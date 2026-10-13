"""Deterministic grading + failure taxonomy.

Every answer gets one label:
  correct           - within tolerance of the gold value (or abstained on a trap)
  fabricated        - gave a number for an unanswerable (trap) question
  wrong_year        - matches the other fiscal year's value
  scale_error       - right digits, wrong unit (x1000 / /1000, e.g. thousands vs millions)
  sign_error        - right magnitude, wrong sign (e.g. cash used vs provided)
  wrong_value       - a number that matches nothing in the gold set for that question
  false_refusal     - abstained although the answer is in the filing
  no_answer         - no parsable number and no clear abstention (e.g. reasoning cut off)
  infra_error       - the model call itself failed (out of memory, API error); excluded from accuracy
"""
from __future__ import annotations

import re

ABSTAIN = "NOT_IN_DOCUMENT"
_abstain_re = re.compile(
    r"not_in_document|not (?:provided|available|disclosed|included|mentioned|found|stated|contain|reported)|"
    r"(?:does|do|did) not (?:contain|include|provide|report|disclose|mention)|cannot (?:be )?(?:determine|find|answer)|"
    r"no (?:information|data|mention)|unable to (?:determine|find|answer)|isn't (?:in|provided|available)",
    re.I,
)
_num_re = re.compile(
    r"(?P<neg>[-−(])?\s*(?P<dollar>\$)?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?![\d,]*-?[KQ]\b)"  # "10-K", "10-Q" are form names, not answers
    r"\s*(?P<scale>trillion|billion|million|thousand|bn|mn|[tbmk](?![a-z]))?\s*(?P<pct>%|percent)?",
    re.I,
)
_SCALE = {"trillion": 1e6, "t": 1e6, "billion": 1e3, "bn": 1e3, "b": 1e3, "million": 1, "mn": 1, "m": 1, "thousand": 1e-3, "k": 1e-3}
_MONTH = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s*$", re.I)


def abstained(text: str) -> bool:
    return bool(_abstain_re.search(text or ""))


def _final_segment(text: str) -> tuple[str, bool]:
    m = re.search(r"final answer\s*[:\-]\s*(.+)", text, re.I)
    return (m.group(1), True) if m else (text, False)


def extract(text: str, unit: str) -> tuple[float, float] | None:
    """Return (value, half_unit) for the answer figure, or None.

    Uses the FINAL ANSWER line when present, else the first answer-like number in the text.
    Answer-like = has a $ sign, a scale word, thousands commas, a % (percent questions), or decimals
    (EPS); bare small integers, years, day-of-month and form names ("10-K") are skipped.
    half_unit is half of the last stated digit after scaling, so "$416.2 billion" may differ from
    416,161 by up to 50 (millions) but "130,141 million" must match to the million.
    """
    if not text:
        return None
    seg, is_final = _final_segment(text)
    for mt in _num_re.finditer(seg):
        raw = mt.group("num").replace(",", "")
        sc = (mt.group("scale") or "").lower()
        has_pct = bool(mt.group("pct"))
        if re.fullmatch(r"(19|20)\d\d", raw) and not sc and not has_pct:
            continue  # a year
        if _MONTH.search(seg[: mt.start()]) and not mt.group("dollar") and not sc:
            continue  # "September 27"
        if unit == "percent":  # a dollar figure in the working is never the answer to a % question
            answer_like = has_pct
        else:
            answer_like = bool(mt.group("dollar") or sc or "," in mt.group("num") or ("." in raw and unit == "usd_per_share"))
        if unit == "percent" and is_final and not has_pct and (mt.group("dollar") or sc or "," in mt.group("num")):
            continue
        if not answer_like and not is_final:
            continue
        val = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        half = 0.5 * 10 ** (-decimals)
        if mt.group("neg"):
            val = -val
        if unit == "percent" and val > 0 and re.search(r"decrease|decline|fell|drop|down|lower|reduc",
                                                       seg[: mt.start()] + seg[mt.end(): mt.end() + 20], re.I):
            val = -val
        if unit == "usd_millions":
            if sc in _SCALE:
                val *= _SCALE[sc]
                half *= _SCALE[sc]
            elif abs(val) > 1e8:  # raw dollars written out in full
                val /= 1e6
                half /= 1e6
        return val, half
    return None


def extract_number(text: str, unit: str) -> float | None:
    r = extract(text, unit)
    return None if r is None else r[0]


_MIN_TOL = {"usd_millions": 0.6, "percent": 0.1, "usd_per_share": 0.006}


def _close(a: float, b: float, unit: str, half: float = 0.0) -> bool:
    return abs(a - b) <= max(half * 1.0001, _MIN_TOL[unit])


def grade(q: dict, answer: str) -> str:
    unit, gold = q["unit"], q["answer"]
    answer = answer or ""
    if answer.startswith("[ERROR]"):
        return "infra_error"  # GPU/API failure: excluded from accuracy, reported separately
    if unit == "abstain":
        if abstained(answer):
            return "correct"
        return "fabricated" if extract_number_for_trap(answer) else "no_answer"
    seg, is_final = _final_segment(answer)
    if is_final and abstained(seg):
        return "false_refusal"
    r = extract(answer, unit)
    if r is None:
        return "false_refusal" if abstained(answer) else "no_answer"
    val, half = r
    if _close(val, gold, unit, half):
        return "correct"
    other = q.get("prior_answer")
    if other is not None and _close(val, other, unit, half):
        return "wrong_year"
    if unit == "usd_millions" and any(_close(val * f, gold, unit, half * f) for f in (1e3, 1e-3, 1e6, 1e-6)):
        return "scale_error"
    if _close(-val, gold, unit, half):
        return "sign_error"
    return "wrong_value"


def extract_number_for_trap(text: str) -> bool:
    """True if the answer asserts a monetary or EPS figure (years, dates and form names ignored)."""
    return extract(text or "", "usd_millions") is not None or extract(text or "", "usd_per_share") is not None
