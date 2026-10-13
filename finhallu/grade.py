"""Deterministic grading + failure taxonomy.

Every answer gets one label:
  correct           - within tolerance of the gold value (or abstained on a trap)
  fabricated        - gave a number for an unanswerable (trap) question
  wrong_year        - matches the other fiscal year's value
  scale_error       - right digits, wrong unit (x1000 / /1000, e.g. thousands vs millions)
  sign_error        - right magnitude, wrong sign (e.g. cash used vs provided)
  wrong_value       - a number that matches nothing in the gold set for that question
  false_refusal     - abstained although the answer is in the filing
  no_answer         - no parsable number and no clear abstention
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
    r"(?P<neg>[-−(])?\s*\$?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<scale>trillion|billion|million|thousand|bn|mn|[tbmk](?![a-z]))?\s*(?P<pct>%|percent)?",
    re.I,
)
_SCALE = {"trillion": 1e6, "t": 1e6, "billion": 1e3, "bn": 1e3, "b": 1e3, "million": 1, "mn": 1, "m": 1, "thousand": 1e-3, "k": 1e-3}


def abstained(text: str) -> bool:
    return bool(_abstain_re.search(text or ""))


def extract_number(text: str, unit: str) -> float | None:
    """Pull the answer number from the FINAL ANSWER line if present, else the first number.

    For usd_millions answers, the number is converted to millions using any scale word.
    A bare number with no scale word is assumed to already be in millions (the filings' unit).
    """
    if not text:
        return None
    m = re.search(r"final answer\s*[:\-]\s*(.+)", text, re.I)
    seg = m.group(1) if m else text
    for mt in _num_re.finditer(seg):
        raw = mt.group("num").replace(",", "")
        if re.fullmatch(r"(19|20)\d\d", raw) and not mt.group("scale") and not mt.group("pct"):
            continue  # skip years
        val = float(raw)
        if mt.group("neg"):
            val = -val
        if unit == "percent" and val > 0 and re.search(r"decrease|decline|fell|drop|down|lower|reduc", seg[: mt.start()] + seg[mt.end(): mt.end() + 20], re.I):
            val = -val
        sc = (mt.group("scale") or "").lower()
        if unit == "usd_millions":
            if sc in _SCALE:
                val *= _SCALE[sc]
            elif val > 1e8:  # raw dollars written out in full
                val /= 1e6
        return val
    return None


def _close(a: float, b: float, unit: str) -> bool:
    if unit == "percent":
        return abs(a - b) <= 0.5  # percentage points
    if unit == "usd_per_share":
        return abs(a - b) <= 0.011
    return abs(a - b) <= max(abs(b) * 0.005, 0.6)  # 0.5% handles "$416.2 billion" style rounding


def grade(q: dict, answer: str) -> str:
    unit, gold = q["unit"], q["answer"]
    if unit == "abstain":
        if extract_number_for_trap(answer) and not abstained(answer):
            return "fabricated"
        return "correct" if abstained(answer) else "no_answer"
    val = extract_number(answer, unit)
    if val is None:
        return "false_refusal" if abstained(answer) else "no_answer"
    if _close(val, gold, unit):
        return "correct"
    other = q.get("prior_answer")
    if other is not None and _close(val, other, unit):
        return "wrong_year"
    if unit == "usd_millions" and any(_close(val * f, gold, unit) for f in (1e3, 1e-3, 1e6, 1e-6)):
        return "scale_error"
    if _close(-val, gold, unit):
        return "sign_error"
    return "wrong_value"


def extract_number_for_trap(text: str) -> bool:
    """True if the answer asserts a monetary/EPS figure (ignoring years)."""
    return extract_number(text or "", "usd_millions") is not None
