"""Build the benchmark question set deterministically from SEC XBRL gold facts.

Five question types:
  lookup      - a single reported figure for the latest fiscal year
  prior_year  - the same figure for the prior year (tests year confusion)
  yoy         - year-over-year % change (tests arithmetic on retrieved numbers)
  ratio       - margin computed from two figures (tests multi-fact reasoning)
  trap        - unanswerable from the filing (tests fabrication vs. abstention)
  multiturn   - a follow-up that depends on the previous turn (tests context carry)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABELS = {
    "revenue": "total revenue (net sales)",
    "net_income": "net income",
    "total_assets": "total assets",
    "equity": "total shareholders' equity",
    "op_cash_flow": "net cash provided by (used in) operating activities",
    "rnd": "research and development expense",
    "eps_diluted": "diluted earnings per share",
    "cash": "cash and cash equivalents",
    "operating_income": "operating income",
}
BALANCE_SHEET = {"total_assets", "equity", "cash"}  # point-in-time items


def _when(m: str, label: str, fy_end: str, prior: bool = False) -> str:
    if m in BALANCE_SHEET:
        return f"as of the end of {label}"
    return f"for {label}"


def build(gold_path: Path = ROOT / "data" / "gold_facts.json") -> list[dict]:
    gold = json.loads(gold_path.read_text())["companies"]
    qs: list[dict] = []

    def add(**q):
        q["id"] = f"{q['ticker']}-{q['type']}-{sum(1 for x in qs if x['ticker'] == q['ticker'] and x['type'] == q['type']) + 1:02d}"
        qs.append(q)

    for tk, c in gold.items():
        name, fy, prior, mets = c["name"], c["fy_label"], c["prior_label"], c["metrics"]
        unit_of = lambda m: "usd_per_share" if m == "eps_diluted" else "usd_millions"

        # 1. lookups (latest year), all available metrics
        for m, (cur, _) in mets.items():
            add(ticker=tk, type="lookup", metric=m, unit=unit_of(m), answer=cur, prior_answer=mets[m][1],
                turns=[f"What was {name}'s {LABELS[m]} {_when(m, fy, c['fy_end'])}?"])

        # 2. prior-year lookups for 3 metrics
        for m in [x for x in ("revenue", "net_income", "op_cash_flow", "total_assets") if x in mets][:3]:
            add(ticker=tk, type="prior_year", metric=m, unit=unit_of(m), answer=mets[m][1], prior_answer=mets[m][0],
                turns=[f"What was {name}'s {LABELS[m]} {_when(m, prior, c['fy_end'])}, as reported in the comparative figures?"])

        # 3. YoY % change for up to 3 flow metrics
        for m in [x for x in ("revenue", "net_income", "operating_income", "op_cash_flow", "rnd") if x in mets][:3]:
            cur, prv = mets[m]
            add(ticker=tk, type="yoy", metric=m, unit="percent", answer=round((cur - prv) / abs(prv) * 100, 2),
                turns=[f"By what percentage did {name}'s {LABELS[m]} change from {prior} to {fy}? Give the percentage change."])

        # 4. margins
        for num in ("net_income", "operating_income"):
            if num in mets and "revenue" in mets:
                add(ticker=tk, type="ratio", metric=f"{num}_margin", unit="percent",
                    answer=round(mets[num][0] / mets["revenue"][0] * 100, 2),
                    turns=[f"What was {name}'s {LABELS[num]} as a percentage of total revenue {_when(num, fy, c['fy_end'])}?"])

        # 5. traps: information that cannot be in this filing
        fy_num = int(fy[-4:]) + 3  # three years after the filing's fiscal year
        add(ticker=tk, type="trap", metric="future_revenue", unit="abstain", answer=None,
            turns=[f"What was {name}'s actual reported total revenue for fiscal year {fy_num}?"])
        add(ticker=tk, type="trap", metric="future_quarter", unit="abstain", answer=None,
            turns=[f"What diluted EPS did {name} report for the third quarter of fiscal year {fy_num}?"])

        # 6. multi-turn follow-up (graded on the last turn)
        if "revenue" in mets:
            add(ticker=tk, type="multiturn", metric="revenue", unit="usd_millions", answer=mets["revenue"][1],
                prior_answer=mets["revenue"][0],
                turns=[f"What was {name}'s {LABELS['revenue']} {_when('revenue', fy, c['fy_end'])}?",
                       "And what was it in the prior fiscal year?"])
    return qs


def main() -> None:
    qs = build()
    out = ROOT / "data" / "questions.jsonl"
    out.write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    from collections import Counter
    print(f"wrote {len(qs)} questions -> {out}")
    print(dict(Counter(q["type"] for q in qs)))


if __name__ == "__main__":
    main()
