"""Validate the automatic grader against human judgement (Cohen's kappa), like TourHalluBench.

Step 1  python scripts/agreement.py sample          -> results/human_labels.csv (40 random rows, blank 'human' column)
Step 2  fill 'human' with 1 (correct) or 0 (wrong) by reading each answer against the filing
Step 3  python scripts/agreement.py score           -> agreement %, Cohen's kappa, disagreements to inspect
"""
import csv
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def sample(n=40, seed=7):
    rows = list(csv.DictReader((R / "results.csv").open()))
    random.Random(seed).shuffle(rows)
    # stratify: half auto-correct, half auto-wrong, so kappa isn't inflated by an easy class
    pick = [r for r in rows if r["label"] == "correct"][: n // 2] + [r for r in rows if r["label"] != "correct"][: n // 2]
    with (R / "human_labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "model", "type", "gold", "answer", "label", "human"])
        w.writeheader()
        for r in pick:
            w.writerow({k: r.get(k, "") for k in ["id", "model", "type", "gold", "answer", "label"]} | {"human": ""})
    print(f"label {len(pick)} rows in {R / 'human_labels.csv'}")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def score():
    rows = [r for r in csv.DictReader((R / "human_labels.csv").open()) if r["human"].strip() in {"0", "1"}]
    auto = [int(r["label"] == "correct") for r in rows]
    human = [int(r["human"]) for r in rows]
    po, k = kappa(auto, human)
    print(f"n={len(rows)}  agreement={po:.1%}  Cohen's kappa={k:.2f}")
    for r, a, h in zip(rows, auto, human):
        if a != h:
            print(f"  DISAGREE {r['id']} [{r['model']}] auto={r['label']} human={h}: {r['answer'][:120]!r}")


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
