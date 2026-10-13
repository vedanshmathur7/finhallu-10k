"""Re-grade saved answers with the current grader (no model calls), then print an audit of every miss.

    python scripts/regrade.py results/results.csv
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from finhallu.grade import grade  # noqa: E402

path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "results.csv")
qs = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data" / "questions.jsonl").read_text().splitlines() if l.strip()}
rows = list(csv.DictReader(path.open()))
changed = 0
for r in rows:
    new = grade(qs[r["id"]], r["answer"])
    changed += new != r["label"]
    r["label"] = new
with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"re-graded {len(rows)} rows, {changed} labels changed -> {path}")
print("\n=== AUDIT: every non-correct answer ===")
for r in rows:
    if r["label"] != "correct":
        print(f"{r['id']} | {r['model']} | {r['label']} | gold={r['gold']} | {r['answer'][:500]!r}")
