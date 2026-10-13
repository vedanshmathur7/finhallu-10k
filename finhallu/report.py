"""Turn results CSV into: summary table (markdown), failure taxonomy, and a chart for the post.

    python -m finhallu.report                       # results/results.csv
    python -m finhallu.report results/results_dryrun.csv
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["lookup", "prior_year", "yoy", "ratio", "multiturn", "trap"]
LABELS = ["correct", "fabricated", "wrong_year", "scale_error", "sign_error", "wrong_value", "false_refusal", "no_answer"]


def main(path: str | None = None) -> None:
    p = Path(path) if path else ROOT / "results" / "results.csv"
    rows = list(csv.DictReader(p.open()))
    models = sorted({r["model"] for r in rows})
    acc = defaultdict(lambda: defaultdict(list))
    fails = defaultdict(Counter)
    for r in rows:
        acc[r["model"]][r["type"]].append(r["label"] == "correct")
        acc[r["model"]]["overall"].append(r["label"] == "correct")
        if r["label"] != "correct":
            fails[r["model"]][r["label"]] += 1

    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"
    md = ["## Accuracy by question type", "", "| Model | Overall | " + " | ".join(TYPES) + " |", "|---" * (len(TYPES) + 2) + "|"]
    for m in sorted(models, key=lambda m: -sum(acc[m]["overall"])):
        md.append(f"| {m} | **{pct(acc[m]['overall'])}** | " + " | ".join(pct(acc[m][t]) for t in TYPES) + " |")
    md += ["", "## Failure taxonomy (count of wrong answers)", "", "| Model | " + " | ".join(LABELS[1:]) + " |", "|---" * len(LABELS) + "|"]
    for m in models:
        md.append(f"| {m} | " + " | ".join(str(fails[m][l]) for l in LABELS[1:]) + " |")

    # trap fabrication rate — usually the headline finding
    md += ["", "## Headline numbers", ""]
    for m in models:
        traps = acc[m]["trap"]
        if traps:
            md.append(f"- **{m}** invented a number on {100 - 100 * sum(traps) / len(traps):.0f}% of unanswerable questions.")
    out = p.with_name(p.stem + "_summary.md")
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    _chart(models, acc, p.with_name(p.stem + "_chart.png"))


def _chart(models, acc, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    w = 0.8 / max(len(models), 1)
    palette = ["#2a6fdb", "#e4572e", "#17a398", "#f3a712", "#7d5ba6", "#555555"]
    for i, m in enumerate(models):
        vals = [100 * sum(acc[m][t]) / len(acc[m][t]) if acc[m][t] else 0 for t in TYPES]
        ax.bar([x + i * w for x in range(len(TYPES))], vals, w, label=m, color=palette[i % len(palette)])
    ax.set_xticks([x + w * (len(models) - 1) / 2 for x in range(len(TYPES))])
    ax.set_xticklabels(["lookup", "prior year", "YoY %", "margin", "follow-up", "unanswerable\n(abstained)"])
    ax.set_ylabel("% correct")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=min(len(models), 3), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.set_title("10-K QA with BM25 retrieval: where LLMs get financial facts wrong", pad=34, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    print(f"chart -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
