"""Run every model on every question with BM25 retrieval, grade, and save results.

    python -m finhallu.run                  # all models in models.yaml
    python -m finhallu.run --models gpt-4.1-mini qwen-local
    python -m finhallu.run --dry-run        # oracle models, no API calls, no filings needed
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .grade import grade
from .models import SYSTEM, ChatModel, ModelSpec, OracleModel

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_questions() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "data" / "questions.jsonl").read_text().splitlines() if l.strip()]


def load_specs(names: list[str] | None) -> list[ModelSpec]:
    import yaml

    cfg = yaml.safe_load((ROOT / "models.yaml").read_text())
    specs = [ModelSpec(**m) for m in cfg["models"]]
    return [s for s in specs if not names or s.name in names]


def ask(model, q: dict, retr) -> dict:
    """Retrieve per turn, carry the conversation, grade the last turn."""
    msgs = [{"role": "system", "content": SYSTEM}]
    answer, ctx_hit = "", None
    for i, turn in enumerate(q["turns"]):
        query = " ".join(q["turns"][: i + 1])  # follow-ups retrieve with the earlier turn as context
        ctx = "\n\n---\n\n".join(retr.search(query)) if retr else ""
        msgs.append({"role": "user", "content": f"10-K excerpts:\n{ctx}\n\nQuestion: {turn}"})
        answer = model.chat(msgs, qid=q["id"])
        msgs[-1] = {"role": "user", "content": f"Question: {turn}"}  # keep history short: drop old excerpts
        msgs.append({"role": "assistant", "content": answer})
    return {
        "id": q["id"], "ticker": q["ticker"], "type": q["type"], "metric": q["metric"], "unit": q["unit"],
        "gold": q["answer"], "model": model.spec.name, "answer": answer.strip(), "label": grade(q, answer),
    }


def run(models, questions, retrievers, workers: int = 6) -> list[dict]:
    rows = []
    for m in models:
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(ask, m, q, retrievers.get(q["ticker"]) if retrievers else None) for q in questions]
            for f in futs:
                rows.append(f.result())
        acc = sum(r["label"] == "correct" for r in rows if r["model"] == m.spec.name) / len(questions)
        print(f"{m.spec.name:28s} accuracy {acc:6.1%}")
    return rows


def save(rows: list[dict], tag: str = "") -> Path:
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"results{tag}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="chunks retrieved per question")
    ap.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    ap.add_argument("--hf", nargs="*", help="Hugging Face model ids to run locally on GPU instead of API models")
    a = ap.parse_args()

    qs = load_questions()[: a.limit] if a.limit else load_questions()
    if a.dry_run:
        by_id = {q["id"]: q for q in qs}
        rows = run([OracleModel(by_id), OracleModel(by_id, noise=True)], qs, None)
        save(rows, "_dryrun")
        return
    from .corpus import load_retrievers

    retr = load_retrievers(a.k)
    if a.hf:  # local GPU models, loaded one at a time to fit in memory
        from .hf_local import HFModel

        rows = []
        for mid in a.hf:
            try:
                m = HFModel(mid)
                rows += run([m], qs, retr, workers=1)
                save(rows)  # checkpoint after each model
                m.close()
            except Exception as e:  # one broken model must not sink the whole run
                import traceback

                print(f"!! {mid} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
        if not rows:
            raise SystemExit("all models failed")
        return
    rows = run([ChatModel(s) for s in load_specs(a.models)], qs, retr)
    save(rows)


if __name__ == "__main__":
    main()
