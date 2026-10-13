# FinHallu-10K

**How often do LLMs get financial facts wrong when answering from a company's 10-K?**

182 questions over the latest annual reports of 10 US public companies (Apple, Microsoft, NVIDIA, JPMorgan, Exxon, J&J, Walmart, Caterpillar, Coca-Cola, Netflix). Every gold answer comes from the company's own XBRL-tagged financial data on SEC EDGAR — not hand-typed, not LLM-generated.

Models answer the way most production "chat with your documents" apps do: BM25 retrieval over the filing, top-8 chunks, one prompt.

## Question types

| Type | n | What it tests |
|---|---|---|
| lookup | 76 | One reported figure (revenue, net income, assets, EPS, cash…) |
| prior_year | 30 | Last year's comparative figure — catches year confusion |
| yoy | 30 | % change between years — retrieval + arithmetic |
| ratio | 16 | Net / operating margin — two facts combined |
| multiturn | 10 | "And the year before?" follow-up — context carry-over |
| trap | 20 | Unanswerable (a future year) — does the model invent a number? |

## Failure taxonomy

Every wrong answer gets exactly one label: `fabricated` (number for an unanswerable question), `wrong_year`, `scale_error` (thousands vs millions), `sign_error`, `wrong_value`, `false_refusal`, `no_answer`. Grading is deterministic (0.5% tolerance for amounts, 0.5 pp for percentages) and validated against human labels with Cohen's kappa (`scripts/agreement.py`).

## Run it

```bash
pip install -r requirements.txt
make test                                   # unit tests + dry run, no API calls
export SEC_USER_AGENT="Your Name you@email.com"
make filings                                # downloads the 10 filings from EDGAR (~1 min)
export OPENROUTER_API_KEY=...  OPENAI_API_KEY=...
make run                                    # all models in models.yaml
make report                                 # results/results_summary.md + results_chart.png
```

Free option: run open models on a Kaggle T4 with vLLM (`vllm serve Qwen/Qwen2.5-7B-Instruct`) and point a `models.yaml` entry at `http://localhost:8000/v1`.

## Results

_Run `make run && make report`, then paste `results/results_summary.md` and the chart here._

## Limitations

- 10 companies, one filing each; BM25 only (no embeddings or rerankers) — this measures a common baseline, not the best possible system.
- Gold values come from XBRL tags; where a company presents a figure differently in the text, the 0.5% tolerance absorbs rounding but not definitional differences. Disagreements found during human labelling are listed by `scripts/agreement.py score`.
- Bank "revenue" (JPM) is the XBRL `Revenues` total, which can differ from managed-basis figures in the narrative.

## Want this for your product?

I build domain-specific eval suites for AI teams: failure taxonomy, graded test cases, a validated judge, CI regression runs. — Vedansh Mathur · [LinkedIn](https://www.linkedin.com/in/vedansh-mathur-86767a300) · vedanshmathur007@gmail.com
