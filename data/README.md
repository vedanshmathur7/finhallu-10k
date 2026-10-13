# Data

gold_facts.json: figures from the SEC EDGAR XBRL companyfacts API, tagged to each company's latest 10-K. Values in USD millions; EPS in USD per diluted share.

questions.jsonl: 182 questions generated from gold_facts.json by `python -m finhallu.questions`.

filings_txt/ and filings_raw/ are created by `make filings` and are not committed.
