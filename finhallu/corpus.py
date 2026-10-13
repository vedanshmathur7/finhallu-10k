"""Download the 10 filings from SEC EDGAR, convert to plain text, chunk, and index with BM25.

SEC asks automated clients to send a descriptive User-Agent with contact info and stay under
10 requests/second. Set SEC_USER_AGENT="Your Name you@example.com" before running.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "filings_raw"
TXT = ROOT / "data" / "filings_txt"


def filing_url(c: dict) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{c['cik']}/{c['accn'].replace('-', '')}/{c['doc']}"


def html_to_text(html: str) -> str:
    """Inline-XBRL 10-K HTML -> readable text, keeping table rows on one line."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    for hidden in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        hidden.decompose()  # the ix:header block of hidden XBRL facts
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", ")", "%"}]
        tr.replace_with(soup.new_string("\n" + " | ".join(cells) + "\n"))
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def download(gold_path: Path = ROOT / "data" / "gold_facts.json") -> None:
    import requests

    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit('Set SEC_USER_AGENT="Your Name your@email.com" (SEC requires a contact User-Agent).')
    RAW.mkdir(parents=True, exist_ok=True)
    TXT.mkdir(parents=True, exist_ok=True)
    for tk, c in json.loads(gold_path.read_text())["companies"].items():
        raw = RAW / f"{tk}.htm"
        if not raw.exists():
            r = requests.get(filing_url(c), headers={"User-Agent": ua}, timeout=60)
            r.raise_for_status()
            raw.write_bytes(r.content)
            time.sleep(0.5)
        txt = TXT / f"{tk}.txt"
        txt.write_text(html_to_text(raw.read_text(errors="ignore")))
        print(f"{tk}: {txt.stat().st_size // 1024} KB text")


def chunk(text: str, size: int = 1500, overlap: int = 250) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


_tok = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _tok.findall(s.lower())


class Retriever:
    """BM25 over one filing's chunks — the cheap baseline most startups actually ship."""

    def __init__(self, text: str, k: int = 8):
        from rank_bm25 import BM25Okapi

        self.chunks = chunk(text)
        self.bm25 = BM25Okapi([tokenize(c) for c in self.chunks])
        self.k = k

    def search(self, query: str) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.k]
        return [self.chunks[i] for i in sorted(top)]  # keep document order


def load_retrievers(k: int = 8) -> dict[str, Retriever]:
    if not TXT.exists() or not any(TXT.glob("*.txt")):
        raise SystemExit("No filings yet. Run: python -m finhallu.corpus")
    return {p.stem: Retriever(p.read_text(), k=k) for p in sorted(TXT.glob("*.txt"))}


if __name__ == "__main__":
    download()
