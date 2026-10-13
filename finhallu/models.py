"""Model clients. Anything OpenAI-compatible works: OpenAI, OpenRouter, Together, Groq, or a local vLLM server.

Configure in models.yaml. API keys are read from the env var named in `api_key_env`; never commit them.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

SYSTEM = (
    "You are a financial analyst assistant. Answer ONLY from the 10-K excerpts provided. "
    "Monetary amounts in these filings are usually in millions of US dollars. "
    "If the excerpts do not contain the answer, reply exactly: NOT_IN_DOCUMENT. "
    "End with one line: 'FINAL ANSWER: <number with units, or NOT_IN_DOCUMENT>'."
)


@dataclass
class ModelSpec:
    name: str
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 400
    extra: dict = field(default_factory=dict)


class ChatModel:
    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.spec = spec
        key = os.environ.get(spec.api_key_env, "EMPTY")  # local vLLM accepts any key
        self.client = OpenAI(base_url=spec.base_url, api_key=key)

    def chat(self, messages: list[dict], qid: str | None = None) -> str:
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(
                    model=self.spec.model, messages=messages,
                    temperature=self.spec.temperature, max_tokens=self.spec.max_tokens, **self.spec.extra,
                )
                return r.choices[0].message.content or ""
            except Exception as e:  # rate limits, transient 5xx
                if attempt == 3:
                    return f"[ERROR] {type(e).__name__}: {e}"
                time.sleep(2 ** attempt * 2)
        return ""


class OracleModel:
    """Test double: answers from gold. Used by tests and `--dry-run` to validate the pipeline."""

    def __init__(self, questions: dict[str, dict], noise: bool = False):
        self.q, self.noise, self.spec = questions, noise, ModelSpec(name="oracle" + ("-noisy" if noise else ""), model="oracle")

    def chat(self, messages: list[dict], qid: str | None = None) -> str:
        q = self.q[qid]
        if q["unit"] == "abstain":
            return "FINAL ANSWER: $512 million" if self.noise else "FINAL ANSWER: NOT_IN_DOCUMENT"
        v = q.get("prior_answer") if (self.noise and q.get("prior_answer") is not None) else q["answer"]
        suffix = {"usd_millions": " million", "percent": "%", "usd_per_share": ""}[q["unit"]]
        return f"FINAL ANSWER: ${v:,}{suffix}" if q["unit"] != "percent" else f"FINAL ANSWER: {v}%"
