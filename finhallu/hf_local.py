"""Run open models locally with Hugging Face transformers (no API key, no server).

Works on a free Kaggle/Colab T4 for models up to ~8B in float16 (use 2xT4 on Kaggle for 7-8B).
"""
from __future__ import annotations

from .models import ModelSpec


def _fold_system(messages: list[dict]) -> list[dict]:
    """For chat templates without a system role: prepend the system prompt to the first user turn."""
    sys_msg, rest = messages[0]["content"], messages[1:]
    return [dict(rest[0], content=sys_msg + "\n\n" + rest[0]["content"])] + rest[1:]


class HFModel:
    def __init__(self, model_id: str, max_new_tokens: int = 300):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = ModelSpec(name=model_id.split("/")[-1], model=model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        n_gpu = torch.cuda.device_count()
        # Leave headroom on the last GPU: generation's logits (seq x vocab) land there.
        max_memory = {i: ("13GiB" if i < n_gpu - 1 else "9GiB") for i in range(n_gpu)} if n_gpu > 1 else None
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto",
                                                          max_memory=max_memory, attn_implementation="sdpa")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = 6000

    def _encode(self, messages: list[dict]) -> dict:
        """Returns {'input_ids', 'attention_mask'} on any transformers version (4.x returns a tensor, 5.x a dict)."""
        try:
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        except Exception:
            enc = self.tok.apply_chat_template(_fold_system(messages), add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        if not hasattr(enc, "keys"):  # very old versions ignore return_dict
            import torch

            enc = {"input_ids": enc, "attention_mask": torch.ones_like(enc)}
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def chat(self, messages: list[dict], qid: str | None = None) -> str:
        import torch

        enc = self._encode(messages)
        n = enc["input_ids"].shape[1]
        if n > self.max_input_tokens:  # keep the tail: the question and the latest excerpts
            enc = {k: v[:, -self.max_input_tokens:] for k, v in enc.items()}
            n = self.max_input_tokens
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        try:
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "[ERROR] CUDA out of memory"  # graded as no_answer; counted, not hidden
        return self.tok.decode(out[0, n:], skip_special_tokens=True)

    def close(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
