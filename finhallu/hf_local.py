"""Run open models locally with Hugging Face transformers (no API key, no server).

Works on a free Kaggle/Colab T4 for models up to ~8B in float16 (use 2xT4 on Kaggle for 7-8B).
"""
from __future__ import annotations

from .models import ModelSpec


class HFModel:
    def __init__(self, model_id: str, max_new_tokens: int = 300):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = ModelSpec(name=model_id.split("/")[-1], model=model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict], qid: str | None = None) -> str:
        import torch

        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        except Exception:  # templates without a system role: fold it into the first user turn
            sys_msg, rest = messages[0]["content"], messages[1:]
            rest = [dict(rest[0], content=sys_msg + "\n\n" + rest[0]["content"])] + rest[1:]
            ids = self.tok.apply_chat_template(rest, add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def close(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
