"""HFModel._encode must work whether apply_chat_template returns a dict (transformers 5) or a tensor (old)."""
import types

import pytest

torch = pytest.importorskip("torch")
from finhallu.hf_local import HFModel


def _model(tok):
    m = HFModel.__new__(HFModel)
    m.tok = tok
    return m


def test_dict_return():
    tok = types.SimpleNamespace(apply_chat_template=lambda *a, **k: {"input_ids": torch.ones(1, 5, dtype=torch.long), "attention_mask": torch.ones(1, 5, dtype=torch.long)})
    enc = _model(tok)._encode([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert enc["input_ids"].shape == (1, 5)


def test_no_system_role_falls_back():
    def tmpl(msgs, **k):
        if msgs[0]["role"] == "system":
            raise ValueError("System role not supported")
        assert msgs[0]["content"].startswith("s\n\nu")
        return {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    enc = _model(types.SimpleNamespace(apply_chat_template=tmpl))._encode([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert enc["attention_mask"].shape == (1, 3)
