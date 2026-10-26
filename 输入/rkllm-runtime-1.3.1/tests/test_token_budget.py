"""prompt + max_tokens 的 context 预算（P0：修 rkllm_run -1）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.agent.pipeline import (CONTEXT_MARGIN_TOKENS, Pipeline,
                                  PROMPT_BYTES_PER_TOKEN)


class _Svc:
    model_name = "budget-test"
    tool_parser = "auto"
    max_tool_calls = 3
    tools_mode = "prompt"
    system_prompt = ""
    json_hint = ""

    def __init__(self, context=128, default_max_tokens=32, limit=0):
        self.context_len = context
        self.default_max_tokens = default_max_tokens
        self.max_tokens_limit = limit


def _pipe(svc, prompt, max_tokens=None):
    body = {"messages": [{"role": "user", "content": prompt}]}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return Pipeline(svc, body, chat=True)


def test_prompt_too_long_rejected_with_clear_error():
    with pytest.raises(ValueError) as exc:
        _pipe(_Svc(context=64), "字" * 300, 32)
    assert "prompt too long" in str(exc.value)
    assert "context=64" in str(exc.value)


def test_max_tokens_clamped_to_remaining_context():
    svc = _Svc(context=128)
    pipe = _pipe(svc, "hi", 1000)
    estimated = max(1, len(pipe._prompt.encode("utf-8")) // PROMPT_BYTES_PER_TOKEN)
    expected = 128 - estimated - CONTEXT_MARGIN_TOKENS
    assert pipe.max_tokens == expected
    assert pipe.max_tokens_clamped is True


def test_config_limit_rejects_explicit_request():
    with pytest.raises(ValueError) as exc:
        _pipe(_Svc(context=4096, limit=100), "hi", 101)
    assert "model.max_tokens_limit=100" in str(exc.value)


def test_default_max_tokens_also_clamped_by_context():
    svc = _Svc(context=64, default_max_tokens=4096)
    pipe = _pipe(svc, "hi")
    assert pipe.max_tokens <= 64
    assert pipe.max_tokens_clamped is True


def test_no_context_limit_keeps_requested_value():
    svc = _Svc(context=0, limit=0)
    pipe = _pipe(svc, "hi", 500)
    assert pipe.max_tokens == 500
    assert pipe.max_tokens_clamped is False
