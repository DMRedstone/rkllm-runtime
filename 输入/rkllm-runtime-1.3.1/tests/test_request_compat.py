"""OpenAI 兼容参数：ignored/unsupported 字段必须先能进 schema。

DSH 等客户端默认会带 `store: false`；若 Pydantic `extra=forbid` 先拒，caps 的
ignored 策略（剥离、不 400）永远执行不到，表现为 422 extra_forbidden。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm import caps  # noqa: E402
from rkllm.server.schemas import ChatCompletionRequest, CompletionRequest  # noqa: E402


def test_chat_accepts_ignored_openai_fields_and_strips():
    body = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        store=False, user="u1", metadata={"k": "v"},
        service_tier="auto", verbosity="low",
    ).model_dump()
    reject, warn = caps.check_request_params(body)
    assert reject == [] and warn == []
    for name in ("store", "user", "metadata", "service_tier", "verbosity"):
        assert name not in body          # caps 已剥离


def test_chat_unsupported_fields_reach_policy():
    body = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        top_logprobs=3, logit_bias={"1": 0.5},
        modalities=["text"], audio={"format": "wav"},
        prediction={"type": "content", "content": "x"},
    ).model_dump()
    reject, warn = caps.check_request_params(body, policy="warn")
    reported = set(reject) | set(warn)
    for name in ("top_logprobs", "logit_bias", "modalities", "audio", "prediction"):
        assert name in reported, (name, reject, warn)


def test_completion_accepts_store_and_user():
    body = CompletionRequest(prompt="hi", store=False, user="u1").model_dump()
    assert body["store"] is False and body["user"] == "u1"
