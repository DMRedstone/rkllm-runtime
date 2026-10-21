"""caps.py 单测：能力三档契约与请求参数检查（D2/G5 验收）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.caps import (CAPS, ENDPOINTS, capabilities_doc, check_request_params,
                        param_tier)


def test_caps_structure():
    assert set(CAPS) == {"endpoints", "params", "limits", "notes"}
    tiers = CAPS["params"]
    assert tiers["native"] and tiers["emulated"] and tiers["ignored"]
    assert tiers["unsupported"]


def test_param_tier():
    assert param_tier("temperature") == "native"
    assert param_tier("tools") == "emulated"
    assert param_tier("user") == "ignored"
    assert param_tier("seed") == "unsupported"
    assert param_tier("response_format.json_schema") == "emulated"
    assert param_tier("completely_unknown") == "unsupported"   # 诚实原则


def test_check_reject_semantic_key_even_in_warn():
    # tool_choice 是声明为恒 reject 的语义关键项：必须真的拦（审计 P2-8）
    reject, warn = check_request_params({"tool_choice": "auto"}, policy="warn")
    assert reject == ["tool_choice"] and not warn
    # json_schema：非流式是 emulated（提示+校验+重试），放行
    reject, _ = check_request_params(
        {"response_format": {"type": "json_schema"}}, policy="warn")
    assert not reject
    # 流式无法在发帧后纠正，明确拒绝
    reject, _ = check_request_params(
        {"response_format": {"type": "json_schema"}}, policy="warn",
        stream=True)
    assert reject and "json_schema" in reject[0]


def test_check_warn_policy():
    body = {"seed": 42, "logprobs": True, "n": 1}
    reject, warn = check_request_params(body, policy="warn")
    assert not reject
    assert set(warn) == {"seed", "logprobs"}


def test_check_reject_policy():
    body = {"seed": 42}
    reject, warn = check_request_params(body, policy="reject")
    assert reject == ["seed"]


def test_check_n_gt_1_always_rejected():
    body = {"n": 3}
    reject, _ = check_request_params(body, policy="warn")
    assert "n>1" in reject


def test_ignored_params_stripped_not_flagged():
    body = {"user": "u1", "metadata": {"a": 1}, "store": False}
    reject, warn = check_request_params(body, policy="reject")
    assert not reject and not warn
    assert "user" not in body and "metadata" not in body   # 静默剥离


def test_endpoints_native_core():
    assert ENDPOINTS["chat.completions"] == "native"
    assert ENDPOINTS["embeddings"] == "unsupported"


def test_capabilities_doc_line():
    line = capabilities_doc()
    assert "native" in line and "emulated" in line
