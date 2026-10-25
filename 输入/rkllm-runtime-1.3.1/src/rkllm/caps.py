"""OpenAI 兼容能力三档表 —— 单一事实源（改造设计 §4.4 / §7，需求 G5/F11）。

三档定义：
- native      语义与 OpenAI 一致，参数真实生效
- emulated    尽力实现，行为可验证但非等价（提示词/校验/重试），响应头标注
- unsupported 返回 400 + code:"unsupported_parameter"（reject 策略）或
              忽略并写 X-RKLLM-Unsupported 头（warn 策略）
- ignored     新 SDK 默认字段，记录但不生效，不 400（否则会打挂客户端）

任何能力变化只改本文件；/v1/capabilities、启动日志摘要、文档支持矩阵、
请求参数检查全部由此渲染。request_params() 是 HTTP 层参数检查的唯一入口。
"""

from __future__ import annotations

from typing import Any

__all__ = ["CAPS", "param_tier", "check_request_params"]

# ---------------------------------------------------------------------------
# 端点三档
# ---------------------------------------------------------------------------

ENDPOINTS: dict[str, str] = {
    "chat.completions": "native",
    "completions": "native",          # 薄封装：prompt 直喂 runtime
    "responses": "emulated",          # 翻译层：Responses -> Chat（02 §4）
    "models": "native",               # 单模型；models/{id} 返回同一对象
    "embeddings": "unsupported",      # 仅当配置 embedding 模型才可用，不假装支持
    "messages(anthropic)": "unsupported",
    "moderations": "unsupported",     # root_alias 有别名，故必须显式 501 而非 404
    "health": "native",               # 非标准扩展
    "ready": "native",
    "metrics": "native",
    "capabilities": "native",         # 本项目自定扩展
    "reset": "native",
}

# ---------------------------------------------------------------------------
# 参数四档（native / emulated / ignored / unsupported）
# ---------------------------------------------------------------------------

NATIVE_PARAMS: tuple[str, ...] = (
    "model", "messages", "stream", "stream_options", "stop",
    "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "top_k", "repeat_penalty",
    "presence_penalty", "frequency_penalty",
    "reasoning_effort", "enable_thinking",
)

EMULATED_PARAMS: tuple[str, ...] = (
    "tools", "tool_choice", "parallel_tool_calls",
    "response_format.json_object", "response_format.json_schema",
)

IGNORED_PARAMS: tuple[str, ...] = (
    "user", "metadata", "store", "service_tier", "verbosity",
)

UNSUPPORTED_PARAMS: tuple[str, ...] = (
    "n>1", "seed", "logprobs", "top_logprobs", "logit_bias",
    "modalities", "audio", "prediction",
)

_PARAM_TIER: dict[str, str] = {
    **{p: "native" for p in NATIVE_PARAMS},
    **{p: "emulated" for p in EMULATED_PARAMS},
    **{p: "ignored" for p in IGNORED_PARAMS},
    **{p: "unsupported" for p in UNSUPPORTED_PARAMS},
}

# ---------------------------------------------------------------------------
# 运行时上限（按档位默认值可被 config 覆盖；此处为 schema 兜底值）
# ---------------------------------------------------------------------------

LIMITS: dict[str, Any] = {
    "context_len": 4096,
    "max_output": 2048,
    "concurrency": 1,
    "queue": 32,
    "request_timeout_s": 900,
}

NOTES: dict[str, str] = {
    # 硬件现实而非缺陷：RK3588 只有一个 NPU 上下文，请求必然串行（limits.
    # concurrency=1）。写进能力表 + 503 文案，避免客户端误判为可并行。
    "concurrency": "single NPU context: requests are serialized; concurrent "
                   "requests are queued (queue.max_wait) or rejected with 503",
    "json_schema": "prompt+validate+retry on non-stream requests; streaming "
                   "returns 400 (cannot correct frames already sent)",
    "tools": "prompt-injected; accuracy depends on model; handle-level "
             "rkllm_set_function_tools avoided by design",
    "seed": "RKLLM runtime exposes no RNG seed interface",
    "logprobs": "per-token logit copy too costly on 27B; not implemented",
}

CAPS: dict[str, Any] = {
    "endpoints": ENDPOINTS,
    "params": {
        "native": list(NATIVE_PARAMS),
        "emulated": list(EMULATED_PARAMS),
        "ignored": list(IGNORED_PARAMS),
        "unsupported": list(UNSUPPORTED_PARAMS),
    },
    "limits": LIMITS,
    "notes": NOTES,
}


# ---------------------------------------------------------------------------
# 查询与请求检查
# ---------------------------------------------------------------------------


def param_tier(param: str) -> str:
    """返回参数档位；未知参数按 'unsupported' 处理（诚实原则）。"""
    if param in _PARAM_TIER:
        return _PARAM_TIER[param]
    if param.startswith("response_format."):
        return _PARAM_TIER.get(param, "unsupported")
    return "unsupported"


def check_request_params(body: dict[str, Any], *,
                         policy: str = "warn",
                         stream: bool = False) -> tuple[list[str], list[str]]:
    """对请求体做参数档位检查，返回 (unsupported_to_reject, unsupported_to_warn)。

    策略（openai.param_policy）：
    - reject：所有 unsupported 项 400；
    - warn：无害项写 X-RKLLM-Unsupported 头放行，语义关键项仍 400。
    ignored 项永不进入这两个列表（D2 的反面：新 SDK 默认字段不能打挂服务）。

    json_schema 是 emulated（提示 + 校验 + 重试），只在**非流式**放行：
    流式响应一旦开始发帧，就无法在结尾"撤回并重答"，因此流式下明确拒绝而不是
    假装支持（设计 §6.1 emulated 的诚实实现方式）。
    """
    reject: list[str] = []
    warn: list[str] = []

    if body.get("n", 1) not in (1, None):
        reject.append("n>1")
    # tool_choice 与 json_schema 同属语义关键项：声明为「必须拦」就必须真的拦
    # ——旧实现只在注释里拒绝、代码里静默忽略（审计 P2-8）。
    tc = body.get("tool_choice")
    if tc is not None:
        # P1-15：只拒绝"强制指定"模式；auto/none 无强制语义，放行（新 SDK 默认
        # 发 tool_choice="auto"，一律 400 会打挂 langchain-openai / Agents SDK）。
        if isinstance(tc, str) and tc in ("auto", "none"):
            pass
        else:
            reject.append("tool_choice")
    for name in ("seed", "logprobs", "top_logprobs", "logit_bias",
                 "modalities", "audio", "prediction"):
        if name in body and body[name] not in (None, False, []):
            (reject if policy == "reject" else warn).append(name)
    rf = body.get("response_format") or {}
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        if stream:
            reject.append("response_format.json_schema(stream)")
        # 非流式：交由 Pipeline 注入提示 + 校验重试（emulated），不在此拦截
    elif isinstance(rf, dict) and rf.get("type") == "json_object":
        pass  # emulated：由 agent/pipeline.py 追加 json_hint 提示，不在此拦截
    for name in IGNORED_PARAMS:
        body.pop(name, None)   # 静默剥离：不生效、不告警、不打挂
    return reject, warn
