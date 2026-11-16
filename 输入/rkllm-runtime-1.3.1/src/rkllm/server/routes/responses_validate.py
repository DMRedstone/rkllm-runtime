"""Responses 入口校验（P2-3 拆分自 routes/responses.py）。

Responses 的字段集大于本项目的实现面；用 `extra="forbid"` 的 pydantic 模型会
把大量"可忽略字段"打成 422，故在路由层做定向校验：只拦会进入 runtime 或会
破坏响应形状的字段（input / max_output_tokens / 采样参数 / tools / stream /
tool_choice）。非法请求一律 OpenAI 400，绝不 500、不回显内部异常。
"""
from __future__ import annotations

import math

from fastapi.responses import JSONResponse

from ..errors import http_error as _http_error

__all__ = ["validate_body"]

_RESP_FLOAT_LIMITS = {"temperature": (0.0, 2.0), "top_p": (0.0, 1.0)}


def _bad(message: str, param: str | None,
         code: str = "invalid_value") -> JSONResponse:
    return _http_error(400, "invalid_request_error", message,
                       param=param, code=code)


def validate_body(body: dict) -> JSONResponse | None:
    """校验 Responses 请求体；返回错误响应或 None。"""
    if not isinstance(body, dict):
        return _bad("request body must be a JSON object", None,
                    code="invalid_request_error")
    raw_input = body.get("input")
    if raw_input is not None and not isinstance(raw_input, (str, list)):
        return _bad("`input` must be a string or an array of input items",
                    "input", code="invalid_input")
    if isinstance(raw_input, list) and any(
            not isinstance(item, dict) for item in raw_input):
        return _bad("`input` array items must be objects", "input",
                    code="invalid_input")

    mot = body.get("max_output_tokens")
    if mot is not None and (isinstance(mot, bool)
                            or not isinstance(mot, int) or mot < 1):
        return _bad("`max_output_tokens` must be a positive integer",
                    "max_output_tokens")

    for name, (low, high) in _RESP_FLOAT_LIMITS.items():
        value = body.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _bad("`%s` must be a number" % name, name)
        v = float(value)
        if not math.isfinite(v) or v < low or v > high or (
                name == "top_p" and v == low):
            return _bad("`%s` must be a finite number in [%s, %s]"
                        % (name, low, high), name)

    if body.get("tools") is not None and not isinstance(body.get("tools"), list):
        return _bad("`tools` must be an array", "tools")
    if body.get("stream") is not None and not isinstance(body.get("stream"),
                                                         bool):
        return _bad("`stream` must be a boolean", "stream")

    tc = body.get("tool_choice")
    if tc is not None and not (isinstance(tc, str) and tc in ("auto", "none")):
        return _http_error(
            400, "invalid_request_error",
            "tool_choice: prompt-injected tools only support 'auto'/'none'; "
            "required/named tool choice is not implemented",
            param="tool_choice", code="unsupported_parameter")
    return None
