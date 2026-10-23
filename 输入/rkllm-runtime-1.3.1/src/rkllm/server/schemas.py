"""pydantic 请求模型 + 参数策略（D2/D7：不校验与静默忽略都不可接受）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .. import caps

__all__ = ["ChatCompletionRequest", "CompletionRequest"]


class ChatCompletionRequest(BaseModel):
    model: str = "rkllm"
    messages: list[dict]
    stream: bool = False
    stream_options: dict | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=128)
    repeat_penalty: float | None = Field(default=None, ge=0.5, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    response_format: dict | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    enable_thinking: bool | None = None
    # ignored 档：必须能进 schema，才能被 caps 剥离；否则 pydantic 先 422，
    # 与「新 SDK 默认字段不能打挂服务」的 caps 口径矛盾（DSH 发送 store 即中招）。
    user: str | None = None
    metadata: dict | None = None
    store: bool | None = None
    service_tier: str | None = None
    verbosity: str | None = None
    # unsupported 档：同样先接受，交给 caps 按 param_policy 返回 400/警告
    top_logprobs: int | None = None
    logit_bias: dict | None = None
    modalities: list[str] | None = None
    audio: dict | None = None
    prediction: dict | None = None
    # ignored 档（caps 剥离后进不了业务逻辑）
    n: int | None = None
    seed: int | None = None
    logprobs: bool | None = None

    model_config = {"extra": "forbid"}

    @field_validator("messages")
    @classmethod
    def _messages_nonempty(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages 不能为空")
        for m in v:
            if not isinstance(m, dict) or "role" not in m:
                raise ValueError("messages 元素必须含 role")
        return v


class CompletionRequest(BaseModel):
    model: str = "rkllm"
    prompt: str
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    # 与 chat 同一套 ignored/unsupported 兼容字段
    user: str | None = None
    metadata: dict | None = None
    store: bool | None = None
    service_tier: str | None = None
    verbosity: str | None = None
    logit_bias: dict | None = None
    prediction: dict | None = None

    model_config = {"extra": "forbid"}
