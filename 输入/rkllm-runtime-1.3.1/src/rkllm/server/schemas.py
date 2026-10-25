"""pydantic 请求模型 + 参数策略（D2/D7：不校验与静默忽略都不可接受）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

__all__ = ["ChatCompletionRequest", "CompletionRequest"]


class _OpenAICompat:
    """chat 与 completions 共享的 OpenAI 兼容字段（caps ignored/unsupported 档）。

    两处曾各写一份且漏了 6 个字段（top_logprobs/modalities/audio/n/seed/
    logprobs + stream_options），`extra:"forbid"` 下客户端一发就 422，caps 的
    reject/warn 策略走不到——2026-10-25 审计发现后合并为单一来源，杜绝再漂移。

    stream_options 属 native 档（流式 usage），其余：ignored 档必须能进 schema
    才能被 caps 剥离（否则 pydantic 先 422，与「新 SDK 默认字段不能打挂服务」
    的 caps 口径矛盾，DSH 发送 store 即中招）；unsupported 档同样先接受，
    交给 caps 按 param_policy 返回 400/警告。
    """

    stream_options: dict | None = None
    user: str | None = None
    metadata: dict | None = None
    store: bool | None = None
    service_tier: str | None = None
    verbosity: str | None = None
    top_logprobs: int | None = None
    logit_bias: dict | None = None
    modalities: list[str] | None = None
    audio: dict | None = None
    prediction: dict | None = None
    n: int | None = None
    seed: int | None = None
    logprobs: bool | None = None


class ChatCompletionRequest(_OpenAICompat, BaseModel):
    model: str = "rkllm"
    messages: list[dict]
    stream: bool = False
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


class CompletionRequest(_OpenAICompat, BaseModel):
    model: str = "rkllm"
    prompt: str
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None

    model_config = {"extra": "forbid"}
