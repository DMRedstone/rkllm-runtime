"""SSE 帧工具：固定 id/created（修 D4）、心跳、错误帧、[DONE]（D9）。"""

from __future__ import annotations

import json
import time
from typing import Any

__all__ = ["SseSession"]


class SseSession:
    """一次 completion 的 SSE 状态：id/created 全程一致（D4）。

    `object_name` / `delta_field` 区分两种 OpenAI 信封：chat 是
    chat.completion.chunk + delta.content；completions 是
    text_completion + delta.text（无 role 首帧）。
    """

    def __init__(self, completion_id: str | None = None, model: str = "rkllm",
                 reasoning_field: str = "reasoning_content",
                 object_name: str = "chat.completion.chunk",
                 delta_field: str = "content"):
        self.id = completion_id or ("chatcmpl-" + id(self).__format__("x"))
        self.created = int(time.time())
        self.model = model
        # openai.reasoning_field：reasoning_content（DeepSeek 约定）/reasoning/none
        self.reasoning_field = reasoning_field
        self.object_name = object_name
        self.delta_field = delta_field
        self.index = 0

    def first_frame(self, role: str | None = "assistant") -> str:
        delta = {self.delta_field: ""}
        if role:
            delta["role"] = role
        return self._frame(delta, finish_reason=None)

    def delta_frame(self, content: str | None = None, *,
                    reasoning: str | None = None,
                    tool_calls: list | None = None) -> str:
        delta: dict[str, Any] = {}
        if content is not None:
            delta[self.delta_field] = content
        if reasoning is not None and self.reasoning_field != "none":
            delta[self.reasoning_field] = reasoning
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        return self._frame(delta, finish_reason=None)

    def finish_frame(self, finish_reason: str = "stop") -> str:
        return self._frame({}, finish_reason=finish_reason)

    def usage_frame(self, usage: dict) -> str:
        return self._frame({}, finish_reason=None, usage=usage)

    def error_frame(self, message: str) -> str:
        # 已发 200 后的失败只能走错误帧（D8）
        return "data: %s\n\n" % json.dumps(
            {"error": {"message": message, "type": "server_error"}},
            ensure_ascii=False)

    def heartbeat(self) -> str:
        return ": ping\n\n"                     # 注释帧：SSE 合法心跳（D9）

    def done(self) -> str:
        return "data: [DONE]\n\n"

    def _frame(self, delta: dict, finish_reason: str | None,
               usage: dict | None = None) -> str:
        chunk: dict[str, Any] = {
            "id": self.id, "object": self.object_name,
            "created": self.created, "model": self.model,
            "choices": [{"index": self.index, "delta": delta,
                         "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        return "data: %s\n\n" % json.dumps(chunk, ensure_ascii=False)


def sse_response_headers() -> dict[str, str]:
    return {"Cache-Control": "no-cache", "Connection": "keep-alive",
            "X-Accel-Buffering": "no"}
