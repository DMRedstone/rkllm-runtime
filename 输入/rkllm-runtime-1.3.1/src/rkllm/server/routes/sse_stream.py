"""OpenAI 两种信封共用的 SSE 生成器（P2-3 拆分）。

`routes/openai.py` 里 chat.completion.chunk 与 text_completion 两份生成器
此前近乎逐行重复（约 110 行）；此处收敛为一处，差异仅两点：
- 首帧 role（chat 为 "assistant"，completions 为 None，即不带 role）；
- 是否在 finish 前发 trailing_frames（仅 chat 需要）。

取消 / 心跳 / 断连 / 错误帧语义完全复用 `server.streaming`（F-1），本模块
只负责帧顺序，不解析模型输出。
"""
from __future__ import annotations

from fastapi import Request

from ..pipeline_runner import StreamTimer
from ..queue import QueueBusy, QueueTimeout
from ..sse import SseSession
from ..streaming import DISCONNECT, DONE, HEARTBEAT, TokenStream, iterate_stream
from ..supervisor import WorkerDown

__all__ = ["stream_events"]


def stream_events(request: Request, svc, pipe, session: SseSession, raw: dict, *,
                  first_role: str | None = "assistant", trailing: bool = True):
    """返回 SSE 异步生成器：首帧 → 增量(含心跳) → finish → usage → [DONE]。"""
    timer = StreamTimer()
    stream = TokenStream(svc, pipe, name="sse-pump",
                         on_slot_acquired=timer.slot_acquired)
    heartbeat_s = max(1, int(svc.stream_heartbeat_s))

    async def generate():
        yield session.first_frame(role=first_role)
        truncated = False
        async for kind, item in iterate_stream(request, stream, heartbeat_s):
            if kind == HEARTBEAT:
                yield session.heartbeat()
                continue
            if kind == DISCONNECT:
                timer.finish(svc, pipe)         # 断连也计入（NPU 已归还）
                return
            if kind == DONE:
                break
            if isinstance(item, BaseException):
                if isinstance(item, (QueueBusy, QueueTimeout, WorkerDown)):
                    yield session.error_frame(str(item)[:300])
                else:
                    yield session.error_frame("internal error")
                yield session.done()
                # 2026-11-20 审计：错误路径此前直接 return，首 token 与耗时
                # 不进指标（成功路径才计）。这里补一次收尾记账。
                timer.finish(svc, pipe)
                stream.mark_settled()           # 已发终帧：收尾不再 abort
                return
            if isinstance(item, dict):
                if item.get("truncated"):
                    truncated = True
                # P2-1：思考阶段的增量在生成时就下发（reasoning_content /
                # reasoning delta），不再等 trailing_frames 在正文后一次性回填。
                delta_reasoning = item.get("reasoning_delta")
                if delta_reasoning and session.reasoning_field != "none":
                    yield session.delta_frame(reasoning=delta_reasoning)
                continue                        # 管线元数据（tool_calls 等）
            timer.first_token()
            yield session.delta_frame(content=item)

        if trailing:
            for frame in pipe.trailing_frames(session):
                yield frame
        yield session.finish_frame("length" if truncated
                                   else pipe.finish_reason())
        want_usage = svc.include_usage_default or bool(
            (raw.get("stream_options") or {}).get("include_usage"))
        if want_usage:
            yield session.usage_frame(pipe.usage())
        timer.finish(svc, pipe)
        yield session.done()

    return generate()
