"""`POST /v1/responses` —— Responses API 翻译层（emulated，02 §4）。

实现方式：翻译成 chat 请求，走同一条 Pipeline（同一个队列、同一套取消与
token 流），再把结果/增量翻译回 Responses 结构与事件序列。

不承诺等价（caps 标 emulated）：
- 无服务端会话：`store`、`previous_response_id` 一律忽略，并在响应头
  `X-RKLLM-Unsupported` 里列出来；
- 事件集会覆盖 Agents SDK 需要的核心序列（created / in_progress /
  output_item.* / output_text.* / function_call_arguments.* / completed），
  但不发 `response.reasoning_summary.*`——板端不做摘要式推理输出。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent import responses as tr
from ...agent.pipeline import Pipeline
from ..pipeline_runner import (BUSY_MESSAGE, SchemaError as _SchemaError,
                               StreamTimer,
                               build_pipeline as _pipeline,
                               http_error as _http_error,
                               run_blocking_checked as _run_blocking_checked,
                               service_of as _svc)
from ..queue import QueueBusy, QueueTimeout
from ..streaming import (DISCONNECT, DONE, HEARTBEAT, TokenStream,
                         iterate_stream)
from ..supervisor import WorkerDown

router = APIRouter(prefix="/v1")

log = logging.getLogger("rkllm.server.responses")


def _sse(event: str, payload: dict) -> str:
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(payload,
                                                          ensure_ascii=False))


def _headers(ignored: list[str]) -> dict:
    headers = {"X-RKLLM-Emulated": "responses"}
    if ignored:
        headers["X-RKLLM-Unsupported"] = ",".join(ignored)
    return headers


def _error(status: int, message: str, code: str, **extra) -> JSONResponse:
    return _http_error(status, "server_error", message, code=code, **extra)


@router.post("/responses")
def create_response(request: Request, body: dict):
    svc = _svc(request)
    chat_body, ignored = tr.to_chat_request(body)
    streaming = bool(body.get("stream"))
    raw_input = body.get("input")
    blank = (raw_input is None or raw_input == [] or
             (isinstance(raw_input, str) and not raw_input.strip()))
    if blank or not chat_body.get("messages"):
        return _http_error(400, "invalid_request_error",
                           "`input` 为空：Responses 请求必须带 input",
                           code="invalid_input")
    # 参数策略与 chat 路径共用（同一张能力表）
    from ... import caps
    reject, warn = caps.check_request_params(chat_body,
                                             policy=svc.param_policy,
                                             stream=streaming)
    if reject:
        return _http_error(400, "invalid_request_error",
                           "unsupported parameter(s): %s" % ", ".join(reject),
                           code="unsupported_parameter")
    headers = _headers(ignored + warn)

    try:
        pipe = _pipeline(svc, chat_body, chat=True)
    except ValueError as exc:
        return _http_error(400, "invalid_request_error", str(exc))

    resp_id = tr.response_id()
    created = tr.now()

    if streaming:
        return StreamingResponse(
            _stream(request, svc, pipe, body, resp_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     **headers})

    # 非流式：复用 chat 的 json_schema 校验重试（审计 P3-7）
    max_tokens = (chat_body.get("max_tokens")
                  or svc.default_max_tokens)
    rf = chat_body.get("response_format") or {}
    schema = None
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        spec = rf.get("json_schema") or {}
        schema = spec.get("schema") if isinstance(spec, dict) else None
    try:
        result, pipe = _run_blocking_checked(svc, chat_body, max_tokens, schema)
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request")
    except _SchemaError as exc:
        return _http_error(422, "invalid_request_error",
                           "response did not match response_format.json_schema "
                           "after retries: %s" % exc, code="schema_mismatch")
    except QueueBusy as exc:
        return _error(503, BUSY_MESSAGE, "queue_full",
                      retry_after_s=exc.retry_after_s)
    except QueueTimeout:
        return _error(503, "queue timeout", "queue_timeout")
    except WorkerDown as exc:
        return _error(503, str(exc)[:300], "worker_down")

    chat_payload = {"choices": [{"message": {
        "role": "assistant",
        "content": result.get("text", ""),
        "tool_calls": result.get("tool_calls") or [],
    }, "finish_reason": ("tool_calls" if result.get("tool_calls") else "stop")}]}
    output, _ = tr.chat_to_output(chat_payload)
    payload = tr.response_object(
        resp_id=resp_id, model=body.get("model", svc.model_name),
        created_at=created, status="completed", output=output,
        usage=tr.usage_of(pipe.usage(result)), request_body=body)
    return JSONResponse(payload, headers=headers)


def _stream(request: Request, svc, pipe: Pipeline, body: dict,
            resp_id: str, created: int):
    """SSE 事件流：把 chat 增量翻译成 Responses 事件序列。

    F-1：泵线程、心跳计时、断连取消统一由 `server.streaming` 提供——与
    `/v1/chat/completions` 走同一套实现（此前两处各写一份，取消路径漂移）。
    """
    heartbeat_s = max(1, int(svc.stream_heartbeat_s))
    stream = TokenStream(svc, pipe, name="responses-pump")

    def item_added(item: dict, index: int) -> str:
        return _sse("response.output_item.added",
                    {"type": "response.output_item.added",
                     "output_index": index, "item": item})

    async def generate():
        timer = StreamTimer()
        seq = 0

        def ev(name: str, payload: dict) -> str:
            nonlocal seq
            payload = {"type": name, "sequence_number": seq, **payload}
            seq += 1
            return _sse(name, payload)

        message_id = "msg_" + resp_id[5:29]
        shell = tr.response_object(resp_id=resp_id,
                                   model=body.get("model", svc.model_name),
                                   created_at=created, status="in_progress",
                                   output=[], request_body=body)
        yield ev("response.created", {"response": shell})
        yield ev("response.in_progress", {"response": shell})
        yield ev("response.output_item.added", {
            "output_index": 0,
            "item": {"id": message_id, "type": "message", "status": "in_progress",
                     "role": "assistant", "content": []}})
        yield ev("response.content_part.added", {
            "item_id": message_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}})

        parts: list[str] = []
        calls: list[dict] = []
        failure: BaseException | None = None
        async for kind, item in iterate_stream(request, stream, heartbeat_s):
            if kind == HEARTBEAT:
                yield ": ping\n\n"        # 长 prefill 保活（与 chat 一致）
                continue
            if kind == DISCONNECT:
                timer.finish(svc, pipe)   # 已两段式取消并归还 NPU
                return
            if kind == DONE:
                break
            if isinstance(item, BaseException):
                failure = item
                break
            if isinstance(item, dict):
                if item.get("tool_calls"):
                    calls = item["tool_calls"]
                continue
            timer.first_token()
            parts.append(item)
            yield ev("response.output_text.delta", {
                "item_id": message_id, "output_index": 0,
                "content_index": 0, "delta": item})

        text = "".join(parts)
        if failure is not None:
            err = ("internal error" if not isinstance(
                failure, (QueueBusy, QueueTimeout, WorkerDown))
                else str(failure)[:300])
            yield ev("response.failed", {
                "response": {**shell, "status": "failed",
                             "error": {"code": "stream_error", "message": err}}})
            timer.finish(svc, pipe)
            yield "data: [DONE]\n\n"
            stream.mark_settled()         # 已发终帧：收尾不再 abort
            return

        # 文本消息收尾
        yield ev("response.output_text.done", {
            "item_id": message_id, "output_index": 0,
            "content_index": 0, "text": text})
        yield ev("response.content_part.done", {
            "item_id": message_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": text,
                     "annotations": []}})
        yield ev("response.output_item.done", {
            "output_index": 0,
            "item": tr.message_item(text, item_id=message_id)})

        output = [tr.message_item(text, item_id=message_id)]
        for offset, call in enumerate(calls, start=1):
            fc = tr.function_call_item(call)
            output.append(fc)
            yield item_added({**fc, "status": "in_progress",
                              "arguments": ""}, offset)
            yield ev("response.function_call_arguments.delta", {
                "item_id": fc["id"], "output_index": offset,
                "delta": fc["arguments"]})
            yield ev("response.function_call_arguments.done", {
                "item_id": fc["id"], "output_index": offset,
                "arguments": fc["arguments"]})
            yield ev("response.output_item.done", {"output_index": offset,
                                                   "item": fc})

        final = tr.response_object(
            resp_id=resp_id, model=body.get("model", svc.model_name),
            created_at=created, status="completed", output=output,
            usage=tr.usage_of(pipe.usage({"text": text})), request_body=body)
        timer.finish(svc, pipe)
        yield ev("response.completed", {"response": final})
        yield "data: [DONE]\n\n"

    return generate()
