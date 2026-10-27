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
import math

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent import responses as tr
from ...agent.pipeline import Pipeline
from ..errors import http_error as _http_error
from ..pipeline_runner import (BUSY_MESSAGE, SchemaError as _SchemaError,
                               StreamTimer,
                               build_pipeline as _pipeline,
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


def _error(status: int, message: str, code: str, *,
           err_type: str = "server_error",
           headers: dict[str, str] | None = None, **extra) -> JSONResponse:
    return _http_error(status, err_type, message, code=code,
                       headers=headers, **extra)


_RESP_FLOAT_LIMITS = {"temperature": (0.0, 2.0), "top_p": (0.0, 1.0)}


def _bad(message: str, param: str | None,
         code: str = "invalid_value") -> JSONResponse:
    return _http_error(400, "invalid_request_error", message,
                       param=param, code=code)


def _validate_body(body: dict) -> JSONResponse | None:
    """Responses 入口校验：非法请求一律 OpenAI 400，绝不 500/回显内部异常。

    为什么在路由层做而不是 pydantic 模型：Responses 的字段集大于本项目实现面，
    用 `extra="forbid"` 的模型会把大量“可忽略字段”打成 422；这里只校验会进入
    runtime 或会破坏响应形状的字段（input / max_output_tokens / 采样参数 /
    tools / stream / tool_choice）。
    """
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


@router.post("/responses")
def create_response(request: Request, body: dict):
    svc = _svc(request)
    invalid = _validate_body(body)
    if invalid is not None:
        return invalid
    chat_body, ignored = tr.to_chat_request(body)
    if chat_body.get("tool_choice") == "none":
        # 与 chat 路径一致：tool_choice=none 表示禁止调用工具，故不注入工具。
        chat_body.pop("tools", None)
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
                           code="unsupported_parameter",
                           param=reject[0])
    headers = _headers(ignored + warn)

    resp_id = tr.response_id()
    created = tr.now()

    if streaming:
        try:
            pipe = _pipeline(svc, chat_body, chat=True)
        except ValueError as exc:
            return _http_error(400, "invalid_request_error", str(exc))
        if pipe.max_tokens_clamped:
            headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
        return StreamingResponse(
            _stream(request, svc, pipe, body, resp_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     **headers})

    # 非流式：复用 chat 的 json_schema 校验重试（审计 P3-7）。
    # 这里只构造一次 Pipeline（此前先建一次做预算、再交给
    # _run_blocking_checked 建第二次，效率项「双构造」在 Responses 未收口）。
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
        return _error(400, str(exc), "invalid_request",
                      err_type="invalid_request_error")
    except _SchemaError as exc:
        return _http_error(422, "invalid_request_error",
                           "response did not match response_format.json_schema "
                           "after retries: %s" % exc, code="schema_mismatch")
    except QueueBusy as exc:
        return _error(503, BUSY_MESSAGE, "queue_full",
                      headers={"Retry-After": str(exc.retry_after_s)})
    except QueueTimeout:
        return _error(503, "queue timeout", "queue_timeout",
                      headers={"Retry-After":
                               str(svc.queue.busy_retry_after_s)})
    except WorkerDown as exc:
        return _error(503, str(exc)[:300], "worker_down",
                      headers={"Retry-After":
                               str(svc.queue.busy_retry_after_s)})

    if pipe.max_tokens_clamped:
        headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
    # max_output_tokens 截断：与 chat 的 finish_reason 同口径，映射到 Responses
    # 的 incomplete 状态（此前固定写 completed，客户端无法感知截断）。
    finish_reason = pipe.finish_reason(result)
    status = "incomplete" if finish_reason == "length" else "completed"
    incomplete_details = ({"reason": "max_output_tokens"}
                          if status == "incomplete" else None)
    content = result.get("text", "")
    if "json" in result:
        # json_schema 校验通过后回填解析结果，保证响应文本本身是合法 JSON。
        content = json.dumps(result["json"], ensure_ascii=False)
    chat_payload = {"choices": [{"message": {
        "role": "assistant",
        "content": content,
        "tool_calls": result.get("tool_calls") or [],
    }, "finish_reason": finish_reason}]}
    output, _ = tr.chat_to_output(chat_payload)
    if status == "incomplete":
        for item in output:
            if item.get("type") == "message":
                item["status"] = "incomplete"
    payload = tr.response_object(
        resp_id=resp_id, model=body.get("model", svc.model_name),
        created_at=created, status=status, output=output,
        usage=tr.usage_of(pipe.usage(result)), request_body=body,
        incomplete_details=incomplete_details)
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
        truncated = False
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
                if item.get("truncated"):
                    truncated = True
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
        # max_output_tokens 触顶（runtime 的 generate_tokens 到达上限）与背压
        # 截断同属“服务端没送完”：统一映射 status=incomplete / reason。
        if pipe.finish_reason() == "length":
            truncated = True
        item_status = "incomplete" if truncated else "completed"
        yield ev("response.output_text.done", {
            "item_id": message_id, "output_index": 0,
            "content_index": 0, "text": text})
        yield ev("response.content_part.done", {
            "item_id": message_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": text,
                     "annotations": []}})
        yield ev("response.output_item.done", {
            "output_index": 0,
            "item": tr.message_item(text, item_id=message_id,
                                    status=item_status)})

        output = [tr.message_item(text, item_id=message_id,
                                  status=item_status)]
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
            created_at=created, status=item_status, output=output,
            usage=tr.usage_of(pipe.usage({"text": text})), request_body=body,
            incomplete_details=({"reason": "max_output_tokens"}
                                if truncated else None))
        timer.finish(svc, pipe)
        # 背压截断不是正常完成：按 Responses 契约发 response.incomplete，
        # 让客户端能区分「模型说完了」与「服务端没来得及送完」。
        yield ev("response.incomplete" if truncated else "response.completed",
                 {"response": final})
        yield "data: [DONE]\n\n"

    return generate()
