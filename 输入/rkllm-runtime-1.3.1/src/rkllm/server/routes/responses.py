"""`POST /v1/responses` —— Responses API 翻译层（emulated，02 §4）。

实现方式：翻译成 chat 请求，走同一条 Pipeline（同一个队列、同一套取消与
token 流），再把结果/增量翻译回 Responses 结构与事件序列。

不承诺等价（caps 标 emulated）：
- 无服务端会话：`store`、`previous_response_id` 一律忽略，并在响应头
  `X-RKLLM-Unsupported` 里列出来；
- 事件集会覆盖 Agents SDK 需要的核心序列（created / in_progress /
  output_item.* / output_text.* / function_call_arguments.* / completed）；
- 思考内容（P2-1）：开启 reasoning_effort/enable_thinking 时按生成顺序**逐帧**
  下发 `response.reasoning_text.delta`（先 reasoning item，后 message item），
  终帧回填 `response.reasoning_text.done` 与 output_item.done；不发
  `response.reasoning_summary.*`——板端不做摘要式推理输出。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent import responses as tr
from ...agent.pipeline import Pipeline
from ..errors import JSON_OBJECT_SCHEMA as _JSON_OBJECT_SCHEMA
from ..errors import http_error as _http_error
from ..errors import validate_response_format as _validate_response_format
from ..errors import validate_text_format as _validate_text_format
from ..pipeline_runner import (BUSY_MESSAGE,
                               DISCONNECTED as _DISCONNECTED,
                               SchemaError as _SchemaError,
                               StreamTimer,
                               build_pipeline as _pipeline,
                               client_disconnected_response as _client_disconnected,
                               run_blocking_checked as _run_blocking_checked,
                               run_blocking_disconnect as _run_blocking_disconnect,
                               service_of as _svc)
from ..queue import QueueBusy, QueueTimeout
from ..streaming import (DISCONNECT, DONE, HEARTBEAT, TokenStream,
                         iterate_stream)
from ..supervisor import WorkerDown
from .responses_validate import validate_body as _validate_body

# 前缀由 app.create_app 在挂载时给（同一 router 挂 /v1 与根路径两份），
# 这里不再自带 prefix，否则双前缀会叠成 /v1/v1/...
router = APIRouter()

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


# 入口校验已移至 routes/responses_validate.py（P2-3 拆分）。


@router.post("/responses")
async def create_response(request: Request, body: dict):
    svc = _svc(request)
    invalid = _validate_body(body)
    if invalid is not None:
        return invalid
    # P2-1：非法 text.format 必须显式 400。agent 的 to_chat_request 对不认识
    # 的 format 是静默跳过的——静默忽略会让客户端以为约束生效了。
    _, _, format_error = _validate_text_format(body.get("text"))
    if format_error is not None:
        return format_error
    chat_body, ignored = tr.to_chat_request(body)
    # 翻译层只映射了有限的抽样参数；这里把 Responses 也支持的字段补进 chat
    # 请求体，让 caps/请求参数检查与真正的采样生效口径一致（P2-1）。
    for name in ("repeat_penalty", "top_k", "parallel_tool_calls"):
        if body.get(name) is not None:
            chat_body[name] = body[name]
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
                           caps.reject_message(reject, chat_body),
                           code="unsupported_parameter",
                           param=reject[0])
    json_mode, schema, fmt_error = _validate_response_format(
        chat_body.get("response_format"))
    if fmt_error is not None:
        return fmt_error
    headers = _headers(ignored + warn)

    resp_id = tr.response_id()
    created = tr.now()

    if streaming:
        try:
            pipe = _pipeline(svc, chat_body, chat=True, endpoint="responses")
        except ValueError as exc:
            return _http_error(400, "invalid_request_error", str(exc))
        if pipe.max_tokens_clamped:
            headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
        return StreamingResponse(
            _stream(request, svc, pipe, body, resp_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     **headers})

    # 非流式：复用 chat 的 json_schema / json_object 校验重试（审计 P3-7 / P2-1）。
    # 这里只构造一次 Pipeline（此前先建一次做预算、再交给
    # _run_blocking_checked 建第二次，效率项「双构造」在 Responses 未收口）。
    max_tokens = (chat_body.get("max_tokens")
                  or svc.default_max_tokens)
    checked_schema = (schema if json_mode == "json_schema"
                      else (_JSON_OBJECT_SCHEMA
                            if json_mode == "json_object" else None))
    try:
        # P2.2：非流式也监听 http.disconnect，200 前断开即取消/abort worker。
        outcome = await _run_blocking_disconnect(
            request,
            lambda **kw: _run_blocking_checked(svc, chat_body, max_tokens,
                                               checked_schema,
                                               endpoint="responses", **kw),
            supervisor=svc.supervisor)
        if outcome is _DISCONNECTED:
            return _client_disconnected()
        result, pipe = outcome
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request",
                      err_type="invalid_request_error")
    except _SchemaError as exc:
        if json_mode == "json_object":
            return _http_error(
                422, "invalid_request_error",
                "model output was not a valid JSON object after retries: %s"
                % exc, code="json_object_invalid")
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
    # P2-1：思考内容的顺序是「先 reasoning 后 message」，非流式也按同序输出
    # reasoning item（此前 reasoning 只在 chat 的 message 字段里，Responses
    # 完全没有；与流式事件序列保持一致）。
    reasoning_text = pipe.reasoning_text
    if reasoning_text and svc.reasoning_field != "none":
        output.insert(0, {"id": "rs_" + resp_id[5:29], "type": "reasoning",
                          "status": "completed",
                          "content": [{"type": "reasoning_text",
                                       "text": reasoning_text}]})
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

    async def generate():
        timer = StreamTimer()
        seq = 0

        def ev(name: str, payload: dict) -> str:
            nonlocal seq
            payload = {"type": name, "sequence_number": seq, **payload}
            seq += 1
            return _sse(name, payload)

        def item_added(item: dict, index: int) -> str:
            # 走 ev()：所有事件都带 sequence_number（Agents SDK/严格客户端依赖）。
            return ev("response.output_item.added",
                      {"output_index": index, "item": item})

        message_id = "msg_" + resp_id[5:29]
        reasoning_id = "rs_" + resp_id[5:29]
        # P2-1：思考开启时先声明 reasoning item（output_index 0），message 顺延
        # 到 1——与真实生成顺序一致（先思考、后正文）。未开启思考时编号不变。
        thinking = (bool(getattr(pipe, "thinking_enabled", False))
                    and getattr(svc, "reasoning_field", "none") != "none")
        message_index = 1 if thinking else 0
        shell = tr.response_object(resp_id=resp_id,
                                   model=body.get("model", svc.model_name),
                                   created_at=created, status="in_progress",
                                   output=[], request_body=body)
        yield ev("response.created", {"response": shell})
        yield ev("response.in_progress", {"response": shell})
        if thinking:
            yield item_added({"id": reasoning_id, "type": "reasoning",
                              "status": "in_progress", "content": []}, 0)
        yield ev("response.output_item.added", {
            "output_index": message_index,
            "item": {"id": message_id, "type": "message", "status": "in_progress",
                     "role": "assistant", "content": []}})
        yield ev("response.content_part.added", {
            "item_id": message_id, "output_index": message_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}})

        parts: list[str] = []
        reasoning_parts: list[str] = []
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
                delta_reasoning = item.get("reasoning_delta")
                if delta_reasoning and thinking:
                    # 思考阶段逐帧下发 reasoning 增量（不再终点一次性回填）。
                    reasoning_parts.append(delta_reasoning)
                    yield ev("response.reasoning_text.delta", {
                        "item_id": reasoning_id, "output_index": 0,
                        "content_index": 0, "delta": delta_reasoning})
                if item.get("tool_calls"):
                    calls = item["tool_calls"]
                continue
            timer.first_token()
            parts.append(item)
            yield ev("response.output_text.delta", {
                "item_id": message_id, "output_index": message_index,
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
        reasoning_text = "".join(reasoning_parts)
        reasoning_item: dict | None = None
        if thinking:
            reasoning_item = {
                "id": reasoning_id, "type": "reasoning", "status": "completed",
                "content": [{"type": "reasoning_text",
                             "text": reasoning_text}]}
            yield ev("response.reasoning_text.done", {
                "item_id": reasoning_id, "output_index": 0,
                "content_index": 0, "text": reasoning_text})
            yield ev("response.output_item.done", {"output_index": 0,
                                                   "item": reasoning_item})
        yield ev("response.output_text.done", {
            "item_id": message_id, "output_index": message_index,
            "content_index": 0, "text": text})
        yield ev("response.content_part.done", {
            "item_id": message_id, "output_index": message_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": text,
                     "annotations": []}})
        yield ev("response.output_item.done", {
            "output_index": message_index,
            "item": tr.message_item(text, item_id=message_id,
                                    status=item_status)})

        output = ([reasoning_item] if reasoning_item is not None else []) + [
            tr.message_item(text, item_id=message_id, status=item_status)]
        for offset, call in enumerate(calls, start=message_index + 1):
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
