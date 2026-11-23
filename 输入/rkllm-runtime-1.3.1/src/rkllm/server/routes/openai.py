"""OpenAI 端点：chat/completions、completions、models（F9）。

流式：SseSession 固定 id（D4）+ 心跳（D9）+ 错误帧（D8）+ 可选 usage 帧。
S3 的 agent 能力（tools 解析 / reasoning / json_schema）在推理管线里以
可插拔管线接入：本文件只负责 OpenAI 信封与流式帧，不解析模型输出。

本批新增（审计整改）：
- 心跳：token 拉取放在工作线程，主生成器按 `openai.stream_heartbeat_s` 超时
  发注释帧——无 token 期间也有字节，27B 慢流不会被客户端误判超时；
- 断连传导：生成器被关闭（客户端断开）时调 supervisor.abort()，释放 NPU；
- json_schema（非流式）：提示注入 + 校验 + 重试，重试次数取
  `openai.json_schema_retries`（流式在 caps 层直接 400，见 caps.py）；
- usage 帧：`stream_options.include_usage` 或 `openai.include_usage_default`。
"""
from __future__ import annotations

import json
import logging
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent.pipeline import Pipeline
from ..errors import JSON_OBJECT_SCHEMA as _JSON_OBJECT_SCHEMA
from ..errors import http_error as _http_error
from ..errors import validate_response_format as _validate_response_format
from ..pipeline_runner import (BUSY_MESSAGE,
                               DISCONNECTED as _DISCONNECTED,
                               SchemaError as _SchemaError,
                               build_pipeline as _pipeline,
                               client_disconnected_response as _client_disconnected,
                               run_blocking as _run_blocking,
                               run_blocking_checked as _run_blocking_checked,
                               run_blocking_disconnect as _run_blocking_disconnect,
                               service_of as _svc)
from ..schemas import ChatCompletionRequest, CompletionRequest
from ..sse import SseSession, sse_response_headers
from ..queue import QueueBusy, QueueTimeout
from ..supervisor import WorkerDown
from ... import caps
from .sse_stream import stream_events

# 前缀由 app.create_app 在挂载时给（同一 router 挂 /v1 与根路径两份），
# 这里不再自带 prefix，否则双前缀会叠成 /v1/v1/...
router = APIRouter()

log = logging.getLogger("rkllm.server.openai")

# /v1/models 的 `created` 是模型时间戳：优先取模型文件 mtime（稳定可复现），
# 文件不可读时退回进程启动时间（同一进程内 /v1/models 与 /models 必须一致）。
_STARTED_AT = int(time.time())


def _model_object(svc) -> dict:
    try:
        created = int(os.stat(svc.model_path).st_mtime)
    except (OSError, TypeError, ValueError):
        created = _STARTED_AT
    return {"id": svc.model_name, "object": "model", "created": created,
            "owned_by": "rkllm-runtime", "context_len": svc.context_len,
            "model_class": svc.model_class}


@router.get("/models")
def list_models(request: Request):
    return {"object": "list", "data": [_model_object(_svc(request))]}


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request):
    """单模型：任意 id 返回同一对象（设计 §4.1）。"""
    return _model_object(_svc(request))


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    svc = _svc(request)
    raw = body.model_dump()
    streaming = bool(raw.get("stream"))
    # caps 参数策略：unsupported 按 reject/warn，ignored 剥离（D2）
    reject, warn = caps.check_request_params(raw, policy=svc.param_policy,
                                            stream=streaming)
    if reject:
        return _http_error(400, "invalid_request_error",
                           caps.reject_message(reject, raw),
                           code="unsupported_parameter",
                           param=reject[0])
    # P2-1：非法 response_format（缺 schema / schema 不是对象 / 类型拼错）
    # 必须显式 400，不能静默当成"没有 response_format"。
    json_mode, schema, format_error = _validate_response_format(
        raw.get("response_format"))
    if format_error is not None:
        return format_error
    if raw.get("tool_choice") == "none":
        # tool_choice=none 的契约是“禁止调用工具”。runtime 没有请求级工具开关，
        # 这里做等效实现：不向 Pipeline 注入工具（也就不会装解析器），模型即使
        # 输出 <tool_call> 文本也只作为普通 content 返回，不会产生 tool_calls。
        raw["tools"] = None
    max_tokens = raw.get("max_tokens") or raw.get("max_completion_tokens") \
        or svc.default_max_tokens
    headers = {"X-RKLLM-Unsupported": ",".join(warn)} if warn else {}

    if streaming:
        # 效率：仅流式路径构造 Pipeline；非流式交给 _run_blocking_checked 自行
        # 构建，避免重复做一次全量 prompt 构建（原实现两条路径各建一次）。
        try:
            pipe = _pipeline(svc, {**raw, "max_tokens": max_tokens}, chat=True,
                               endpoint="chat.completions")
        except ValueError as exc:
            return _http_error(400, "invalid_request_error", str(exc))
        if pipe.max_tokens_clamped:
            headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
        return StreamingResponse(
            _stream_chat(request, svc, pipe, raw),
            media_type="text/event-stream",
            headers={**sse_response_headers(), **headers})

    checked_schema = (schema if json_mode == "json_schema"
                      else (_JSON_OBJECT_SCHEMA
                            if json_mode == "json_object" else None))
    try:
        # P2.2：非流式也监听 http.disconnect——客户端在 200 前断开即取消排队/
        # abort worker，不再把请求算完（或干等 queue.request_timeout_s）。
        outcome = await _run_blocking_disconnect(
            request,
            lambda **kw: _run_blocking_checked(svc, raw, max_tokens,
                                               checked_schema, **kw),
            supervisor=svc.supervisor)
        if outcome is _DISCONNECTED:
            return _client_disconnected()
        result, pipe = outcome
    except ValueError as exc:
        return _http_error(400, "invalid_request_error", str(exc))
    except _SchemaError as exc:
        if json_mode == "json_object":
            # P2-1：json_object 也必须保证 content 是合法 JSON 对象；重试耗尽
            # 后明确失败，不能把非 JSON 文本当成功返回（客户端 json.loads 崩）。
            return _http_error(
                422, "invalid_request_error",
                "model output was not a valid JSON object after retries: %s"
                % exc, code="json_object_invalid")
        return _http_error(422, "invalid_request_error",
                           "response did not match response_format.json_schema "
                           "after retries: %s" % exc, code="schema_mismatch")
    except QueueBusy as exc:
        return _http_error(503, "server_error", BUSY_MESSAGE,
                           code="queue_full",
                           headers={"Retry-After": str(exc.retry_after_s)})
    except QueueTimeout:
        return _http_error(503, "server_error", "queue timeout",
                           code="queue_timeout",
                           headers={"Retry-After":
                                    str(svc.queue.busy_retry_after_s)})
    except WorkerDown as exc:
        return _http_error(503, "server_error", str(exc)[:300],
                           code="worker_down",
                           headers={"Retry-After":
                                    str(svc.queue.busy_retry_after_s)})
    if pipe.max_tokens_clamped:
        headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
    return JSONResponse(_completion_payload(svc, pipe, result,
                                            pipe.max_tokens or max_tokens),
                        headers=headers)


def _stream_chat(request: Request, svc, pipe: Pipeline, raw: dict):
    """chat.completion.chunk 信封的 SSE 生成器（帧序列见 routes/sse_stream）。"""
    session = SseSession(model=svc.model_name,
                         reasoning_field=svc.reasoning_field)
    return stream_events(request, svc, pipe, session, raw,
                         first_role="assistant", trailing=True)


def _completion_payload(svc, pipe: Pipeline, result: dict,
                        max_tokens: int) -> dict:
    usage = pipe.usage(result)
    content = result["text"]
    if "json" in result:
        # json_schema 校验成功后用解析结果回填，保证 content 本身是合法 JSON
        # （原始 result["text"] 可能带 ``` 围栏；校验器已剥壳，此处同步剥壳）。
        content = json.dumps(result["json"], ensure_ascii=False)
    message: dict = {"role": "assistant", "content": content}
    reasoning = result.get("reasoning") or getattr(pipe, "reasoning_text", "")
    if reasoning and svc.reasoning_field != "none":
        message[svc.reasoning_field] = reasoning
    if result.get("tool_calls"):
        # OpenAI 语义：带 tool_calls 时 content 是模型**调用前**输出的文本。
        # `result["text"]` 正是这部分——Pipeline._route_content 只把
        # kind=="content" 的片段 yield 成文本，tool_call 块走 dict，
        # _run_blocking 的 parts 就是这些文本的拼接。
        # 此前这里用 `result.get("content_before_tools", "")` 覆盖，而该键
        # 从未被产出，等于把前置文本抹成空串（已修）。
        message["tool_calls"] = result["tool_calls"]
    return {
        "id": "chatcmpl-" + str(id(result))[-8:],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": svc.model_name,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": pipe.finish_reason(result)}],
        "usage": usage,
    }


@router.post("/completions")
async def completions(body: CompletionRequest, request: Request):
    """legacy completions：prompt 直喂 runtime（设计 §4.1 native）。

    流式与非流式同 chat 语义；流式用 legacy text + cmpl- id（P1-O1）。
    """
    svc = _svc(request)
    raw = body.model_dump()
    streaming = bool(raw.get("stream"))
    # completions 同样走 caps 参数策略（审计 P3-9）
    reject, warn = caps.check_request_params(raw, policy=svc.param_policy,
                                            stream=streaming)
    if reject:
        return _http_error(400, "invalid_request_error",
                           caps.reject_message(reject, raw),
                           code="unsupported_parameter",
                           param=reject[0])
    headers = {"X-RKLLM-Unsupported": ",".join(warn)} if warn else {}
    max_tokens = raw.get("max_tokens") or svc.default_max_tokens

    if streaming:
        try:
            pipe = _pipeline(svc, {**raw, "max_tokens": max_tokens}, chat=False,
                           endpoint="completions")
        except ValueError as exc:
            return _http_error(400, "invalid_request_error", str(exc))
        if pipe.max_tokens_clamped:
            headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
        return StreamingResponse(
            _stream_completions(request, svc, pipe, raw),
            media_type="text/event-stream",
            headers={**sse_response_headers(), **headers})

    try:
        pipe = _pipeline(svc, {**raw, "max_tokens": max_tokens}, chat=False,
                           endpoint="completions")
    except ValueError as exc:
        return _http_error(400, "invalid_request_error", str(exc))
    try:
        # P2.2：同上，非流式断连取消。
        result = await _run_blocking_disconnect(
            request,
            lambda **kw: _run_blocking(svc, pipe, **kw),
            supervisor=svc.supervisor)
        if result is _DISCONNECTED:
            return _client_disconnected()
    except QueueBusy as exc:
        return _http_error(503, "server_error", BUSY_MESSAGE,
                           code="queue_full",
                           headers={"Retry-After": str(exc.retry_after_s)})
    except QueueTimeout:
        return _http_error(503, "server_error", "queue timeout",
                           code="queue_timeout",
                           headers={"Retry-After":
                                    str(svc.queue.busy_retry_after_s)})
    except WorkerDown as exc:
        return _http_error(503, "server_error", str(exc)[:200],
                           code="worker_down",
                           headers={"Retry-After":
                                    str(svc.queue.busy_retry_after_s)})
    if pipe.max_tokens_clamped:
        headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
    return JSONResponse({
        "id": "cmpl-" + str(id(result))[-8:],
        "object": "text_completion",
        "created": int(time.time()),
        "model": svc.model_name,
        "choices": [{"index": 0, "text": result["text"],
                     "finish_reason": pipe.finish_reason(result)}],
        "usage": pipe.usage(result),
    }, headers=headers)


def _stream_completions(request: Request, svc, pipe: Pipeline, raw: dict):
    """text_completion 信封的 SSE 生成器（无 role、无 trailing_frames）。

    `legacy=True`：帧形状是 OpenAI legacy completions——`choices[0].text`
    而非 `delta.text`，id 前缀 `cmpl-`（与非流式 completions 一致）。
    """
    session = SseSession(model=svc.model_name,
                         object_name="text_completion",
                         delta_field="text", legacy=True)
    return stream_events(request, svc, pipe, session, raw,
                         first_role=None, trailing=False)
