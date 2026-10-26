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

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent.pipeline import Pipeline
from ..pipeline_runner import (BUSY_MESSAGE,
                               SchemaError as _SchemaError,
                               StreamTimer,
                               build_pipeline as _pipeline,
                               http_error as _http_error,
                               run_blocking as _run_blocking,
                               run_blocking_checked as _run_blocking_checked,
                               service_of as _svc)
from ..schemas import ChatCompletionRequest, CompletionRequest
from ..sse import SseSession, sse_response_headers
from ..queue import QueueBusy, QueueTimeout
from ..streaming import (DISCONNECT, DONE, HEARTBEAT, TokenStream,
                         iterate_stream)
from ..supervisor import WorkerDown
from ... import caps

router = APIRouter(prefix="/v1")

log = logging.getLogger("rkllm.server.openai")



@router.get("/models")
def list_models(request: Request):
    svc = _svc(request)
    obj = {"id": svc.model_name, "object": "model",
           "owned_by": "rkllm-runtime", "context_len": svc.context_len,
           "model_class": svc.model_class}
    return {"object": "list", "data": [obj]}


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request):
    """单模型：任意 id 返回同一对象（设计 §4.1）。"""
    svc = _svc(request)
    return {"id": svc.model_name, "object": "model",
            "owned_by": "rkllm-runtime", "context_len": svc.context_len,
            "model_class": svc.model_class}


@router.post("/chat/completions")
def chat_completions(body: ChatCompletionRequest, request: Request):
    svc = _svc(request)
    raw = body.model_dump()
    streaming = bool(raw.get("stream"))
    # caps 参数策略：unsupported 按 reject/warn，ignored 剥离（D2）
    reject, warn = caps.check_request_params(raw, policy=svc.param_policy,
                                            stream=streaming)
    if reject:
        return _http_error(400, "invalid_request_error",
                           "unsupported parameter(s): %s" % ", ".join(reject),
                           code="unsupported_parameter")
    max_tokens = raw.get("max_tokens") or raw.get("max_completion_tokens") \
        or svc.default_max_tokens
    headers = {"X-RKLLM-Unsupported": ",".join(warn)} if warn else {}

    rf = raw.get("response_format") or {}
    schema = None
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        spec = rf.get("json_schema") or {}
        schema = spec.get("schema") if isinstance(spec, dict) else None

    if streaming:
        # 效率：仅流式路径构造 Pipeline；非流式交给 _run_blocking_checked 自行
        # 构建，避免重复做一次全量 prompt 构建（原实现两条路径各建一次）。
        try:
            pipe = _pipeline(svc, {**raw, "max_tokens": max_tokens}, chat=True)
        except ValueError as exc:
            return _http_error(400, "invalid_request_error", str(exc))
        if pipe.max_tokens_clamped:
            headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
        return StreamingResponse(
            _stream_chat(request, svc, pipe, raw),
            media_type="text/event-stream",
            headers={**sse_response_headers(), **headers})

    try:
        result, pipe = _run_blocking_checked(svc, raw, max_tokens, schema)
    except ValueError as exc:
        return _http_error(400, "invalid_request_error", str(exc))
    except _SchemaError as exc:
        return _http_error(422, "invalid_request_error",
                           "response did not match response_format.json_schema "
                           "after retries: %s" % exc, code="schema_mismatch")
    except QueueBusy as exc:
        return _http_error(503, "server_error", BUSY_MESSAGE,
                           code="queue_full",
                           **{"Retry-After": str(exc.retry_after_s)})
    except QueueTimeout:
        return _http_error(503, "server_error", "queue timeout",
                           code="queue_timeout")
    except WorkerDown as exc:
        return _http_error(503, "server_error", str(exc)[:300],
                           code="worker_down",
                           **{"Retry-After": str(svc.queue.busy_retry_after_s)})
    if pipe.max_tokens_clamped:
        headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
    return JSONResponse(_completion_payload(svc, pipe, result,
                                            pipe.max_tokens or max_tokens),
                        headers=headers)


def _stream_chat(request: Request, svc, pipe: Pipeline, raw: dict):
    """SSE 生成器（异步）：首帧 role → 增量（含心跳）→ finish_reason → usage → [DONE]。

    为什么必须是 async 生成器：客户端断开时同步生成器**收不到**任何信号——
    uvicorn 不会因为写失败而中断它（实测：800 token 的长任务在客户端断开后
    仍跑满 99 s，NPU 白转）。

    F-1：泵线程、心跳计时、断连取消统一由 `server.streaming` 提供（此前与
    `responses.py` 各有一份，取消路径互相漂移，见 O-2）。
    """
    session = SseSession(model=svc.model_name,
                         reasoning_field=svc.reasoning_field)
    stream = TokenStream(svc, pipe, name="sse-pump")
    heartbeat_s = max(1, int(svc.stream_heartbeat_s))

    async def generate():
        timer = StreamTimer()
        yield session.first_frame()
        truncated = False
        async for kind, item in iterate_stream(request, stream, heartbeat_s):
            if kind == HEARTBEAT:
                yield session.heartbeat()       # D9：无 token 期间续命
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
                stream.mark_settled()           # 已发终帧：收尾不再 abort
                return
            if isinstance(item, dict):
                if item.get("truncated"):
                    truncated = True
                continue                        # 管线元数据（tool_calls 等）
            timer.first_token()
            yield session.delta_frame(content=item)

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


def _completion_payload(svc, pipe: Pipeline, result: dict,
                        max_tokens: int) -> dict:
    usage = pipe.usage(result)
    message: dict = {"role": "assistant", "content": result["text"]}
    if result.get("reasoning") and svc.reasoning_field != "none":
        message[svc.reasoning_field] = result["reasoning"]
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
def completions(body: CompletionRequest, request: Request):
    """legacy completions：prompt 直喂 runtime（设计 §4.1 native）。

    流式与非流式同 chat 端点语义；信封为 text_completion（delta.text）。
    """
    svc = _svc(request)
    raw = body.model_dump()
    streaming = bool(raw.get("stream"))
    # completions 同样走 caps 参数策略（审计 P3-9）
    reject, warn = caps.check_request_params(raw, policy=svc.param_policy,
                                            stream=streaming)
    if reject:
        return _http_error(400, "invalid_request_error",
                           "unsupported parameter(s): %s" % ", ".join(reject),
                           code="unsupported_parameter")
    headers = {"X-RKLLM-Unsupported": ",".join(warn)} if warn else {}
    max_tokens = raw.get("max_tokens") or svc.default_max_tokens

    if streaming:
        try:
            pipe = _pipeline(svc, {**raw, "max_tokens": max_tokens}, chat=False)
        except ValueError as exc:
            return _http_error(400, "invalid_request_error", str(exc))
        if pipe.max_tokens_clamped:
            headers["X-RKLLM-Max-Tokens-Clamped"] = str(pipe.max_tokens)
        return StreamingResponse(
            _stream_completions(request, svc, pipe, raw),
            media_type="text/event-stream",
            headers={**sse_response_headers(), **headers})

    try:
        pipe = _pipeline(svc, {**raw, "max_tokens": max_tokens}, chat=False)
    except ValueError as exc:
        return _http_error(400, "invalid_request_error", str(exc))
    try:
        result = _run_blocking(svc, pipe)
    except QueueBusy as exc:
        return _http_error(503, "server_error", BUSY_MESSAGE,
                           code="queue_full",
                           **{"Retry-After": str(exc.retry_after_s)})
    except (QueueTimeout, WorkerDown) as exc:
        return _http_error(503, "server_error", str(exc)[:200])
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
    """completions 的 SSE 生成器：text_completion 信封（delta.text，无 role）。

    复用与 chat 完全相同的两段式取消 / 心跳 / 错误帧机制（F-1），只换信封。
    """
    session = SseSession(model=svc.model_name,
                         object_name="text_completion",
                         delta_field="text")
    stream = TokenStream(svc, pipe, name="sse-pump")
    heartbeat_s = max(1, int(svc.stream_heartbeat_s))

    async def generate():
        timer = StreamTimer()
        yield session.first_frame(role=None)
        truncated = False
        async for kind, item in iterate_stream(request, stream, heartbeat_s):
            if kind == HEARTBEAT:
                yield session.heartbeat()
                continue
            if kind == DISCONNECT:
                timer.finish(svc, pipe)
                return
            if kind == DONE:
                break
            if isinstance(item, BaseException):
                if isinstance(item, (QueueBusy, QueueTimeout, WorkerDown)):
                    yield session.error_frame(str(item)[:300])
                else:
                    yield session.error_frame("internal error")
                yield session.done()
                stream.mark_settled()
                return
            if isinstance(item, dict):
                if item.get("truncated"):
                    truncated = True
                continue
            timer.first_token()
            yield session.delta_frame(content=item)

        yield session.finish_frame("length" if truncated
                                   else pipe.finish_reason())
        want_usage = svc.include_usage_default or bool(
            (raw.get("stream_options") or {}).get("include_usage"))
        if want_usage:
            yield session.usage_frame(pipe.usage())
        timer.finish(svc, pipe)
        yield session.done()

    return generate()
