"""OpenAI 端点：chat/completions、completions、models（F9）。

流式：SseSession 固定 id（D4）+ 心跳（D9）+ 错误帧（D8）。
S3 的 agent 能力（tools 解析 / reasoning / json_schema）在推理管线里以
可插拔管线接入：本文件只负责 OpenAI 信封与流式帧，不解析模型输出。
"""

from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent.pipeline import Pipeline
from ..schemas import ChatCompletionRequest, CompletionRequest
from ..sse import SseSession, sse_response_headers
from ..queue import QueueBusy, QueueTimeout
from ..supervisor import WorkerDown
from ... import caps

router = APIRouter(prefix="/v1")


def _svc(request: Request):
    return request.app.state.svc


def _http_error(status: int, err_type: str, message: str, **extra) -> JSONResponse:
    body = {"error": {"type": err_type, "message": message}}
    body["error"].update(extra)
    return JSONResponse(body, status_code=status)


@router.get("/models")
def list_models(request: Request):
    svc = _svc(request)
    obj = {"id": svc.model_name, "object": "model",
           "owned_by": "rkllm-runtime", "context_len": svc.context_len}
    return {"object": "list", "data": [obj]}


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request):
    """单模型：任意 id 返回同一对象（设计 §4.1）。"""
    svc = _svc(request)
    return {"id": svc.model_name, "object": "model",
            "owned_by": "rkllm-runtime", "context_len": svc.context_len}


def _pipeline(svc, body: dict, *, chat: bool) -> Pipeline:
    """从请求构造推理管线（agent 能力在此按需挂载）。"""
    return Pipeline(svc, body, chat=chat)


@router.post("/chat/completions")
def chat_completions(body: ChatCompletionRequest, request: Request):
    svc = _svc(request)
    raw = body.model_dump()
    # caps 参数策略：unsupported 按 reject/warn，ignored 剥离（D2）
    reject, warn = caps.check_request_params(raw, policy=svc.param_policy)
    if reject:
        return _http_error(400, "invalid_request_error",
                           "unsupported parameter(s): %s" % ", ".join(reject),
                           code="unsupported_parameter")
    messages = raw["messages"]
    headers = {}
    if warn:
        headers["X-RKLLM-Unsupported"] = ",".join(warn)
    max_tokens = raw.get("max_tokens") or raw.get("max_completion_tokens") \
        or svc.default_max_tokens

    try:
        pipe = _pipeline(svc, {**raw, "max_tokens": max_tokens}, chat=True)
    except ValueError as exc:
        return _http_error(400, "invalid_request_error", str(exc))

    if raw.get("stream"):
        return StreamingResponse(
            _stream_chat(svc, pipe, warn),
            media_type="text/event-stream",
            headers={**sse_response_headers(), **headers})
    try:
        result = _run_blocking(svc, pipe)
    except QueueBusy as exc:
        return _http_error(503, "server_error", "model busy",
                           code="queue_full",
                           **{"Retry-After": str(exc.retry_after_s)})
    except QueueTimeout:
        return _http_error(503, "server_error", "queue timeout",
                           code="queue_timeout")
    except WorkerDown as exc:
        return _http_error(503, "server_error", str(exc)[:300],
                           code="worker_down",
                           **{"Retry-After": str(svc.queue.busy_retry_after_s)})
    return JSONResponse(_completion_payload(svc, pipe, result, max_tokens),
                        headers=headers)


def _run_blocking(svc, pipe: Pipeline) -> dict:
    """排队 + 收齐整段（非流式路径；队列在 worker 线程里同步执行）。"""
    parts: list[str] = []
    meta: dict = {}
    for item in svc.queue.run(lambda: pipe.iter_tokens()):
        if isinstance(item, dict):
            meta.update(item)          # perf / tool_calls / reasoning 等管线产物
        else:
            parts.append(item)
    return {"text": "".join(parts), **meta}


def _stream_chat(svc, pipe: Pipeline, warn: list[str]):
    """SSE 生成器：首帧 role → 增量 → finish_reason → usage → [DONE]。"""
    session = SseSession(model=svc.model_name)
    yield session.first_frame()
    try:
        for item in svc.queue.run(lambda: pipe.iter_tokens()):
            if isinstance(item, dict):
                continue
            yield session.delta_frame(content=item)
        finish = pipe.finish_reason()
        for frame in pipe.trailing_frames(session):
            yield frame
        yield session.finish_frame(finish)
        yield session.done()
    except QueueBusy:
        yield session.error_frame("model busy; retry later")
        yield session.done()
    except (WorkerDown, QueueTimeout) as exc:
        yield session.error_frame(str(exc)[:300])
        yield session.done()


def _completion_payload(svc, pipe: Pipeline, result: dict,
                        max_tokens: int) -> dict:
    usage = pipe.usage(result)
    message: dict = {"role": "assistant", "content": result["text"]}
    if result.get("reasoning"):
        message["reasoning_content"] = result["reasoning"]
    if result.get("tool_calls"):
        message["tool_calls"] = result["tool_calls"]
        message["content"] = result.get("content_before_tools", "")
    return {
        "id": "chatcmpl-" + str(id(result))[-8:],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": svc.model_name,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": pipe.finish_reason()}],
        "usage": usage,
    }


@router.post("/completions")
def completions(body: CompletionRequest, request: Request):
    """薄封装：prompt 直喂 runtime（设计 §4.1 native）。"""
    svc = _svc(request)
    pipe = _pipeline(svc, body.model_dump(), chat=False)
    try:
        result = _run_blocking(svc, pipe)
    except QueueBusy as exc:
        return _http_error(503, "server_error", "model busy",
                           code="queue_full",
                           **{"Retry-After": str(exc.retry_after_s)})
    except (QueueTimeout, WorkerDown) as exc:
        return _http_error(503, "server_error", str(exc)[:200])
    return {
        "id": "cmpl-" + str(id(result))[-8:],
        "object": "text_completion",
        "created": int(time.time()),
        "model": svc.model_name,
        "choices": [{"index": 0, "text": result["text"],
                     "finish_reason": pipe.finish_reason()}],
        "usage": pipe.usage(result),
    }
