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
import queue as queue_mod
import threading
import time
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent.json_schema import validate as validate_schema
from ...agent.pipeline import Pipeline
from ..schemas import ChatCompletionRequest, CompletionRequest
from ..sse import SseSession, sse_response_headers
from ..queue import QueueBusy, QueueTimeout
from ..supervisor import WorkerDown
from ... import caps

router = APIRouter(prefix="/v1")

log = logging.getLogger("rkllm.server.openai")

_STREAM_DONE = object()


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


def _pipeline(svc, body: dict, *, chat: bool) -> Pipeline:
    """从请求构造推理管线（agent 能力在此按需挂载）。"""
    return Pipeline(svc, body, chat=chat)


def _json_payload(text: str):
    """宽容提取 JSON（容忍 ``` 代码块与前后缀）；失败抛 ValueError。"""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
    start = candidate.find("{")
    if start < 0:
        start = candidate.find("[")
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    if start >= 0 and end > start:
        candidate = candidate[start:end + 1]
    return json.loads(candidate)


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
        return StreamingResponse(
            _stream_chat(request, svc, pipe, raw),
            media_type="text/event-stream",
            headers={**sse_response_headers(), **headers})

    try:
        result, pipe = _run_blocking_checked(svc, raw, max_tokens, schema)
    except _SchemaError as exc:
        return _http_error(422, "invalid_request_error",
                           "response did not match response_format.json_schema "
                           "after retries: %s" % exc, code="schema_mismatch")
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


class _SchemaError(Exception):
    pass


def _run_blocking(svc, pipe: Pipeline) -> dict:
    """排队 + 收齐整段（非流式路径；队列在 worker 线程里同步执行）。"""
    parts: list[str] = []
    meta: dict = {}
    started = time.monotonic()
    first_token_at: float | None = None
    for item in svc.queue.run(lambda: pipe.iter_tokens()):
        if isinstance(item, dict):
            meta.update(item)          # perf / tool_calls / reasoning 等管线产物
        else:
            if first_token_at is None:
                first_token_at = time.monotonic()
            parts.append(item)
    elapsed = time.monotonic() - started
    metrics = svc.queue.metrics
    metrics["duration_sum"] = metrics.get("duration_sum", 0.0) + elapsed
    metrics["last_ttft_s"] = ((first_token_at - started)
                              if first_token_at else 0.0)
    return {"text": "".join(parts), **meta}


def _run_blocking_checked(svc, raw: dict, max_tokens: int,
                          schema: dict | None) -> tuple[dict, Pipeline]:
    """非流式 + 可选 json_schema 校验重试（emulated，F6）。"""
    attempts = max(0, int(svc.json_schema_retries)) + 1
    messages = list(raw.get("messages") or [])
    last_errors: list[str] = []
    body = {**raw, "max_tokens": max_tokens}
    pipe = _pipeline(svc, body, chat=True)
    result: dict = {}
    for attempt in range(attempts):
        if attempt:
            body = {**body, "messages": messages}
            pipe = _pipeline(svc, body, chat=True)
        result = _run_blocking(svc, pipe)
        if schema is None:
            return result, pipe
        try:
            parsed = _json_payload(result["text"])
        except (ValueError, json.JSONDecodeError) as exc:
            last_errors = ["输出不是合法 JSON：%s" % exc]
        else:
            last_errors = validate_schema(parsed, schema)
            if not last_errors:
                result["json"] = parsed
                return result, pipe
        messages = messages + [
            {"role": "assistant", "content": result["text"]},
            {"role": "user", "content":
             "上一次输出不符合 JSON Schema：%s\n请只输出符合该 Schema 的 JSON，"
             "不要包含解释或代码块标记。" % "；".join(last_errors[:5])},
        ]
    raise _SchemaError("；".join(last_errors[:5]) or "unknown")


def _stream_chat(request: Request, svc, pipe: Pipeline, raw: dict):
    """SSE 生成器（异步）：首帧 role → 增量（含心跳）→ finish_reason → usage → [DONE]。

    为什么必须是 async 生成器：客户端断开时同步生成器**收不到**任何信号——
    uvicorn 不会因为写失败而中断它（实测：800 token 的长任务在客户端断开后
    仍跑满 99 s，NPU 白转）。`await request.is_disconnected()` 是 starlette
    提供的标准探测方式，轮询 0.2 s 一次，断开后立即 abort。
    """
    import asyncio

    session = SseSession(model=svc.model_name,
                         reasoning_field=svc.reasoning_field)

    items: queue_mod.Queue = queue_mod.Queue(maxsize=512)

    def pump() -> None:
        truncated = False
        try:
            for item in svc.queue.run(lambda: pipe.iter_tokens()):
                try:
                    items.put(item, timeout=5)
                except queue_mod.Full:
                    truncated = True
                    break                 # 消费者已消失，别再攒
        except BaseException as exc:      # 交给生成器决定如何呈现
            try:
                items.put(exc, timeout=1)
            except queue_mod.Full:
                pass
        finally:
            if truncated:
                # P2：显式告知生成器发生了截断，收尾改用 finish_reason=length，
                # 而不是让客户端把被截断的响应当成完整 stop。
                try:
                    items.put({"truncated": True}, timeout=1)
                except queue_mod.Full:
                    pass
            try:
                items.put(_STREAM_DONE, timeout=1)
            except queue_mod.Full:
                pass

    heartbeat_s = max(1, int(svc.stream_heartbeat_s))
    threading.Thread(target=pump, daemon=True, name="sse-pump").start()

    def _abort() -> None:
        if svc.supervisor is not None:
            try:
                svc.supervisor.abort()
            except Exception:
                pass

    async def generate():
        yield session.first_frame()
        finished = False
        truncated = False
        last_emit = time.monotonic()
        stop_watch = asyncio.Event()

        async def watch_disconnect() -> None:
            """直接长轮询 receive()：uvicorn 在 connection_lost 时推 http.disconnect。

            不用 request.is_disconnected()：它内部用 CancelScope 立即取消等待，
            只读取"已经入队"的消息，流式期间极易漏掉（实测断连后永不触发）。
            """
            try:
                while not stop_watch.is_set():
                    message = await request.receive()
                    if message.get("type") == "http.disconnect":
                        log.info("client disconnected -> abort worker")
                        _abort()
                        return
            except Exception:
                return

        watcher = asyncio.create_task(watch_disconnect())
        try:
            while True:
                try:
                    item = items.get_nowait()
                except queue_mod.Empty:
                    if time.monotonic() - last_emit >= heartbeat_s:
                        yield session.heartbeat()   # D9：无 token 期间续命
                        last_emit = time.monotonic()
                    await asyncio.sleep(0.02)   # 0.2s 会给首帧叠加固定延迟
                    continue
                if stop_watch.is_set():
                    return
                if item is _STREAM_DONE:
                    break
                if isinstance(item, BaseException):
                    if isinstance(item, (QueueBusy, QueueTimeout, WorkerDown)):
                        yield session.error_frame(str(item)[:300])
                    else:
                        yield session.error_frame("internal error")
                    yield session.done()
                    finished = True
                    return
                if isinstance(item, dict):
                    if item.get("truncated"):
                        truncated = True
                    continue                    # 管线元数据（tool_calls 等）
                yield session.delta_frame(content=item)
                last_emit = time.monotonic()

            for frame in pipe.trailing_frames(session):
                yield frame
            yield session.finish_frame("length" if truncated else pipe.finish_reason())
            want_usage = svc.include_usage_default or bool(
                (raw.get("stream_options") or {}).get("include_usage"))
            if want_usage:
                yield session.usage_frame(pipe.usage())
            yield session.done()
            finished = True
        finally:
            stop_watch.set()
            watcher.cancel()
            if not finished:
                _abort()

    return generate()


def _completion_payload(svc, pipe: Pipeline, result: dict,
                        max_tokens: int) -> dict:
    usage = pipe.usage(result)
    message: dict = {"role": "assistant", "content": result["text"]}
    if result.get("reasoning") and svc.reasoning_field != "none":
        message[svc.reasoning_field] = result["reasoning"]
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
    raw = body.model_dump()
    # completions 同样走 caps 参数策略（审计 P3-9）
    reject, warn = caps.check_request_params(raw, policy=svc.param_policy,
                                            stream=bool(raw.get("stream")))
    if reject:
        return _http_error(400, "invalid_request_error",
                           "unsupported parameter(s): %s" % ", ".join(reject),
                           code="unsupported_parameter")
    headers = {"X-RKLLM-Unsupported": ",".join(warn)} if warn else {}
    pipe = _pipeline(svc, raw, chat=False)
    try:
        result = _run_blocking(svc, pipe)
    except QueueBusy as exc:
        return _http_error(503, "server_error", "model busy",
                           code="queue_full",
                           **{"Retry-After": str(exc.retry_after_s)})
    except (QueueTimeout, WorkerDown) as exc:
        return _http_error(503, "server_error", str(exc)[:200])
    return JSONResponse({
        "id": "cmpl-" + str(id(result))[-8:],
        "object": "text_completion",
        "created": int(time.time()),
        "model": svc.model_name,
        "choices": [{"index": 0, "text": result["text"],
                     "finish_reason": pipe.finish_reason()}],
        "usage": pipe.usage(result),
    }, headers=headers)
