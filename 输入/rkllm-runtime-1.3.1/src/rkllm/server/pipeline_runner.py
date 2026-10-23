"""推理管线装配与阻塞执行（共享层，审查 A-2）。

为什么单独一层：改造前 `responses.py` 从 `openai.py` 导入 7 个**私有**符号
（`_pipeline` / `_run_blocking` / `_run_blocking_checked` / `_SchemaError` /
`_json_payload` / `_http_error` / `_svc`），`app.py` 又反向导入
`routes.ops._cache_stats`——路由之间、app 与路由之间互相穿透私有实现，
任何一个函数改名都会连带改三处，且无法单独测试。

这里把「怎么建管线、怎么跑阻塞推理、怎么发错误响应」收敛为一条明面 API；
两个路由与 app 只依赖本模块的公开名字，不再互相 import 私有实现。
"""

from __future__ import annotations

import json
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from ..agent.json_schema import validate as validate_schema
from ..agent.pipeline import Pipeline


class SchemaError(Exception):
    """json_schema 重试耗尽（供路由转 422）。"""


def service_of(request: Request):
    """`app.state.svc` 的唯一取用点（避免各路由各写一遍）。"""
    return request.app.state.svc


def http_error(status: int, err_type: str, message: str,
               **extra) -> JSONResponse:
    body = {"error": {"type": err_type, "message": message}}
    body["error"].update(extra)
    return JSONResponse(body, status_code=status)


def build_pipeline(svc, body: dict, *, chat: bool) -> Pipeline:
    """从请求构造推理管线（agent 能力在此按需挂载）。"""
    return Pipeline(svc, body, chat=chat)


def json_payload(text: str):
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


def run_blocking(svc, pipe: Pipeline) -> dict:
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


def run_blocking_checked(svc, raw: dict, max_tokens: int,
                        schema: dict | None) -> tuple[dict, Pipeline]:
    """非流式 + 可选 json_schema 校验重试（emulated，F6）。"""
    attempts = max(0, int(svc.json_schema_retries)) + 1
    messages = list(raw.get("messages") or [])
    last_errors: list[str] = []
    body = {**raw, "max_tokens": max_tokens}
    pipe = build_pipeline(svc, body, chat=True)
    result: dict = {}
    for attempt in range(attempts):
        if attempt:
            body = {**body, "messages": messages}
            pipe = build_pipeline(svc, body, chat=True)
        result = run_blocking(svc, pipe)
        if schema is None:
            return result, pipe
        try:
            parsed = json_payload(result["text"])
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
    raise SchemaError("；".join(last_errors[:5]) or "unknown")
