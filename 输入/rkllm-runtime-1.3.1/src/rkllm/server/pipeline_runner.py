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
from ..agent import tokens as token_counter
from ..agent.pipeline import Pipeline


# 单 NPU 上下文串行（硬件现实，非缺陷）：忙时明确告知客户端，别让它以为
# 并行重试就能过。两个路由共用，放共享层避免各写一份文案而漂移。
BUSY_MESSAGE = "model busy (single NPU context: requests are serialized)"


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


def build_pipeline(svc, body: dict, *, chat: bool,
                   endpoint: str = "chat.completions") -> Pipeline:
    """从请求构造推理管线（agent 能力在此按需挂载）。"""
    return Pipeline(svc, body, chat=chat, endpoint=endpoint)


def record_timing(queue, endpoint: str, ttft_s: float | None,
                  decode_s: float | None, gen_tokens: int) -> None:
    """请求耗时分解：TTFT（含 prefill）、解码时长、生成速率。

    三者都带 `endpoint` 标签——`/v1/responses` 是翻译层，混进同一个桶就看不出
    它的额外代价。参数为 None 表示该阶段没发生（例如 0 token 就结束）。
    """
    if ttft_s is not None:
        queue.ttft_hist.labels(endpoint=endpoint).observe(ttft_s)
    if decode_s is not None and decode_s > 0:
        queue.decode_hist.labels(endpoint=endpoint).observe(decode_s)
        if gen_tokens > 0:
            queue.tok_s_hist.labels(endpoint=endpoint).observe(
                gen_tokens / decode_s)


def json_payload(text: str):
    """宽容提取 JSON（容忍 ``` 代码块、前后缀、多个候选）；失败抛 ValueError。

    旧实现取"第一个 { 到最后一个 }"：模型输出两个对象、或结尾带一句带括号的说明
    （"（注意 {} 不要留空）"）时整段就废了。改用 stdlib `JSONDecoder.raw_decode`
    在每个 `{`/`[` 起点尝试解析——**由标准库判定边界**，不做手写括号配平。

    只尝试前 8 个起点：响应体里的 JSON 一定靠前，全量扫描反而会被长文本拖慢。
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(candidate) if ch in "{["][:8]
    for start in starts:
        try:
            value, _end = decoder.raw_decode(candidate[start:])
            return value
        except ValueError:
            continue
    raise ValueError("no JSON object found in model output")


def _calibrate_tokens(pipe, perf: dict) -> None:
    """用 runtime 的 prefill_tokens（真值）校准 bytes/token 估算。

    只在估算档生效（装了精确 tokenizer 时 `calibrate()` 自己会跳过）。样本异常
    由 `tokens.calibrate()` 丢弃，这里不做二次判断。
    """
    try:
        actual = int((perf or {}).get("prefill_tokens") or 0)
        prompt_bytes = int(getattr(pipe, "prompt_bytes", 0) or 0)
        if actual > 0 and prompt_bytes > 0:
            token_counter.calibrate(prompt_bytes, actual)
    except Exception:                       # noqa: BLE001 - 校准失败不影响主流程
        pass


def run_blocking(svc, pipe: Pipeline) -> dict:
    """排队 + 收齐整段（非流式路径；队列在 worker 线程里同步执行）。"""
    parts: list[str] = []
    meta: dict = {}
    started = time.monotonic()
    first_token_at: float | None = None
    for item in svc.queue.run(lambda: pipe.iter_tokens(),
                              endpoint=pipe.endpoint):
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
    perf = meta.get("perf") or {}
    _calibrate_tokens(pipe, perf)
    gen_tokens = int(perf.get("generate_tokens") or 0)
    ttft = (first_token_at - started) if first_token_at else None
    decode = (elapsed - ttft) if ttft is not None else None
    record_timing(svc.queue, pipe.endpoint, ttft, decode, gen_tokens)
    # token 计数（/health stats 兼容字段的来源；perf 的 exact counts 优先）
    metrics["prompt_tokens_total"] = (metrics.get("prompt_tokens_total", 0)
                                      + int(perf.get("prefill_tokens") or 0))
    metrics["completion_tokens_total"] = (
        metrics.get("completion_tokens_total", 0)
        + int(perf.get("generate_tokens") or 0))
    return {"text": "".join(parts), **meta}


def run_blocking_checked(svc, raw: dict, max_tokens: int,
                        schema: dict | None,
                        endpoint: str = "chat.completions"
                        ) -> tuple[dict, Pipeline]:
    """非流式 + 可选 json_schema 校验重试（emulated，F6）。"""
    attempts = max(0, int(svc.json_schema_retries)) + 1
    messages = list(raw.get("messages") or [])
    last_errors: list[str] = []
    body = {**raw, "max_tokens": max_tokens}
    pipe = build_pipeline(svc, body, chat=True, endpoint=endpoint)
    result: dict = {}
    for attempt in range(attempts):
        if attempt:
            body = {**body, "messages": messages}
            pipe = build_pipeline(svc, body, chat=True, endpoint=endpoint)
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
    raise SchemaError("；".join(last_errors[:5]) or "JSON Schema 校验未通过（无具体错误）")


class StreamTimer:
    """流式路径的计时（此前只有非流式更新 last_ttft_s / duration_sum）。

    指标名不区分路径，流式却是主要用法（客户端几乎都开 stream），不更新的话
    `/metrics` 里的 TTFT 与耗时长期停在 0，容量判断会失真。用法：进入生成器
    时 `t = StreamTimer()`，收到第一个 token 时 `t.first_token()`，收尾时
    `t.finish(svc, pipe)`。
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.first_token_at: float | None = None

    def first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()

    def finish(self, svc, pipe) -> None:
        metrics = svc.queue.metrics
        metrics["duration_sum"] = (metrics.get("duration_sum", 0.0)
                                   + (time.monotonic() - self.started))
        perf = pipe.last_perf() if hasattr(pipe, "last_perf") else None
        if isinstance(perf, dict):
            _calibrate_tokens(pipe, perf)
        elif perf:
            _calibrate_tokens(pipe, {"prefill_tokens": perf[0]})
        # last_perf 可能是 (prefill, generate) 元组，也可能是 dict（fake 路径）
        if isinstance(perf, dict):
            gen_tokens = int(perf.get("generate_tokens") or 0)
        elif perf:
            gen_tokens = int(perf[1])
        else:
            gen_tokens = 0
        ttft = (self.first_token_at - self.started
                if self.first_token_at is not None else None)
        decode = None
        if ttft is not None:
            metrics["last_ttft_s"] = ttft
            decode = time.monotonic() - self.started - ttft
        record_timing(svc.queue, getattr(pipe, "endpoint", "unknown"),
                      ttft, decode, gen_tokens)
        if perf:
            metrics["prompt_tokens_total"] = (
                metrics.get("prompt_tokens_total", 0)
                + int(perf.get("prefill_tokens") or 0))
            metrics["completion_tokens_total"] = (
                metrics.get("completion_tokens_total", 0)
                + int(perf.get("generate_tokens") or 0))
