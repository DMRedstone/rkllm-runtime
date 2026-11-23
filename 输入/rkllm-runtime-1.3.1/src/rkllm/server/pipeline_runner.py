"""推理管线装配与阻塞执行（共享层，审查 A-2）。

为什么单独一层：改造前 `responses.py` 从 `openai.py` 导入 7 个**私有**符号
（`_pipeline` / `_run_blocking` / `_run_blocking_checked` / `_SchemaError` /
`_json_payload` / `_http_error` / `_svc`），`app.py` 又反向导入
`routes.ops._cache_stats`——路由之间、app 与路由之间互相穿透私有实现，
任何一个函数改名都会连带改三处，且无法单独测试。

这里把「怎么建管线、怎么跑阻塞推理、怎么发错误响应」收敛为一条明面 API；
两个路由与 app 只依赖本模块的公开名字，不再互相 import 私有实现。

P2.2（2026-11-23）：非流式路径补客户端断连取消——`run_blocking*` 接受可选的
`cancelled/on_acquire/on_release` 并透传给 `InferenceQueue.run`；路由用
`run_blocking_disconnect()` 在事件循环里并发监听 `http.disconnect`，断开时
两段式取消（先置位、持槽时才 abort worker），不再让请求算到 900s。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from ..agent.json_schema import validate as validate_schema
from ..agent import tokens as token_counter
from ..agent.pipeline import Pipeline
from .streaming import CancelToken

log = logging.getLogger("rkllm.server.pipeline")


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


def run_blocking(svc, pipe: Pipeline, *, cancelled=None,
                 on_acquire=None, on_release=None) -> dict:
    """排队 + 收齐整段（非流式路径；队列在 worker 线程里同步执行）。

    `cancelled` / `on_acquire` / `on_release` 是 P2.2 的断连取消透传：
    - `cancelled()`：每 token 之间检查，客户端断开后尽快停止消费；
    - 两个槽位回调供调用方实现「只在持槽时 abort」的两段式取消
      （见 `run_blocking_disconnect` / `streaming.CancelToken`）。
    默认全为 None，与旧行为（不可取消）完全一致。
    """
    parts: list[str] = []
    meta: dict = {}
    started = time.monotonic()
    slot_at: float | None = None
    first_token_at: float | None = None

    def _acquire() -> None:
        nonlocal slot_at
        slot_at = time.monotonic()
        if on_acquire is not None:
            on_acquire()

    for item in svc.queue.run(lambda: pipe.iter_tokens(),
                              cancelled=cancelled, on_acquire=_acquire,
                              on_release=on_release, endpoint=pipe.endpoint):
        if isinstance(item, dict):
            meta.update(item)          # perf / tool_calls / reasoning 等管线产物
        else:
            if first_token_at is None:
                first_token_at = time.monotonic()
            parts.append(item)
    finished = time.monotonic()
    # P2.2 / 指标口径：TTFT 与 duration 从**拿到执行槽**起算，排队等待由
    # `queue.run` 单独计入 queue_wait_seconds_sum / queue_wait_histogram；
    # 此前 duration 从入队前开始算，把排队时间混进了推理耗时。
    infer_start = slot_at if slot_at is not None else started
    duration = finished - infer_start
    # 计数一律走 queue.bump/gauge（原子），不要直接改 metrics dict
    svc.queue.bump("duration_sum", duration)
    svc.queue.gauge("last_ttft_s",
                    (first_token_at - infer_start) if first_token_at else 0.0)
    perf = meta.get("perf") or {}
    _calibrate_tokens(pipe, perf)
    gen_tokens = int(perf.get("generate_tokens") or 0)
    ttft = (first_token_at - infer_start) if first_token_at else None
    decode = (duration - ttft) if ttft is not None else None
    record_timing(svc.queue, pipe.endpoint, ttft, decode, gen_tokens)
    # token 计数（/health stats 兼容字段的来源；perf 的 exact counts 优先）
    svc.queue.bump("prompt_tokens_total", int(perf.get("prefill_tokens") or 0))
    svc.queue.bump("completion_tokens_total",
                   int(perf.get("generate_tokens") or 0))
    return {"text": "".join(parts), **meta}


def run_blocking_checked(svc, raw: dict, max_tokens: int,
                        schema: dict | None,
                        endpoint: str = "chat.completions",
                        *, cancelled=None, on_acquire=None, on_release=None,
                        ) -> tuple[dict, Pipeline]:
    """非流式 + 可选 json_schema 校验重试（emulated，F6）。

    取消参数同 `run_blocking`：透传到每次尝试的阻塞执行，客户端断开后
    重试链也会立刻停止（而不是把剩余 2 次重试也算完）。
    """
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
        result = run_blocking(svc, pipe, cancelled=cancelled,
                              on_acquire=on_acquire, on_release=on_release)
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

    P2.2 / 指标口径：`slot_acquired()` 报告「拿到执行槽」的时刻后，TTFT 与
    duration 改从该时刻起算（排队等待交给 `queue.run` 的 queue_wait_* 单独
    统计）；未调用时保持旧口径（从生成器入口起算），向后兼容。
    """

    def __init__(self, started: float | None = None) -> None:
        self.started = started if started is not None else time.monotonic()
        self.slot_at: float | None = None
        self.first_token_at: float | None = None

    def slot_acquired(self, when: float | None = None) -> None:
        """报告执行槽获取时刻：之后的 TTFT/duration 以它为原点。"""
        self.slot_at = when if when is not None else time.monotonic()

    @property
    def _origin(self) -> float:
        return self.slot_at if self.slot_at is not None else self.started

    def first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()

    def finish(self, svc, pipe) -> None:
        now = time.monotonic()
        origin = self._origin
        svc.queue.bump("duration_sum", now - origin)
        # `Pipeline.last_perf()` 恒返回 dict | None（agent/pipeline.py）；此前为
        # 早期 fake 路径保留的"元组"分支是死代码，已删（2026-11-20 审计）。
        perf = pipe.last_perf() if hasattr(pipe, "last_perf") else None
        if not isinstance(perf, dict):
            perf = None
        else:
            _calibrate_tokens(pipe, perf)
        gen_tokens = int(perf.get("generate_tokens") or 0) if perf else 0
        ttft = (self.first_token_at - origin
                if self.first_token_at is not None else None)
        decode = None
        if ttft is not None:
            svc.queue.gauge("last_ttft_s", ttft)
            decode = now - self.first_token_at
        record_timing(svc.queue, getattr(pipe, "endpoint", "unknown"),
                      ttft, decode, gen_tokens)
        if perf:
            svc.queue.bump("prompt_tokens_total",
                           int(perf.get("prefill_tokens") or 0))
            svc.queue.bump("completion_tokens_total",
                           int(perf.get("generate_tokens") or 0))


def _discard_exception(future) -> None:
    try:
        future.result()
    except BaseException:
        pass


# ---------------------------------------------------------------------------
# P2.2：非流式（200 之前）客户端断连取消
# ---------------------------------------------------------------------------

DISCONNECTED = object()             # run_blocking_disconnect 的"已断连"哨兵
CLIENT_DISCONNECTED_STATUS = 499    # 客户端断开；响应仅用于闭合 ASGI 语义
_WATCH_DISCONNECT_SLEEP_S = 0.05    # 断连轮询退避（同 streaming 的 watch）
_DISCONNECT_SETTLE_S = 10.0         # 断开后等阻塞路径收尾的上限（秒）


def client_disconnected_response() -> JSONResponse:
    """客户端在 200 前断开：返回 499（客户端已走，只用于收尾）。"""
    return http_error(CLIENT_DISCONNECTED_STATUS, "server_error",
                      "client disconnected before response",
                      code="client_disconnected")


def _on_slot_releasing_after_cancel(token: CancelToken):
    """queue 归还执行槽前的回调：仍持槽且已取消时补一次 abort。

    这是两段式取消的关键窗口（同 `streaming.TokenStream._on_slot_releasing`）：
    取消置位与队列循环检查之间，若直接释放槽位，worker 会继续把旧请求算完。
    """
    def _release() -> None:
        if token.cancelled:
            token.abort_once()
        token.release_slot()
    return _release


async def _watch_disconnect(request: Request) -> bool:
    """长轮询 `request.receive()`，等到 `http.disconnect` 返回 True。

    与 streaming 的断连监视同一套路：不要用 `request.is_disconnected()`
    （它立即取消等待、只读已入队消息，流式/阻塞期都极易漏事件）。

    返回值区分「真的断开」与「监视不可用」：某些 ASGI 适配层/测试桩的
    `receive()` 会在请求处理中途抛错或提前结束，此时返回 False——绝不能
    当成客户端断连（否则会把正常请求误取消）。
    """
    try:
        while True:
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                return True
            await asyncio.sleep(_WATCH_DISCONNECT_SLEEP_S)
    except Exception as exc:        # noqa: BLE001 - 连接层异常不再监视
        log.warning("非流式断连监视退出（按未断连处理）：%r", exc)
        return False


async def run_blocking_disconnect(request: Request, work, *, supervisor=None,
                                  settle_timeout_s: float = _DISCONNECT_SETTLE_S):
    """在工作线程跑阻塞 `work`，同时监听客户端断连；断开即取消并归还 NPU。

    `work` 必须接受三个关键字参数：`cancelled`（传给 queue.run 的轮询回调）、
    `on_acquire` / `on_release`（两段式取消的槽位归属回调）。返回：

    - `work` 的返回值（正常完成）；
    - 或原样抛出 `work` 的异常（HTTP 错误语义与旧实现一致）；
    - `DISCONNECTED` 哨兵：客户端在响应前断开。此时 `work` 的异常不再有意义
      （客户端已走），但会等它收尾（有界）确保执行槽归还。

    为什么用 `CancelToken` 而不是直接 `supervisor.abort()`：阻塞请求可能还
    在排队（未持槽），无条件 abort 会打断无关的在飞请求（F-2 回归）。
    """
    loop = asyncio.get_running_loop()
    token = CancelToken(supervisor)

    def _work():
        return work(cancelled=lambda: token.cancelled,
                    on_acquire=token.hold_slot,
                    on_release=_on_slot_releasing_after_cancel(token))

    task = loop.run_in_executor(None, _work)
    watcher = asyncio.create_task(_watch_disconnect(request))
    try:
        done, _pending = await asyncio.wait(
            {task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            # 正常完成/报错：取消监视并原样返回（等待响应期间不再关心断连）
            watcher.cancel()
            return task.result()
        if not watcher.result():
            # 监视不可用（receive 提前结束/报错）：退回旧行为，只等阻塞执行，
            # 绝不误判为客户端断开。
            await asyncio.wait({task})
            return task.result()
        # 客户端断开：第一段置位 + 第二段归还 NPU（阻塞写，放线程池，
        # 不卡事件循环）；未持槽时 abort_once 返回 False，不会误伤他人。
        token.cancel()
        try:
            loop.run_in_executor(None, token.abort_once)
        except RuntimeError:        # 事件循环收尾：同步兜底
            token.abort_once()
        try:
            await asyncio.wait({task}, timeout=settle_timeout_s)
        except asyncio.CancelledError:
            raise
        if task.done():
            _discard_exception(task)      # 取消导致的异常不回传 HTTP 层
        else:
            # 卡在 C 调用里也要能结束请求：后台收尾，别让客户端(已断)占住
            # 事件循环；槽位由 cancelled/abort 路径负责归还。
            task.add_done_callback(_discard_exception)
        return DISCONNECTED
    finally:
        if not watcher.done():
            watcher.cancel()

