"""长任务 job 状态机（MCP 设计 §5.3 / §5.4）。

- 判定：估算耗时 = prefill/prefill_tok_s + max_tokens/decode_tok_s，超过阈值 →
  建任务返回 job_id。**速率优先取板端 `/health` 的实测值**（`perf.decode_tok_s`，
  由 runtime 的 `perf.generate_tokens` 汇总而来）；只有板端尚无样本（冷启动）时
  才回落到 `DECODE_TOK_S` 档位预设。这样换模型/换板子不需要改连接器配置。
- 状态：working → completed | failed | cancelled；
- 内存表 + TTL（job_ttl_s 默认 600）+ job_max 上限；
- 取消：job.cancel() 幂等，推理侧通过 cancel_check 协作点感知。
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from typing import Any

__all__ = ["JobTable", "estimate_seconds", "sync_token_cap", "job_timeout_s",
           "DECODE_TOK_S", "COLD_PREFILL_TOK_S", "DEFAULT_BYTES_PER_TOKEN",
           "estimate_tokens", "prefill_seconds"]

MIN_SYNC_TOKENS = 8
# 数值安全上限：防止客户端传超大整数（如 10**1000）或板端回报非有限值时
# 在估算里溢出成 OverflowError（P2-A）。超过此值即夹紧，估算上界也随之有界。
MAX_ESTIMATE_TOKENS = 10 ** 7

# 冷启动兜底（板端 `/health.perf.source == "cold"` 时才用）：
# 2026-10-21 真机校准，4B W8A8 45 token / 9.3 s ≈ 4.8 tok/s，取 5 略乐观、
# 长文批量时更低；宁可多转 job 也不让客户端撞 60 s 硬超时。
DECODE_TOK_S = {"4b": 5.0, "7b": 3.5}
# P2.5（A 方案补齐 prefill）：优先用板端 `/health.perf.ttft_s` 的实测首 token
# 延迟做 prefill 估算；只有板端还没有 ttft 样本（冷启动/老版本）时才用这个
# 由 2026-10-21 真机校准的 44 tok/s 兜底。命名保留 `COLD_` 前缀，明确它只是
# 冷启动常量，不再是"prefill 速率"的默认事实源。
COLD_PREFILL_TOK_S = 44.0
# 板端 /health.tokens.bytes_per_token 的兜底（agent/tokens.py 同为 3 字节/token）。
DEFAULT_BYTES_PER_TOKEN = 3.0


def _finite_positive(value) -> float | None:
    """返回有限正 float，否则 None（json.loads 接受 Infinity/NaN，必须拦住）。"""
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return rate if rate > 0 and math.isfinite(rate) else None


def _decode_rate(model_class: str, decode_tok_s: float | None) -> float:
    """解码速率：板端实测优先，冷启动回落档位表，再兜底保守值。"""
    measured = _finite_positive(decode_tok_s)
    return measured if measured is not None \
        else DECODE_TOK_S.get(model_class, 1.5)


def prefill_seconds(prefill_tokens: int, ttft_s: float | None = None) -> float:
    """估算 prefill 耗时（秒）。

    - 板端有实测 `ttft_s`：取 `max(ttft_s, 冷启动折算)`——短 prompt 用实测值，
      长 prompt 不被"短 prompt 的 ttft"低估（保守转 job）。
    - 板端无样本/不可达：按冷启动 44 tok/s 折算，行为与旧实现一致。
    """
    tokens = max(0, min(int(prefill_tokens), MAX_ESTIMATE_TOKENS))
    cold = tokens / COLD_PREFILL_TOK_S
    measured = _finite_positive(ttft_s)
    if measured is None:
        return cold
    return max(measured, cold)


def estimate_tokens(text: str, bytes_per_token: float | None = None) -> int:
    """按板端 tokenizer 口径估算 token 数（UTF-8 字节 / bytes_per_token）。

    与板端 `agent/tokens.py` 的估算口径一致；`bytes_per_token` 来自
    `/health.tokens`，取不到时回落到 3。输入裁剪、prompt 预算、prefill 估算
    共用本函数，避免三处各写一套字符/字节口径（P2.5）。
    """
    if not text:
        return 0
    bpt = _finite_positive(bytes_per_token) or DEFAULT_BYTES_PER_TOKEN
    return max(1, int(len(text.encode("utf-8")) / bpt))


def estimate_seconds(model_class: str, prefill_tokens: int,
                     max_tokens: int,
                     decode_tok_s: float | None = None,
                     ttft_s: float | None = None) -> float:
    """估算耗时。`decode_tok_s`/`ttft_s` 由调用方从板端实测值传入，优先于兜底。"""
    decode = _decode_rate(model_class, decode_tok_s)
    # P2-A：max_tokens 可能是超大合法 JSON 整数（10**1000），直接转 float 溢出。
    # 夹紧到有界上限，估算值随之有界；真实生成仍受板端 max_tokens_limit 约束。
    tokens = max(0, min(int(max_tokens), MAX_ESTIMATE_TOKENS))
    return prefill_seconds(prefill_tokens, ttft_s) + tokens / decode


def sync_token_cap(model_class: str, prefill_tokens: int,
                   threshold_s: float,
                   decode_tok_s: float | None = None,
                   ttft_s: float | None = None) -> int:
    """同步调用的安全生成上限（token）。

    同步路径传给后端/工具的 max_tokens 会被夹到这个值：让最坏情况下的估算
    耗时仍落在阈值内（留 1 s 余量），从而不会因为"默认 max_tokens 很大"而
    真跑过客户端硬超时。被夹断时板端会返回 `finish_reason=length`。

    **prefill/解码速率都来自板端实测**（`ttft_s`/`decode_tok_s`），所以换模型后
    上限自动跟随，不需要改连接器配置。
    """
    decode = _decode_rate(model_class, decode_tok_s)
    try:
        threshold = float(threshold_s)
        if not math.isfinite(threshold):
            threshold = 0.0
    except (TypeError, ValueError, OverflowError):
        threshold = 0.0
    room_s = threshold - prefill_seconds(prefill_tokens, ttft_s) - 1.0
    raw = room_s * decode
    # P2-A：decode 已保证有限，threshold 也做了 isfinite 检查；这里再兜一层，
    # 任何非有限结果都回落到最小安全上限，绝不抛 OverflowError 出去。
    if not math.isfinite(raw):
        return MIN_SYNC_TOKENS
    return max(MIN_SYNC_TOKENS, int(raw))


def job_timeout_s(estimate_s: float, base_s: float = 120.0,
                  factor: float = 1.5, max_s: float = 4200.0) -> float:
    """job 的读超时：估算的 factor 倍 + 30 s 余量，但不小于 base。

    `factor`/`max_s` 是**传输层**常量而非模型参数（板端不知道客户端会等多久），
    因此不再作为配置暴露；上限与板端 `job_timeout_max_s` 的默认对齐。
    """
    return min(float(max_s),
               max(float(base_s), float(estimate_s) * float(factor) + 30.0))



class Job:
    __slots__ = ("id", "state", "result", "error", "error_meta", "created",
                 "short", "_cancel_requested", "_cond")

    def __init__(self, job_id: str, short: str = ""):
        self.id = job_id
        self.state = "working"
        self.result: Any = None
        self.error: str | None = None
        # P2.5：失败终帧的结构化语义（kind/httpStatus/retryAfter），轮询时
        # 放进 _meta，客户端可据此区分鉴权失败/参数错误/可重试忙，而不是只有
        # 一段错误文本。
        self.error_meta: dict = {}
        self.created = time.monotonic()
        # 工具短名（extract 等声明了 outputSchema 的工具，终帧需要补
        # structuredContent；失败/取消轮询也按它决定是否塑形）
        self.short = short
        self._cancel_requested = threading.Event()
        self._cond = threading.Condition()

    def cancel(self) -> bool:
        """幂等取消；返回是否发生了状态转移。"""
        with self._cond:
            if self.state != "working":
                return False
            self.state = "cancelled"
        self._cancel_requested.set()
        with self._cond:
            self._cond.notify_all()
        return True

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def complete(self, result: Any) -> None:
        with self._cond:
            if self.state == "working":
                self.state = "completed"
                self.result = result
            self._cond.notify_all()

    def fail(self, message: str, meta: dict | None = None) -> None:
        with self._cond:
            if self.state == "working":
                self.state = "failed"
                self.error = message
                if meta:
                    self.error_meta = dict(meta)
            self._cond.notify_all()

    def summary(self) -> dict:
        return {"job_id": self.id, "status": self.state}


class JobTable:
    """线程安全的 job 表：TTL 清理 + 容量上限。"""

    def __init__(self, ttl_s: int = 600, max_jobs: int = 8):
        self.ttl_s = ttl_s
        self.max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, short: str = "") -> Job:
        with self._lock:
            self._evict_locked()
            if len(self._jobs) >= self.max_jobs:
                self._drop_oldest_locked()
            job = Job(uuid.uuid4().hex[:12], short)
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and time.monotonic() - job.created > self.ttl_s \
                    and job.state != "working":
                del self._jobs[job_id]
                return None
            return job

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def _evict_locked(self) -> None:
        now = time.monotonic()
        for job_id in [jid for jid, j in self._jobs.items()
                       if j.state != "working"
                       and now - j.created > self.ttl_s]:
            del self._jobs[job_id]

    def _drop_oldest_locked(self) -> None:
        if not self._jobs:
            return
        oldest = min(self._jobs.values(), key=lambda j: j.created)
        if oldest.state == "working":
            oldest.cancel()
        del self._jobs[oldest.id]
