"""长任务 job 状态机（MCP 设计 §5.3 / §5.4）。

- 判定：估算耗时 = prefill/50 + max_tokens/decode（档位预设，2026-10-21 真机
  校准：4B 4.5 / 7B 3.5 / 27B 1.5 tok/s），超过 task_threshold_s（默认 45）→
  建任务返回 job_id；
- 状态：working → completed | failed | cancelled；
- 内存表 + TTL（job_ttl_s 默认 600）+ job_max 上限；
- 取消：job.cancel() 幂等，推理侧通过 cancel_check 协作点感知。
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

__all__ = ["JobTable", "estimate_seconds", "sync_token_cap", "job_timeout_s",
           "DECODE_TOK_S"]

MIN_SYNC_TOKENS = 8

# 解码速度（tok/s）：设计 §5.3 给的是 8 / 5 / 2，2026-10-21 真机校准后下调——
# OrangePi 5 Plus / qwen3.5-4b W8A8 实测 45 token / 9.3 s ≈ 4.8 tok/s（单发），
# 长文批量时更低。按 8 tok/s 估算会把 300 token 的请求算成 38.7 s（< 45 阈值）
# 而同步执行，实际却要 80 s，直接撞上客户端 60 s 硬超时——这是把长任务判错的
# 方向性错误，因此以实测值下压，宁可多转 job 也不让客户端超时。
DECODE_TOK_S = {"small": 4.5, "medium": 3.5, "large": 1.5}
PREFILL_TOK_S = 50


def estimate_seconds(model_class: str, prefill_tokens: int,
                     max_tokens: int) -> float:
    decode = DECODE_TOK_S.get(model_class, 5)
    return prefill_tokens / PREFILL_TOK_S + max_tokens / decode


def sync_token_cap(model_class: str, prefill_tokens: int,
                   threshold_s: float) -> int:
    """同步调用的安全生成上限（token）。

    同步路径传给后端/工具的 max_tokens 会被夹到这个值：让最坏情况下的估算
    耗时仍落在 task_threshold_s 内（留 1 s 余量），从而不会因为"默认 max_tokens
    很大"而把短任务误判成 job，也不会让同步调用真的跑过客户端 60 s 硬超时。
    被夹断时后端会返回 finish_reason=length，工具层会给出明确提示。
    """
    decode = DECODE_TOK_S.get(model_class, 5)
    room_s = float(threshold_s) - prefill_tokens / PREFILL_TOK_S - 1.0
    return max(MIN_SYNC_TOKENS, int(room_s * decode))


def job_timeout_s(estimate_s: float, base_s: float = 120.0,
                  factor: float = 1.5, max_s: float = 1800.0) -> float:
    """job 的读超时：估算的 factor 倍 + 30 s 余量，但不小于 base。

    旧实现 job 与同步共用 target_timeout_s=120：27B/large 512 token 的估算
    约 341 s，必然被 120 s 总超时掐断。job 是后台任务，读超时应按估算放宽。
    """
    return min(float(max_s),
               max(float(base_s), float(estimate_s) * float(factor) + 30.0))


class Job:
    __slots__ = ("id", "state", "result", "error", "created", "short",
                 "_cancel_requested", "_cond")

    def __init__(self, job_id: str, short: str = ""):
        self.id = job_id
        self.state = "working"
        self.result: Any = None
        self.error: str | None = None
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

    def fail(self, message: str) -> None:
        with self._cond:
            if self.state == "working":
                self.state = "failed"
                self.error = message
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
