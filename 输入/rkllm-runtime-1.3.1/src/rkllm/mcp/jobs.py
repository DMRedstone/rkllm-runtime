"""长任务 job 状态机（MCP 设计 §5.3 / §5.4）。

- 判定：估算耗时 = prefill/50 + max_tokens/decode（4B 8 / 7B 5 / 27B 2 tok/s），
  超过 task_threshold_s（默认 45）→ 建任务返回 job_id；
- 状态：working → completed | failed | cancelled；
- 内存表 + TTL（job_ttl_s 默认 600）+ job_max 上限；
- 取消：job.cancel() 幂等，推理侧通过 cancel_check 协作点感知。
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

__all__ = ["JobTable", "estimate_seconds", "DECODE_TOK_S"]

DECODE_TOK_S = {"small": 8, "medium": 5, "large": 2}
PREFILL_TOK_S = 50


def estimate_seconds(model_class: str, prefill_tokens: int,
                     max_tokens: int) -> float:
    decode = DECODE_TOK_S.get(model_class, 5)
    return prefill_tokens / PREFILL_TOK_S + max_tokens / decode


class Job:
    __slots__ = ("id", "state", "result", "error", "created", "request_id",
                 "_cancel_requested", "_cond")

    def __init__(self, job_id: str):
        self.id = job_id
        self.state = "working"
        self.result: Any = None
        self.error: str | None = None
        self.created = time.monotonic()
        self.request_id: str | None = None      # 取消传导到后端的句柄
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

    def create(self, request_id: str | None = None) -> Job:
        with self._lock:
            self._evict_locked()
            if len(self._jobs) >= self.max_jobs:
                self._drop_oldest_locked()
            job = Job(uuid.uuid4().hex[:12])
            job.request_id = request_id
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
