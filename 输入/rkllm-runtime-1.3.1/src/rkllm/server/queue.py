"""单槽推理队列：背压模型（沿用旧实现并实测过的口径）。

吞吐恒等于 1/延迟，排队只增尾延迟——有界队列 + 快速失败优于无限排队。
max_wait=0 表示不排队；等待超时与推理时长上限按配置执行。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterator

__all__ = ["QueueBusy", "QueueTimeout", "InferenceQueue"]

log = logging.getLogger("rkllm.server.queue")


class QueueBusy(Exception):
    def __init__(self, retry_after_s: int):
        super().__init__("model busy")
        self.retry_after_s = retry_after_s


class QueueTimeout(Exception):
    pass


class InferenceQueue:
    """全局单执行槽 + 有界等待队列；等待中可取消（取消不占额度）。"""

    def __init__(self, max_wait: int = 32, wait_timeout_s: int = 600,
                 request_timeout_s: int = 900, busy_retry_after_s: int = 5,
                 max_concurrent: int = 1):
        # queue.max_concurrent：真正的并发上限。单 NPU 只有一个引擎，engine 内部
        # 还会拒绝并发 stream()，所以 >1 在这里只会有害无益——读取该键、夹紧为 1
        # 并告警，比"键存在但不生效"诚实。
        self.max_concurrent = 1
        if int(max_concurrent) > 1:
            # A-3：告警走日志系统（原先 sys.stderr.write，配 logging.file 时看不到）
            log.warning("queue.max_concurrent=%d 已夹紧为 1：单 NPU 单引擎串行",
                        int(max_concurrent))
        self.max_wait = max_wait
        self.wait_timeout_s = wait_timeout_s
        self.request_timeout_s = request_timeout_s
        self.busy_retry_after_s = busy_retry_after_s
        self._slot = threading.Semaphore(1)
        self._waiting = 0
        self._lock = threading.Lock()
        # 指标（Prometheus 文本由 routes/ops 渲染）
        self.metrics = {"requests_total": 0, "rejected_total": 0,
                        "queue_wait_seconds_sum": 0.0, "tokens_total": 0}

    @property
    def queue_depth(self) -> int:
        return self._waiting

    def run(self, submit: Callable[[], Iterator[str]], *,
            cancelled: Callable[[], bool] | None = None) -> Iterator[str]:
        """排队 → 拿槽 → 执行；submit() 返回 token 迭代器（由 worker 客户端给）。

        max_wait<=0 表示「不排队」：仅在槽立即空闲时放行，否则立即 503（P1-2）。
        """
        acquired = False
        waiting_counted = False
        with self._lock:
            if self.max_wait <= 0:
                # 不排队模式：非阻塞抢槽，抢不到立即 503（旧实现 0>=0 恒真→全线 503）
                if not self._slot.acquire(blocking=False):
                    self.metrics["rejected_total"] += 1
                    raise QueueBusy(self.busy_retry_after_s)
                acquired = True
                self.metrics["requests_total"] += 1
            else:
                if self._waiting >= self.max_wait:
                    self.metrics["rejected_total"] += 1
                    raise QueueBusy(self.busy_retry_after_s)
                self._waiting += 1
                waiting_counted = True
                self.metrics["requests_total"] += 1
        try:
            if not acquired:
                deadline = time.monotonic() + self.wait_timeout_s
                while True:
                    if cancelled and cancelled():
                        raise QueueTimeout("cancelled while waiting")
                    if self._slot.acquire(timeout=0.5):
                        acquired = True
                        break
                    if time.monotonic() > deadline:
                        self.metrics["rejected_total"] += 1
                        raise QueueTimeout("queue wait timeout %ds"
                                           % self.wait_timeout_s)
            start = time.monotonic()
            try:
                for token in submit():
                    if cancelled and cancelled():
                        break
                    yield token
                    self.metrics["tokens_total"] += 1
                self.metrics["queue_wait_seconds_sum"] += time.monotonic() - start
            finally:
                self._slot.release()
        finally:
            if waiting_counted:
                with self._lock:
                    self._waiting -= 1
