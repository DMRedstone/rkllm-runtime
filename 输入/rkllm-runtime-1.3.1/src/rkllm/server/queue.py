"""单槽推理队列：背压模型（沿用旧实现并实测过的口径）。

吞吐恒等于 1/延迟，排队只增尾延迟——有界队列 + 快速失败优于无限排队。
max_wait=0 表示不排队；等待超时与推理时长上限按配置执行。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterator

from .metricsutil import (QUEUE_WAIT_BUCKETS, TTFT_BUCKETS, init_hist,
                          observe)

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
                        "errors_total": 0, "queue_wait_seconds_sum": 0.0,
                        "infer_seconds_sum": 0.0, "tokens_total": 0,
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "max_tokens_clamped_total": 0}
        init_hist(self.metrics, "ttft", TTFT_BUCKETS)
        init_hist(self.metrics, "queue_wait", QUEUE_WAIT_BUCKETS)

    @property
    def queue_depth(self) -> int:
        return self._waiting

    def run(self, submit: Callable[[], Iterator[str]], *,
            cancelled: Callable[[], bool] | None = None,
            on_acquire: Callable[[], None] | None = None,
            on_release: Callable[[], None] | None = None) -> Iterator[str]:
        """排队 → 拿槽 → 执行；submit() 返回 token 迭代器（由 worker 客户端给）。

        max_wait<=0 表示「不排队」：仅在槽立即空闲时放行，否则立即 503（P1-2）。

        `on_acquire` / `on_release`：执行槽的持/放回调（F-2，2026-10-27 审核）。
        `streaming.CancelToken` 靠它们判断「本请求是否真的持有 NPU 槽」：只有
        持槽期间才允许 `supervisor.abort()`，排队期断连不得打断别人的在飞请求。
        两个回调在同一 try/finally 里成对调用；拿到槽后立刻再查一次 `cancelled`，
        已取消就不进入 `submit()`（不写 run 帧、不启动推理，也就不需要 abort）。
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
                enqueued_at = time.monotonic()
                deadline = enqueued_at + self.wait_timeout_s
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
                # 排队时长 = 入队到拿槽（此前 start 记在拿槽之后，
                # queue_wait_seconds_sum 实际累加的是推理执行时长）
                waited = time.monotonic() - enqueued_at
                self.metrics["queue_wait_seconds_sum"] += waited
                observe(self.metrics, "queue_wait", QUEUE_WAIT_BUCKETS, waited)
            # -- 已持有执行槽：以下任何路径都必须成对 on_release + release ----
            infer_start = time.monotonic()
            acquire_notified = False
            try:
                # 拿到槽后立刻再查一次取消：排队期断连的请求到此为止，
                # 不 submit()（不写 run 帧、不启动 NPU），也不需要 abort。
                if cancelled and cancelled():
                    raise QueueTimeout("cancelled after acquiring slot")
                # on_acquire 放在取消检查之后、submit() 之前：回调一旦触发
                # 就表示"本请求即将/已经开始推理"，此时才允许 abort worker。
                if on_acquire is not None:
                    on_acquire()
                acquire_notified = True
                for token in submit():
                    if cancelled and cancelled():
                        break
                    yield token
                    self.metrics["tokens_total"] += 1
                self.metrics["infer_seconds_sum"] += (time.monotonic()
                                                     - infer_start)
            finally:
                # 回调严格配对：on_acquire 没触发过就不调 on_release；
                # 即使 on_release 抛异常也必须先归还槽位（否则整条队列死锁）。
                try:
                    if acquire_notified and on_release is not None:
                        on_release()
                finally:
                    self._slot.release()
        except (QueueBusy, QueueTimeout):
            raise                      # 已计 rejected，不重复计 errors
        except Exception:
            # 请求级失败（RequestFailed/WorkerDown 等）：errors 计数供
            # /health stats 兼容字段与 Prometheus 使用
            self.metrics["errors_total"] += 1
            raise
        finally:
            if waiting_counted:
                with self._lock:
                    self._waiting -= 1
