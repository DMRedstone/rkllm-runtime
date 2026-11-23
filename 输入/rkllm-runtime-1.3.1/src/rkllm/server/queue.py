"""单槽推理队列：背压模型（沿用旧实现并实测过的口径）。

吞吐恒等于 1/延迟，排队只增尾延迟——有界队列 + 快速失败优于无限排队。
max_wait=0 表示不排队；等待超时与推理时长上限按配置执行。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterator

from .metricsutil import (DECODE_BUCKETS, INFERENCE_BUCKETS,
                          QUEUE_WAIT_BUCKETS, TOK_S_BUCKETS, TTFT_BUCKETS,
                          new_labelled_histogram, new_registry)

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
        # metrics 是普通 dict，`metrics[k] += 1` 在多线程下会丢更新
        # （2026-11-20 审计）。所有写点改走 bump()/gauge()，在此锁内更新；
        # 锁序固定为 _lock → _metrics_lock，无反向获取。
        self._metrics_lock = threading.Lock()
        # 指标（Prometheus 文本由 routes/ops 渲染）
        self.metrics = {"requests_total": 0, "rejected_total": 0,
                        "errors_total": 0, "queue_wait_seconds_sum": 0.0,
                        "infer_seconds_sum": 0.0, "tokens_total": 0,
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "max_tokens_clamped_total": 0}
        # 直方图由 prometheus_client 持有（独立 registry，见 metricsutil）
        self.metrics_registry = new_registry()
        self.ttft_hist = new_labelled_histogram(
            self.metrics_registry, "rkllm_ttft_histogram_seconds",
            "首 token 延迟（秒）", TTFT_BUCKETS)
        self.queue_wait_hist = new_labelled_histogram(
            self.metrics_registry, "rkllm_queue_wait_histogram_seconds",
            "排队等待时长（秒）", QUEUE_WAIT_BUCKETS)
        # P2-4：推理时长与排队等待分开计量。口径：
        #   queue_wait_seconds = 入队 → 拿到执行槽（立即拿到则记 0）
        #   inference_seconds  = 拿到执行槽 → 本次请求结束（不含排队）
        # 两者相加 ≈ 请求总耗时；/metrics 里各自独立，便于分别告警。
        self.infer_hist = new_labelled_histogram(
            self.metrics_registry, "rkllm_inference_histogram_seconds",
            "推理时长（拿到执行槽后，秒）", INFERENCE_BUCKETS)
        self.decode_hist = new_labelled_histogram(
            self.metrics_registry, "rkllm_decode_histogram_seconds",
            "解码阶段时长（首 token 之后，秒）", DECODE_BUCKETS)
        self.tok_s_hist = new_labelled_histogram(
            self.metrics_registry, "rkllm_tokens_per_second_histogram",
            "生成速率（token/s）", TOK_S_BUCKETS)

    def bump(self, key: str, delta=1) -> None:
        """原子累加计数（并发请求下必须走这里，不要直接 `metrics[k] += x`）。"""
        with self._metrics_lock:
            self.metrics[key] = self.metrics.get(key, 0) + delta

    def gauge(self, key: str, value) -> None:
        """写入瞬时值（TTFT 等），同样在锁内更新。"""
        with self._metrics_lock:
            self.metrics[key] = value

    # -- 口径分离的计量（P2-4） --------------------------------------------
    def _record_queue_wait(self, endpoint: str, waited_s: float) -> None:
        """排队等待时长：入队 → 拿到执行槽（立即拿槽记 0，口径完整）。

        与 `inference_seconds` 分离：总量 = 排队 + 推理，两段分别可告警；
        不再让"总耗时"混进排队时间（此前 duration/TTFT 从请求开始计时，
        会把队列等待算进模型性能）。
        """
        waited = max(0.0, float(waited_s))
        self.bump("queue_wait_seconds_sum", waited)
        self.queue_wait_hist.labels(endpoint=endpoint).observe(waited)

    def _record_inference(self, endpoint: str, infer_s: float) -> None:
        """推理时长：拿到执行槽 → 本次请求结束（成功/失败/取消都记）。"""
        infer = max(0.0, float(infer_s))
        self.bump("infer_seconds_sum", infer)
        self.infer_hist.labels(endpoint=endpoint).observe(infer)

    @property
    def queue_depth(self) -> int:
        return self._waiting

    def run(self, submit: Callable[[], Iterator[str]], *,
            cancelled: Callable[[], bool] | None = None,
            on_acquire: Callable[[], None] | None = None,
            on_release: Callable[[], None] | None = None,
            endpoint: str = "unknown") -> Iterator[str]:
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
                    self.bump("rejected_total")
                    raise QueueBusy(self.busy_retry_after_s)
                acquired = True
                self.bump("requests_total")
            else:
                if self._waiting >= self.max_wait:
                    self.bump("rejected_total")
                    raise QueueBusy(self.busy_retry_after_s)
                self._waiting += 1
                waiting_counted = True
                self.bump("requests_total")
        try:
            enqueued_at = time.monotonic()
            if not acquired:
                deadline = enqueued_at + self.wait_timeout_s
                try:
                    while True:
                        if cancelled and cancelled():
                            raise QueueTimeout("cancelled while waiting")
                        if self._slot.acquire(timeout=0.5):
                            acquired = True
                            break
                        if time.monotonic() > deadline:
                            self.bump("rejected_total")
                            raise QueueTimeout("queue wait timeout %ds"
                                               % self.wait_timeout_s)
                except QueueTimeout:
                    # 排队超时也要记录已等待时长：只统计成功请求会让直方图
                    # 系统性低估排队压力（运维据此判断该不该加队列）。
                    if not (cancelled and cancelled()):
                        self._record_queue_wait(
                            endpoint, time.monotonic() - enqueued_at)
                    raise
            # 立即拿槽也记 0，保证「queue_wait = 等待时长」这个口径完整
            self._record_queue_wait(endpoint, time.monotonic() - enqueued_at)
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
                    # P2-4：只把真实 token 计入 tokens_total。管线会同时 yield
                    # dict 元数据（perf / reasoning_delta / tool_calls），旧实现
                    # 对每个 item 都 bump 一次，导致指标被系统性高估。
                    if isinstance(token, str):
                        self.bump("tokens_total")
            finally:
                # 推理时长口径：拿到执行槽 → 结束（成功/失败/取消都记录；
                # 与排队等待分开，/metrics 侧可见两段各自的分布）。
                if acquire_notified:
                    self._record_inference(
                        endpoint, time.monotonic() - infer_start)
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
            self.bump("errors_total")
            raise
        finally:
            if waiting_counted:
                with self._lock:
                    self._waiting -= 1
