"""Prometheus 指标：直方图交给 `prometheus_client`（Debian `python3-prometheus-client`）。

此前是手写的"桶计数 + 渲染时累加"，两个直方图就要自己保证累计语义、`le` 排序、
`+Inf` 桶与线程安全；换成成熟库后这些都不必再维护，新增指标也只是加一个对象。

分工：
- **直方图**：观测时就地累加（`Histogram.observe`），对象挂在 `SemaphoreQueue` 上，
  用一个独立的 `CollectorRegistry`，避免多个实例/测试互相污染。
- **标量**：仍以 dict 计数（`/health` 也读它），渲染时临时灌进一个新 registry，
  这样 `/metrics` 输出不需要任何全局可变注册状态。
"""

from __future__ import annotations

from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                               generate_latest)

__all__ = ["TTFT_BUCKETS", "QUEUE_WAIT_BUCKETS", "DECODE_BUCKETS",
           "INFERENCE_BUCKETS", "TOK_S_BUCKETS", "new_registry", "new_histogram",
           "new_labelled_histogram", "render_text", "scalar_registry"]

TTFT_BUCKETS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
QUEUE_WAIT_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
# 解码阶段（首 token 之后到结束）：板端 4b 实测 ~5 tok/s，512 token 约 100 s
DECODE_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
# P2-4：推理阶段总时长（拿到执行槽 → 结束，**不含排队等待**）。口径与
# queue_wait 分离：两个直方图相加 ≈ 请求总耗时，但各自可独立告警。
INFERENCE_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0,
                     320.0, 640.0)
# 生成速率（tok/s）：8G 板 4b 约 5、7b 约 3.5，低于 1 说明严重退化
TOK_S_BUCKETS = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0)


def new_registry() -> CollectorRegistry:
    """独立注册表：一个服务实例一份，测试之间不串味。"""
    return CollectorRegistry()


def new_histogram(registry: CollectorRegistry, name: str, help_text: str,
                  buckets: tuple[float, ...]) -> Histogram:
    """建一个直方图；名字沿用旧的 `rkllm_*_histogram_seconds` 口径。"""
    return Histogram(name, help_text, buckets=buckets, registry=registry)


def new_labelled_histogram(registry: CollectorRegistry, name: str,
                           help_text: str, buckets: tuple[float, ...],
                           labelnames: tuple[str, ...] = ("endpoint",)) -> Histogram:
    """带标签的直方图（当前只有 `endpoint` 一个维度）。

    为什么值得打标签：`/v1/responses` 是翻译层，会多一次内部构造；它的 TTFT 与
    解码耗时若和 `chat.completions` 混在一个桶里，就永远看不出代价。端点只有
    3–4 个，基数可忽略。
    """
    return Histogram(name, help_text, labelnames=labelnames, buckets=buckets,
                     registry=registry)


def scalar_registry(counters: dict[str, tuple[str, float]],
                    gauges: dict[str, tuple[str, float]] | None = None,
                    labelled: list[tuple[str, str, dict[str, str], float]]
                    | None = None) -> CollectorRegistry:
    """把当次快照的标量灌进一个临时 registry（只用于本次渲染）。

    counters / gauges：{指标名: (help, 值)}
    labelled：[(指标名, help, 标签, 值)]，用于 `rkllm_worker_state{state="ready"}`
    这类带标签的 gauge——prometheus 要求同名同 help 同标签集，因此用
    `labels(**tag).set(v)` 而不是每次新建。
    """
    registry = CollectorRegistry()
    for name, (help_text, value) in counters.items():
        counter = Counter(name, help_text, registry=registry)
        if value:
            counter.inc(float(value))
    for name, (help_text, value) in (gauges or {}).items():
        Gauge(name, help_text, registry=registry).set(float(value))
    for name, help_text, tags, value in (labelled or []):
        gauge = Gauge(name, help_text, labelnames=tuple(tags), registry=registry)
        gauge.labels(**tags).set(float(value))
    return registry


def render_text(*registries: CollectorRegistry) -> str:
    """按 Prometheus 文本格式拼接多个 registry（同名指标不会跨段重复）。"""
    return b"".join(generate_latest(r) for r in registries).decode("utf-8")
