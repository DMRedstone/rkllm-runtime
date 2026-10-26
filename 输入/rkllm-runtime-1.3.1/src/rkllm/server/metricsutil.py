"""Prometheus 直方图的最小实现（无第三方依赖）。

当前只保留少量高价值观测：TTFT 与队列等待。桶值按秒，`observe` 记录原始
桶计数，渲染时再累加为 Prometheus 的累计桶，避免运行期做前缀和。
"""

from __future__ import annotations

__all__ = ["TTFT_BUCKETS", "QUEUE_WAIT_BUCKETS", "init_hist", "observe",
           "render_hist"]

TTFT_BUCKETS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
QUEUE_WAIT_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


def init_hist(metrics: dict, prefix: str, buckets: tuple[float, ...]) -> None:
    metrics.setdefault(prefix + "_count", 0)
    metrics.setdefault(prefix + "_sum", 0.0)
    for le in buckets:
        metrics.setdefault("%s_bucket_%s" % (prefix, le), 0)


def observe(metrics: dict, prefix: str, buckets: tuple[float, ...],
            value: float) -> None:
    metrics[prefix + "_count"] = metrics.get(prefix + "_count", 0) + 1
    metrics[prefix + "_sum"] = metrics.get(prefix + "_sum", 0.0) + value
    for le in buckets:
        if value <= le:
            key = "%s_bucket_%s" % (prefix, le)
            metrics[key] = metrics.get(key, 0) + 1
            break


def render_hist(metrics: dict, name: str, prefix: str,
                buckets: tuple[float, ...]) -> list[str]:
    lines = []
    cumulative = 0
    for le in buckets:
        cumulative += metrics.get("%s_bucket_%s" % (prefix, le), 0)
        lines.append('%s_bucket{le="%s"} %d' % (name, le, cumulative))
    lines.append('%s_bucket{le="+Inf"} %d'
                 % (name, metrics.get(prefix + "_count", 0)))
    lines.append("%s_count %d" % (name, metrics.get(prefix + "_count", 0)))
    lines.append("%s_sum %f" % (name, metrics.get(prefix + "_sum", 0.0)))
    return lines
