"""Prometheus 指标：直方图与标量快照回归。

直方图已改用 `prometheus_client`（Debian python3-prometheus-client）：
本文件验证"喂进观测值后，exposition 里的累计桶/count/sum 正确"，
以及标量快照能正常渲染成标准文本。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.server.metricsutil import (QUEUE_WAIT_BUCKETS, TTFT_BUCKETS,
                                      new_histogram, new_registry, render_text,
                                      scalar_registry)


def _buckets(text: str, name: str) -> dict[str, float]:
    found = {}
    for line in text.splitlines():
        if line.startswith(name + "_bucket{"):
            key, value = line.split()
            found[key.split('le="')[1].rstrip('"}')] = float(value)
    return found


def test_histogram_is_cumulative_on_render():
    registry = new_registry()
    hist = new_histogram(registry, "rkllm_ttft_histogram_seconds",
                         "首 token 延迟（秒）", TTFT_BUCKETS)
    for value in (0.1, 0.3, 0.8, 2.5, 30.0):
        hist.observe(value)
    text = render_text(registry)
    buckets = _buckets(text, "rkllm_ttft_histogram_seconds")
    assert buckets["0.25"] == 1.0            # 0.1
    assert buckets["1.0"] == 3.0             # 0.1 0.3 0.8
    assert buckets["+Inf"] == 5.0
    assert "rkllm_ttft_histogram_seconds_count 5.0" in text
    assert "rkllm_ttft_histogram_seconds_sum 33.7" in text


def test_two_registries_do_not_collide():
    """多实例/多测试各自建 registry，同名指标互不冲突。"""
    reg_a, reg_b = new_registry(), new_registry()
    first = new_histogram(reg_a, "rkllm_queue_wait_histogram_seconds",
                          "排队等待时长（秒）", QUEUE_WAIT_BUCKETS)
    second = new_histogram(reg_b, "rkllm_queue_wait_histogram_seconds",
                           "排队等待时长（秒）", QUEUE_WAIT_BUCKETS)
    first.observe(0.05)
    second.observe(3.0)
    assert "rkllm_queue_wait_histogram_seconds_count 1.0" in render_text(reg_a)
    assert "rkllm_queue_wait_histogram_seconds_count 1.0" in render_text(reg_b)


def test_scalar_registry_renders_counters_gauges_labels():
    registry = scalar_registry(
        {"rkllm_requests_total": ("受理的请求数", 7)},
        {"rkllm_queue_depth": ("当前排队请求数", 2)},
        [("rkllm_worker_state", "worker 状态", {"state": "ready"}, 1.0)],
    )
    text = render_text(registry)
    assert "rkllm_requests_total 7.0" in text
    assert "rkllm_queue_depth 2.0" in text
    assert 'rkllm_worker_state{state="ready"} 1.0' in text


def test_zero_counter_is_still_exposed():
    """值为 0 也要出现（监控端需要区分"没有该指标"与"计数为 0"）。"""
    registry = scalar_registry({"rkllm_rejected_total": ("被拒绝数", 0)})
    assert "rkllm_rejected_total 0.0" in render_text(registry)
