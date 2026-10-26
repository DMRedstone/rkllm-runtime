"""Prometheus 直方图与 clamp 指标回归。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.server.metricsutil import (QUEUE_WAIT_BUCKETS, TTFT_BUCKETS,
                                      init_hist, observe, render_hist)


def test_histogram_is_cumulative_on_render():
    metrics = {}
    init_hist(metrics, "ttft", TTFT_BUCKETS)
    for value in (0.1, 0.3, 0.8, 2.5, 30.0):
        observe(metrics, "ttft", TTFT_BUCKETS, value)
    lines = render_hist(metrics, "rkllm_ttft_histogram_seconds", "ttft",
                        TTFT_BUCKETS)
    buckets = {}
    for line in lines:
        if "_bucket{" in line:
            key, value = line.split()
            buckets[key] = int(value)
    assert buckets['rkllm_ttft_histogram_seconds_bucket{le="0.25"}'] == 1
    assert buckets['rkllm_ttft_histogram_seconds_bucket{le="1.0"}'] == 3
    assert buckets['rkllm_ttft_histogram_seconds_bucket{le="+Inf"}'] == 5
    assert "rkllm_ttft_histogram_seconds_count 5" in lines
    assert "rkllm_ttft_histogram_seconds_sum 33.700000" in lines


def test_queue_wait_histogram_init_keys():
    metrics = {}
    init_hist(metrics, "queue_wait", QUEUE_WAIT_BUCKETS)
    assert metrics["queue_wait_count"] == 0
    assert metrics["queue_wait_bucket_%s" % QUEUE_WAIT_BUCKETS[0]] == 0
