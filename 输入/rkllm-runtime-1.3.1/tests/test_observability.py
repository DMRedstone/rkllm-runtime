"""观测性回归：端点维度、耗时分解、资源时序。"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import load_config
from rkllm.server.app import create_app
from rkllm.server.metricsutil import new_labelled_histogram, new_registry, render_text
from rkllm.server.pipeline_runner import record_timing
from rkllm.server.queue import InferenceQueue
from rkllm.server.sysmetrics import ResourceSampler


class _FakeSupervisor:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "你"
        yield "好"
        yield {"perf": {"prefill_tokens": 10, "generate_tokens": 2}}

    def abort(self):
        pass


def _client():
    cfg = load_config(env={"RKLLM_MODEL": "/tmp/x.rkllm"})
    return TestClient(create_app(cfg, supervisor=_FakeSupervisor()))


# -- 端点维度 ---------------------------------------------------------------

def test_labelled_histogram_renders_endpoint():
    reg = new_registry()
    hist = new_labelled_histogram(reg, "rkllm_ttft_histogram_seconds",
                                  "首 token 延迟（秒）", (0.5, 1.0, 2.0))
    hist.labels(endpoint="chat.completions").observe(0.3)
    hist.labels(endpoint="responses").observe(1.2)
    text = render_text(reg)
    assert 'endpoint="chat.completions"' in text
    assert 'endpoint="responses"' in text
    # 两个端点各算各的桶
    assert 'rkllm_ttft_histogram_seconds_count{endpoint="chat.completions"} 1.0' in text
    assert 'rkllm_ttft_histogram_seconds_count{endpoint="responses"} 1.0' in text


def test_responses_gets_its_own_endpoint_label():
    """翻译层必须自成一档：否则它的额外代价被混进 chat 的桶里。"""
    client = _client()
    r = client.post("/v1/responses", json={
        "model": "x",
        "input": [{"role": "user",
                   "content": [{"type": "input_text", "text": "hi"}]}],
    })
    assert r.status_code == 200, r.text
    text = client.get("/metrics").text
    assert 'endpoint="responses"' in text
    assert 'endpoint="chat.completions"' not in text     # 不同端点各自成档


def test_metrics_has_endpoint_after_request():
    client = _client()
    assert client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 8}).status_code == 200
    text = client.get("/metrics").text
    assert 'endpoint="chat.completions"' in text
    assert "rkllm_decode_histogram_seconds" in text
    assert "rkllm_tokens_per_second_histogram" in text


# -- 耗时分解 ---------------------------------------------------------------

def test_record_timing_fills_three_histograms():
    queue = InferenceQueue(max_wait=1, wait_timeout_s=1,
                           request_timeout_s=10)
    record_timing(queue, "responses", ttft_s=0.8, decode_s=4.0, gen_tokens=20)
    text = render_text(queue.metrics_registry)
    assert 'rkllm_ttft_histogram_seconds_count{endpoint="responses"} 1.0' in text
    assert 'rkllm_decode_histogram_seconds_count{endpoint="responses"} 1.0' in text
    # 20 tok / 4 s = 5 tok/s
    assert 'rkllm_tokens_per_second_histogram_sum{endpoint="responses"} 5.0' in text


def test_record_timing_skips_missing_stages():
    queue = InferenceQueue(max_wait=1, wait_timeout_s=1, request_timeout_s=10)
    record_timing(queue, "chat.completions", ttft_s=None, decode_s=None,
                  gen_tokens=0)
    text = render_text(queue.metrics_registry)
    assert "rkllm_ttft_histogram_seconds_count" not in text
    assert "rkllm_decode_histogram_seconds_count" not in text


# -- 资源时序 ---------------------------------------------------------------

def test_sampler_snapshot_keys():
    sampler = ResourceSampler(interval_s=60)
    snap = sampler.sample_once()
    assert set(snap) == {"npu_freq_mhz", "ddr_freq_mhz", "mem_available_mb",
                         "mem_total_mb", "proc_rss_mb"}
    # 内存一定能读到；频率节点在非板端为 None（不报错）
    assert snap["mem_total_mb"] and snap["mem_total_mb"] > 0
    assert snap["proc_rss_mb"] and snap["proc_rss_mb"] > 0


def test_metrics_exposes_resource_gauges():
    client = _client()
    text = client.get("/metrics").text
    for name in ("rkllm_npu_freq_mhz", "rkllm_ddr_freq_mhz",
                 "rkllm_mem_available_mb", "rkllm_mem_total_mb",
                 "rkllm_process_rss_mb"):
        assert name in text
