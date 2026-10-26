"""路由级 P0/可观测性回归：400、clamp 头、直方图、clamp 计数。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import load_config
from rkllm.server.app import create_app


class _Fake:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self):
        self.model_name = "fake"

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "你"
        yield "好"
        yield {"perf": {"prefill_tokens": 3, "generate_tokens": 2}}

    def abort(self):
        pass


@pytest.fixture()
def client():
    cfg = load_config(env={})
    app = create_app(cfg, supervisor=_Fake())
    with TestClient(app) as c:
        yield c


def test_long_prompt_returns_400_instead_of_rkllm_run_minus_one(client):
    huge = "字" * 20000            # ~20000 tokens, context 4096
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": huge}],
                          "max_tokens": 32})
    assert r.status_code == 400
    assert "prompt too long" in r.json()["error"]["message"]


def test_clamped_max_tokens_header_and_metric(client):
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 999999})
    assert r.status_code == 200
    assert "X-RKLLM-Max-Tokens-Clamped" in r.headers
    assert int(r.headers["X-RKLLM-Max-Tokens-Clamped"]) < 4096
    metrics = client.get("/metrics").text
    assert "rkllm_max_tokens_clamped_total 1" in metrics


def test_responses_clamp_header(client):
    r = client.post("/v1/responses",
                    json={"input": "hi", "max_output_tokens": 999999})
    assert r.status_code == 200, r.text
    assert "X-RKLLM-Max-Tokens-Clamped" in r.headers


def test_metrics_exposes_ttft_and_queue_histograms(client):
    client.post("/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 16})
    text = client.get("/metrics").text
    assert "rkllm_ttft_histogram_seconds_bucket" in text
    assert "rkllm_queue_wait_histogram_seconds_bucket" in text
    assert "rkllm_worker_heartbeat_timeouts_total" in text
