"""Token 计数回归：自校准估算 + 精确 tokenizer 的优先级。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.agent import tokens
from rkllm.config import load_config
from rkllm.server.app import create_app


@pytest.fixture(autouse=True)
def _reset_tokens():
    tokens.reset()
    yield
    tokens.reset()


def test_cold_start_uses_three_bytes_per_token():
    # "你好" = 6 字节 → 2 token（中文 ≈1 token/字）
    assert tokens.count_tokens("你好") == 2
    # 空串不算 token
    assert tokens.count_tokens("") == 0
    assert tokens.stats()["backend"] == "estimate"


def test_calibration_tracks_runtime_truth():
    text = "x" * 300                      # 300 字节
    before = tokens.count_tokens(text)    # 默认 3.0 → 100
    # runtime 实报 150 token（≈2 字节/token，例如英文占比高）
    tokens.calibrate(len(text.encode()), 150)
    after = tokens.count_tokens(text)
    assert after > before, "校准后应更接近真值（更多 token）"
    assert tokens.stats()["samples"] == 1
    assert tokens.stats()["bytes_per_token"] == pytest.approx(2.0, rel=0.01)


def test_calibration_is_ewma_not_last_wins():
    tokens.calibrate(200, 100)            # 2.0
    tokens.calibrate(400, 100)            # 观测 4.0
    ratio = tokens.stats()["bytes_per_token"]
    assert 2.0 < ratio < 4.0, "EWMA 不应直接跳到最新样本"


def test_calibration_ignores_bad_samples():
    tokens.reset()
    # 空提示 / 0 token / 比例 1000（离谱）/ 提示太短（噪声大）
    for body in ((0, 10), (100, 0), (1000, 1), (10, 100)):
        tokens.calibrate(*body)
    assert tokens.stats()["samples"] == 0
    assert tokens.stats()["bytes_per_token"] == 3.0


def test_short_prompt_is_not_a_sample():
    """短提示的比例受模板开销影响太大，不能用来校准。"""
    tokens.reset()
    tokens.calibrate(20, 8)               # 20 字节 < 64，忽略
    assert tokens.stats()["samples"] == 0


def test_exact_tokenizer_takes_priority(tmp_path):
    """装了 tokenizers 且有词表时，校准不应该再动估算值。"""
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "hello": 1, "world": 2, "你好": 3}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))

    state = tokens.configure(str(tmp_path / "model.rkllm"))
    assert state["backend"] == "tokenizers"
    assert tokens.count_tokens("hello world") == 2
    # 精确档下校准被忽略
    tokens.calibrate(11, 999)
    assert tokens.stats()["samples"] == 0


def test_no_tokenizer_file_keeps_estimate(tmp_path):
    (tmp_path / "model.rkllm").write_bytes(b"x")
    state = tokens.configure(str(tmp_path / "model.rkllm"))
    assert state["backend"] == "estimate"


class _FakeSupervisor:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "ok"
        # 与真实量级一致：~100 字节提示约 40 token（2.5 字节/token）
        yield {"perf": {"prefill_tokens": 40, "generate_tokens": 2}}

    def abort(self):
        pass


def test_request_calibrates_and_exposes_state():
    cfg = load_config(env={"RKLLM_MODEL": "/tmp/x.rkllm"})
    long_text = "请把下面这段话改写成一句话摘要：" + "内容" * 40   # >64 字节
    with TestClient(create_app(cfg, supervisor=_FakeSupervisor())) as client:
        assert client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user",
                                               "content": long_text}],
                                 "max_tokens": 8}).status_code == 200
        health = client.get("/health").json()
        assert health["tokens"]["samples"] >= 1
        assert health["tokens"]["backend"] == "estimate"
        text = client.get("/metrics").text
        assert "rkllm_token_bytes_per_token" in text
        assert "rkllm_token_calibrations_total" in text
