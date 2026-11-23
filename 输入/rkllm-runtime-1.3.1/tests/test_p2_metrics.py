"""P2-4 指标 / DoS / 资源回归（真实路径优先）。

覆盖（对应 task-18）：
1. `tokens_total` 只统计真实 token，排除 perf/tool_calls 等 dict 元数据；
2. TTFT/duration 与 queue-wait 口径分离：queue 侧记录
   `queue_wait_seconds`（入队 → 拿槽，立即拿槽记 0）与
   `inference_seconds`（拿槽 → 结束）；两者分直方图、互不混入；
3. `/health.perf` 跨端点聚合（只走 completions/responses 的部署不再 cold）；
4. chunked body 先鉴权再读体：未授权直接 401 且**不读** body，授权后仍执行
   体积上限（413）；未启用鉴权的 chunked 行为不变；
5. prompt cache 的 meta/stats 文件 0600、目录 0700（POSIX 精确断言；
   Windows 权限语义不同，只验证功能不回归）。

说明：真实 `load_config()` + `create_app()` + TestClient（或直接 ASGI 调用，
用于精确断言"body 没有被读"）；supervisor 是唯一替身（板端 NPU 不可用）。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import load_config                       # noqa: E402
from rkllm.server.app import create_app                    # noqa: E402
from rkllm.server.metricsutil import (INFERENCE_BUCKETS,   # noqa: E402
                                      render_text)
from rkllm.server.queue import InferenceQueue, QueueTimeout  # noqa: E402
from rkllm.worker.promptcache import PromptCache            # noqa: E402

MSG = {"messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# 假的 supervisor 边界
# ---------------------------------------------------------------------------


class FakeSupervisor:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None, perf=None):
        self.tokens = list(tokens if tokens is not None else ["你", "好"])
        self.perf = perf
        self.prompts: list[str] = []

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        self.prompts.append(prompt)
        for token in self.tokens:
            yield token
        if self.perf:
            yield {"perf": self.perf}

    def abort(self):
        pass

    def shutdown(self):
        pass


def _client(env=None, supervisor=None):
    cfg = load_config(env=env or {})
    app = create_app(cfg, supervisor=supervisor or FakeSupervisor())
    return TestClient(app)


def _metric_value(text: str, name: str) -> float | None:
    match = re.search(r"^%s ([0-9.eE+-]+)$" % re.escape(name), text, re.M)
    return float(match.group(1)) if match else None


def _hist_totals(hist) -> tuple[float, int]:
    """返回直方图所有标签下的 (sum, count)。"""
    total, count = 0.0, 0
    for metric in hist.collect():
        for sample in metric.samples:
            if sample.name.endswith("_sum"):
                total += float(sample.value)
            elif sample.name.endswith("_count"):
                count += int(sample.value)
    return total, count


# ---------------------------------------------------------------------------
# 1) tokens_total 只统计真实 token
# ---------------------------------------------------------------------------


def test_queue_tokens_total_excludes_metadata():
    """dict 元数据（perf/tool_calls/reasoning_delta）不是 token。"""
    queue = InferenceQueue(max_wait=1)
    items = list(queue.run(lambda: iter([
        "a", {"perf": {"generate_tokens": 99}}, "b",
        {"tool_calls": [{"id": "c"}]}, {"reasoning_delta": "思考"}])))
    assert len(items) == 5
    assert queue.metrics["tokens_total"] == 2


def test_metrics_tokens_total_counts_only_real_tokens():
    supervisor = FakeSupervisor(perf={"prefill_tokens": 5,
                                      "generate_tokens": 2})
    client = _client(supervisor=supervisor)
    assert client.post("/v1/chat/completions", json={
        **MSG, "stream": True}).status_code == 200
    text = client.get("/metrics").text
    assert _metric_value(text, "rkllm_tokens_total") == 2.0


def test_nonstream_tokens_total_also_filtered():
    supervisor = FakeSupervisor(perf={"prefill_tokens": 5,
                                      "generate_tokens": 2})
    client = _client(supervisor=supervisor)
    assert client.post("/v1/chat/completions", json=MSG).status_code == 200
    assert _metric_value(client.get("/metrics").text,
                         "rkllm_tokens_total") == 2.0


# ---------------------------------------------------------------------------
# 2) queue-wait 与 inference 口径分离
# ---------------------------------------------------------------------------


def test_immediate_slot_records_zero_queue_wait():
    queue = InferenceQueue(max_wait=1)
    assert list(queue.run(lambda: iter(["x"]),
                          endpoint="chat.completions")) == ["x"]
    waited_sum, waited_count = _hist_totals(queue.queue_wait_hist)
    assert waited_count == 1
    assert waited_sum == pytest.approx(0.0)       # 立即拿槽 = 等待 0
    infer_sum, infer_count = _hist_totals(queue.infer_hist)
    assert infer_count == 1 and infer_sum >= 0.0
    assert queue.metrics["tokens_total"] == 1


def test_queue_wait_and_inference_are_separate():
    """一个请求占住执行槽，第二个请求排队：两段耗时进入不同直方图。"""
    queue = InferenceQueue(max_wait=4, wait_timeout_s=5)
    gate = threading.Event()
    first_started = threading.Event()

    def first_submit():
        first_started.set()
        yield "a"
        gate.wait(10)
        yield "b"

    def first():
        list(queue.run(first_submit, endpoint="chat.completions"))

    def second():
        list(queue.run(lambda: iter(["c"]), endpoint="chat.completions"))

    t1 = threading.Thread(target=first, daemon=True)
    t1.start()
    assert first_started.wait(5)
    t2 = threading.Thread(target=second, daemon=True)
    t2.start()
    time.sleep(0.2)
    gate.set()
    t1.join(10)
    t2.join(10)

    assert queue.metrics["tokens_total"] == 3
    waited_sum, waited_count = _hist_totals(queue.queue_wait_hist)
    assert waited_count == 2
    # 第一个立即拿槽（0），第二个等了 ~0.2s
    assert 0.1 <= waited_sum <= 5.0
    assert queue.metrics["queue_wait_seconds_sum"] == pytest.approx(waited_sum)

    infer_sum, infer_count = _hist_totals(queue.infer_hist)
    assert infer_count == 2
    # 第一个请求占槽 >= 0.2s，推理总时长应明显大于排队等待
    assert infer_sum > 0.15
    assert queue.metrics["infer_seconds_sum"] == pytest.approx(infer_sum)

    # 两个直方图都在 /metrics 可见（口径分开、各自可告警）
    text = render_text(queue.metrics_registry)
    assert "rkllm_queue_wait_histogram_seconds_count" in text
    assert "rkllm_inference_histogram_seconds_count" in text


def test_queue_wait_timeout_records_wait_time():
    """排队超时的请求也要把已等待时长记入 queue_wait（否则低估排队压力）。"""
    queue = InferenceQueue(max_wait=4, wait_timeout_s=0)
    queue._slot.acquire()
    try:
        with pytest.raises(QueueTimeout):
            list(queue.run(lambda: iter(["x"])))
    finally:
        queue._slot.release()
    waited_sum, waited_count = _hist_totals(queue.queue_wait_hist)
    assert waited_count == 1 and waited_sum > 0.0
    assert queue.metrics["rejected_total"] == 1


def test_metrics_exposes_inference_histogram_after_request():
    client = _client()
    assert client.post("/v1/chat/completions",
                       json={**MSG, "max_tokens": 16}).status_code == 200
    text = client.get("/metrics").text
    assert "rkllm_inference_histogram_seconds_bucket" in text
    assert "rkllm_queue_wait_histogram_seconds_bucket" in text
    assert INFERENCE_BUCKETS == tuple(sorted(INFERENCE_BUCKETS))


# ---------------------------------------------------------------------------
# 3) /health.perf 跨端点聚合
# ---------------------------------------------------------------------------


def test_health_perf_aggregates_completions_endpoint():
    """只调用 /v1/completions（无 chat 请求）时 perf 不能是 cold。"""
    client = _client(supervisor=FakeSupervisor(
        perf={"prefill_tokens": 5, "generate_tokens": 2}))
    assert client.post("/v1/completions",
                       json={"prompt": "hi", "max_tokens": 8}).status_code == 200
    perf = client.get("/health").json()["perf"]
    assert perf["source"] == "measured"
    assert perf["samples"] >= 1
    assert perf["ttft_s"] is not None
    assert perf["endpoints"] == 1


def test_health_perf_aggregates_multiple_endpoints():
    client = _client(supervisor=FakeSupervisor(
        perf={"prefill_tokens": 5, "generate_tokens": 2}))
    assert client.post("/v1/completions",
                       json={"prompt": "hi", "max_tokens": 8}).status_code == 200
    assert client.post("/v1/responses",
                       json={"input": "hi", "max_output_tokens": 8}
                       ).status_code == 200
    perf = client.get("/health").json()["perf"]
    assert perf["source"] == "measured"
    assert perf["samples"] == 2
    assert perf["endpoints"] == 2


# ---------------------------------------------------------------------------
# 4) chunked body：先鉴权再读体
# ---------------------------------------------------------------------------


def _asgi_call(app, path: str, *, headers: dict[str, str],
               chunks: list[bytes]) -> dict:
    """直接驱动 ASGI app，精确统计 body 被 receive 的次数。

    TestClient 会在客户端侧缓冲/发送 body，无法证明"服务端没读"；这里用
    裸 ASGI scope + 计数 receive，验证未授权请求在 body_limit 阶段就被拒。
    """
    pending = list(chunks)
    state = {"body_reads": 0, "messages": []}

    async def receive():
        state["body_reads"] += 1
        if pending:
            chunk = pending.pop(0)
            return {"type": "http.request", "body": chunk,
                    "more_body": bool(pending)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        state["messages"].append(message)

    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1"))
                   for key, value in headers.items()]
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode("latin-1"), "root_path": "",
        "query_string": b"", "headers": raw_headers,
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8080),
        "state": {},
    }
    asyncio.run(app(scope, receive, send))
    messages = state["messages"]
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages
                    if m["type"] == "http.response.body")
    resp_headers = {name.decode("latin-1").lower(): value.decode("latin-1")
                    for name, value in start["headers"]}
    return {"status": start["status"], "headers": resp_headers,
            "json": json.loads(body or b"{}"), "body_reads": state["body_reads"]}


def _chunked_app(env=None, supervisor=None):
    cfg = load_config(env=env or {})
    return create_app(cfg, supervisor=supervisor or FakeSupervisor())


def test_unauthorized_chunked_body_is_rejected_without_reading():
    app = _chunked_app(env={"RKLLM_SERVER_API_KEY": "s3cret"})
    result = _asgi_call(
        app, "/v1/chat/completions",
        headers={"content-type": "application/json"},
        chunks=[b'{"messages": [{"role": "user", "content": "hi"}]}'])
    assert result["status"] == 401
    assert result["json"]["error"]["type"] == "authentication_error"
    assert result["json"]["error"]["code"] == "invalid_api_key"
    assert result["headers"].get("www-authenticate") == "Bearer"
    assert result["headers"].get("x-rkllm-auth") == "enabled"
    # 关键：body 一次都没被读（未授权的大/慢 chunked body 不再进内存）
    assert result["body_reads"] == 0


def test_unauthorized_content_length_body_rejected_before_size_check():
    """鉴权先于体积判断：未授权 + 超大 Content-Length → 401（不是 413）。"""
    app = _chunked_app(env={"RKLLM_SERVER_API_KEY": "s3cret",
                            "RKLLM_SERVER_MAX_BODY_BYTES": "1024"})
    result = _asgi_call(
        app, "/v1/chat/completions",
        headers={"content-type": "application/json",
                 "content-length": "999999"},
        chunks=[])
    assert result["status"] == 401
    assert result["body_reads"] == 0


def test_authorized_chunked_body_still_enforces_size_limit():
    """授权后仍按 max_body_bytes 计数并 413；body 确实被读取。"""
    app = _chunked_app(env={"RKLLM_SERVER_API_KEY": "s3cret",
                            "RKLLM_SERVER_MAX_BODY_BYTES": "1024"})
    result = _asgi_call(
        app, "/v1/chat/completions",
        headers={"content-type": "application/json",
                 "authorization": "Bearer s3cret"},
        chunks=[b"x" * 800, b"x" * 800])
    assert result["status"] == 413
    assert result["json"]["error"]["code"] == "request_too_large"
    assert result["body_reads"] >= 1


def test_authorized_chunked_body_reaches_route():
    app = _chunked_app(env={"RKLLM_SERVER_API_KEY": "s3cret"})
    payload = json.dumps(MSG).encode()
    result = _asgi_call(
        app, "/v1/chat/completions",
        headers={"content-type": "application/json",
                 "authorization": "Bearer s3cret"},
        chunks=[payload])
    assert result["status"] == 200, result
    assert result["json"]["choices"][0]["message"]["content"] == "你好"


def test_chunked_without_auth_still_works():
    """未启用鉴权（默认回环）时 chunked 行为不变。

    用 /v1/reset（不经过推理管线）验证 body_limit 仍会读入 chunked body 并把
    请求正常交给路由——避免把 P2.4 测试耦合到并行的推理路由改动上。
    """
    app = _chunked_app()
    result = _asgi_call(app, "/v1/reset",
                        headers={"content-type": "application/json"},
                        chunks=[b"{}"])
    assert result["status"] == 200
    assert result["json"] == {"reset": True}
    assert result["body_reads"] >= 1


# ---------------------------------------------------------------------------
# 5) prompt cache 权限（prompt 明文只给属主）
# ---------------------------------------------------------------------------


def _write_runtime_cache(pc: PromptCache, data: bytes = b"kv") -> None:
    """模拟 runtime 以进程默认权限写出 KV 缓存文件。"""
    with open(pc.cache_path, "wb") as fh:
        fh.write(data)


@pytest.mark.skipif(os.name != "posix", reason="Windows 无 POSIX 权限位语义")
def test_prompt_cache_meta_and_stats_are_0600(tmp_path):
    pc = PromptCache(str(tmp_path / "cache"))
    pc.commit("这是用户的 prompt 明文")
    _write_runtime_cache(pc)
    pc.commit("这是用户的 prompt 明文")          # 提交后收紧 runtime 写出的 KV 文件
    for path in (pc.meta_path, pc.stats_path, pc.cache_path):
        assert os.path.isfile(path), path
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600, path
    assert stat.S_IMODE(os.stat(pc.dir).st_mode) == 0o700
    # 内容仍正确（权限收紧不影响功能）
    assert pc._read_meta() == "这是用户的 prompt 明文"
    # 原子写的临时文件不残留
    assert not os.path.exists(pc.meta_path + ".tmp")
    assert not os.path.exists(pc.stats_path + ".tmp")


@pytest.mark.skipif(os.name != "posix", reason="Windows 无 POSIX 权限位语义")
def test_prompt_cache_stats_written_by_prepare_are_0600(tmp_path):
    pc = PromptCache(str(tmp_path / "cache"))
    assert pc.prepare("第一轮内容") == (pc.cache_path, 1)   # miss → 写 stats
    assert stat.S_IMODE(os.stat(pc.stats_path).st_mode) == 0o600


def test_private_write_requests_0600_mode(monkeypatch, tmp_path):
    """跨平台验证写入路径确实请求 0600（Windows 无 POSIX 位，但 mode 参数可查）。"""
    import rkllm.worker.promptcache as cache_mod

    recorded: list[int] = []
    real_open = os.open

    def _spy(path, flags, mode=0o777):
        recorded.append(mode)
        return real_open(path, flags, mode)

    monkeypatch.setattr(cache_mod.os, "open", _spy)
    pc = PromptCache(str(tmp_path / "cache"))
    pc.commit("secret")
    pc.prepare("secret")          # miss → 再写一次 stats
    assert recorded, "缓存文件必须经 os.open 写入（才能显式指定权限）"
    assert all(mode == 0o600 for mode in recorded), recorded
    assert cache_mod._FILE_MODE == 0o600
    assert cache_mod._DIR_MODE == 0o700


def test_prompt_cache_still_functions_on_current_platform(tmp_path):
    """权限修复不改变缓存语义（非 POSIX 平台的主要回归）。"""
    pc = PromptCache(str(tmp_path / "cache"))
    pc.commit("前缀内容")
    _write_runtime_cache(pc)
    assert pc.prepare("前缀内容") == (pc.cache_path, 0)     # 命中
    stats = pc.stats()
    assert stats["hits"] == 1
    assert json.loads(Path(pc.stats_path).read_text(encoding="utf-8"))["hits"] == 1
