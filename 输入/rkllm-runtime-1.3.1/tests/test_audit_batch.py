"""审计整改第一批的回归用例：鉴权、档位、内存预算、取消贯通、
batch 隔离、extract 校验、config init、未接线标注。

保持与既有测试一致的风格：Fake 替身，不依赖 NPU / 板端。
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import config_docs, config_init, config_show, load_config
from rkllm.mcp import tools as tools_mod
from rkllm.platform import (MemoryBudgetError, derive_class, kv_mb,
                            max_tokens_for, plan)
from rkllm.server.app import create_app


# ---------------------------------------------------------------------------
# A. 鉴权（N4）
# ---------------------------------------------------------------------------


class _FakeSup:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "好"

    def abort(self):
        pass


def _client(**env):
    cfg = load_config(env=env)
    return TestClient(create_app(cfg, supervisor=_FakeSup()))


def test_auth_required_when_key_set():
    c = _client(RKLLM_SERVER_API_KEY="secret-token")
    assert c.get("/health").status_code == 401
    assert c.get("/v1/models").status_code == 401
    assert c.get("/metrics").status_code == 401
    ok = c.get("/health", headers={"Authorization": "Bearer secret-token"})
    assert ok.status_code == 200
    bad = c.get("/health", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401


def test_auth_ready_is_exempt():
    c = _client(RKLLM_SERVER_API_KEY="secret-token")
    assert c.get("/ready").status_code == 200      # 编排探活豁免
    assert c.get("/health").status_code == 401     # 数据面仍受保护


def test_auth_disabled_when_key_empty():
    c = _client()                                   # 与 1.3.1-15 行为一致
    assert c.get("/health").status_code == 200


def test_auth_key_file_preferred(tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("from-file\n", encoding="utf-8")
    key_file.chmod(0o600)
    c = _client(RKLLM_SERVER_API_KEY="from-env",
                RKLLM_SERVER_API_KEY_FILE=str(key_file))
    assert c.get("/health", headers={"Authorization": "Bearer from-file"}
                 ).status_code == 200
    assert c.get("/health", headers={"Authorization": "Bearer from-env"}
                 ).status_code == 401


def test_body_limit_413():
    c = _client(RKLLM_SERVER_MAX_BODY_BYTES="1024")
    big = {"messages": [{"role": "user", "content": "x" * 4096}]}
    r = c.post("/v1/chat/completions", json=big)
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# C. 档位与内存预算（N10 / F12）
# ---------------------------------------------------------------------------


def test_derive_class_from_name():
    assert derive_class("/m/qwen3.5-4b.rkllm") == "small"
    assert derive_class("/m/qwen2.5-7b.rkllm") == "medium"
    assert derive_class("/m/qwen3-32b.rkllm") == "large"
    assert derive_class("/m/model.rkllm", configured="large") == "large"


def test_derive_class_from_size(tmp_path):
    f = tmp_path / "model.rkllm"
    f.write_bytes(b"\0" * (7 * 1024 * 1024))        # 7 MiB → small
    assert derive_class(str(f)) == derive_class(str(f))  # 稳定
    assert max_tokens_for("small") == 2048
    assert max_tokens_for("large") == 512


def test_kv_scales_with_context():
    assert kv_mb("small", 4096) > kv_mb("small", 2048)
    assert kv_mb("large", 4096) > kv_mb("small", 4096)


def test_budget_passes_when_enough(tmp_path):
    model = tmp_path / "m.rkllm"
    model.write_bytes(b"\0" * 1024)
    p = plan("small", 4096, model_path=str(model), reserve_mb=1024,
             available=64000)
    assert p.context == 4096 and not p.downgraded


def test_budget_downgrades_context(tmp_path):
    model = tmp_path / "m-7b.rkllm"
    model.write_bytes(b"\0" * (7 * 1024 * 1024))     # 7 GiB 量级按 MiB 计
    # 可用只够 2048：模型 7 MiB + KV + 预留 1024
    avail = 7 + kv_mb("medium", 2048) + 1024
    p = plan("medium", 4096, model_path=str(model), reserve_mb=1024,
             min_context=1024, available=avail)
    assert p.context == 2048 and p.downgraded


def test_budget_rejects_when_hopeless(tmp_path):
    model = tmp_path / "m-27b.rkllm"
    model.write_bytes(b"\0" * (15 * 1024 * 1024))
    with pytest.raises(MemoryBudgetError) as exc:
        plan("large", 4096, model_path=str(model), reserve_mb=1024,
             min_context=1024, available=100)     # 降到 min_context 也不够
    assert exc.value.required_mb > exc.value.available_mb


def test_health_reports_model_class():
    c = _client(RKLLM_MODEL_PATH="/m/qwen3.5-4b.rkllm")
    body = c.get("/health").json()
    assert body["model_class"] == "small"


# ---------------------------------------------------------------------------
# E. 契约诚实（P1-4 / P2-8）
# ---------------------------------------------------------------------------


def test_config_show_marks_unwired_key():
    """未接线键必须在 config show 里显式标出来（不静默）。

    2026-10-26：queue.max_concurrent 标注为未接线（单 NPU 恒为 1），是全表
    唯一的例外，其余键都必须生效。
    """
    from rkllm.config import SCHEMA

    assert [k.full for k in SCHEMA if not k.implemented] == ["queue.max_concurrent"]
    text = config_show(load_config(env={}))
    assert "queue.max_concurrent" in text
    assert text.count("[未接线]") == 1
    assert "enabled" in text


def test_config_docs_lists_all_cache_keys():
    docs = config_docs()
    assert docs.count("`cache.") == 5
    # 唯一允许出现的未接线标注（queue.max_concurrent）
    assert docs.count("（未接线）") == 1


def test_config_init_board_parses(tmp_path):
    text = config_init("board")
    data = tomllib.loads(text)
    assert data["model"]["path"].endswith(".rkllm")
    assert "engine" in data


def test_config_init_client_parses():
    data = tomllib.loads(config_init("client"))
    assert data["mcp"]["task_threshold_s"] == 45
    assert "target_endpoint" in data["mcp"]


def test_capabilities_responses_emulated():
    """2026-10-21：按 02 §4 实现翻译层后，responses 由 unsupported 升为 emulated。"""
    caps_body = _client().get("/v1/capabilities").json()
    assert caps_body["endpoints"]["responses"] == "emulated"


# ---------------------------------------------------------------------------
# B/F. 取消贯通、batch 隔离、extract 校验
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, replies=None, cancel_after=0):
        self.replies = list(replies or ["ok"])
        self.calls = 0
        self.cancel_after = cancel_after
        self.received_cancel_check = False

    def chat(self, messages, *, max_tokens=512, temperature=0.3,
             cancel_check=None, timeout_s=None):
        from rkllm.mcp.backend import BackendError
        self.calls += 1
        self.received_cancel_check = cancel_check is not None
        if cancel_check is not None and self.calls > self.cancel_after:
            if cancel_check():
                raise BackendError("任务已取消", kind="cancelled")
        text = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return {"text": text, "usage": None, "finish_reason": "stop"}

    def status(self):
        return {"state": "ready"}


_CFG = {"tool_prefix": "local_llm_", "default_max_tokens": 64,
        "default_temperature": 0.3, "max_input_chars": 1000,
        "max_result_chars": 4000, "batch_max_segments": 20}


def test_cancel_flag_reaches_backend():
    backend = _FakeBackend(replies=["ok"])
    handlers = tools_mod.build_handlers(backend, _CFG, None)
    handlers["ask"]({"prompt": "hi"}, should_cancel=lambda: False)
    assert backend.received_cancel_check is True


def test_cancelled_tool_reports_iserror():
    backend = _FakeBackend(replies=["ok"])
    handlers = tools_mod.build_handlers(backend, _CFG, None)
    with pytest.raises(tools_mod.Cancelled):
        handlers["ask"]({"prompt": "hi"}, should_cancel=lambda: True)


def test_batch_isolates_single_segment_failure():
    class Flaky(_FakeBackend):
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return {"text": "ok", "usage": None, "finish_reason": "stop"}

    handlers = tools_mod.build_handlers(Flaky(), _CFG, None)
    result = handlers["batch"]({"segments": ["a", "b", "c"]})
    payload = json.loads(result["content"][0]["text"])
    assert len(payload) == 3
    assert payload[0] == "ok" and "error" in payload[1] and payload[2] == "ok"


def test_extract_retries_until_schema_matches():
    backend = _FakeBackend(replies=['{"name": 1}', '{"name": "ok"}'])
    handlers = tools_mod.build_handlers(backend, _CFG, None)
    schema = {"type": "object", "required": ["name"],
              "properties": {"name": {"type": "string"}}}
    result = handlers["extract"]({"text": "x", "schema": schema})
    assert result["structuredContent"] == {"name": "ok"}
    assert backend.calls == 2


def test_extract_fails_after_retries():
    backend = _FakeBackend(replies=['{"name": 1}'])
    handlers = tools_mod.build_handlers(backend, _CFG, None, extract_retries=1)
    schema = {"type": "object",
              "properties": {"name": {"type": "string"}}}
    result = handlers["extract"]({"text": "x", "schema": schema})
    assert result["isError"] is True
    assert "schema" in result["content"][0]["text"]


def test_sampling_reaches_supervisor():
    """请求级采样参数必须真的下发（此前只进 usage 估算、从未传给 runtime）。"""
    class CapSup:
        state = "ready"
        metrics = {"worker_restarts_total": 0}

        def __init__(self):
            self.got = None

        def is_ready(self):
            return True

        def execute(self, prompt, **kwargs):
            self.got = kwargs.get("sampling")
            yield "好"

        def abort(self):
            pass

    cfg = load_config(env={})
    sup = CapSup()
    app = create_app(cfg, supervisor=sup)
    app.state.svc.default_sampling = {"temperature": 0.8, "top_k": 1}
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.2, "top_p": 0.5})
    assert r.status_code == 200
    assert sup.got["temperature"] == 0.2      # 请求体优先于进程默认
    assert sup.got["top_p"] == 0.5
    assert sup.got["top_k"] == 1              # 进程默认保留


def test_frame_carries_sampling():
    """worker 帧必须带 sampling 字段（IPC 侧不丢参数）。"""
    import io
    from rkllm.worker.frames import read_frame, write_frame

    buf = io.BytesIO()
    write_frame(buf, {"op": "run", "id": "r1", "prompt": "x",
                      "params": {"sampling": {"temperature": 0.3}}})
    buf.seek(0)
    frame = read_frame(buf)
    assert frame["params"]["sampling"] == {"temperature": 0.3}


def test_sampling_fills_missing_fields():
    """请求只给 temperature 时，其余字段必须取基线值而非 0。

    真机故障：旧实现新建 _SamplingParam 只填请求字段，top_k/top_p/
    repeat_penalty 落到 ctypes 默认 0 → 采样崩坏（词语重复 + </think> 残留）。
    """
    from rkllm.worker.engine import DEFAULT_SAMPLING, build_sampling

    got = build_sampling({"temperature": 0.3})
    assert got["temperature"] == 0.3
    assert got["top_k"] == DEFAULT_SAMPLING["top_k"] == 1
    assert got["top_p"] == DEFAULT_SAMPLING["top_p"] == 0.9
    assert got["repeat_penalty"] == DEFAULT_SAMPLING["repeat_penalty"] == 1.1
    assert got["mirostat"] == 0
    # 无采样参数时也返回完整基线
    assert build_sampling(None)["top_k"] == 1


def test_sampling_clamps_illegal_values():
    """0/越界值不得直接进结构体。"""
    from rkllm.worker.engine import build_sampling

    got = build_sampling({"top_k": 0, "top_p": 0.0, "repeat_penalty": 0.5,
                          "temperature": -1})
    assert got["top_k"] == 1
    assert got["top_p"] == 0.9
    assert got["repeat_penalty"] == 1.1
    assert got["temperature"] == 0.8
    # 合法的贪心解码保留
    assert build_sampling({"temperature": 0.0})["temperature"] == 0.0


def test_sampling_ignores_unknown_keys():
    from rkllm.worker.engine import build_sampling

    got = build_sampling({"temperature": 0.5, "bogus": 123, "top_k": None})
    assert "bogus" not in got
    assert got["temperature"] == 0.5
    assert got["top_k"] == 1          # None 不覆盖基线
