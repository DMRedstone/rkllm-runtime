"""LoRA adapter 回归：加载 / 选择 / 显性拒绝。

不碰真 NPU：引擎侧用桩替换 `rkllm_load_lora`，HTTP 侧用 FakeSupervisor。
"""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.abi import RKLLMError, _LoraAdapter
from rkllm.config import load_config
from rkllm.server.app import create_app
from rkllm.server.state import ServiceState
from rkllm.worker.engine import RealEngine


class _StubLib:
    """只实现 rkllm_load_lora；其余符号本用例不用。"""

    rkllm_optional_missing = ()

    def __init__(self, code=0):
        self.code = code
        self.calls = []

    def rkllm_load_lora(self, handle, adapter_ref):
        adapter = adapter_ref._obj if hasattr(adapter_ref, "_obj") else None
        self.calls.append(adapter)
        return self.code


def _bare_engine(lib) -> RealEngine:
    """绕开真实 __init__（那要模型与 NPU），只装 load_loras 需要的字段。"""
    engine = RealEngine.__new__(RealEngine)
    engine.lib = lib
    engine._handle = None
    engine._keep = []
    engine._lora_params = {}
    engine.verbose = False
    return engine


def test_load_loras_registers_by_stem(tmp_path):
    adapter = tmp_path / "domain.rkllm"
    adapter.write_bytes(b"fake")
    lib = _StubLib()
    engine = _bare_engine(lib)
    loaded = engine.load_loras([str(adapter)])
    assert list(loaded) == ["domain"]
    assert engine.loras == ("domain",)
    assert isinstance(lib.calls[0], _LoraAdapter)
    assert lib.calls[0].lora_adapter_name == b"domain"
    assert lib.calls[0].scale == 1.0
    # ctypes 保活：结构体必须留在 _keep 里，否则 C 侧拿到悬垂指针
    assert any(isinstance(obj, _LoraAdapter) for obj in engine._keep)


def test_load_loras_missing_file_is_explicit(tmp_path):
    engine = _bare_engine(_StubLib())
    with pytest.raises(RKLLMError, match="不存在"):
        engine.load_loras([str(tmp_path / "nope.rkllm")])


def test_load_loras_without_symbol_is_explicit(tmp_path):
    adapter = tmp_path / "d.rkllm"
    adapter.write_bytes(b"x")

    class _OldLib(_StubLib):
        rkllm_optional_missing = ("rkllm_load_lora",)

    engine = _bare_engine(_OldLib())
    with pytest.raises(RKLLMError, match="不导出 rkllm_load_lora"):
        engine.load_loras([str(adapter)])


def test_load_loras_propagates_nonzero_code(tmp_path):
    adapter = tmp_path / "d.rkllm"
    adapter.write_bytes(b"x")
    engine = _bare_engine(_StubLib(code=7))
    with pytest.raises(RKLLMError, match="code 7"):
        engine.load_loras([str(adapter)])


def test_state_exposes_adapter_names():
    cfg = load_config(env={"RKLLM_MODEL": "/tmp/x.rkllm",
                           "RKLLM_MODEL_LORA_PATHS": "/tmp/a.rkllm,/tmp/b.rkllm"})
    svc = ServiceState(cfg)
    assert svc.loras == ("a", "b")


class _FakeSupervisor:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self):
        self.seen = []

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        self.seen.append(kwargs.get("lora"))
        yield "ok"

    def abort(self):
        pass


def _client(**env):
    cfg = load_config(env={"RKLLM_MODEL": "/tmp/x.rkllm", **env})
    sup = _FakeSupervisor()
    return TestClient(create_app(cfg, supervisor=sup)), sup


def test_unknown_lora_is_rejected_400():
    client, sup = _client(RKLLM_MODEL_LORA_PATHS="/tmp/domain.rkllm")
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}],
                          "lora": "nope"})
    assert r.status_code == 400
    assert "nope" in r.json()["error"]["message"]
    assert sup.seen == []            # 拒绝发生在推理之前


def test_known_lora_is_forwarded():
    client, sup = _client(RKLLM_MODEL_LORA_PATHS="/tmp/domain.rkllm")
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}],
                          "lora": "domain"})
    assert r.status_code == 200
    assert sup.seen == ["domain"]


def test_run_frame_carries_lora_to_worker():
    """参数必须真的落到 run 帧里——板端实测过"某一层没接住"的 AttributeError。"""
    from rkllm.server.supervisor import WorkerClient
    from rkllm.worker.frames import OP_END, read_frame, write_frame

    left, right = socket.socketpair()
    client = WorkerClient(left)
    rfile, wfile = right.makefile("rb"), right.makefile("wb")
    seen = {}

    def worker():
        frame = read_frame(rfile)
        seen.update(frame.get("params") or {})
        write_frame(wfile, {"op": "token", "id": frame["id"], "text": "x"})
        write_frame(wfile, {"op": OP_END, "id": frame["id"]})

    threading.Thread(target=worker, daemon=True).start()
    try:
        items = list(client.run("p", lora="domain", request_timeout_s=5))
        assert seen.get("lora") == "domain"
        assert "".join(t for t in items if isinstance(t, str)) == "x"
    finally:
        client.close()
        right.close()


# -- 2026-11-20 审计回归：RealEngine.__init__ 必须真正设置 verbose ------------

class _FullStubLib(_StubLib):
    """完整路径替身：rkllm_init 返回 0，其余沿用 _StubLib。"""

    def rkllm_init(self, handle_ref, param_ref, callback_ref):
        return 0


def test_real_engine_init_sets_verbose_and_loads_lora(tmp_path, monkeypatch):
    """配置 LoRA 时 __init__ 不得因 self.verbose 未定义而崩（真实 bug：
    实例属性从未赋值，测试替身手工补齐才通过）。此处用库替身走完整
    __init__ → load_loras 路径。"""
    from rkllm.worker import engine as eng

    stub = _FullStubLib()
    monkeypatch.setattr(eng, "load_library", lambda _p: (stub, "stub.so", {}))
    monkeypatch.setattr(eng, "bind_library", lambda _lib: None)
    monkeypatch.setattr("rkllm.platform.soc.resolve_platform",
                        lambda _p: "rk3588")

    adapter = tmp_path / "domain.rkllm"
    adapter.write_bytes(b"fake")
    model = tmp_path / "m.rkllm"
    model.write_bytes(b"fake")

    real = eng.RealEngine(str(model), "auto", lora_paths=[str(adapter)])
    assert real.verbose is False
    assert real.loras == ("domain",)
    assert stub.calls, "rkllm_load_lora 应被调用一次"
