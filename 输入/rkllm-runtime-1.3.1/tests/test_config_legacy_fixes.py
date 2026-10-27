"""2026-10-27 审核 P1/P2 回归（运行时/平台/deb）。

覆盖：
1. conffile 旧变量（1.3.1-15 名）真正生效：RKLLM_CONTEXT / RKLLM_MAX_NEW_TOKENS /
   RKLLM_QUIET / RKLLM_MEMORY_CHECK / RKLLM_MEMORY_RESERVE_MB / RKLLM_MIN_CONTEXT
   都要映射到当前 schema 键（P1-3）。
2. rk3576 与 rk3588 一样使用大核掩码（P1-2）。
3. 27B 20-29B 区间的档位推导（P2-5）。
4. worker 心跳只在有进度（或 prefill 宽限期内）时续命；C 调用卡死时停止心跳，
   supervisor 的心跳超时逻辑因此能杀 worker（P1-1）。
5. deb 工具链：conffile↔代码逐键检查能发现死键；crosscheck_deb 默认用法有 ROOT。
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]        # 输入/rkllm-runtime-1.3.1
SRC = PKG / "src"
REPO = PKG.parents[1]
TOOLS = REPO / "工具" / "编译工具链"
sys.path.insert(0, str(SRC))

from rkllm.abi import BIG_CORE_PLATFORMS                     # noqa: E402
from rkllm.cli import build_parser                           # noqa: E402
from rkllm.cli_env import _legacy_cli_overrides              # noqa: E402
from rkllm.config import LEGACY_ENV_MAP, SCHEMA, load_config  # noqa: E402
from rkllm.platform.modelclass import _class_from_name, derive_class  # noqa: E402
from rkllm.worker.fake import FakeEngine                     # noqa: E402
from rkllm.worker.frames import (OP_HEARTBEAT, FrameError,   # noqa: E402
                                 read_frame)
from rkllm.worker.main import Worker                         # noqa: E402


def _env(**overrides) -> dict[str, str]:
    """干净环境：去掉宿主机可能残留的 RKLLM_*，再叠加用例变量。"""
    base = {k: v for k, v in os.environ.items() if not k.startswith("RKLLM_")}
    base.update(overrides)
    return base


def _serve_cfg(**overrides):
    """按 serve 的真实路径装配配置（含 argparse defer 与 cli 覆盖层）。"""
    args = build_parser().parse_args(["serve", "m.rkllm"])
    return load_config(cli=_legacy_cli_overrides(args), env=_env(**overrides),
                       require_model_path=False)


# ---------------------------------------------------------------------------
# 1) conffile 旧变量（P1-3）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name,raw,key,want", [
    ("RKLLM_CONTEXT", "8192", "model.context", 8192),
    ("RKLLM_MAX_NEW_TOKENS", "512", "model.max_tokens", 512),
    ("RKLLM_QUIET", "1", "logging.level", "warning"),
    ("RKLLM_MEMORY_CHECK", "false", "engine.memory_check", False),
    ("RKLLM_MEMORY_RESERVE_MB", "512", "engine.memory_reserve_mb", 512),
    ("RKLLM_MIN_CONTEXT", "512", "model.min_context", 512),
])
def test_conffile_legacy_key_is_live(env_name, raw, key, want):
    cfg = _serve_cfg(**{env_name: raw})
    section, name = key.split(".")
    got = getattr(getattr(cfg, section), name)
    assert got == want, "%s 未生效：%s=%r" % (env_name, key, got)
    assert cfg.source_of(key) in ("env", "cli")


def test_rkllm_context_8192_reaches_serve_and_wins_over_default():
    """验收口径：conffile 里写 RKLLM_CONTEXT=8192，serve 的 context 必须是 8192。"""
    cfg = _serve_cfg(RKLLM_CONTEXT="8192")
    assert cfg.model.context == 8192
    assert cfg.source_of("model.context") == "env"


def test_quiet_zero_maps_back_to_info():
    assert _serve_cfg(RKLLM_QUIET="0").logging.level == "info"
    assert _serve_cfg(RKLLM_QUIET="false").logging.level == "info"


def test_new_style_env_name_wins_over_legacy():
    cfg = _serve_cfg(RKLLM_CONTEXT="8192", RKLLM_MODEL_CONTEXT="2048")
    assert cfg.model.context == 2048
    assert cfg.source_of("model.context") == "env"


def test_legacy_map_targets_all_exist_in_schema():
    known = {key.full for key in SCHEMA}
    broken = sorted("%s -> %s" % (legacy, target)
                    for legacy, target in LEGACY_ENV_MAP.items()
                    if target not in known)
    assert broken == [], "LEGACY_ENV_MAP 映射到不存在的配置键：%s" % broken


# ---------------------------------------------------------------------------
# 2) rk3576 大核（P1-2）
# ---------------------------------------------------------------------------


def test_rk3576_uses_big_core_mask_like_official_demo():
    # 官方 flask_server.py: platform in ("rk3576", "rk3588") → CPU4-7
    assert "rk3588" in BIG_CORE_PLATFORMS
    assert "rk3576" in BIG_CORE_PLATFORMS
    # rk3562（4×A53）/ rv1126b（4×A7）没有大核，必须保持低掩码
    assert "rk3562" not in BIG_CORE_PLATFORMS
    assert "rv1126b" not in BIG_CORE_PLATFORMS


# ---------------------------------------------------------------------------
# 3) 27B 档位正则（P2-5）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,want", [
    ("qwen3-27b", "large"),
    ("27b.rkllm", "large"),
    ("Qwen3-27B-RKLLM-v1.rkllm", "large"),
    ("model-20b", "large"),
    ("qwen3-12b", "large"),
    ("qwen3-30b", "large"),
    ("qwen3-235b", "large"),
    ("qwen3-7b", "medium"),
    ("qwen3.5-4b", "small"),
    ("qwen2.5-0.5b", "small"),
])
def test_model_class_from_name(name, want):
    assert _class_from_name(name) == want


def test_derive_class_27b_by_name_not_size():
    """名称优先于体积：文件名带 27b 时即使体积未知也必须判 large。"""
    assert derive_class("/nonexistent/qwen3-27b.rkllm") == "large"
    assert derive_class("/nonexistent/qwen3-27b.rkllm", configured="auto") == "large"


# ---------------------------------------------------------------------------
# 4) worker 心跳：只在有进度/宽限期时续命（P1-1）
# ---------------------------------------------------------------------------


class _Recorder:
    """线程安全的 wfile 替身：收下 write() 的字节，稍后按帧解析。"""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()

    def write(self, data) -> None:
        with self._lock:
            self._buf.extend(data)

    def flush(self) -> None:
        pass

    def frames(self) -> list[dict]:
        with self._lock:
            blob = bytes(self._buf)
        out = []
        stream = io.BytesIO(blob)
        while True:
            try:
                frame = read_frame(stream)
            except FrameError:      # 记录到一半的尾帧，忽略
                break
            if not frame:
                break
            out.append(frame)
        return out


def _start_heartbeat(worker: Worker, rec: _Recorder) -> threading.Thread:
    thread = threading.Thread(target=worker._heartbeat_loop, args=(rec,),
                              daemon=True)
    thread.start()
    return thread


def _stop_heartbeat(worker: Worker, thread: threading.Thread) -> None:
    worker._conn_closed.set()
    thread.join(timeout=3)
    assert not thread.is_alive(), "心跳线程未退出"


def test_heartbeat_stops_when_inference_makes_no_progress():
    worker = Worker(FakeEngine(), heartbeat_interval_s=1.0, heartbeat_grace_s=0.0)
    worker.heartbeat_interval_s = 0.05          # 只影响测试速度
    worker._run_started = time.monotonic() - 100  # 宽限期早已过去
    worker._infer_active.set()
    rec = _Recorder()
    thread = _start_heartbeat(worker, rec)
    try:
        time.sleep(0.35)
        assert rec.frames() == [], "无进度且宽限期已过时不应再发心跳（假心跳）"
        worker._progress += 1                    # 重新有进度
        time.sleep(0.3)
        frames = rec.frames()
        assert frames, "有进度后心跳应恢复"
        assert all(f["op"] == OP_HEARTBEAT for f in frames)
        assert frames[-1]["progress"] == 1
    finally:
        _stop_heartbeat(worker, thread)


def test_heartbeat_continues_during_prefill_grace_window():
    """首 token 之前（prefill）没有进度，但宽限期内必须继续发心跳，
    否则 27B 的长 prefill 会被 supervisor 误杀。"""
    worker = Worker(FakeEngine(), heartbeat_interval_s=1.0, heartbeat_grace_s=10.0)
    worker.heartbeat_interval_s = 0.05
    worker._run_started = time.monotonic()
    worker._infer_active.set()
    rec = _Recorder()
    thread = _start_heartbeat(worker, rec)
    try:
        time.sleep(0.35)
        frames = rec.frames()
        assert frames, "prefill 宽限期内应持续发心跳"
        assert all(f["progress"] == 0 for f in frames)
        assert all(f["op"] == OP_HEARTBEAT for f in frames)
    finally:
        _stop_heartbeat(worker, thread)


def test_supervisor_kills_worker_when_inference_stalls():
    """端到端：推理卡住（fake 引擎 30 s 不出首 token）时，worker 停止心跳，
    supervisor 按 heartbeat_timeout_s 杀进程并计入指标（P1-1 回归）。"""
    from rkllm.server.supervisor import Supervisor

    sup = Supervisor(
        ["fake", "--model", "fake", "--context", "4096",
         "--delay", "30",                    # 首 token 前 30 s：模拟卡死
         "--heartbeat-interval", "1",
         "--heartbeat-grace-s", "1"],        # 宽限期 1 s，便于测试
        load_timeout_s=60, restart_backoff_s=1, restart_backoff_max_s=2,
        restart_max_per_hour=10, health_interval_s=1,
        heartbeat_timeout_s=2)
    sup.start()
    try:
        done = threading.Event()

        def consume() -> None:
            try:
                for _ in sup.execute("stall", max_new_tokens=4,
                                     request_timeout_s=60):
                    pass
            except Exception:      # 卡死被杀后应拿到 WorkerDown；不关心类型
                pass
            finally:
                done.set()

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        deadline = time.time() + 20
        while time.time() < deadline and \
                sup.metrics["worker_heartbeat_timeouts_total"] == 0:
            time.sleep(0.2)
        assert sup.metrics["worker_heartbeat_timeouts_total"] >= 1, (
            "推理无进展超过 heartbeat_timeout_s 后 supervisor 未杀 worker："
            "心跳看门狗仍是假心跳（P1-1 回归）")
        assert sup.state in ("error", "loading", "ready")
        assert done.wait(timeout=20), "卡死请求未被唤醒"
    finally:
        sup.shutdown()


def test_supervisor_keeps_worker_alive_when_tokens_keep_coming():
    """反向回归：有进度的推理不能因为「心跳不再是纯定时器」而被误杀。"""
    from rkllm.server.supervisor import Supervisor

    sup = Supervisor(
        ["fake", "--model", "fake", "--context", "4096",
         "--delay", "0.15",                  # 每 token 150 ms，持续有进度
         "--heartbeat-interval", "1",
         "--heartbeat-grace-s", "1"],
        load_timeout_s=60, restart_backoff_s=1, restart_backoff_max_s=2,
        restart_max_per_hour=10, health_interval_s=1,
        heartbeat_timeout_s=2)
    sup.start()
    try:
        parts = []
        for item in sup.execute("hello", max_new_tokens=40,
                                request_timeout_s=60):
            if not isinstance(item, dict):
                parts.append(item)
        text = "".join(parts)
        assert text.startswith("answer")
        assert sup.metrics["worker_heartbeat_timeouts_total"] == 0, (
            "有 token 进度的推理被心跳看门狗误杀（假阳性回归）")
        assert sup.state == "ready"
    finally:
        sup.shutdown()


# ---------------------------------------------------------------------------
# 5) deb 工具链（P2-6 / P2-7）
# ---------------------------------------------------------------------------


def _import_tool(name: str):
    if not (TOOLS / (name + ".py")).exists():
        pytest.skip("工具链目录不在仓库里：%s" % (TOOLS / (name + ".py")))
    sys.path.insert(0, str(TOOLS))
    module = __import__(name)
    return module


def test_conffile_env_coverage_finds_no_dead_or_undocumented_key():
    verify_deb = _import_tool("verify_deb")
    conffile = (PKG / "default" / "rkllm-server").read_text(encoding="utf-8")
    unit = (PKG / "systemd" / "rkllm-server.service").read_text(encoding="utf-8")
    config_text = (SRC / "rkllm" / "config.py").read_text(encoding="utf-8")
    cli_texts = [path.read_text(encoding="utf-8")
                 for path in (SRC / "rkllm").rglob("*.py")
                 if path.name != "config.py"]
    coverage = verify_deb.conffile_env_coverage(conffile, unit, config_text,
                                                cli_texts)
    assert coverage["dead"] == [], "conffile 里仍有代码不读的死键：%s" % (
        coverage["dead"][:8],)
    assert coverage["forward_missing"] == [], (
        "有 serve/unit 入口没写进 conffile：%s" % (coverage["forward_missing"][:8],))
    assert coverage["bad_targets"] == []
    assert coverage["uncovered"] == []
    assert "RKLLM_CONTEXT" in coverage["documented"]
    assert "RKLLM_CONTEXT" in coverage["consumed"]


def test_conffile_env_coverage_detects_dead_key():
    verify_deb = _import_tool("verify_deb")
    coverage = verify_deb.conffile_env_coverage(
        "RKLLM_TOTALLY_DEAD_KEY=1\n", "", "", [])
    assert coverage["dead"] == ["RKLLM_TOTALLY_DEAD_KEY"]


def test_crosscheck_deb_default_discovery_has_root():
    crosscheck = _import_tool("crosscheck_deb")
    # ROOT 与 make_deb.py 同口径：仓库的 工具/ 目录（dist/ 在它下面）。
    tool_root = TOOLS.parent
    assert Path(crosscheck.ROOT) == tool_root
    # 默认用法（无参数）就是扫 ROOT/dist；ROOT 未定义时会 NameError（P2-6）。
    assert (Path(crosscheck.ROOT) / "dist").name == "dist"
