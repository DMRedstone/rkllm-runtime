"""P2.3 配置/启动/打包回归：真实 load_config / 真实 CLI 启动链。

覆盖待办 §2.3 的六项：

1. 显式 `RKLLM_TOML`（serve）时 `.d` 语义与文档一致：`RKLLM_TOML` 只换掉 TOML
   这一层，`.d` 部署层仍在其上生效；程序化 `load_config(file=...)` 默认仍保持
   "显式输入即隔离"的既有契约（test_p1_realism 也锁了这条），`use_dropins=True`
   才读系统 `.d`，`system=False` 完全离线装配。
2. list 类型环境变量同时接受 `,` 与 `:`：路径列表的 `:` 会切开，URL / Windows
   盘符里的 `:` 不会被误切（旧 `RKLLM_MODEL_DIR=/a:/b` 继续有效）。
3. `RKLLM_TOML` / `RKLLM_TOKENIZER_JSON` / `RKLLM_RUNTIME_VERSION` /
   `RKLLM_DEBUG` 明确登记为运行时环境变量（`RUNTIME_ENV_VARS`），`config docs`
   与打包 example 都会列出，且逐条核对真实读取点。
4. systemd `TimeoutStartSec` 覆盖 `engine.load_timeout_s` 的允许范围（旧单元没
   写，systemd 对 Type=notify 默认 90s 会先杀死 300s 的冷加载）。
5. 启动失败清理：`create_app` / `uvicorn.run` / supervisor `start()` 抛错时，
   已 spawn 的 worker 必须 shutdown、已加的锁频必须 restore，不留孤儿持 NPU。
6. `rkllm selftest` 的 HTTP 回环不读板端 `/etc/rkllm/rkllm.toml` / `.d`，
   板端坏配置不会把安装自检变成失败。

测试原则与 realism_helpers（task-12）一致：只重定向路径常量/替换外部边界
（uvicorn 的 socket、worker 子进程），`load_config`、CLI 装配、`cli.main` 的
参数解析与控制流都走真实实现。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "src"
REPO = PKG.parents[1]
sys.path.insert(0, str(SRC))

import realism_helpers as rh                                    # noqa: E402
from rkllm import cli, config as cfgmod                         # noqa: E402
from rkllm.cli_engine import _build_supervisor                  # noqa: E402
from rkllm.cli_serve import _cmd_serve                          # noqa: E402
from rkllm.config import (RUNTIME_ENV_VARS, SCHEMA,             # noqa: E402
                          config_docs, load_config)

UNIT = PKG / "systemd" / "rkllm-server.service"
EXAMPLE = PKG / "default" / "rkllm-server-example"
DEFAULT_TOML = PKG / "default" / "rkllm-server"


# ---------------------------------------------------------------------------
# 夹具：外部边界替身（只替 NPU worker / uvicorn socket / 锁频 sysfs）
# ---------------------------------------------------------------------------


class _FakeSup:
    """真实 create_app/AppState 需要的最小组件，不加载 NPU。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self):
        self.shut = 0

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "好"

    def abort(self):
        pass

    def shutdown(self):
        self.shut += 1


class _FakeLock:
    def __init__(self):
        self.restored = 0

    def restore(self):
        self.restored += 1


def _model(tmp_path: Path, name: str = "qwen3-4b.rkllm") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\0" * 4096)
    return path


def _isolate_paths(monkeypatch, tmp_path, *, toml_path=None, dropin_dir=None):
    """只重定向落点常量；load_config 本体保持真实（同 realism_helpers）。"""
    rh.install_config_paths(
        monkeypatch,
        toml_path=toml_path or (tmp_path / "absent-rkllm.toml"),
        dropin_dir=dropin_dir or (tmp_path / "absent-rkllm-server.d"))


def _clean_process_env(monkeypatch, **overrides):
    """_cmd_serve/_cmd_config 读 os.environ：清掉宿主 RKLLM_* 再显式注入。"""
    for name in list(os.environ):
        if name.upper().startswith("RKLLM_"):
            monkeypatch.delenv(name, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, str(value))


def _capture_serve_run(monkeypatch, *, sup=None, lock=None):
    """让真实 `cli.main(["serve",...])` 走到 uvicorn.run 为止并返回捕获结果。"""
    import uvicorn

    captured: dict = {}

    def fake_build(cfg, *, use_fake):
        captured["cfg"] = cfg
        captured["use_fake"] = use_fake
        return sup if sup is not None else _FakeSup()

    monkeypatch.setattr("rkllm.cli_serve._build_supervisor", fake_build)
    import rkllm.platform as platform_mod
    monkeypatch.setattr(platform_mod, "apply_freq_lock",
                        lambda *a, **k: lock if lock is not None else _FakeLock())
    monkeypatch.setattr(uvicorn, "run",
                        lambda app, **kwargs: captured.update(
                            app=app, kwargs=kwargs))
    return captured


# ===========================================================================
# 1) 显式 RKLLM_TOML / file= 与 `.d` 的语义
# ===========================================================================


def test_serve_explicit_rkllm_toml_still_gets_dropin_overrides(
        monkeypatch, tmp_path, capsys):
    """文档优先级 TOML < .d：RKLLM_TOML 换掉 TOML 后 `.d` 仍要覆盖它。"""
    model = _model(tmp_path)
    toml = rh.write_toml(tmp_path, {
        "server": {"bind_host": "127.0.0.1", "port": 9001},
        "model": {"path": str(model)},
    })
    dropin = rh.write_dropins(tmp_path / "rkllm-server.d", {
        "20-port.conf": "RKLLM_SERVER_PORT=9002\n",
    })
    _isolate_paths(monkeypatch, tmp_path, dropin_dir=dropin)
    _clean_process_env(monkeypatch, RKLLM_TOML=str(toml),
                       RKLLM_ENGINE_MEMORY_CHECK="false")
    captured = _capture_serve_run(monkeypatch)

    code = cli.main(["serve", str(model), "--fake-engine"])
    assert code == 0, capsys.readouterr().err[-800:]
    cfg = captured["cfg"]
    assert cfg.server.port == 9002, "显式 RKLLM_TOML 时 `.d` 没生效"
    assert cfg.source_of("server.port") == ".d"


def test_programmatic_explicit_file_keeps_isolated_contract(
        monkeypatch, tmp_path):
    """程序化 load_config(file=) 默认仍是"隔离输入"：不吃系统 `.d`。"""
    toml = rh.write_toml(tmp_path, {"server": {"port": 9001}})
    dropin = rh.write_dropins(tmp_path / "d", {
        "99-override.conf": "RKLLM_SERVER_PORT=9999\n",
    })
    _isolate_paths(monkeypatch, tmp_path, dropin_dir=dropin)

    cfg = load_config(file=str(toml), env=rh.clean_env())
    assert (cfg.server.port, cfg.source_of("server.port")) == (9001, "file")


def test_use_dropins_true_opts_a_file_caller_into_deployment_layer(
        monkeypatch, tmp_path):
    toml = rh.write_toml(tmp_path, {"server": {"port": 9001}})
    dropin = rh.write_dropins(tmp_path / "d", {
        "99-override.conf": "RKLLM_SERVER_PORT=9999\n",
    })
    _isolate_paths(monkeypatch, tmp_path, dropin_dir=dropin)

    cfg = load_config(file=str(toml), env=rh.clean_env(), use_dropins=True)
    assert (cfg.server.port, cfg.source_of("server.port")) == (9999, ".d")


def test_system_false_ignores_board_toml_and_dropins_but_keeps_explicit_file(
        monkeypatch, tmp_path):
    """selftest 用的离线装配：默认 TOML / .d 都不读，显式 file 仍生效。"""
    toml = rh.write_toml(tmp_path, {"server": {"port": 9001}})
    _isolate_paths(monkeypatch, tmp_path, toml_path=toml,
                   dropin_dir=rh.write_dropins(
                       tmp_path / "d", {"20.conf": "RKLLM_SERVER_PORT=9002\n"}))

    cfg = load_config(env=rh.clean_env(), system=False)
    assert (cfg.server.port, cfg.source_of("server.port")) == (8080, "default")

    cfg = load_config(file=str(toml), env=rh.clean_env(), system=False)
    assert (cfg.server.port, cfg.source_of("server.port")) == (9001, "file")


def test_config_show_uses_the_same_layers_as_serve(monkeypatch, tmp_path, capsys):
    """`rkllm config show` 必须打印 serve 真正会用的值与来源（含 RKLLM_TOML/.d）。"""
    toml = rh.write_toml(tmp_path, {"server": {"port": 9001}})
    dropin = rh.write_dropins(tmp_path / "d", {
        "20-port.conf": "RKLLM_SERVER_PORT=9002\n",
    })
    _isolate_paths(monkeypatch, tmp_path, dropin_dir=dropin)
    _clean_process_env(monkeypatch, RKLLM_TOML=str(toml))

    code = cli.main(["config", "show"])
    out = capsys.readouterr().out
    assert code == 0
    assert "9002" in out and "# .d" in out, out[-800:]


# ===========================================================================
# 2) list 类型环境变量的分隔符
# ===========================================================================


def test_list_env_accepts_comma_and_colon_for_paths():
    cfg = load_config(env=rh.clean_env(
        RKLLM_MODEL_SEARCH_DIRS="/a:/b,/c",
        RKLLM_MODEL_LORA_PATHS="/l1:/l2"), system=False)
    assert cfg.model.search_dirs == ("/a", "/b", "/c")
    assert cfg.model.lora_paths == ("/l1", "/l2")


def test_list_env_colon_does_not_split_urls():
    """cors_origins 是 URL 列表：'http://host:port' 里的 ':' 不能被当分隔符。"""
    cfg = load_config(env=rh.clean_env(
        RKLLM_SERVER_CORS_ORIGINS="http://a.example:3000,http://b.example:4000"),
        system=False)
    assert cfg.server.cors_origins == ("http://a.example:3000",
                                       "http://b.example:4000")


def test_list_env_windows_drive_path_is_not_split():
    """Windows 盘符路径（客户机侧配置）不能被 ':' 切成 "C" + "\\models"。"""
    cfg = load_config(env=rh.clean_env(
        RKLLM_MODEL_SEARCH_DIRS=r"C:\models"), system=False)
    assert cfg.model.search_dirs == (r"C:\models",)


def test_legacy_model_dir_colon_convention_still_works():
    """旧 conffile 的 RKLLM_MODEL_DIR=/a:/b 语义不能回退。"""
    cfg = load_config(env=rh.clean_env(
        RKLLM_MODEL_DIR="/var/a:/var/b"), system=False)
    assert cfg.model.search_dirs == ("/var/a", "/var/b")


# ===========================================================================
# 3) 运行时环境变量登记
# ===========================================================================

_RUNTIME_READERS = {
    # 变量 -> 真实读取点所在文件（交叉核对登记表与实现，不只看常量本身）
    "RKLLM_TOML": "cli_serve.py",
    "RKLLM_TOKENIZER_JSON": "tokens.py",
    "RKLLM_RUNTIME_VERSION": "__init__.py",
    "RKLLM_DEBUG": "cli.py",
}


def test_runtime_env_vars_registered_and_actually_read():
    assert set(RUNTIME_ENV_VARS) == set(_RUNTIME_READERS)
    for name, filename in _RUNTIME_READERS.items():
        candidates = list(SRC.rglob(filename))
        assert candidates, "找不到读取点文件 %s" % filename
        text = "\n".join(path.read_text(encoding="utf-8")
                         for path in candidates)
        assert name in text, "%s 已登记但代码里没人读" % name
        # 不重复进 84 键表 / 旧名映射：运行时变量有独立的读取语义。
        assert name not in {key.env for key in SCHEMA}
        assert name not in cfgmod.LEGACY_ENV_MAP


def test_config_docs_lists_runtime_env_vars():
    docs = config_docs()
    for name in RUNTIME_ENV_VARS:
        assert "| `%s` |" % name in docs, "%s 没进 config docs" % name


def test_shipped_example_and_conffile_document_runtime_env_vars():
    example = EXAMPLE.read_text(encoding="utf-8")
    conffile = DEFAULT_TOML.read_text(encoding="utf-8")
    for name in RUNTIME_ENV_VARS:
        assert name in example, "example 没登记 %s" % name
    assert "RKLLM_TOML" in conffile     # 优先级说明里点名显式 TOML 路径


# ===========================================================================
# 4) systemd TimeoutStartSec 与 engine.load_timeout_s 对齐
# ===========================================================================


def test_unit_timeout_start_covers_load_timeout_range():
    text = UNIT.read_text(encoding="utf-8")
    match = re.search(r"^TimeoutStartSec=(\d+)\s*$", text, re.M)
    assert match, "unit 没设 TimeoutStartSec（systemd 默认 90s 会杀 300s 加载）"
    timeout = int(match.group(1))
    load_key = next(key for key in SCHEMA if key.full == "engine.load_timeout_s")
    assert load_key.default == 300
    assert load_key.max_v == 900
    assert timeout >= load_key.max_v, (
        "TimeoutStartSec=%d 小于 engine.load_timeout_s 上限 %d"
        % (timeout, load_key.max_v))
    # 解释性注释要留在单元里，避免以后又被人删成默认 90s
    assert "load_timeout_s" in text


def test_example_documents_timeout_alignment():
    example = EXAMPLE.read_text(encoding="utf-8")
    assert "TimeoutStartSec" in example
    assert "load_timeout_s" in example


# ===========================================================================
# 5) 启动失败清理：worker / 锁频不得泄漏
# ===========================================================================


def _serve_args(model: Path, extra=()):
    return cli.build_parser().parse_args(["serve", str(model), *extra])


def _prepare_serve_call(monkeypatch, tmp_path, *, sup, lock):
    """隔离默认 TOML/.d、清环境，并替换 worker/锁频/uvicorn 外部边界。"""
    model = _model(tmp_path)
    _isolate_paths(monkeypatch, tmp_path)
    _clean_process_env(monkeypatch, RKLLM_ENGINE_MEMORY_CHECK="false")
    _capture_serve_run(monkeypatch, sup=sup, lock=lock)
    return model


def test_create_app_failure_releases_worker_and_freq_lock(
        monkeypatch, tmp_path):
    """旧实现 finally 只包住 uvicorn.run：create_app 抛错会漏 worker/锁频。"""
    sup, lock = _FakeSup(), _FakeLock()
    model = _prepare_serve_call(monkeypatch, tmp_path, sup=sup, lock=lock)

    def boom(*args, **kwargs):
        raise RuntimeError("create_app failed")

    monkeypatch.setattr("rkllm.server.app.create_app", boom)
    with pytest.raises(RuntimeError):
        _cmd_serve(_serve_args(model, ["--fake-engine"]))
    assert sup.shut == 1, "worker 没有 shutdown"
    assert lock.restored == 1, "锁频没有 restore"


def test_uvicorn_failure_releases_worker_and_freq_lock(monkeypatch, tmp_path):
    import uvicorn

    sup, lock = _FakeSup(), _FakeLock()
    model = _prepare_serve_call(monkeypatch, tmp_path, sup=sup, lock=lock)

    def boom(*args, **kwargs):
        raise RuntimeError("listen failed")

    monkeypatch.setattr(uvicorn, "run", boom)
    with pytest.raises(RuntimeError):
        _cmd_serve(_serve_args(model, ["--fake-engine"]))
    assert sup.shut == 1
    assert lock.restored == 1


def test_successful_run_still_cleans_up(monkeypatch, tmp_path):
    """恢复正常路径也要 finally 收尾（防止把清理逻辑只修在异常分支）。"""
    sup, lock = _FakeSup(), _FakeLock()
    model = _prepare_serve_call(monkeypatch, tmp_path, sup=sup, lock=lock)

    code = cli.main(["serve", str(model), "--fake-engine"])
    assert code == 0
    assert sup.shut == 1 and lock.restored == 1


def _clean_cfg(monkeypatch, tmp_path, **env):
    _isolate_paths(monkeypatch, tmp_path)
    return load_config(env=rh.clean_env(
        RKLLM_MODEL=str(_model(tmp_path)), **env))


def test_build_supervisor_shuts_down_when_start_fails(monkeypatch, tmp_path):
    """subprocess 模式：Supervisor.start() 失败（worker 未就绪）必须杀子进程。"""
    import rkllm.server.supervisor as supmod

    cfg = _clean_cfg(monkeypatch, tmp_path)
    created: list = []

    class BoomSupervisor:
        def __init__(self, worker_args, **kwargs):
            self.worker_args = list(worker_args)
            self.shut = 0
            created.append(self)

        def start(self):
            raise RuntimeError("worker did not become ready")

        def shutdown(self):
            self.shut += 1

    monkeypatch.setattr(supmod, "Supervisor", BoomSupervisor)
    with pytest.raises(RuntimeError):
        _build_supervisor(cfg, use_fake=True)
    assert created and created[0].shut == 1, "start() 失败后没 shutdown"


def test_local_supervisor_shuts_down_when_start_fails(monkeypatch, tmp_path):
    """in_process 模式：LocalSupervisor.start() 失败同样要收尾。"""
    import rkllm.server.supervisor as supmod

    cfg = _clean_cfg(monkeypatch, tmp_path, RKLLM_ENGINE_MODE="in_process")
    created: list = []

    class BoomLocal:
        def __init__(self, engine, *, load_timeout_s=300):
            self.engine = engine
            self.shut = 0
            created.append(self)

        def start(self):
            raise RuntimeError("model load timeout")

        def shutdown(self):
            self.shut += 1

    monkeypatch.setattr(supmod, "LocalSupervisor", BoomLocal)
    with pytest.raises(RuntimeError):
        _build_supervisor(cfg, use_fake=True)
    assert created and created[0].shut == 1


# ===========================================================================
# 6) selftest 隔离板端 TOML / `.d`
# ===========================================================================


def test_selftest_http_ignores_broken_board_configuration(
        monkeypatch, tmp_path):
    """板端 TOML/.d 会拒绝启动时，安装自检仍必须能跑完 HTTP 回环。"""
    hostile_toml = tmp_path / "rkllm.toml"
    hostile_toml.write_text('[mcp]\ntransport = "http"\n', encoding="utf-8")
    hostile_dropin = tmp_path / "rkllm-server.d"
    hostile_dropin.mkdir()
    (hostile_dropin / "10-broken.conf").write_text(
        "RKLLM_SERVER_PORT=not-a-number\n", encoding="utf-8")
    monkeypatch.setattr(cfgmod, "DEFAULT_TOML_PATH", str(hostile_toml))
    monkeypatch.setattr(cfgmod, "DEFAULT_DROPIN_DIR", str(hostile_dropin))

    # 前提：如果真去读系统层，这份配置确实会让 load_config 失败。
    with pytest.raises(Exception):
        load_config(env={})

    from rkllm.cli_info import _selftest_http
    checks = _selftest_http()
    assert checks, "selftest 没有任何结论"
    failed = [(name, detail) for name, ok, detail in checks
              if not ok and name in ("http e2e (deps)", "http e2e (boot)",
                                     "http e2e (abort)")]
    assert not failed, "板端配置污染了 selftest：%r" % (failed,)
    assert ("http models", True) in [(name, ok) for name, ok, _ in checks]
