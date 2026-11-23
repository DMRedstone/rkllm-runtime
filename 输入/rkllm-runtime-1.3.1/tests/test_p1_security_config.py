"""P1-S1 / P1-C1 / P1-C2 回归：**真实路径**（不 patch `load_config`）。

用户指出的教训：很多缺陷是测试不仿真导致的（例如 `load_cfg` 被跳过）。
所以本文件刻意用真实的 `load_config()` 读取临时目录里的真实 TOML / `.d`
文件，并真实调用 `_cmd_serve` / `_build_supervisor`（推理引擎可用 FakeEngine
或捕获参数），而不是只对中间函数做 mock。

覆盖：
- P1-S1：默认无 key 只允许回环；非回环 + 无 key 启动拒绝（load_config 抛
  ConfigError，_cmd_serve 返回 2）；api_key_file 缺失/不可读/为空 fail-closed；
  默认 conffile 绑回环；启动横幅与 /health 如实报告鉴权状态。
- P1-C1：打包的 `10-model.conf` 不再有任何生效赋值；TOML 的 context=8192、
  search_dirs / lora_paths / logging.file / json_hint 不被 `.d` 覆盖或清空；
  `.d` 非空值仍按部署级覆盖；env 空串的旧"显式清空"语义保留。
- P1-C2：8B 文件名 / 显式 14B、27B / 无名文件构造 `_build_supervisor` 时
  `--max-tokens` 恒为正数；in_process FakeEngine 能真正启动。
"""
from __future__ import annotations

import ipaddress
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from rkllm import cli, config  # noqa: E402
from rkllm.cli_serve import _cmd_serve, _startup_banner  # noqa: E402
from rkllm.config import ConfigError, config_show, load_config  # noqa: E402
from rkllm.server.app import create_app  # noqa: E402

REAL_DROPIN_DIR = PKG / "default" / "rkllm-server.d"
REAL_CONFFILE = PKG / "default" / "rkllm-server"


# ---------------------------------------------------------------------------
# 测试夹具 / 辅助
# ---------------------------------------------------------------------------


class _FakeSup:
    """/health 只需的最小组件（不加载任何 NPU 库）。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "好"

    def abort(self):
        pass

    def shutdown(self):
        pass


def _model(tmp_path: Path, name: str = "qwen3-4b.rkllm",
           size: int = 4096) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _write_toml(tmp_path: Path, text: str, name: str = "rkllm.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _isolate(monkeypatch, toml_path: Path, dropin_dir: Path | None = None) -> None:
    """把 DEFAULT_TOML_PATH / DEFAULT_DROPIN_DIR 指向临时真实文件。

    只改这两个路径常量（生产里也是这两个落点），**不 patch load_config**，
    因此装配、校验、来源标注全部走真实实现。
    """
    monkeypatch.setattr(config, "DEFAULT_TOML_PATH", str(toml_path))
    monkeypatch.setattr(
        config, "DEFAULT_DROPIN_DIR",
        str(dropin_dir) if dropin_dir is not None
        else str(toml_path.parent / "rkllm-server.d"))


def _clean_rkllm_env(monkeypatch) -> None:
    """_cmd_serve 读 os.environ；清掉外部 RKLLM_* 以避免测试互相污染。"""
    for name in list(os.environ):
        if name.startswith("RKLLM_"):
            monkeypatch.delenv(name, raising=False)


def _serve_args(model: Path, extra=()):
    return cli.build_parser().parse_args(["serve", str(model), *extra])


def _patch_runtime(monkeypatch):
    """让 _cmd_serve 走到横幅为止：不真起 uvicorn，由捕获函数记录 cfg。"""
    captured: dict = {}

    def fake_build(cfg, *, use_fake):
        captured["cfg"] = cfg
        captured["use_fake"] = use_fake
        return _FakeSup()

    monkeypatch.setattr("rkllm.cli_serve._build_supervisor", fake_build)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: captured.setdefault(
        "uvicorn_kwargs", k))
    return captured


# ---------------------------------------------------------------------------
# P1-S1：安全默认 / 鉴权 fail-closed
# ---------------------------------------------------------------------------


def test_default_toml_is_loopback_and_health_reports_auth_disabled(
        monkeypatch, tmp_path):
    """默认（无 key、无 .d）允许启动，但只绑回环；/health 如实显示未鉴权。"""
    model = _model(tmp_path)
    toml = _write_toml(tmp_path,
                       '[model]\npath = "%s"\n' % model.as_posix())
    _isolate(monkeypatch, toml)

    cfg = load_config(env={}, require_model_path=True)
    assert cfg.server.bind_host == "127.0.0.1"
    assert cfg.server.api_key == "" and cfg.server.api_key_file == ""

    client = TestClient(create_app(cfg, supervisor=_FakeSup()))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["auth"] == {
        "enabled": False, "source": "none", "bind_host": "127.0.0.1",
        "loopback_only": True}
    # 响应头同源（未鉴权时也能直接看出真实状态）
    assert resp.headers["X-RKLLM-Auth"] == "disabled"
    assert resp.headers["X-RKLLM-Auth-Source"] == "none"


def test_non_loopback_without_key_is_refused(monkeypatch, tmp_path, capsys):
    """非回环 + 无 key：load_config 拒绝，_cmd_serve 返回 2（不再是 warning）。"""
    model = _model(tmp_path)
    toml = _write_toml(tmp_path,
                       '[server]\nbind_host = "0.0.0.0"\n'
                       '[model]\npath = "%s"\n' % model.as_posix())
    _isolate(monkeypatch, toml)

    with pytest.raises(ConfigError) as excinfo:
        load_config(env={}, require_model_path=True)
    assert "bind_host" in str(excinfo.value)
    assert "fail-closed" in str(excinfo.value)

    # 真实 serve 入口：配置错误在 spawn worker/uvicorn 之前返回 2
    _clean_rkllm_env(monkeypatch)
    monkeypatch.setenv("RKLLM_TOML", str(toml))
    code = _cmd_serve(_serve_args(model, ["--host", "0.0.0.0"]))
    err = capsys.readouterr().err
    assert code == 2
    assert "fail-closed" in err


def test_explicit_api_key_allows_non_loopback(monkeypatch, tmp_path):
    """同样的非回环绑定，显式配置 key 后允许启动。"""
    model = _model(tmp_path)
    toml = _write_toml(tmp_path,
                       '[server]\nbind_host = "0.0.0.0"\napi_key = "s3cret"\n'
                       '[model]\npath = "%s"\n' % model.as_posix())
    _isolate(monkeypatch, toml)
    cfg = load_config(env={}, require_model_path=True)
    assert cfg.server.bind_host == "0.0.0.0"
    client = TestClient(create_app(cfg, supervisor=_FakeSup()))
    assert client.get("/health").status_code == 401
    ok = client.get("/health", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert ok.json()["auth"]["enabled"] is True
    assert ok.json()["auth"]["source"] == "api_key"


def test_missing_api_key_file_fails_closed(monkeypatch, tmp_path, capsys):
    """api_key_file 不存在：拒绝启动（即使 server.api_key 也有值，不猜意图）。"""
    model = _model(tmp_path)
    missing = tmp_path / "no-such-key"
    toml = _write_toml(tmp_path,
                       '[server]\napi_key = "backup"\napi_key_file = "%s"\n'
                       '[model]\npath = "%s"\n'
                       % (missing.as_posix(), model.as_posix()))
    _isolate(monkeypatch, toml)

    with pytest.raises(ConfigError) as excinfo:
        load_config(env={}, require_model_path=True)
    assert "api_key_file" in str(excinfo.value)

    _clean_rkllm_env(monkeypatch)
    monkeypatch.setenv("RKLLM_TOML", str(toml))
    assert _cmd_serve(_serve_args(model)) == 2
    assert "api_key_file" in capsys.readouterr().err


def test_api_key_file_that_is_a_directory_fails_closed(monkeypatch, tmp_path):
    """不可读的一种真实形态：路径存在但不是普通文件（目录）。"""
    model = _model(tmp_path)
    keydir = tmp_path / "keys"
    keydir.mkdir()
    toml = _write_toml(tmp_path,
                       '[server]\napi_key_file = "%s"\n'
                       '[model]\npath = "%s"\n'
                       % (keydir.as_posix(), model.as_posix()))
    _isolate(monkeypatch, toml)
    with pytest.raises(ConfigError) as excinfo:
        load_config(env={}, require_model_path=True)
    assert "不是普通文件" in str(excinfo.value)


def test_empty_api_key_file_fails_closed(monkeypatch, tmp_path):
    """空 key 文件同样 fail-closed（不得静默变成"无鉴权"）。"""
    model = _model(tmp_path)
    key_file = tmp_path / "api_key"
    key_file.write_text("   \n", encoding="utf-8")
    toml = _write_toml(tmp_path,
                       '[server]\napi_key_file = "%s"\n'
                       '[model]\npath = "%s"\n'
                       % (key_file.as_posix(), model.as_posix()))
    _isolate(monkeypatch, toml)
    with pytest.raises(ConfigError) as excinfo:
        load_config(env={}, require_model_path=True)
    assert "文件为空" in str(excinfo.value)


def test_key_file_deleted_after_load_config_refuses_to_build_app(
        monkeypatch, tmp_path):
    """load_config 之后 key 文件被删：create_app 仍 fail-closed（应用层兜底）。"""
    model = _model(tmp_path)
    key_file = tmp_path / "api_key"
    key_file.write_text("s3cret\n", encoding="utf-8")
    toml = _write_toml(tmp_path,
                       '[server]\napi_key_file = "%s"\n'
                       '[model]\npath = "%s"\n'
                       % (key_file.as_posix(), model.as_posix()))
    _isolate(monkeypatch, toml)
    cfg = load_config(env={}, require_model_path=True)   # 此时文件存在
    key_file.unlink()
    with pytest.raises(ConfigError):
        create_app(cfg, supervisor=_FakeSup())


def test_api_key_file_enables_auth_and_banner_reports_true_state(
        monkeypatch, tmp_path):
    """key 文件可用：/health 401→200、body/头/banner 都显示 enabled + 来源。"""
    model = _model(tmp_path)
    key_file = tmp_path / "api_key"
    key_file.write_text("from-file\n", encoding="utf-8")
    try:
        key_file.chmod(0o600)
    except OSError:                                   # Windows 上 chmod 语义有限
        pass
    toml = _write_toml(tmp_path,
                       '[server]\napi_key_file = "%s"\n'
                       '[model]\npath = "%s"\n'
                       % (key_file.as_posix(), model.as_posix()))
    _isolate(monkeypatch, toml)
    cfg = load_config(env={}, require_model_path=True)

    client = TestClient(create_app(cfg, supervisor=_FakeSup()))
    unauth = client.get("/health")
    assert unauth.status_code == 401
    assert unauth.headers["X-RKLLM-Auth"] == "enabled"
    assert "Bearer" in unauth.headers["WWW-Authenticate"]

    ok = client.get("/health", headers={"Authorization": "Bearer from-file"})
    assert ok.status_code == 200
    assert ok.json()["auth"]["enabled"] is True
    assert ok.json()["auth"]["source"] == "api_key_file"

    banner = _startup_banner(cfg, "qwen3-4b", "4b")
    assert "api key : enabled（来源 api_key_file）" in banner
    assert "127.0.0.1" in banner


def test_banner_does_not_claim_key_when_none_configured():
    """没有 key 时横幅必须写 disabled，而不是旧实现的 'none'/'set' 混淆。"""
    cfg = load_config(env={})
    banner = _startup_banner(cfg, "rkllm", "4b")
    assert "api key : disabled" in banner


def test_packaged_conffile_binds_loopback_without_key():
    """打包的 /etc/default/rkllm-server：默认绑回环、未内置 API key。"""
    parsed: dict[str, str] = {}
    for line in REAL_CONFFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        parsed[key.strip()] = value.strip()
    assert parsed.get("RKLLM_HOST") in ("127.0.0.1", "::1", "localhost")
    assert ipaddress.ip_address(parsed["RKLLM_HOST"]).is_loopback
    assert not parsed.get("RKLLM_API_KEY")
    assert not parsed.get("RKLLM_SERVER_API_KEY")


def test_serve_with_shipped_conffile_binds_loopback(monkeypatch, tmp_path):
    """真实 serve 路径：systemd 传入的 --host 来自 conffile → 绑回环可启动。"""
    model = _model(tmp_path)
    toml = _write_toml(tmp_path,
                       '[model]\npath = "%s"\n' % model.as_posix())
    _isolate(monkeypatch, toml)
    _clean_rkllm_env(monkeypatch)
    monkeypatch.setenv("RKLLM_TOML", str(toml))
    captured = _patch_runtime(monkeypatch)

    code = _cmd_serve(_serve_args(model, ["--host", "127.0.0.1",
                                          "--fake-engine"]))
    assert code == 0
    assert captured["cfg"].server.bind_host == "127.0.0.1"
    assert captured["cfg"].server.api_key == ""


# ---------------------------------------------------------------------------
# P1-C1：`.d` 不得覆盖/清空 TOML
# ---------------------------------------------------------------------------


def test_packaged_dropin_has_no_active_assignments():
    """打包的 10-model.conf 不得有任何生效赋值（否则必然盖过 TOML）。"""
    active = [line for line in
              (REAL_DROPIN_DIR / "10-model.conf").read_text(
                  encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    assert active == [], "打包 drop-in 仍有生效赋值：%r" % (active[:5],)


def test_toml_values_survive_packaged_dropin(monkeypatch, tmp_path):
    """TOML 的 context/search_dirs/lora_paths/logging.file/json_hint 全部生效。"""
    model = _model(tmp_path)
    search = tmp_path / "models"
    search.mkdir()
    lora = tmp_path / "domain.rkllm"
    lora.write_bytes(b"lora")
    logfile = tmp_path / "server.log"
    toml = _write_toml(tmp_path, """
[model]
path = "%s"
context = 8192
min_context = 1024
search_dirs = ["%s", "/var/lib/rkllm/models"]
lora_paths = ["%s"]

[logging]
file = "%s"

[openai]
json_hint = "回答用 JSON"
""" % (model.as_posix(), search.as_posix(), lora.as_posix(),
       logfile.as_posix()))
    # 用真实打包 drop-in 目录，不传 file=（传 file= 时按设计跳过 .d）
    _isolate(monkeypatch, toml, dropin_dir=REAL_DROPIN_DIR)
    cfg = load_config(env={}, require_model_path=True)

    assert cfg.model.context == 8192
    assert cfg.sources["model.context"] == "file"
    assert cfg.model.search_dirs == (search.as_posix(), "/var/lib/rkllm/models")
    assert cfg.sources["model.search_dirs"] == "file"
    assert cfg.model.lora_paths == (lora.as_posix(),)
    assert cfg.sources["model.lora_paths"] == "file"
    assert cfg.logging.file == logfile.as_posix()
    assert cfg.sources["logging.file"] == "file"
    assert cfg.openai.json_hint == "回答用 JSON"
    assert cfg.sources["openai.json_hint"] == "file"

    shown = json.loads(config_show(cfg, as_json=True))
    for key in ("model.context", "model.search_dirs",
                "model.lora_paths", "logging.file", "openai.json_hint"):
        assert shown["sources"][key] == "file", key


def test_dropin_empty_values_do_not_clear_toml(monkeypatch, tmp_path):
    """`.d` 空串 = 未配置：不得清空 TOML 的 str/list 键（P1-C1 核心）。"""
    model = _model(tmp_path)
    search = tmp_path / "models"
    search.mkdir()
    dropin = tmp_path / "rkllm-server.d"
    dropin.mkdir()
    (dropin / "10-model.conf").write_text(
        "RKLLM_MODEL_SEARCH_DIRS=\n"
        "RKLLM_MODEL_LORA_PATHS=\n"
        "RKLLM_LOGGING_FILE=\n"
        "RKLLM_OPENAI_JSON_HINT=\n", encoding="utf-8")
    toml = _write_toml(tmp_path, """
[model]
path = "%s"
search_dirs = ["%s"]
lora_paths = ["%s"]

[logging]
file = "%s"

[openai]
json_hint = "keep me"
""" % (model.as_posix(), search.as_posix(),
       (tmp_path / "a.rkllm").as_posix(),
       (tmp_path / "s.log").as_posix()))
    _isolate(monkeypatch, toml, dropin_dir=dropin)
    cfg = load_config(env={}, require_model_path=True)

    assert cfg.model.search_dirs == (search.as_posix(),)
    assert cfg.model.lora_paths == ((tmp_path / "a.rkllm").as_posix(),)
    assert cfg.logging.file == (tmp_path / "s.log").as_posix()
    assert cfg.openai.json_hint == "keep me"
    for key in ("model.search_dirs", "model.lora_paths",
                "logging.file", "openai.json_hint"):
        assert cfg.sources[key] == "file", key


def test_dropin_nonempty_value_still_overrides_toml(monkeypatch, tmp_path):
    """`.d` 非空值仍是部署级覆盖（优先级：TOML < .d）。"""
    model = _model(tmp_path)
    dropin = tmp_path / "rkllm-server.d"
    dropin.mkdir()
    (dropin / "20-runtime.conf").write_text("RKLLM_MODEL_CONTEXT=2048\n",
                                            encoding="utf-8")
    toml = _write_toml(tmp_path,
                       '[model]\npath = "%s"\ncontext = 8192\n'
                       % model.as_posix())
    _isolate(monkeypatch, toml, dropin_dir=dropin)
    cfg = load_config(env={}, require_model_path=True)
    assert cfg.model.context == 2048
    assert cfg.sources["model.context"] == ".d"


def test_env_empty_still_clears_toml(monkeypatch, tmp_path):
    """env 空串的旧"显式清空"语义保留（旧 conffile RKLLM_SERVER_API_KEY= 依赖）。"""
    model = _model(tmp_path)
    toml = _write_toml(tmp_path,
                       '[model]\npath = "%s"\n'
                       '[logging]\nfile = "/tmp/x.log"\n' % model.as_posix())
    _isolate(monkeypatch, toml)
    cfg = load_config(env={"RKLLM_LOGGING_FILE": ""},
                      require_model_path=True)
    assert cfg.logging.file == ""
    assert cfg.sources["logging.file"] == "env"


# ---------------------------------------------------------------------------
# P1-C2：非 4B/7B 档位不得生成 --max-tokens 0
# ---------------------------------------------------------------------------


class _RecordingSupervisor:
    """捕获 worker 命令行的 Supervisor 替身：不 spawn 真实 worker。"""

    last = None

    def __init__(self, worker_args, **kwargs):
        self.worker_args = list(worker_args)
        self.kwargs = kwargs
        _RecordingSupervisor.last = self
        self.state = "ready"
        self.metrics = {"worker_restarts_total": 0}

    def start(self):
        pass

    def shutdown(self):
        pass

    def is_ready(self):
        return True


@pytest.fixture
def recorded_supervisor(monkeypatch):
    import rkllm.server.supervisor as supervisor_mod
    _RecordingSupervisor.last = None
    monkeypatch.setattr(supervisor_mod, "Supervisor", _RecordingSupervisor)
    return _RecordingSupervisor


def _config_for(monkeypatch, tmp_path, *, filename="model.rkllm",
                model_class=None, extra_sections=""):
    model = _model(tmp_path, filename)
    lines = ["[model]", 'path = "%s"' % model.as_posix()]
    if model_class:
        lines.append('class = "%s"' % model_class)
    toml = _write_toml(tmp_path, "\n".join(lines) + "\n" + extra_sections)
    _isolate(monkeypatch, toml)
    return load_config(env={}, require_model_path=True), model


def _max_tokens_from(worker_args) -> int:
    assert "--max-tokens" in worker_args, worker_args
    value = worker_args[worker_args.index("--max-tokens") + 1]
    return int(value)


@pytest.mark.parametrize("filename, model_class, expected", [
    ("qwen3-8b.rkllm", None, 512),      # 文件名 8B：无内置 → 保守默认
    ("qwen2.5-7b.rkllm", None, 384),    # 7B 内置值（防止误伤）
    ("model.rkllm", "14b", 512),        # 显式 14B：无内置 → 保守默认
    ("model.rkllm", "27b", 512),        # 显式 27B：无内置 → 保守默认
    ("model.rkllm", None, 512),         # 无名小文件 → 4B 内置值
])
def test_supervisor_never_passes_zero_max_tokens(
        monkeypatch, tmp_path, recorded_supervisor, filename, model_class,
        expected):
    cfg, _ = _config_for(monkeypatch, tmp_path, filename=filename,
                         model_class=model_class,
                         extra_sections="[engine]\nmemory_check = false\n")
    from rkllm.cli_engine import _build_supervisor
    _build_supervisor(cfg, use_fake=False)
    args = recorded_supervisor.last.worker_args
    assert _max_tokens_from(args) == expected
    assert _max_tokens_from(args) > 0


def test_large_class_derives_from_filename(monkeypatch, tmp_path):
    """8B 文件名被 derive_class 归到 8b（无内置档位），仍拿到正数默认。"""
    from rkllm.platform import derive_class
    cfg, model = _config_for(monkeypatch, tmp_path,
                             filename="qwen3-8b.rkllm")
    assert derive_class(cfg.model.path, "auto") == "8b"


def test_serve_adopts_conservative_max_tokens_for_27b(monkeypatch, tmp_path):
    """真实 serve 路径：显式 27B 未配 max_tokens 时 cfg 得到正数默认。"""
    model = _model(tmp_path, "model.rkllm")
    toml = _write_toml(tmp_path, (
        '[model]\npath = "%s"\nclass = "27b"\n'
        '[engine]\nmemory_check = false\n' % model.as_posix()))
    _isolate(monkeypatch, toml)
    _clean_rkllm_env(monkeypatch)
    monkeypatch.setenv("RKLLM_TOML", str(toml))
    captured = _patch_runtime(monkeypatch)

    code = _cmd_serve(_serve_args(model, ["--fake-engine"]))
    assert code == 0
    assert captured["cfg"].model.max_tokens > 0
    assert captured["cfg"].model.max_tokens == 512


def test_fake_engine_starts_for_27b_in_process(monkeypatch, tmp_path):
    """27B 档位 + in_process FakeEngine 至少能真正启动（P1-C2 验收项）。"""
    model = _model(tmp_path, "model.rkllm")
    toml = _write_toml(tmp_path, (
        '[model]\npath = "%s"\nclass = "27b"\n'
        '[engine]\nmode = "in_process"\nmemory_check = false\n'
        % model.as_posix()))
    _isolate(monkeypatch, toml)
    cfg = load_config(env={}, require_model_path=True)
    from rkllm.cli_engine import _build_supervisor
    sup = _build_supervisor(cfg, use_fake=True)
    try:
        assert sup.is_ready()
        assert sup.state == "ready"
    finally:
        sup.shutdown()
