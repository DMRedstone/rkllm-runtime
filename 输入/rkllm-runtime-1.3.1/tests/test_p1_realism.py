"""真实路径集成测试（task-12）：不 mock ``load_config`` / ``create_app`` / ``_build_supervisor``。

背景（用户 2026-11-23 反馈）：很多缺陷因测试不仿真而被漏掉——
``load_cfg`` 被 monkeypatch 跳过、默认 conffile / ``.d`` / TOML 从未真正读入、
app 启动与鉴权只测 mock、CLI 参数构建被整体替换。本文件用真实文件与真实函数
补上这些路径：

1. **真实配置栈**：临时目录写真实 conffile（systemd EnvironmentFile 格式）、
   真实 ``rkllm.toml``、真实 ``rkllm-server.d/*.conf``，调用真实
   ``load_config()``（只重定向 ``DEFAULT_TOML_PATH`` / ``DEFAULT_DROPIN_DIR``
   两个路径常量），验证 default < file < .d < env < CLI 优先级链。
2. **P1-S1**：``server.api_key_file`` 缺失/空/不可读 fail-closed；非回环且无
   key 拒绝启动；回环无 key 仍可用；``/health`` 如实报告鉴权状态。
3. **P1-C1**：``.d`` 空串不得清空 TOML 的 str/list 键（env 空串仍保留"显式
   清空"语义）；打包 ``.d/10-model.conf`` 不得再有生效赋值。
4. **P1-C2**：真实 ``_build_supervisor`` 构造 8B/14B/27B/无名文件，断言
   ``--max-tokens`` 不是 0（保守默认 512）；FakeEngine 真实启动一次。
5. **真实 app/CLI**：``create_app(cfg)`` + ``TestClient``（cfg 来自真实
   TOML/conffile）；P1-O1 ``/v1/completions`` 流式 legacy 契约；真实
   ``cli.main(["serve", ...])`` 启动链（只把 uvicorn 的 socket 监听换成记录）。

预期失败：S1/C1/C2 的生产修复若尚未落地，对应断言用
``realism_helpers.expect_contract`` 记为 ``xfail``（原因写清）；修复落地后
自动转严格。CI 建议设 ``RKLLM_REALISM_STRICT=1`` 让 xfail 直接失败。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from realism_helpers import (  # noqa: E402
    SHIPPED_DROPIN_DIR,
    capture_supervisor,
    clean_env,
    load_real_config,
    make_app,
    parse_env_file,
    run_serve,
    shipped_conffile_env,
    shipped_dropin_env,
    worker_arg,
    expect_contract,
)

from rkllm.config import ConfigError  # noqa: E402
from rkllm.server.app import create_app  # noqa: E402

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
NON_LOOPBACK_HOSTS = ("0.0.0.0", "::", "192.168.10.20")


# ---------------------------------------------------------------------------
# 公共夹具 / 小工具
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_rkllm_env(monkeypatch):
    """每个用例都从干净环境开始（宿主机残留 RKLLM_* 不得影响结果）。"""
    for key in list(os.environ):
        if key.upper().startswith("RKLLM_"):
            monkeypatch.delenv(key, raising=False)


def _model_file(tmp_path: Path, name: str = "m.rkllm", size: int = 4096) -> Path:
    path = tmp_path / name
    path.write_bytes(b"\0" * size)
    return path


def _serve_payload():
    return {"messages": [{"role": "user", "content": "打个招呼"}]}


def _sse_frames(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


# ===========================================================================
# 1) 真实 load_config：conffile / TOML / .d 三层文件真的被读入
# ===========================================================================


def test_toml_only_real_load_sets_values_and_sources(tmp_path, monkeypatch):
    """真实 rkllm.toml 被读入；来源标注为 file，默认值不被污染。"""
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={
            "server": {"bind_host": "127.0.0.1", "port": 9099,
                       "api_key": "toml-key"},
            "model": {"path": "/models/qwen3.5-4b.rkllm", "context": 8192},
            "queue": {"max_wait": 7},
        },
        require_model_path=True)
    assert cfg.server.port == 9099
    assert cfg.server.api_key == "toml-key"
    assert cfg.model.context == 8192
    assert cfg.queue.max_wait == 7
    assert cfg.source_of("server.port") == "file"
    assert cfg.source_of("model.context") == "file"
    # 未在 TOML 出现的键仍取内置默认
    assert cfg.server.read_timeout_s == 30
    assert cfg.source_of("server.read_timeout_s") == "default"


def test_real_conffile_env_maps_legacy_names(tmp_path, monkeypatch):
    """真实 conffile（EnvironmentFile 格式）中的旧变量名全部生效。

    这是 systemd 部署的实际形态：``/etc/default/rkllm-server`` 的每一行都会
    变成 ``RKLLM_*`` 环境变量交给 ``load_config``。旧测试只手工传 env dict，
    没有经过"文件 → 解析 → 映射"这条链。
    """
    conffile = """\
# 注释行不得生效
RKLLM_MODEL=/var/lib/rkllm/models/qwen3.5-4b.rkllm
RKLLM_HOST=127.0.0.1
RKLLM_PORT=9098
RKLLM_CONTEXT=8192
RKLLM_MAX_NEW_TOKENS=321
RKLLM_QUIET=1
RKLLM_MEMORY_CHECK=false
RKLLM_MEMORY_RESERVE_MB=512
RKLLM_MIN_CONTEXT=512
RKLLM_SERVER_API_KEY=from-conffile

RKLLM_EXTRA_ARGS=--不存在的历史变量
"""
    cfg = load_real_config(monkeypatch, tmp_path, conffile_text=conffile,
                           require_model_path=True)
    assert cfg.model.path.endswith("qwen3.5-4b.rkllm")
    assert cfg.server.bind_host == "127.0.0.1"
    assert cfg.server.port == 9098
    # P1-3 的 6 个"conffile 死键"必须真正映射到 schema 键
    assert cfg.model.context == 8192
    assert cfg.model.max_tokens == 321
    assert cfg.logging.level == "warning"          # RKLLM_QUIET=1
    assert cfg.engine.memory_check is False
    assert cfg.engine.memory_reserve_mb == 512
    assert cfg.model.min_context == 512
    assert cfg.server.api_key == "from-conffile"
    for key in ("model.context", "model.max_tokens", "logging.level",
                "engine.memory_check", "engine.memory_reserve_mb",
                "model.min_context", "server.api_key"):
        assert cfg.source_of(key) == "env", key


def test_priority_chain_default_file_dropin_env_cli(tmp_path, monkeypatch):
    """优先级链逐层验证：default < file(TOML) < .d < env < CLI。"""
    toml = {"server": {"port": 9000}}
    dropins = {"10-port.conf": "RKLLM_SERVER_PORT=9001\n"}

    cfg = load_real_config(monkeypatch, tmp_path, toml_sections=toml)
    assert (cfg.server.port, cfg.source_of("server.port")) == (9000, "file")

    cfg = load_real_config(monkeypatch, tmp_path, toml_sections=toml,
                           dropins=dropins)
    assert (cfg.server.port, cfg.source_of("server.port")) == (9001, ".d")

    cfg = load_real_config(monkeypatch, tmp_path, toml_sections=toml,
                           dropins=dropins,
                           env={"RKLLM_SERVER_PORT": "9002"})
    assert (cfg.server.port, cfg.source_of("server.port")) == (9002, "env")

    cfg = load_real_config(monkeypatch, tmp_path, toml_sections=toml,
                           dropins=dropins,
                           env={"RKLLM_SERVER_PORT": "9002"},
                           cli={"server.port": 9003})
    assert (cfg.server.port, cfg.source_of("server.port")) == (9003, "cli")

    cfg = load_real_config(monkeypatch, tmp_path)   # 三层都没有
    assert (cfg.server.port, cfg.source_of("server.port")) == (8080, "default")


def test_dropin_layer_wins_over_toml_and_later_file_wins(tmp_path, monkeypatch):
    """``.d`` 目录真实读取：后文件名覆盖先文件，export/引号按 systemd 语义解析。"""
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"server": {"port": 9000, "bind_host": "127.0.0.1"}},
        dropins={
            "10-model.conf": "RKLLM_SERVER_PORT=9001\n"
                             "export RKLLM_LOGGING_FORMAT='json'\n",
            "20-later.conf": "RKLLM_SERVER_PORT=9002\n",
        })
    assert cfg.server.port == 9002
    assert cfg.source_of("server.port") == ".d"
    assert cfg.logging.format == "json"            # 引号/export 解析正确
    assert cfg.source_of("logging.format") == ".d"


def test_missing_toml_and_dropin_are_zero_change(tmp_path, monkeypatch):
    """默认 TOML/.d 不存在是正常状态：默认行为零变化，且不被宿主机 /etc 污染。"""
    cfg = load_real_config(monkeypatch, tmp_path)
    assert (cfg.server.port, cfg.source_of("server.port")) == (8080, "default")
    assert cfg.server.bind_host == "127.0.0.1"     # 默认只本机
    assert cfg.model.context == 4096


def test_empty_numeric_in_dropin_is_ignored_not_an_error(tmp_path, monkeypatch):
    """空串对数值键 = 未设置（旧 conffile 里 ``RKLLM_PORT=`` 不能炸启动）。"""
    cfg = load_real_config(monkeypatch, tmp_path,
                           dropins={"10-empty.conf": "RKLLM_SERVER_PORT=\n"
                                                     "RKLLM_MODEL_CONTEXT=\n"})
    assert cfg.server.port == 8080
    assert cfg.model.context == 4096
    assert cfg.source_of("server.port") == "default"


def test_explicit_file_arg_still_skips_dropin_dir(tmp_path, monkeypatch):
    """显式 ``file=`` 时不吃系统 ``.d``（排障/客户端隔离配置的既有契约）。"""
    from realism_helpers import write_dropins, write_toml
    from rkllm import config as cfgmod

    toml_path = write_toml(tmp_path, {"server": {"port": 9000}})
    dropin = write_dropins(tmp_path / "d",
                           {"99-override.conf": "RKLLM_SERVER_PORT=9999\n"})
    monkeypatch.setattr(cfgmod, "DEFAULT_DROPIN_DIR", str(dropin))
    cfg = cfgmod.load_config(file=str(toml_path), env=clean_env())
    assert cfg.server.port == 9000
    assert cfg.source_of("server.port") == "file"


# ===========================================================================
# 2) P1-C1：`.d` 空串不得清空 TOML
# ===========================================================================


def _c1_toml(model_path: Path) -> dict[str, dict]:
    return {
        "server": {"bind_host": "127.0.0.1"},
        "model": {"path": str(model_path), "context": 8192,
                  "search_dirs": ["/models/a", "/models/b"],
                  "lora_paths": ["/loras/one.rkllm"]},
        "logging": {"file": "/var/log/rkllm-server.log"},
        "openai": {"json_hint": "只输出 JSON"},
    }


def test_c1_dropin_empty_strings_do_not_clear_toml(tmp_path, monkeypatch):
    """临时 ``.d`` 里写空串（旧打包默认形态）不得清空 TOML 的 str/list 键。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections=_c1_toml(model),
        dropins={"10-model.conf": "\n".join([
            "RKLLM_MODEL_SEARCH_DIRS=",
            "RKLLM_MODEL_LORA_PATHS=",
            "RKLLM_LOGGING_FILE=",
            "RKLLM_OPENAI_JSON_HINT=",
            "RKLLM_MODEL_CONTEXT=",
        ]) + "\n"},
        require_model_path=True)

    expected = {
        "model.search_dirs": ("/models/a", "/models/b"),
        "model.lora_paths": ("/loras/one.rkllm",),
        "logging.file": "/var/log/rkllm-server.log",
        "openai.json_hint": "只输出 JSON",
        "model.context": 8192,
    }
    for key, want in expected.items():
        had = cfg.values[key]
        expect_contract(
            had == want and cfg.sources[key] == "file",
            "P1-C1: `.d` 空串清空了 TOML 的 %s（实际 %r，来源 %s）"
            % (key, had, cfg.sources[key]))


def test_c1_shipped_dropin_does_not_clear_toml(tmp_path, monkeypatch):
    """用仓库里真正会打进 deb 的 ``default/rkllm-server.d`` 验证同一契约。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(monkeypatch, tmp_path,
                           toml_sections=_c1_toml(model),
                           dropin_dir=SHIPPED_DROPIN_DIR,
                           require_model_path=True)

    expected = {
        "model.search_dirs": ("/models/a", "/models/b"),
        "model.lora_paths": ("/loras/one.rkllm",),
        "logging.file": "/var/log/rkllm-server.log",
        "openai.json_hint": "只输出 JSON",
        "model.context": 8192,
    }
    bad = {k: cfg.values[k] for k, want in expected.items()
           if cfg.values[k] != want or cfg.sources[k] != "file"}
    expect_contract(
        not bad,
        "P1-C1: 打包 `.d` 覆盖了 TOML（实际 %r）；应去掉/注释空串与固定默认" % bad)


def test_c1_shipped_default_dropin_has_no_active_assignment():
    """打包的 ``.d/*.conf`` 不得含生效赋值（否则默认部署必然覆盖 TOML）。"""
    active = shipped_dropin_env()
    expect_contract(
        active == {},
        "P1-C1: 打包 `.d` 仍有生效赋值 %r；应改成全注释示例" % active)


def test_c1_nonempty_dropin_still_overrides_toml(tmp_path, monkeypatch):
    """修 C1 不能把 ``.d`` 的覆盖能力修没：非空值仍覆盖 TOML。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections=_c1_toml(model),
        dropins={"10-model.conf": "RKLLM_MODEL_CONTEXT=2048\n"
                                  "RKLLM_LOGGING_FILE=/tmp/x.log\n"},
        require_model_path=True)
    assert (cfg.model.context, cfg.source_of("model.context")) == (2048, ".d")
    assert cfg.logging.file == "/tmp/x.log"
    assert cfg.source_of("logging.file") == ".d"


def test_c1_env_empty_still_clears_toml(tmp_path, monkeypatch):
    """env 层空串保留"显式清空"语义（旧 conffile 关 key 依赖它）。"""
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"server": {"api_key": "secret"}},
        conffile_text="RKLLM_SERVER_API_KEY=\n")
    assert cfg.server.api_key == ""
    assert cfg.source_of("server.api_key") == "env"


# ===========================================================================
# 3) P1-S1：默认安全姿态 / key_file fail-closed / 鉴权状态如实上报
# ===========================================================================


def test_s1_shipped_conffile_binds_loopback():
    """默认 conffile 不得再绑 0.0.0.0 且无 key（旧 P2-11/S1 缺陷）。"""
    env = shipped_conffile_env()
    bind = env.get("RKLLM_HOST", "")
    expect_contract(
        bind in LOOPBACK_HOSTS,
        "P1-S1: 默认 conffile RKLLM_HOST=%r 不是回环地址" % bind)


@pytest.mark.parametrize("host", NON_LOOPBACK_HOSTS)
def test_s1_non_loopback_without_key_is_refused(tmp_path, monkeypatch, host):
    """非回环 + 无 key：``load_config(require_model_path=True)`` 必须拒绝启动。"""
    model = _model_file(tmp_path)
    try:
        cfg = load_real_config(
            monkeypatch, tmp_path,
            toml_sections={"server": {"bind_host": host, "api_key": ""},
                           "model": {"path": str(model)}},
            require_model_path=True)
    except ConfigError as exc:
        assert "key" in str(exc).lower() or "鉴权" in str(exc) \
            or "bind" in str(exc).lower()
    else:
        expect_contract(
            False,
            "P1-S1: bind_host=%r 且无 key 时仍允许启动（bind_host=%r）"
            % (host, cfg.server.bind_host))


@pytest.mark.parametrize("host", LOOPBACK_HOSTS)
def test_s1_loopback_without_key_is_allowed(tmp_path, monkeypatch, host):
    """回环地址 + 无 key 是合法部署（保留 1.3.1-15 的升级兼容）。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"server": {"bind_host": host},
                       "model": {"path": str(model)}},
        require_model_path=True)
    assert cfg.server.bind_host == host
    assert cfg.server.api_key == ""


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_s1_key_file_unusable_is_fail_closed(tmp_path, monkeypatch, kind):
    """``server.api_key_file`` 配置了但缺失/空/不可读：必须 ConfigError。"""
    model = _model_file(tmp_path)
    key_path = tmp_path / "api_key"
    if kind == "empty":
        key_path.write_text("", encoding="utf-8")
    elif kind == "directory":
        key_path.mkdir()
    # missing: 不创建
    try:
        load_real_config(
            monkeypatch, tmp_path,
            toml_sections={
                "server": {"bind_host": "127.0.0.1",
                           "api_key_file": str(key_path)},
                "model": {"path": str(model)}},
            require_model_path=True)
    except ConfigError as exc:
        assert "api_key_file" in str(exc)
    else:
        expect_contract(
            False,
            "P1-S1: api_key_file=%s（%s）时仍静默以无鉴权启动"
            % (key_path, kind))


def test_s1_key_file_unusable_fails_even_if_api_key_also_set(tmp_path, monkeypatch):
    """key_file 优先于 api_key：文件不可用时不得悄悄回退到 api_key。"""
    model = _model_file(tmp_path)
    try:
        load_real_config(
            monkeypatch, tmp_path,
            toml_sections={
                "server": {"bind_host": "127.0.0.1",
                           "api_key": "fallback-key",
                           "api_key_file": str(tmp_path / "absent.key")},
                "model": {"path": str(model)}},
            require_model_path=True)
    except ConfigError as exc:
        assert "api_key_file" in str(exc)
    else:
        expect_contract(
            False, "P1-S1: key_file 缺失时静默回退到 api_key，未 fail-closed")


def test_s1_create_app_rechecks_key_file(tmp_path, monkeypatch):
    """load 之后 key 文件被删：``create_app`` 构造时也必须拒绝（strict 解析）。"""
    model = _model_file(tmp_path)
    key_file = tmp_path / "api_key"
    key_file.write_text("from-file\n", encoding="utf-8")
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={
            "server": {"bind_host": "127.0.0.1",
                       "api_key_file": str(key_file)},
            "model": {"path": str(model)}},
        require_model_path=True)
    key_file.unlink()
    try:
        create_app(cfg)
    except ConfigError:
        pass
    else:
        expect_contract(
            False, "P1-S1: key 文件在 load 后被删，create_app 仍以无鉴权构造成功")


def test_s1_real_key_file_enables_bearer(tmp_path, monkeypatch):
    """真实 key 文件（0600）经真实 create_app 生效：无 Bearer 401。"""
    model = _model_file(tmp_path)
    key_file = tmp_path / "api_key"
    key_file.write_text("from-file\n", encoding="utf-8")
    try:
        key_file.chmod(0o600)
    except OSError:
        pass
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={
            "server": {"bind_host": "127.0.0.1",
                       "api_key_file": str(key_file)},
            "model": {"path": str(model)}},
        require_model_path=True)
    app, _ = make_app(cfg)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health",
                          headers={"Authorization": "Bearer from-file"}
                          ).status_code == 200
        assert client.get("/health",
                          headers={"Authorization": "Bearer wrong"}
                          ).status_code == 401


def test_s1_app_auth_and_health_from_real_toml(tmp_path, monkeypatch):
    """真实 TOML 驱动 app：401/200、/ready 豁免、/health 报告鉴权状态。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={
            "server": {"bind_host": "127.0.0.1", "api_key": "s3cret",
                       "port": 8199},
            "model": {"path": str(model), "context": 4096}},
        require_model_path=True)
    app, _ = make_app(cfg)
    with TestClient(app) as client:
        for path in ("/health", "/metrics", "/v1/models", "/"):
            assert client.get(path).status_code == 401, path
        allowed = {"Authorization": "Bearer s3cret"}
        health = client.get("/health", headers=allowed)
        assert health.status_code == 200
        body = health.json()
        assert body["context_len"] == 4096 and body["model"]
        assert client.get("/ready").status_code == 200     # 探活豁免
        assert client.get("/v1/models", headers=allowed).status_code == 200
        expect_contract(
            isinstance(body.get("auth"), dict)
            and body["auth"].get("enabled") is True,
            "P1-S1: /health 未如实报告 auth 状态（实际 %r）" % (body.get("auth"),))
        if isinstance(body.get("auth"), dict):
            assert body["auth"]["source"] == "api_key"
            assert body["auth"]["bind_host"] == "127.0.0.1"
            assert body["auth"]["loopback_only"] is True


def test_s1_loopback_no_key_app_state_reports_auth_disabled(tmp_path, monkeypatch):
    """回环无 key：app 仍可用，``app.state.auth`` 如实标记未启用。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"server": {"bind_host": "127.0.0.1"},
                       "model": {"path": str(model)}},
        require_model_path=True)
    app, _ = make_app(cfg)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    expect_contract(
        getattr(app.state, "auth", None) is not None,
        "P1-S1: create_app 未暴露 app.state.auth（鉴权状态不可观测）")
    if getattr(app.state, "auth", None) is not None:
        assert app.state.auth["enabled"] is False
        assert app.state.auth["source"] == "none"


# ---------------------------------------------------------------------------
# S1 的 serve 启动链落点：真实 cli.main → _cmd_serve → 返回码
# ---------------------------------------------------------------------------


def _serve_env_hermetic(tmp_path: Path, **overrides) -> dict[str, str]:
    """serve 测试用环境：干净 RKLLM_*，并把默认 /etc 配置路径指向空目录。"""
    env = clean_env(**overrides)
    return env


def test_s1_serve_cli_exits_2_when_key_file_missing(tmp_path, monkeypatch, capsys):
    """真实 serve 启动链：key_file 不可用必须非 0 退出（_cmd_serve 捕获→2）。"""
    from realism_helpers import install_config_paths
    install_config_paths(monkeypatch,
                         toml_path=tmp_path / "absent.toml",
                         dropin_dir=tmp_path / "absent.d")
    missing = tmp_path / "absent.key"
    probe = run_serve(
        monkeypatch,
        ["serve", "m.rkllm", "--fake-engine"],
        env=_serve_env_hermetic(tmp_path,
                                RKLLM_SERVER_API_KEY_FILE=str(missing)))
    err = capsys.readouterr().err
    expect_contract(
        probe.code == 2 and "api_key_file" in err,
        "P1-S1: serve 未因 key_file 缺失拒绝启动（code=%r，stderr=%r）"
        % (probe.code, err[-400:]))


def test_s1_serve_cli_refuses_non_loopback_without_key(tmp_path, monkeypatch, capsys):
    """真实 serve 启动链：非回环 + 无 key 必须拒绝启动（返回 2）。"""
    from realism_helpers import install_config_paths
    install_config_paths(monkeypatch,
                         toml_path=tmp_path / "absent.toml",
                         dropin_dir=tmp_path / "absent.d")
    probe = run_serve(
        monkeypatch,
        ["serve", "m.rkllm", "--fake-engine"],
        env=_serve_env_hermetic(tmp_path, RKLLM_BIND="0.0.0.0"))
    err = capsys.readouterr().err
    if probe.code == 0:
        # 未拒绝启动时补一刀：确认它真的把匿名数据面暴露出来了
        from realism_helpers import anonymous_health_status
        anon = anonymous_health_status(probe.app)
        expect_contract(
            False,
            "P1-S1: 非回环无 key 时仍启动（uvicorn host=%r，匿名 /health=%r）"
            % (probe.kwargs.get("host"), anon))
    expect_contract(probe.code == 2,
                    "P1-S1: serve 拒绝非回环无 key 的返回码应为 2，实际 %r（stderr=%r）"
                    % (probe.code, err[-300:]))


def test_s1_serve_cli_loopback_no_key_starts_and_serves(tmp_path, monkeypatch,
                                                        capsys):
    """回环无 key 的合法部署：真实 serve 启动链跑通，app 可服务。"""
    from realism_helpers import install_config_paths
    install_config_paths(monkeypatch,
                         toml_path=tmp_path / "absent.toml",
                         dropin_dir=tmp_path / "absent.d")
    model = _model_file(tmp_path)
    probe = run_serve(
        monkeypatch,
        ["serve", str(model), "--fake-engine"],
        env=_serve_env_hermetic(tmp_path, RKLLM_BIND="127.0.0.1"))
    err = capsys.readouterr().err
    assert probe.code == 0, "serve 启动失败：%s" % err[-800:]
    assert probe.app is not None
    assert probe.kwargs.get("host") == "127.0.0.1"
    with TestClient(probe.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
        # /ready：uvicorn 返回后 CLI 已 shutdown supervisor，状态为 stopped
        assert client.get("/ready").status_code == 503


# ===========================================================================
# 4) P1-C2：真实 cli_engine 构造不得产生 ``--max-tokens 0``
# ===========================================================================


@pytest.mark.parametrize("model_name,configured_class", [
    ("qwen3-8b.rkllm", None),        # 8B：非内置档位，按文件名推导
    ("qwen3-27b.rkllm", None),       # 27B：项目目标模型，非内置档位
    ("model.rkllm", "14b"),          # 无名文件 + 显式 14b
])
def test_c2_non_builtin_class_never_passes_zero_max_tokens(
        tmp_path, monkeypatch, model_name, configured_class):
    """8B/14B/27B：``--max-tokens`` 必须是保守正数（契约为 512），不能 0。"""
    model = tmp_path / model_name
    model.write_bytes(b"\0" * 4096)
    sections = {"model": {"path": str(model)}}
    if configured_class:
        sections["model"]["class"] = configured_class
    cfg = load_real_config(monkeypatch, tmp_path, toml_sections=sections,
                           require_model_path=True)
    captured = capture_supervisor(monkeypatch)
    from rkllm.cli_engine import _build_supervisor
    _build_supervisor(cfg, use_fake=False)
    args = captured.instances[-1].worker_args
    got = worker_arg(args, "--max-tokens")
    expect_contract(
        got == "512",
        "P1-C2: %s（档位 %r）生成 --max-tokens %r；应保守默认 512（不能 0）"
        % (model_name, configured_class, got))
    # 同一命令行必须携带真实 context/platform/线程数（不是 argparse 默认值）
    assert worker_arg(args, "--context") == str(cfg.model.context)
    assert worker_arg(args, "--platform") not in (None, "auto")
    assert args[0] == "real"


def test_c2_unnamed_tiny_file_defaults_to_4b_positive(tmp_path, monkeypatch):
    """无名小文件按体积归 4b：既有内置默认 512，回归防 0。"""
    model = _model_file(tmp_path, "model.rkllm")
    cfg = load_real_config(monkeypatch, tmp_path,
                           toml_sections={"model": {"path": str(model)}},
                           require_model_path=True)
    captured = capture_supervisor(monkeypatch)
    from rkllm.cli_engine import _build_supervisor
    _build_supervisor(cfg, use_fake=False)
    assert worker_arg(captured.instances[-1].worker_args,
                      "--max-tokens") == "512"


def test_c2_explicit_max_tokens_wins_over_class_default(tmp_path, monkeypatch):
    """显式 ``model.max_tokens`` 永远优先于档位默认（27B 运维手动配置路径）。"""
    model = _model_file(tmp_path, "qwen3-27b.rkllm")
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"model": {"path": str(model), "max_tokens": 777}},
        require_model_path=True)
    captured = capture_supervisor(monkeypatch)
    from rkllm.cli_engine import _build_supervisor
    _build_supervisor(cfg, use_fake=False)
    args = captured.instances[-1].worker_args
    assert worker_arg(args, "--max-tokens") == "777"
    assert cfg.source_of("model.max_tokens") == "file"


def test_c2_fake_engine_in_process_really_starts(tmp_path, monkeypatch):
    """真实 ``_build_supervisor`` + 真实 FakeEngine/LocalSupervisor 启动一次。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"model": {"path": str(model)},
                       "engine": {"mode": "in_process"}},
        require_model_path=True)
    from rkllm.cli_engine import _build_supervisor
    sup = _build_supervisor(cfg, use_fake=True)
    try:
        assert sup.is_ready()
        text = "".join(item for item in sup.execute("hello", max_new_tokens=8)
                       if not isinstance(item, dict))
        assert text.startswith("answer")
    finally:
        sup.shutdown()


def test_c2_fake_worker_subprocess_really_starts(tmp_path, monkeypatch):
    """真实 Supervisor + fake worker 子进程：``_build_supervisor(use_fake=True)``。"""
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"model": {"path": str(model)},
                       "engine": {"mode": "subprocess"}},
        require_model_path=True)
    from rkllm.cli_engine import _build_supervisor
    sup = _build_supervisor(cfg, use_fake=True)
    try:
        assert sup.is_ready()
        text = "".join(item for item in sup.execute("hello", max_new_tokens=8)
                       if not isinstance(item, dict))
        assert text.startswith("answer")
    finally:
        sup.shutdown()


# ===========================================================================
# 5) 真实 CLI 参数构建：argparse 默认值不得击穿文件/环境层
# ===========================================================================


def _serve_args(argv: list[str]):
    from rkllm.cli import build_parser
    return build_parser().parse_args(["serve"] + argv)


def test_cli_serve_defaults_do_not_override_file_layer(tmp_path, monkeypatch):
    """``serve <model>`` 未显式给的参数必须是 None，不能击穿 TOML 的 context。"""
    from rkllm.cli_env import _legacy_cli_overrides
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"server": {"bind_host": "127.0.0.1"},
                       "model": {"path": str(model), "context": 8192}},
        cli=_legacy_cli_overrides(_serve_args([str(model)])),
        require_model_path=True)
    assert cfg.model.context == 8192
    assert cfg.source_of("model.context") == "file"   # 未被 argparse 默认值覆盖
    assert cfg.source_of("model.path") == "cli"       # 位置参数确实进 cli 层


def test_cli_serve_explicit_flags_land_in_cli_layer(tmp_path, monkeypatch):
    """显式 ``--context/--max-new-tokens/--host/--port`` 进 cli 层且真实生效。"""
    from rkllm.cli_env import _legacy_cli_overrides
    model = _model_file(tmp_path)
    args = _serve_args([str(model), "--context", "2048",
                        "--max-new-tokens", "256",
                        "--host", "127.0.0.1", "--port", "18100"])
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"model": {"path": str(model), "context": 4096}},
        cli=_legacy_cli_overrides(args),
        require_model_path=True)
    assert (cfg.model.context, cfg.source_of("model.context")) == (2048, "cli")
    assert cfg.model.max_tokens == 256
    assert cfg.server.port == 18100
    assert cfg.server.bind_host == "127.0.0.1"


def test_cli_serve_toml_path_via_env_boots_real_app(tmp_path, monkeypatch, capsys):
    """真实 ``cli.main(["serve", ...])`` → 真实 TOML → 真实 create_app。

    这是最接近生产的启动链：只把 uvicorn 的 socket 监听替换为记录调用。
    """
    model = _model_file(tmp_path, "m.rkllm")
    from realism_helpers import write_toml
    toml_path = write_toml(tmp_path, {
        "server": {"bind_host": "127.0.0.1", "port": 18099},
        "model": {"path": str(model), "context": 4096},
    })
    probe = run_serve(
        monkeypatch, ["serve", str(model), "--fake-engine"],
        env=clean_env(RKLLM_TOML=str(toml_path)))
    err = capsys.readouterr().err
    assert probe.code == 0, "serve 启动失败：%s" % err[-800:]
    assert probe.kwargs.get("host") == "127.0.0.1"
    assert probe.kwargs.get("port") == 18099      # 来自真实 TOML，不是默认 8080
    assert probe.app is not None
    svc = probe.app.state.svc
    assert svc.model_name == "m"
    assert svc.context_len == 4096
    assert svc.model_class == "4b"
    with TestClient(probe.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["context_len"] == 4096
        assert client.get("/v1/models").status_code == 200


def test_app_from_real_toml_and_conffile_layers(tmp_path, monkeypatch):
    """真实 TOML + 真实 conffile 环境：env 覆盖 TOML，app 用装配结果启动。"""
    model = _model_file(tmp_path)
    conffile = """\
RKLLM_PORT=18101
RKLLM_CONTEXT=2048
RKLLM_THREADS=2
"""
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={
            "server": {"bind_host": "127.0.0.1"},
            "model": {"path": str(model), "context": 4096},
        },
        conffile_text=conffile,
        require_model_path=True)
    assert cfg.server.port == 18101 and cfg.source_of("server.port") == "env"
    assert cfg.model.context == 2048 and cfg.source_of("model.context") == "env"
    assert cfg.model.threads == 2
    app, _ = make_app(cfg)
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["context_len"] == 2048
        assert body["max_queue"] == 32


# ===========================================================================
# 6) P1-O1：``/v1/completions`` 流式 legacy completions 契约（真 app + TestClient）
# ===========================================================================


def _completions_client(tmp_path, monkeypatch, *, tokens=None, perf=None):
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"server": {"bind_host": "127.0.0.1"},
                       "model": {"path": str(model)}},
        require_model_path=True)
    app, sup = make_app(cfg, tokens=tokens, perf=perf)
    return TestClient(app), sup


def test_o1_completions_stream_legacy_contract(tmp_path, monkeypatch):
    """流式帧必须是 legacy ``choices[0].text``，无 delta，id 前缀 ``cmpl-``。"""
    client, _ = _completions_client(tmp_path, monkeypatch,
                                    tokens=["你", "好"],
                                    perf={"prefill_tokens": 5,
                                          "generate_tokens": 2})
    response = client.post("/v1/completions",
                           json={"prompt": "你好", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _sse_frames(response.text)
    assert frames, "SSE 无数据帧"
    assert response.text.rstrip().endswith("data: [DONE]")

    for frame in frames:
        assert frame["object"] == "text_completion"
        assert frame["id"].startswith("cmpl-")
        assert "delta" not in frame["choices"][0], (
            "legacy completions 帧不得再有 delta：%r" % (frame,))
        assert "text" in frame["choices"][0]
    ids = {f["id"] for f in frames}
    created = {f["created"] for f in frames}
    assert len(ids) == 1 and len(created) == 1, "流内 id/created 必须全程一致"
    assert frames[0]["choices"][0]["text"] == ""
    assert (frames[-1]["choices"][0]["finish_reason"] == "stop")
    text = "".join(f["choices"][0]["text"] for f in frames)
    assert text == "你好"


def test_o1_completions_stream_include_usage_chunk(tmp_path, monkeypatch):
    """``stream_options.include_usage``：finish 后一个 choices=[] 的 usage 帧。"""
    client, _ = _completions_client(tmp_path, monkeypatch,
                                    tokens=["你", "好"],
                                    perf={"prefill_tokens": 5,
                                          "generate_tokens": 2})
    response = client.post("/v1/completions",
                           json={"prompt": "你好", "stream": True,
                                 "stream_options": {"include_usage": True}})
    frames = _sse_frames(response.text)
    usage_frames = [f for f in frames if f.get("usage") is not None]
    assert len(usage_frames) == 1
    usage_frame = usage_frames[0]
    assert usage_frame["choices"] == []              # OpenAI 文档形状
    assert usage_frame["object"] == "text_completion"
    assert usage_frame["id"].startswith("cmpl-")
    usage = usage_frame["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (5, 2)
    assert usage["total_tokens"] == 7
    assert usage["estimated"] is False               # perf 帧生效
    # usage 帧在 finish 帧之后、[DONE] 之前
    finish_index = next(i for i, f in enumerate(frames)
                        if f["choices"] and f["choices"][0]["finish_reason"])
    assert finish_index < frames.index(usage_frame)


def test_o1_completions_nonstream_legacy_shape(tmp_path, monkeypatch):
    """非流式 completions 回到同一信封：``text`` + ``cmpl-`` + text_completion。"""
    client, _ = _completions_client(tmp_path, monkeypatch, tokens=["你", "好"])
    response = client.post("/v1/completions", json={"prompt": "你好"})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert body["choices"][0]["text"] == "你好"


def test_chat_stream_keeps_delta_envelope(tmp_path, monkeypatch):
    """回归：O1 修复不得动 chat 的 ``chat.completion.chunk`` + delta 信封。"""
    client, _ = _completions_client(tmp_path, monkeypatch, tokens=["你", "好"])
    response = client.post("/v1/chat/completions",
                           json={**_serve_payload(), "stream": True})
    assert response.status_code == 200
    frames = _sse_frames(response.text)
    assert frames[0]["object"] == "chat.completion.chunk"
    assert frames[0]["id"].startswith("chatcmpl-")
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(f["choices"][0].get("delta", {}).get("content", "")
                   for f in frames)
    assert text == "你好"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
