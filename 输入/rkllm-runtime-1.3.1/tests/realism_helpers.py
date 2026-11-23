"""真实路径测试辅助（task-12）：临时 conffile / rkllm.toml / `.d` 构造器。

为什么需要这个模块（2026-11-23 用户反馈）：
现有测试大量使用 ``load_config(env={})`` 或直接构造配置字典，导致
``/etc/default/rkllm-server``（systemd EnvironmentFile）、``/etc/rkllm/rkllm.toml``
和 ``/etc/default/rkllm-server.d/*.conf`` 三层真实来源从未被真正读入；CLI
``serve`` 的参数构建、``create_app`` 的鉴权解析也被 mock 替换。于是
"conffile 死键 / `.d` 清空 TOML / 非 4B 档位 max-tokens=0 / 鉴权静默失效"
这类缺陷在测试里完全不可见。

本模块的隔离原则（不 mock 被测算子）：

1. **只重定向路径常量**：``realism_helpers.install_config_paths`` 把
   ``rkllm.config.DEFAULT_TOML_PATH`` / ``DEFAULT_DROPIN_DIR`` 指向 ``tmp_path``。
   ``load_config`` 本体一行不改——读取、优先级、``_normalize_legacy``、校验全是
   真实代码。
2. **conffile 按 systemd EnvironmentFile 语义解析**（前缀 ``RKLLM_`` 环境变量就
   是 conffile 的运行时形态），``load_config(env=...)`` 的 ``env`` 参数正是生产
   里 systemd 注入的环境，所以这条路径与真实 serve 一致。
3. **外部边界才用替身**：NPU worker（``FakeSupervisor`` 产 token / ``FakeEngine``
   进程内）、uvicorn 的 socket 监听（``run_serve`` 只 patch ``uvicorn.run``）。
   ``create_app`` / ``TestClient`` / ``_build_supervisor`` 都是真实调用。

已知未落地的 P1 生产修复用 ``expect_contract`` 记为 xfail（报告里列明原因）；
修复落地后同一测试自动转为严格断言。设置 ``RKLLM_REALISM_STRICT=1`` 时
xfail 视为失败，供 CI 使用（防止修复过期/回归后仍被 xfail 掩盖）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]          # 输入/rkllm-runtime-1.3.1
SRC = PKG / "src"
REPO = PKG.parents[1]                              # rkllm-runtime/
SHIPPED_CONFFILE = PKG / "default" / "rkllm-server"
SHIPPED_DROPIN_DIR = PKG / "default" / "rkllm-server.d"
SHIPPED_EXAMPLE = PKG / "default" / "rkllm-server-example"

STRICT_ENV = "RKLLM_REALISM_STRICT"


# ---------------------------------------------------------------------------
# 预期失败开关（生产修复未落地时的诚实标注）
# ---------------------------------------------------------------------------


def expect_contract(ok: bool, pending: str) -> None:
    """``ok`` 为 False 时：普通运行记 xfail，严格模式（CI）直接失败。

    ``pending`` 必须写清"哪条 P1 修复还没落地"，报告里据此列出预期失败项。
    修复落地后 ``ok`` 变为 True，测试自动严格；之后再回归就是真失败。
    """
    if ok:
        return
    if os.environ.get(STRICT_ENV) == "1":
        pytest.fail("契约未满足（%s=1 严格模式）：%s" % (STRICT_ENV, pending))
    pytest.xfail("生产修复尚未落地：" + pending)


# ---------------------------------------------------------------------------
# 环境 / conffile / TOML / .d 构造器
# ---------------------------------------------------------------------------


def clean_env(**overrides: object) -> dict[str, str]:
    """干净环境：去掉宿主机残留的 ``RKLLM_*``，只保留显式覆盖。

    宿主或 CI 上可能残留 ``RKLLM_*``（例如板端调试 shell），不清掉会让测试
    结果依赖机器状态——这本身就是旧测试的仿真度问题之一。
    """
    base = {k: v for k, v in os.environ.items()
            if not k.upper().startswith("RKLLM_")}
    base.update({str(k).upper(): str(v) for k, v in overrides.items()})
    return base


def parse_env_file(path: str | os.PathLike) -> dict[str, str]:
    """按 systemd ``EnvironmentFile=`` 语义解析 conffile 或 ``.d/*.conf``。

    支持：``KEY=value``、``#`` 整行注释、``export KEY=value``、成对引号。
    与生产 ``rkllm.config._load_dropins`` 的解析规则一致（该函数只吃
    ``*.conf`` 文件名，conffile 本体无后缀，所以这里做等价解析）。
    """
    out: dict[str, str] = {}
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def _toml_literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    raise TypeError("不支持写入 TOML 的测试值：%r" % (value,))


def write_toml(directory: str | os.PathLike,
               sections: dict[str, dict] | str,
               name: str = "rkllm.toml") -> Path:
    """写真实 TOML 文件：``sections`` 为 {section: {key: value}} 或原始文本。"""
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(sections, str):
        path.write_text(sections, encoding="utf-8")
        return path
    lines: list[str] = []
    for section, payload in sections.items():
        lines.append("[%s]" % section)
        for key, value in payload.items():
            lines.append("%s = %s" % (key, _toml_literal(value)))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_dropins(directory: str | os.PathLike,
                  files: dict[str, str] | str) -> Path:
    """写真实 ``rkllm-server.d`` 目录（``files`` 为 {文件名: 文本} 或单文本）。"""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(files, str):
        (d / "10-model.conf").write_text(files, encoding="utf-8")
    else:
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8")
    return d


def write_conffile(directory: str | os.PathLike, text: str,
                   name: str = "rkllm-server") -> Path:
    """写真实 conffile（``/etc/default/rkllm-server`` 的临时替身）。"""
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def install_config_paths(monkeypatch, *, toml_path=None, dropin_dir=None) -> None:
    """只重定向路径常量；``load_config`` 实现保持真实。"""
    from rkllm import config as cfgmod
    if toml_path is not None:
        monkeypatch.setattr(cfgmod, "DEFAULT_TOML_PATH", str(toml_path))
    if dropin_dir is not None:
        monkeypatch.setattr(cfgmod, "DEFAULT_DROPIN_DIR", str(dropin_dir))


def load_real_config(monkeypatch, tmp_path, *,
                     toml_sections: dict[str, dict] | str | None = None,
                     dropins: dict[str, str] | str | None = None,
                     dropin_dir: str | os.PathLike | None = None,
                     conffile_text: str | None = None,
                     env: dict[str, str] | None = None,
                     cli: dict | None = None,
                     require_model_path: bool = False):
    """用真实 ``load_config()`` 装配一层临时但完整的配置栈。

    - ``toml_sections``：写入 tmp_path/rkllm.toml，并作为 ``DEFAULT_TOML_PATH``；
    - ``dropins`` / ``dropin_dir``：``DEFAULT_DROPIN_DIR``（可指向仓库真实
      ``default/rkllm-server.d``）；
    - ``conffile_text``：写成真实 conffile 文件后按 systemd 语义解析成 env
      （等价 systemd ``EnvironmentFile=`` 注入）；
    - 不存在的 TOML/.d 会被重定向到 tmp 里的空路径，保证不读宿主机 ``/etc``。

    返回 ``rkllm.config.Config``；校验失败抛 ``ConfigError``（与生产一致）。
    """
    env_full = clean_env()
    if conffile_text is not None:
        conf_path = write_conffile(tmp_path, conffile_text)
        env_full.update(parse_env_file(conf_path))
    if env:
        env_full.update({str(k).upper(): str(v) for k, v in env.items()})

    toml_path = None
    if toml_sections is not None:
        toml_path = write_toml(tmp_path, toml_sections)
    resolved_dropin = None
    if dropins is not None:
        resolved_dropin = write_dropins(tmp_path / "rkllm-server.d", dropins)
    elif dropin_dir is not None:
        resolved_dropin = Path(dropin_dir)

    install_config_paths(
        monkeypatch,
        toml_path=toml_path or (tmp_path / "absent-rkllm.toml"),
        dropin_dir=resolved_dropin or (tmp_path / "absent-rkllm-server.d"))

    from rkllm.config import load_config
    return load_config(env=env_full, cli=cli,
                       require_model_path=require_model_path)


def shipped_conffile_env() -> dict[str, str]:
    """仓库里真正会打进 deb 的 conffile → systemd 环境变量。"""
    return parse_env_file(SHIPPED_CONFFILE)


def shipped_dropin_env() -> dict[str, str]:
    """仓库里真正会打进 deb 的 ``.d`` 目录 → 合并后的环境变量（后文件覆盖先文件）。"""
    env: dict[str, str] = {}
    for path in sorted(SHIPPED_DROPIN_DIR.glob("*.conf")):
        env.update(parse_env_file(path))
    return env


# ---------------------------------------------------------------------------
# 外部边界替身（只在真的无法在本机模拟的边界使用）
# ---------------------------------------------------------------------------


class FakeSupervisor:
    """真实 WorkerClient 的替身：确定性 token 流 + end 帧同款 perf 元数据。

    只替换 NPU worker 边界（本机无 RK3588/NPU）；HTTP 栈、队列、Pipeline、
    SSE 全部走真实代码，与 ``test_server_app.FakeSupervisor`` 同形状但可配置。
    """

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None, perf=None):
        self.tokens = list(tokens or ["你", "好"])
        self.perf = perf
        self.aborted = False
        self.calls: list[tuple[str, dict]] = []

    def is_ready(self) -> bool:
        return self.state == "ready"

    def execute(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        for token in self.tokens:
            yield token
        if self.perf:
            yield {"perf": self.perf}

    def abort(self) -> None:
        self.aborted = True


def make_app(cfg, *, tokens=None, perf=None):
    """真实 ``create_app(cfg, supervisor=FakeSupervisor())``。"""
    from rkllm.server.app import create_app
    sup = FakeSupervisor(tokens=tokens, perf=perf)
    return create_app(cfg, supervisor=sup), sup


def capture_supervisor(monkeypatch):
    """把 ``rkllm.server.supervisor.Supervisor`` 换成只记录 ``worker_args`` 的类。

    用途：真实执行 ``_build_supervisor``（含档位推导、platform 解析、
    ``--max-tokens`` 计算）但**不 spawn 真 worker**（本机无 NPU、也不该在
    C2 回归里验证 C 库）。返回捕获类，``.instances`` 里每个实例有
    ``worker_args`` / ``kwargs``。
    """
    import rkllm.server.supervisor as supmod

    class CapturedSupervisor:
        instances: list["CapturedSupervisor"] = []

        def __init__(self, worker_args, **kwargs):
            self.worker_args = list(worker_args)
            self.kwargs = kwargs
            self.started = False
            CapturedSupervisor.instances.append(self)

        state = "ready"
        metrics = {"worker_restarts_total": 0}

        def start(self) -> None:
            self.started = True

        def shutdown(self) -> None:
            self.stopped = True

        def is_ready(self) -> bool:
            return True

    monkeypatch.setattr(supmod, "Supervisor", CapturedSupervisor)
    return CapturedSupervisor


def worker_arg(args: list[str], flag: str) -> str | None:
    """取 worker 命令行参数值（``--max-tokens 512`` → ``"512"``）。"""
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


class ServeProbe:
    """一次真实 CLI ``serve`` 启动链的捕获结果。"""

    def __init__(self, code: int, app, kwargs: dict, argv: list[str]):
        self.code = code
        self.app = app                  # uvicorn.run 收到的 ASGI app（None=未走到监听）
        self.kwargs = kwargs            # uvicorn.run 的关键字（host/port/... 来自真实 cfg）
        self.argv = argv


def run_serve(monkeypatch, argv, *, env: dict[str, str] | None = None) -> ServeProbe:
    """真实 ``cli.main(["serve", ...])``：只替换 uvicorn 的 socket 监听。

    覆盖 ``cli.main → cli_serve._cmd_serve → load_config(require_model_path=True)
    → _resolve_model_path / 档位默认 / _apply_memory_budget → _build_supervisor
    （``--fake-engine`` 时是真实 fake worker 子进程）→ create_app → uvicorn``。
    ``uvicorn.run`` 换成记录调用（不真占端口），因此本函数还能拿到 ASGI app
    继续用 TestClient 验证鉴权/health。

    ``env`` 会整体替换进程环境里的 ``RKLLM_*``（其余变量保留），模拟 systemd
    的干净 EnvironmentFile 注入。
    """
    import uvicorn
    import rkllm.platform as platform_mod

    captured: dict = {}
    real_run = uvicorn.run

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    class _NullLock:
        def restore(self) -> None:
            pass

    monkeypatch.setattr(uvicorn, "run", fake_run)
    # 锁频需要 root/sysfs；本机只验证启动链，不碰 sysfs
    monkeypatch.setattr(platform_mod, "apply_freq_lock",
                        lambda *a, **k: _NullLock())

    if env is not None:
        for key in list(os.environ):
            if key.upper().startswith("RKLLM_") and key not in env:
                monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))

    from rkllm import cli
    code = cli.main(list(argv))
    monkeypatch.setattr(uvicorn, "run", real_run)
    return ServeProbe(code, captured.get("app"), captured.get("kwargs") or {},
                      list(argv))


# ---------------------------------------------------------------------------
# 安全启动契约探针（S1 的多个合法落点共用同一语义）
# ---------------------------------------------------------------------------


def anonymous_health_status(app) -> int | None:
    """不带 Bearer 访问 ``/health`` 的状态码（app 为 None 时返回 None）。"""
    if app is None:
        return None
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        return client.get("/health").status_code
