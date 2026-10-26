"""只读子命令：list / info / selftest / config（A-4：从 cli.py 拆出）。

这些命令不启动服务、不加载模型（selftest 会 dlopen 库做一次校验），
与 serve/run/chat 的依赖面完全不同，单独成文件便于按需阅读。
"""

from __future__ import annotations

import os
import sys
import time

from .cli_engine import _search_dirs
from . import VERSION

__all__ = ["_cmd_list", "_cmd_info", "_cmd_selftest", "_cmd_config"]


def _cmd_list(args) -> int:
    import json
    found = []
    for directory in _search_dirs(getattr(args, "model_dir", None)):
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if not name.endswith(".rkllm"):
                continue
            path = os.path.join(directory, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            found.append({"name": name, "path": path,
                          "size_mb": round(size / 1048576, 1)})
    if getattr(args, "json", False):
        print(json.dumps(found, ensure_ascii=False, indent=2))
        return 0
    if not found:
        sys.stderr.write("未找到 .rkllm 模型；用 --model-dir 指定目录\n")
        return 1
    for item in found:
        print("%-60s %8.1f MB  %s" % (item["name"], item["size_mb"], item["path"]))
    return 0
def _cmd_info(args) -> int:
    import json
    import platform as py_platform
    from .platform import available_mb, detect_platform, npu_status
    npu_available, npu_detail = npu_status()
    info = {
        "rkllm": VERSION,
        "python": sys.version.split()[0],
        "platform_detected": detect_platform(),
        "machine": py_platform.machine(),
        "mem_available_mb": (round(available_mb(), 0)
                             if available_mb() is not None else None),
        "npu_available": npu_available,
        "npu_detail": npu_detail,
    }
    try:
        from .abi import load_library
        _lib, lib_path, fingerprint = load_library(getattr(args, "lib", None))
        info.update({"library": lib_path,
                     "library_sha256": fingerprint.get("sha256"),
                     "library_size": fingerprint.get("size")})
    except Exception as exc:                      # noqa: BLE001
        info["library_error"] = str(exc)[:200]
    info["model_dirs"] = _search_dirs(getattr(args, "model_dir", None))
    if getattr(args, "json", False):
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    for key, value in info.items():
        print("%-18s %s" % (key, value))
    return 0
def _cmd_selftest(args) -> int:
    """安装自检：解释器 / 依赖 / 库 / 模型 / 配置表五步。"""
    checks: list[tuple[str, bool, str]] = [
        ("python>=3.13", sys.version_info[:2] >= (3, 13),
         sys.version.split()[0])]
    try:
        import fastapi
        import pydantic
        import uvicorn
        checks.append(("fastapi/uvicorn/pydantic", True,
                       "%s / %s / %s" % (fastapi.__version__,
                                         uvicorn.__version__,
                                         pydantic.VERSION)))
    except ImportError as exc:
        checks.append(("fastapi/uvicorn/pydantic", False, str(exc)))
    try:
        from .abi import load_library
        _lib, lib_path, fingerprint = load_library(getattr(args, "lib", None))
        checks.append(("librkllmrt", True,
                       "%s sha256=%s" % (lib_path,
                                         (fingerprint.get("sha256") or "?")[:12])))
    except Exception as exc:                      # noqa: BLE001
        checks.append(("librkllmrt", False, str(exc)[:120]))
    models = []
    for directory in _search_dirs(getattr(args, "model_dir", None)):
        try:
            models += [n for n in os.listdir(directory) if n.endswith(".rkllm")]
        except OSError:
            pass
    checks.append(("model files", bool(models), "%d 个" % len(models)))
    from .config import SCHEMA
    checks.append(("config schema", len(SCHEMA) == 83, "%d 键" % len(SCHEMA)))
    checks.extend(_selftest_http())

    failed = 0
    for name, ok, detail in checks:
        print("%s  %-26s %s" % ("PASS" if ok else "FAIL", name, detail))
        failed += 0 if ok else 1
    print("\n%d/%d 通过" % (len(checks) - failed, len(checks)))
    return 0 if failed == 0 else 1


def _selftest_http() -> list[tuple[str, bool, str]]:
    """HTTP 端到端回环：真实 uvicorn + 临时端口 + FakeEngine，不依赖 NPU。

    1.3.1-15 的 selftest 覆盖 models/401/chat/流式/stop/usage（「装完即用」
    的验收强度）；新栈一度缩水为纯静态检查，2026-10-25 审计后恢复。

    为什么不用 `fastapi.testclient`：TestClient 的 ASGI transport 会让
    `request.receive()`（断连监视用）立刻返回并空转，事件循环被饿死——
    实测每个 token 要等满一个心跳周期（15s）才投递到 SSE，一条 30 token
    的回环要跑几分钟，selftest 形同卡死。真机/uvicorn 下没有该现象
    （真 socket 的 receive() 会真正挂起），所以回环走真实端口：既贴近
    生产路径，又不引入测试栈特有的假故障。
    """
    try:
        import asyncio
        import http.client
        import json as _json
        import socket
        import threading

        import uvicorn

        from .config import load_config
        from .server.app import create_app
        from .server.supervisor import LocalSupervisor
        from .worker.fake import FakeEngine
    except Exception as exc:                      # noqa: BLE001
        return [("http e2e (deps)", False, str(exc)[:120])]

    key = "selftest-probe-key"
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    cfg = load_config(env={"RKLLM_API_KEY": key})
    app = create_app(cfg, supervisor=LocalSupervisor(FakeEngine()))
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
        access_log=False))
    server.install_signal_handlers = lambda: None   # 非主线程装 handler 会炸

    def _serve() -> None:
        asyncio.run(server.serve())

    th = threading.Thread(target=_serve, daemon=True, name="selftest-http")
    th.start()
    for _ in range(100):                            # 至多等 10s
        if server.started:
            break
        time.sleep(0.1)
    if not server.started:
        server.should_exit = True
        return [("http e2e (boot)", False, "uvicorn 未能在 10s 内起来")]

    def call(method: str, path: str, body=None, auth: bool = True):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + key
        try:
            conn.request(method, path,
                         body=_json.dumps(body) if body is not None else None,
                         headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, raw
        finally:
            conn.close()

    out: list[tuple[str, bool, str]] = []
    try:
        status, raw = call("GET", "/v1/models")
        body = _json.loads(raw) if status == 200 else {}
        out.append(("http models", status == 200 and body.get("object") == "list"
                    and bool(body.get("data")), "GET /v1/models"))

        status, _ = call("GET", "/v1/models", auth=False)
        out.append(("http auth 401", status == 401, "无 Authorization → 401"))

        status, raw = call("POST", "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 16})
        body = _json.loads(raw) if status == 200 else {}
        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage") or {}
        out.append(("http chat",
                    status == 200
                    and "answer" in (choice.get("message") or {}).get(
                        "content", "")
                    and "completion_tokens" in usage,
                    "chat.completion + usage（%s）"
                    % (choice.get("finish_reason") or "?")))

        status, raw = call("POST", "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": "ping"}],
                            "stream": True})
        out.append(("http chat stream",
                    status == 200 and "chat.completion.chunk" in raw
                    and raw.rstrip().endswith("data: [DONE]"),
                    "SSE + [DONE]"))

        status, raw = call("POST", "/v1/completions",
                           {"prompt": "ping", "stream": True})
        out.append(("http completions stream",
                    status == 200 and "text_completion" in raw
                    and raw.rstrip().endswith("data: [DONE]"),
                    "text_completion 信封（P0-2 修复后）"))

        status, raw = call("POST", "/v1/chat/completions",
                           {"messages": [{"role": "user",
                                          "content": "一二三四五"}],
                            "stop": ["三"], "max_tokens": 64})
        body = _json.loads(raw) if status == 200 else {}
        text = ((body.get("choices") or [{}])[0].get("message")
                or {}).get("content") or ""
        out.append(("http stop 截断", status == 200 and "三" not in text,
                    repr(text[:24])))

        status, raw = call("GET", "/health")
        stats = (_json.loads(raw) if status == 200 else {}).get("stats") or {}
        out.append(("http health stats",
                    status == 200
                    and {"requests", "rejected", "errors"} <= set(stats),
                    "15 版兼容计数（requests=%s）"
                    % stats.get("requests", "?")))
    except Exception as exc:                      # noqa: BLE001
        out.append(("http e2e (abort)", False, str(exc)[:120]))
    finally:
        server.should_exit = True
        th.join(timeout=5)
    return out


def _cmd_config(args) -> int:
    from .config import load_config, config_docs, config_init, config_show
    if args.action == "docs":
        print(config_docs())
        return 0
    if args.action == "init":
        target = "client" if args.client else "board"
        path = args.path or ("mcp.toml" if args.client
                             else "/etc/rkllm/rkllm.toml")
        path = os.path.expanduser(path)
        if os.path.exists(path) and not args.force:
            sys.stderr.write("[rkllm] 已存在，未覆盖：%s（用 --force 覆盖）\n" % path)
            return 1
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(config_init(target))
        print("已写入 %s（%s 模板）" % (path, target))
        return 0
    cfg = load_config(file=args.config, require_model_path=False)
    if args.action == "show":
        print(config_show(cfg, as_json=args.json))
        return 0
    return 0     # validate：load_config 抛错即非零退出
