"""只读子命令：list / info / selftest / config（A-4：从 cli.py 拆出）。

这些命令不启动服务、不加载模型（selftest 会 dlopen 库做一次校验），
与 serve/run/chat 的依赖面完全不同，单独成文件便于按需阅读。
"""

from __future__ import annotations

import os
import sys

from .cli_engine import _search_dirs

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
        "rkllm": "1.3.1-16",
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
    checks.append(("config schema", len(SCHEMA) == 80, "%d 键" % len(SCHEMA)))

    failed = 0
    for name, ok, detail in checks:
        print("%s  %-26s %s" % ("PASS" if ok else "FAIL", name, detail))
        failed += 0 if ok else 1
    print("\n%d/%d 通过" % (len(checks) - failed, len(checks)))
    return 0 if failed == 0 else 1
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
