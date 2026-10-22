#!/usr/bin/env python3
"""golden_diff.py — 新栈 golden 基线（S6「golden 全绿」工具）。

背景：1.3.1-16 起旧单文件 `cli/rkllm.py` 已从仓库删除，golden 不再做
“旧栈 vs 新栈”对照，改为“新栈 vs 已录基线”的回归检测：

    python golden_diff.py --save tests/golden_baseline.json   # 录基线
    python golden_diff.py --compare tests/golden_baseline.json
    python golden_diff.py                                     # 用默认基线

默认基线路径：输入/rkllm-runtime-1.3.1/tests/golden_baseline.json。
退出码 0 = 与基线一致；1 = 存在差异（差异清单打到 stdout）。

归一化：剥除 id / created / system_fingerprint 这类每次请求都变的字段；
`/health` 含 uptime/queue 等运行时量，不参与逐字段对比（启动健康由
wait_up 覆盖）。
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "输入" / "rkllm-runtime-1.3.1" / "src"
DEFAULT_BASELINE = (ROOT / "输入" / "rkllm-runtime-1.3.1" / "tests"
                    / "golden_baseline.json")

REQUESTS = [
    ("GET", "/v1/models", None),
    ("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "max_tokens": 64}),
    ("POST", "/v1/chat/completions", {
        "messages": [{"role": "system", "content": "回答保持一句话"},
                     {"role": "user", "content": "天空为什么是蓝的"}],
        "temperature": 0.5, "max_tokens": 64}),
    ("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "写三行诗"}],
        "stream": True, "max_tokens": 48}),
    ("POST", "/v1/completions", {"prompt": "数到三：", "max_tokens": 32}),
]

# /health 的 uptime/queue/cache 会随运行状态变化，不参与基线对比。
SKIP_COMPARE_PATHS = ("/health",)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_up(port: int, deadline_s: float = 60) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with opener.open("http://127.0.0.1:%d/health" % port,
                             timeout=3) as resp:
                if resp.status in (200, 401):     # 401 = 服务已起、仅需鉴权
                    return
        except (urllib.error.HTTPError, OSError):
            time.sleep(0.5)
    raise SystemExit("service on :%d did not become healthy in %ss"
                     % (port, deadline_s))


def start_new(port: int) -> subprocess.Popen:
    # 1.3.1-16 的假引擎开关沿用旧名 --fake-engine；`--engine fake` 是早期草案。
    return subprocess.Popen(
        [sys.executable, "-m", "rkllm.cli", "serve", "fake", "--fake-engine",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(SRC), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={"SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
             "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"})


def request(port: int, method: str, path: str, body: dict | None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
    if body and body.get("stream"):
        return _normalize_sse(raw)
    return _normalize_json(raw)


def _normalize_json(raw: str):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    obj.pop("id", None)
    obj.pop("created", None)
    obj.pop("system_fingerprint", None)
    obj.pop("service_tier", None)
    return obj


def _normalize_sse(raw: str):
    """流式：保留 delta 序列与 finish_reason 序列，剥掉每帧 id/created。"""
    deltas, finishes, usage_seen = [], [], False
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            deltas.append("[DONE]")
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            deltas.append({"_bad": payload[:80]})
            continue
        if chunk.get("usage"):
            usage_seen = True
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                deltas.append(delta["content"])
            if choice.get("finish_reason"):
                finishes.append(choice["finish_reason"])
    return {"deltas": deltas, "finish_reasons": finishes,
            "usage_frame": usage_seen}


def capture(port: int) -> dict:
    out = {}
    for method, path, body in REQUESTS:
        if any(skip in path for skip in SKIP_COMPARE_PATHS):
            continue
        key = "%s %s|%s" % (method, path,
                            json.dumps(body, ensure_ascii=False, sort_keys=True))
        out[key] = request(port, method, path, body)
    return out


def deep_diff(a, b, path: str = "$", out: list | None = None) -> list:
    out = out if out is not None else []
    if type(a) is not type(b):
        out.append("%s: 类型 %s != %s" % (path, type(a).__name__, type(b).__name__))
    elif isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append("%s.%s: 仅基线有 = %r" % (path, key, b[key]))
            elif key not in b:
                out.append("%s.%s: 仅当前有 = %r" % (path, key, a[key]))
            else:
                deep_diff(a[key], b[key], "%s.%s" % (path, key), out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append("%s: 长度 %d != %d" % (path, len(a), len(b)))
        for index, (left, right) in enumerate(zip(a, b)):
            deep_diff(left, right, "%s[%d]" % (path, index), out)
    elif a != b:
        out.append("%s: %r != %r" % (path, a, b))
    return out


def _run_and_capture() -> dict:
    port = free_port()
    proc = start_new(port)
    try:
        wait_up(port)
        return capture(port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="新栈 golden 基线对比")
    parser.add_argument("--save", metavar="FILE", help="录新栈基线")
    parser.add_argument("--compare", metavar="FILE", help="与基线对比")
    args = parser.parse_args()

    if args.save:
        data = _run_and_capture()
        Path(args.save).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print("基线已存: %s（%d 个请求）" % (args.save, len(data)))
        return 0

    baseline = Path(args.compare) if args.compare else DEFAULT_BASELINE
    if not baseline.is_file():
        print("缺少基线文件：%s\n先运行：python %s --save %s"
              % (baseline, Path(__file__).name, baseline))
        return 1

    expected = json.loads(baseline.read_text(encoding="utf-8"))
    actual = _run_and_capture()
    diffs = deep_diff(expected, actual)
    if not diffs:
        print("golden 全绿：%d 个请求与基线一致" % len(expected))
        return 0
    print("golden 存在差异：%d 处" % len(diffs))
    for item in diffs:
        print("  -", item)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
