#!/usr/bin/env python3
"""golden_diff.py — 新旧栈 golden 差分基线（S6 验收 DoD「golden 全绿」工具）。

对旧实现（cli/rkllm.py --fake-engine）与新实现（src/rkllm --engine fake）各起
一个 FakeEngine 服务，发同一请求集，归一化后逐项对比：

  归一化：剥除 id / created / system_fingerprint / perf / usage 计数 /
  timing / model 名（两栈 fake 模型名不同）；流式剥 chunk 级 id/created，
  只留 delta 序列与 finish_reason。

用法：
  python golden_diff.py                # 双栈自起，直接对比
  python golden_diff.py --save base.json   # 存基线（板端真机录帧用）
  python golden_diff.py --compare base.json  # 与已存基线对比

退出码 0 = 全绿；1 = 存在差异（差异清单打到 stdout）。
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
CLI = ROOT / "输入" / "rkllm-runtime-1.3.1" / "cli" / "rkllm.py"

REQUESTS = [
    ("GET", "/health", None),
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

STREAM = "text/event-stream"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_up(port: int, deadline_s: float = 60) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.HTTPError, OSError):
            time.sleep(0.5)
    raise SystemExit(f"service on :{port} did not become healthy in {deadline_s}s")


def start_old(port: int, model_path: Path) -> subprocess.Popen:
    # fake 模式绕过引擎实例化，但模型路径解析仍要求文件存在：
    # 占位空文件 dummy.rkllm（审核报告附录 A 口径）
    return subprocess.Popen(
        [sys.executable, str(CLI), "serve", str(model_path), "--fake-engine",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(SRC.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_new(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "rkllm.cli", "serve", "--engine", "fake",
         "--model", "fake", "--port", str(port)],
        cwd=str(SRC), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={"SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
             "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"})


def request(port: int, method: str, path: str, body: dict | None):
    # 系统代理会拦 127.0.0.1（实测 "upstream connect failed"），全部直连
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{port}{path}"
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
    if isinstance(obj.get("usage"), dict):
        obj["usage"].pop("estimated", None)
    for choice in obj.get("choices") or []:
        msg = choice.get("message") or {}
        msg.pop("reasoning_content", None)
    return obj


def _normalize_sse(raw: str):
    """流式：只保留 delta 文本序列与 finish_reason 序列（D4 归一化）。"""
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
    return {"deltas": deltas, "finish_reasons": finishes, "usage_frame": usage_seen}


def capture(port: int) -> dict:
    out = {}
    for method, path, body in REQUESTS:
        key = f"{method} {path}" + (f" #{REQUESTS.index((method, path, body))}" if body else "")
        out[f"{method} {path}|{json.dumps(body, ensure_ascii=False, sort_keys=True)}"] = \
            request(port, method, path, body)
    return out


def deep_diff(a, b, path="$", out: list | None = None) -> list:
    out = out if out is not None else []
    if type(a) is not type(b):
        out.append(f"{path}: 类型 {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: 仅新栈有 = {b[k]!r:.120}")
            elif k not in b:
                out.append(f"{path}.{k}: 仅旧栈有 = {a[k]!r:.120}")
            else:
                deep_diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r:.120} != {b!r:.120}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="把新栈响应存为基线 JSON（板端真机录帧用）")
    ap.add_argument("--compare", help="与已存基线对比而不是与旧栈对比")
    args = ap.parse_args()

    p_old, p_new = free_port(), free_port()
    procs = []
    dummy = Path(__import__("tempfile").gettempdir()) / "golden-dummy.rkllm"
    dummy.write_bytes(b"")          # 占位空模型：fake 引擎不读内容，路径需存在
    try:
        if args.compare:
            base = json.loads(Path(args.compare).read_text(encoding="utf-8"))
            proc = start_new(p_new)
            procs.append(proc)
            wait_up(p_new)
            new = capture(p_new)
            diffs = deep_diff(base, new)
            print(f"对比基线 {args.compare}：{len(diffs)} 处差异")
            for d in diffs:
                print("  -", d)
            return 0 if not diffs else 1

        proc_old = start_old(p_old, dummy)
        procs.append(proc_old)
        proc_new = start_new(p_new)
        procs.append(proc_new)
        wait_up(p_old)
        wait_up(p_new)
        old, new = capture(p_old), capture(p_new)
        if args.save:
            Path(args.save).write_text(
                json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"基线已存: {args.save}")
        all_diffs = {}
        for key in sorted(set(old) | set(new)):
            diffs = deep_diff(old.get(key), new.get(key), f"{key}")
            if diffs:
                all_diffs[key] = diffs
        if not all_diffs:
            print(f"golden 全绿：{len(old)} 个请求，新旧栈归一化后无差异")
            return 0
        total = sum(len(v) for v in all_diffs.values())
        print(f"golden 存在差异：{total} 处，涉及 {len(all_diffs)} 个请求")
        for key, diffs in all_diffs.items():
            print(f"\n[{key}]")
            for d in diffs:
                print("  -", d)
        return 1
    finally:
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
