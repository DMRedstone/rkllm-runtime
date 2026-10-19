#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""隔离并定位「并发请求导致板端服务崩溃」的复现脚本。

背景
----
在仿真测试中观察到：当后台「生成标题」请求与带历史的对话请求并发时，
板端进程死亡（客户端收到 RemoteDisconnected，随后 /health 连接被拒）。

本脚本的作用
------------
从「单请求串行」逐步升级到「多个并发模式」，每个场景之后做一次存活探测，
从而把肇事模式定位出来，而不是凭空猜测。同时测量守护进程把服务拉起的耗时。

每个场景会重复 N 次（默认 3），以给出「崩溃复现率」——偶发崩溃只跑一次
是抓不到的。

用法
----
    python 04_crash_probe.py --host 192.0.2.10:8080 --key KEY
    python 04_crash_probe.py --host 192.0.2.10:8080 --key KEY --repeat 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 复刻客户端「生成会话标题」用的 system 提示词
TITLE_SYSTEM = (
    "Generate a concise, sentence-case title (3-7 words) that captures the main "
    "topic or goal of this session. Respond with EXACTLY one JSON object and "
    'nothing else: {"isNewTopic": boolean, "title": string}')

# 一段多轮会话历史，用于构造「带历史的对话请求」
HISTORY = [
    {"role": "user", "content": "你好，请简单介绍一下你自己。"},
    {"role": "assistant", "content": "你好！我是 Qwen3.5，一个本地部署的语言模型。"},
    {"role": "user", "content": "刚才你说的是什么模型？"},
    {"role": "assistant", "content": "我是 Qwen3.5-4B，运行在 RK3588 NPU 上。"},
    {"role": "user", "content": "帮我把「今天天气不错」翻译成英文。"},
    {"role": "assistant", "content": "The weather is nice today."},
    {"role": "user", "content": "再把它翻译成日文。"},
    {"role": "assistant", "content": "今日は天気が良いです。"},
]

RESULTS = []


def req(base, key, payload, timeout=120, raw_method="POST", path="/v1/chat/completions"):
    """发送一次请求。

    返回 (status, elapsed, err)：status 为 HTTP 码或 "EXC"；err 为客户端的
    异常描述（用于识别 RemoteDisconnected 这类连接中断）。
    """
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=data, method=raw_method, headers=headers)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status, time.time() - started, None
    except urllib.error.HTTPError as exc:
        return exc.code, time.time() - started, None
    except Exception as exc:  # noqa: BLE001
        return "EXC", time.time() - started, repr(exc)


def health_ok(base, key):
    """探测 /health 是否返回 200（服务是否存活）。"""
    status, _, _ = req(base, key, None, timeout=5, raw_method="GET", path="/health")
    return status == 200


def wait_ready(base, key, timeout=90):
    """等待服务恢复（守护进程自动拉起），返回等待秒数；超时返回 None。"""
    started = time.time()
    while time.time() - started < timeout:
        if health_ok(base, key):
            return time.time() - started
        time.sleep(1.0)
    return None


def fire_concurrently(base, key, payloads, timeout=120):
    """同时发起多个请求（每个一个线程），返回每个请求的结果列表。"""
    lock = threading.Lock()
    out = []

    def worker(payload):
        status, elapsed, err = req(base, key, payload, timeout=timeout)
        with lock:
            out.append({"status": status, "elapsed": round(elapsed, 2), "err": err})

    threads = [threading.Thread(target=worker, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def scenario(base, key, name, payloads, description, repeat=3):
    """执行一个并发场景 repeat 次，统计崩溃次数与恢复耗时。

    name        -- 场景名
    payloads    -- 本次同时发起的请求体列表
    description -- 场景说明
    repeat      -- 重复次数（用于评估偶发崩溃的复现率）
    返回该场景的崩溃次数。
    """
    print("\n=== %s  (x%d)" % (name, repeat))
    print("    %s" % description)
    crashes = 0
    runs = []
    for i in range(repeat):
        calls = fire_concurrently(base, key, payloads)
        alive = health_ok(base, key)
        recovery = None
        if not alive:
            crashes += 1
            print("    run %d: 崩溃 ✗  等待守护进程重启…" % (i + 1))
            recovery = wait_ready(base, key)
            print("    run %d: 恢复用时 %s" % (
                i + 1, "%.1fs" % recovery if recovery is not None else "超时未恢复"))
        else:
            print("    run %d: 存活 ✓  %s" % (i + 1, json.dumps(calls, ensure_ascii=False)))
        runs.append({"calls": calls, "crashed": not alive,
                     "recovery_s": round(recovery, 1) if recovery is not None else None})
    print("    小结: %d/%d 次触发崩溃" % (crashes, repeat))
    RESULTS.append({"scenario": name, "repeat": repeat, "crashes": crashes, "runs": runs})
    return crashes


def main():
    """命令行入口：逐个场景做并发崩溃复现实验。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("RKLLM_HOST", "192.0.2.10:8080"),
                        help="host:port（可用 RKLLM_HOST 覆盖）")
    parser.add_argument("--key", default=os.environ.get("RKLLM_KEY") or os.environ.get("RKLLM_API_KEY", ""),
                        help="API key（可用 RKLLM_KEY 覆盖）")
    parser.add_argument("--repeat", type=int, default=3, help="每个场景的重复次数")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "results"),
                        help="结果 JSON 输出目录")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    base = args.host if args.host.startswith(("http://", "https://")) else "http://" + args.host
    rep = max(1, args.repeat)

    # 实验开始前确保服务在线（若刚崩过，等它自动拉起）
    if not health_ok(base, args.key):
        print("服务当前不可达，先等待其就绪…")
        if wait_ready(base, args.key) is None:
            print("服务未就绪，退出")
            return 1

    # 各类用于构造并发负载的请求体
    short = {"messages": [{"role": "user", "content": "请回答 ok"}], "max_tokens": 8}
    chat_with_history = {"messages": [{"role": "system", "content": "You are a helpful assistant."}]
                         + HISTORY + [{"role": "user", "content": "顺便问一下，列表和元组有什么区别？"}],
                         "max_tokens": 64}
    title = {"messages": [{"role": "system", "content": TITLE_SYSTEM},
                          {"role": "user", "content": "帮我写一个 Python 快速排序"}],
             "max_tokens": 48}
    long_a = {"messages": [{"role": "user", "content": "请概括：" + "嵌入式设备部署大模型需要考虑内存与算力。" * 60}],
              "max_tokens": 32}
    long_b = {"messages": [{"role": "user", "content": "请分析：" + "模型量化会带来精度损失。" * 60}],
              "max_tokens": 32}

    # 从最温和到最敌对逐步升级
    scenario(base, args.key, "S1 serial title request", [title],
             "单个『生成标题』请求，串行 —— 验证它本身是否有害", rep)
    scenario(base, args.key, "S2 2x concurrent short", [short, short],
             "两个并发短请求 —— 压测已验证安全", rep)
    scenario(base, args.key, "S3 concurrent title + chat(history)", [title, chat_with_history],
             "复刻仿真崩溃点：标题请求 与 带8条历史的对话请求 并发", rep)
    scenario(base, args.key, "S4 2x concurrent long prompt", [long_a, long_b],
             "两个并发长 prompt —— 验证是否与 prompt 长度/并发 prefill 有关", rep)
    scenario(base, args.key, "S5 4x concurrent chat", [chat_with_history] * 4,
             "四个并发中长对话请求", rep)
    scenario(base, args.key, "S6 8x concurrent chat", [chat_with_history] * 8,
             "八个并发中长对话请求 —— 并发阶梯升到 8，覆盖更陡的峰值", rep)

    out = os.path.join(args.outdir, "crashprobe_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"target": base, "repeat": rep, "scenarios": RESULTS}, fh,
                  ensure_ascii=False, indent=2)
    print("\n结果写入 %s" % out)
    crashed = ["%s(%d/%d)" % (r["scenario"], r["crashes"], r["repeat"])
               for r in RESULTS if r["crashes"]]
    print("触发崩溃的场景: %s" % (crashed or "无——崩溃未能复现"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
