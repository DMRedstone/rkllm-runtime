#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端推理服务健康 / 连通性检查 —— 三个同功能脚本合并后的唯一实现。

合并来源
--------
此前有三份功能重叠的检查脚本，本文件把它们合并为一：

  * check_connectivity.py —— 仅探测 /health 与 /v1/models（无对话检查、无退出码）
  * health_check.py       —— 本文件，在连通性之上加了端到端对话与退出码
  * verify_board.py       —— 端到端验收，含 6 项深度检测（鉴权拒绝、system 消息
                             污染、system+多轮、system-only 探测、流式 [DONE] 终止、
                             超长 prompt 拒绝）

检查分两级，用开关切换
----------------------
  默认      基础 3 项：/health、/v1/models、端到端对话
            快（数秒），适合压测 / 仿真之前的预检
  --full    追加 7 项深度检测，即原 verify_board.py 的验收口径
            适合交付、升级、换板之后做一次完整体检

退出码：0 全部通过，1 存在问题（可直接接入 CI 或 run_all.py 的编排）。

用法
----
    python health_check.py --host 192.0.2.10:8080 --key KEY
    python health_check.py --host 192.0.2.10:8080 --key KEY --full
    python health_check.py --host 192.0.2.10:8080 --key KEY --prompt "你好"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from board_api import answer_of, build_base, request   # 共享 HTTP 层（同目录 board_api.py）

# 强制 UTF-8 输出（Windows 控制台默认 GBK，会打不出中文）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# --- 原 verify_board.py 用的常量：探测「system 模板被污染」的标记 -----------------
# 若这些标记出现在普通回答里，说明上一条请求的 system prompt 泄漏了过来，
# 即服务端是「替换」而非「前置」了内置 chat 模板。
POISON_MARKERS = ("isnewtopic", '"title"', "session-title", "session_title")
TITLE_SYSTEM = (
    "Generate a concise, sentence-case title (3-7 words) that captures the main "
    "topic or goal of this session. Respond with EXACTLY one JSON object and "
    'nothing else: {"isNewTopic": boolean, "title": string}'
)

# request() / answer_of() 曾在 01、02、03、04、08、00 各写一份，签名与判读口径
# 互不一致；现统一由 board_api 提供，本脚本只保留检查逻辑。


# ---------------------------------------------------------------------------
# 基础检查（默认执行）
# ---------------------------------------------------------------------------

def check_basic(base, key, prompt, timeout):
    """基础 3 项：/health、/v1/models、端到端对话。返回 bool（是否健康）。"""
    healthy = True

    # 1) /health —— 同时用于区分「网络不可达」与「鉴权失败」
    status, body, elapsed, err = request(base, key, "/health", method="GET", timeout=8)
    if status == 200:
        try:
            info = json.loads(body)
            print("[PASS] /health           %.2fs  model=%s  requests=%s errors=%s"
                  % (elapsed, info.get("model"), info.get("stats", {}).get("requests"),
                     info.get("stats", {}).get("errors")))
            stats = info.get("stats") or {}
            required = ("requests", "rejected", "errors",
                        "prompt_tokens", "completion_tokens")
            missing = [name for name in required if name not in stats]
            if missing:
                print("[FAIL] /health stats     缺少字段：%s" % ", ".join(missing))
                healthy = False
        except Exception:  # noqa: BLE001
            print("[WARN] /health           返回 200 但响应无法解析：%s" % body[:80])
            healthy = False
    elif status == 401:
        # 服务在线但拒绝鉴权 —— 指向 key 问题，而不是网络问题
        print("[FAIL] /health           HTTP 401 鉴权失败：key 是否正确？")
        healthy = False
    elif status == "EXC":
        print("[FAIL] /health           无法连接：%s" % err)
        healthy = False
    else:
        print("[FAIL] /health           HTTP %s  %s" % (status, body[:80]))
        healthy = False

    # 服务不可达时后续检查没有意义
    if not healthy:
        return False

    # 2) /v1/models —— 确认模型列表可用
    status, body, elapsed, err = request(base, key, "/v1/models", method="GET", timeout=8)
    if status == 200:
        try:
            ids = [m["id"] for m in json.loads(body).get("data", [])]
            print("[PASS] /v1/models        %.2fs  %s" % (elapsed, ids))
        except Exception:  # noqa: BLE001
            print("[WARN] /v1/models        响应无法解析：%s" % body[:80])
    else:
        print("[FAIL] /v1/models        HTTP %s  %s" % (status, body[:80]))
        healthy = False

    # 3) 端到端对话 —— 确认推理链路真的能出结果
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 32}
    status, body, elapsed, err = request(base, key, "/v1/chat/completions",
                                         payload=payload, timeout=timeout)
    if status == 200:
        answer = answer_of(body)
        if answer is None:
            print("[WARN] chat/completions  响应无法解析：%s" % body[:80])
            healthy = False
        else:
            print("[PASS] chat/completions  %.2fs  %r" % (elapsed, answer[:60]))
    else:
        print("[FAIL] chat/completions  HTTP %s  %s" % (status, body[:80]))
        healthy = False

    return healthy


# ---------------------------------------------------------------------------
# 深度检查（--full，口径取自原 verify_board.py）
# ---------------------------------------------------------------------------

def check_full(base, key, timeout):
    """7 项深度检测 + 1 项收尾存活核对。返回 (通过数, 失败数)。"""
    passed = failed = 0
    # B-3：errors 是服务启动以来的累计值——记录基线，检测 8 改比「本次检查期间增量」
    _status0, _body0, _e0, _err0 = request(base, key, "/health", method="GET", timeout=8)
    _errors_base = 0
    try:
        _errors_base = int((json.loads(_body0).get("stats") or {}).get("errors") or 0)
    except Exception:  # noqa: BLE001
        pass

    def rec(name, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
        else:
            failed += 1
        print("[%s] %-22s %s" % ("PASS" if ok else "FAIL", name, detail))

    def ask(payload, label, timeout_override=None):
        return request(base, key, "/v1/chat/completions", payload=payload,
                       timeout=timeout_override or timeout)

    # 1) 鉴权：不带 key 必须被拒（401）。服务未启用鉴权会返回 200，属配置选择。
    if key:
        status, body, _, _ = request(base, "", "/health", method="GET", timeout=8,
                                     send_key=False)
        rec("鉴权拒绝(401)", status == 401,
            "HTTP %s%s" % (status, "  注意：服务未启用鉴权" if status == 200 else ""))
    else:
        print("[SKIP] %-22s 未提供 --key" % "鉴权拒绝(401)")

    # 2) 长 system 消息必须能被应答（历史上曾出现永久挂起）
    status, body, elapsed, err = ask(
        {"messages": [{"role": "system", "content": TITLE_SYSTEM},
                      {"role": "user", "content": "帮我写一个python快排"}],
         "max_tokens": 64}, "长system应答")
    rec("长system消息应答", status == 200 and bool(answer_of(body)),
        "%s  %.1fs" % (status, elapsed))

    # 3) 关键回归：system 请求之后，普通请求不能被「污染」成模板回显
    #    先 /v1/reset 清掉检测 2 的 system 上下文——污染与历史残留两个变量分离
    _rst_status, _, _, _ = request(base, key, "/v1/reset", payload={}, timeout=8)
    rec("检测3前置reset", _rst_status in (200, 204), "HTTP %s" % _rst_status)
    status, body, elapsed, err = ask(
        {"messages": [{"role": "user", "content": "用一句话介绍杭州"}],
         "max_tokens": 64}, "污染检测")
    answer = answer_of(body) or ""
    poisoned = any(marker in answer.lower() for marker in POISON_MARKERS)
    rec("无system模板污染", status == 200 and len(answer.strip()) > 4 and not poisoned,
        "%s  %.1fs  %r%s" % (status, elapsed, answer[:36],
                             "  POISONED" if poisoned else ""))

    # 4) system + 多轮对话
    status, body, elapsed, err = ask(
        {"messages": [{"role": "system", "content": "你是一个简洁的助手。"},
                      {"role": "user", "content": "1+1等于几"},
                      {"role": "assistant", "content": "2"},
                      {"role": "user", "content": "那2+2呢"}], "max_tokens": 32},
        "system+多轮")
    rec("system+多轮对话", status == 200 and bool(answer_of(body)),
        "%s  %.1fs" % (status, elapsed))

    # 5) 仅 system 的探测请求必须被应答（旧版返回 400，让能用的模型看起来是坏的）
    status, body, elapsed, err = ask(
        {"messages": [{"role": "system", "content": "be nice"}], "max_tokens": 16},
        "system-only")
    rec("system-only 探测", status == 200, "HTTP %s" % status)

    # 6) 流式必须以 [DONE] 正常收尾（否则客户端会一直等）
    status, body, elapsed, err = ask(
        {"messages": [{"role": "user", "content": "说三个水果名"}],
         "max_tokens": 32, "stream": True}, "流式收尾")
    lines = [ln for ln in body.splitlines() if ln.startswith("data: ")]
    rec("流式以[DONE]收尾",
        status == 200 and bool(lines) and lines[-1].strip() == "data: [DONE]",
        "%s  %.1fs  %d chunks" % (status, elapsed, len(lines)))

    # 7) 超长 prompt 必须被干净拒绝（400），而不是把服务打崩
    status, body, elapsed, err = ask(
        {"messages": [{"role": "user", "content": "x" * 200000}], "max_tokens": 8},
        "超长拒绝")
    rec("超长prompt被拒(400)", status == 400, "HTTP %s" % status)

    # 8) 全部检查之后，服务仍存活且错误计数为 0
    status, body, _, _ = request(base, key, "/health", method="GET", timeout=8)
    info = {}
    try:
        info = json.loads(body)
    except Exception:  # noqa: BLE001
        pass
    stats = info.get("stats", {}) if isinstance(info, dict) else {}
    _errors_now = int(stats.get("errors") or 0)
    rec("检查后服务存活", status == 200 and _errors_now == _errors_base,
        "requests=%s errors=%s（本次检查期间增量 %d）"
        % (stats.get("requests"), _errors_now, _errors_now - _errors_base))

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("RKLLM_HOST", "192.0.2.10:8080"),
                        help="host:port（可用 RKLLM_HOST 覆盖）")
    parser.add_argument("--key", default=os.environ.get("RKLLM_KEY") or os.environ.get("RKLLM_API_KEY", ""),
                        help="API key（可用 RKLLM_KEY 覆盖）")
    parser.add_argument("--prompt", default="你好，请用一句话自我介绍",
                        help="端到端对话检查使用的提示词")
    parser.add_argument("--full", action="store_true",
                        help="在基础 3 项之外追加 7 项深度检测（原 verify_board 口径）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次对话检查超时（秒）")
    args = parser.parse_args()

    base = build_base(args.host)
    print("检查目标 : %s" % base)
    print("检查级别 : %s" % ("基础 + 深度（--full）" if args.full else "基础"))

    healthy = check_basic(base, args.key, args.prompt, args.timeout)
    if not healthy:
        print("\n结论：服务不健康 ✗")
        return 1

    if not args.full:
        print("\n结论：服务健康 ✓")
        return 0

    print("\n--- 深度检测 ---")
    passed, failed = check_full(base, args.key, args.timeout)
    print("\n结论：%s（深度检测 %d 项通过, %d 项失败）"
          % ("服务健康 ✓" if failed == 0 else "服务存在异常 ✗", passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
