#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端大模型服务压力 / 负载测试脚本。

用途
----
对以 OpenAI 兼容协议暴露的板端推理服务（rkllm serve / RK3588 NPU）执行
分层的压力测试，覆盖：单请求延迟基线、并发-吞吐曲线、持续耐力、过载背压、
以及边界与畸形输入。全程每个子段结束后做一次存活探测，以便把「服务崩溃」
精确定位到引发它那一次请求，而不是笼统地归因于「压测把服务压垮了」。

相对原版 tools/stress_board.py 的改进（原脚本已并入本脚本，不再单独维护）
--------------------------------------------------------------------
0. 原版全部用例已并入：原 B 段「流式 vs 阻塞」现为本脚本 A2 段，其余段落一一对应；
1. B 段从「一次性并发突发」升级为真正的并发→吞吐曲线：多档并发各发固定数量
   请求，统计 p50/p90/p99 延迟与聚合吞吐；
2. 每次请求都读取服务端返回的 usage，吞吐按服务端上报的 completion_tokens
   聚合计算，而不是靠字符数估算；
3. 超长 prompt 的守卫探测不再写死上下文长度，改为由命令行参数传入；
4. 全部结果落盘为 JSON，供报告生成器（make_report.py）消费；
5. 段序保持「先温和、后敌意」：可能压垮服务的极限用例放在最后。

段落对照
--------
    A   单请求延迟基线          A2  流式 vs 阻塞（首字延迟）
    B   并发 → 吞吐曲线         C   持续耐力（延迟漂移）
    D   过载 / 背压            E   超长 prompt 守卫 + 畸形body

用法
----
    python llm_stress.py --host 192.0.2.10:8080 --key KEY
    python llm_stress.py --host 192.0.2.10:8080 --key KEY --quick          # 快速模式
    python llm_stress.py --host 192.0.2.10:8080 --key KEY --skip-limits    # 跳过可能压垮服务的 E 段

注意
----
* NPU 一次只能处理一个请求，因此「并发」测的是排队行为，不是并行吞吐；
* E 段可能把服务打到崩溃（这正是它的目的）。若崩溃，请在板端重启服务后
  依据输出定位是哪个用例触发的；
* 服务为独占资源，本脚本运行期间不要同时跑其他压测 / 仿真，否则结果互相污染。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

from board_api import Board, build_base, pct   # 共享 HTTP 层与百分位口径
from quality import (check_attribution, collect_sample,   # 内容形态分析（不判语义）
                     finish_reason_stats, overrun_stats, shape_metrics, stream_parity)

# Windows 控制台默认可能不是 UTF-8，中文输出会乱码，这里强制切到 UTF-8。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 全局结果收集器：每个 note() 调用追加一条记录，最终整体写入 JSON。
RESULTS = []


def note(section, metric, value, extra=None):
    """记录一条结果：既打印到控制台，也存入 RESULTS 供落盘。

    section -- 所属测试段（如 "A"、"B"）；空串表示全局提示
    metric  -- 指标名
    value   -- 指标值（字符串）
    extra   -- 可选的附加结构化数据（如并发曲线的完整统计字典）
    """
    row = {"section": section, "metric": metric, "value": value}
    if extra:
        row.update(extra)
    RESULTS.append(row)
    shown = value if extra is None else "%s  %s" % (value, json.dumps(extra, ensure_ascii=False))
    print("  %-40s %s" % (metric, shown))


def header(text):
    """打印分节标题。"""
    print("\n=== %s" % text)


# HTTP 客户端与 pct() 曾在本文件内实现（Board 类 + 局部 pct）；现统一由
# board_api 提供，本脚本只保留压测逻辑。alive() 包装保留原「不可达时 note
# 一条告警」的行为——board_api.Board.alive 只返回布尔与 info。

def alive(board, label):
    """存活探测 + 不可达告警。"""
    ok, info = board.alive()
    if not ok:
        note("", "  !! SERVICE UNREACHABLE after %s" % label,
             board.last_health_problem or "no /health")
    return ok, info



# ---------------------------------------------------------------------------
# 0. 环境探测
# ---------------------------------------------------------------------------

def probe_environment(board):
    """读取并打印服务的基础信息（模型名、uptime、计数器）。返回 health dict。"""
    header("0. environment")
    info = board.health()
    if info is None:
        print("  cannot reach /health: %s" % (board.last_health_problem or "no response"))
        return None
    note("0", "model", info.get("model"))
    note("0", "uptime_s", info.get("uptime_s"))
    note("0", "counters", "requests=%s rejected=%s errors=%s"
         % (info["stats"]["requests"], info["stats"]["rejected"], info["stats"]["errors"]))
    return info


# ---------------------------------------------------------------------------
# A. 单请求延迟基线
# ---------------------------------------------------------------------------

def baseline(board, quick):
    """A 段：串行发送若干代表性请求，建立无排队时的延迟与生成速率基线。

    覆盖短/中/长 prompt 与不同输出长度，用于区分「prefill 慢」与「decode 慢」。
    任一请求后服务不可用则立即返回 False。
    """
    header("A. latency baseline (single request, no queue)")
    probes = [
        ("short prompt, 32 tok", "你好", 32),
        ("medium prompt, 32 tok", "请把下面这句话翻译成英文：今天天气很好。" * 10, 32),
        ("long prompt, 32 tok", "请把下面这句话翻译成英文：今天天气很好。" * 60, 32),
        ("generate 128 tok", "请详细介绍杭州这座城市，越多越好。", 128),
        ("generate 256 tok", "请详细介绍杭州这座城市，越多越好。", 256),
    ]
    for label, content, max_tokens in probes:
        status, text, elapsed, usage = board.ask(content, max_tokens=max_tokens)
        ok, _ = alive(board, label)
        if status != 200:
            note("A", label, "%s  %.2fs  %s" % (status, elapsed, text))
        else:
            ct = (usage or {}).get("completion_tokens")
            # tok/s 仅按生成阶段大致估算（含少量 prefill 误差），用于横向对比
            rate = (ct / elapsed) if ct and elapsed else None
            note("A", label,
                 "%s  %.2fs  in=%d out=%d  %s tok/s" % (
                     status, elapsed, len(content), ct if ct is not None else len(text),
                     "%.1f" % rate if rate else "n/a"))
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# A2. 流式 vs 阻塞（首字延迟）
# ---------------------------------------------------------------------------

def streaming_vs_blocking(board, quick):
    """A2 段：对比流式与阻塞两种取数方式，量化「首字延迟」（TTFT）。

    合并自原 stress_board.py 的 B 段。意义：对话式前端靠流式才能「一有字就显示」；
    本段给出首字能比整段返回提前多久出现，用于判断交互体感是否可接受。
    """
    header("A2. streaming vs blocking (time to first token)")
    prompt = "请用三句话介绍西湖。"
    tokens = 48 if quick else 96

    # 流式：记录到「第一个有内容的增量」为止的耗时
    status, stream_text, elapsed, ttft = board.stream(
        {"messages": [{"role": "user", "content": prompt}],
         "max_tokens": tokens, "stream": True})
    ok, _ = alive(board, "A2 streaming")
    note("A2", "流式 %d tok" % tokens,
         "%s  total %.2fs  首字 %s  %d chars  %s"
         % (status, elapsed, ("%.2fs" % ttft) if ttft else "n/a", len(stream_text),
            "alive" if ok else "DIED"))
    if not ok:
        return False

    # 阻塞：同样内容，一次返回
    status, block_text, elapsed_block, usage = board.ask(prompt, max_tokens=tokens)
    note("A2", "阻塞 %d tok" % tokens,
         "%s  total %.2fs  %d chars" % (status, elapsed_block, len(block_text)))

    if ttft and elapsed_block:
        note("A2", "流式的首字提前量",
             "阻塞需 %.2fs 才有内容，流式 %.2fs 即有首字（提前约 %.0f%%）"
             % (elapsed_block, ttft, 100.0 * (elapsed_block - ttft) / elapsed_block))

    # 只比 TTFT 看不出流式丢字、重复首字、[DONE] 前缺尾块——必须比内容。
    parity = stream_parity(block_text, stream_text)
    note("A2", "流式内容一致性",
         "%s  阻塞 %d 字 / 流式 %d 字，公共前缀 %d"
         % (parity["verdict"], parity["blocking_chars"], parity["streamed_chars"],
            parity["common_prefix"]),
         extra=parity)
    return parity["verdict"] in ("identical", "equivalent")


# ---------------------------------------------------------------------------
# A3. 响应归属（串包）
# ---------------------------------------------------------------------------

def attribution(board, quick):
    """A3 段：并发下每个响应是否回到了发起它的那个请求。

    并发压测只看状态码与延迟，「响应跑到别人的请求上」完全看不出来，而后果是
    用户拿到别人的答案。做法：每个请求的 prompt 里埋一个只属于它的标记，然后看
    响应里出现的是自己的标记还是别人的。
    """
    header("A3. response attribution under concurrency (cross-talk)")
    count = 4 if quick else 8
    markers = ["MK%02dQZ" % i for i in range(count)]
    collected = {}
    lock = threading.Lock()

    def worker(index):
        marker = markers[index]
        prompt = "请原样输出下面这串字符，不要加任何解释、标点或换行：%s" % marker
        status, text, _elapsed, _usage = board.ask(prompt, max_tokens=24, timeout=180)
        with lock:
            collected[index] = (marker, text if status == 200 else "")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    verdict = check_attribution([collected[i] for i in sorted(collected)])
    note("A3", "x%d concurrent attributed" % count,
         "%s  matched %d / missing %d / foreign %d"
         % (verdict["verdict"], verdict["matched"], verdict["missing"], verdict["foreign"]),
         extra=verdict)
    return alive(board, "A3 attribution")[0]


# ---------------------------------------------------------------------------
# A5. 输出形态：截断 / overrun / 退化 / 乱码
# ---------------------------------------------------------------------------

def output_shape(board, quick):
    """A5 段：输出在形态上还能不能用（不判语义）。

    语义质量（改写得好不好）只能人看；这里只抓机器能判的四种形态：
      · finish_reason=length 或请求长度与实际严重不符（截断）
      · 实际输出 / 请求 max_tokens 的比值（overrun 的依据，配置项靠它定）
      · 塌成重复词流（越长约 1000 token 越常见）
      · 多字节字符在字节预算处被切断（U+FFFD）

    跑完顺便 reset 一次：这一段会灌进较长的输出，不清掉会污染后续段的上下文。
    """
    header("A5. output shape (truncation / overrun / degeneration)")
    tiers = [32, 128] if quick else [32, 128, 384]
    repeats = 2 if quick else 3
    for requested in tiers:
        samples, texts = [], []
        for _ in range(repeats):
            status, body, elapsed, _usage = board.call(
                payload={"messages": [{"role": "user",
                                       "content": "请写一段约 %d 字的说明文，主题是秋天的早晨，"
                                                  "语言平实，不要分点。" % (requested * 2)}],
                         "max_tokens": requested},
                timeout=900)
            if status != 200:
                continue
            try:
                data = json.loads(body)
                choice = (data.get("choices") or [{}])[0]
                text = ((choice.get("message") or {}).get("content") or "")
                tokens = int(((data.get("usage") or {}).get("completion_tokens")) or 0)
                reason = choice.get("finish_reason")
            except (ValueError, AttributeError):
                continue
            samples.append(collect_sample(requested, reason, tokens, text, elapsed))
            texts.append(text)
        if not samples:
            note("A5", "max_tokens=%d" % requested, "all requests failed")
            continue
        reasons = finish_reason_stats(samples)
        overrun = overrun_stats(samples)
        shape = shape_metrics(max(texts, key=len))
        note("A5", "max_tokens=%d  x%d" % (requested, len(samples)),
             "finish %s  trunc %.0f%%  overrun中位 %.2f 最大 %.2f  degen %.2f  %d 字"
             % (reasons["counts"], reasons["truncated_pct"], overrun.get("median") or 0,
                overrun.get("max") or 0, shape["degen"], shape["chars"]),
             extra={"requested": requested, "finish": reasons, "overrun": overrun,
                    "shape": shape, "samples": samples})
        if shape["degen"] > 0.55:
            note("A5", "  !! degeneration at max_tokens=%d" % requested,
                 "degen %.2f，最长整段复现 %d 字" % (shape["degen"], shape["longest_repeat"]))
        if shape["mojibake"] > 0:
            note("A5", "  !! mojibake at max_tokens=%d" % requested,
                 "U+FFFD 占比 %.3f（多字节字符被切断）" % shape["mojibake"])

    board.call("/v1/reset", payload={}, timeout=15)
    return alive(board, "A5 shape")[0]


# ---------------------------------------------------------------------------
# B. 并发 -> 吞吐曲线
# ---------------------------------------------------------------------------

def concurrency_curve(board, quick, levels, reqs_per_level, max_tokens):
    """B 段：在多个并发档位下各发固定数量请求，绘制「并发→吞吐」曲线。

    levels          -- 并发档位列表，如 [1,2,4,8,16]
    reqs_per_level  -- 每档总请求数
    max_tokens      -- 该段请求的输出上限

    每档并发按 level 个一组分批下发（组内并发、组间串行），统计：
      wall 总墙钟、QPS、p50/p90/p99 延迟、聚合 completion_tokens 吞吐、失败数。
    对单 NPU 串行服务，预期现象是 QPS 不随并发上升、延迟随并发线性抬升（排队）。
    """
    header("B. concurrency -> throughput curve (NPU serves one request at a time)")
    for level in levels:
        # 请求数取 level 的整数倍，保证各档批结构一致（跨档 p50/p99 可比）
        reqs = max(level, ((reqs_per_level + level - 1) // level) * level)
        lock = threading.Lock()
        rows = []

        def worker(index):
            """单次请求工作线程：发请求并记录 (状态, 延迟, 输出token数)。"""
            status, text, elapsed, usage = board.ask(
                "第 %d 号请求，请只回答 ok" % index, max_tokens=max_tokens, timeout=300)
            with lock:
                rows.append((status, elapsed, (usage or {}).get("completion_tokens")))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(reqs)]
        started = time.time()
        # 按 level 分批：一批内并发，批间等待，从而精确控制并发度
        for chunk_start in range(0, len(threads), level):
            batch = threads[chunk_start:chunk_start + level]
            for th in batch:
                th.start()
            for th in batch:
                th.join()
        wall = time.time() - started

        ok_rows = [r for r in rows if r[0] == 200]
        lat = [r[1] for r in ok_rows]
        toks = sum(r[2] or 0 for r in ok_rows)
        errors = len(rows) - len(ok_rows)
        stats = {
            "requests": reqs, "concurrency": level,
            "ok": len(ok_rows), "errors": errors,
            "wall_s": round(wall, 2),
            "qps": round(reqs / wall, 3) if wall else None,
            "completion_tokens": toks,
            "agg_tok_s": round(toks / wall, 2) if wall and toks else None,
            "p50_s": round(pct(lat, 50), 3) if lat else None,
            "p90_s": round(pct(lat, 90), 3) if lat else None,
            "p99_s": round(pct(lat, 99), 3) if lat else None,
            "max_s": round(max(lat), 3) if lat else None,
        }
        note("B", "concurrency=%d  x%d requests" % (level, reqs),
             "wall %.1fs  qps %.2f  p50 %.2fs  p90 %.2fs  p99 %.2fs  agg %.1f tok/s  err %d"
             % (wall, stats["qps"] or 0, stats["p50_s"] or 0, stats["p90_s"] or 0,
                stats["p99_s"] or 0, stats["agg_tok_s"] or 0, errors),
             extra=stats)
    ok, info = alive(board, "concurrency")
    if info:
        note("B", "  server counters", "requests=%s rejected=%s errors=%s"
             % (info["stats"]["requests"], info["stats"]["rejected"], info["stats"]["errors"]))
    return ok


# ---------------------------------------------------------------------------
# C. 持续耐力
# ---------------------------------------------------------------------------

def endurance(board, quick):
    """C 段：连续串行请求，观察延迟是否随时间漂移（内存泄漏 / 热降频的迹象）。"""
    header("C. endurance (sequential, watches for drift)")
    rounds = 15 if quick else 40
    latencies, toks, failures = [], [], 0
    for index in range(rounds):
        status, text, elapsed, usage = board.ask(
            "连续第 %d 次提问，回答 ok 就行" % index, max_tokens=16)
        board.reset()   # 每轮清历史：钉死上下文长度，让 drift 只归因于时间/温度，而非历史累积
        if status == 200:
            latencies.append(elapsed)
            toks.append((usage or {}).get("completion_tokens") or 0)
        else:
            failures += 1
            note("C", "  request %d failed" % index, "%s  %s" % (status, text))
    ok, info = alive(board, "endurance")
    if latencies:
        # 比较「前四分之一」与「后四分之一」的中位延迟，量化漂移
        first = statistics.median(latencies[:max(1, len(latencies) // 4)])
        last = statistics.median(latencies[-max(1, len(latencies) // 4):])
        note("C", "%d sequential requests" % rounds,
             "%d ok  %d failed  median %.2fs  agg %.1f tok/s"
             % (len(latencies), failures, statistics.median(latencies),
                sum(toks) / sum(latencies) if sum(latencies) else 0))
        note("C", "  latency drift",
             "first quarter %.2fs -> last quarter %.2fs  (delta %+.2fs)"
             % (first, last, last - first))
    if info:
        note("C", "  server counters", "requests=%s errors=%s"
             % (info["stats"]["requests"], info["stats"]["errors"]))
    return ok


# ---------------------------------------------------------------------------
# D. 过载 / 背压
# ---------------------------------------------------------------------------

def overload(board, quick, burst_size):
    """D 段：突发大量并发短请求，观察服务是否对超额负载做背压（503/429）还是照单全收排队。

    burst_size -- 每轮同时发起的请求数
    """
    header("D. overload / backpressure (how does it shed load?)")
    # 并发阶梯：必须越过服务端队列容量（request_queue_size=256）才可能触发背压 503/429
    ladder = [min(burst_size, 16), 64, 256] if not quick else [min(burst_size, 16), 64]
    ladder = sorted({s for s in ladder if s > 0})
    all_rows = []
    lock = threading.Lock()

    def flooder():
        """单次短请求（max_tokens=4），模拟瞬时洪峰中的一个客户端。"""
        status, body, elapsed, _ = board.call(payload={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}, timeout=300)
        with lock:
            all_rows.append((status, elapsed))

    for index, burst_size in enumerate(ladder):
        note("D", "burst", "flooding with %d concurrent short requests" % burst_size)
        threads = [threading.Thread(target=flooder) for _ in range(burst_size)]
        started = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.time() - started
        statuses = {}
        for status, _ in all_rows[-burst_size:]:
            statuses[status] = statuses.get(status, 0) + 1
        note("D", "burst %d: %d simultaneous" % (index + 1, burst_size),
             "wall %.1fs  statuses %s" % (wall, statuses))
    statuses = {}
    for status, _ in all_rows:
        statuses[status] = statuses.get(status, 0) + 1
    note("D", "totals", "statuses %s" % statuses)
    if 503 in statuses:
        note("D", "  overflow shed with 503", "%d of %d refused" % (statuses[503], len(all_rows)))
    else:
        note("D", "  no explicit shed observed",
             "%d requests all queued/accepted" % len(all_rows))
    ok, info = alive(board, "overload")
    if info:
        note("D", "  server counters", "requests=%s rejected=%s errors=%s"
             % (info["stats"]["requests"], info["stats"]["rejected"], info["stats"]["errors"]))
    return ok


# ---------------------------------------------------------------------------
# E. 边界与健壮性（最具敌意的用例放最后）
# ---------------------------------------------------------------------------

def limits(board, context_len, quick):
    """E 段：从温和到敌意依次施加参数边界、超长 prompt 与畸形请求体。

    目标：确认服务是「干净拒绝（4xx）」还是「崩溃 / 5xx」。
    每个用例后做存活探测；一旦服务不可用，立即返回并指明肇事用例。
    """
    header("E. limit conditions (least -> most hostile)")
    budget = (context_len or 4096) * 3   # 服务端守卫按 3 字节/token 估算
                                         # （cli/rkllm.py: estimated_tokens = (bytes+2)//3；8 是已废弃的旧版值）
    note("E", "advertised guard", "context %s tokens, %s byte budget" % (context_len, budget))

    cases = [
        ("max_tokens = 0", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}),
        ("max_tokens = -1", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": -1}),
        ("max_tokens huge", {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 100000000}),
        ("temperature = -1", {"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 8, "temperature": -1}),
        ("top_p = 2.0", {"messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 8, "top_p": 2.0}),
        ("2000 messages", {"messages": [{"role": "user", "content": "x"}] * 2000,
                           "max_tokens": 8}),
        ("unknown role", {"messages": [{"role": "wizard", "content": "hi"}], "max_tokens": 8}),
        ("control chars", {"messages": [{"role": "user",
                                         "content": "\x00\x01\x07\x1b[31mred\x1b[0m"}],
                           "max_tokens": 8}),
        ("emoji + CJK + rtl", {"messages": [{"role": "user",
                                             "content": "🌸 中文 العربية עברית مرحبا"}],
                               "max_tokens": 8}),
        ("prompt just under budget", {"messages": [{"role": "user", "content": "a" * (budget - 100)}],
                                      "max_tokens": 8}),
        ("prompt just over budget", {"messages": [{"role": "user", "content": "a" * (budget + 100)}],
                                     "max_tokens": 8}),
        ("CJK prompt near budget", {"messages": [{"role": "user", "content": "测" * (budget // 3)}],
                                    "max_tokens": 8}),
    ]
    for label, payload in cases:
        # 超长用例给更宽松的超时，其余走默认
        timeout = 300 if "budget" in label or "2000" in label else None
        status, body, elapsed, _ = board.call(payload=payload, timeout=timeout)
        detail = "%s  %.2fs  %s" % (status, elapsed, body[:70].replace("\n", " "))
        ok, _ = alive(board, label)
        note("E", label, detail + ("" if ok else "   <== SERVICE DIED"))
        if not ok:
            return False, label

    # E2：直接发送非合法 JSON / 超大体积等畸形请求体，验证协议层健壮性
    header("E2. malformed bodies")
    raw_cases = [
        ("not JSON", b"this is not json"),
        ("JSON array", b"[1,2,3]"),
        ("JSON string", b'"hello"'),
        ("truncated JSON", b'{"messages":[{"role":"user",'),
        ("empty body", b""),
        ("deeply nested", b'{"messages":[{"role":"user","content":' + b"[" * 500 + b"]" * 500 + b'}]}'),
        ("8MB body", json.dumps({"messages": [{"role": "user",
                                               "content": "a" * (8 * 1024 * 1024 + 1000)}]}).encode()),
    ]

    # E2-chunked：Transfer-Encoding: chunked（服务端只认 Content-Length——已知裂缝，见全库报告 P0-5）
    header("E2c. chunked transfer (server only reads Content-Length)")
    try:
        import http.client as _hc
        _host = board.base.split("//", 1)[-1]
        _conn = _hc.HTTPConnection(_host, timeout=120)
        _body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 4}).encode()
        _conn.putrequest("POST", "/v1/chat/completions")
        _conn.putheader("Transfer-Encoding", "chunked")
        if board.key:
            _conn.putheader("Authorization", "Bearer " + board.key)
        _conn.endheaders()
        for _i in range(0, len(_body), 64):
            _chunk = _body[_i:_i + 64]
            _conn.send(b"%x\r\n" % len(_chunk) + _chunk + b"\r\n")
        _conn.send(b"0\r\n\r\n")
        _resp = _conn.getresponse()
        _status, _bodytxt = _resp.status, _resp.read().decode("utf-8", "replace")
        _conn.close()
    except Exception as exc:  # noqa: BLE001
        _status, _bodytxt = "EXC", repr(exc)
    _ok, _ = alive(board, "chunked transfer")
    note("E2c", "chunked body (no Content-Length)",
         "%s  %s%s" % (_status, _bodytxt[:60].replace("\n", " "),
                       "" if _ok else "   <== SERVICE DIED"))
    if not _ok:
        return False, "chunked transfer"
    for label, raw in raw_cases:
        request = board._request("/v1/chat/completions", raw)
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status, body = response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            status, body = "EXC", repr(exc)
        elapsed = time.time() - started
        ok, _ = alive(board, label)
        note("E2", label, "%s  %.2fs  %s%s" % (status, elapsed, body[:60].replace("\n", " "),
                                               "" if ok else "  <== SERVICE DIED"))
        if not ok:
            return False, label
    return True, None


# ---------------------------------------------------------------------------

def main():
    """命令行入口：解析参数、按段执行、落盘 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("RKLLM_HOST", "192.0.2.10:8080"),
                        help="host:port（可用环境变量 RKLLM_HOST 覆盖）")
    parser.add_argument("--key", default=os.environ.get("RKLLM_KEY") or os.environ.get("RKLLM_API_KEY", ""),
                        help="API key（环境变量 RKLLM_KEY 或 RKLLM_API_KEY）")
    parser.add_argument("--quick", action="store_true", help="快速模式：减少请求轮数")
    parser.add_argument("--skip-limits", action="store_true",
                        help="跳过 E 段（该段可能把服务压垮）")
    parser.add_argument("--levels", default="1,2,4,8,16",
                        help="并发档位，逗号分隔")
    parser.add_argument("--concurrency-requests", type=int, default=24,
                        help="并发段每个档位的请求总数")
    parser.add_argument("--concurrency-tokens", type=int, default=32,
                        help="并发段单请求 max_tokens")
    parser.add_argument("--context-len", type=int, default=4096,
                        help="服务声明的上下文窗口（token），用于超长 prompt 守卫探测")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "results"),
                        help="结果 JSON 输出目录")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    board = Board(args.host, args.key, 240)
    started_at = time.time()
    info = probe_environment(board)
    if info is None:
        print("cannot use http://%s : %s"
              % (args.host, board.last_health_problem or "no response from /health"))
        return 1

    print("target   : http://%s" % args.host)
    print("model    : %s" % info.get("model"))
    print("mode     : %s" % ("quick" if args.quick else "full"))

    # 各段按序执行；任一段探测到服务已死即停止后续（避免把崩溃归因到错误的用例）
    steps = [
        ("A", lambda: baseline(board, args.quick)),
        ("A2", lambda: streaming_vs_blocking(board, args.quick)),
        ("A3", lambda: attribution(board, args.quick)),
        ("A5", lambda: output_shape(board, args.quick)),
        ("B", lambda: concurrency_curve(board, args.quick, levels,
                                        args.concurrency_requests, args.concurrency_tokens)),
        ("C", lambda: endurance(board, args.quick)),
        ("D", lambda: overload(board, args.quick, 24 if args.quick else 32)),
    ]
    if not args.skip_limits:
        steps.append(("E", lambda: limits(board, args.context_len, args.quick)[0]))

    survived = True
    stopped_at = None
    for name, fn in steps:
        if not fn():
            survived = False
            stopped_at = name
            break

    # 汇总落盘，供报告生成器读取
    elapsed = time.time() - started_at
    summary = {
        "target": build_base(args.host),
        "model": info.get("model"),
        "mode": "quick" if args.quick else "full",
        "started_at": started_at,
        "elapsed_s": round(elapsed, 1),
        "completed": survived,
        "stopped_at_section": stopped_at,
        "results": RESULTS,
    }
    out = os.path.join(args.outdir, "stress_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\nresults written to %s" % out)
    if not survived:
        print("*** the service did not survive section %s - that is a finding" % stopped_at)
        return 2   # 完成但有 finding
    return 0


if __name__ == "__main__":
    sys.exit(main())
