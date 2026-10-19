#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_realworld.py —— 真实公文场景测试（基于用户提供的 10 份真实文档）

为什么做这个测试
----------------
先前的压力/仿真测试用的是合成 prompt，只能反映「速度与稳定性」。
本脚本改用**真实公文**作为负载，回答三个更贴近落地的问题：

  1. 可行性：模型能否完成真实的公文任务（润色 / 摘要 / 扩写 / 审校 / 条目化改写）？
  2. 边界：真实文档 1000~2100 字，而服务上下文仅 4096 token —— 整篇处理会不会溢出？
  3. 真实吞吐：按真实文档规模，处理一份要多久；并发时是否稳定。

测试分组
--------
  A 串行任务   ：6 类真实任务各跑一次，观察成败、耗时、输出形态
  B 分段处理   ：把最长文档按段落切分逐段润色（真实落地最可行的方式）
  C 多轮迭代   ：还原真实迭代链（草稿→润色→审校），每轮输出喂下一轮
  D 并发稳定性 ：多个「文档润色」任务并发，观察是否触发已知的偶发崩溃
  E 批量吞吐   ：10 份文档依次处理，统计端到端效率

只读语料
--------
语料由 docx_corpus.py 生成于 corpus/corpus.json，本脚本不接触原始 docx。

用法
----
    python llm_realworld.py --host 192.0.2.10:8080 --key <KEY>
    python llm_realworld.py --host ... --key ... --only a,b       # 只跑指定组
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from board_api import build_base, health_ok, post_chat, wait_ready   # 共享 HTTP 层

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus", "corpus.json")
RESULTS_DIR = os.path.join(HERE, "results")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "realworld_samples")

# ---------------------------------------------------------------------------
# 真实任务模板：尽量贴近「把公文丢给助手」时的自然指令
# ---------------------------------------------------------------------------
PROMPTS = {
    "polish": (
        "你是公文写作专家。请对下面这份公文进行润色：保持原意与结构不变，"
        "改进用词、语句通顺度与公文规范性。只输出润色后的正文，不要任何解释。\n\n"
        "【原文】\n{text}"
    ),
    "summarize": (
        "请为下面这份公文写一段内容摘要，不超过 200 字，概括核心事项与工作要求。"
        "只输出摘要正文。\n\n【公文】\n{text}"
    ),
    "expand": (
        "下面是一份工作流程草稿。请依据公文写作规范（分条列项、用语规范、"
        "逻辑清晰、层级分明）将其补充完善为一份正式稿。只输出正式稿正文。\n\n"
        "【草稿】\n{text}"
    ),
    "proofread": (
        "请审校下面这份公文，逐一指出其中的错别字、标点误用、语句不通顺或表述不严谨之处，"
        "并给出修改建议。用条目列表输出。\n\n【公文】\n{text}"
    ),
    "restructure": (
        "请把下面这份公文改写成条目清晰的汇报要点，每条一句话，便于口头汇报。"
        "只输出要点列表。\n\n【公文】\n{text}"
    ),
}

# ---------------------------------------------------------------------------
# HTTP 调用
# ---------------------------------------------------------------------------


# post_chat / health_ok / wait_ready 曾在本文件内各写一份；现由 board_api 提供，
# 返回结构与判读口径与其余脚本一致。


# 质量粗判（无需人工即可给出「像不像样」的量化线索）
# ---------------------------------------------------------------------------


def cjk_ratio(s):
    """中文占比，用于识别乱码/英文跑飞。"""
    if not s:
        return 0.0
    return len(re.findall(r"[\u3400-\u9fff]", s)) / len(s)


def ngram_overlap(src, out, n=4):
    """输出覆盖了多少「输入的 4-gram」——越高越像照抄原文。"""
    a = re.sub(r"\s+", "", src)
    b = re.sub(r"\s+", "", out)
    A = set(a[i:i + n] for i in range(max(0, len(a) - n + 1)))
    if not A:
        return 0.0
    B = set(b[i:i + n] for i in range(max(0, len(b) - n + 1)))
    return len(A & B) / len(A)


def judge(task, src, res):
    """给出该次调用的粗判结论，返回 (verdict, note)。"""
    if not res["ok"]:
        return "失败", (res.get("error") or "")[:60]
    out = res["text"]
    if not out.strip():
        return "空输出", "返回内容为空"
    notes = []
    cr = cjk_ratio(out)
    if cr < 0.2:
        notes.append("中文占比低(%.0f%%)" % (cr * 100))
    ov = ngram_overlap(src, out)
    # 改写类任务若与原文高度重合，视为「疑似照抄、未实质改写」
    if task in ("polish", "restructure", "expand", "summarize") and ov > 0.90:
        notes.append("疑似照抄(重合%.0f%%)" % (ov * 100))
    fr = res.get("finish_reason")
    if fr and fr != "stop":
        notes.append("finish=%s" % fr)
    return ("成功" if not notes else "存疑"), "；".join(notes) or "正常"


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------


def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {f["key"]: f for f in data["files"]}


def run_task(base, key, task, text, max_tokens=1024, timeout=600, retries=3):
    """
    单次真实任务调用（含偶发崩溃自动重试），附带粗判。

    重要：服务在真实负载下存在低概率崩溃，崩溃后由守护进程自动拉起。
    这里对失败请求「等待恢复 → 重试」，并把崩溃次数计入测试数据，
    既保证测试能跑完，也把稳定性缺陷量化记录下来。
    """
    prompt = PROMPTS[task].format(text=text)
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    # F-5：失败分类——HTTP 4xx 是请求级拒绝（重试无意义，记 rejected）；连接中断/超时
    # 才是崩溃信号（记 crashes 并重试）；每条记录存 recovery_s，数据可自证。
    crashes, rejected, wait_s, recovery_s, res = 0, 0, 0.0, None, None
    for _ in range(retries + 1):
        res = post_chat(base, key, payload, timeout=timeout)
        if res["ok"]:
            break
        _err = res.get("error", "")
        if _err.startswith("HTTP 4"):
            rejected += 1
            break
        crashes += 1
        rec = wait_ready(base, key, max_wait=120)
        if rec is None:
            break
        wait_s += rec
        recovery_s = rec
    verdict, note = judge(task, text, res)
    if crashes:
        prefix = "" if note in ("正常", "") else note + "；"
        note = prefix + "遭遇崩溃 %d 次（已自动重试）" % crashes
    elif rejected:
        prefix = "" if note in ("正常", "") else note + "；"
        note = prefix + "请求被服务端拒绝（HTTP 4xx，不重试）"
    return {
        "task": task, "ok": res["ok"], "verdict": verdict, "note": note,
        "dt": round(res["dt"], 2), "wait_s": round(wait_s, 1), "crashes": crashes,
        "rejected": rejected, "recovery_s": round(recovery_s, 1) if recovery_s else None,
        "in_chars": len(text), "in_tok": res.get("prompt_tokens"),
        "out_chars": len(res["text"]), "out_tok": res.get("completion_tokens"),
        "finish_reason": res.get("finish_reason"),
        "echo": round(ngram_overlap(text, res["text"]), 3),
        "preview": re.sub(r"\s+", " ", res["text"])[:120],
        "full": res["text"],          # 完整输出（仅内存，落盘前会剥离）
        "error": res.get("error"),
    }


# ---------------------------------------------------------------------------
# A 组：串行真实任务
# ---------------------------------------------------------------------------


def group_a(base, key, corpus, out_dir):
    plan = [
        ("A1", "polish", "build_report_v1", 512, "短汇报润色"),
        ("A2", "summarize", "build_report", 384, "调研报告摘要"),
        ("A3", "expand", "plan_0312_draft", 1024, "草稿扩写为正式稿"),
        ("A4", "proofread", "plan_0312_draft", 640, "公文审校"),
        ("A5", "restructure", "plan_0320", 768, "长文改条目要点"),
        ("A6", "polish", "plan_0320", 2048, "★整篇润色(考验上下文边界)"),
    ]
    rows = []
    for tag, task, key_name, mt, desc in plan:
        rec = corpus[key_name]
        print("  [%s] %s | %s | 输入 %d 字 | max_tokens=%d ..." %
              (tag, desc, task, rec["chars"], mt), flush=True)
        r = run_task(base, key, task, rec["text"], max_tokens=mt)
        r.update({"tag": tag, "desc": desc, "source": key_name})
        rows.append(r)
        print("       -> %s %s | 用时 %.1fs | 输出 %s 字/%s tok | %s" % (
            r["verdict"], "(HTTP/连接失败)" if not r["ok"] else "",
            r["dt"], r["out_chars"], r["out_tok"], r["note"]), flush=True)
        # 保存完整输出供人工评审（不进结果 JSON）
        with open(os.path.join(out_dir, "A_%s_%s.txt" % (tag, key_name)), "w", encoding="utf-8") as fh:
            fh.write("任务：%s（%s）\n输入：%d 字\n%s\n" % (desc, task, rec["chars"], "=" * 50))
            fh.write("【原始语料】\n%s\n\n" % rec["text"])
            fh.write("【完整输出】\n%s\n" % r.get("full", ""))
    return rows


# ---------------------------------------------------------------------------
# B 组：分段处理（长文按段落切，逐段润色）
# ---------------------------------------------------------------------------


def split_paragraphs(text, target=260):
    """按自然段落聚合到约 target 字/段，模拟「分段喂给模型」。"""
    paras = [p for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > target:
            chunks.append(buf)
            buf = p
        else:
            buf = (buf + "\n" + p) if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def group_b(base, key, corpus, out_dir):
    rec = corpus["plan_0320"]
    chunks = split_paragraphs(rec["text"])
    print("  分段润色：%d 字 → %d 段（平均 %d 字/段）" %
          (rec["chars"], len(chunks), sum(len(c) for c in chunks) // len(chunks)), flush=True)
    rows = []
    for i, ch in enumerate(chunks, 1):
        r = run_task(base, key, "polish", ch, max_tokens=512)
        r.update({"seq": i, "in_chars": len(ch)})
        rows.append(r)
        print("    段 %2d/%d | %s | %.1fs | 出 %s 字 | %s" %
              (i, len(chunks), r["verdict"], r["dt"], r["out_chars"], r["note"]), flush=True)
    ok = sum(1 for r in rows if r["verdict"] == "成功")
    return {"chunks": len(chunks), "success": ok,
            "total_s": round(sum(r["dt"] for r in rows), 1),
            "rows": rows}


# ---------------------------------------------------------------------------
# C 组：多轮迭代（草稿→润色→审校，输出喂下一轮）
# ---------------------------------------------------------------------------


def group_c(base, key, corpus, out_dir):
    text = corpus["plan_0312_draft"]["text"]
    chain = [("第1轮·润色", "polish", 1024), ("第2轮·再润色", "polish", 1024),
             ("第3轮·审校", "proofread", 768)]
    rows = []
    cur = text
    for name, task, mt in chain:
        print("  %s | 输入 %d 字 ..." % (name, len(cur)), flush=True)
        r = run_task(base, key, task, cur, max_tokens=mt)
        r["round"] = name
        rows.append(r)
        print("    -> %s | %.1fs | 出 %s 字 | %s" %
              (r["verdict"], r["dt"], r["out_chars"], r["note"]), flush=True)
        # 关键：把本轮输出作为下一轮输入（还原真实迭代）；输出为空则保留上轮原文
        nxt = (r.get("full") or "").strip()
        if nxt:
            cur = nxt
    return rows


# ---------------------------------------------------------------------------
# D 组：并发稳定性（多个文档润色同时提交）
# ---------------------------------------------------------------------------


def group_d(base, key, corpus, out_dir, n=3):
    texts = [corpus[k]["text"] for k in ("build_report", "build_report_v1", "build_report_v2",
                                         "plan_0312_draft", "plan_0312_polish")][:n]
    print("  %d 个文档润色任务并发提交..." % n, flush=True)
    results = [None] * len(texts)
    t0 = time.time()

    def worker(i, t):
        results[i] = run_task(base, key, "polish", t, max_tokens=512)

    ths = [threading.Thread(target=worker, args=(i, t)) for i, t in enumerate(texts)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.time() - t0
    alive = health_ok(base, key)
    recovery = None
    if not alive:
        print("    服务无响应，等待守护进程重启…", flush=True)
        recovery = wait_ready(base, key)
    ok = sum(1 for r in results if r and r["verdict"] == "成功")
    print("    完成 %d/%d，墙钟 %.1fs，服务%s" %
          (ok, len(texts), wall, "存活" if alive else "崩溃(已恢复 %s)" % (
              "%.0fs" % recovery if recovery else "超时")), flush=True)
    return {"concurrency": n, "ok": ok, "wall_s": round(wall, 1),
            "alive_after": alive, "recovery_s": round(recovery, 1) if recovery else None,
            "rows": results}


# ---------------------------------------------------------------------------
# E 组：批量吞吐（10 份文档依次处理）
# ---------------------------------------------------------------------------


def group_e(base, key, corpus, out_dir):
    order = ["build_report_v1", "build_report_v2", "build_report", "plan_0312_draft",
             "plan_0312_polish", "plan_0313_draft", "plan_0313", "plan_0318", "plan_0320"]
    rows = []
    t_all = time.time()
    for i, k in enumerate(order, 1):
        rec = corpus[k]
        r = run_task(base, key, "summarize", rec["text"], max_tokens=384)
        r.update({"seq": i, "source": k, "in_chars": rec["chars"]})
        rows.append(r)
        print("  %2d/%d %-16s %5d字 | %s | %.1fs | 出 %s 字" %
              (i, len(order), k, rec["chars"], r["verdict"], r["dt"], r["out_chars"]), flush=True)
    wall = time.time() - t_all
    return {"count": len(order), "wall_s": round(wall, 1),
            "avg_s": round(wall / len(order), 1), "rows": rows}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("RKLLM_HOST", "192.0.2.10:8080"))
    ap.add_argument("--key", default=os.environ.get("RKLLM_KEY") or os.environ.get("RKLLM_API_KEY", ""))
    ap.add_argument("--only", default="a,b,c,d,e", help="只跑指定组，逗号分隔")
    ap.add_argument("--concurrency", type=int, default=3, help="D 组并发数")
    args = ap.parse_args()

    base = build_base(args.host)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    if not health_ok(base, args.key):
        print("服务不可达，等待重启…")
        if wait_ready(base, args.key) is None:
            print("服务未就绪，退出")
            return 1

    corpus = load_corpus()
    groups = set(x.strip().lower() for x in args.only.split(","))
    result = {"target": base, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "corpus_size": len(corpus)}

    if "a" in groups:
        print("\n########## A 组：串行真实任务 ##########", flush=True)
        result["group_a"] = group_a(base, args.key, corpus, SAMPLES_DIR)
    if "b" in groups:
        print("\n########## B 组：分段处理 ##########", flush=True)
        result["group_b"] = group_b(base, args.key, corpus, SAMPLES_DIR)
    if "c" in groups:
        print("\n########## C 组：多轮迭代 ##########", flush=True)
        result["group_c"] = group_c(base, args.key, corpus, SAMPLES_DIR)
    if "d" in groups:
        print("\n########## D 组：并发稳定性 ##########", flush=True)
        result["group_d"] = group_d(base, args.key, corpus, SAMPLES_DIR, n=args.concurrency)
    if "e" in groups:
        print("\n########## E 组：批量吞吐 ##########", flush=True)
        result["group_e"] = group_e(base, args.key, corpus, SAMPLES_DIR)

    # 落盘前剥离内存中的完整输出，避免结果 JSON 过大
    def strip_full(o):
        if isinstance(o, dict):
            o.pop("full", None)
            for v in o.values():
                strip_full(v)
        elif isinstance(o, list):
            for v in o:
                strip_full(v)

    strip_full(result)
    out = os.path.join(RESULTS_DIR, "realworld_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print("\n结果写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
