#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_rw_report.py —— 真实公文场景测试数据格式化器。

职责边界
--------
本脚本**只把 results/realworld_*.json 里的测试数据格式化为可读表格**，
不下任何结论、不做可用性判断、不给工程建议。结论由阅读报告的人（或 Agent）
基于表中数据自行做出。

输入
----
  results/realworld_*.json   由 llm_realworld.py 产生的分组结果
                             （按文件时间排序，每个分组取最新且非空的一份）
  corpus/corpus.json         语料画像（只用于展示语料清单）

输出
----
  report_realworld.html      数据表（HTML）
  report_realworld.md        数据表（Markdown）

用法
----
  python make_rw_report.py

模型名 / 版本 / 上下文窗口 / NPU 均由测试脚本写入结果 JSON 的 `meta` 字段
（采集侧调 `board_api.fetch_server_meta()` 从板端 /health 自动读取），
本脚本不在运行时联网、不写死任何型号或版本；必要时可用 --model 覆盖，
或设 RKLLM_MODEL 环境变量。读不到时显示 unknown。
内网地址在报告中一律替换为 RFC5737 文档段，避免真实拓扑进入仓库。
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CORPUS_PATH = os.path.join(HERE, "corpus", "corpus.json")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "realworld_samples")


class Raw(str):
    """标记为原样 HTML 的字符串，table() 遇到时不转义。"""
    pass


def esc(x):
    return html.escape(str(x if x is not None else ""))


def fmt(x, digits=1):
    """数字格式化；None / 空值显示 -。"""
    if x is None or x == "":
        return "-"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, (int, float)):
        return ("%%.%df" % digits) % x if isinstance(x, float) else format(x, ",")
    return str(x)


def redact(value):
    """内网地址 -> RFC5737 文档段，避免真实拓扑进入报告文件。

    先匹配完整四段 IPv4 再判断网段，避免只替换前三段后把剩余一段
    留在文档段地址末尾（曾产生形如 <文档段>.<残留段> 的错误结果）。
    """
    s = str(value or "-")

    def _repl(m):
        octets = [int(x) for x in m.group(0).split(".")]
        a, b = octets[0], octets[1]
        if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
            return "192.0.2.10"
        return m.group(0)

    return re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", _repl, s)


def table(headers, rows):
    out = ["<table><thead><tr>"]
    out += ["<th>%s</th>" % esc(h) for h in headers]
    out += ["</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>")
        for c in row:
            if c is None:
                out.append("<td>-</td>")
            elif isinstance(c, Raw):
                out.append("<td>%s</td>" % str(c))
            else:
                out.append("<td>%s</td>" % esc(c))
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") if c is not None else "-"
                 for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 数据加载（纯合并，不加工语义）
# ---------------------------------------------------------------------------

def load_merged():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "realworld_*.json")))
    merged, meta, sources = {}, {}, []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        sources.append(os.path.basename(f))
        for k, v in d.items():
            if k.startswith("group_"):
                if v:
                    merged[k] = v
            elif k in ("target", "generated", "model", "release", "corpus_size", "meta"):
                meta[k] = v
    merged.update(meta)
    merged["_sources"] = sources
    return merged


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        return []
    try:
        with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh).get("files", [])
    except (OSError, ValueError):
        return []


def count_crashes(data):
    """结果里记录到的崩溃重试次数（数据汇总，非判断）。"""
    n = 0
    for r in (data.get("group_a") or []):
        n += r.get("crashes") or 0
    for r in (data.get("group_c") or []):
        n += r.get("crashes") or 0
    for grp in (data.get("group_b"), data.get("group_e")):
        for r in ((grp or {}).get("rows") or []):
            n += r.get("crashes") or 0
    for r in (((data.get("group_d") or {}).get("rows")) or []):
        if r:
            n += r.get("crashes") or 0
    return n


def server_meta(data):
    """从结果里取采集时记录的板端元信息（不联网、不猜）。"""
    m = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    return {
        "model": m.get("model") or data.get("model") or "unknown",
        "version": m.get("version") or data.get("release") or "unknown",
        "context_len": m.get("context_len"),
        "npu": m.get("npu") or "",
    }


def model_name(data, override=""):
    return override or os.environ.get("RKLLM_MODEL") or server_meta(data)["model"]


# ---------------------------------------------------------------------------
# 行数据（只取字段，不解释）
# ---------------------------------------------------------------------------

TASK_HEADERS = ["#", "任务/来源", "类型", "输入字数", "输出字数", "输出tok",
                "端到端(s)", "判定", "备注"]


def task_rows(rows, label_key=None):
    out = []
    for i, r in enumerate(rows or [], 1):
        if not r:
            out.append([i, "-", "-", "-", "-", "-", "-", "-", "无返回"])
            continue
        label = r.get(label_key) if label_key else None
        if not label:
            label = r.get("tag") or r.get("round") or r.get("source") or r.get("seq") or i
        out.append([label, r.get("desc") or r.get("task") or "-", r.get("task") or "-",
                    fmt(r.get("in_chars")), fmt(r.get("out_chars")), fmt(r.get("out_tok")),
                    fmt(r.get("dt")), r.get("verdict") or "-", r.get("note") or "-"])
    return out


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

STYLE = """
:root{--bg:#f6f7f9;--card:#fff;--line:#e5e7eb;--ink:#1f2328;--sub:#5b6470;--accent:#1a56db}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 64px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:20px;margin:34px 0 12px;padding-left:10px;border-left:4px solid var(--accent)}
.sub{color:var(--sub);font-size:14px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.k{color:var(--sub);font-size:13px;margin-bottom:4px}.v{font-size:20px;font-weight:600}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px;background:#fff}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f0f2f5;font-weight:600}tr:nth-child(even) td{background:#fafbfc}
code{font-family:ui-monospace,Consolas,monospace}
footer{margin-top:38px;padding-top:14px;border-top:1px solid var(--line);color:var(--sub);font-size:13px}
"""


def build_html(data, corpus, model):
    ga = data.get("group_a") or []
    gb = data.get("group_b") or {}
    gc = data.get("group_c") or []
    gd = data.get("group_d") or {}
    ge = data.get("group_e") or {}
    P = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>板端推理服务 · 真实公文场景测试数据</title>",
         "<style>%s</style></head><body><div class='wrap'>" % STYLE,
         "<h1>板端推理服务 · 真实公文场景测试数据</h1>",
         "<div class='sub'>目标：<code>%s</code> ｜ 模型：<code>%s</code> ｜ 版本：<code>%s</code> ｜ 上下文：<code>%s</code> ｜ 生成时间：%s</div>"
         % (esc(redact(data.get("target"))), esc(model),
            esc(server_meta(data)["version"]), fmt(server_meta(data)["context_len"]),
            esc(data.get("generated") or time.strftime("%Y-%m-%d %H:%M:%S")))]

    P.append("<h2>一、数据概览</h2><div class='grid'>")
    for k, v in (("语料份数", len(corpus)),
                 ("语料总字数", format(sum(c.get("chars", 0) for c in corpus), ",")),
                 ("A 组记录", len(ga)), ("B 组分段", gb.get("chunks") or 0),
                 ("C 组轮次", len(gc)), ("D 组并发", gd.get("concurrency") or 0),
                 ("E 组份数", ge.get("count") or 0),
                 ("崩溃重试次数", count_crashes(data))):
        P.append("<div class='card'><div class='k'>%s</div><div class='v'>%s</div></div>" % (k, v))
    P.append("</div>")

    P.append("<h2>二、测试语料</h2>")
    P.append(table(["语料 ID", "业务分组", "阶段", "字符数", "段落", "≈tokens"],
                   [[c.get("key"), c.get("group"), c.get("stage"),
                     fmt(c.get("chars")), fmt(c.get("paras")), fmt(c.get("est_tokens"))]
                    for c in corpus]) or "<p>（无语料数据）</p>")

    P.append("<h2>三、A 组：串行真实任务</h2>")
    P.append(table(TASK_HEADERS, task_rows(ga)) if ga else "<p>（无 A 组数据）</p>")

    P.append("<h2>四、B 组：分段处理</h2>")
    if gb:
        P.append("<div class='card'>分段数 %s ｜ 记录数 %s ｜ 总耗时 %s s ｜ 墙钟 %s s</div>"
                 % (fmt(gb.get("chunks")), len(gb.get("rows") or []),
                    fmt(gb.get("total_s")), fmt(gb.get("wall_s"))))
        P.append(table(["段号", "输入字数", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                       [[r.get("seq"), fmt(r.get("in_chars")), fmt(r.get("out_chars")),
                         fmt(r.get("out_tok")), fmt(r.get("dt")), r.get("verdict") or "-",
                         r.get("note") or "-"] for r in (gb.get("rows") or [])]))
    else:
        P.append("<p>（无 B 组数据）</p>")

    P.append("<h2>五、C 组：多轮迭代</h2>")
    P.append(table(["轮次", "任务", "输入字数", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                   [[r.get("round"), r.get("task"), fmt(r.get("in_chars")),
                     fmt(r.get("out_chars")), fmt(r.get("out_tok")), fmt(r.get("dt")),
                     r.get("verdict") or "-", r.get("note") or "-"] for r in gc])
            if gc else "<p>（无 C 组数据）</p>")

    P.append("<h2>六、D 组：并发稳定性</h2>")
    if gd:
        P.append("<div class='card'>并发数 %s ｜ 成功 %s ｜ 墙钟 %s s ｜ 服务存活 %s ｜ 恢复耗时 %s</div>"
                 % (fmt(gd.get("concurrency")), fmt(gd.get("ok")), fmt(gd.get("wall_s")),
                    fmt(gd.get("alive_after")), fmt(gd.get("recovery_s"))))
        P.append(table(["#", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                       [[i, fmt(r.get("out_chars")) if r else "-",
                         fmt(r.get("out_tok")) if r else "-",
                         fmt(r.get("dt")) if r else "-",
                         (r.get("verdict") if r else None) or "-",
                         (r.get("note") if r else None) or "无返回"]
                        for i, r in enumerate(gd.get("rows") or [], 1)]))
    else:
        P.append("<p>（无 D 组数据）</p>")

    P.append("<h2>七、E 组：批量吞吐</h2>")
    if ge:
        P.append("<div class='card'>份数 %s ｜ 总耗时 %s s ｜ 平均 %s s/份</div>"
                 % (fmt(ge.get("count")), fmt(ge.get("wall_s")), fmt(ge.get("avg_s"))))
        P.append(table(["#", "语料", "输入字数", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                       [[r.get("seq"), r.get("source"), fmt(r.get("in_chars")),
                         fmt(r.get("out_chars")), fmt(r.get("out_tok")), fmt(r.get("dt")),
                         r.get("verdict") or "-", r.get("note") or "-"]
                        for r in (ge.get("rows") or [])]))
    else:
        P.append("<p>（无 E 组数据）</p>")

    P.append("<h2>八、结果文件</h2>")
    P.append("<div class='card'>合并来源：<code>%s</code><br>完整输出样本目录：<code>%s</code></div>"
             % (esc(", ".join(data.get("_sources") or []) or "-"),
                esc(os.path.relpath(SAMPLES_DIR, HERE))))
    P.append("<footer>本页仅格式化展示测试脚本产生的原始数据，不含结论与建议。"
             "判定/备注字段为测试脚本的原始输出，未做二次解释。</footer>")
    P.append("</div></body></html>")
    return "\n".join(P)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_md(data, corpus, model):
    ga = data.get("group_a") or []
    gb = data.get("group_b") or {}
    gc = data.get("group_c") or []
    gd = data.get("group_d") or {}
    ge = data.get("group_e") or {}
    L = ["# 板端推理服务 · 真实公文场景测试数据", "",
         "- 目标：`%s`" % redact(data.get("target")),
         "- 模型：`%s`" % model,
         "- 版本：`%s`" % server_meta(data)["version"],
         "- 上下文窗口：`%s`" % fmt(server_meta(data)["context_len"]),
         "- NPU：`%s`" % (server_meta(data)["npu"] or "-"),
         "- 生成时间：%s" % (data.get("generated") or "-"),
         "- 结果来源：`%s`" % (", ".join(data.get("_sources") or []) or "-"), "",
         "> 本文仅格式化展示测试脚本产生的原始数据，不含结论与建议；"
         "判定/备注字段为脚本原始输出，未做二次解释。", "",
         "## 一、数据概览", "",
         md_table(["项", "值"],
                  [["语料份数", len(corpus)],
                   ["语料总字数", sum(c.get("chars", 0) for c in corpus)],
                   ["A 组记录", len(ga)], ["B 组分段", gb.get("chunks") or 0],
                   ["C 组轮次", len(gc)], ["D 组并发", gd.get("concurrency") or 0],
                   ["E 组份数", ge.get("count") or 0],
                   ["崩溃重试次数", count_crashes(data)]]), "",
         "## 二、测试语料", "",
         md_table(["语料 ID", "业务分组", "阶段", "字符数", "段落", "≈tokens"],
                  [[c.get("key"), c.get("group"), c.get("stage"), c.get("chars"),
                    c.get("paras"), c.get("est_tokens")] for c in corpus]), "",
         "## 三、A 组：串行真实任务", ""]
    L.append(md_table(TASK_HEADERS, task_rows(ga)) if ga else "（无 A 组数据）")
    L += ["", "## 四、B 组：分段处理", ""]
    if gb:
        L.append("- 分段数 %s ｜ 记录数 %s ｜ 总耗时 %s s ｜ 墙钟 %s s"
                 % (fmt(gb.get("chunks")), len(gb.get("rows") or []),
                    fmt(gb.get("total_s")), fmt(gb.get("wall_s"))))
        L.append("")
        L.append(md_table(["段号", "输入字数", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                          [[r.get("seq"), r.get("in_chars"), r.get("out_chars"),
                            r.get("out_tok"), r.get("dt"), r.get("verdict"), r.get("note")]
                           for r in (gb.get("rows") or [])]))
    else:
        L.append("（无 B 组数据）")
    L += ["", "## 五、C 组：多轮迭代", ""]
    L.append(md_table(["轮次", "任务", "输入字数", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                      [[r.get("round"), r.get("task"), r.get("in_chars"), r.get("out_chars"),
                        r.get("out_tok"), r.get("dt"), r.get("verdict"), r.get("note")]
                       for r in gc]) if gc else "（无 C 组数据）")
    L += ["", "## 六、D 组：并发稳定性", ""]
    if gd:
        L.append("- 并发数 %s ｜ 成功 %s ｜ 墙钟 %s s ｜ 服务存活 %s ｜ 恢复耗时 %s"
                 % (fmt(gd.get("concurrency")), fmt(gd.get("ok")), fmt(gd.get("wall_s")),
                    fmt(gd.get("alive_after")), fmt(gd.get("recovery_s"))))
        L.append("")
        L.append(md_table(["#", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                          [[i, r.get("out_chars") if r else "-", r.get("out_tok") if r else "-",
                            r.get("dt") if r else "-", (r.get("verdict") if r else None) or "-",
                            (r.get("note") if r else None) or "无返回"]
                           for i, r in enumerate(gd.get("rows") or [], 1)]))
    else:
        L.append("（无 D 组数据）")
    L += ["", "## 七、E 组：批量吞吐", ""]
    if ge:
        L.append("- 份数 %s ｜ 总耗时 %s s ｜ 平均 %s s/份"
                 % (ge.get("count"), ge.get("wall_s"), ge.get("avg_s")))
        L.append("")
        L.append(md_table(["#", "语料", "输入字数", "输出字数", "输出tok", "耗时(s)", "判定", "备注"],
                          [[r.get("seq"), r.get("source"), r.get("in_chars"),
                            r.get("out_chars"), r.get("out_tok"), r.get("dt"),
                            r.get("verdict"), r.get("note")]
                           for r in (ge.get("rows") or [])]))
    else:
        L.append("（无 E 组数据）")
    L += ["", "## 八、结果文件", "",
          "- 合并来源：`%s`" % (", ".join(data.get("_sources") or []) or "-"),
          "- 完整输出样本：`%s`" % os.path.relpath(SAMPLES_DIR, HERE), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="被测模型名（也可由 RKLLM_MODEL 或结果 JSON 提供）")
    args = ap.parse_args()
    data = load_merged()
    corpus = load_corpus()
    model = model_name(data, args.model)
    html_path = os.path.join(HERE, "report_realworld.html")
    md_path = os.path.join(HERE, "report_realworld.md")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(build_html(data, corpus, model))
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_md(data, corpus, model))
    print("model = %s" % model)
    print("wrote %s (%d bytes)" % (html_path, os.path.getsize(html_path)))
    print("wrote %s (%d bytes)" % (md_path, os.path.getsize(md_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
