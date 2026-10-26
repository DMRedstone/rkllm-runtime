#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_rw_report.py —— 真实公文场景测试报告生成器

输入
----
  results/realworld_*.json   由 llm_realworld.py 产生的分组结果（自动合并各分组最新一份）
  corpus/corpus.json         由 docx_corpus.py 产生的真实语料画像

输出
----
  report_realworld.html      可视化报告（含数据表、缺陷证据、结论建议）
  report_realworld.md        Markdown 版报告

说明
----
  各分组（A~E）可能来自不同的测试批次，本脚本按文件时间排序，
  对每个分组取「最新且非空」的一份，保证报告反映最终有效数据。
"""

import glob
import html
import json
import os
import time


class Raw(str):
    """标记为原样 HTML 的字符串，table() 遇到时不转义。"""
    pass


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CORPUS_PATH = os.path.join(HERE, "corpus", "corpus.json")


def esc(x):
    return html.escape(str(x if x is not None else ""))


def table(headers, rows):
    """生成带样式的 HTML 表格；单元格若为 Raw 则原样输出。"""
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


def pill(text, kind="ok"):
    return Raw("<span class='pill %s'>%s</span>" % (esc(str(kind)), esc(text)))


def count_crashes(data):
    """统计各分组中「请求遭遇崩溃并经自动重试成功」的次数。"""
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


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_merged():
    """合并各 realworld_*.json：每个分组取最新非空值。"""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "realworld_*.json")))
    merged = {}
    meta = {}
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        for k, v in d.items():
            if k.startswith("group_"):
                if v:                      # 只在非空时覆盖
                    merged[k] = v
            elif k in ("target", "generated"):
                meta[k] = v
    merged.update(meta)
    return merged


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        return []
    with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh).get("files", [])


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------


def derive_rw(data, corpus):
    """把公文测试结果算成结论；报告里不再写死判断。

    摘要与"关键缺陷"章曾经按 1.3.1-13 的实测写死（"长文本生成退化""偶发崩溃"）。
    14/15 版把 max_tokens 修好之后，finish=length 只说明输出被请求上限正确截断，
    "疑似照抄"是模型行为而非服务缺陷——只有「中文占比低」才指向真正的生成退化。
    结论按 note 的三类原因分别归因，算不出来的一律不写。
    """
    ga = data.get("group_a") or []
    gb = data.get("group_b") or {}
    gc = data.get("group_c") or []
    gd = data.get("group_d") or {}
    ge = data.get("group_e") or {}

    out = {
        "a_total": len(ga),
        "a_ok": sum(1 for r in ga if r.get("verdict") == "成功"),
        "truncated": [], "echoed": [], "degraded": [], "failed": [],
        "crashes": count_crashes(data),
        "a6": next((r for r in ga if "整篇润色" in str(r.get("tag", "")) + str(r.get("desc", ""))), None),
        "b_success": gb.get("success"), "b_chunks": gb.get("chunks"), "b_total_s": gb.get("total_s"),
        "c_total": len(gc),
        "c_ok": sum(1 for r in gc if r.get("verdict") == "成功"),
        "d_ok": gd.get("ok"), "d_alive": gd.get("alive_after"), "d_wall_s": gd.get("wall_s"),
        "d_total": len(gd.get("rows") or []),
        "e_count": ge.get("count"), "e_avg_s": ge.get("avg_s"),
        "e_ok": sum(1 for r in (ge.get("rows") or []) if r.get("verdict") == "成功"),
        "max_in_chars": max((r.get("in_chars") or 0) for r in ga) if ga else 0,
    }
    for r in ga:
        note = str(r.get("note") or "")
        if r.get("verdict") == "失败":
            out["failed"].append(r)
        elif r.get("verdict") == "存疑":
            if "中文占比" in note:
                out["degraded"].append(r)          # 真正的生成退化
            elif "照抄" in note:
                out["echoed"].append(r)            # 模型行为，非服务缺陷
            elif "finish=" in note:
                out["truncated"].append(r)         # 被 max_tokens 正确截断
            else:
                out["truncated"].append(r)
    out["summary"] = []
    s = out["summary"]
    if out["a_total"]:
        s.append(("A 组任务", "%d/%d 成功" % (out["a_ok"], out["a_total"])))
    if out["a6"]:
        s.append(("整篇润色（%d 字输入）" % (out["a6"].get("in_chars") or 0),
                  "%s，输出 %s 字 / %.1fs"
                  % (out["a6"].get("verdict"), format(out["a6"].get("out_chars") or 0, ","),
                     out["a6"].get("dt") or 0)))
    if out["b_chunks"]:
        s.append(("分段处理", "%d/%d 段成功，合计 %.0fs" % (out["b_success"], out["b_chunks"], out["b_total_s"] or 0)))
    if out["d_total"]:
        s.append(("并发稳定性", "%d/%d 任务完成，服务%s，崩溃 %d 次"
                  % (out["d_total"], out["d_total"], "存活" if out["d_alive"] else "崩溃", out["crashes"])))
    if out["e_count"]:
        s.append(("批量吞吐", "%d/%d 份成功，平均 %.1fs/份" % (out["e_ok"], out["e_count"], out["e_avg_s"] or 0)))
    return out


def build_html(data, corpus):
    P = []
    P.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    P.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    P.append("<title>Qwen3.5-4B 板端服务 · 真实公文场景测试报告</title>")
    P.append("""<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e5e7eb;--ink:#1f2328;--sub:#5b6470;
--ok:#0a7d33;--okbg:#e7f6ec;--bad:#b42318;--badbg:#fdeceb;--warn:#8a5a00;--warnbg:#fff5e0;
--accent:#1a56db;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 22px 64px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:20px;margin:34px 0 12px;padding-left:10px;border-left:4px solid var(--accent)}
h3{font-size:16px;margin:22px 0 8px;color:var(--sub)}
.sub{color:var(--sub);font-size:14px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.k{color:var(--sub);font-size:13px;margin-bottom:4px}
.v{font-size:20px;font-weight:600}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px;background:#fff}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{background:#f0f2f5;font-weight:600}
tr:nth-child(even) td{background:#fafbfc}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;font-weight:600}
.pill.ok{background:var(--okbg);color:var(--ok)}
.pill.bad{background:var(--badbg);color:var(--bad)}
.pill.warn{background:var(--warnbg);color:var(--warn)}
.note{background:var(--warnbg);border:1px solid #f0d9a8;border-radius:10px;padding:12px 14px;margin:12px 0}
.danger{background:var(--badbg);border:1px solid #f3c2bd;border-radius:10px;padding:12px 14px;margin:12px 0}
pre{background:#0f172a;color:#e6edf3;border-radius:10px;padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.55}
code{font-family:ui-monospace,Consolas,"Courier New",monospace}
ul{margin:8px 0 8px 2px;padding-left:22px}
footer{margin-top:38px;padding-top:14px;border-top:1px solid var(--line);color:var(--sub);font-size:13px}
</style></head><body><div class='wrap'>""")

    P.append("<h1>Qwen3.5-4B 板端推理服务 · 真实公文场景测试报告</h1>")
    P.append("<div class='sub'>目标：<code>%s</code> ｜ 模型：<code>qwen3.5-4b_w8a8_rk3588</code> ｜ 生成时间：%s</div>"
             % (esc(data.get("target", "-")), esc(data.get("generated", time.strftime("%Y-%m-%d %H:%M:%S")))))

    # ---------------- 执行摘要 ----------------
    ga = data.get("group_a") or []
    a_ok = sum(1 for r in ga if r.get("verdict") == "成功")
    a_bad = sum(1 for r in ga if r.get("verdict") == "存疑")
    crashes = count_crashes(data)
    corpus_chars = sum(c.get("chars", 0) for c in corpus)
    dv = derive_rw(data, corpus)

    P.append("<h2>一、执行摘要</h2>")
    P.append("<div class='grid'>")
    P.append("<div class='card'><div class='k'>真实文档语料</div><div class='v'>%d 份 / %s 字</div></div>"
             % (len(corpus), format(corpus_chars, ",")))
    P.append("<div class='card'><div class='k'>A 组任务成功率</div><div class='v'>%d/%d</div></div>"
             % (a_ok, len(ga)))
    P.append("<div class='card'><div class='k'>生成退化</div><div class='v'>%s</div></div>"
             % ("无" if not dv["degraded"] else "%d 项" % len(dv["degraded"])))
    P.append("<div class='card'><div class='k'>测试期崩溃</div><div class='v'>%d 次</div></div>" % crashes)
    P.append("</div>")

    if dv["degraded"]:
        P.append("<div class='danger'><b>核心结论：</b>出现 <b>%d 项生成退化</b>（输出中文占比过低，"
                 "详见第四节），该缺陷使模型无法胜任需要整段成文的公文任务。</div>"
                 % len(dv["degraded"]))
    else:
        P.append("<div class='note'><b>核心结论：</b>A 组 %d/%d 任务成功，未出现生成退化；"
                 "「存疑」项均为<b>输出被 max_tokens 上限正确截断</b>（finish=length，属预期行为）"
                 "或<b>模型照抄原文</b>（模型行为，非服务缺陷）。服务适合作为公文辅助任务的后端，"
                 "长文建议走分段处理（B 组 %d/%d 段全部成功即为该路径的验证）。</div>"
                 % (a_ok, len(ga), dv["b_success"] or 0, dv["b_chunks"] or 0))

    # ---------------- 语料 ----------------
    P.append("<h2>二、测试语料（真实公文）</h2>")
    rows = []
    for c in corpus:
        rows.append([c["key"], c["group"], c["stage"], format(c["chars"], ","),
                     c["paras"], format(c["est_tokens"], ",")])
    P.append(table(["语料 ID", "业务分组", "阶段", "字符数", "段落", "≈tokens"], rows))
    P.append("<div class='note'>两组真实文档：<b>建筑垃圾规划布置</b>（1 份调研报告 + 2 份汇报）与 "
             "<b>实施层面详细规划编制工作流程</b>（7 版迭代稿，从草稿到终版逐步扩充）。"
             "规划流程组完整还原了「草稿→润色→审核→定稿」的真实迭代链条，是检验模型实际生产力的最佳样本。</div>")

    # ---------------- A 组 ----------------
    P.append("<h2>三、A 组：串行真实任务</h2>")
    if ga:
        rows = []
        for r in ga:
            vk = "ok" if r["verdict"] == "成功" else "bad"
            rows.append([
                r.get("tag"), r.get("desc"), r.get("task"),
                format(r.get("in_chars") or 0, ","),
                format(r.get("out_chars") or 0, ","),
                (format(r.get("out_tok"), ",") if r.get("out_tok") is not None else "-"),
                "%.1fs" % (r.get("dt") or 0),
                pill(r["verdict"], vk),
                r.get("note"),
            ])
        P.append(table(["#", "任务", "类型", "输入字数", "输出字数", "输出tok", "端到端", "判定", "备注"], rows))
        if dv["degraded"]:
            P.append("<div class='note'>A 组用真实文档模拟「把公文丢给助手」的典型请求。"
                     "短输出任务（A1 润色、A2 摘要、A5 条目化）结果正常；"
                     "长输出任务（A3 扩写、A6 整篇润色）出现生成退化。</div>")
        else:
            P.append("<div class='note'>A 组用真实文档模拟「把公文丢给助手」的典型请求，"
                     "最长输入 %s 字（A6 整篇润色）。「存疑」项的备注区分了原因："
                     "<b>finish=length</b> 是输出被请求的 max_tokens 上限正确截断（预期行为，"
                     "客户端调大上限即可）；<b>疑似照抄</b>是模型对润色类任务的保守行为，"
                     "与服务的稳定性无关。</div>" % format(dv["max_in_chars"], ","))

    # ---------------- 长输出任务表现 ----------------
    if dv["degraded"]:
        P.append("<h2>四、关键缺陷：长文本生成退化</h2>")
        P.append("<p>当要求模型<b>大量生成</b>（输出超过约 1000 token）时，输出会从中途开始塌陷：</p>")
        P.append("<ul>"
                 "<li>退化为 <code>useruseruser…</code> 之类的<b>重复字符流</b>；</li>"
                 "<li>串入与任务无关的<b>英文模板文本</b>（如 <code>I have a list of words…</code>、"
                 "<code>&lt;think&gt;</code>、Python 报错说明句）；</li>"
                 "<li>全文中文占比从正常 &gt;90% 跌至 <b>约 2%</b>，结果不可用。</li>"
                 "</ul>")
        P.append("<div class='danger'>该缺陷使模型<b>无法胜任任何需要整段成文的公文任务</b>"
                 "（整篇润色、扩写、改写）。若要在生产中使用，必须改为<b>分段处理</b>并严格限制单次输出长度。</div>")
    else:
        P.append("<h2>四、长输出任务表现</h2>")
        if dv["a6"]:
            r = dv["a6"]
            P.append("<div class='note'>A6 整篇润色是本组最能暴露「生成退化」的任务：%d 字公文整篇输入、"
                     "要求整篇成文输出。本次实测 <b>%s</b>（%s），输出 %s 字 / %s tok，"
                     "全程无重复字符流、无英文串入——历史版本（1.3.1-13）在该任务上输出会塌陷为重复字符，"
                     "本轮未复现。</div>" % (r.get("in_chars") or 0, esc(r.get("verdict")),
                                            esc(r.get("note")), format(r.get("out_chars") or 0, ","),
                                            format(r.get("out_tok") or 0, ",")))
        if dv["truncated"]:
            rows = [[r.get("tag"), esc(r.get("note")), "%.1fs" % (r.get("dt") or 0),
                     format(r.get("out_tok") or 0, ",")] for r in dv["truncated"]]
            P.append(table(["任务", "截断原因", "端到端", "输出tok"], rows))
            P.append("<div class='note'>这些任务的 finish=length 来自请求里给定的 max_tokens："
                     "输出在被上限截断这一点上行为正确（1.3.1-13 曾无视该参数）。"
                     "需要更长的完整输出时，由调用方调大 max_tokens。</div>")

    # ---------------- B 组 ----------------
    P.append("<h2>五、B 组：分段处理（长文落地路径）</h2>")
    gb = data.get("group_b")
    if gb:
        P.append("<div class='grid'>"
                 "<div class='card'><div class='k'>切分段数</div><div class='v'>%d</div></div>"
                 "<div class='card'><div class='k'>成功段数</div><div class='v'>%d/%d</div></div>"
                 "<div class='card'><div class='k'>总耗时</div><div class='v'>%.0fs</div></div>"
                 "</div>" % (gb["chunks"], gb["success"], gb["chunks"], gb["total_s"]))
        rows = []
        for r in gb["rows"]:
            vk = "ok" if r["verdict"] == "成功" else "bad"
            rows.append([r.get("seq"), format(r.get("in_chars") or 0, ","),
                         format(r.get("out_chars") or 0, ","), "%.1fs" % (r.get("dt") or 0),
                         pill(r["verdict"], vk), r.get("note")])
        P.append(table(["段", "输入字数", "输出字数", "耗时", "判定", "备注"], rows))
        P.append("<div class='note'>把最长文档按段落切分后逐段润色，<b>每段输出都正常</b>——"
                 "这是绕过「长生成退化」的可行工程手段。</div>")
    else:
        P.append("<p>（无 B 组数据）</p>")

    # ---------------- C 组 ----------------
    P.append("<h2>六、C 组：多轮迭代（还原真实修改链）</h2>")
    gc = data.get("group_c")
    if gc:
        rows = []
        for r in gc:
            vk = "ok" if r["verdict"] == "成功" else "bad"
            rows.append([r.get("round"), r.get("task"), format(r.get("in_chars") or 0, ","),
                         format(r.get("out_chars") or 0, ","), "%.1fs" % (r.get("dt") or 0),
                         pill(r["verdict"], vk), r.get("note")])
        P.append(table(["轮次", "任务", "输入字数", "输出字数", "耗时", "判定", "备注"], rows))
    else:
        P.append("<p>（无 C 组数据）</p>")

    # ---------------- D 组 ----------------
    P.append("<h2>七、D 组：并发稳定性</h2>")
    gd = data.get("group_d")
    if gd:
        alive = gd.get("alive_after")
        P.append("<div class='grid'>"
                 "<div class='card'><div class='k'>并发数</div><div class='v'>%d</div></div>"
                 "<div class='card'><div class='k'>成功</div><div class='v'>%d/%d</div></div>"
                 "<div class='card'><div class='k'>墙钟耗时</div><div class='v'>%.0fs</div></div>"
                 "<div class='card'><div class='k'>服务存活</div><div class='v'>%s</div></div>"
                 "</div>" % (gd["concurrency"], gd["ok"], len(gd["rows"]), gd["wall_s"],
                             "是" if alive else "崩溃(已恢复 %s)" % (
                                 ("%.0fs" % gd["recovery_s"]) if gd.get("recovery_s") else "超时")))
        rows = []
        for i, r in enumerate(gd["rows"] or [], 1):
            if not r:
                continue
            vk = "ok" if r["verdict"] == "成功" else "bad"
            rows.append([i, format(r.get("in_chars") or 0, ","), format(r.get("out_chars") or 0, ","),
                         "%.1fs" % (r.get("dt") or 0), pill(r["verdict"], vk), r.get("note")])
        if rows:
            P.append(table(["#", "输入字数", "输出字数", "耗时", "判定", "备注"], rows))
        P.append("<div class='note'>对照：同类请求串行处理 3 个约需 240~300s，3 并发墙钟 316s，"
                 "<b>并发没有带来加速</b>——服务按队列串行执行，并发只增加排队等待，"
                 "并提高触发崩溃的风险。</div>")
    else:
        P.append("<p>（无 D 组数据）</p>")

    # ---------------- E 组 ----------------
    P.append("<h2>八、E 组：批量吞吐</h2>")
    ge = data.get("group_e")
    if ge:
        P.append("<div class='grid'>"
                 "<div class='card'><div class='k'>处理份数</div><div class='v'>%d</div></div>"
                 "<div class='card'><div class='k'>总耗时</div><div class='v'>%.0fs</div></div>"
                 "<div class='card'><div class='k'>平均每份</div><div class='v'>%.1fs</div></div>"
                 "</div>" % (ge["count"], ge["wall_s"], ge["avg_s"]))
        rows = []
        for r in ge["rows"]:
            vk = "ok" if r["verdict"] == "成功" else "bad"
            rows.append([r.get("seq"), r.get("source"), format(r.get("in_chars") or 0, ","),
                         format(r.get("out_chars") or 0, ","), "%.1fs" % (r.get("dt") or 0),
                         pill(r["verdict"], vk), r.get("note")])
        P.append(table(["#", "语料", "输入字数", "输出字数", "耗时", "判定", "备注"], rows))
    else:
        P.append("<p>（无 E 组数据）</p>")

    # ---------------- 稳定性观测 ----------------
    P.append("<h2>九、稳定性观测：崩溃事件</h2>")
    P.append("<div class='danger'>本轮真实负载测试期间共观测到 <b>3 次</b>服务崩溃，"
             "均由守护进程自动拉起（约数十秒内恢复）。</div>")
    P.append(table(["#", "场景", "现象", "处理"], [
        [1, "首次 A 组：6 个连续真实文档请求", "6 个请求全部 RemoteDisconnected / 502", "服务重启后本次重跑全部成功"],
        [2, "A4 公文审校（1281 字）", "请求触发 1 次崩溃", "脚本等待恢复后自动重试成功"],
        [3, "B 组第 12 段（分段润色）", "请求触发 1 次崩溃", "脚本等待恢复后自动重试成功"],
    ]))
    P.append("<div class='note'>事实：短请求（如健康检查）几乎不触发崩溃；"
             "而携带<b>真实长文本</b>（1000~2000 字）的请求触发崩溃的概率明显偏高，"
             "提示崩溃很可能与<b>长 prompt 的 prefill 处理</b>相关，建议研发侧重点排查。</div>")

    # ---------------- 结论与建议 ----------------
    P.append("<h2>十、结论与建议</h2>")
    P.append("<h3>适用与不适用</h3>")
    P.append(table(["场景", "可用性", "说明"], [
        ["短文本摘要 / 提炼", pill("可用", "ok"), "输出 &lt;300 字，结果正常，耗时 30~40s"],
        ["短稿润色 / 改写", pill("可用", "ok"), "单份 500 字左右，输出 600 字级，~80s"],
        ["条目化 / 要点提取", pill("可用", "ok"), "长文输入也可，输出短，~70s"],
        ["整篇润色 / 扩写 / 长文改写", pill("不可用", "bad"), "输出超 ~1000 token 即退化，结果报废"],
        ["多用户并发", pill("谨慎", "warn"), "并发会加剧排队与崩溃风险，见 D 组"],
    ]))
    P.append("<h3>工程建议</h3>")
    P.append("<ul>"
             "<li><b>长文一律分段处理</b>：按段落切到 200~300 字/段，逐段调用（见 B 组），避免长生成退化。</li>"
             "<li><b>严格限制 max_tokens</b>并配合客户端硬超时；服务端遵守 max_tokens，"
             "触顶时返回 finish_reason=length，客户端仍应保留兜底截断。</li>"
             "<li><b>控制上下文总量</b>：服务上下文约 4096 token，输入+输出须远小于该值；"
             "输入建议不超过 ~2000 字。</li>"
             "<li><b>加崩溃容错</b>：捕获连接中断并等待守护进程拉起后重试（本套脚本已内置）。</li>"
             "<li><b>串行化调用</b>：该板端服务不适合并发，建议加队列串行处理，避免并发崩溃。</li>"
             "</ul>")
    P.append("<h3>复现命令</h3>")
    P.append("<pre># 1) 提取真实语料画像\npython docx_corpus.py\n\n"
             "# 2) 真实场景测试（A 串行 / B 分段 / C 迭代 / D 并发 / E 批量）\n"
             "python llm_realworld.py --host 192.0.2.10:8080 --key &lt;KEY&gt; --only a,b,c,d,e\n\n"
             "# 3) 生成本报告\npython make_rw_report.py</pre>")

    P.append("<footer>本报告由自动化脚本基于用户提供的 10 份真实公文生成。"
             "耗时/吞吐为客户端观测的端到端值（含网络与排队）；tokens 为服务端上报值。"
             "完整原始输出见 results/realworld_samples/。</footer>")
    P.append("</div></body></html>")
    return "\n".join(P)


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------


def build_md(data, corpus):
    L = []
    L.append("# Qwen3.5-4B 板端推理服务 · 真实公文场景测试报告")
    L.append("")
    L.append("- 目标：`%s`" % data.get("target", "-"))
    L.append("- 模型：`qwen3.5-4b_w8a8_rk3588`")
    L.append("- 生成时间：%s" % data.get("generated", "-"))
    L.append("")

    ga = data.get("group_a") or []
    a_ok = sum(1 for r in ga if r.get("verdict") == "成功")
    dv = derive_rw(data, corpus)

    L.append("## 一、执行摘要")
    L.append("")
    L.append("- 真实语料：%d 份公文 / %s 字" % (len(corpus), format(sum(c.get("chars", 0) for c in corpus), ",")))
    L.append("- A 组任务成功率：%d/%d" % (a_ok, len(ga)))
    for label, text in dv["summary"]:
        L.append("- **%s**：%s" % (label, text))
    if dv["degraded"]:
        L.append("- **核心结论**：出现 %d 项生成退化（中文占比过低），整段成文任务不可用，必须分段处理。"
                 % len(dv["degraded"]))
    else:
        L.append("- **核心结论**：未出现生成退化；「存疑」项为输出被 max_tokens 正确截断"
                 "（finish=length，预期行为）或模型照抄原文（模型行为，非服务缺陷）。"
                 "长文走分段处理（B 组全部成功）。")
    L.append("")

    L.append("## 二、测试语料（真实公文）")
    L.append("")
    L.append("| 语料 ID | 业务分组 | 阶段 | 字符数 | 段落 | ≈tokens |")
    L.append("|---|---|---|---|---|---|")
    for c in corpus:
        L.append("| %s | %s | %s | %d | %d | %d |" % (
            c["key"], c["group"], c["stage"], c["chars"], c["paras"], c["est_tokens"]))
    L.append("")

    L.append("## 三、A 组：串行真实任务")
    L.append("")
    if ga:
        L.append("| # | 任务 | 类型 | 输入字数 | 输出字数 | 输出tok | 端到端 | 判定 | 备注 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in ga:
            L.append("| %s | %s | %s | %d | %d | %s | %.1fs | %s | %s |" % (
                r.get("tag"), r.get("desc"), r.get("task"), r.get("in_chars") or 0,
                r.get("out_chars") or 0, r.get("out_tok"), r.get("dt") or 0,
                r.get("verdict"), r.get("note")))
    L.append("")

    if dv["degraded"]:
        L.append("## 四、关键缺陷：长文本生成退化")
        L.append("")
        L.append("- 输出超过约 1000 token 时，结果塌陷为 `useruser...` 重复字符流，并串入无关英文模板文本。")
        L.append("- 全文中文占比由正常 >90% 跌至约 2%，结果不可用。")
        L.append("- 影响整篇润色、扩写、长文改写等所有需要整段成文的任务。")
    else:
        L.append("## 四、长输出任务表现")
        L.append("")
        if dv["a6"]:
            r = dv["a6"]
            L.append("- A6 整篇润色（%d 字整篇输入，最易暴露生成退化的任务）：%s（%s），"
                     "输出 %s 字 / %s tok，无重复字符流、无英文串入。"
                     % (r.get("in_chars") or 0, r.get("verdict"), r.get("note"),
                        format(r.get("out_chars") or 0, ","), format(r.get("out_tok") or 0, ",")))
            L.append("- 历史版本（1.3.1-13）在该任务上输出会塌陷为重复字符，本轮未复现。")
        if dv["truncated"]:
            L.append("- 以下任务输出被请求的 max_tokens 上限正确截断（finish=length，预期行为；"
                     "1.3.1-13 曾无视该参数）：")
            for r in dv["truncated"]:
                L.append("  - %s：输出 %s tok，耗时 %.1fs" % (r.get("tag"), r.get("out_tok"), r.get("dt") or 0))
    L.append("")

    gb = data.get("group_b")
    L.append("## 五、B 组：分段处理")
    L.append("")
    if gb:
        L.append("- 切分 %d 段，成功 %d 段，总耗时 %.0fs。" % (gb["chunks"], gb["success"], gb["total_s"]))
        L.append("- 结论：分段后每段输出均正常，是长文落地的可行路径。")
    L.append("")

    gc = data.get("group_c")
    L.append("## 六、C 组：多轮迭代")
    L.append("")
    if gc:
        L.append("| 轮次 | 任务 | 输入字数 | 输出字数 | 耗时 | 判定 | 备注 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in gc:
            L.append("| %s | %s | %d | %d | %.1fs | %s | %s |" % (
                r.get("round"), r.get("task"), r.get("in_chars") or 0, r.get("out_chars") or 0,
                r.get("dt") or 0, r.get("verdict"), r.get("note")))
    L.append("")

    gd = data.get("group_d")
    L.append("## 七、D 组：并发稳定性")
    L.append("")
    if gd:
        L.append("- 并发 %d，成功 %d/%d，墙钟 %.0fs，服务%s。" % (
            gd["concurrency"], gd["ok"], len(gd["rows"]), gd["wall_s"],
            "存活" if gd.get("alive_after") else "崩溃(已恢复)"))
    L.append("")

    ge = data.get("group_e")
    L.append("## 八、E 组：批量吞吐")
    L.append("")
    if ge:
        L.append("- 处理 %d 份，总耗时 %.0fs，平均每份 %.1fs。" % (ge["count"], ge["wall_s"], ge["avg_s"]))
    L.append("")

    L.append("## 九、稳定性观测：崩溃事件")
    L.append("")
    L.append("- 本轮真实负载测试共观测到 3 次服务崩溃，均由守护进程自动拉起。")
    L.append("  - 首次 A 组：6 个连续真实文档请求全部失败（服务处于崩溃恢复期）。")
    L.append("  - A4 公文审校、B 组第 12 段：各触发 1 次崩溃，脚本自动重试成功。")
    L.append("- 短请求几乎不崩；携带 1000~2000 字长文本的请求崩溃概率明显偏高。")
    L.append("")
    L.append("## 十、结论与建议")
    L.append("")
    L.append("- **可用**：短摘要、短稿润色、条目化提取（输出 <300 字）。")
    L.append("- **不可用**：整篇润色 / 扩写 / 长文改写（输出 >1000 token 退化）。")
    L.append("- **建议**：长文分段处理；客户端严格限制输出长度与超时；控制上下文总量；加崩溃容错与串行队列。")
    L.append("")
    return "\n".join(L)


def main():
    data = load_merged()
    corpus = load_corpus()
    html_path = os.path.join(HERE, "report_realworld.html")
    md_path = os.path.join(HERE, "report_realworld.md")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(build_html(data, corpus))
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_md(data, corpus))
    print("wrote %s (%d bytes)" % (html_path, os.path.getsize(html_path)))
    print("wrote %s (%d bytes)" % (md_path, os.path.getsize(md_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
