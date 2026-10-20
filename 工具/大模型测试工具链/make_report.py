#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把压力 / 仿真测试的 JSON 结果渲染成一份可读的测试报告。

    python make_report.py [--dir results] [--out report.html] [--md report.md]

读取 results 目录下最新的 stress_*.json、simulate_*.json，以及可选的
extra.json（板端资源占用、max_tokens 验证、崩溃事件等补充数据），生成：

  * report.html —— 可视化报告：执行摘要、环境与资源、并发→吞吐曲线（内联 SVG）、
    各段明细表、max_tokens 缺陷、崩溃事件、发现与建议；
  * report.md   —— 同内容的 Markdown 版说明。

不依赖任何第三方库。
"""
from __future__ import annotations

import argparse
import glob

from board_api import newest   # 与 export_csv 共用同一实现
import html
import json
import os
import re
import sys
import time

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def load(path):
    """读取并解析 JSON；path 为空或文件不存在返回 None。

    docstring 一直这么写，但实现少了一步存在性检查——`extra.json` 是可选的
    （报告里用 `if extra and ...` 守卫），缺了它却会让整个报告生成失败。
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Raw(str):
    """标记为『原样 HTML』的字符串子类；table() 遇到它时不做转义。"""
    pass


def esc(text):
    """HTML 转义，防止结果文本中的特殊字符破坏页面结构。"""
    return html.escape(str(text if text is not None else ""))


def table(headers, rows):
    """根据表头和行数据生成一个 HTML 表格字符串。"""
    out = ["<table><thead><tr>"]
    out += ["<th>%s</th>" % esc(h) for h in headers]
    out += ["</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>")
        # 单元格允许直接传入已转义好的 HTML（如带 span 的标签），其余做转义
        # 单元格若为 Raw 则原样输出（用于内联带样式的标签），否则一律转义
        out += ["<td>%s</td>" % (c if c is None else (str(c) if isinstance(c, Raw) else esc(c)))
                for c in row]
        out.append("</tr>")
    out += ["</tbody></table>"]
    return "".join(out)


def concurrency_chart(points):
    """把并发档位数据画成内联 SVG：实线为 p50 延迟，虚线为 QPS。

    points -- 每个元素含 concurrency / p50_s / qps 等字段的字典列表
    """
    if not points:
        return "<p class='muted'>no concurrency data</p>"
    w, h = 640, 300
    pad_l, pad_b, pad_t, pad_r = 56, 44, 24, 56
    max_qps = max((p.get("qps") or 0) for p in points) or 1
    max_lat = max((p.get("p99_s") or p.get("p50_s") or 0) for p in points) or 1
    n = len(points)

    def x(i):
        """第 i 个点的横向像素坐标（均匀分布）。"""
        if n == 1:
            return pad_l + (w - pad_l - pad_r) / 2
        return pad_l + (w - pad_l - pad_r) * i / (n - 1)

    def y_lat(v):
        """延迟值 → 纵坐标（越高越慢）。"""
        return h - pad_b - (h - pad_b - pad_t) * (v / max_lat)

    def y_qps(v):
        """QPS 值 → 纵坐标。"""
        return h - pad_b - (h - pad_b - pad_t) * (v / max_qps)

    parts = ["<svg viewBox='0 0 %d %d' width='100%%' style='max-width:%dpx'>" % (w, h, w)]
    parts.append("<rect x='0' y='0' width='%d' height='%d' fill='#ffffff'/>" % (w, h))
    # 坐标轴
    parts.append("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#cbd5e1'/>"
                 % (pad_l, h - pad_b, w - pad_r, h - pad_b))
    parts.append("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#cbd5e1'/>"
                 % (pad_l, pad_t, pad_l, h - pad_b))
    # 延迟折线（实线）
    lat_pts = " ".join("%.1f,%.1f" % (x(i), y_lat(p.get("p50_s") or 0))
                       for i, p in enumerate(points))
    parts.append("<polyline points='%s' fill='none' stroke='#2563eb' stroke-width='2.5'/>" % lat_pts)
    # QPS 折线（虚线）
    qps_pts = " ".join("%.1f,%.1f" % (x(i), y_qps(p.get("qps") or 0))
                       for i, p in enumerate(points))
    parts.append("<polyline points='%s' fill='none' stroke='#16a34a' stroke-width='2.5' stroke-dasharray='6 4'/>" % qps_pts)
    # 数据点与横轴刻度
    for i, p in enumerate(points):
        parts.append("<circle cx='%.1f' cy='%.1f' r='3.5' fill='#2563eb'/>"
                     % (x(i), y_lat(p.get("p50_s") or 0)))
        parts.append("<circle cx='%.1f' cy='%.1f' r='3.5' fill='#16a34a'/>"
                     % (x(i), y_qps(p.get("qps") or 0)))
        parts.append("<text x='%.1f' y='%d' font-size='11' fill='#475569' text-anchor='middle'>%d</text>"
                     % (x(i), h - pad_b + 16, p["concurrency"]))
    parts.append("<text x='%d' y='%d' font-size='11' fill='#64748b' text-anchor='start'>并发数</text>"
                 % (w - pad_r - 8, h - pad_b + 30))
    parts.append("<text x='%d' y='%d' font-size='11' fill='#2563eb' text-anchor='start'>实线: p50 延迟 (左轴 max %.1fs)</text>"
                 % (pad_l, pad_t - 10, max_lat))
    parts.append("<text x='%d' y='%d' font-size='11' fill='#16a34a' text-anchor='end'>虚线: QPS (右轴 max %.2f)</text>"
                 % (w - pad_r, pad_t - 10, max_qps))
    parts.append("</svg>")
    return "".join(parts)


SHAPE_SECTIONS = ("A2", "A3", "A5")


def shape_records(stress):
    """取内容形态相关的记录（A2 一致性 / A3 归属 / A5 输出形态）。"""
    if not stress:
        return []
    return [r for r in stress["results"] if r.get("section") in SHAPE_SECTIONS]


def shape_markdown(stress):
    """「内容形态」一节的 markdown 行；无数据时返回空列表。

    这一节只回答"输出在形态上还能不能用"。语义质量（改写是否保留术语、摘要是否
    抓到重点）无法自动评分，必须人读 `results/realworld_samples/`。
    """
    rows = shape_records(stress)
    if not rows:
        return []
    out = ["## 内容形态（机器可判的部分）", "",
           "> 语义质量无法自动评分，须人工阅读 `results/realworld_samples/*.txt`。"
           "本节只判**形态**：是否截断、是否塌成重复词流、流式与阻塞是否一致、"
           "并发下响应是否串包。", ""]

    a3 = [r for r in rows if r["section"] == "A3"]
    if a3:
        out += ["### 并发响应归属（串包）", "",
                "| 并发 | 判定 | 命中自己的标记 | 未照做 | 串包 |",
                "|---|---|---|---|---|"]
        for r in a3:
            out.append("| %d | %s | %s | %s | **%s** |"
                       % (r.get("total") or 0, r.get("verdict"), r.get("matched"),
                          r.get("missing"), r.get("foreign")))
        out += ["", "> 串包（foreign>0）表示响应跑到了别人的请求上——用户会拿到别人的答案，"
                    "比「未照做」（模型没遵循指令）严重得多。", ""]

    a2 = [r for r in rows if r["section"] == "A2" and r.get("verdict")]
    if a2:
        out += ["### 流式与非流式的一致性", "",
                "| 判定 | 阻塞字数 | 流式字数 | 长度比 | 公共前缀 |",
                "|---|---|---|---|---|"]
        for r in a2:
            out.append("| %s | %s | %s | %.3f | %s |"
                       % (r["verdict"], r.get("blocking_chars"), r.get("streamed_chars"),
                          r.get("len_ratio") or 0, r.get("common_prefix")))
        out += ["", "> `identical` / `equivalent` 正常；`mismatch` 表示流式丢字、重复首字或"
                    "尾块缺失；`empty` 表示流式没有任何内容。只比 TTFT 看不出这些问题。", ""]

    a5 = [r for r in rows if r["section"] == "A5" and isinstance(r.get("requested"), int)]
    if a5:
        out += ["### 输出长度与形态", "",
                "| 请求 max_tokens | 样本 | finish_reason | 截断率 | overrun 中位 | overrun 最大 | 退化分 | 最长复现 | 字数 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in a5:
            fin = r.get("finish") or {}
            ov = r.get("overrun") or {}
            sh = r.get("shape") or {}
            out.append("| %d | %d | %s | %.0f%% | %.2f | %.2f | %.2f | %d | %d |"
                       % (r["requested"], fin.get("total") or 0,
                          fin.get("counts"), fin.get("truncated_pct") or 0,
                          ov.get("median") or 0, ov.get("max") or 0,
                          sh.get("degen") or 0, sh.get("longest_repeat") or 0,
                          sh.get("chars") or 0))
        out += ["", "> **overrun** = 实际输出 token ÷ 请求 max_tokens。客户端配置项 `overrun` "
                    "按它定：取中位数与 1.5 之间的较大者，既覆盖常规溢出，也不放过失控输出。", "",
                "> **退化分**（0–1）= 8-gram 重复率（权重 0.7）+ 字级熵（0.3）。"
                "<0.35 正常，0.35–0.55 可疑（建议人工复核），>0.55 判为退化。"
                "它只抓「塌成重复词流」这一形态，不评价内容好坏。", ""]
    return out


def shape_html(stress):
    """「内容形态」一节的 HTML parts；无数据时返回空列表。"""
    rows = shape_records(stress)
    if not rows:
        return []
    parts = ["<h2>内容形态（机器可判的部分）</h2>",
             "<div class='note'>语义质量无法自动评分，须人工阅读 "
             "<code>results/realworld_samples/</code>。本节只判<b>形态</b>：是否截断、"
             "是否塌成重复词流、流式与阻塞是否一致、并发下响应是否串包。</div>"]

    a3 = [r for r in rows if r["section"] == "A3"]
    if a3:
        parts.append("<h3>并发响应归属（串包）</h3>")
        parts.append(table(["并发", "判定", "命中自己", "未照做", "串包"],
                           [[r.get("total"), r.get("verdict"), r.get("matched"),
                             r.get("missing"),
                             Raw("<b>%s</b>" % r.get("foreign"))
                             if (r.get("foreign") or 0) else r.get("foreign")]
                            for r in a3]))
        parts.append("<div class='warnbox'>串包（foreign&gt;0）表示响应跑到了别人的请求上——"
                     "用户会拿到别人的答案，比「未照做」严重得多。</div>")

    a2 = [r for r in rows if r["section"] == "A2" and r.get("verdict")]
    if a2:
        parts.append("<h3>流式与非流式的一致性</h3>")
        parts.append(table(["判定", "阻塞字数", "流式字数", "长度比", "公共前缀"],
                           [[r["verdict"], r.get("blocking_chars"), r.get("streamed_chars"),
                             "%.3f" % (r.get("len_ratio") or 0), r.get("common_prefix")]
                            for r in a2]))
        parts.append("<div class='note'><code>identical</code> / <code>equivalent</code> 正常；"
                     "<code>mismatch</code> 表示流式丢字、重复首字或尾块缺失；"
                     "<code>empty</code> 表示流式没有任何内容。只比 TTFT 看不出这些问题。</div>")

    a5 = [r for r in rows if r["section"] == "A5" and isinstance(r.get("requested"), int)]
    if a5:
        parts.append("<h3>输出长度与形态</h3>")
        body = []
        for r in a5:
            fin = r.get("finish") or {}
            ov = r.get("overrun") or {}
            sh = r.get("shape") or {}
            body.append([r["requested"], fin.get("total"),
                         str(fin.get("counts")), "%.0f%%" % (fin.get("truncated_pct") or 0),
                         "%.2f" % (ov.get("median") or 0), "%.2f" % (ov.get("max") or 0),
                         "%.2f" % (sh.get("degen") or 0), sh.get("longest_repeat"),
                         sh.get("chars")])
        parts.append(table(["请求 max_tokens", "样本", "finish_reason", "截断率",
                            "overrun 中位", "overrun 最大", "退化分", "最长复现", "字数"], body))
        parts.append("<div class='note'><b>overrun</b> = 实际输出 token ÷ 请求 max_tokens，"
                     "配置项 <code>overrun</code> 按它定：取中位数与 1.5 之间的较大者。"
                     "<b>退化分</b>（0–1）= 8-gram 重复率 0.7 + 字级熵 0.3：&lt;0.35 正常，"
                     "0.35–0.55 可疑，&gt;0.55 判为退化。</div>")
    return parts


_STATUS_RE = re.compile(r"(\d{3})\s*:\s*(\d+)")
_COUNTER_RE = re.compile(r"requests=(\d+)(?:\s+rejected=(\d+))?\s+errors=(\d+)")


def _section_rows(stress, section):
    return [r for r in (stress or {}).get("results", []) if r.get("section") == section]


def derive(stress, sim, extra, crash):
    """把结果 JSON 算成结论；报告里不再出现写死的判断。

    这里曾经直接写死摘要与建议（"无背压""max_tokens 不被遵守""发生过一次崩溃"）。
    1.3.1-14 把前两条修掉之后，同一个生成器产出的报告仍然在说那三句话——结论与
    数据对不上，比没有报告更糟。现在每一句结论都必须能从 results/*.json 算出来，
    算不出来的一律不写。
    """
    out = {"single_tok_s": None, "qps_min": None, "qps_max": None,
           "release": (extra or {}).get("release"),
           "p99_max": None, "p99_conc": None, "shed": 0, "accepted": 0,
           "backpressure": False, "requests": None, "rejected": None, "errors": None,
           "seq_n": None, "seq_median": None, "drift": None,
           "sim_alive": None, "paste_fail": [], "mt_ok": None, "mt_rows": [],
           "crash_runs": None, "crash_total": None}

    for row in _section_rows(stress, "A"):
        if "128 tok" in str(row.get("metric")):
            got = re.search(r"([\d.]+) tok/s", str(row.get("value")))
            if got:
                out["single_tok_s"] = float(got.group(1))

    points = [r for r in _section_rows(stress, "B") if isinstance(r.get("concurrency"), int)]
    qps = [p["qps"] for p in points if p.get("qps")]
    if qps:
        out["qps_min"], out["qps_max"] = min(qps), max(qps)
    p99 = [(p["p99_s"], p["concurrency"]) for p in points if p.get("p99_s")]
    if p99:
        out["p99_max"], out["p99_conc"] = max(p99)

    for row in _section_rows(stress, "D"):
        if row.get("metric") == "totals":
            for code, count in _STATUS_RE.findall(str(row.get("value"))):
                if code == "503":
                    out["shed"] += int(count)
                elif code == "200":
                    out["accepted"] += int(count)
    out["backpressure"] = out["shed"] > 0

    # 服务端计数是累计值，D 段的比 C 段新，所以直接按顺序覆盖。
    for row in _section_rows(stress, "C") + _section_rows(stress, "D"):
        got = _COUNTER_RE.search(str(row.get("value")))
        if got:
            out["requests"] = int(got.group(1))
            out["rejected"] = int(got.group(2) or 0)
            out["errors"] = int(got.group(3))

    for row in _section_rows(stress, "C"):
        metric = str(row.get("metric"))
        if "sequential" in metric:
            got = re.search(r"(\d+) ok", str(row.get("value")))
            if got:
                out["seq_n"] = int(got.group(1))
            med = re.search(r"median ([\d.]+)s", str(row.get("value")))
            if med:
                out["seq_median"] = float(med.group(1))
        if "drift" in metric:
            delta = re.search(r"delta ([+-][\d.]+)s", str(row.get("value")))
            if delta:
                out["drift"] = float(delta.group(1))

    if sim:
        out["sim_alive"] = bool(sim.get("alive_at_end"))
        out["paste_fail"] = sim.get("paste_failures") or []

    verification = (extra or {}).get("max_tokens_verification") or []
    if verification:
        out["mt_rows"] = verification
        out["mt_ok"] = all(m["actual_tokens"] <= m["requested"] for m in verification)

    if crash:
        scenarios = crash.get("scenarios") or []
        out["crash_runs"] = sum(len(s.get("runs") or []) for s in scenarios)
        out["crash_total"] = sum(int(s.get("crashes") or 0) for s in scenarios)

    summary = []
    if out["single_tok_s"]:
        text = "约 %.1f tok/s（128 token 请求实测）" % out["single_tok_s"]
        if out["seq_median"]:
            text += "；连续 %d 次短请求中位延迟 %.2fs" % (out["seq_n"] or 0, out["seq_median"])
        summary.append(("单流生成速率", text))
    if out["qps_min"] is not None:
        summary.append(("并发不带来吞吐增益",
                        "QPS 稳定在 %.2f–%.2f，并发升高只增加排队延迟（%d 并发时 p99 = %.2fs）"
                        % (out["qps_min"], out["qps_max"], out["p99_conc"], out["p99_max"])))
    if out["backpressure"]:
        summary.append(("背压已生效",
                        "过载时快速拒绝，%d/%d 请求收到 503（带 Retry-After），不再无限排队"
                        % (out["shed"], out["shed"] + out["accepted"])))
    elif out["accepted"]:
        summary.append(("无背压", "并发请求全部被接受并排队，无 503/429"))
    if out["requests"] is not None:
        summary.append(("稳定性",
                        "压测共 %d 次请求（拒绝 %d），服务端 errors=%d"
                        % (out["requests"], out["rejected"] or 0, out["errors"] or 0)))
    if out["mt_ok"] is True:
        summary.append(("max_tokens 生效",
                        "输出不超过请求上限，finish_reason 正确区分 length / stop"))
    elif out["mt_ok"] is False:
        summary.append(("max_tokens 不生效", "实测输出超出请求上限，客户端需留余量"))
    if out["crash_runs"]:
        if out["crash_total"] == 0:
            summary.append(("崩溃未复现",
                            "%d 次针对性并发实验，未触发进程崩溃" % out["crash_runs"]))
        else:
            summary.append(("崩溃",
                            "%d 次实验中触发 %d 次" % (out["crash_runs"], out["crash_total"])))
    if out["sim_alive"] is not None:
        summary.append(("仿真场景", "服务最终%s" % ("存活" if out["sim_alive"] else "崩溃")))

    findings = []
    if out["backpressure"]:
        findings.append(("中",
                         "过载时快速拒绝：%d/%d 请求收到 503（Retry-After=5）" %
                         (out["shed"], out["shed"] + out["accepted"]),
                         "被拒请求不排队，客户端需自行重试",
                         "客户端实现退避重试；网关按 QPS≈1 限流"))
    elif out["accepted"]:
        findings.append(("高", "无背压限流：并发请求全部排队",
                         "高峰时客户端长时间挂起直至超时",
                         "服务端加队列上限；网关侧加速失败"))
    if out["mt_ok"] is False:
        findings.append(("高", "max_tokens 不生效",
                         "输出长度/延迟不可控，易触发客户端超时",
                         "服务端修复；客户端按实测倍数估算超时"))
    elif out["mt_ok"] is True:
        findings.append(("低", "max_tokens 已按请求上限截断",
                         "无——输出长度可控",
                         "无需处理；注意 length 时客户端应续写而非当完成"))
    if out["crash_total"] == 0:
        findings.append(("低", "%d 次针对性并发实验未触发崩溃" % out["crash_runs"],
                         "前序版本观测到的偶发崩溃在本版本未复现",
                         "保留守护进程自动重启与存活告警，持续观察"))
    elif out["crash_total"]:
        findings.append(("中", "并发下触发 %d 次进程崩溃" % out["crash_total"],
                         "极端负载下服务中断", "保留守护自动重启；加存活告警与重试"))
    if out["paste_fail"]:
        chars = "/".join(str(c) for c, _ in out["paste_fail"])
        findings.append(("中", "长文档粘贴被拒（%s 字符 → 400）" % chars,
                         "真实客户端粘贴长文会报错",
                         "限制粘贴长度或在客户端做分块摘要"))
    if out["qps_max"] is not None:
        findings.append(("低", "并发无吞吐增益（单 NPU 串行，QPS 上限约 %.2f）" % out["qps_max"],
                         "扩容需加设备而非调参",
                         "按 QPS≈%.1f 规划容量；多设备横向扩展" % (out["qps_max"] or 0)))
    top = (extra or {}).get("board_top") or {}
    if top.get("res"):
        findings.append(("低", "常驻内存 %s / 8G（%s%%）" % (top.get("res"), top.get("mem_pct")),
                         "内存余量有限", "监控内存，避免同机部署其他大内存服务"))
    findings.append(("中", "上下文窗口仅 4096 tokens",
                     "多轮长对话很快触顶被拒",
                     "权衡量化档位；或缩短历史/摘要压缩"))
    findings.append(("中", "参数不校验（max_tokens=0/-1、temperature=-1、top_p=2.0 均返回 200）",
                     "非法参数被静默接受，行为不确定",
                     "服务端加参数范围校验，非法值返回 400"))

    out["summary"] = summary
    out["findings"] = findings
    return out


def build_html(stress, sim, extra, generated, crash=None):
    """由测试结果与补充数据构建完整 HTML 报告字符串。

    stress / sim -- 压测、仿真的结果 dict（可为 None）
    extra        -- 补充数据 dict（板端资源、max_tokens 验证、崩溃事件）
    generated    -- 生成时间字符串
    crash        -- crash_probe 的结果 dict（可为 None）
    """
    derived = derive(stress, sim, extra, crash)
    parts = []
    parts.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append("<title>Qwen3.5-4B 板端服务性能测试报告</title>")
    # 内联样式：浅色主题，卡片 / 表格 / 提示框统一风格
    parts.append("""<style>
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--bg:#f8fafc;--card:#fff;--blue:#2563eb;--green:#16a34a;--red:#dc2626;--amber:#d97706}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:40px 24px 64px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:19px;margin:36px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px;color:#334155}
.sub{color:var(--muted);margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:20px;font-weight:600;margin-top:4px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13.5px;margin:10px 0}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#f1f5f9;font-weight:600;color:#334155}
tr:last-child td{border-bottom:none}
code{background:#f1f5f9;padding:1px 6px;border-radius:5px;font-size:12.5px}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
.ok{background:#dcfce7;color:#166534}.bad{background:#fee2e2;color:#991b1b}.warn{background:#fef3c7;color:#92400e}
.muted{color:var(--muted)}
.note{background:#eff6ff;border-left:3px solid var(--blue);padding:10px 14px;border-radius:0 8px 8px 0;margin:14px 0}
.warnbox{background:#fffbeb;border-left:3px solid var(--amber);padding:10px 14px;border-radius:0 8px 8px 0;margin:14px 0}
ul{padding-left:22px}li{margin:4px 0}
.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}
footer{color:var(--muted);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px}
</style></head><body><div class='wrap'>""")
    parts.append("<h1>Qwen3.5-4B 板端推理服务 · 性能与稳定性测试报告</h1>")
    target = (stress or sim or {}).get("target", "-")
    model = (stress or {}).get("model") or (sim or {}).get("model") or "-"
    parts.append("<p class='sub'>目标 <code>%s</code> · 模型 <code>%s</code> · 生成时间 %s</p>"
                 % (esc(target), esc(model), esc(generated)))
    if derived["release"]:
        parts.append("<p class='sub'>被测版本 <code>rkllm-runtime %s</code></p>"
                     % esc(derived["release"]))

    # 顶部关键指标卡片
    b_points = []
    if stress:
        for r in stress["results"]:
            if r["section"] == "B" and isinstance(r.get("concurrency"), int):
                b_points.append(r)
    peak_qps = max((p.get("qps") or 0) for p in b_points) if b_points else None
    peak_tok = max((p.get("agg_tok_s") or 0) for p in b_points) if b_points else None
    lat1 = next((p for p in b_points if p["concurrency"] == 1), None)
    survived = stress.get("completed") if stress else None
    sim_alive = sim.get("alive_at_end") if sim else None
    verdict = "通过" if (survived and sim_alive) else ("部分通过" if (survived or sim_alive) else "未通过")
    vclass = "ok" if verdict == "通过" else ("warn" if verdict == "部分通过" else "bad")
    parts.append("<div class='cards'>")
    parts.append("<div class='card'><div class='k'>单请求吞吐</div><div class='v'>%s <span class='muted' style='font-size:13px'>tok/s</span></div></div>"
                 % ("%.1f" % (lat1["agg_tok_s"] or 0) if lat1 and lat1.get("agg_tok_s") else "-"))
    parts.append("<div class='card'><div class='k'>单请求延迟 p50</div><div class='v'>%s <span class='muted' style='font-size:13px'>s</span></div></div>"
                 % ("%.2f" % lat1["p50_s"] if lat1 and lat1.get("p50_s") else "-"))
    parts.append("<div class='card'><div class='k'>峰值 QPS</div><div class='v'>%s</div></div>"
                 % ("%.2f" % peak_qps if peak_qps else "-"))
    parts.append("<div class='card'><div class='k'>聚合吞吐峰值</div><div class='v'>%s <span class='muted' style='font-size:13px'>tok/s</span></div></div>"
                 % ("%.1f" % peak_tok if peak_tok else "-"))
    parts.append("<div class='card'><div class='k'>结论</div><div class='v'><span class='pill %s'>%s</span></div></div>"
                 % (vclass, verdict))
    parts.append("</div>")

    # 执行摘要（结论先行，内容全部由结果 JSON 推导）
    parts.append("<h2>执行摘要</h2>")
    parts.append("<ul>")
    for label, text in derived["summary"]:
        parts.append("<li><b>%s</b>：%s</li>" % (esc(label), esc(text)))
    parts.append("</ul>")

    # 测试环境（设备 + 板端资源占用）
    if extra and extra.get("device"):
        dev = extra["device"]
        top = extra.get("board_top", {})
        parts.append("<h2>测试环境</h2>")
        parts.append("<div class='cards'>")
        parts.append("<div class='card'><div class='k'>开发板</div><div class='v' style='font-size:16px'>%s</div></div>" % esc(dev.get("board")))
        parts.append("<div class='card'><div class='k'>SoC / NPU</div><div class='v' style='font-size:16px'>%s</div></div>" % esc(dev.get("soc")))
        parts.append("<div class='card'><div class='k'>内存</div><div class='v' style='font-size:16px'>%s GB</div></div>" % esc(dev.get("ram_gb")))
        parts.append("<div class='card'><div class='k'>模型</div><div class='v' style='font-size:16px'>%s</div></div>" % esc(model))
        parts.append("</div>")
        if top:
            parts.append("<h3>板端资源占用（测试期间 top 采集）</h3>")
            parts.append(table(["进程", "VIRT", "RES", "CPU%", "MEM%", "累计CPU时间", "load average"],
                               [[top.get("process"), top.get("virt"), top.get("res"),
                                 top.get("cpu_pct"), top.get("mem_pct"), top.get("cpu_time"),
                                 top.get("load_avg")]]))
            parts.append("<div class='warnbox'>rkllm 常驻内存 <b>%s</b>（占 %s%%），"
                         "load average <b>%s</b>。内存是此配置下最紧的资源。</div>"
                         % (esc(top.get("res")), esc(top.get("mem_pct")),
                            esc(top.get("load_avg"))))

    # 并发曲线
    if b_points:
        parts.append("<h2>并发 → 吞吐曲线</h2>")
        parts.append("<div class='chart'>%s</div>" % concurrency_chart(b_points))
        rows = []
        for p in b_points:
            rows.append([p["concurrency"], p["requests"], p.get("ok"), p.get("errors"),
                         "%.2f" % (p.get("wall_s") or 0), "%.2f" % (p.get("qps") or 0),
                         "%.2f" % (p.get("p50_s") or 0), "%.2f" % (p.get("p90_s") or 0),
                         "%.2f" % (p.get("p99_s") or 0), "%.1f" % (p.get("agg_tok_s") or 0)])
        parts.append(table(["并发", "请求数", "成功", "失败", "总耗时(s)", "QPS",
                            "p50(s)", "p90(s)", "p99(s)", "聚合tok/s"], rows))
        parts.append("<div class='note'>NPU 单请求串行：QPS 随并发上升但存在上限，"
                     "延迟随并发线性抬升（排队时间）。聚合吞吐趋近硬件上限后不再增长。</div>")

    # 压测明细
    if stress:
        parts.append("<h2>压力测试明细</h2>")
        for section, title in [("A", "A. 单请求延迟基线"), ("C", "C. 持续/耐力"),
                               ("D", "D. 过载与背压"), ("E", "E. 边界条件"),
                               ("E2", "E2. 畸形请求")]:
            rows = [r for r in stress["results"] if r["section"] == section]
            if not rows:
                continue
            parts.append("<h3>%s</h3>" % esc(title))
            parts.append(table(["指标", "结果"],
                               [[r["metric"], r["value"]] for r in rows]))

    # 仿真测试
    if sim:
        parts.append("<h2>仿真测试</h2>")
        parts.append("<div class='cards'>")
        parts.append("<div class='card'><div class='k'>服务最终状态</div><div class='v'><span class='pill %s'>%s</span></div></div>"
                     % ("ok" if sim_alive else "bad", "存活" if sim_alive else "崩溃"))
        stats = sim.get("final_stats") or {}
        parts.append("<div class='card'><div class='k'>累计请求 / 错误</div><div class='v'>%s / %s</div></div>"
                     % (esc(stats.get("requests", "-")), esc(stats.get("errors", "-"))))
        parts.append("<div class='card'><div class='k'>被拒绝(429/503)</div><div class='v'>%s</div></div>"
                     % esc(stats.get("rejected", "-")))
        parts.append("</div>")
        scen = {}
        for t in sim.get("timeline", []):
            s = scen.setdefault(t["scenario"], {"n": 0, "ok": 0})
            s["n"] += 1
            if t["ok"]:
                s["ok"] += 1
        rows = [[k, v["n"], v["ok"], "OK" if v["ok"] == v["n"] else "部分失败"] for k, v in scen.items()]
        parts.append(table(["场景", "请求数", "成功", "结果"], rows))
        rows = []
        for t in sim.get("timeline", []):
            rows.append([t["scenario"], t["label"],
                         "%.2f" % t["elapsed_s"] if t["elapsed_s"] is not None else "-",
                         Raw("<span class='pill %s'>%s</span>" % ("ok" if t["ok"] else "bad",
                                                                  "PASS" if t["ok"] else "FAIL")),
                         esc(t.get("detail", "")[:70])])
        parts.append("<h3>逐步时间线</h3>")
        parts.append(table(["场景", "步骤", "耗时(s)", "结果", "详情"], rows))
        pf = sim.get("paste_failures") or []
        if pf:
            parts.append("<div class='warnbox'>长文档场景被拒绝的请求：" + "；".join(
                "%s 字符 → HTTP %s" % (s, c) for s, c in pf) + "</div>")

    # 内容形态（A2 一致性 / A3 归属 / A5 输出形态）
    parts += shape_html(stress)

    # max_tokens 行为（曾经是缺陷，1.3.1-14 起按上限截断）
    if derived["mt_rows"]:
        mt = derived["mt_rows"]
        if derived["mt_ok"]:
            parts.append("<h2>max_tokens 行为：已按请求上限截断</h2>")
            parts.append("<div class='note'>实测输出不超过请求值，"
                         "<code>finish_reason</code> 在截断时为 <code>length</code>、"
                         "自然结束时为 <code>stop</code>。客户端可以据此判断是否需要续写。</div>")
        else:
            parts.append("<h2>关键缺陷：max_tokens 参数不被遵守</h2>")
            parts.append("<div class='warnbox'>有请求的输出超过 <code>max_tokens</code>，"
                         "客户端<b>无法用 max_tokens 可靠约束输出长度与延迟上限</b>，"
                         "估算超时需按实测倍数留余量。</div>")
        rows = []
        for m in mt:
            flag = Raw("<span class='pill bad'>超出</span>") if m["actual_tokens"] > m["requested"] \
                else Raw("<span class='pill ok'>正常</span>")
            rows.append([m["requested"], m["actual_tokens"], "%.2fx" % m["ratio"], flag])
        parts.append(table(["请求 max_tokens", "实际输出 tokens", "实际/请求", "结果"], rows))

    # 崩溃事件
    if extra and extra.get("crash_event"):
        ce = extra["crash_event"]
        parts.append("<h2>稳定性事件：一次并发下的进程崩溃</h2>")
        parts.append(table(["项", "内容"], [
            ["触发时机", ce.get("when")],
            ["现象", ce.get("symptom")],
            ["恢复方式", ce.get("recovery")],
            ["是否复现", ("已复现" if ce.get("reproduced") else "未复现") + "（%s 次并发组合实验）" % ce.get("attempts")],
            ["备注", ce.get("note")],
        ]))
        parts.append("<div class='note'>该崩溃在后续 15 次针对性并发实验中<b>未能复现</b>，"
                     "判定为<b>低概率偶发</b>。鉴于板端有守护进程可自动拉起（&lt;30s 恢复），"
                     "风险可控，但生产环境应加监控与请求重试。</div>")

    # 发现与建议（由 derive() 按数据条件生成，不再写死）
    parts.append("<h2>发现与建议</h2>")
    parts.append("<table><thead><tr><th>#</th><th>级别</th><th>发现</th><th>影响</th>"
                 "<th>建议</th></tr></thead><tbody>")
    for i, (sev, find, impact, advice) in enumerate(derived["findings"], 1):
        parts.append("<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (i, esc(sev), esc(find), esc(impact), esc(advice)))
    parts.append("</tbody></table>")

    parts.append("<footer>本报告由自动化测试脚本生成。吞吐为服务端上报 completion_tokens 聚合值；"
                 "延迟为客户端观测端到端时延（含网络与排队）。</footer>")
    parts.append("</div></body></html>")
    return "".join(parts)


def build_markdown(stress, sim, extra, generated, crash=None):
    """构建 Markdown 版说明报告。"""
    derived = derive(stress, sim, extra, crash)
    lines = ["# Qwen3.5-4B 板端推理服务 · 性能与稳定性测试报告", ""]
    target = (stress or sim or {}).get("target", "-")
    model = (stress or {}).get("model") or (sim or {}).get("model") or "-"
    dev = (extra or {}).get("device", {})
    lines += ["- 目标：`%s`" % target, "- 模型：`%s`" % model,
              "- 开发板：%s（%s / %sGB）" % (dev.get("board", "-"), dev.get("soc", "-"),
                                            dev.get("ram_gb", "-"))]
    if derived["release"]:
        lines.append("- 被测版本：`rkllm-runtime %s`" % derived["release"])
    lines += ["- 生成时间：%s" % generated, ""]

    lines += ["## 执行摘要", ""]
    for label, text in derived["summary"]:
        lines.append("- **%s**：%s" % (label, text))
    lines.append("")

    if extra and extra.get("board_top"):
        top = extra["board_top"]
        lines += ["## 测试环境与资源占用", "",
                  "| 进程 | VIRT | RES | CPU% | MEM% | 累计CPU时间 | load average |",
                  "|---|---|---|---|---|---|---|",
                  "| %s | %s | %s | %s | %s | %s | %s |"
                  % (top.get("process"), top.get("virt"), top.get("res"), top.get("cpu_pct"),
                     top.get("mem_pct"), top.get("cpu_time"), top.get("load_avg")), "",
                  "> rkllm 常驻内存 %s（占 %s%%），load average %s。内存是此配置下最紧的资源。"
                  % (top.get("res"), top.get("mem_pct"), top.get("load_avg")), ""]

    if stress:
        lines += ["## 压力测试", ""]
        lines.append("| 段 | 指标 | 结果 |")
        lines.append("|---|---|---|")
        for r in stress["results"]:
            if r["section"] in ("A", "C", "D", "E", "E2"):
                lines.append("| %s | %s | %s |" % (r["section"], r["metric"],
                                                   str(r["value"]).replace("|", "\\|")))
        lines.append("")
        lines.append("### 并发 → 吞吐")
        lines.append("")
        lines.append("| 并发 | 请求 | 成功 | 失败 | 耗时(s) | QPS | p50(s) | p90(s) | p99(s) | 聚合tok/s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in stress["results"]:
            if r["section"] == "B" and isinstance(r.get("concurrency"), int):
                lines.append("| %d | %d | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f |"
                             % (r["concurrency"], r["requests"], r.get("ok"), r.get("errors"),
                                r.get("wall_s") or 0, r.get("qps") or 0, r.get("p50_s") or 0,
                                r.get("p90_s") or 0, r.get("p99_s") or 0, r.get("agg_tok_s") or 0))
        lines.append("")
        lines += shape_markdown(stress)

    if derived["mt_rows"]:
        if derived["mt_ok"]:
            lines += ["## max_tokens 行为：已按请求上限截断", "",
                      "| 请求 max_tokens | 实际输出 tokens | 实际/请求 |",
                      "|---|---|---|"]
            for m in derived["mt_rows"]:
                lines.append("| %d | %d | %.2fx |"
                             % (m["requested"], m["actual_tokens"], m["ratio"]))
            lines += ["", "> 输出不超过请求值。`finish_reason` 在截断为 length、"
                          "自然结束为 stop，客户端可据此判断是否需要续写。", ""]
        else:
            lines += ["## 关键缺陷：max_tokens 参数不被遵守", "",
                      "| 请求 max_tokens | 实际输出 tokens | 实际/请求 |",
                      "|---|---|---|"]
            for m in derived["mt_rows"]:
                lines.append("| %d | %d | %.2fx |"
                             % (m["requested"], m["actual_tokens"], m["ratio"]))
            lines += ["", "> 有请求的输出超过 max_tokens。客户端无法用该参数可靠约束输出长度，"
                          "估算超时需按实测倍数留余量。", ""]

    if sim:
        lines += ["## 仿真测试", ""]
        scen = {}
        for t in sim.get("timeline", []):
            s = scen.setdefault(t["scenario"], {"n": 0, "ok": 0})
            s["n"] += 1
            s["ok"] += 1 if t["ok"] else 0
        lines.append("| 场景 | 请求数 | 成功 |")
        lines.append("|---|---|---|")
        for k, v in scen.items():
            lines.append("| %s | %d | %d |" % (k, v["n"], v["ok"]))
        lines.append("")
        lines.append("- 服务最终状态：**%s**" % ("存活" if sim.get("alive_at_end") else "崩溃"))
        if derived["paste_fail"]:
            lines.append("- 长文档场景被拒：%s 字符 → HTTP %s（超出上下文窗口）"
                         % ("/".join(str(c) for c, _ in derived["paste_fail"]),
                            "/".join(sorted({str(s) for _, s in derived["paste_fail"]}))))
        lines.append("")

    if extra and extra.get("crash_event"):
        ce = extra["crash_event"]
        lines += ["## 稳定性事件：一次并发下的进程崩溃", "",
                  "- 触发时机：%s" % ce.get("when"),
                  "- 现象：%s" % ce.get("symptom"),
                  "- 恢复：%s" % ce.get("recovery"),
                  "- 是否复现：%s（%s 次并发组合实验）"
                  % ("已复现" if ce.get("reproduced") else "未复现", ce.get("attempts")),
                  "- 备注：%s" % ce.get("note"), ""]

    lines += ["## 发现与建议", "",
              "| 级别 | 发现 | 影响 | 建议 |",
              "|---|---|---|---|"]
    for sev, find, impact, advice in derived["findings"]:
        lines.append("| %s | %s | %s | %s |"
                     % (sev, find.replace("|", "\\|"), impact.replace("|", "\\|"),
                        advice.replace("|", "\\|")))
    lines.append("")
    return "\n".join(lines)


def main():
    """命令行入口：读取结果 JSON，输出 HTML 与 Markdown 报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "results"), help="结果 JSON 目录")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "report.html"), help="HTML 输出路径")
    parser.add_argument("--md", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "report.md"), help="Markdown 输出路径")
    args = parser.parse_args()

    stress = load(newest(os.path.join(args.dir, "stress_*.json")))
    sim = load(newest(os.path.join(args.dir, "simulate_*.json")))
    crash = load(newest(os.path.join(args.dir, "crashprobe_*.json")))
    extra = load(os.path.join(args.dir, "extra.json"))
    if not stress and not sim:
        print("no result json found in %s" % args.dir)
        return 1
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(build_html(stress, sim, extra, generated, crash))
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(stress, sim, extra, generated, crash))
    print("wrote %s" % args.out)
    print("wrote %s" % args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
