"""F3 七类场景验收：每类 5 条，板端本地跑（127.0.0.1），输出可供人工判定。

七类：摘要 / 改写 / 抽取 / 分类 / 翻译 / 短问答 / 批量。
每条记录：耗时、输出字数、输出片段（前 120 字），写入 f3_result.json。
判据（设计 G2）：单条 < 60 s 返回、输出与任务类型匹配、无乱码/标签残留。
"""

import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080/v1"
KEY = os.environ["RKLLM_API_KEY"]

DOC1 = ("关于加强建筑垃圾全过程管理的通知。为进一步规范建筑垃圾产生、收集、运输、"
        "处置各环节管理，经研究决定，自2026年11月1日起，全县建筑垃圾实行申报登记制度。"
        "产生单位应在开工前15日内向属地乡镇（街道）申报，内容包括工程名称、施工地址、"
        "预计产生量、运输单位、消纳场所。运输车辆必须安装定位装置并接入监管平台。")

DOC2 = ("城市绿化是改善人居环境的重要抓手。近年来，我县累计新增绿地面积120万平方米，"
        "建成口袋公园36处，人均公园绿地面积达到14.2平方米。下一步将重点推进老旧小区"
        "绿化补植和立体绿化试点，力争三年内人均指标达到15平方米。")

DOC3 = ("会议听取了县住建局关于农村危房改造工作进展的汇报。截至9月底，全县已完成"
        "改造1276户，占年度计划的92.4%，剩余105户因地质条件复杂，预计11月底前完成。"
        "会议要求，要严把质量关，确保改造后房屋安全达标。")

CASES = [
    # -- 摘要 -------------------------------------------------------------
    ("摘要", "用不超过100字总结要点：\n" + DOC1),
    ("摘要", "用不超过100字总结要点：\n" + DOC2),
    ("摘要", "用不超过100字总结要点：\n" + DOC3),
    ("摘要", "用一句话概括这段文字：\n" + DOC1),
    ("摘要", "分三点列出这段文字的关键信息：\n" + DOC2),
    # -- 改写 -------------------------------------------------------------
    ("改写", "把下面这段公文改成口语化表达，保留全部关键信息：\n" + DOC1),
    ("改写", "把下面这段改成更正式的公文表述：\n城市绿化搞得不错，老百姓挺满意，"
             "以后还要接着干。"),
    ("改写", "把下面这段压缩到一半长度，不丢关键数字：\n" + DOC2),
    ("改写", "把下面这段改成对居民的通知口吻：\n" + DOC3),
    ("改写", "把下面这段改写成 80 字以内的简报：\n" + DOC1),
    # -- 抽取（json_object 走提示约束）-------------------------------------
    ("抽取", "从下面文本中抽取 JSON，字段为 title、doc_number、date、units、urgent，"
             "只输出 JSON：\n柞政发〔2026〕19号 关于加强建筑垃圾全过程管理的通知 "
             "2026年10月21日 县住建局、县生态环境局"),
    ("抽取", "从下面文本中抽取 JSON，字段为 area、parks、per_capita，只输出 JSON：\n"
             + DOC2),
    ("抽取", "从下面文本中抽取 JSON，字段为 households、percent、deadline，只输出 JSON：\n"
             + DOC3),
    ("抽取", "从下面文本中抽取 JSON，字段为 start_date、report_days，只输出 JSON：\n"
             + DOC1),
    ("抽取", "从下面文本中抽取 JSON，字段为 title、date，只输出 JSON：\n"
             "关于开展安全生产大检查的通知\n2026年10月20日"),
    # -- 分类 -------------------------------------------------------------
    ("分类", "判断下面文本属于哪一类（通知/报告/请示/纪要），只答类别：\n" + DOC1),
    ("分类", "判断下面文本属于哪一类（通知/报告/请示/纪要），只答类别：\n" + DOC2),
    ("分类", "判断下面文本属于哪一类（通知/报告/请示/纪要），只答类别：\n" + DOC3),
    ("分类", "判断下面这段情感倾向（正面/中性/负面），只答一个词：\n"
             "改造进度滞后，群众意见较大。"),
    ("分类", "判断下面这段情感倾向（正面/中性/负面），只答一个词：\n" + DOC2),
    # -- 翻译 -------------------------------------------------------------
    ("翻译", "把下面这句翻译成英文：\n建筑垃圾实行申报登记制度。"),
    ("翻译", "把下面这句翻译成英文：\n人均公园绿地面积达到14.2平方米。"),
    ("翻译", "把下面这句翻译成中文：\nThe county completed 1,276 rural housing "
             "renovations by the end of September."),
    ("翻译", "把下面这句翻译成英文：\n会议要求严把质量关，确保房屋安全达标。"),
    ("翻译", "把下面这段翻译成日文，保留数字：\n" + DOC3),
    # -- 短问答 -----------------------------------------------------------
    ("短问答", "建筑垃圾申报应提前多少天？只答数字和单位。"),
    ("短问答", "人均公园绿地面积现状是多少？只答数字。"),
    ("短问答", "危房改造年度计划完成率是多少？只答百分比。"),
    ("短问答", "运输车辆需要安装什么装置？一句话回答。"),
    ("短问答", "口袋公园建成了多少处？只答数字。"),
    # -- 批量（一次请求处理多段）------------------------------------------
    ("批量", "分别用一句话概括下面三段，每段一行，前缀 1./2./3.：\n"
             "1) " + DOC1[:120] + "\n2) " + DOC2[:120] + "\n3) " + DOC3[:120]),
    ("批量", "把下面三段分别改成一个标题，每段一行：\n"
             "1) " + DOC1[:120] + "\n2) " + DOC2[:120] + "\n3) " + DOC3[:120]),
    ("批量", "给下面三段各配一个 10 字以内的摘要标签：\n"
             "1) " + DOC1[:120] + "\n2) " + DOC2[:120] + "\n3) " + DOC3[:120]),
    ("批量", "把下面三段分别判断类别（通知/报告/请示/纪要），每段一行：\n"
             "1) " + DOC1[:100] + "\n2) " + DOC2[:100] + "\n3) " + DOC3[:100]),
    ("批量", "把下面三段分别抽取一个最关键的数字，每段一行：\n"
             "1) " + DOC1[:120] + "\n2) " + DOC2[:120] + "\n3) " + DOC3[:120]),
]


def ask(prompt: str, max_tokens: int = 256, timeout: int = 240) -> tuple[str, float]:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"] or "", time.time() - t0


def main() -> int:
    results = []
    for i, (kind, prompt) in enumerate(CASES, 1):
        try:
            text, elapsed = ask(prompt)
            ok = bool(text.strip()) and "</think>" not in text
            print("[%2d/%d] %-4s %5.1fs %3d字 %s" %
                  (i, len(CASES), kind, elapsed, len(text), text[:40].replace("\n", " ")),
                  flush=True)
        except Exception as exc:
            text, elapsed, ok = "ERROR: %s" % exc, -1.0, False
            print("[%2d/%d] %-4s FAILED %s" % (i, len(CASES), kind, exc), flush=True)
        results.append({"index": i, "kind": kind, "prompt": prompt[:80],
                        "elapsed_s": round(elapsed, 1), "chars": len(text),
                        "ok": ok, "output": text})

    summary = {}
    for kind in ("摘要", "改写", "抽取", "分类", "翻译", "短问答", "批量"):
        rows = [r for r in results if r["kind"] == kind]
        summary[kind] = {
            "count": len(rows),
            "ok": sum(1 for r in rows if r["ok"]),
            "avg_s": round(sum(r["elapsed_s"] for r in rows) / len(rows), 1),
            "max_s": max(r["elapsed_s"] for r in rows),
            "all_under_60s": all(r["elapsed_s"] < 60 for r in rows),
        }
    with open("/tmp/f3_result.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": results}, fh,
                  ensure_ascii=False, indent=1)
    print("\n=== 汇总 ===")
    for kind, s in summary.items():
        print("%-4s %d/%d 通过  平均 %5.1fs  最慢 %5.1fs  均<60s=%s"
              % (kind, s["ok"], s["count"], s["avg_s"], s["max_s"],
                 s["all_under_60s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
