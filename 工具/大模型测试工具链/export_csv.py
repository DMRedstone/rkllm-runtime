#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把测试结果 JSON 导出为 CSV，便于做表、画图或二次分析。

    python export_csv.py [--dir results] [--outdir csv]

读取 results 目录下最新的 stress_*.json 与 simulate_*.json，导出三张表：

  * stress_concurrency_<ts>.csv —— 并发→吞吐曲线（每档一行）
  * stress_sections_<ts>.csv    —— 压测各段明细（长表，便于筛选）
  * simulate_timeline_<ts>.csv  —— 仿真逐步时间线（每步一行）

不依赖第三方库。生成的 CSV 均为 UTF-8 带 BOM，可直接用 Excel 打开。
"""
from __future__ import annotations

import argparse
import csv
import glob

from board_api import newest   # 与 make_report 共用同一实现
import json
import os
import sys
import time

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def load(path):
    """读取 JSON；path 为空返回 None。"""
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path, headers, rows):
    """写一个 UTF-8-BOM 的 CSV（Excel 友好），返回路径。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def export_stress_concurrency(stress, outdir, tag):
    """导出并发→吞吐曲线 CSV。"""
    points = [r for r in stress["results"]
              if r["section"] == "B" and isinstance(r.get("concurrency"), int)]
    if not points:
        return None
    headers = ["concurrency", "requests", "ok", "errors", "wall_s", "qps",
               "p50_s", "p90_s", "p99_s", "max_s", "completion_tokens", "agg_tok_s"]
    rows = [[p.get(h) for h in headers] for p in points]
    return write_csv(os.path.join(outdir, "stress_concurrency_%s.csv" % tag), headers, rows)


def export_stress_sections(stress, outdir, tag):
    """导出压测各段明细 CSV（每行一个指标）。"""
    headers = ["section", "metric", "value"]
    rows = [[r.get("section"), r.get("metric"), r.get("value")] for r in stress["results"]]
    return write_csv(os.path.join(outdir, "stress_sections_%s.csv" % tag), headers, rows)


def export_simulate_timeline(sim, outdir, tag):
    """导出仿真时间线 CSV。"""
    timeline = sim.get("timeline") or []
    if not timeline:
        return None
    headers = ["scenario", "label", "ok", "elapsed_s", "detail"]
    rows = [[t.get("scenario"), t.get("label"), t.get("ok"),
             t.get("elapsed_s"), t.get("detail")] for t in timeline]
    return write_csv(os.path.join(outdir, "simulate_timeline_%s.csv" % tag), headers, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "results"), help="结果 JSON 目录")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "csv"), help="CSV 输出目录")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    stress = load(newest(os.path.join(args.dir, "stress_*.json")))
    sim = load(newest(os.path.join(args.dir, "simulate_*.json")))
    if not stress and not sim:
        print("no result json found in %s" % args.dir)
        return 1

    tag = time.strftime("%Y%m%d_%H%M%S")
    written = []
    if stress:
        for fn in (export_stress_concurrency, export_stress_sections):
            path = fn(stress, args.outdir, tag)
            if path:
                written.append(path)
    if sim:
        path = export_simulate_timeline(sim, args.outdir, tag)
        if path:
            written.append(path)

    if not written:
        print("no exportable data found")
        return 1
    for path in written:
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
