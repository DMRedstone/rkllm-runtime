#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""space_name_check —— 云端资料库 ↔ 本地目录「按名称」核对（忽略层级）+ 数量统计。

与 space_sync_check.py 的区别：
  space_sync_check.py  逐「路径」比对（相对目录 + 名字），发现结构差异；
  space_name_check.py  只比「名字」（扩展名容错），用于判断"同一批文件是否两边都在"，
                       顺带给出云端叶子数 / 本地文件数。

用法：
  python space_name_check.py --cloud-root <节点id> --local-dir <本地目录>
                             [--skip-cloud 目录名 ...] [--local-only 本地相对路径 ...]
                             [--space-api /path/to/space_api.py]

输出：仅云端有 / 仅本地有 两份清单 + 数量统计；退出码 0=一致 1=有差异。
纯标准库；只读；自动定位 space_api.py（--space-api → $SPACE_API → 常见插件目录）。
"""
import argparse
import json
import os
import subprocess
import sys
import time

EXTS = [".docx", ".html", ".htm", ".json", ".csv", ".md", ".py", ".txt", ".conf",
        ".deb", ".tar.gz", ".tgz", ".pyc", ".sh", ".service", ".in", ".head",
        ".gz", ".1", ".pc", ".cmake", ".json", ".jpg", ".png"]


def find_space_api(explicit):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("SPACE_API"):
        cands.append(os.environ["SPACE_API"])
    home = os.path.expanduser("~")
    cands += [
        os.path.join(home, ".workbuddy", "plugins", "cache", "workbuddy-builtin",
                     "skills", "library", "space_api.py"),
        r"C:/Program Files/WorkBuddy/resources/app.asar.unpacked/resources/plugins/"
        r"workbuddy-builtin/skills/library/space_api.py",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    for root, dirs, files in os.walk(os.path.join(home, ".workbuddy", "plugins")):
        if "space_api.py" in files:
            return os.path.join(root, "space_api.py")
    return None


def strip_any(name):
    out = {name}
    for e in EXTS:
        if name.endswith(e):
            out.add(name[: -len(e)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud-root", required=True, help="云端起始节点 id（子节点递归）")
    ap.add_argument("--local-dir", required=True, help="本地目录")
    ap.add_argument("--space-id", default=None, help="空间 id（可选，仅用于 list-nodes）")
    ap.add_argument("--skip-cloud", action="append", default=[], help="根层要跳过的云端节点名（可多次）")
    ap.add_argument("--local-only", action="append", default=[], help="本地要跳过的相对路径（可多次）")
    ap.add_argument("--space-api", default=None)
    ap.add_argument("--retries", type=int, default=4)
    a = ap.parse_args()

    api = find_space_api(a.space_api)
    if not api:
        print("找不到 space_api.py，请用 --space-api 指定", file=sys.stderr)
        return 2
    python = sys.executable
    skip = set(a.skip_cloud)
    local_only = set(a.local_only)

    def ls(pid):
        for _ in range(a.retries):
            cmd = [python, api, "space.workspace.list-nodes", "--limit", "200",
                   "--parent-node-id", pid]
            if a.space_id:
                cmd += ["--space-id", a.space_id]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            out = (p.stdout or "").strip()
            i = out.find("{")
            if i >= 0:
                try:
                    return json.JSONDecoder().raw_decode(out[i:])[0]["data"]["nodes"]
                except Exception:
                    pass
            time.sleep(0.8)
        return None

    cloud = []
    def walk(pid, depth_root):
        ns = ls(pid)
        if ns is None:
            return
        for x in ns:
            t, k = x["title"], x["kind"]
            if depth_root and t in skip:
                continue
            cloud.append((t, k))
            if k == "folder":
                walk(x["id"], False)
    walk(a.cloud_root, True)

    local = []
    ldirs = set()
    for root, dirs, files in os.walk(a.local_dir):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for d in dirs:
            ldirs.add(d)
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), a.local_dir)
            if rel in local_only or os.path.dirname(rel).split(os.sep)[0] in local_only:
                continue
            local.append(rel)

    ctitle = {}
    for t, k in cloud:
        ctitle.setdefault(t, k)
    lname = {}
    for d in ldirs:
        lname.setdefault(d, "<dir>")
    for rel in local:
        for s in strip_any(os.path.basename(rel)):
            lname.setdefault(s, rel)

    only_c = sorted(t for t in ctitle if t not in lname)
    only_l = sorted(rel for rel in local
                    if os.path.basename(rel) not in ctitle
                    and not (strip_any(os.path.basename(rel)) & set(ctitle)))

    n_leaf = sum(1 for _, k in cloud if k != "folder")
    print("云端叶子文件 = %d   本地文件 = %d" % (n_leaf, len(local)))
    print("\n[仅云端有] %d：" % len(only_c))
    for t in only_c:
        print("   %s  [%s]" % (t, ctitle[t]))
    print("\n[仅本地有] %d：" % len(only_l))
    for r in only_l:
        print("   %s" % r)
    return 1 if (only_c or only_l) else 0


if __name__ == "__main__":
    sys.exit(main())
