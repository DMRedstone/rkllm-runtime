#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资料库空间 <-> 本地目录 双向比对（结构级 / 内容级）。

比对口径
--------
* 结构（默认）：云端每个节点的标题，在本地对应目录下应有同名文件或同名目录
  （扩展名不限）；反之本地文件的基名应在云端同级目录里能找到同名节点。
* 内容（--content）：对结构上已配对的文件，下载云端内容做**归一化指纹**比对
  （去 BOM、统一换行、去行尾空白），报告「名字对得上但内容不同」的项。
  --content-doc 额外覆盖 doc 节点（读正文转纯文本），适合核对文档是否同步。

为什么归一化：同名文件因为换行符（CRLF/LF）、行尾空格、BOM 造成 MD5 不同是
常态，直接的二进制哈希会把噪声当成差异，故默认按归一化文本比对；需要严格
字节比对时加 --raw。

用法示例
--------
    # 结构级：整棵空间树对本地目录
    python space_sync_check.py --space-id <SPACE_ID> --local-dir ./myproject

    # 只比某个节点子树（空间 ID 从节点信息自动取）
    python space_sync_check.py --cloud-root <NODE_ID> --local-dir ./myproject

    # 排除项（可重复）：云端忽略的标题 / 本地专有项
    python space_sync_check.py --space-id <S> --local-dir ./p \\
        --skip-cloud 文档 --skip-cloud 交付 --local-only 本地过程存档

    # 结构 + 内容级，覆盖 py/sh/json/md，并连带核对 doc 节点
    python space_sync_check.py --cloud-root <NODE_ID> --local-dir ./p \\
        --content --content-doc

退出码：0 = 无差异；1 = 有差异；2 = 运行错误。
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_TEXT_EXT = "py,sh,json,txt,md,conf,ini,cfg,cmake,in,service,yml,yaml,csv,html,css,js,ts"
TEXT_EXT = set(DEFAULT_TEXT_EXT.split(","))


def find_space_api(explicit=None):
    """定位 skill-library 里的 space_api.py。"""
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("SPACE_API"):
        cands.append(os.environ["SPACE_API"])
    home = os.path.expanduser("~")
    cands += glob.glob(os.path.join(home, ".workbuddy", "plugins", "cache", "*", "*",
                                    "skill-library", "*", "space_api.py"))
    cands += glob.glob(os.path.join(home, ".workbuddy", "plugins", "cache", "*",
                                    "skill-library", "*", "space_api.py"))
    for root in ("C:/Program Files/WorkBuddy/resources/app.asar.unpacked/resources/plugins",
                 "C:/Program Files/WorkBuddy/resources/app.asar.unpacked/resources/plugins/workbuddy-builtin/skills"):
        cands += glob.glob(os.path.join(root, "**", "space_api.py"), recursive=True)
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.normpath(c)
    return None


def strip_ext(name):
    return os.path.splitext(name)[0]


def skeleton(text):
    """去空白 + 去 Markdown/标题标记，用于跨格式（doc 正文 vs .md）的宽松比对。"""
    s = norm_text(text)
    s = re.sub(r"^\s{0,3}(#{1,6}|[-*+>]|\d+\.)\s*", "", s, flags=re.M)
    s = s.replace("```", "")
    return re.sub(r"[\s|`*_\\]+", "", s)


def norm_text(s):
    s = s.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(x.rstrip() for x in s.split("\n"))


def digest_bytes(data, raw=False):
    if not raw:
        try:
            data = norm_text(data.decode("utf-8", errors="replace")).encode("utf-8")
        except Exception:
            pass
    return hashlib.md5(data).hexdigest()


def digest_file(path, raw=False):
    try:
        with open(path, "rb") as f:
            return digest_bytes(f.read(), raw)
    except Exception:
        return None


class Space:
    def __init__(self, api, python, retries=4):
        self.api, self.python, self.retries = api, python, retries
        self.calls = 0
        self.root = os.path.dirname(api)
        self._tmp = None

    def _json(self, script, *args):
        last = ""
        for attempt in range(self.retries):
            self.calls += 1
            proc = subprocess.run([self.python, script] + list(args),
                                  capture_output=True, text=True, encoding="utf-8")
            out = (proc.stdout or "").strip()
            i = out.find("{")
            if i >= 0:
                try:
                    return json.JSONDecoder().raw_decode(out[i:])[0]
                except Exception as exc:
                    last = "解析失败: %s" % exc
            else:
                last = out[:120] or (proc.stderr or "")[:120]
            if attempt < self.retries - 1:
                time.sleep(1.2)
        return {"__error__": "接口调用失败 %s -> %s" % (args, last)}

    def api_call(self, *args):
        d = self._json(self.api, *args)
        if "__error__" in d:
            raise RuntimeError(d["__error__"])
        return d

    def node_info(self, node_id):
        return self.api_call("space.workspace.node-info", "--node-id", node_id)["data"]["node"]

    def children(self, space_id, parent_id=None):
        args = ["space.workspace.list-nodes", "--space-id", space_id, "--limit", "200"]
        if parent_id:
            args += ["--parent-node-id", parent_id]
        return self.api_call(*args)["data"]["nodes"]

    def tmpdir(self):
        if not self._tmp:
            self._tmp = tempfile.mkdtemp(prefix="spacediff_")
        return self._tmp

    def download(self, node_id, name, max_size):
        """取 drive 节点内容到临时文件；返回路径或 None。"""
        d = self._json(os.path.join(self.root, "drive", "get_download_link.py"),
                       "--node-id", node_id)
        url = d.get("download_url")
        if not url:
            return None
        safe = re.sub(r"[^\w.\-]", "_", name) or "node"
        dst = os.path.join(self.tmpdir(), "%s_%s" % (node_id, safe))
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception:
            return None
        if max_size and os.path.getsize(dst) > max_size:
            return None
        return dst

    def _raw(self, script, *args):
        """调用接口脚本，返回原始 stdout（doc 正文接口不返回 JSON）。"""
        for attempt in range(self.retries):
            self.calls += 1
            proc = subprocess.run([self.python, script] + list(args),
                                  capture_output=True, text=True, encoding="utf-8")
            out = (proc.stdout or "").strip()
            if out:
                return out
            if attempt < self.retries - 1:
                time.sleep(1.2)
        return ""

    def doc_text(self, node_id):
        """读 doc 节点正文，剥去 block 标签后返回纯文本。"""
        out = self._raw(os.path.join(self.root, "doc", "get_doc_reviews.py"),
                        "--page-id", node_id)
        return re.sub(r"<[^>]+>", "", out)


def main():
    ap = argparse.ArgumentParser(description="资料库空间 <-> 本地目录 结构/内容 双向比对")
    ap.add_argument("--space-id", help="空间 ID；给了 --cloud-root 时可省略")
    ap.add_argument("--cloud-root", help="比对起点节点 ID；省略则从空间根开始")
    ap.add_argument("--local-dir", required=True, help="本地目录")
    ap.add_argument("--skip-cloud", action="append", default=[], help="云端根层忽略的节点标题（可重复）")
    ap.add_argument("--local-only", action="append", default=[], help="本地专有项（文件或一级目录，可重复）")
    ap.add_argument("--max-depth", type=int, default=12, help="云端递归深度上限（默认 12）")
    ap.add_argument("--space-api", help="space_api.py 路径；默认自动查找")
    ap.add_argument("--python", default=sys.executable, help="跑接口脚本的解释器")
    # 内容级
    ap.add_argument("--content", action="store_true", help="开启内容级比对（drive 文本文件）")
    ap.add_argument("--content-doc", action="store_true", help="内容级比对时连带核对 doc 节点")
    ap.add_argument("--content-ext", default=DEFAULT_TEXT_EXT, help="参与内容比对的扩展名（逗号分隔）")
    ap.add_argument("--raw", action="store_true", help="内容比对用原始字节（不做换行/行尾归一化）")
    ap.add_argument("--max-size", type=int, default=4 * 1024 * 1024, help="云端下载体积上限，超出跳过（默认 4MiB）")
    args = ap.parse_args()

    api = find_space_api(args.space_api)
    if not api:
        sys.exit("找不到 space_api.py，请用 --space-api 指定")
    if not os.path.isdir(args.local_dir):
        sys.exit("本地目录不存在: %s" % args.local_dir)
    if not (args.space_id or args.cloud_root):
        sys.exit("至少要给 --space-id 或 --cloud-root")

    sp = Space(api, args.python)
    space_id = args.space_id
    if args.cloud_root:
        node = sp.node_info(args.cloud_root)
        space_id = space_id or node.get("spaceId")
        start_id = args.cloud_root
    else:
        start_id = None

    skip = set(args.skip_cloud)
    local_only = set(args.local_only)
    content_ext = {e.strip().lower() for e in args.content_ext.split(",") if e.strip()}

    cloud = []  # dict: rel, title, kind, ext, id

    def walk(parent_id, rel, depth):
        if depth > args.max_depth:
            return
        for n in sp.children(space_id, parent_id):
            title, kind = n.get("title"), n.get("kind")
            if depth == 0 and title in skip:
                continue
            cloud.append({"rel": rel, "title": title, "kind": kind,
                          "ext": (n.get("ext") or "").lower(), "id": n["id"]})
            if kind == "folder":
                walk(n["id"], (rel + "/" + title) if rel else title, depth + 1)

    walk(start_id, "", 0)
    local_dir = os.path.abspath(args.local_dir)

    def local_path_for(item):
        """云端节点 -> 本地路径（文件；目录返回其路径）。"""
        d = os.path.join(local_dir, item["rel"].replace("/", os.sep)) if item["rel"] else local_dir
        p = os.path.join(d, item["title"])
        if item["kind"] == "folder":
            return p if os.path.isdir(p) else None
        if os.path.exists(p):
            return p
        hits = glob.glob(os.path.join(d, glob.escape(item["title"]) + ".*"))
        return hits[0] if hits else None

    # ---------- 结构：云端 -> 本地 ----------
    print("=== 云端 -> 本地：缺失 ===")
    missing = [it for it in cloud if local_path_for(it) is None]
    for it in missing:
        print("  MISS  %-52s [%s]" % ((it["rel"] + "/" if it["rel"] else "") + it["title"], it["kind"]))
    print("  —— 缺失 %d / 云端共 %d 项" % (len(missing), len(cloud)))

    # ---------- 结构：本地 -> 云端 ----------
    print()
    print("=== 本地 -> 云端：多余 ===")
    by_dir = {}
    for it in cloud:
        by_dir.setdefault(it["rel"], set()).add(it["title"])
        by_dir[it["rel"]].add(strip_ext(it["title"]))
    extra = []
    for dirpath, dirnames, filenames in os.walk(local_dir):
        rel = os.path.relpath(dirpath, local_dir).replace(os.sep, "/")
        rel = "" if rel == "." else rel
        if rel.split("/")[0] in local_only or "__pycache__" in rel.split("/"):
            continue
        for name in sorted(filenames):
            if rel == "" and name in local_only:
                continue
            titles = by_dir.get(rel, set())
            if name not in titles and strip_ext(name) not in titles:
                extra.append((rel, name))
    for rel, name in extra:
        print("  EXTRA %s" % ((rel + "/" if rel else "") + name))
    print("  —— 多余 %d 项" % len(extra))

    # ---------- 内容级 ----------
    content_diff, content_ok, content_skip = [], 0, 0
    if args.content or args.content_doc:
        print()
        print("=== 内容级：同名但内容不同 ===")
        for it in cloud:
            if it["kind"] == "folder":
                continue
            want_doc = it["kind"] == "doc"
            if want_doc and not args.content_doc:
                continue
            if not want_doc:
                if not args.content:
                    continue
                e = it["ext"] or strip_ext(it["title"]).rsplit(".", 1)[-1].lower()
                if e not in content_ext:
                    continue
            lp = local_path_for(it)
            if lp is None:
                continue  # 缺失已在上面报过
            label = (it["rel"] + "/" if it["rel"] else "") + it["title"]
            rel_local = os.path.relpath(lp, local_dir)
            if want_doc:
                # doc 与 .md 格式天然不同，按「文本骨架长度」宽松比对
                a = skeleton(sp.doc_text(it["id"]))
                b = skeleton(norm_text(open(lp, encoding="utf-8", errors="replace").read()))
                if not a or not b:
                    content_skip += 1
                    continue
                ratio = abs(len(a) - len(b)) / max(len(a), len(b))
                if ratio <= 0.03:
                    content_ok += 1
                else:
                    content_diff.append((label, rel_local,
                                         "正文字符 %d vs %d（差 %.1f%%）" % (len(a), len(b), ratio * 100)))
                continue
            cp = sp.download(it["id"], it["title"], args.max_size)
            if cp is None:
                content_skip += 1
                continue
            cdig, ldig = digest_file(cp, args.raw), digest_file(lp, args.raw)
            if cdig is None or ldig is None:
                content_skip += 1
                continue
            if cdig == ldig:
                content_ok += 1
            else:
                content_diff.append((label, rel_local, "内容不一致"))
        for c, l, note in content_diff:
            print("  DIFF  %-52s  <->  %s   %s" % (c, l, note))
        print("  —— 一致 %d 项 / 不同 %d 项 / 跳过 %d 项" % (content_ok, len(content_diff), content_skip))

    # ---------- 摘要 ----------
    print()
    print("=== 摘要 ===")
    folders = [c for c in cloud if c["kind"] == "folder"]
    local_files = sum(len(fn) for dp, dn, fn in os.walk(local_dir)
                      if os.path.relpath(dp, local_dir).split(os.sep)[0] not in local_only
                      and "__pycache__" not in os.path.relpath(dp, local_dir).split(os.sep))
    print("  云端 %d 个节点（%d 目录 / %d 叶子）｜本地 %d 个文件（不含本地专有项）"
          % (len(cloud), len(folders), len(cloud) - len(folders), local_files))
    print("  接口调用 %d 次" % sp.calls)

    return 0 if (not missing and not extra and not content_diff) else 1


if __name__ == "__main__":
    sys.exit(main())
