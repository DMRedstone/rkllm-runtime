#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""space_rename_strip —— 批量剥离云端 doc 标题中的时间戳（前缀建档日期 + 后缀最后修改日期）。

背景：项目内容迁入 git 仓库后，版本由提交历史表达，项目文档标题不再携带时间戳；
行为类文档（如「智能体行为记录」夹）仍保留时间戳，用不跑该子树（或 --skip）排除。

用法：
  python space_rename_strip.py --space-root <节点id> [--space-id <空间id>]
                               [--skip 名称 ...] [--apply]
默认 dry-run：仅打印「旧标题 -> 新标题」映射；--apply 才真正调用 rename-node。
纯标准库；dry-run 只读；自动定位 space_api.py（--space-api → $SPACE_API → 插件目录）。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

PAT = re.compile(r'^(\d{8})\s+(.+?)（\d{4}\.\d{1,2}\.\d{1,2}）$')


def find_space_api(explicit):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get('SPACE_API'):
        cands.append(os.environ['SPACE_API'])
    home = os.path.expanduser('~')
    cands += [
        os.path.join(home, '.workbuddy', 'plugins', 'cache', 'workbuddy-builtin',
                     'skills', 'library', 'space_api.py'),
        r'C:/Program Files/WorkBuddy/resources/app.asar.unpacked/resources/plugins/'
        r'workbuddy-builtin/skills/library/space_api.py',
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    for root, dirs, files in os.walk(os.path.join(home, '.workbuddy', 'plugins')):
        if 'space_api.py' in files:
            return os.path.join(root, 'space_api.py')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--space-root', required=True, help='起始节点 id（递归其子树）')
    ap.add_argument('--space-id', default=None)
    ap.add_argument('--skip', action='append', default=[], help='跳过的节点名（含其子树）')
    ap.add_argument('--space-api', default=None)
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--apply', action='store_true', help='真正执行 rename（默认 dry-run）')
    a = ap.parse_args()
    api = find_space_api(a.space_api)
    if not api:
        print('找不到 space_api.py，请用 --space-api 指定', file=sys.stderr)
        return 2
    python = sys.executable
    skip = set(a.skip)

    def api_call(args):
        for _ in range(a.retries):
            cmd = [python, api] + args
            if a.space_id:
                cmd += ['--space-id', a.space_id]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            out = (p.stdout or '').strip()
            i = out.find('{')
            if i >= 0:
                try:
                    return json.JSONDecoder().raw_decode(out[i:])[0]
                except Exception:
                    pass
            time.sleep(0.8)
        return None

    renames = []

    def walk(pid, is_root):
        d = api_call(['space.workspace.list-nodes', '--limit', '200', '--parent-node-id', pid])
        if not d:
            return
        for x in d['data']['nodes']:
            t, k = x['title'], x['kind']
            if is_root and t in skip:
                continue
            if k == 'doc':
                m = PAT.match(t)
                if m:
                    renames.append((x['id'], t, m.group(2).strip()))
            elif k == 'folder':
                walk(x['id'], False)

    walk(a.space_root, True)
    print('待改名 doc：%d' % len(renames))
    for nid, old, new in renames:
        print('  %s\n    -> %s' % (old, new))
    if not renames:
        return 0
    if not a.apply:
        print('（dry-run；加 --apply 执行）')
        return 0
    ok = fail = 0
    for nid, old, new in renames:
        r = api_call(['space.workspace.rename-node', '--node-id', nid, '--title', new])
        if r and r.get('data', {}).get('title') == new:
            ok += 1
        else:
            fail += 1
            print('FAIL:', old.encode('ascii', 'replace').decode())
    print('完成：成功 %d，失败 %d' % (ok, fail))
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
