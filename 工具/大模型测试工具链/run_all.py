#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键编排：按正确顺序跑完整套板端测试，并生成报告。

    python run_all.py --host 192.0.2.10:8080 --key KEY

为什么需要它
------------
板端 NPU 是**独占资源**，压测与仿真不能同时跑，否则结果互相污染；且极限
用例可能把服务打到崩溃。本脚本把整套流程串起来，并在每步之间做就绪等待：

    1. health_check.py      预检（服务器在线 / key 正确 / 推理链路通）
    2. llm_stress.py      压力测试
    3. llm_simulate.py    客户端行为仿真
    4. crash_probe.py      并发崩溃隔离实验
    5. make_report.py  汇总生成 report.html / report.md

任一步骤让服务崩溃时，脚本会轮询 /health 等待守护进程自动拉起后再继续。

用法
----
    python run_all.py --host 192.0.2.10:8080 --key KEY
    python run_all.py --host 192.0.2.10:8080 --key KEY --quick           # 快速全流程
    python run_all.py --host 192.0.2.10:8080 --key KEY --skip-stress     # 只跑仿真等
    python run_all.py --host 192.0.2.10:8080 --key KEY --ready-timeout 120
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from board_api import build_base, wait_ready   # 共享 HTTP 层

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(title, script, extra_args=None, timeout=None):
    timeout = timeout or 1800   # 默认单步上限 30 分钟：子脚本死锁时不至于挂死编排器
    """运行同目录下的一个子脚本，实时把输出透传到当前终端。

    返回子进程退出码（负数表示异常/超时）。
    """
    print("\n" + "=" * 70)
    print(">>> %s  (%s)" % (title, script))
    print("=" * 70)
    cmd = [sys.executable, os.path.join(HERE, script)] + (extra_args or [])
    try:
        proc = subprocess.run(cmd, timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print("!!! %s 超时（%ss）" % (script, timeout))
        return -1
    except Exception as exc:  # noqa: BLE001
        print("!!! %s 启动失败：%r" % (script, exc))
        return -2


# wait_ready 由 board_api 提供（空 200 也视为存活，vLLM 就是那么回的）。


def check_board_api_sync():
    """两份 board_api.py 必须一致。

    工具链与连接器共用同一份 HTTP 实现，但连接器要能整目录拷走，所以各自存
    一份物理副本。这里做内容比对，防止改了一边忘了另一边——两边判读不一致
    正是这次要消灭的问题。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    ours = os.path.join(here, "board_api.py")
    theirs = os.path.join(here, "..", "..", "本地模型连接器", "board_api.py")
    if not os.path.exists(theirs):
        return True                     # 只部署了工具链：无从比较
    try:
        with open(ours, "rb") as fh:
            mine = fh.read()
        with open(theirs, "rb") as fh:
            other = fh.read()
    except OSError as exc:
        print("    无法比较 board_api.py：%r" % exc)
        return True
    if mine != other:
        print("    !! 两份 board_api.py 内容不一致：")
        print("       %s" % ours)
        print("       %s" % theirs)
        print("       改一份后必须同步另一份，否则连接器与工具链的判读会不同。")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("RKLLM_HOST"))
    parser.add_argument("--key", default=os.environ.get("RKLLM_KEY") or os.environ.get("RKLLM_API_KEY", ""))
    parser.add_argument("--quick", action="store_true", help="快速模式（传给压测/仿真）")
    parser.add_argument("--skip-stress", action="store_true", help="跳过压力测试")
    parser.add_argument("--skip-simulate", action="store_true", help="跳过仿真测试")
    parser.add_argument("--skip-crash", action="store_true", help="跳过崩溃复现实验")
    parser.add_argument("--limits", action="store_true",
                        help="压测中包含可能压垮服务的 E 段（默认跳过）")
    parser.add_argument("--vu", type=int, default=3, help="仿真虚拟用户数")
    parser.add_argument("--ready-timeout", type=float, default=90, help="步骤间等待服务恢复的上限秒数")
    args = parser.parse_args()

    common = ["--host", args.host, "--key", args.key]
    check_board_api_sync()
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not args.host:
        parser.error("必须用 --host 或环境变量 RKLLM_HOST 指定板端地址（示例地址不作为默认，防止静默打向不可达目标）")

    # 1) 预检：服务不可达 / key 错误 时提前失败，避免后续白跑
    if run_step("预检 健康检查", "health_check.py", common) != 0:
        print("\n预检未通过，终止。请确认服务在线且 key 正确。")
        return 1

    # 退出码语义：0=全绿；2=完成但有 finding（子脚本返回 2）；1=有步骤失败/未完成
    codes = []

    # 2) 压力测试
    if not args.skip_stress:
        stress_args = list(common)
        if args.quick:
            stress_args.append("--quick")
        if not args.limits:
            stress_args.append("--skip-limits")   # 默认不跑会压垮服务的 E 段
        codes.append(("压力测试", run_step("压力测试", "llm_stress.py", stress_args)))
        wait_ready(args.host, args.key, args.ready_timeout)

    # 3) 仿真测试
    if not args.skip_simulate:
        sim_args = list(common) + ["--vu", str(args.vu)]
        if args.quick:
            sim_args += ["--vu-seconds", "20"]
        codes.append(("仿真测试", run_step("仿真测试", "llm_simulate.py", sim_args)))
        wait_ready(args.host, args.key, args.ready_timeout)

    # 4) 崩溃隔离实验
    if not args.skip_crash:
        codes.append(("崩溃复现", run_step("崩溃复现", "crash_probe.py", common)))
        wait_ready(args.host, args.key, args.ready_timeout)

    # 5) 汇总报告
    run_step("生成报告", "make_report.py", [])

    findings = [(n, c) for n, c in codes if c == 2]
    failed = [(n, c) for n, c in codes if c not in (0, 2)]

    print("\n" + "=" * 70)
    print("全流程完成。报告：%s" % os.path.join(HERE, "report.html"))
    print("步骤退出码（0=全绿 2=有finding）：")
    for n, c in codes:
        print("  %-10s -> %d" % (n, c))
    print("=" * 70)
    if failed:
        print("有步骤失败（退出码非 0/2）：", failed)
        return 1
    if findings:
        print("完成但有 finding（详见各段结果）：", [n for n, _ in findings])
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
