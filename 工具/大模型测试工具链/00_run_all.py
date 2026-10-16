#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键编排：按正确顺序跑完整套板端测试，并生成报告。

    python 00_run_all.py --host 192.0.2.10:8080 --key KEY

为什么需要它
------------
板端 NPU 是**独占资源**，压测与仿真不能同时跑，否则结果互相污染；且极限
用例可能把服务打到崩溃。本脚本把整套流程串起来，并在每步之间做就绪等待：

    1. 01_health_check.py      预检（服务器在线 / key 正确 / 推理链路通）
    2. 02_llm_stress.py      压力测试
    3. 03_llm_simulate.py    客户端行为仿真
    4. 04_crash_probe.py      并发崩溃隔离实验
    5. 05_make_report.py  汇总生成 report.html / report.md

任一步骤让服务崩溃时，脚本会轮询 /health 等待守护进程自动拉起后再继续。

用法
----
    python 00_run_all.py --host 192.0.2.10:8080 --key KEY
    python 00_run_all.py --host 192.0.2.10:8080 --key KEY --quick           # 快速全流程
    python 00_run_all.py --host 192.0.2.10:8080 --key KEY --skip-stress     # 只跑仿真等
    python 00_run_all.py --host 192.0.2.10:8080 --key KEY --ready-timeout 120
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

# 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(title, script, extra_args=None, timeout=None):
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


def wait_ready(host, key, timeout=90):
    """轮询 /health，等待服务就绪。返回是否就绪。

    用于某一步把服务打崩之后、下一步开始之前的等待。
    """
    url = "http://%s/health" % host
    started = time.time()
    while time.time() - started < timeout:
        try:
            req = urllib.request.Request(url, method="GET",
                                         headers={"Authorization": "Bearer " + key} if key else {})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    waited = time.time() - started
                    if waited > 1:
                        print("    服务已恢复（等待 %.1fs）" % waited)
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    print("    !! 服务在 %ss 内未恢复" % timeout)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("RKLLM_HOST", "192.0.2.10:8080"))
    parser.add_argument("--key", default=os.environ.get("RKLLM_KEY", ""))
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
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1) 预检：服务不可达 / key 错误 时提前失败，避免后续白跑
    if run_step("预检 健康检查", "01_health_check.py", common) != 0:
        print("\n预检未通过，终止。请确认服务在线且 key 正确。")
        return 1

    # 2) 压力测试
    if not args.skip_stress:
        stress_args = list(common)
        if args.quick:
            stress_args.append("--quick")
        if not args.limits:
            stress_args.append("--skip-limits")   # 默认不跑会压垮服务的 E 段
        run_step("压力测试", "02_llm_stress.py", stress_args)
        wait_ready(args.host, args.key, args.ready_timeout)

    # 3) 仿真测试
    if not args.skip_simulate:
        sim_args = list(common) + ["--vu", str(args.vu)]
        if args.quick:
            sim_args += ["--vu-seconds", "20"]
        run_step("仿真测试", "03_llm_simulate.py", sim_args)
        wait_ready(args.host, args.key, args.ready_timeout)

    # 4) 崩溃隔离实验
    if not args.skip_crash:
        run_step("崩溃复现", "04_crash_probe.py", common)
        wait_ready(args.host, args.key, args.ready_timeout)

    # 5) 汇总报告
    run_step("生成报告", "05_make_report.py", [])

    print("\n" + "=" * 70)
    print("全流程完成。报告：%s" % os.path.join(HERE, "report.html"))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
