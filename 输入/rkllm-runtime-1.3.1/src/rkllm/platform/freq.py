"""CPU / NPU 锁频（model.freq_lock / model.freq_helper）。

为什么需要：RK3588 的 NPU devfreq 与 CPU cpufreq 默认是 ondemand/simple_ondemand，
推理首 token 阶段会经历一次升频滞后（实测首 token 0.8 s 里约有 0.1–0.2 s 是等
频率爬升）；长任务里 governor 也可能在中途降频，导致吞吐抖动。锁频把这两处
固定到最高可用频率，代价是空载功耗上升——所以它是可配置项，默认开（板端供电
场景），不适合电池设备。

两种实现，按配置选择：
- `model.freq_helper` 指向脚本时，交给脚本处理（不同板子的 sysfs 布局差异大，
  脚本是逃生舱），本模块只负责调用与记录；
- 否则内建 sysfs 方案：NPU devfreq 的 governor 设 userspace 并把 min=max 设为
  最高可用频率；CPU 逐核设 performance。

安全边界：非 root 只告警不阻断（服务必须能起来）；任何 sysfs 写入失败都回滚
已改的项并告警——锁频失败不该让服务启动失败。
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

__all__ = ["FreqLock", "apply_freq_lock"]

_NPU_GLOBS = ("/sys/class/devfreq/*npu*", "/sys/class/devfreq/*.npu")
_CPU_CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq"


def _write(path: str, value: str) -> bool:
    try:
        with open(path, "w", encoding="ascii") as fh:
            fh.write(value)
        return True
    except OSError:
        return False


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="ascii") as fh:
            return fh.read().strip()
    except OSError:
        return ""


class FreqLock:
    """记录改动以便恢复；字段为空表示该项未动。"""

    def __init__(self) -> None:
        self.npu_governor: tuple[str, str] | None = None     # (path, 原值)
        self.npu_limits: tuple[str, str, str] | None = None  # (min_path, max_path, 原值)
        self.cpu_governors: list[tuple[str, str]] = []
        self.external_cmd: str | None = None
        self.locked = False

    # -- 上锁 ---------------------------------------------------------------
    def _lock_npu(self) -> bool:
        nodes = []
        for pattern in _NPU_GLOBS:
            nodes.extend(glob.glob(pattern))
        nodes = sorted(set(nodes))
        if not nodes:
            sys.stderr.write("[rkllm] 锁频：未找到 NPU devfreq 节点，跳过 NPU 锁频\n")
            return False
        node = nodes[0]
        gov_path = os.path.join(node, "governor")
        min_path = os.path.join(node, "min_freq")
        max_path = os.path.join(node, "max_freq")
        freqs = _read(os.path.join(node, "available_frequencies")).split()
        target = max(freqs, key=int) if freqs else ""
        if not target:
            target = _read(max_path)
        if not target:
            sys.stderr.write("[rkllm] 锁频：%s 无可用频率，跳过\n" % node)
            return False

        orig_gov = _read(gov_path)
        orig_min, orig_max = _read(min_path), _read(max_path)
        ok = _write(gov_path, "userspace")
        ok = _write(min_path, target) and ok
        ok = _write(max_path, target) and ok
        if not ok:
            # 回滚，保持设备原状
            if orig_min:
                _write(min_path, orig_min)
            if orig_max:
                _write(max_path, orig_max)
            if orig_gov:
                _write(gov_path, orig_gov)
            sys.stderr.write("[rkllm] 锁频：NPU 写入失败（需要 root），已回滚\n")
            return False
        self.npu_governor = (gov_path, orig_gov)
        self.npu_limits = (min_path, max_path, "%s %s" % (orig_min, orig_max))
        sys.stderr.write("[rkllm] 锁频：NPU %s -> %s Hz (userspace)\n"
                         % (os.path.basename(node), target))
        return True

    def _lock_cpu(self) -> bool:
        locked = 0
        for cf in sorted(glob.glob(_CPU_CPUFREQ_GLOB)):
            gov_path = os.path.join(cf, "scaling_governor")
            orig = _read(gov_path)
            if not orig or orig == "performance":
                continue
            if _write(gov_path, "performance"):
                self.cpu_governors.append((gov_path, orig))
                locked += 1
        if locked:
            sys.stderr.write("[rkllm] 锁频：%d 个 CPU 核心 -> performance\n" % locked)
        return bool(locked)

    def acquire(self) -> "FreqLock":
        self.locked = self._lock_npu() or self._lock_cpu()
        return self

    # -- 恢复 ---------------------------------------------------------------
    def restore(self) -> None:
        for gov_path, orig in self.cpu_governors:
            if orig:
                _write(gov_path, orig)
        if self.npu_limits:
            min_path, max_path, orig = self.npu_limits
            if orig:
                parts = orig.split()
                if len(parts) == 2:
                    _write(min_path, parts[0])
                    _write(max_path, parts[1])
        if self.npu_governor:
            gov_path, orig = self.npu_governor
            if orig:
                _write(gov_path, orig)
        self.cpu_governors = []
        self.npu_limits = None
        self.npu_governor = None
        self.locked = False


def apply_freq_lock(enabled: bool, helper: str = "") -> FreqLock:
    """按配置上锁。返回对象总是可用；调用方负责在退出时 restore()。"""
    lock = FreqLock()
    if not enabled:
        return lock
    # 非 POSIX 平台没有 geteuid：锁频无意义，跳过但说清楚（不能让 serve 崩）
    getuid = getattr(os, "geteuid", None)
    if getuid is None:
        sys.stderr.write("[rkllm] 锁频：当前平台无 os.geteuid，跳过\n")
        return lock
    if getuid() != 0:
        sys.stderr.write("[rkllm] 锁频：非 root 用户，跳过（不阻断启动）\n")
        return lock
    if helper and helper != "auto":
        if not os.path.isfile(helper):
            sys.stderr.write("[rkllm] 锁频：freq_helper 不存在（%s），改用内建方案\n"
                             % helper)
        else:
            try:
                subprocess.run([helper], check=False, timeout=30)
                lock.external_cmd = helper
                sys.stderr.write("[rkllm] 锁频：已交给外部脚本 %s\n" % helper)
                return lock
            except (OSError, subprocess.SubprocessError) as exc:
                sys.stderr.write("[rkllm] 锁频：外部脚本失败（%s），改用内建方案\n" % exc)
    return lock.acquire()
