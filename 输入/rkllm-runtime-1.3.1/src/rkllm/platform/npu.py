"""NPU 驱动探测（设计 §4.4 platform/ 的驱动侧）。

为什么不能只看 `/dev/rknpu`：0.9.x 及更新的 rknpu 驱动把 NPU 注册成
DRM render node，**根本不创建字符设备**，只查 `/dev/rknpu` 会在完全正常的
板子上报“没有 NPU”。探测顺序按可信度从高到低：

1. `/sys/kernel/debug/rknpu/version`（最权威，能拿到驱动版本）；
2. debugfs 目录存在（版本不可读时仍算可用）；
3. `/sys/class/drm/renderD*/device/driver` 指向 rknpu（新驱动的正常形态）；
4. `/dev/rknpu`（旧驱动的设备节点）；
5. `/sys/module/rknpu`（模块已加载、节点尚未创建）。

只做只读探测，不写 sysfs；非 Linux 平台自然落到“未找到”，不抛异常。
"""
from __future__ import annotations

import os

__all__ = ["npu_status", "NPU_DEBUGFS", "LEGACY_DEVICE", "DRM_CLASS", "MODULE_PATH"]

NPU_DEBUGFS = "/sys/kernel/debug/rknpu"
LEGACY_DEVICE = "/dev/rknpu"
DRM_CLASS = "/sys/class/drm"
MODULE_PATH = "/sys/module/rknpu"


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def npu_status() -> tuple[bool, str]:
    """返回 (可用, 说明)。说明面向运维，可直接打印。"""
    version = _read_text(os.path.join(NPU_DEBUGFS, "version"))
    if version:
        return True, version.replace("\n", " ")

    if os.path.isdir(NPU_DEBUGFS):
        return True, "present at %s (version not readable)" % NPU_DEBUGFS

    try:
        entries = sorted(os.listdir(DRM_CLASS))
    except OSError:
        entries = []
    for entry in entries:
        if not entry.startswith("renderD"):
            continue
        target = os.path.realpath(
            os.path.join(DRM_CLASS, entry, "device", "driver"))
        if "rknpu" in target.lower():
            return True, "bound to /dev/dri/%s" % entry

    if os.path.exists(LEGACY_DEVICE):
        return True, "present at %s (legacy device node)" % LEGACY_DEVICE
    if os.path.isdir(MODULE_PATH):
        return True, "kernel module loaded, no device node yet"

    return False, (
        "no NPU driver found: neither %s/version, nor a DRM render node "
        "bound to rknpu, nor %s" % (NPU_DEBUGFS, LEGACY_DEVICE))
