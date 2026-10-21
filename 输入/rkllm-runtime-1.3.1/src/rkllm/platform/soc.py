"""SoC 平台检测（设计 §4.4 `platform/` 的另一半：soc）。

为什么必须有：`model.platform=auto` 若原样传给 worker，`engine.py` 里
    base_mask = CPU_MASK_HIGH if platform in BIG_CORE_PLATFORMS else CPU_MASK_LOW
会因为 "auto" 不在名单里而落到**小核**（A55），prefill/解码慢 3-4 倍。
2026-10-21 真机实测：auto 未解析时 4B 首 token 3.1 s，解析为 rk3588 后 0.76 s。

检测顺序：
1. `/proc/device-tree/compatible`（Armbian/Debian 上最可靠）；
2. `/proc/cpuinfo` 的 Hardware/Model 字段兜底；
3. 都不认识 → rk3588（本项目唯一已验收平台，且它用大核，宁可快）。
"""
from __future__ import annotations

import re

__all__ = ["KNOWN_PLATFORMS", "detect_platform", "resolve_platform"]

KNOWN_PLATFORMS = ("rk3588", "rk3576", "rk3562", "rv1126b")

# device-tree compatible 里的片段 → 平台名（按特异性从高到低）
_TOKENS = (
    ("rv1126b", "rv1126b"),
    ("rk3576", "rk3576"),
    ("rk3562", "rk3562"),
    ("rk3588", "rk3588"),
)

_CPUINFO_RE = re.compile(r"(rk3588|rk3576|rk3562|rv1126b)", re.I)


def detect_platform() -> str:
    """返回 KNOWN_PLATFORMS 之一。"""
    try:
        with open("/proc/device-tree/compatible", "rb") as fh:
            compat = fh.read().decode("utf-8", "replace").lower()
    except OSError:
        compat = ""
    for token, name in _TOKENS:
        if token in compat:
            return name
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        head = ""
    match = _CPUINFO_RE.search(head)
    if match:
        return match.group(1).lower()
    return "rk3588"


def resolve_platform(configured: str) -> str:
    """`auto` → 实测平台；显式值原样返回（允许强制指定，便于排障）。"""
    configured = (configured or "auto").strip().lower()
    if configured in KNOWN_PLATFORMS:
        return configured
    return detect_platform()
