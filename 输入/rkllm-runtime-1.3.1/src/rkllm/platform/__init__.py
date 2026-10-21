"""平台相关能力：模型档位推导、内存预算、SoC 检测、（板端）锁频。

设计 §4.4 的 platform/ 模块。本文件是包入口，只导出稳定 API。
"""
from __future__ import annotations

from .memory import (MemoryBudgetError, Plan, available_mb, check_budget,
                     kv_mb, model_size_mb, plan)
from .freq import FreqLock, apply_freq_lock
from .modelclass import CLASSES, derive_class, max_tokens_for
from .soc import KNOWN_PLATFORMS, detect_platform, resolve_platform

__all__ = ["CLASSES", "derive_class", "max_tokens_for", "MemoryBudgetError",
           "Plan", "available_mb", "check_budget", "kv_mb", "model_size_mb",
           "plan", "KNOWN_PLATFORMS", "detect_platform", "resolve_platform",
           "FreqLock", "apply_freq_lock"]
