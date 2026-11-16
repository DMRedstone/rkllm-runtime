"""模型档位推导（设计 §7.1，F12）。

档位以**参数量**命名（`4b` / `7b` / `14b` / `27b` …），不再使用 small/medium/large。
推导顺序：
1. 显式配置（model.class != auto）优先，直接采用；兼容旧名 small/medium/large；
2. 文件名含参数量（27b/14b/8b/7b/4b/1.5b…）→ 取该参数量作为档位名；
3. 文件名无信息（如 model.rkllm）→ 按文件体积分档（≥12GiB→14b，≥6GiB→7b，其余 4b）。

**内置默认仅覆盖 4b 与 7b 两档**：4b 取实测值，7b 取保守估计。
更大模型（如 14b/27b）不内置参数，需在配置文件中手动设置 model.max_tokens /
max_tokens_limit / cache.max_mb；本模块对无内置档位返回 0（=不设默认），由
cli_serve 打印显式告警，避免"猜一个值"造成隐性错误。

体积阈值是经验值：W8A8 量化下 1B ≈ 1 GiB。低比特量化（W4）会让体积偏小，
此时第 2 步通常已命中；只有文件名无参数量时才落到第 3 步，因此风险有限。
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ["CLASSES", "derive_class", "max_tokens_for",
           "max_tokens_limit_for", "cache_max_mb_for",
           "DEFAULT_MAX_TOKENS", "MAX_TOKENS_LIMIT", "CACHE_MAX_MB"]

# 内置默认档位（以参数量命名）：仅 4b / 7b。
CLASSES = ("4b", "7b")

DEFAULT_MAX_TOKENS = {"4b": 512, "7b": 384}
MAX_TOKENS_LIMIT = {"4b": 2048, "7b": 1536}
CACHE_MAX_MB = {"4b": 512, "7b": 1024}

# 兼容旧档位名（历史配置/旧金标）
_LEGACY_LABELS = {"small": "4b", "medium": "7b", "large": "14b"}

# 文件名里的参数量（B）→ 档位名（保留小数，如 1.5b/0.5b）
_PARAM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)b(?!\d)", re.I)

# 体积分档（MiB）：≥12 GiB → 14b，≥6 GiB → 7b，其余 4b
_SIZE_LARGE_MB = 12 * 1024
_SIZE_MEDIUM_MB = 6 * 1024


def _label(num: str) -> str:
    return str(num).lower() + "b"


def _class_from_name(name: str) -> str | None:
    match = _PARAM_RE.search(name)
    return _label(match.group(1)) if match else None


def _class_from_size(size_mb: float) -> str:
    if size_mb >= _SIZE_LARGE_MB:
        return "14b"
    if size_mb >= _SIZE_MEDIUM_MB:
        return "7b"
    return "4b"


def derive_class(model_path: str, configured: str = "auto") -> str:
    """返回参数量档位名（如 4b/7b/14b/27b）。

    显式配置优先（兼容 small/medium/large）；否则按文件名参数量；再否则按体积；
    文件不存在且无信息 → 4b（保守：内置档位中 4b 上限最大，由用户显式配置纠正）。
    """
    if configured and configured != "auto":
        value = str(configured).strip().lower()
        if value in _LEGACY_LABELS:
            return _LEGACY_LABELS[value]
        match = _PARAM_RE.fullmatch(value)
        return _label(match.group(1)) if match else value
    if not model_path:
        return "4b"
    name = Path(model_path).name
    by_name = _class_from_name(name)
    if by_name:
        return by_name
    try:
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    except OSError:
        return "4b"
    return _class_from_size(size_mb)


def max_tokens_for(model_class: str) -> int:
    """档位默认生成上限；无内置档位（14b/27b/未知）返回 0（需手动配置）。"""
    return DEFAULT_MAX_TOKENS.get(model_class, 0)


def max_tokens_limit_for(model_class: str) -> int:
    """档位单次输出硬上限；无内置档位返回 0（需手动配置）。"""
    return MAX_TOKENS_LIMIT.get(model_class, 0)


def cache_max_mb_for(model_class: str) -> int:
    """档位前缀缓存上限（MB）；无内置档位返回 0（需手动配置）。"""
    return CACHE_MAX_MB.get(model_class, 0)
