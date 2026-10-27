"""模型档位推导（设计 §7.1「按模型规模（4B/7B/27B）推导的一组默认参数」，F12）。

档位只影响三件事：默认 max_tokens、KV 估算系数、MCP 侧的耗时估算。
推导顺序：
1. 显式配置（model.class != auto）优先，直接采用；
2. 文件名含参数量（27b/14b/8b/7b/4b/1.5b…）→ 按参数量；
3. 文件名无信息（如 model.rkllm）→ 按文件体积分档。

体积阈值是经验值：W8A8 量化下 1B ≈ 1 GiB。低比特量化（W4）会让体积偏小，
此时第 2 步通常已命中；只有文件名无参数量时才落到第 3 步，因此风险有限。
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ["CLASSES", "derive_class", "max_tokens_for", "DEFAULT_MAX_TOKENS"]

CLASSES = ("small", "medium", "large")

# 设计 §7.1：默认 max_tokens 按档位（model.max_tokens=0 时生效）
DEFAULT_MAX_TOKENS = {"small": 2048, "medium": 1024, "large": 512}

# 体积分档（MiB）：≥12 GiB → large，≥6 GiB → medium，其余 small
_SIZE_LARGE_MB = 12 * 1024
_SIZE_MEDIUM_MB = 6 * 1024

# 文件名里的参数量（B）→ 档位；顺序有意义，长匹配优先。
# 2026-10-27 审核 P2-5：原先的 large 正则只有 200-299/30-99/12-19，漏了
# 20-29 区间——恰好包括本项目的目标 27B（"qwen3-27b" 推导不出 large，
# 只能靠文件体积兜底；一旦模型文件 < 12 GiB（低比特量化/裁剪）就会被
# 错分到 medium，KV 系数与默认生成长度都随之偏小）。
_PARAM_PATTERNS = (
    (re.compile(r"(?<!\d)(2[0-9]{2}|[2-9][0-9]|1[2-9])b(?!\d)", re.I), "large"),
    (re.compile(r"(?<!\d)(7|8|9|10|11)b(?!\d)", re.I), "medium"),
    (re.compile(r"(?<![\d.])(0\.5|1|1\.5|2|3|4|5|6)b(?![\d.])", re.I), "small"),
)


def _class_from_name(name: str) -> str | None:
    low = name.lower()
    for pattern, cls in _PARAM_PATTERNS:
        if pattern.search(low):
            return cls
    return None


def _class_from_size(size_mb: float) -> str:
    if size_mb >= _SIZE_LARGE_MB:
        return "large"
    if size_mb >= _SIZE_MEDIUM_MB:
        return "medium"
    return "small"


def derive_class(model_path: str, configured: str = "auto") -> str:
    """返回 small / medium / large。

    显式配置优先；否则按文件名参数量；再否则按体积；文件不存在且无信息 → small
    （保守：small 的 max_tokens 最大，由用户显式配置纠正）。
    """
    if configured in CLASSES:
        return configured
    if not model_path:
        return "small"
    name = Path(model_path).name
    by_name = _class_from_name(name)
    if by_name:
        return by_name
    try:
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    except OSError:
        return "small"
    return _class_from_size(size_mb)


def max_tokens_for(model_class: str) -> int:
    """档位默认生成上限（设计 §7.1）。"""
    return DEFAULT_MAX_TOKENS.get(model_class, DEFAULT_MAX_TOKENS["small"])
