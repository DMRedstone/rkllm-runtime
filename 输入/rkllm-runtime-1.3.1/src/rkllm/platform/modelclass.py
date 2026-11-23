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

**但 0 不能进 worker**（2026-11-20 审核 P1-C2）：worker 的 argparse 会把
`--max-tokens 0` 当成非法值直接 exit 2，表现成"worker 起不来"。因此凡是要把
"档位默认"变成真实参数的地方（cli_serve / cli_engine）都必须走
`effective_max_tokens_for()`——它把无内置档位回退到 `CONSERVATIVE_MAX_TOKENS`
（4B 实测值 512，保守正数），既保证 8B/显式 14B/27B 可启动，也保留
`max_tokens_for()` 返回 0 的"不内置"语义给 config show/文档使用。

体积阈值是经验值：W8A8 量化下 1B ≈ 1 GiB。低比特量化（W4）会让体积偏小，
此时第 2 步通常已命中；只有文件名无参数量时才落到第 3 步，因此风险有限。
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ["CLASSES", "derive_class", "max_tokens_for",
           "max_tokens_limit_for", "cache_max_mb_for", "effective_max_tokens_for",
           "DEFAULT_MAX_TOKENS", "MAX_TOKENS_LIMIT", "CACHE_MAX_MB",
           "CONSERVATIVE_MAX_TOKENS"]

# 内置默认档位（以参数量命名）：仅 4b / 7b。
CLASSES = ("4b", "7b")

DEFAULT_MAX_TOKENS = {"4b": 512, "7b": 384}
MAX_TOKENS_LIMIT = {"4b": 2048, "7b": 1536}
CACHE_MAX_MB = {"4b": 512, "7b": 1024}

# P1-C2：无内置档位（8b/14b/27b/未知）的保守正数默认。取 4B 实测值——
# 生成长度是上限而非目标值，比 7B 的 384 更宽松、比"猜"大模型更安全；
# 关键是恒 > 0，worker 不会再因 --max-tokens 0 而 argparse exit 2。
CONSERVATIVE_MAX_TOKENS = 512

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


def effective_max_tokens_for(model_class: str) -> int:
    """真正下发给 worker/pipeline 的档位默认生成上限（恒为正数）。

    P1-C2：无内置档位时回退到 CONSERVATIVE_MAX_TOKENS，而不是返回 0。
    用户显式配置 model.max_tokens 时仍由调用方优先使用显式值。
    """
    return max_tokens_for(model_class) or CONSERVATIVE_MAX_TOKENS


def max_tokens_limit_for(model_class: str) -> int:
    """档位单次输出硬上限；无内置档位返回 0（需手动配置）。"""
    return MAX_TOKENS_LIMIT.get(model_class, 0)


def cache_max_mb_for(model_class: str) -> int:
    """档位前缀缓存上限（MB）；无内置档位返回 0（需手动配置）。"""
    return CACHE_MAX_MB.get(model_class, 0)
