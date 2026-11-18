"""Token 计数：可选精确 tokenizer + 自校准估算。

**为什么要三档**：板端 Debian 没有 `python3-tokenizers`，`.rkllm` 也不带
`tokenizer.json`，所以"拿一个 HF tokenizer 直接数"在这台机器上不可行。但 runtime
每次推理都会在 `perf` 里回报 **prefill_tokens（真值）**——用它反过来校正"字节→token"
的比例，就能让估算跟着**实际模型**收敛：不需要词表，也不需要新依赖。

三档优先级：

1. **精确**：装了 `tokenizers` 且能定位 `tokenizer.json`（模型同目录的
   `<model>.tokenizer.json` / `tokenizer.json`，或环境变量 `RKLLM_TOKENIZER_JSON`）。
   开发机或将来有词表时启用；板端默认不启用。
2. **自校准估算**（板端主路径）：`calibrate(prompt_bytes, actual_tokens)` 用 EWMA
   更新 bytes/token，夹在 [1.0, 8.0]；一次请求就能明显纠偏，几次后趋于稳定。
3. **冷启动默认**：3.0 字节/token（历史口径：中文 ≈1 token/字）。

口径统一：预算、usage、测试都走 `count_tokens()`。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

__all__ = ["BYTES_PER_TOKEN", "count_tokens", "estimate_tokens", "calibrate",
           "configure", "stats", "reset", "exact_available"]

log = logging.getLogger("rkllm.tokens")

# 冷启动默认值：中文 1 token/字（3 字节）；英文会被高估——宁可少生成
BYTES_PER_TOKEN = 3.0
# 比例边界：英文约 4 字节/token；极短提示会因为 BOS/模板开销低于 1（"hi" → 3~4 token），
# 所以下界放到 0.8 而不是 1.0。
_MIN_RATIO, _MAX_RATIO = 0.8, 8.0
# 样本门槛：太短的提示比例噪声极大（模板开销占比过高），不参与校准
_MIN_BYTES = 64
_EWMA_ALPHA = 0.3          # 新样本权重：一次请求就能明显纠偏，又不被单次异常带跑

_lock = threading.Lock()
_ratio = BYTES_PER_TOKEN
_samples = 0
_backend = "estimate"      # estimate / tokenizers
_encoder = None
_source: str | None = None


# -- 精确路径（可选） --------------------------------------------------------

def _discover(model_path: str | None) -> Path | None:
    env = (os.environ.get("RKLLM_TOKENIZER_JSON") or "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    if not model_path:
        return None
    model = Path(model_path)
    for candidate in (model.with_suffix(".tokenizer.json"),
                      model.parent / "tokenizer.json"):
        if candidate.is_file():
            return candidate
    return None


def configure(model_path: str | None = None) -> dict:
    """尝试启用精确 tokenizer；不可用就留在估算档（返回当前状态）。"""
    global _backend, _encoder, _source
    path = _discover(model_path)
    if path is None:
        return stats()
    try:
        from tokenizers import Tokenizer      # 可选依赖（Debian 无此包）
    except ImportError:
        log.info("发现 %s 但未安装 tokenizers，继续用自校准估算", path)
        return stats()
    try:
        encoder = Tokenizer.from_file(str(path))
    except Exception as exc:                  # noqa: BLE001 - 词表损坏不该拦启动
        log.warning("加载 tokenizer 失败（%s）：%r；继续用估算", path, exc)
        return stats()
    with _lock:
        _encoder = encoder
        _backend = "tokenizers"
        _source = str(path)
    log.info("token 计数使用精确 tokenizer：%s", path)
    return stats()


def exact_available() -> bool:
    return _backend == "tokenizers" and _encoder is not None


# -- 计数 -------------------------------------------------------------------

def _estimate(text: str, ratio: float) -> int:
    if not text:
        return 0
    return max(1, int(len(text.encode("utf-8")) / ratio))


def count_tokens(text: str) -> int:
    """精确优先；否则按当前（可能已校准的）bytes/token 估算。"""
    if not text:
        return 0
    encoder = _encoder
    if encoder is not None:
        try:
            return len(encoder.encode(text, add_special_tokens=False).ids)
        except Exception:                     # noqa: BLE001 - 静默退回估算
            pass
    with _lock:
        ratio = _ratio
    return _estimate(text, ratio)


def estimate_tokens(text: str) -> int:
    """历史别名（`pipeline.py` 与旧测试使用）。"""
    return count_tokens(text)


# -- 自校准 ----------------------------------------------------------------

def calibrate(prompt_bytes: int, actual_tokens: int) -> float:
    """用 runtime 回报的 prefill_tokens 校正比例；返回新的 bytes/token。

    `actual_tokens` 必须来自 `perf.prefill_tokens`（真值）。异常样本（≤0 或
    折算后超出 [1,8] 字节/token）直接丢弃——否则一次空提示就会把比例带偏。
    """
    global _ratio, _samples
    if exact_available() or prompt_bytes < _MIN_BYTES or actual_tokens <= 0:
        return _ratio
    observed = prompt_bytes / float(actual_tokens)
    if not (_MIN_RATIO <= observed <= _MAX_RATIO):
        return _ratio
    with _lock:
        if _samples == 0:
            _ratio = observed
        else:
            _ratio = (1 - _EWMA_ALPHA) * _ratio + _EWMA_ALPHA * observed
        _ratio = min(_MAX_RATIO, max(_MIN_RATIO, _ratio))
        _samples += 1
        ratio = _ratio
    return ratio


def stats() -> dict:
    with _lock:
        return {"backend": _backend, "bytes_per_token": round(_ratio, 3),
                "samples": _samples, "source": _source}


def reset(ratio: float = BYTES_PER_TOKEN) -> None:
    """测试用：回到冷启动状态。"""
    global _ratio, _samples, _backend, _encoder, _source
    with _lock:
        _ratio, _samples = float(ratio), 0
        _backend, _encoder, _source = "estimate", None, None
