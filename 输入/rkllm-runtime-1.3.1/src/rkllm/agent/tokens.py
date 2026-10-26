"""Token 估算的单一口径（无 tokenizer 时的保守折算）。

预算、usage、测试都用同一函数：UTF-8 字节 / 3。中文约 1 token/字（3 字节），
英文 4 字节/token 会被高估约 33%——宁可少生成，也不把请求送过 context。
"""

from __future__ import annotations

__all__ = ["BYTES_PER_TOKEN", "estimate_tokens"]

BYTES_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.encode("utf-8")) // BYTES_PER_TOKEN)
