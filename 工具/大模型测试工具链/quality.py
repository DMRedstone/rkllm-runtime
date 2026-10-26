# -*- coding: utf-8 -*-
"""内容形态分析：截断、重复、乱码、流式一致性。

本模块**不判断语义**——改写得好不好、摘要抓没抓到重点，只有人能判。它只回答
"这段输出在形态上还能不能用"：

* 被截断（`finish_reason=length`，或请求 max_tokens 与服务端实际输出严重不符）
* 塌成重复词流（本项目实测：输出越过约 1000 token 后开始重复同一词组）
* 空响应 / 只剩空白
* 乱码（多字节字符在字节预算处被切断，出现 U+FFFD）
* 流式拼接结果与非流式不一致（丢字、重复首字、尾块缺失）

纯标准库、无 HTTP 依赖，供 `llm_stress` 采集与 `make_report` 渲染共用。
"""
from __future__ import annotations

import math
import re
from collections import Counter

# 多字节字符被切断时 decode(errors="replace") 会产生这个字符
REPLACEMENT = "\ufffd"


# ---------------------------------------------------------------------------
# 截断与 overrun
# ---------------------------------------------------------------------------

def collect_sample(requested, finish_reason, completion_tokens, text, elapsed=None):
    """把一次调用的观测结果整理成样本，供下面的统计函数使用。"""
    return {
        "requested": int(requested or 0),
        "finish_reason": finish_reason,
        "completion_tokens": int(completion_tokens or 0),
        "chars": len(text or ""),
        "elapsed_s": round(elapsed, 3) if elapsed else None,
        "empty": not (text or "").strip(),
    }


def finish_reason_stats(samples):
    """统计 finish_reason 分布 → {"stop": n, "length": n, "other": {...}, "truncated_pct": x}。

    `length` 是"被硬截断"，也是内容不可用最常见的形态：模型还没说完就被切断。
    服务端遵守 max_tokens：触顶时 `finish_reason=length`；`stop` 只表示模型
    自然结束，是否超过请求上限仍要看 overrun。
    """
    counts = Counter()
    for s in samples:
        counts[str(s.get("finish_reason"))] += 1
    total = sum(counts.values()) or 1
    truncated = sum(n for k, n in counts.items() if k not in ("stop", "None"))
    return {
        "counts": dict(counts),
        "total": sum(counts.values()),
        "truncated": truncated,
        "truncated_pct": round(100.0 * truncated / total, 1),
        "empty": sum(1 for s in samples if s.get("empty")),
    }


def overrun_stats(samples):
    """实际输出 token / 请求 max_tokens 的分布。

    配置项 `overrun` 就是按这个比值定的（客户端据此硬截断）。没有实测值时只能猜，
    猜小了会误切正常输出，猜大了会放过失控输出。
    """
    ratios = []
    for s in samples:
        requested = s.get("requested") or 0
        actual = s.get("completion_tokens") or 0
        if requested > 0 and actual > 0:
            ratios.append(actual / float(requested))
    if not ratios:
        return {"count": 0}
    ratios.sort()
    return {
        "count": len(ratios),
        "median": round(ratios[len(ratios) // 2], 2),
        "p90": round(ratios[min(len(ratios) - 1, int(len(ratios) * 0.9))], 2),
        "max": round(ratios[-1], 2),
        "over_1": sum(1 for r in ratios if r > 1.0),
        "over_1_5": sum(1 for r in ratios if r > 1.5),
    }


# ---------------------------------------------------------------------------
# 重复与退化
# ---------------------------------------------------------------------------

def _ngram_repeat_ratio(text, n=8):
    """n-gram 重复率 = 1 - 唯一窗口数 / 总窗口数。

    取 n=8 而不是 n=2/3：中文里"的、了、是"天然高频，短窗口的重复率对正常文本
    也很高，分不出退化。八字符的短语整段复现才是塌陷特征。
    """
    if len(text) < n * 2:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1.0 - len(set(grams)) / float(len(grams))


def longest_repeat_run(text, max_unit=60):
    """最长"整段复现"的长度（字符数）。

    找形如 `abcabcabc` 的最长单元。退化输出的典型形态是同一句话连续出现三五次，
    这个值会比正常文本高一个量级。
    """
    if not text:
        return 0
    best = 0
    for unit_len in range(6, min(max_unit, len(text) // 2) + 1):
        pattern = re.compile(r"(.{%d})\1{1,}" % unit_len)
        for match in pattern.finditer(text):
            best = max(best, len(match.group(0)))
    return best


def char_entropy(text):
    """字级香农熵（bits/char）。

    中文正常输出大致 4–6；塌成重复词流时会明显下降（信息量变小）。
    只作辅助指标：短文本的熵本身就低，不能单独判定。
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = float(len(text))
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def degen_score(text):
    """退化综合分 0–1：重复率为主（0.7），熵为辅（0.3）。

    权重这样分是因为实测区分度差一个量级——正常长文本的 8-gram 重复率 0–0.1，
    重复词流能到 0.9；而两者的字级熵只差约 6.2 vs 3.5。拿熵当半个权重，会让
    "字符种类多但整段复现"的塌陷（如 `杭州是一座很好的城市。` ×12）只判到 0.52。

    重复率先做归一：0.10 以下视为自然重复（中文固定搭配、标点节奏），0.50 视为满分。
    阈值：<0.35 正常，0.35–0.55 可疑（人工复核），>0.55 判定退化。
    """
    if not text or len(text) < 120:
        return 0.0                      # 太短，重复率与熵都不稳定
    repeat = _ngram_repeat_ratio(text)
    repeat_part = max(0.0, min(1.0, (repeat - 0.10) / 0.40))
    entropy_part = max(0.0, min(1.0, (4.0 - char_entropy(text)) / 4.0))
    return round(min(1.0, 0.7 * repeat_part + 0.3 * entropy_part), 3)


def shape_metrics(text):
    """一次性给出形态指标 → dict，便于直接塞进结果 JSON。"""
    text = text or ""
    return {
        "chars": len(text),
        "repeat_ratio": round(_ngram_repeat_ratio(text), 3),
        "longest_repeat": longest_repeat_run(text),
        "entropy": round(char_entropy(text), 2),
        "degen": degen_score(text),
        "mojibake": round(mojibake_ratio(text), 3),
        "empty": not text.strip(),
    }


def mojibake_ratio(text):
    """替换符（U+FFFD）占比。非零就说明有字符在字节边界被切断。"""
    if not text:
        return 0.0
    return text.count(REPLACEMENT) / float(len(text))


# ---------------------------------------------------------------------------
# 流式一致性
# ---------------------------------------------------------------------------

_TAIL_CHARS = "。．.!！?？、,，;；:： \n\t\r"


def _strip_tail(text):
    """去掉末尾的标点与空白，用于"只差一个句号"的比对。"""
    return text.rstrip(_TAIL_CHARS)


def stream_parity(blocking_text, streamed_text):
    """对比同一 prompt 的阻塞输出与流式拼接结果 → dict。

    流式丢字、重复首字、`[DONE]` 前缺尾块，都表现为两者不一致；而这些在只比
    TTFT 的用例里完全看不出来。
    """
    a = blocking_text or ""
    b = streamed_text or ""
    shortest = min(len(a), len(b))
    prefix = 0
    while prefix < shortest and a[prefix] == b[prefix]:
        prefix += 1
    ratio = (len(b) / float(len(a))) if a else (1.0 if not b else 0.0)
    if a == b:
        verdict = "identical"
    elif _strip_tail(a) == _strip_tail(b):
        # 只差末尾标点/空白：短文本按比例算会被误判（8 字少一个句号就是 12.5%），
        # 所以先剥掉尾部再比。
        verdict = "equivalent"
    elif ratio >= 0.98 and prefix >= shortest * 0.9:
        verdict = "equivalent"
    elif not b:
        verdict = "empty"
    else:
        verdict = "mismatch"
    return {
        "verdict": verdict,
        "blocking_chars": len(a),
        "streamed_chars": len(b),
        "len_ratio": round(ratio, 3),
        "common_prefix": prefix,
    }


# ---------------------------------------------------------------------------
# 并发响应归属（串包检测）
# ---------------------------------------------------------------------------

def check_attribution(expectations):
    """检查每个响应是否回到了发起它的那个请求。

    expectations -- [(marker, text), ...]，marker 是只出现在该请求 prompt 里的字符串。

    返回 {"total", "matched", "foreign", "missing"}：
      matched  响应里含自己的 marker（正常）
      foreign  响应里含**别人的** marker（串包——用户会拿到别人的答案）
      missing  自己的 marker 都没出现（模型没照做，或响应被截断）
    串包（foreign>0）比 missing 严重得多：前者是服务端串了请求，后者常是模型没听话。
    """
    markers = [m for m, _ in expectations]
    matched = foreign = missing = 0
    details = []
    for marker, text in expectations:
        body = text or ""
        own = marker in body
        others = [m for m in markers if m != marker and m in body]
        if others:
            foreign += 1
            details.append({"marker": marker, "foreign": others[:3]})
        elif own:
            matched += 1
        else:
            missing += 1
            details.append({"marker": marker, "foreign": []})
    return {
        "total": len(expectations),
        "matched": matched,
        "foreign": foreign,
        "missing": missing,
        "verdict": "ok" if foreign == 0 else "CROSS_TALK",
        "details": details[:5],
    }
