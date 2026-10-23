"""prompt 前缀缓存（cache.* 四键）的纯函数行为。"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.worker.promptcache import (  # noqa: E402
    PROMPT_SIMILARITY, PromptCache, common_prefix_len)


def _touch_cache(pc: PromptCache) -> None:
    """模拟 runtime 写出的缓存文件（真实写入由 librkllmrt 完成）。"""
    with open(pc.cache_path, "wb") as fh:
        fh.write(b"kv-bytes")


def test_first_call_misses(tmp_path):
    pc = PromptCache(str(tmp_path))
    path, save = pc.prepare("第一轮")
    assert path == pc.cache_path
    assert save == 1                     # 未命中 -> 本次结束后保存


def test_repeat_prompt_hits(tmp_path):
    pc = PromptCache(str(tmp_path))
    pc.commit("第一轮：请总结这段文字")
    _touch_cache(pc)
    assert pc.prepare("第一轮：请总结这段文字") == (pc.cache_path, 0)


def test_extended_prompt_hits_prefix(tmp_path):
    """多轮对话：新一轮 prompt 以上一轮为前缀 -> 命中复用。"""
    pc = PromptCache(str(tmp_path))
    pc.commit("系统提示\n用户：第一问\n助手：第一答")
    _touch_cache(pc)
    nxt = "系统提示\n用户：第一问\n助手：第一答\n用户：第二问"
    assert pc.prepare(nxt) == (pc.cache_path, 0)


def test_unrelated_prompt_does_not_take_over_slot(tmp_path):
    """无关会话不接管单槽（O-1）：返回 None，meta 与 cache 配对保持不变。

    改造前这里是 `save=1`——runtime 会立刻用新会话的内容覆盖缓存文件，
    而 meta 随后也被改写，等于把热点会话的缓存冲掉；交错请求下形成
    「每次 miss、每次覆盖」的抖动。
    """
    pc = PromptCache(str(tmp_path))
    pc.commit("完全无关的内容")
    _touch_cache(pc)
    assert pc.prepare("另一段完全不同的问题") == (None, 0)
    assert pc.skipped_saves == 1
    assert pc.last_similarity == 0.0
    assert pc._read_meta() == "完全无关的内容"      # meta 未被改写


def test_partial_prefix_at_threshold_rebuilds(tmp_path):
    """相似度正好达到阈值 -> 同一会话的延续，允许本轮重建缓存。"""
    pc = PromptCache(str(tmp_path))
    pc.commit("abcdefgh")                          # 8 字节
    _touch_cache(pc)
    assert pc.prepare("abcdXXXX") == (pc.cache_path, 1)
    assert pc.last_similarity == PROMPT_SIMILARITY  # 4/8，等于阈值


def test_partial_prefix_below_threshold_skips(tmp_path):
    """相似度低于阈值 -> 视为另一会话，不加载也不保存。"""
    pc = PromptCache(str(tmp_path))
    pc.commit("abcdefgh")
    _touch_cache(pc)
    assert pc.prepare("aXXXXXXX") == (None, 0)     # 1/8 = 0.125
    assert pc.skipped_saves == 1


def test_similarity_high_still_misses_but_rebuilds(tmp_path):
    """接近但不等于前缀（用户在中间插了字）仍可重建缓存。

    注意不能写成末尾追加——那是严格前缀，会走 hit 分支（save=0）。
    """
    pc = PromptCache(str(tmp_path))
    pc.commit("系统提示\n用户：第一问\n助手：第一答")
    _touch_cache(pc)
    nxt = "系统提示\n用户：第一问（改）\n助手：第一答"
    path, save = pc.prepare(nxt)
    assert (path, save) == (pc.cache_path, 1)
    assert PROMPT_SIMILARITY <= pc.last_similarity < 1.0


def test_common_prefix_len_utf8_boundary():
    """LCP 按字节比较但必须落在字符边界上（不能切在多字节字符中间）。"""
    assert common_prefix_len("中文", "中") == len("中".encode("utf-8")) == 3
    assert common_prefix_len("😀a", "😀b") == len("😀".encode("utf-8")) == 4
    assert common_prefix_len("abc", "abc") == 3
    assert common_prefix_len("abc", "xyz") == 0


def test_stats_expose_similarity_and_skipped(tmp_path):
    pc = PromptCache(str(tmp_path))
    pc.commit("第一轮内容")
    _touch_cache(pc)
    pc.prepare("完全不同的另一段")
    stats = pc.stats()
    assert stats["skipped_saves"] == 1
    assert "similarity" in stats and stats["similarity"] == 0.0


def test_missing_cache_file_means_miss(tmp_path):
    """元数据在但缓存文件被删（例如别人清了目录）必须当未命中。"""
    pc = PromptCache(str(tmp_path))
    pc.commit("前缀")
    assert pc.prepare("前缀加后缀")[1] == 1


def test_invalidate_clears_slot(tmp_path):
    pc = PromptCache(str(tmp_path))
    pc.commit("前缀")
    _touch_cache(pc)
    pc.invalidate()
    assert not os.path.exists(pc.cache_path)
    assert not os.path.exists(pc.meta_path)
    assert pc.prepare("前缀")[1] == 1


def test_ttl_expiry_removes_files(tmp_path):
    pc = PromptCache(str(tmp_path), ttl_s=1)
    _touch_cache(pc)
    pc.commit("x")
    old = time.time() - 10
    for path in (pc.cache_path, pc.meta_path):
        os.utime(path, (old, old))
    pc.maintain()
    assert not os.path.exists(pc.cache_path)


def test_max_mb_evicts_oldest(tmp_path):
    pc = PromptCache(str(tmp_path), max_mb=1)      # 1 MiB
    big = tmp_path / "old.cache"
    big.write_bytes(b"\0" * (2 * 1024 * 1024))
    old = time.time() - 100
    os.utime(big, (old, old))
    pc.maintain()
    assert not big.exists()


def test_unwritable_dir_degrades_silently(tmp_path):
    """目录不可用时退化为"不用缓存"，不抛异常（缓存不能影响可用性）。"""
    target = tmp_path / "sub"
    target.mkdir()
    probe = target / "probe"
    probe.write_text("x")
    # 用一个文件当目录：必然 makedirs 失败
    pc = PromptCache(str(probe / "nested"))
    assert pc.prepare("任何内容") == (None, 0)
    assert pc.commit("任何内容") is None      # 不抛
