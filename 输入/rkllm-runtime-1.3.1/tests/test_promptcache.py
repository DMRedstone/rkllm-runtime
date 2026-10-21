"""prompt 前缀缓存（cache.* 四键）的纯函数行为。"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.worker.promptcache import PromptCache  # noqa: E402


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


def test_unrelated_prompt_misses(tmp_path):
    pc = PromptCache(str(tmp_path))
    pc.commit("完全无关的内容")
    _touch_cache(pc)
    assert pc.prepare("另一段完全不同的问题")[1] == 1


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
