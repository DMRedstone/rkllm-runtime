"""Prompt 前缀缓存（ADR-07 / cache.* 四键）。

解决的问题：服务默认无状态，每轮都把整段 messages 全量 prefill。多轮对话
里前 N-1 轮完全重复，重算 4k 前缀在 4B 上约 1–3 s、27B 上 20–60 s。

实现：单槽（single-slot）前缀缓存。rkllm 的 `prompt_cache_params` 会把本次
prompt 的 KV 保存到文件，下一次若用同一文件则直接加载作为前缀——只有当新
prompt 以缓存内容为前缀时才有意义。所以这里维护「上一次的 prompt」：

- 命中：`new_prompt.startswith(上次 prompt)` → save=0（加载并按前缀复用）；
- 未命中：save=1（本次推理结束后由 runtime 写入缓存）。

单槽而非多槽：多轮对话是串行追加的，上一轮就是最长可复用前缀；多槽需要
会话标识，而服务按设计是无状态的。容量与 TTL 仍按配置执行，用于清理异常
增长（历史遗留文件、模型更换后失效的缓存）。

失败降级：缓存文件损坏或 runtime 拒绝加载时，engine 会清槽并重试一次不带
缓存的推理——缓存只能是加速器，不能成为可用性的单点。
"""

from __future__ import annotations

import hashlib
import os
import time
import sys

__all__ = ["PromptCache"]

_CACHE_NAME = "prefix.cache"
_META_NAME = "prefix.meta"
_STATS_NAME = "stats.json"


class PromptCache:
    """管理一个缓存目录下的单槽前缀缓存。"""

    def __init__(self, directory: str, max_mb: int = 512, ttl_s: int = 604800):
        self.dir = directory
        self.max_mb = max_mb
        self.ttl_s = ttl_s
        self.cache_path = os.path.join(directory, _CACHE_NAME)
        self.meta_path = os.path.join(directory, _META_NAME)
        self.stats_path = os.path.join(directory, _STATS_NAME)
        self.hits = 0
        self.misses = 0
        self._warned = False

    # -- 查询 ---------------------------------------------------------------
    def prepare(self, prompt: str) -> tuple[str | None, int]:
        """返回 (缓存路径, save 标志)。路径为 None 表示本次不使用缓存。

        save=0 表示"加载已有缓存"，save=1 表示"推理结束后保存"。
        """
        if not self._usable():
            return None, 0
        self.maintain()
        meta = self._read_meta()
        if (meta is not None and os.path.isfile(self.cache_path)
                and len(meta) <= len(prompt) and prompt.startswith(meta)):
            self.hits += 1
            self._write_stats()
            return self.cache_path, 0
        self.misses += 1
        self._write_stats()
        return self.cache_path, 1

    def commit(self, prompt: str) -> None:
        """推理成功后记录本次 prompt，作为下一次的前缀判断依据。"""
        if not self._usable():
            return
        try:
            os.makedirs(self.dir, exist_ok=True)
            tmp = self.meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(prompt)
            os.replace(tmp, self.meta_path)
        except OSError as exc:
            self._warn("写入缓存元数据失败：%s" % exc)

    def invalidate(self) -> None:
        """清空单槽（缓存文件与元数据一起删，避免"文件是新的、元数据是旧的"）。"""
        for path in (self.cache_path, self.meta_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._warn("删除缓存文件失败：%s" % exc)

    # -- 维护 ---------------------------------------------------------------
    def maintain(self) -> None:
        """TTL 过期与容量上限清理。只在单槽上有意义，但保留目录级清扫。"""
        try:
            if not os.path.isdir(self.dir):
                return
            now = time.time()
            entries = []
            for name in os.listdir(self.dir):
                path = os.path.join(self.dir, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if self.ttl_s and now - st.st_mtime > self.ttl_s:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    continue
                entries.append((st.st_mtime, st.st_size, path))
            if not self.max_mb:
                return
            total = sum(size for _, size, _ in entries)
            limit = self.max_mb * 1024 * 1024
            if total <= limit:
                return
            for _, size, path in sorted(entries):        # 最旧优先淘汰
                try:
                    os.unlink(path)
                except OSError:
                    continue
                total -= size
                if total <= limit:
                    break
        except OSError as exc:
            self._warn("缓存维护失败：%s" % exc)

    # -- 内部 ---------------------------------------------------------------
    def _usable(self) -> bool:
        """目录不可写时不报错，只告警并退化为"不用缓存"。"""
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as exc:
            self._warn("缓存目录不可用（%s），本次不使用前缀缓存" % exc)
            return False
        return True

    def bytes_used(self) -> int:
        total = 0
        try:
            for name in os.listdir(self.dir):
                try:
                    total += os.path.getsize(os.path.join(self.dir, name))
                except OSError:
                    continue
        except OSError:
            return 0
        return total

    def stats(self) -> dict:
        """跨进程可读的统计（server 的 /health 直接读这个文件）。"""
        return {"hits": self.hits, "misses": self.misses,
                "bytes": self.bytes_used()}

    def _write_stats(self) -> None:
        """原子写 stats.json：worker 是唯一写者，server 只读。"""
        import json
        try:
            tmp = self.stats_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"hits": self.hits, "misses": self.misses,
                           "bytes": self.bytes_used()}, fh)
            os.replace(tmp, self.stats_path)
        except OSError:
            pass

    def _read_meta(self) -> str | None:
        try:
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                return fh.read()
        except (OSError, UnicodeDecodeError):
            return None

    def _warn(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        try:
            sys.stderr.write("[rkllm] %s\n" % message)
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
