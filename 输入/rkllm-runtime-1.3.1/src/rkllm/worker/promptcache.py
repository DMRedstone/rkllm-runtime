"""Prompt 前缀缓存（ADR-07 / cache.* 四键）。

解决的问题：服务默认无状态，每轮都把整段 messages 全量 prefill。多轮对话
里前 N-1 轮完全重复，重算 4k 前缀在 4B 上约 1–3 s、27B 上 20–60 s。

实现：单槽（single-slot）前缀缓存。rkllm 的 `prompt_cache_params` 会把本次
prompt 的 KV 保存到文件，下一次若用同一文件则直接加载作为前缀——只有当新
prompt 以缓存内容为前缀时才有意义。所以这里维护「上一次的 prompt」：

- 命中：`new_prompt.startswith(上次 prompt)` → save=0（加载并按前缀复用）；
- 未命中：save=1（本次推理结束后由 runtime 写入缓存）。

存储位置：默认放在 tmpfs（/dev/shm/rkllm/prefix），完全在内存中，避免
eMMC/SD 的 I/O 瓶颈与写入磨损。若 tmpfs 不可用则自动降级到 /tmp；再不可用
则停用缓存。

单槽而非多槽：多轮对话是串行追加的，上一轮就是最长可复用前缀；多槽需要
会话标识，而服务按设计是无状态的。容量与 TTL 仍按配置执行，用于清理异常
增长（历史遗留文件、模型更换后失效的缓存）。

失败降级：缓存文件损坏或 runtime 拒绝加载时，engine 会清槽并重试一次不带
缓存的推理——缓存只能是加速器，不能成为可用性的单点。

O-1（字节级 LCP + 相似度阈值，对齐 llama.cpp `--slot-prompt-similarity`）：
- 命中判定从 `str.startswith` 换成**字节级最长公共前缀**（UTF-8 边界安全），
  由此得到「缓存内容中可续用的比例」`similarity = lcp / len(meta)`；
- 为什么不做"部分前缀复用"：llama.cpp 能把 slot 的 KV 裁到公共前缀，本
  runtime 的 `prompt_cache_params` 是**整文件加载、不截断**——注入比新 prompt
  更长的旧 KV 会污染上下文。所以只有 `lcp == len(meta)`（严格前缀）才安全
  命中，这一点不打折扣；
- 阈值真正生效的位置是**是否让本请求接管单槽**（在 prepare 阶段决定，不是
  commit）：`meta` 与 `cache` 文件必须始终配对——runtime 在 `save=1` 时会
  直接覆盖缓存文件，若只跳过 meta 就会造成「文件是新会话的、meta 是旧会话
  的」错位，下一次旧会话命中后加载到别人的 KV，上下文直接错。
  因此：严格前缀 → `save=0` 复用；相似度 ≥ 阈值（同一会话的延续，例如用户
  改了最后一句）→ `save=1` 重建；相似度低于阈值（另一个会话）→ **本次不
  加载也不保存**（返回 None），既有配对保持不动，热点会话下一轮继续命中。
  这避免了交错请求下「每次 miss、每次覆盖」的抖动，代价是无关会话在单槽
  存续期间得不到缓存——比注入错误上下文的代价小得多。
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

# 相似度阈值：与 llama.cpp `--slot-prompt-similarity` 默认值一致（0.5）。
# 单槽下用于区分「同一会话的延续」与「另一个会话」——低于阈值的 commit
# 会保留既有缓存而不是覆盖（见模块 docstring 的 O-1 说明）。
PROMPT_SIMILARITY = 0.5

# 内存优先：tmpfs 是 Linux 标准内存文件系统，无需外部依赖。
_DEFAULT_TMPFS_DIR = "/dev/shm/rkllm/prefix"
_FALLBACK_TMP_DIR = "/tmp/rkllm/prefix"


def common_prefix_len(left: str, right: str) -> int:
    """字节级最长公共前缀长度（UTF-8 边界安全）。

    先按 UTF-8 字节比较，再把落点回退到字符起始处——避免返回一个切在
    多字节字符中间的偏移（那种偏移既不能用于切片，也无法与长度比较）。
    """
    a = left.encode("utf-8")
    b = right.encode("utf-8")
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    # 0b10xxxxxx 是 UTF-8 续字节：回退到该字符首字节
    while 0 < index < len(a) and (a[index] & 0xC0) == 0x80:
        index -= 1
    return index



def _pick_cache_dir(requested: str | None) -> str:
    """选择实际可用的缓存目录。

    - 用户显式指定目录：优先用它，不可用时不自动回退（保持原退化语义，
      让 _usable 决定本次是否使用缓存）。
    - 未指定：优先 tmpfs，回退 /tmp；都失败则返回 tmpfs 路径，由 _usable
      报告不可用并退化。
    """
    if requested:
        return requested
    for path in (_DEFAULT_TMPFS_DIR, _FALLBACK_TMP_DIR):
        try:
            os.makedirs(path, exist_ok=True)
            # 试探写：某些挂载点允许创建目录但禁止写（如只读 rootfs）
            probe = os.path.join(path, ".probe")
            with open(probe, "w") as fh:
                fh.write("1")
            os.unlink(probe)
            return path
        except OSError:
            continue
    return _DEFAULT_TMPFS_DIR   # 都失败也返回一个路径，让后续 _usable 再报一次


class PromptCache:
    """管理一个缓存目录下的单槽前缀缓存。"""

    def __init__(self, directory: str | None = None, max_mb: int = 512,
                 ttl_s: int = 604800):
        self.dir = _pick_cache_dir(directory)
        self.max_mb = max_mb
        self.ttl_s = ttl_s
        self.cache_path = os.path.join(self.dir, _CACHE_NAME)
        self.meta_path = os.path.join(self.dir, _META_NAME)
        self.stats_path = os.path.join(self.dir, _STATS_NAME)
        self.hits = 0
        self.misses = 0
        # O-1：上一次 prepare 的相似度（缓存内容中可续用的比例）与因阈值
        # 未接管单槽的次数——都进 stats.json，供板端判断前缀复用是否有效。
        self.last_similarity = 0.0
        self.skipped_saves = 0
        self._warned = False
        self._disabled = False

    # -- 查询 ---------------------------------------------------------------
    def prepare(self, prompt: str) -> tuple[str | None, int]:
        """返回 (缓存路径, save 标志)。路径为 None 表示本次不使用缓存。

        save=0 表示"加载已有缓存"，save=1 表示"推理结束后保存"。

        命中判定用字节级 LCP（O-1）：只有 `lcp == len(meta)`（缓存内容是
        本次 prompt 的严格前缀）才安全复用——runtime 整文件加载、不截断，
        部分前缀命中会把多余 KV 注入上下文。
        """
        if not self._usable():
            return None, 0
        self.maintain()
        meta = self._read_meta()
        self.last_similarity = 0.0
        if meta is None or not os.path.isfile(self.cache_path):
            # 首次写入（或缓存文件被外部清掉）：本次建立配对
            self.misses += 1
            self._write_stats()
            return self.cache_path, 1
        meta_bytes = len(meta.encode("utf-8"))
        prefix = common_prefix_len(prompt, meta)
        if meta_bytes:
            self.last_similarity = prefix / meta_bytes
        if prefix >= meta_bytes:
            self.hits += 1
            self._write_stats()
            return self.cache_path, 0
        if self.last_similarity >= PROMPT_SIMILARITY:
            # 同一会话的延续（例如用户改了最后一句）：允许本轮重建缓存。
            # save=1 会覆盖缓存文件，commit 随后同步 meta，配对保持一致。
            self.misses += 1
            self._write_stats()
            return self.cache_path, 1
        # 无关会话：不加载、不保存。既避免把别人的 KV 当前缀注入，也避免
        # 覆盖掉热点会话的缓存文件（meta 与 cache 必须始终配对）。
        self.skipped_saves += 1
        self.misses += 1
        self._write_stats()
        return None, 0

    def commit(self, prompt: str) -> None:
        """推理成功后记录本次 prompt，作为下一次的前缀判断依据。

        只在 `prepare` 返回了路径（save=1）时由 engine 调用，因此这里
        无需再判相似度——能否接管单槽已在 prepare 阶段决定。
        """
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
    def disable(self) -> None:
        """本 worker 生命周期内彻底停用缓存（加载失败的熔断开关）。

        2026-10-22 实测：当前 runtime 对 cache 文件的**加载**路径返回
        rkllm_run -1（内部码 10000），且失败后即使换 save=1 重跑仍失败——
        运行时状态已脏。熔断后本次进程不再碰缓存，宁可全量 prefill。
        """
        self._disabled = True

    def _usable(self) -> bool:
        """目录不可写时不报错，只告警并退化为"不用缓存"。"""
        if self._disabled:
            return False
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
        return self._stats_dict()

    def _stats_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "bytes": self.bytes_used(),
                # O-1：相似度 gauge 与因阈值未接管单槽的次数
                "similarity": round(self.last_similarity, 3),
                "skipped_saves": self.skipped_saves}

    def _write_stats(self) -> None:
        """原子写 stats.json：worker 是唯一写者，server 只读。"""
        import json
        try:
            tmp = self.stats_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._stats_dict(), fh)
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
