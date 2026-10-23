"""缓存统计视图（审查 A-2：app 与 `/health` 共用的公开入口）。

原先 `app.py` 反向导入 `routes.ops._cache_stats`——app 依赖路由的**私有**
实现。这里提到独立模块，`app`（/metrics）与 `routes.ops`（/health）都只
依赖公开名字。

数据源始终是 worker 落盘的 `stats.json`（跨进程唯一可靠来源）；目录不可读
或文件不存在时返回零值而不报错——缓存是加速器，指标缺失不该让 /health 失败。
"""

from __future__ import annotations

import json
import os

__all__ = ["cache_stats"]


def cache_stats(cfg) -> dict:
    """读取 worker 落盘的缓存统计（含 O-1 的相似度与跳过次数）。"""
    if not getattr(cfg.cache, "enabled", False):
        return {"enabled": False, "hits": 0, "misses": 0, "bytes": 0,
                "similarity": 0.0, "skipped_saves": 0}
    directory = cfg.cache.dir
    stats = {"enabled": True, "hits": 0, "misses": 0, "bytes": 0,
             "similarity": 0.0, "skipped_saves": 0}
    bytes_from_stats = None
    try:
        with open(os.path.join(directory, "stats.json"), "r",
                  encoding="utf-8") as fh:
            data = json.load(fh)
        stats["hits"] = int(data.get("hits", 0))
        stats["misses"] = int(data.get("misses", 0))
        # O-1：相似度 gauge 与因阈值未接管单槽的次数——worker 侧算了就
        # 必须有人读，否则等于空转的配置/指标
        stats["similarity"] = float(data.get("similarity", 0.0))
        stats["skipped_saves"] = int(data.get("skipped_saves", 0))
        if "bytes" in data:
            bytes_from_stats = int(data["bytes"])
    except (OSError, ValueError):
        pass
    if bytes_from_stats is not None:
        # 效率：worker 已把 bytes 落盘，避免每次 /metrics 都 listdir+逐文件 getsize
        stats["bytes"] = bytes_from_stats
    else:
        total = 0
        try:
            for name in os.listdir(directory):
                try:
                    total += os.path.getsize(os.path.join(directory, name))
                except OSError:
                    continue
        except OSError:
            pass
        stats["bytes"] = total
    return stats
