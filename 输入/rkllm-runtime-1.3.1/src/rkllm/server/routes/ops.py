"""运维端点：health / ready / capabilities / reset。

/metrics 不在此注册：它的开关与路径来自 metrics.enabled / metrics.path，
由 app.create_app 按配置动态挂载（N5，键不再空转）。
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ... import caps

router = APIRouter()


def _svc(request: Request):
    return request.app.state.svc


def _cache_stats(cfg) -> dict:
    """读取 worker 落盘的缓存统计（跨进程唯一可靠来源）。

    目录不可读或文件不存在时返回零值，不报错——缓存是加速器，指标缺失不该
    让 /health 失败。
    """
    if not getattr(cfg.cache, "enabled", False):
        return {"enabled": False, "hits": 0, "misses": 0, "bytes": 0}
    directory = cfg.cache.dir
    stats = {"enabled": True, "hits": 0, "misses": 0, "bytes": 0}
    try:
        with open(os.path.join(directory, "stats.json"), "r",
                  encoding="utf-8") as fh:
            data = json.load(fh)
        stats["hits"] = int(data.get("hits", 0))
        stats["misses"] = int(data.get("misses", 0))
    except (OSError, ValueError):
        pass
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


@router.get("/health")
def health(request: Request):
    """永远 200：进程活着。/ready 才反映模型可用性（D3：补 context_len / 档位）。"""
    svc = _svc(request)
    return {
        "state": svc.supervisor.state if svc.supervisor else "unconfigured",
        "context_len": svc.context_len,
        "model_class": svc.model_class,          # F12 验收项
        "max_queue": svc.queue.max_wait,
        "queue_timeout": svc.queue.wait_timeout_s,
        "queue_length": svc.queue.queue_depth,
        "model": svc.model_name,
        "model_path": svc.model_path,
        "version": svc.version,
        "lib_sha256": (svc.lib_fingerprint or {}).get("sha256"),
        "lib_size": (svc.lib_fingerprint or {}).get("size"),
        # 设计 §6.4(4)：缓存命中与占用对运维可见
        "cache": _cache_stats(svc.cfg),
    }


@router.get("/ready")
def ready(request: Request):
    """模型就绪才 200；加载中/崩溃 503 + Retry-After（编排系统可用）。"""
    svc = _svc(request)
    if svc.supervisor and svc.supervisor.is_ready():
        return {"ready": True, "model": svc.model_name}
    return JSONResponse({"ready": False,
                         "state": svc.supervisor.state if svc.supervisor
                         else "unconfigured"},
                        status_code=503,
                        headers={"Retry-After": str(svc.queue.busy_retry_after_s)})


@router.get("/v1/capabilities")
def capabilities(request: Request):
    """机器可读能力表（F11/G5：单一事实源 caps.py）。"""
    svc = _svc(request)
    sup = svc.supervisor
    view = dict(caps.CAPS)
    view["model"] = {"name": svc.model_name,
                     "class": svc.model_class,
                     "context_len": svc.context_len,
                     "state": sup.state if sup else "none"}
    return view


@router.post("/v1/reset")
def reset(request: Request):
    """清 KV / 会话状态（保留旧版扩展语义）。"""
    svc = _svc(request)
    try:
        svc.on_reset()
    except Exception as exc:                    # worker 不可用也不 500
        return {"reset": False, "reason": str(exc)[:200]}
    return {"reset": True}
