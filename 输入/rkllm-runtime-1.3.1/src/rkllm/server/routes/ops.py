"""运维端点：health / ready / metrics / capabilities / reset。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from ... import caps

router = APIRouter()


def _svc(request: Request):
    return request.app.state.svc


@router.get("/health")
def health(request: Request):
    """永远 200：进程活着。/ready 才反映模型可用性（D3：补 context_len）。"""
    svc = _svc(request)
    return {
        "state": svc.supervisor.state if svc.supervisor else "unconfigured",
        "context_len": svc.context_len,
        "max_queue": svc.queue.max_wait,
        "queue_timeout": svc.queue.wait_timeout_s,
        "queue_length": svc.queue.queue_depth,
        "model": svc.model_name,
        "model_path": svc.model_path,
        "version": svc.version,
        "lib_sha256": (svc.lib_fingerprint or {}).get("sha256"),
        "lib_size": (svc.lib_fingerprint or {}).get("size"),
    }


@router.get("/ready")
def ready(request: Request):
    """模型就绪才 200；加载中/崩溃 503 + Retry-After（编排系统可用）。"""
    svc = _svc(request)
    if svc.supervisor and svc.supervisor.is_ready():
        return {"ready": True, "model": svc.model_name}
    return JSONResponse({"ready": False, "state": svc.supervisor.state},
                        status_code=503,
                        headers={"Retry-After": str(svc.queue.busy_retry_after_s)})


@router.get("/metrics")
def metrics(request: Request):
    """Prometheus 文本（改造设计 §6.6 指标集）。"""
    svc = _svc(request)
    m = svc.queue.metrics
    sup = svc.supervisor.metrics if svc.supervisor else {}
    lines = [
        "# TYPE rkllm_requests_total counter",
        "rkllm_requests_total %d" % m["requests_total"],
        "# TYPE rkllm_rejected_total counter",
        "rkllm_rejected_total %d" % m["rejected_total"],
        "# TYPE rkllm_queue_depth gauge",
        "rkllm_queue_depth %d" % svc.queue.queue_depth,
        "# TYPE rkllm_queue_wait_seconds_sum counter",
        "rkllm_queue_wait_seconds_sum %f" % m["queue_wait_seconds_sum"],
        "# TYPE rkllm_tokens_total counter",
        "rkllm_tokens_total %d" % m["tokens_total"],
        "# TYPE rkllm_worker_restarts_total counter",
        "rkllm_worker_restarts_total %d" % sup.get("worker_restarts_total", 0),
        "# TYPE rkllm_worker_state gauge",
        'rkllm_worker_state{state="%s"} 1' % (
            svc.supervisor.state if svc.supervisor else "none"),
    ]
    return PlainTextResponse("\n".join(lines) + "\n",
                             media_type="text/plain; version=0.0.4")


@router.get("/v1/capabilities")
def capabilities(request: Request):
    """机器可读能力表（F11/G5：单一事实源 caps.py）。"""
    sup = _svc(request).supervisor
    view = dict(caps.CAPS)
    view["model"] = {"name": _svc(request).model_name,
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
