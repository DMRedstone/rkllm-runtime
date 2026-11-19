"""运维端点：health / ready / capabilities / reset。

/metrics 不在此注册：它的开关与路径来自 metrics.enabled / metrics.path，
由 app.create_app 按配置动态挂载（N5，键不再空转）。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ... import caps
from ...platform import npu_status
from ...agent.tokens import stats as _token_stats
from ...agent.pipeline import CONTEXT_MARGIN_TOKENS
from ..cacheview import cache_stats as _cache_stats

router = APIRouter()
# 数据面同款端点：由 app.create_app 以 `/v1` 前缀挂一次、根路径再挂一次
v1_router = APIRouter()


def _svc(request: Request):
    return request.app.state.svc


def _hist_mean(hist, endpoint: str) -> float | None:
    """取某 endpoint 标签下直方图的均值（`_sum/_count`）；无样本返回 None。

    直方图由 prometheus_client 持有（`metricsutil`），`_sum`/`_count` 是它
    对外的累计值，读它们不需要额外的聚合器。任何异常都返回 None，避免
    `/health` 因为指标对象缺失而 500——探针端点必须永远可用。
    """
    if hist is None:
        return None
    try:
        child = hist.labels(endpoint=endpoint)
        count = float(child._count.get() or 0)
        if count <= 0:
            return None
        return float(child._sum.get()) / count
    except Exception:
        return None


def _hist_count(hist, endpoint: str) -> int:
    if hist is None:
        return 0
    try:
        return int(hist.labels(endpoint=endpoint)._count.get() or 0)
    except Exception:
        return 0


def _perf_stats(svc) -> dict:
    """板端实测性能 + 能力上限，供客户端自行判断同步/异步与预算。

    为什么放在 `/health` 而不是让客户端按档位表估：解码速率、上下文、输出上限
    都是板端**已经测出来/算出来**的事实，客户端再写一份档位表只会漂移
    （换模型/换板子即失效）。客户端读到 `source=measured` 即可直接使用；
    `cold`（尚无样本）时自行回落保守值。
    """
    queue = getattr(svc, "queue", None)
    tok_s = _hist_mean(getattr(queue, "tok_s_hist", None), "chat.completions")
    ttft = _hist_mean(getattr(queue, "ttft_hist", None), "chat.completions")
    samples = _hist_count(getattr(queue, "tok_s_hist", None), "chat.completions")
    limits = {
        "max_tokens_limit": int(getattr(svc, "max_tokens_limit", 0) or 0),
        "default_max_tokens": int(getattr(svc, "default_max_tokens", 0) or 0),
        "context_margin_tokens": CONTEXT_MARGIN_TOKENS,
        "request_timeout_s": (int(queue.wait_timeout_s) if queue else 0),
        "busy_retry_after_s": (int(queue.busy_retry_after_s) if queue else 0),
    }
    return {
        "perf": {"decode_tok_s": round(tok_s, 2) if tok_s else None,
                 "ttft_s": round(ttft, 3) if ttft else None,
                 "samples": samples,
                 "source": "measured" if samples else "cold"},
        "limits": limits,
    }


@router.get("/health")
def health(request: Request):
    """永远 200：进程活着。/ready 才反映模型可用性（D3：补 context_len / 档位）。"""
    svc = _svc(request)
    npu_ok, npu_detail = npu_status()
    # D10：subprocess 模式下库由 worker 加载，指纹从 supervisor 的 ready 帧来；
    # in_process 模式下 supervisor 直接暴露 engine 的指纹。
    fingerprint = (getattr(svc.supervisor, "lib_fingerprint", None)
                   or svc.lib_fingerprint or {})
    # 1.3.1-15 兼容：旧客户端（health_check 等）读 stats 计数。计数现以
    # Prometheus /metrics 为正本，这里原样暴露同一组计数器（2026-10-25
    # 审计 P1：对外契约不能无声变化）。
    m = svc.queue.metrics
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
        "lib_sha256": fingerprint.get("sha256"),
        "lib_size": fingerprint.get("size"),
        "stats": {
            "requests": m.get("requests_total", 0),
            "rejected": m.get("rejected_total", 0),
            "errors": m.get("errors_total", 0),
            "prompt_tokens": m.get("prompt_tokens_total", 0),
            "completion_tokens": m.get("completion_tokens_total", 0),
        },
        # 驱动可用性对运维/本地 status 工具可见（0.9.x 起是 DRM render node）
        "npu": {"available": npu_ok, "detail": npu_detail},
        # 设计 §6.4(4)：缓存命中与占用对运维可见
        "cache": _cache_stats(svc.cfg),
        # token 计数状态：backend=tokenizers（精确）/estimate（自校准估算）
        "tokens": _token_stats(),
        # 实测性能 + 能力上限：客户端据此判断同步/异步与预算，不再自带档位表
        **_perf_stats(svc),
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


@v1_router.get("/capabilities")
def capabilities(request: Request):
    """机器可读能力表（F11/G5：单一事实源 caps.py）。"""
    svc = _svc(request)
    sup = svc.supervisor
    view = dict(caps.CAPS)
    view["model"] = {"name": svc.model_name,
                     "class": svc.model_class,
                     "context_len": svc.context_len,
                     "state": sup.state if sup else "none"}
    # limits 必须反映**实际生效**的配置（M-8）：它是客户端计算请求预算的唯一
    # 依据，报出与行为不符的容量会让客户端在真正发请求时收到 400/503。
    limits = dict(caps.LIMITS)
    q = getattr(svc, "queue", None)
    limits.update({
        "context_len": svc.context_len,
        "max_output": (int(getattr(svc, "max_tokens_limit", 0) or 0)
                       or int(getattr(svc, "default_max_tokens", 0) or 0)),
        "concurrency": 1,                     # 单 NPU 串行，queue 侧已夹紧
        "queue": (q.max_wait if q else limits.get("queue", 0)),
        "request_timeout_s": (int(q.wait_timeout_s) if q
                              else limits.get("request_timeout_s", 0)),
    })
    view["limits"] = limits
    return view


@v1_router.post("/reset")
def reset(request: Request):
    """清 KV / 会话状态（保留旧版扩展语义）。"""
    svc = _svc(request)
    try:
        svc.on_reset()
    except Exception as exc:                    # worker 不可用也不 500
        return {"reset": False, "reason": str(exc)[:200]}
    return {"reset": True}


@router.get("/admin/models")
def admin_models(request: Request) -> JSONResponse:
    """只读：当前模型 + 搜索目录里可用的 `.rkllm` 文件。

    只列举，不加载、不删除——多模型热切换需要新的资源与路由设计，尚未做。
    """
    svc = request.app.state.svc
    cfg = svc.cfg
    roots = [d for d in (getattr(cfg.model, "search_dirs", ()) or ()) if d]
    found: list[dict] = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.rkllm")):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            found.append({
                "id": path.stem,
                "path": str(path),
                "size_mb": round(size / (1024 * 1024), 1),
                "current": str(path) == str(cfg.model.path),
            })
    return JSONResponse({
        "current": {"id": svc.model_name, "path": cfg.model.path,
                    "model_class": svc.model_class,
                    "context_len": svc.context_len},
        "search_dirs": roots,
        # 说明为什么只有只读：切换模型要重启 worker，属未实现项
        "switchable": False,
        "models": found,
    })
