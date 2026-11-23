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
from ..auth import auth_status
from ..cacheview import cache_stats as _cache_stats

router = APIRouter()
# 数据面同款端点：由 app.create_app 以 `/v1` 前缀挂一次、根路径再挂一次
v1_router = APIRouter()


def _svc(request: Request):
    return request.app.state.svc


def _auth_state(request: Request) -> dict:
    """`/health` 里的鉴权状态（P1-S1）：与启动横幅/响应头同源，绝不含 key。

    鉴权启用时 /health 本身要求 Bearer；这里再报 enabled/source 供客户端确认
    "带对了 key"，未启用时也能直接看出"当前无鉴权、只允许回环"。
    """
    state = getattr(request.app.state, "auth", None)
    if isinstance(state, dict):
        return dict(state)
    return auth_status(_svc(request).cfg)


def _hist_stats(hist, endpoint: str | None = None) -> tuple[float | None, int]:
    """取直方图的 (均值, 样本数)；无样本返回 (None, 0)。

    `endpoint=None` 时**跨所有端点聚合**（P2-4）：`/health.perf` 此前只读
    `chat.completions` 标签，导致只走 `/v1/completions` 或 `/v1/responses`
    的部署永远 `source=cold`。指定 endpoint 时保持原口径（既有调用方）。

    用公开的 `collect()` 解析 samples，不再读 `child._sum` / `child._count`
    私有属性（2026-11-20 审计：私有属性在 prometheus_client 升级后可能消失，
    表现为 /health 的实测速率静默变 None）。任何异常都返回空值——探针端点
    必须永远可用。
    """
    if hist is None:
        return None, 0
    try:
        total = 0.0
        count = 0
        metrics = (hist.labels(endpoint=endpoint).collect() if endpoint
                   else hist.collect())
        for metric in metrics:
            for sample in metric.samples:
                if sample.name.endswith("_sum"):
                    total += float(sample.value)
                elif sample.name.endswith("_count"):
                    count += int(sample.value)
        if count <= 0:
            return None, 0
        return total / count, count
    except Exception:
        return None, 0


def _hist_endpoint_count(hist) -> int:
    """直方图里已有样本的端点数（0=无样本；多端点部署据此判断覆盖度）。

    注意：带标签 Histogram 的 `collect()` 只返回**一个** Metric，所有 label
    组合的 samples 都在里面（sample.labels 带 endpoint），所以按 label 值去重，
    不能按 Metric 个数计。
    """
    if hist is None:
        return 0
    try:
        endpoints: set[str] = set()
        for metric in hist.collect():
            for sample in metric.samples:
                if sample.name.endswith("_count") and sample.value > 0:
                    labels = getattr(sample, "labels", None) or {}
                    endpoints.add(str(labels.get("endpoint", "")))
        return len(endpoints)
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
    # P2-4：跨端点聚合（不再只读 chat.completions），否则 completions /
    # responses 单端点部署的 /health.perf 永远是 cold。
    tok_s, tok_samples = _hist_stats(getattr(queue, "tok_s_hist", None))
    ttft, ttft_samples = _hist_stats(getattr(queue, "ttft_hist", None))
    endpoints = _hist_endpoint_count(getattr(queue, "ttft_hist", None))
    # samples = 至少观测到 TTFT 的请求数（每个请求都会观测 TTFT；tok/s 只在
    # 解码阶段 > 0 时才有样本）。用 max 避免"模型很快、decode=0 无速率样本"
    # 被误报成 cold。
    samples = max(tok_samples, ttft_samples)
    limits = {
        "max_tokens_limit": int(getattr(svc, "max_tokens_limit", 0) or 0),
        "default_max_tokens": int(getattr(svc, "default_max_tokens", 0) or 0),
        "context_margin_tokens": CONTEXT_MARGIN_TOKENS,
        # request_timeout_s 对外取 wait_timeout_s（默认 600s）而非 queue 的
        # request_timeout_s（900s）：这是 2026-11-19 拍板的保守值——连接器
        # 拿它当 job 读超时的 base，600s 已足够覆盖单请求，且留更多余量给
        # factor 放大。不是漏报，勿改回 900s（审查报告 P1-B 已记录该决策）。
        "request_timeout_s": (int(queue.wait_timeout_s) if queue else 0),
        "busy_retry_after_s": (int(queue.busy_retry_after_s) if queue else 0),
    }
    return {
        # 0.0 是合法测量值（异步/极快响应），不能用真值判断把它误报成 cold。
        "perf": {"decode_tok_s": (round(tok_s, 2) if tok_s is not None
                                  else None),
                 "ttft_s": round(ttft, 3) if ttft is not None else None,
                 "samples": samples,
                 "endpoints": endpoints,
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
        # 鉴权真实状态（P1-S1）：enabled/source/bind_host/loopback_only
        "auth": _auth_state(request),
        # token 计数状态：backend=tokenizers（精确）/estimate（自校准估算）
        "tokens": _token_stats(),
        # 实测性能 + 能力上限：客户端据此判断同步/异步与预算，不再自带档位表
        **_perf_stats(svc),
    }


@router.get("/ready")
def ready(request: Request):
    """模型就绪才 200；加载中/崩溃 503 + Retry-After（编排系统可用）。

    P2-1：/ready 是唯一豁免鉴权且不限本机的探针端点，因此**不得泄露模型名**
    （此前 200 体里带 `model`，等于匿名可见部署信息）。只返回布尔与状态枚举。
    """
    svc = _svc(request)
    if svc.supervisor and svc.supervisor.is_ready():
        return {"ready": True}
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
        # 同 /health.limits：对外保守取 wait_timeout_s（600s），见上方说明。
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
