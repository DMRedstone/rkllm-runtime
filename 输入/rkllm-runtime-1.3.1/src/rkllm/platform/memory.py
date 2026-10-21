"""内存预算（设计 §7.1 / N10 / F12）。

公式（设计原文）：
    需要 = 模型文件大小 + KV(按档位与 context) + engine.memory_reserve_mb
不足时按 model.context → 2048 → 1024（不低于 model.min_context）逐级降并发告警；
仍不足则拒绝启动并打印所需内存。

可用内存口径：MemAvailable + SwapFree。把 swap 计入是有意为之——设计 §7.1
对 8 GB 板的 7B 明确给的是「建议 context 2048 + swap」，即 swap 本就是 7B 的
可行条件之一。分项保留在 Plan 里，拒绝时能说清"差多少、要不要加 swap"。

KV 系数是量级估算（不是精确测量）：RKLLM 的 KV cache 与层数/头数/head_dim
相关，不同模型族有差异。这里取保守偏低的系数，目的是让"8 GB 上 27B 必须拒绝、
7B 必须降 context"这两个设计结论成立；精确标定属 S0 spike 的「内存预算」项。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["MemoryBudgetError", "Plan", "available_mb", "model_size_mb",
           "kv_mb", "plan", "check_budget", "CONTEXT_STEPS"]

# 每 token KV 占用（MiB）。经验值，见模块 docstring 的说明。
KV_MB_PER_TOKEN = {"small": 0.125, "medium": 0.156, "large": 0.281}

# 降级步进（设计 §7.1 的三个档）
CONTEXT_STEPS = (4096, 2048, 1024)


class MemoryBudgetError(RuntimeError):
    """内存不足且已无法再降 context。"""

    def __init__(self, required_mb: float, available_mb_: float,
                 tried_context: int):
        self.required_mb = required_mb
        self.available_mb = available_mb_
        self.tried_context = tried_context
        super().__init__(
            "内存不足：context=%d 仍需 %.0f MB，可用 %.0f MB（含 swap）。"
            "请改用更小的模型/量化、降低 model.context、配置 swap，"
            "或在 ≥32 GB 环境改用 27B。" %
            (tried_context, required_mb, available_mb_))


@dataclass(frozen=True)
class Plan:
    context: int            # 校验后建议使用的 context
    required_mb: float      # 该 context 下的需求
    available_mb: float     # 实测可用（MemAvailable + SwapFree）
    model_mb: float
    kv_mb: float
    reserve_mb: float
    downgraded: bool        # 是否发生了降级

    def describe(self) -> str:
        return ("内存预算：模型 %.0f MB + KV %.0f MB + 预留 %.0f MB = %.0f MB，"
                "可用 %.0f MB，context=%d%s" %
                (self.model_mb, self.kv_mb, self.reserve_mb, self.required_mb,
                 self.available_mb, self.context,
                 "（已降级）" if self.downgraded else ""))


def model_size_mb(path: str) -> float:
    """模型文件体积（MiB）；不存在返回 0（交由档位经验值兜底）。"""
    if not path:
        return 0.0
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def kv_mb(model_class: str, context: int) -> float:
    per_token = KV_MB_PER_TOKEN.get(model_class, KV_MB_PER_TOKEN["small"])
    return per_token * max(1, int(context))


def available_mb() -> float | None:
    """MemAvailable + SwapFree（MiB）；非 Linux 返回 None（跳过校验）。"""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    info[key.strip()] = int(parts[0])     # kB
    except (OSError, ValueError):
        return None
    if "MemAvailable" not in info:
        return None
    total_kb = info.get("MemAvailable", 0) + info.get("SwapFree", 0)
    return total_kb / 1024.0


def plan(model_class: str, configured_context: int, *, model_path: str = "",
         reserve_mb: int = 1024, min_context: int = 1024,
         available: float | None = None) -> Plan:
    """按设计口径求可行 context。不抛错，只在无法满足时抛 MemoryBudgetError。

    available 显式传入用于测试；None 时读 /proc/meminfo，读不到则视为不校验
    （返回 configured_context 的 Plan，available 记为 -1）。
    """
    model = model_size_mb(model_path)
    avail = available_mb() if available is None else available
    if avail is None:
        return Plan(context=int(configured_context),
                    required_mb=model + kv_mb(model_class, configured_context)
                    + reserve_mb,
                    available_mb=-1.0, model_mb=model,
                    kv_mb=kv_mb(model_class, configured_context),
                    reserve_mb=float(reserve_mb), downgraded=False)

    # 候选 context：配置值本身 + 标准步进（只降不升），去重后从大到小
    candidates = []
    for ctx in [int(configured_context)] + list(CONTEXT_STEPS):
        ctx = max(int(min_context), min(int(ctx), int(configured_context)))
        if ctx not in candidates:
            candidates.append(ctx)
    candidates.sort(reverse=True)

    last_required = 0.0
    for idx, ctx in enumerate(candidates):
        kv = kv_mb(model_class, ctx)
        required = model + kv + reserve_mb
        last_required = required
        if avail >= required:
            return Plan(context=ctx, required_mb=required, available_mb=avail,
                        model_mb=model, kv_mb=kv, reserve_mb=float(reserve_mb),
                        downgraded=(ctx < int(configured_context)))
    raise MemoryBudgetError(last_required, avail, int(min_context))


def check_budget(model_class: str, configured_context: int, **kwargs) -> Plan:
    """plan() 的语义化别名：供 cli 直接调用。"""
    return plan(model_class, configured_context, **kwargs)
