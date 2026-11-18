"""资源时序采样：NPU / DDR 频率、内存可用量、worker 常驻内存。

为什么单独一个采样线程：`/metrics` 是拉取式的，抓取时刻只能看到瞬时值；调参
（context、threads、锁频策略）要看的是**一段时间内的走势**——比如锁频是否真的
生效、prefill 时内存峰值多少、worker 有没有缓慢泄漏。这里每 `interval_s` 采
一次，`/metrics` 每次抓取把最新值作为 gauge 暴露（Prometheus 侧按 scrape 间隔
自己存时间序列，我们不需要在进程内保存历史）。

只读 `/sys`，失败一律记 None（不抛、不影响服务）；非板端环境（无 devfreq 节点）
自然得到空值，不影响其他指标。
"""
from __future__ import annotations

import glob
import logging
import threading
import time

import psutil

__all__ = ["ResourceSampler", "read_freqs"]

log = logging.getLogger("rkllm.server.sysmetrics")

_NPU_GLOBS = ("/sys/class/devfreq/*npu*", "/sys/class/devfreq/*.npu")
_DDR_GLOBS = ("/sys/class/devfreq/*dmc*", "/sys/class/devfreq/*ddr*",
              "/sys/class/devfreq/*dfi*")


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="ascii", errors="replace") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _first_freq_mhz(patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        for node in sorted(glob.glob(pattern)):
            hz = _read_int(node + "/cur_freq")
            if hz:
                return round(hz / 1_000_000.0, 1)     # Hz → MHz
    return None


def read_freqs() -> tuple[float | None, float | None]:
    """(NPU MHz, DDR MHz)；节点不存在时返回 (None, None)。"""
    return _first_freq_mhz(_NPU_GLOBS), _first_freq_mhz(_DDR_GLOBS)


class ResourceSampler(threading.Thread):
    """后台采样线程；`snapshot()` 返回最近一次结果（无锁读字典即可）。"""

    def __init__(self, interval_s: float = 5.0):
        super().__init__(name="rkllm-res-sampler", daemon=True)
        self.interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._snapshot: dict[str, float | None] = {
            "npu_freq_mhz": None, "ddr_freq_mhz": None,
            "mem_available_mb": None, "mem_total_mb": None,
            "proc_rss_mb": None,
        }

    # -- 采样 -----------------------------------------------------------
    def sample_once(self) -> dict:
        npu, ddr = read_freqs()
        snap: dict[str, float | None] = {"npu_freq_mhz": npu, "ddr_freq_mhz": ddr}
        try:
            mem = psutil.virtual_memory()
            snap["mem_available_mb"] = round(mem.available / 1048576.0, 1)
            snap["mem_total_mb"] = round(mem.total / 1048576.0, 1)
        except Exception as exc:                      # noqa: BLE001
            log.debug("读取内存失败：%r", exc)
            snap["mem_available_mb"] = None
            snap["mem_total_mb"] = None
        try:
            snap["proc_rss_mb"] = round(
                psutil.Process().memory_info().rss / 1048576.0, 1)
        except Exception:                             # noqa: BLE001
            snap["proc_rss_mb"] = None
        self._snapshot = snap
        return snap

    def snapshot(self) -> dict:
        return dict(self._snapshot)

    # -- 线程 -----------------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:                  # noqa: BLE001
                log.debug("资源采样失败：%r", exc)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
