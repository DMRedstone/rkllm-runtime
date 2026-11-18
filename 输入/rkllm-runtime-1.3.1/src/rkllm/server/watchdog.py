"""systemd 看门狗接入（`Type=notify` + `WatchdogSec`）。

为什么交给 systemd：自研的心跳只能判断"worker 进程还活着"，判断不了"HTTP
进程自己卡死"（事件循环被同步调用阻塞、死锁）。systemd 的看门狗语义正是
"进程必须周期性报到，否则重启"——这部分不用我们实现。

实现要点（用 Debian `python3-systemd` 的 `sd_notify`，不自己拼 AF_UNIX 报文）：

* 不在 systemd 下运行时（`NOTIFY_SOCKET` 未设置）整段是 no-op，命令行直跑
  不受影响；
* 报到**必须证明事件循环还在跑**：ping 线程把一次 no-op 协程丢回事件循环并
  等结果，超时就这一次不报到 —— 否则"只看线程活着"的 ping 会在主循环死锁
  时继续骗 systemd；
* 间隔取 `WATCHDOG_USEC/3`（systemd 推荐的 1/3 余量），未设置则 20 s。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

try:                                  # Debian: python3-systemd
    from systemd.daemon import notify as _sd_notify
except ImportError:                   # 未安装时不报错，只是没有看门狗
    _sd_notify = None

__all__ = ["available", "notify", "ready", "stopping", "watchdog_interval_s",
           "start_loop_watchdog"]

log = logging.getLogger("rkllm.server.watchdog")


def available() -> bool:
    return _sd_notify is not None and bool(os.environ.get("NOTIFY_SOCKET"))


def notify(state: str) -> None:
    if not available():
        return
    try:
        _sd_notify(state)
    except Exception as exc:          # noqa: BLE001 - 报到失败不该拖垮服务
        log.warning("sd_notify(%s) 失败：%s", state, exc)


def ready() -> None:
    notify("READY=1")


def stopping() -> None:
    notify("STOPPING=1")


def watchdog_interval_s(default: float = 20.0) -> float:
    """systemd 推荐按 WatchdogSec 的 1/3 报到。"""
    try:
        usec = int(os.environ.get("WATCHDOG_USEC", "0"))
    except ValueError:
        usec = 0
    if usec <= 0:
        return default
    return max(1.0, usec / 1_000_000.0 / 3.0)


async def _probe() -> None:
    """一次空转：只要事件循环还能调度就立刻返回。"""
    await asyncio.sleep(0)


def start_loop_watchdog(loop: asyncio.AbstractEventLoop,
                        interval_s: float | None = None) -> threading.Thread | None:
    """起一个 ping 线程；返回线程（不可用时返回 None）。

    每轮先确认事件循环可调度，再报到。用 `run_coroutine_threadsafe` 而不是
    `call_soon_threadsafe`：前者能拿到结果，才有"循环真的动了"的证据。
    """
    if not available():
        return None
    interval = interval_s or watchdog_interval_s()

    def _run() -> None:
        while True:
            time.sleep(interval)
            try:
                future = asyncio.run_coroutine_threadsafe(_probe(), loop)
                future.result(timeout=max(1.0, interval * 0.5))
            except Exception:         # noqa: BLE001 - 循环卡住/已关闭：不报到
                continue
            notify("WATCHDOG=1")

    thread = threading.Thread(target=_run, name="rkllm-watchdog", daemon=True)
    thread.start()
    log.info("systemd 看门狗已接入，报到间隔 %.1fs", interval)
    return thread
