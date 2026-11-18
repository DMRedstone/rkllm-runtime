"""systemd 看门狗接入回归（不依赖真实 systemd）。"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.server import watchdog


def test_no_socket_means_noop(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert watchdog.available() is False
    # 不该抛异常，也不该尝试报到
    watchdog.notify("READY=1")
    watchdog.ready()
    watchdog.stopping()
    assert watchdog.start_loop_watchdog(asyncio.new_event_loop()) is None


def test_interval_from_watchdog_usec(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "90000000")     # 90 s → 1/3
    assert watchdog.watchdog_interval_s() == 30.0
    monkeypatch.setenv("WATCHDOG_USEC", "0")
    assert watchdog.watchdog_interval_s(default=7.0) == 7.0
    monkeypatch.setenv("WATCHDOG_USEC", "abc")
    assert watchdog.watchdog_interval_s(default=5.0) == 5.0


def test_notify_reaches_sd_notify(monkeypatch):
    seen = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setattr(watchdog, "_sd_notify", lambda state: seen.append(state))
    assert watchdog.available() is True
    watchdog.ready()
    watchdog.stopping()
    assert seen == ["READY=1", "STOPPING=1"]


def test_ping_requires_running_loop(monkeypatch):
    """事件循环停摆时不得报到——否则看门狗被自己骗过去。"""
    seen = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setattr(watchdog, "_sd_notify", lambda state: seen.append(state))

    loop = asyncio.new_event_loop()
    started = threading.Event()

    # 用极短间隔，避免测试等 1 s 以上
    thread = watchdog.start_loop_watchdog(loop, interval_s=0.05)
    assert thread is not None
    try:
        # 循环没跑 → run_coroutine_threadsafe 永远不会完成 → 不报到
        threading.Event().wait(0.3)
        assert "WATCHDOG=1" not in seen
        # 让循环跑起来 → 下一轮应当报到
        runner = threading.Thread(target=loop.run_forever, daemon=True)
        runner.start()
        started.set()
        threading.Event().wait(0.4)
        assert "WATCHDOG=1" in seen
    finally:
        loop.call_soon_threadsafe(loop.stop)
