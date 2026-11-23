"""P2.2 取消 / 自愈 / 资源回归。

覆盖（真实路径优先，伪 worker/引擎只替换 NPU 边界）：

1. worker abort 按请求 id 匹配：迟到的旧 id 不误杀在飞请求；匹配的 id 正常中止。
2. prefill 阶段（0 个 token）被 abort 必须回 `aborted: true` 的 end，而不是
   "假完成"（正常 end + finish_reason=stop）；abort 打断 C 调用抛异常时同理。
3. `_send` 写失败不能杀死推理线程：只标记连接死亡、置 shutdown，由 supervisor
   重建 worker（恢复可服务状态）。
4. supervisor 崩溃恢复：健康线程主动后台重启（不等下一个请求），请求撞上恢复
   时只重启一次、不死等 load_timeout；加载失败不泄漏 worker 子进程。
5. `queue.request_timeout_s` 真正落到 supervisor/连接读超时；
6. queue 的 `cancelled=` 能力对非流式阻塞提交可用（routes 透传由 Lead 收口，
   本任务不改 routes/pipeline_runner）。
7. worker 侧 `--cache-similarity` CLI 与 PromptCache 接线（cli_engine 一行待 Lead）。
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.server.queue import InferenceQueue  # noqa: E402
from rkllm.server.supervisor import (RequestAborted, Supervisor,  # noqa: E402
                                     WorkerClient, WorkerDown)
from rkllm.worker.fake import FakeEngine  # noqa: E402
from rkllm.worker.frames import (OP_ABORT, OP_END, OP_ERROR,  # noqa: E402
                                 OP_RUN, OP_TOKEN, read_frame, write_frame)
from rkllm.worker.main import Worker  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# 1) worker 侧：abort id 匹配 / prefill abort / 写失败
# ---------------------------------------------------------------------------


class _PrefillGateEngine:
    """首个 token 之前阻塞（模拟长 prefill）；abort() 放行。"""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.aborts = 0

    def stream(self, prompt, **kwargs):
        self.started.set()
        self.release.wait(15)
        if False:                     # 让函数成为 generator
            yield ""

    def last_perf(self):
        return None

    def abort(self):
        self.aborts += 1
        self.release.set()


def _start_worker(engine):
    left, right = socket.socketpair()
    worker = Worker(engine, heartbeat_interval_s=3600, heartbeat_grace_s=3600)
    thread = threading.Thread(target=worker.serve_connection, args=(left,),
                              daemon=True)
    thread.start()
    return worker, right, thread


def _stop_worker(right, thread):
    try:
        right.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    right.close()
    thread.join(timeout=5)


def _collect_end(rfile) -> tuple[list[str], dict]:
    tokens: list[str] = []
    while True:
        frame = read_frame(rfile)
        assert frame, "worker 在 end/error 帧之前关闭了连接"
        if frame.get("op") == OP_TOKEN:
            tokens.append(frame.get("text", ""))
        elif frame.get("op") in (OP_END, OP_ERROR):
            return tokens, frame


def test_worker_abort_during_prefill_reports_aborted_not_fake_stop():
    """prefill 阶段（还没有 token）被 abort → 必须回 aborted end，不能回正常 end。"""
    engine = _PrefillGateEngine()
    worker, right, thread = _start_worker(engine)
    right.settimeout(10)
    rfile, wfile = right.makefile("rb"), right.makefile("wb")
    try:
        write_frame(wfile, {"op": OP_RUN, "id": "pre-1", "prompt": "很长的前缀"})
        assert engine.started.wait(5), "推理未进入 prefill"
        write_frame(wfile, {"op": OP_ABORT, "id": "pre-1"})
        tokens, end = _collect_end(rfile)
        assert tokens == []
        assert end.get("op") == OP_END, end
        assert end.get("aborted") is True, (
            "prefill 阶段 abort 被当成正常结束（假完成）：%r" % (end,))
        assert engine.aborts >= 1, "abort 未传导到 engine.abort()"
    finally:
        _stop_worker(right, thread)


def test_worker_ignores_stale_abort_id():
    """迟到的旧请求 abort（id 不匹配）不得误杀下一条在飞请求。"""
    engine = FakeEngine(delay=0.01)
    worker, right, thread = _start_worker(engine)
    right.settimeout(10)
    rfile, wfile = right.makefile("rb"), right.makefile("wb")
    try:
        write_frame(wfile, {"op": OP_RUN, "id": "live-1", "prompt": "hi"})
        first = read_frame(rfile)
        assert first.get("op") == OP_TOKEN and first.get("id") == "live-1"
        write_frame(wfile, {"op": OP_ABORT, "id": "stale-run-0"})
        tokens, end = _collect_end(rfile)
        assert end.get("op") == OP_END and not end.get("aborted"), (
            "旧 id 的迟到 abort 误杀了当前请求：%r" % (end,))
        assert "".join([first["text"], *tokens]) == "answer 1 to: hi"
    finally:
        _stop_worker(right, thread)


def test_worker_aborts_when_id_matches():
    engine = FakeEngine(delay=0.01)
    worker, right, thread = _start_worker(engine)
    right.settimeout(10)
    rfile, wfile = right.makefile("rb"), right.makefile("wb")
    try:
        write_frame(wfile, {"op": OP_RUN, "id": "live-2", "prompt": "hi"})
        first = read_frame(rfile)
        assert first.get("op") == OP_TOKEN
        write_frame(wfile, {"op": OP_ABORT, "id": "live-2"})
        _tokens, end = _collect_end(rfile)
        assert end.get("op") == OP_END and end.get("aborted") is True
    finally:
        _stop_worker(right, thread)


def test_worker_abort_after_final_token_reports_aborted():
    """abort 在最后一个 token 之后、循环结束之前到达：不能发正常 end（假完成）。"""
    worker_holder: dict = {}

    class _LastTokenAbortEngine:
        def stream(self, prompt, **kwargs):
            yield "only"
            # 模拟 abort 恰在最后一个 token 之后到达：旧实现走 for-else 发正常
            # end，客户端会拿到 finish_reason=stop 的"完整答案"。
            worker_holder["worker"]._abort_requested.set()

        def last_perf(self):
            return (1, 1)

        def abort(self):
            pass

    engine = _LastTokenAbortEngine()
    worker, right, thread = _start_worker(engine)
    worker_holder["worker"] = worker
    right.settimeout(10)
    rfile, wfile = right.makefile("rb"), right.makefile("wb")
    try:
        write_frame(wfile, {"op": OP_RUN, "id": "tail-1", "prompt": "hi"})
        tokens, end = _collect_end(rfile)
        assert tokens == ["only"]
        assert end.get("op") == OP_END and end.get("aborted") is True, (
            "最后一个 token 之后到达的 abort 被当成正常结束：%r" % (end,))
    finally:
        _stop_worker(right, thread)


class _BrokenWFile:
    def write(self, _data):
        raise OSError("broken pipe")

    def flush(self):
        raise OSError("broken pipe")


def test_worker_send_failure_marks_connection_dead_without_raising():
    """写失败只标连接死亡（不向推理线程抛异常），并触发进程退出由 supervisor 重建。"""
    worker = Worker(FakeEngine(), heartbeat_interval_s=3600)

    class _Conn:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self, _how):
            self.shutdown_calls += 1

    conn = _Conn()
    worker._conn = conn
    assert worker._send(_BrokenWFile(), {"op": OP_TOKEN, "text": "x"}) is False
    assert worker._conn_closed.is_set() and worker._shutdown.is_set()
    assert conn.shutdown_calls == 1, "写失败后应关闭 socket 唤醒读线程"

    # 推理线程遇到写失败应正常退出（而不是抛异常死掉/卡死）
    worker2 = Worker(FakeEngine(), heartbeat_interval_s=3600)
    worker2._run_queue.put({"op": OP_RUN, "id": "r1", "prompt": "hi"})
    thread = threading.Thread(target=worker2._inference_loop,
                              args=(_BrokenWFile(),), daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert worker2._conn_closed.is_set() and worker2._shutdown.is_set()


# ---------------------------------------------------------------------------
# 2) supervisor 侧：request_timeout / abort id / 崩溃恢复
# ---------------------------------------------------------------------------


def _fake_args(*extra):
    return ["fake", "--model", "fake", "--context", "4096", *extra]


def _make_supervisor(*extra, **kwargs):
    defaults = dict(load_timeout_s=60, restart_backoff_s=1,
                    restart_backoff_max_s=2, restart_max_per_hour=10,
                    health_interval_s=1, heartbeat_timeout_s=30)
    defaults.update(kwargs)
    return Supervisor(_fake_args(*extra), **defaults)


class _RecordingConn:
    def __init__(self):
        self.timeouts: list[float] = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)


class _RecordingClient:
    def __init__(self):
        self.conn = _RecordingConn()
        self.got: dict = {}

    def run(self, prompt, **kwargs):
        self.got = kwargs
        yield "ok"


def test_supervisor_applies_queue_request_timeout():
    """queue.request_timeout_s（经 Pipeline 传入）必须落到 supervisor/连接读超时。"""
    sup = Supervisor(["fake"])
    sup.state = "ready"
    client = _RecordingClient()
    sup._client = client
    assert list(sup.execute("p", request_timeout_s=7)) == ["ok"]
    assert client.got["request_timeout_s"] == 7
    assert client.conn.timeouts and client.conn.timeouts[-1] == 7
    assert sup.request_timeout_s == 7


def test_worker_client_abort_only_targets_matching_run_id():
    left, right = socket.socketpair()
    client = WorkerClient(left)
    right.settimeout(5)
    rfile, wfile = right.makefile("rb"), right.makefile("wb")
    seen: dict = {}

    def worker():
        run = read_frame(rfile)
        seen["run"] = run
        write_frame(wfile, {"op": OP_END, "id": run["id"]})
        seen["abort"] = read_frame(rfile)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert list(client.run("p", request_timeout_s=5)) == []
        run_id = seen["run"]["id"]
        assert client.abort("some-other-id") is False, (
            "非当前 run 的 abort 不应写帧")
        assert client.abort(run_id) is True
        thread.join(5)
        assert seen["abort"]["op"] == OP_ABORT
        assert seen["abort"].get("id") == run_id
    finally:
        client.close()
        right.close()


def test_supervisor_health_thread_actively_restarts_crashed_worker():
    """kill worker 后健康线程主动重启：不需要发请求，state 回到 ready。"""
    sup = _make_supervisor()
    sup.start()
    try:
        assert sup.is_ready()
        old = sup._proc
        base = sup.metrics["worker_restarts_total"]
        old.kill()
        old.wait(timeout=5)
        deadline = time.time() + 20
        while time.time() < deadline:
            proc = sup._proc
            if (sup.state == "ready" and proc is not None
                    and proc is not old and proc.poll() is None
                    and sup._client is not None):
                break
            time.sleep(0.1)
        assert sup.state == "ready", (
            "worker 崩溃后健康线程没有主动重启（仍等下一个请求）")
        assert sup.metrics["worker_restarts_total"] == base + 1
        text = "".join(item for item in sup.execute(
            "hi", max_new_tokens=8, request_timeout_s=20)
            if not isinstance(item, dict))
        assert text.startswith("answer")
    finally:
        sup.shutdown()


def test_request_during_recovery_restarts_once_and_returns_fast():
    """请求撞上崩溃窗口：只触发一次重启（不重复 spawn），且快速返回。

    崩溃瞬间第一个请求可能：① 健康线程已开始恢复 → 等恢复后成功；② 还没被
    发现 → 写失败快速 503（不阻塞到 load_timeout），恢复后下一条请求可用。
    两种语义都合法，关键是**同一次崩溃只重启一次**。
    """
    sup = _make_supervisor()
    sup.start()
    try:
        assert sup.is_ready()
        base = sup.metrics["worker_restarts_total"]
        sup._proc.kill()
        sup._proc.wait(timeout=5)
        started = time.monotonic()
        try:
            text = "".join(item for item in sup.execute(
                "hi", max_new_tokens=8, request_timeout_s=20)
                if not isinstance(item, dict))
            assert text.startswith("answer")
        except WorkerDown:
            # 崩溃窗口内快速 503（不长时间阻塞）是可接受语义
            pass
        elapsed = time.monotonic() - started
        assert elapsed < 20, "请求在恢复路径上耗时过长（%.1fs）" % elapsed
        # 恢复完成后，下一条请求必须可用
        deadline = time.time() + 20
        while time.time() < deadline and sup.state != "ready":
            time.sleep(0.1)
        assert sup.state == "ready"
        text2 = "".join(item for item in sup.execute(
            "hi", max_new_tokens=8, request_timeout_s=20)
            if not isinstance(item, dict))
        assert text2.startswith("answer")
        time.sleep(0.5)                    # 给可能的后台恢复线程收尾
        assert sup.metrics["worker_restarts_total"] == base + 1, (
            "同一次崩溃被重启了多次（健康线程与请求路径重复 spawn）")
    finally:
        sup.shutdown()


def test_spawn_failure_kills_worker_child(monkeypatch):
    """加载超时/失败不得留下仍在占 NPU 的孤儿 worker（P2-2 回归）。"""
    import rkllm.server.supervisor as sup_mod

    created = []
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        proc = real_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(sup_mod.subprocess, "Popen", fake_popen)
    sup = Supervisor(["fake", "--model", "fake"], load_timeout_s=1,
                     restart_backoff_s=1)
    with pytest.raises(WorkerDown):
        sup.start()
    assert sup.state == "error"
    assert created, "未发生子进程 spawn"
    deadline = time.time() + 5
    while time.time() < deadline and created[0].poll() is None:
        time.sleep(0.1)
    assert created[0].poll() is not None, (
        "加载失败后 worker 子进程仍存活（孤儿占 NPU）")


# ---------------------------------------------------------------------------
# 3) queue 侧：非流式阻塞提交的 cancelled 能力（routes 接线由 Lead 收口）
# ---------------------------------------------------------------------------


def test_queue_cancelled_works_for_blocking_nonstream_submit():
    """queue.run(cancelled=) + supervisor.abort() 可打断阻塞提交并归还执行槽。

    当前 routes/pipeline_runner 尚未透传断连回调（Lead 在 P2.1/P2.3 收口后
    接线）；这里固定 queue/supervisor 一侧的能力契约，避免接线时再改底层。
    """

    class _BlockingSupervisor:
        metrics = {"worker_restarts_total": 0}

        def __init__(self):
            self.started = threading.Event()
            self.stop = threading.Event()

        def execute(self, prompt, **kwargs):
            self.started.set()
            if not self.stop.wait(10):
                raise AssertionError("未被 abort：阻塞提交无法取消")
            raise RequestAborted("inference aborted before completion")

        def abort(self):
            self.stop.set()

    sup = _BlockingSupervisor()
    queue = InferenceQueue(max_wait=1, wait_timeout_s=2)
    cancelled = threading.Event()
    outcome: dict = {}

    def consume():
        try:
            list(queue.run(lambda: sup.execute("x"),
                           cancelled=cancelled.is_set))
            outcome["ok"] = True
        except RequestAborted:
            outcome["aborted"] = True

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    assert sup.started.wait(5)
    cancelled.set()
    sup.abort()
    thread.join(5)
    assert not thread.is_alive()
    assert outcome.get("aborted") is True
    # 取消后执行槽必须已归还：下一条请求立即可执行
    assert list(queue.run(lambda: iter(["y"]))) == ["y"]


# ---------------------------------------------------------------------------
# 4) worker 侧 cache.similarity 接线（cli_engine 一行待 Lead）
# ---------------------------------------------------------------------------


def test_worker_cache_similarity_reaches_promptcache(monkeypatch):
    import rkllm.worker.engine as engine_mod
    import rkllm.worker.main as wmain
    import rkllm.worker.promptcache as cache_mod

    captured: dict = {}

    class _FakeCache:
        def __init__(self, directory, max_mb=512, ttl_s=604800,
                     similarity=0.5):
            captured["dir"] = directory
            captured["similarity"] = similarity

    class _FakeRealEngine:
        def __init__(self, **kwargs):
            captured["prompt_cache"] = kwargs.get("prompt_cache")

    monkeypatch.setattr(cache_mod, "PromptCache", _FakeCache)
    monkeypatch.setattr(engine_mod, "RealEngine", _FakeRealEngine)

    args = SimpleNamespace(
        engine="real", model="m.rkllm", platform="rk3588", context=4096,
        max_tokens=512, no_history=True, threads=2, lora_path=[],
        cache_dir="/tmp/rkllm-cache", no_cache=False, cache_max_mb=8,
        cache_ttl_s=60, cache_similarity=0.33)
    wmain.build_engine(args)
    assert captured["dir"] == "/tmp/rkllm-cache"
    assert abs(captured["similarity"] - 0.33) < 1e-9
    assert captured["prompt_cache"] is not None


def test_worker_cli_parses_cache_similarity(monkeypatch):
    import rkllm.worker.main as wmain

    captured: dict = {}

    def fake_build_engine(args):
        captured["args"] = args
        return FakeEngine()

    class _NoopWorker:
        def __init__(self, *a, **k):
            pass

        def serve_forever(self, *a, **k):
            pass

    monkeypatch.setattr(wmain, "build_engine", fake_build_engine)
    monkeypatch.setattr(wmain, "Worker", _NoopWorker)
    assert wmain.main(["--address", "127.0.0.1:1", "--engine", "fake",
                       "--cache-similarity", "0.25"]) == 0
    assert abs(captured["args"].cache_similarity - 0.25) < 1e-9
    with pytest.raises(SystemExit):
        wmain.main(["--address", "127.0.0.1:1", "--engine", "fake",
                    "--cache-similarity", "2"])
