"""worker 帧协议与 TCP 回环测试（S1：D2 验收「kill worker 后 HTTP 存活」的
worker 侧基座；supervisor 侧在 S2 补齐）。

Windows 无 AF_UNIX（Python 层不可用），传输层参数化：--tcp 走回环。
"""

from __future__ import annotations

import io
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.worker.frames import (FrameError, read_frame, write_frame,
                                 MAX_FRAME_BYTES)
from rkllm.worker.fake import FakeEngine

SRC = Path(__file__).resolve().parents[1] / "src"


# -- 帧编解码（纯函数） -------------------------------------------------------


def test_frame_roundtrip():
    buf = io.BytesIO()
    write_frame(buf, {"op": "token", "id": "r1", "text": "你"})
    buf.seek(0)
    assert read_frame(buf) == {"op": "token", "id": "r1", "text": "你"}


def test_frame_eof():
    assert read_frame(io.BytesIO(b"")) == {}


def test_frame_bad_length():
    buf = io.BytesIO(struct.pack(">I", MAX_FRAME_BYTES + 1))
    with pytest.raises(FrameError):
        read_frame(buf)


def test_frame_truncated_body():
    buf = io.BytesIO(struct.pack(">I", 10) + b"abc")
    with pytest.raises(FrameError):
        read_frame(buf)


def test_frame_bad_json():
    body = json.dumps({"op": "x"}).encode()[:-3]
    buf = io.BytesIO(struct.pack(">I", len(body)) + body)
    with pytest.raises(FrameError):
        read_frame(buf)


# -- worker 子进程 TCP 回环（fake 引擎） ---------------------------------------


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def worker_proc():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "rkllm.worker.main", "--address",
         "127.0.0.1:%d" % port, "--tcp", "--engine", "fake",
         "--model", "fake-model"],
        cwd=str(SRC), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 等 ready 帧
    header = proc.stdout.read(4)
    assert len(header) == 4, "worker 未启动: " + proc.stderr.read(2000).decode()
    (length,) = struct.unpack(">I", header)
    ready = json.loads(proc.stdout.read(length))
    assert ready["op"] == "ready"
    yield port, proc
    proc.terminate()
    proc.wait(timeout=10)


def _connect(port):
    conn = socket.create_connection(("127.0.0.1", port), timeout=15)
    return conn, conn.makefile("rb"), conn.makefile("wb")


def test_worker_run_and_end(worker_proc):
    port, _ = worker_proc
    conn, rfile, wfile = _connect(port)
    write_frame(wfile, {"op": "run", "id": "r1", "prompt": "hi"})
    tokens, end = [], None
    while end is None:
        frame = read_frame(rfile)
        if frame["op"] == "token":
            tokens.append(frame["text"])
        elif frame["op"] == "end":
            end = frame
    assert "".join(tokens) == "answer 1 to: hi"
    assert end["id"] == "r1" and "aborted" not in end
    assert end["perf"]["generate_tokens"] > 0
    conn.close()


def test_worker_ping_state(worker_proc):
    port, _ = worker_proc
    conn, rfile, wfile = _connect(port)
    write_frame(wfile, {"op": "ping"})
    assert read_frame(rfile) == {"op": "state", "state": "ready"}
    conn.close()


def test_worker_abort_midstream(worker_proc):
    port, _ = worker_proc
    conn, rfile, wfile = _connect(port)
    write_frame(wfile, {"op": "run", "id": "r2", "prompt": "long"})
    # 读到一个 token 后立刻发 abort
    frame = read_frame(rfile)
    assert frame["op"] == "token"
    write_frame(wfile, {"op": "abort"})
    while True:
        frame = read_frame(rfile)
        if frame["op"] == "end":
            break
    assert frame.get("aborted") is True
    conn.close()


def test_worker_two_sequential_runs_increment_turns(worker_proc):
    port, _ = worker_proc
    conn, rfile, wfile = _connect(port)
    texts = []
    for i in (3, 4):
        write_frame(wfile, {"op": "run", "id": "r%d" % i, "prompt": "p"})
        parts = []
        while True:
            frame = read_frame(rfile)
            if frame["op"] == "token":
                parts.append(frame["text"])
            elif frame["op"] == "end":
                break
        texts.append("".join(parts))
    assert texts[0].startswith("answer ")
    assert texts[1].startswith("answer ")
    # turn 计数跨连接递增（单 worker 单实例）
    n0 = int(texts[0].split()[1])
    n1 = int(texts[1].split()[1])
    assert n1 == n0 + 1
    assert texts[0].endswith("p") and texts[1].endswith("p")
    conn.close()
