"""推理 worker：UDS 帧协议 + 引擎隔离（改造设计 §6.1，S1）。

进程模型：
- HTTP 进程（S2）只做协议与调度，不加载模型；
- 本 worker 进程持有 ctypes 句柄与 NPU，段错误只死自己；
- 控制通道与 token 流在同一 socket 上按帧复用：abort 帧由控制线程消费
  （rkllm_abort 从非推理线程调用，与旧实现行为一致）。

帧格式：4 字节大端长度前缀 + UTF-8 JSON。op 一览：
  客户端 → worker: run / abort / ping / shutdown
  worker → 客户端: ready / token / end / error / state
"""

from __future__ import annotations

import json
import struct
from typing import Any

__all__ = ["FrameError", "read_frame", "write_frame", "MAX_FRAME_BYTES"]

MAX_FRAME_BYTES = 8 * 1024 * 1024   # 与 MCP 消息上限同源


class FrameError(Exception):
    pass


def read_frame(rfile) -> dict[str, Any]:
    """从二进制文件对象读一帧；对端正常关闭返回 {}（EOF 语义由调用方判定）。"""
    head = rfile.read(4)
    if not head:
        return {}
    if len(head) < 4:
        raise FrameError("truncated frame header")
    (length,) = struct.unpack(">I", head)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise FrameError("bad frame length %d" % length)
    body = rfile.read(length)
    if len(body) < length:
        raise FrameError("truncated frame body")
    try:
        frame = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError("bad frame json: %s" % exc)
    if not isinstance(frame, dict) or "op" not in frame:
        raise FrameError("frame must be an object with 'op'")
    return frame


def write_frame(wfile, frame: dict[str, Any]) -> None:
    body = json.dumps(frame, ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError("frame too large: %d" % len(body))
    wfile.write(struct.pack(">I", len(body)))
    wfile.write(body)
    wfile.flush()
