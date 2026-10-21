"""stdio 传输：读线程 + 工作线程池 + 取消传导 + 进度通知（MCP 设计 §6.1）。

线程模型（本批重写，审计 P0-2）：
- reader 线程只做三件事：读一行、处理通知（cancelled/ping）、把 tools/call
  提交给工作线程池。**绝不在 reader 里执行工具**——否则一次 ask（最长 45 s）
  期间收不到取消通知，也收不到 job 轮询，job 机制形同虚设。
- 工作线程池 max_workers=4：既保证调用中仍能响应 job/status，又不会对板端
  形成压力（板端本来是单槽排队，背压在服务端）。
- stdout 写入全局锁：一条协议报文绝不拆开。
- 单条消息上限 8 MiB；非法 JSON 回 -32700 后继续服务，不退出。
- 取消：notifications/cancelled 用 JSON-RPC 请求 id 匹配（此前错用
  progressToken，取消对不上号），命中后置位取消标志，工具层把它传给
  Backend → 在 SSE chunk 边界中断。
- 进程退出时取消在飞任务。
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .jobs import JobTable, estimate_seconds
from .protocol import (Dispatcher, JsonRpcError, PARSE_ERROR,
                       SERVER_INFO, detect_protocol)
from . import tools as tools_mod

MAX_MESSAGE_BYTES = 8 * 1024 * 1024
POOL_WORKERS = 4


class CancelFlag:
    """同步调用的取消句柄（不占 job 表，避免伪 job 把真 job 挤掉）。"""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> bool:
        self._event.set()
        return True

    def is_set(self) -> bool:
        return self._event.is_set()


class StdioServer:
    def __init__(self, backend, cfg: dict, model_class: str = "small",
                 protocol_version: str | None = None,
                 job_estimate: Callable[[dict], float] | None = None):
        self.backend = backend
        self.cfg = cfg
        self.model_class = model_class
        self._job_estimate = job_estimate
        self.jobs = JobTable(ttl_s=cfg.get("job_ttl_s", 600),
                             max_jobs=cfg.get("job_max", 8))
        self._write_lock = threading.Lock()
        self._inflight: dict[str, Any] = {}     # request_id -> Job | CancelFlag
        self._lock = threading.Lock()
        self._protocol_version = protocol_version
        # mcp.max_concurrent：同时在飞的工具调用数。默认 1（板端串行），
        # 但读循环线程不占池，所以 job/status/cancel 随时可响应。
        workers = max(1, int(cfg.get("max_concurrent") or POOL_WORKERS))
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="mcp-call")
        self._handlers = tools_mod.build_handlers(
            backend, cfg, self.jobs,
            extract_retries=cfg.get("extract_retries", 2))
        self.dispatcher = self._build_dispatcher()
        # 档位从板端 /health 取（F9）：失败不影响启动，退回 env 默认
        threading.Thread(target=self._refresh_model_class, daemon=True,
                         name="mcp-model-class").start()

    def _refresh_model_class(self) -> None:
        try:
            info = self.backend.status()
            if isinstance(info, dict):
                found = info.get("model_class")
                if found in ("small", "medium", "large"):
                    self.model_class = found
        except Exception:
            pass

    # -- 底层 IO ------------------------------------------------------------

    def _send(self, obj: dict) -> None:
        with self._write_lock:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    def _notify_progress(self, progress_token, progress: float,
                         total: float | None = None) -> None:
        if not self.cfg.get("progress_interval_s", 5):
            return
        note: dict = {"jsonrpc": "2.0",
                      "method": "notifications/progress",
                      "params": {"progressToken": progress_token,
                                 "progress": progress}}
        if total is not None:
            note["params"]["total"] = total
        self._send(note)

    # -- 方法注册 ------------------------------------------------------------

    def _build_dispatcher(self) -> Dispatcher:
        version = self._protocol_version or "2024-11-05"
        supported = self.cfg.get("protocol_versions") or None
        disp = Dispatcher(version, supported=supported)
        prefix = self.cfg["tool_prefix"]

        def initialize(params: dict) -> dict:
            return {"protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO}

        def tools_list(params: dict) -> dict:
            return {"tools": tools_mod.tool_definitions(prefix)}

        def server_discover(params: dict) -> dict:
            return {"serverInfo": SERVER_INFO,
                    "protocolVersion": version,
                    "tools": [t["name"] for t in
                              tools_mod.tool_definitions(prefix)]}

        def ping(params: dict) -> dict:
            return {}

        def tools_call(params: dict) -> dict:
            return self._tools_call(params)

        def cancelled(params: dict) -> dict:
            request_id = str(params.get("requestId"))
            with self._lock:
                handle = self._inflight.get(request_id)
            if handle is not None:
                handle.cancel()
            return {}

        disp.register("initialize", initialize)
        disp.register("tools/list", tools_list)
        disp.register("tools/call", tools_call)
        disp.register("ping", ping)
        disp.register("notifications/cancelled", cancelled)
        if version == "2026-07-28":
            disp.register("server/discover", server_discover)
        return disp

    # -- tools/call 编排：job 判定 + 进度 + 取消 ------------------------------

    def _tools_call(self, params: dict) -> dict:
        name = str(params.get("name", ""))
        progress_token = params.get("_meta", {}).get("progressToken")
        request_id = self.dispatcher.current_request_id
        prefix = self.cfg["tool_prefix"]
        short = name[len(prefix):] if name.startswith(prefix) else name
        arguments = params.get("arguments") or {}

        if short == "job":
            return self._job_tool(arguments)
        handler = self._handlers.get(short)
        if handler is None:
            raise JsonRpcError(-32602, f"unknown tool: {name}")

        # status 只读不推理，恒同步秒回（实测发现：job 化判定按 max_tokens
        # 估算会把 status 也转后台，违反设计 §4.1「秒回」语义）
        if short == "status":
            return handler(arguments)

        # job 化判定：估算耗时超过阈值（task_mode=never 强制同步）
        estimate = (self._job_estimate or self._default_estimate)(arguments)
        task_mode = self.cfg.get("task_mode", "auto")
        threshold = self.cfg.get("task_threshold_s", 45)
        if task_mode != "never" and estimate > threshold:
            return self._run_as_job(short, handler, arguments, progress_token,
                                    request_id)

        # 同步执行：用轻量取消标志注册到 in-flight（不占 job 表）
        flag = CancelFlag()
        key = str(request_id) if request_id is not None else None
        if key is not None:
            with self._lock:
                self._inflight[key] = flag
        try:
            result = handler(arguments, should_cancel=flag.is_set)
            if flag.is_set():
                return {"content": [{"type": "text", "text": "任务已取消"}],
                        "isError": True}
            return result
        except tools_mod.Cancelled:
            return {"content": [{"type": "text", "text": "任务已取消"}],
                    "isError": True}
        finally:
            if key is not None:
                with self._lock:
                    self._inflight.pop(key, None)

    def _default_estimate(self, arguments: dict) -> float:
        max_tokens = int(arguments.get("max_tokens")
                         or self.cfg.get("default_max_tokens", 512))
        text = json.dumps(arguments, ensure_ascii=False)
        prefill = max(1, len(text) // 2)   # 中文约 2 字符/token
        return estimate_seconds(self.model_class, prefill, max_tokens)

    def _run_as_job(self, short: str, handler, arguments: dict,
                    progress_token, request_id=None) -> dict:
        job = self.jobs.create(request_id=str(request_id)
                               if request_id is not None else None)
        if self.jobs.get(job.id) is None:      # 防御：刚建即被逐出
            raise JsonRpcError(-32603, "job table overflow")
        key = str(request_id) if request_id is not None else None
        if key is not None:
            with self._lock:
                self._inflight[key] = job

        def runner():
            try:
                if progress_token is not None:
                    interval = max(1, self.cfg.get("progress_interval_s", 5))
                    for step in range(1, 13):   # 最多 12 次，够覆盖 27B 长任务
                        if job.cancel_requested:
                            break
                        import time
                        time.sleep(interval)
                        self._notify_progress(progress_token, step * interval,
                                              None)
                if job.cancel_requested:
                    return
                # 注意：必须传**可调用对象**——`job.cancel_requested` 是 property，
                # 直接传会得到当时的 bool（永远 False），取消检查就等于没接
                # （2026-10-21 真机发现：cancel 后槽位仍被占到任务结束）。
                result = handler(arguments,
                                 should_cancel=lambda: job.cancel_requested)
                job.complete(result)
            except tools_mod.Cancelled:
                job.cancel()
            except Exception as exc:
                if job.cancel_requested:
                    return
                job.fail(str(exc)[:300])
            finally:
                if key is not None:
                    with self._lock:
                        self._inflight.pop(key, None)

        thread = threading.Thread(target=runner, daemon=True,
                                  name="mcp-job-" + job.id)
        thread.start()
        return {"content": [{"type": "text", "text": json.dumps(
            {"job_id": job.id, "status": "working",
             "poll_hint_s": self.cfg.get("job_poll_hint_s", 2),
             "message": "任务预计超过 %d 秒，已转后台任务；请用 %sjob 轮询"
                        % (self.cfg.get("task_threshold_s", 45),
                           self.cfg["tool_prefix"])},
            ensure_ascii=False)}]}

    def _job_tool(self, arguments: dict) -> dict:
        job_id = str(arguments.get("job_id", ""))
        action = arguments.get("action", "get")
        job = self.jobs.get(job_id)
        if job is None:
            return {"content": [{"type": "text",
                                 "text": "job 不存在或已过期：" + job_id}],
                    "isError": True}
        if action == "cancel":
            job.cancel()
            return {"content": [{"type": "text",
                                 "text": json.dumps(job.summary(),
                                                    ensure_ascii=False)}]}
        return {"content": [{"type": "text", "text": json.dumps(
            {**job.summary(),
             "result": job.result if job.state == "completed" else None,
             "error": job.error if job.state == "failed" else None},
            ensure_ascii=False)}]}

    # -- 主循环 ---------------------------------------------------------------

    def serve_forever(self) -> None:
        reader = threading.Thread(target=self._read_loop, daemon=True,
                                  name="mcp-reader")
        reader.start()
        reader.join()
        # 进程退出：取消在飞任务（M4：断开后 NPU 不空转）
        for job in self.jobs.all():
            job.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _read_loop(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if len(line.encode("utf-8", "ignore")) > MAX_MESSAGE_BYTES:
                self._send({"jsonrpc": "2.0", "id": None,
                            "error": {"code": PARSE_ERROR,
                                      "message": "message too large"}})
                continue
            if self._protocol_version is None:
                try:
                    first = json.loads(line)
                    self._protocol_version = detect_protocol(
                        first, self.cfg.get("protocol_versions") or None)
                    self.dispatcher = self._build_dispatcher()
                    self._log_first_connect(first)
                except json.JSONDecodeError:
                    pass
            # tools/call 交给工作线程：reader 必须随时可读，才能收到 cancelled
            if self._is_tool_call(line):
                self._pool.submit(self._handle_and_send, line)
                continue
            response = self.dispatcher.handle_message(line)
            if response is not None:
                self._send(response)

    @staticmethod
    def _is_tool_call(line: str) -> bool:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return False
        return (isinstance(msg, dict) and msg.get("method") == "tools/call"
                and msg.get("id") is not None)

    def _handle_and_send(self, line: str) -> None:
        try:
            response = self.dispatcher.handle_message(line)
        except Exception as exc:              # 兜底：工作线程异常不能静默
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32603, "message": str(exc)[:300]}}
        if response is not None:
            self._send(response)

    def _log_first_connect(self, first: dict) -> None:
        """首连事实记录（M10）：协议版本、clientInfo、时间——实测回写配置的依据。

        stderr 在 WorkBuddy 拉起下不可见，落文件；路径可由
        RKLLM_MCP_CONNECT_LOG 覆盖，默认 ~/.config/rkllm/connect.log。
        """
        import os
        import time as _time
        try:
            path = os.environ.get(
                "RKLLM_MCP_CONNECT_LOG",
                os.path.join(os.path.expanduser("~"), ".config", "rkllm",
                             "connect.log"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            params = first.get("params") or {}
            entry = {
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                "method": first.get("method"),
                "detected_protocol": self._protocol_version,
                "requested_protocol": (params.get("_meta") or
                                       {}).get("protocolVersion")
                                       or params.get("protocolVersion"),
                "clientInfo": params.get("clientInfo"),
                "meta_keys": sorted((params.get("_meta") or {}).keys()),
            }
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass                                   # 记录失败不影响服务
