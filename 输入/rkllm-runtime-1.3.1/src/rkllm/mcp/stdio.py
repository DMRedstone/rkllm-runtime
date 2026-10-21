"""stdio 传输：读循环线程 + 写锁 + 取消传导 + 进度通知（MCP 设计 §6.1）。

线程模型（M9/M4 处置）：
- 读循环在独立线程：调用进行中仍能收到 notifications/cancelled 与
  local_llm_job 轮询——这是 job 机制能工作的前提；
- stdout 写入全局锁：一条协议报文绝不拆开；
- 单条消息上限 8 MiB；非法 JSON 回 -32700 后继续服务，不退出；
- 进程退出时取消在飞任务。
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Callable

from .jobs import JobTable, estimate_seconds
from .protocol import (Dispatcher, JsonRpcError, PARSE_ERROR,
                       SERVER_INFO, detect_protocol, result_payload)
from . import tools as tools_mod

MAX_MESSAGE_BYTES = 8 * 1024 * 1024


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
        self._inflight: dict[str, Any] = {}     # request_id -> cancel flag holder
        self._lock = threading.Lock()
        self._protocol_version = protocol_version
        self.dispatcher = self._build_dispatcher()

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
        disp = Dispatcher(version)
        prefix = self.cfg["tool_prefix"]
        handlers = tools_mod.build_handlers(
            self.backend, self.cfg, self.jobs,
            extract_retries=self.cfg.get("extract_retries", 2))

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
                job = self._inflight.get(request_id)
            if job is not None:
                job.cancel()
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
        call_id = params.get("_meta", {}).get("progressToken")
        prefix = self.cfg["tool_prefix"]
        short = name[len(prefix):] if name.startswith(prefix) else name
        arguments = params.get("arguments") or {}

        if short == "job":
            return self._job_tool(arguments)
        handler = self._handler_for(short)
        if handler is None:
            raise JsonRpcError(-32602, f"unknown tool: {name}")

        # job 化判定：估算耗时超过阈值（task_mode=never 强制同步）
        estimate = (self._job_estimate or self._default_estimate)(arguments)
        task_mode = self.cfg.get("task_mode", "auto")
        threshold = self.cfg.get("task_threshold_s", 45)
        if task_mode != "never" and estimate > threshold:
            return self._run_as_job(short, handler, arguments, call_id)

        # 同步执行（注册取消句柄；cancelled 通知可打断后续重试轮）
        job = self.jobs.create(request_id=None)
        with self._lock:
            self._inflight[str(call_id)] = job if call_id is not None else None
        try:
            result = handler(arguments)
            if job.cancel_requested:
                raise JsonRpcError(-32800, "cancelled")
            return result
        finally:
            with self._lock:
                self._inflight.pop(str(call_id), None)

    def _handler_for(self, short: str):
        handlers = tools_mod.build_handlers(
            self.backend, self.cfg, self.jobs,
            extract_retries=self.cfg.get("extract_retries", 2))
        return handlers.get(short)

    def _default_estimate(self, arguments: dict) -> float:
        max_tokens = int(arguments.get("max_tokens")
                         or self.cfg.get("default_max_tokens", 512))
        text = json.dumps(arguments, ensure_ascii=False)
        prefill = max(1, len(text) // 2)   # 中文约 2 字符/token
        return estimate_seconds(self.model_class, prefill, max_tokens)

    def _run_as_job(self, short: str, handler, arguments: dict,
                    call_id) -> dict:
        job = self.jobs.create()
        if self.jobs.get(job.id) is None:      # 防御：刚建即被逐出
            raise JsonRpcError(-32603, "job table overflow")

        def runner():
            try:
                if call_id is not None:
                    interval = max(1, self.cfg.get("progress_interval_s", 5))
                    for step in range(1, 13):   # 最多 12 次，够覆盖 27B 长任务
                        if job.cancel_requested:
                            return
                        import time
                        time.sleep(interval)
                        self._notify_progress(call_id, step * interval, None)
                result = handler(arguments)
                job.complete(result)
            except Exception as exc:
                job.fail(str(exc)[:300])

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
                    self._protocol_version = detect_protocol(first)
                    self.dispatcher = self._build_dispatcher()
                except json.JSONDecodeError:
                    pass
            response = self.dispatcher.handle_message(line)
            if response is not None:
                self._send(response)
