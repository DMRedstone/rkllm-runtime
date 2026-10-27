"""stdio 传输：读线程 + 推理池 + 控制池 + 取消传导 + 进度通知（MCP 设计 §6.1）。

线程模型：
- reader 线程只做三件事：读一行、处理通知（cancelled/ping）、把 tools/call
  提交给线程池。**绝不在 reader 里执行工具**——否则一次 ask（最长 45 s）
  期间收不到取消通知，也收不到 job 轮询，job 机制形同虚设。
- 两个池：推理池 max_workers=mcp.max_concurrent（默认 1，板端串行）；控制池
  固定 2 线程，只跑 status/job。长同步 ask 占住推理池时，job/status 仍能秒回
  （审计项：单池时 job 轮询会被排队的同步调用挡住）。
- stdout 写入全局锁：一条协议报文绝不拆开。
- 单条消息上限 8 MiB；非法 JSON 回 -32700 后继续服务，不退出。
- 取消：notifications/cancelled 用 JSON-RPC 请求 id 匹配，命中后置位取消标志，
  工具层把它传给 Backend → 在 SSE chunk 边界中断。
- 进度：独立心跳线程在任务执行期间按 interval 发 notifications/progress，
  不再像旧实现那样先睡满 12 个周期才真正开始推理。
- 进程退出时取消在飞任务。
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .jobs import (JobTable, estimate_seconds, job_timeout_s,
                   sync_token_cap)
from .protocol import (Dispatcher, JsonRpcError, INVALID_PARAMS, PARSE_ERROR,
                       PROTOCOL_2026, SERVER_INFO,
                       TOOLS_LIST_CACHE_META_2026, detect_protocol)
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
        # O-3：协议输出走私有 fd（serve_forever 里 dup 出来），None 时回退 sys.stdout
        self._out = None
        self._inflight: dict[str, Any] = {}     # request_id -> Job | CancelFlag
        self._lock = threading.Lock()
        self._sessions: dict[str, list[dict]] = {}
        self._sessions_lock = threading.Lock()
        self._protocol_version = protocol_version
        # mcp.max_concurrent：同时在飞**推理**工具数（默认 1，板端串行）。
        # status/job 不改推理状态，走独立控制池，长 ask 期间仍可秒回。
        workers = max(1, int(cfg.get("max_concurrent") or POOL_WORKERS))
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="mcp-call")
        self._control_pool = ThreadPoolExecutor(max_workers=2,
                                                thread_name_prefix="mcp-ctl")
        # P2：推理并发统一闸门——同步工具走 _pool，异步 job 用裸线程会绕过
        # _pool 的 max_concurrent，job_max=8 时 8 路推理同打板端。用同一信号量
        # 把同步与异步都限制在 max_concurrent。
        self._infer_sem = threading.Semaphore(workers)
        self._handlers = tools_mod.build_handlers(
            backend, cfg, self.jobs,
            extract_retries=cfg.get("extract_retries", 2),
            sessions=self._sessions, sessions_lock=self._sessions_lock)
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
                    self.cfg["model_class"] = found
                context_len = info.get("context_len")
                if isinstance(context_len, int) and context_len > 0:
                    # 工具层的输入预算按板端实际上下文收窄（避免 4k+ 汉字 400）
                    self.cfg["context_len"] = context_len
        except Exception:
            pass

    # -- 底层 IO ------------------------------------------------------------

    def _send(self, obj: dict) -> None:
        with self._write_lock:
            handle = self._out if self._out is not None else sys.stdout
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
            handle.flush()

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
            result = {"tools": tools_mod.tool_definitions(prefix)}
            if version == PROTOCOL_2026:
                # 2026 列表结果必须带缓存元数据（设计 §1.1；工具表是静态的）。
                # 元数据常量集中在 protocol.py，传输层不再写版本分支。
                result.update(TOOLS_LIST_CACHE_META_2026)
            return result

        def server_discover(params: dict) -> dict:
            # 全版本实现：新客户端探针 + 旧客户端兼容探测（设计 §1.2）
            return {"serverInfo": SERVER_INFO,
                    "protocolVersion": version,
                    "supportedVersions": list(disp.supported),
                    "capabilities": {"tools": {}},
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
        disp.register("server/discover", server_discover)
        return disp

    # -- tools/call 编排：job 判定 + 进度 + 取消 ------------------------------

    def _tools_call(self, params: dict) -> dict:
        if not isinstance(params, dict):
            raise JsonRpcError(INVALID_PARAMS, "tools/call params 必须是对象")
        name = str(params.get("name", ""))
        meta = params.get("_meta")
        progress_token = (meta.get("progressToken")
                          if isinstance(meta, dict) else None)
        request_id = self.dispatcher.current_request_id
        prefix = self.cfg["tool_prefix"]
        short = name[len(prefix):] if name.startswith(prefix) else name
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_PARAMS, "arguments 必须是对象")

        if short == "job":
            return self._job_tool(arguments)
        handler = self._handlers.get(short)
        if handler is None:
            raise JsonRpcError(INVALID_PARAMS, f"unknown tool: {name}")

        # status 只读不推理，恒同步秒回（实测发现：job 化判定按 max_tokens
        # 估算会把 status 也转后台，违反设计 §4.1「秒回」语义）
        if short == "status":
            return handler(arguments)

        # job 化判定：估算耗时超过阈值（task_mode=never 强制同步 / always 强制 job）
        estimator = self._job_estimate
        estimate = (estimator(arguments) if estimator
                    else self._default_estimate(arguments, short))
        task_mode = self.cfg.get("task_mode", "auto")
        try:
            threshold = float(self.cfg.get("task_threshold_s", 45))
        except (TypeError, ValueError):
            threshold = 45.0
        force_job = task_mode == "always"
        if task_mode != "never" and (force_job or estimate > threshold):
            # job 的读超时必须覆盖"真的生成满 max_tokens"的最坏情况，而不是
            # 只覆盖判定用的预期值（否则 large 默认 ask 判 job 后仍会被超时掐断）。
            worst = self._job_worst_estimate(arguments, short)
            return self._run_as_job(short, handler, arguments, progress_token,
                                    request_id, estimate=estimate,
                                    timeout_s=self._job_timeout(worst))

        # 同步执行：复用 reader 在入池前预注册的 CancelFlag（排队期间也能
        # 被 notifications/cancelled 取消）。若未预注册则兜底新建。
        key = str(request_id) if request_id is not None else None
        with self._lock:
            flag = self._inflight.get(key) if key is not None else None
        if flag is None:
            flag = CancelFlag()
            if key is not None:
                with self._lock:
                    self._inflight[key] = flag
        try:
            with self._infer_sem:
                # 同步调用的生成上限夹到"估算仍落在阈值内"的安全值：默认
                # max_tokens=512 不再把短任务恒转 job，也不会真跑过客户端
                # 60 s 硬超时；被夹断时 finish_reason=length 会给出提示。
                result = handler(
                    arguments, should_cancel=flag.is_set,
                    max_tokens_cap=self._sync_token_cap(arguments, short),
                    timeout_s=float(self.cfg.get("target_timeout_s", 120)))
            if flag.is_set():
                return {"content": [{"type": "text", "text": "任务已取消"}],
                        "isError": True, "_meta": {"cancelled": True}}
            return result
        except tools_mod.Cancelled:
            return {"content": [{"type": "text", "text": "任务已取消"}],
                    "isError": True, "_meta": {"cancelled": True}}
        finally:
            if key is not None:
                with self._lock:
                    self._inflight.pop(key, None)

    def _estimate_decode_tokens(self, arguments: dict,
                                short: str | None) -> int:
        """判定 job 时用的"预期生成 token"。

        旧实现直接拿 max_tokens 当预期输出：默认 512 在 4B 上估算 114 s，
        导致 ask/chat/batch/extract **恒转 job**。现在：
        - 未显式传 max_tokens 时按 estimate_tokens_cap（默认 128）估；
        - ask 显式传 max_tokens 时按显式值估（大请求继续转 job，避免超时）；
        - extract 按 (重试次数+1) 次尝试估（校验重试是真实的多次推理）；
        - batch 按段数估（串行逐段推理，旧估算漏乘段数）。
        """
        try:
            cap = int(self.cfg.get("estimate_tokens_cap", 128) or 128)
        except (TypeError, ValueError):
            cap = 128
        try:
            default = int(self.cfg.get("default_max_tokens", 512) or 512)
        except (TypeError, ValueError):
            default = 512
        per_call = min(default, cap)
        explicit = arguments.get("max_tokens")
        if explicit is not None:
            try:
                per_call = max(1, int(explicit))
            except (TypeError, ValueError):
                per_call = min(default, cap)
        if short == "extract":
            try:
                retries = max(0, int(self.cfg.get("extract_retries", 2) or 0))
            except (TypeError, ValueError):
                retries = 0
            return min(default * 2, cap) * (retries + 1)
        if short == "batch":
            segments = arguments.get("segments")
            count = len(segments) if isinstance(segments, list) else 1
            try:
                max_segments = max(1, int(self.cfg.get("batch_max_segments", 20) or 1))
            except (TypeError, ValueError):
                max_segments = 1
            return per_call * max(1, min(count, max_segments))
        return per_call

    def _default_estimate(self, arguments: dict, short: str | None = None) -> float:
        text = json.dumps(arguments, ensure_ascii=False)
        prefill = max(1, len(text) // 2)   # 中文约 2 字符/token
        return estimate_seconds(self.model_class, prefill,
                                self._estimate_decode_tokens(arguments, short))

    def _sync_token_cap(self, arguments: dict, short: str | None) -> int:
        """同步路径的安全生成上限（见 jobs.sync_token_cap）。"""
        text = json.dumps(arguments, ensure_ascii=False)
        prefill = max(1, len(text) // 2)
        try:
            threshold = float(self.cfg.get("task_threshold_s", 45))
        except (TypeError, ValueError):
            threshold = 45.0
        cap = sync_token_cap(self.model_class, prefill, threshold)
        if short == "batch":
            segments = arguments.get("segments")
            count = len(segments) if isinstance(segments, list) else 1
            try:
                max_segments = max(1, int(self.cfg.get("batch_max_segments", 20) or 1))
            except (TypeError, ValueError):
                max_segments = 1
            cap = max(8, cap // max(1, min(count, max_segments)))
        return cap

    def _job_worst_estimate(self, arguments: dict,
                            short: str | None) -> float:
        """job 的最坏耗时估算（按实际会请求的 max_tokens/段数/重试次数）。

        判定 job 用"预期生成"（_estimate_decode_tokens，含 cap），但 job 路径
        实际会按 default_max_tokens 生成：读超时必须按最坏值放宽，否则
        large 默认 ask（预期 85 s 判 job、实际可能 341 s）仍会被 120 s 掐断。
        """
        try:
            default = int(self.cfg.get("default_max_tokens", 512) or 512)
        except (TypeError, ValueError):
            default = 512
        explicit = arguments.get("max_tokens")
        if short == "extract":
            try:
                retries = max(0, int(self.cfg.get("extract_retries", 2) or 0))
            except (TypeError, ValueError):
                retries = 0
            tokens = default * 2 * (retries + 1)
        elif short == "batch":
            segments = arguments.get("segments")
            count = len(segments) if isinstance(segments, list) else 1
            try:
                max_segments = max(1, int(self.cfg.get("batch_max_segments", 20) or 1))
            except (TypeError, ValueError):
                max_segments = 1
            tokens = default * max(1, min(count, max_segments))
        else:
            try:
                tokens = max(1, int(explicit)) if explicit is not None else default
            except (TypeError, ValueError):
                tokens = default
        text = json.dumps(arguments, ensure_ascii=False)
        prefill = max(1, len(text) // 2)
        return estimate_seconds(self.model_class, prefill, tokens)

    def _job_timeout(self, estimate: float) -> float:
        """job 读超时 = max(target_timeout, 估算×factor+30)，上限可配。

        旧实现 job 与同步共用 120 s：27B/large 512 token（估算 341 s）必被掐断。
        """
        try:
            base = float(self.cfg.get("target_timeout_s", 120) or 120)
        except (TypeError, ValueError):
            base = 120.0
        try:
            factor = float(self.cfg.get("job_timeout_factor", 1.5) or 1.5)
        except (TypeError, ValueError):
            factor = 1.5
        try:
            max_s = float(self.cfg.get("job_timeout_max_s", 1800) or 1800)
        except (TypeError, ValueError):
            max_s = 1800.0
        return job_timeout_s(estimate, base_s=base, factor=factor, max_s=max_s)

    def _run_as_job(self, short: str, handler, arguments: dict,
                    progress_token, request_id=None, *,
                    estimate: float | None = None,
                    timeout_s: float | None = None) -> dict:
        job = self.jobs.create(short=short)
        if self.jobs.get(job.id) is None:      # 防御：刚建即被逐出
            raise JsonRpcError(-32603, "job table overflow")
        job_timeout = float(timeout_s or self.cfg.get("target_timeout_s", 120))
        key = str(request_id) if request_id is not None else None
        # 复用 reader 在入池前预注册的 CancelFlag：若已取消则立即 job.cancel()
        with self._lock:
            pre_flag = self._inflight.get(key) if key is not None else None
        if isinstance(pre_flag, CancelFlag):
            if pre_flag.is_set():
                job.cancel()
        if key is not None:
            with self._lock:
                self._inflight[key] = job

        def runner():
            stop_progress = threading.Event()

            def heartbeat() -> None:
                interval = max(1, int(self.cfg.get("progress_interval_s", 5)))
                step = 0
                while not stop_progress.wait(interval):
                    if job.cancel_requested:
                        return
                    step += 1
                    self._notify_progress(progress_token, step * interval, None)

            progress_thread = None
            if progress_token is not None and self.cfg.get("progress_interval_s", 5):
                progress_thread = threading.Thread(
                    target=heartbeat, daemon=True, name="mcp-job-progress-" + job.id)
                progress_thread.start()
            try:
                if job.cancel_requested:
                    return
                # 注意：必须传**可调用对象**——`job.cancel_requested` 是 property，
                # 直接传会得到当时的 bool（永远 False），取消检查就等于没接
                # （2026-10-21 真机发现：cancel 后槽位仍被占到任务结束）。
                def _should_cancel():
                    if isinstance(pre_flag, CancelFlag) and pre_flag.is_set():
                        job.cancel()
                        return True
                    return job.cancel_requested
                with self._infer_sem:
                    result = handler(arguments, should_cancel=_should_cancel,
                                     max_tokens_cap=None,
                                     timeout_s=job_timeout)
                if job.cancel_requested:
                    job.cancel()          # batch 等吞掉 Cancelled 时仍要记取消
                    return
                job.complete(result)
            except tools_mod.Cancelled:
                job.cancel()
            except Exception as exc:
                if job.cancel_requested:
                    return
                job.fail(str(exc)[:300])
            finally:
                stop_progress.set()
                if key is not None:
                    with self._lock:
                        self._inflight.pop(key, None)

        thread = threading.Thread(target=runner, daemon=True,
                                  name="mcp-job-" + job.id)
        thread.start()
        info = {"job_id": job.id, "status": "working",
                "poll_hint_s": self.cfg.get("job_poll_hint_s", 2),
                "estimated_s": round(float(estimate), 1) if estimate else None,
                "timeout_s": int(job_timeout),
                "message": "任务预计超过 %d 秒，已转后台任务；请用 %sjob 轮询"
                           % (self.cfg.get("task_threshold_s", 45),
                              self.cfg["tool_prefix"])}
        reply = {"content": [{"type": "text",
                              "text": json.dumps(info, ensure_ascii=False)}]}
        # MCP 客户端会对**每一次** tools/call 结果按 outputSchema 校验，包括
        # 「已转 job、正在运行」这个首次响应。extract 声明了 outputSchema，
        # 所以首帧也必须带 structuredContent；这里给出 job 信封（extract 的
        # outputSchema 是宽松 object），客户端随后按 job_id 轮询最终结果。
        if short == "extract":
            reply["structuredContent"] = dict(info)
        return reply

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
        # 任务完成时把原始工具结果原样返回，尤其要保留 structuredContent /
        # isError：声明了 output_schema 的工具（目前是 extract）如果只在
        # content[0].text 里塞一段 JSON，客户端会判定「有 output schema 却没有
        # structuredContent」并抛 -32600。同步路径本来就直通 handler 结果，
        # 这里只修 job 轮询这条旁路。
        if job.state == "completed" and isinstance(job.result, dict):
            payload = dict(job.result)
            payload.setdefault("content", [{
                "type": "text",
                "text": json.dumps({**job.summary(), "result": job.result},
                                   ensure_ascii=False)}])
            return payload
        if job.state == "failed":
            reply = {"content": [{"type": "text",
                                  "text": job.error or "job failed"}],
                     "isError": True,
                     "job_id": job.id,
                     "status": job.state}
            if job.short == "extract":
                # extract 声明了 outputSchema：轮询结果也带宽松 object，避免
                # 客户端对失败终帧做 schema 校验时报 -32600。
                reply["structuredContent"] = {"error": job.error or "job failed",
                                               "job_id": job.id}
            return reply
        summary = {**job.summary(),
                   "result": job.result if job.state == "completed" else None,
                   "error": job.error if job.state == "failed" else None}
        reply = {"content": [{"type": "text",
                              "text": json.dumps(summary, ensure_ascii=False)}]}
        if job.short == "extract":
            # working / cancelled 的轮询也返回宽松 object 形状
            reply["structuredContent"] = dict(summary)
        return reply

    # -- 主循环 ---------------------------------------------------------------

    def serve_forever(self) -> None:
        # O-3：fd 隔离（对齐 Python SDK stdio_server 的 _claim_fd）。
        # 协议走 dup 出的私有 fd；把 fd 0/1 重定向到空设备/stderr，此后
        # print()、子进程输出、C 库日志都无法污染 JSON-RPC 线。
        try:
            import os
            # 仅在真实 fd 0/1 上隔离；测试注入 StringIO 等非 fd 流时跳过，
            # 回退 sys.stdin/sys.stdout（行为与改动前一致）。
            if sys.stdin.fileno() != 0 or sys.stdout.fileno() != 1:
                raise OSError("stdio not bound to real fd 0/1")
            rfd = os.dup(0)
            wfd = os.dup(1)
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.close(devnull)
            os.dup2(2, 1)
            sys.stdin = os.fdopen(rfd, "r", encoding="utf-8", newline="")
            self._out = os.fdopen(wfd, "w", encoding="utf-8",
                                  newline="\n", buffering=1)
            sys.stdout = sys.stderr          # print() 不再进协议线
        except (OSError, ValueError, AttributeError):
            self._out = None
        reader = threading.Thread(target=self._read_loop, daemon=True,
                                  name="mcp-reader")
        reader.start()
        reader.join()
        # 进程退出：取消在飞任务（M4：断开后 NPU 不空转）
        # P2：原实现只取消 job，排队/执行中的**同步**调用（_inflight 里的
        # CancelFlag）没人置位，NPU 会把这些请求一路算完。这里一并取消；
        # Job 与 CancelFlag 都有幂等的 cancel()，重复调用无副作用。
        with self._lock:
            handles = list(self._inflight.values())
        for handle in handles:
            try:
                handle.cancel()
            except Exception:
                pass
        for job in self.jobs.all():
            job.cancel()
        self._control_pool.shutdown(wait=False, cancel_futures=True)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _read_loop(self) -> None:
        for line in sys.stdin:
            try:
                self._read_line(line)
            except Exception as exc:          # P0-1：单条畸形消息不得杀死 reader 线程
                try:
                    self._send({"jsonrpc": "2.0", "id": None,
                                "error": {"code": -32603,
                                          "message": "internal error: %s"
                                                     % str(exc)[:200]}})
                except Exception:
                    pass

    def _read_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if len(line.encode("utf-8", "ignore")) > MAX_MESSAGE_BYTES:
            self._send({"jsonrpc": "2.0", "id": None,
                        "error": {"code": PARSE_ERROR,
                                  "message": "message too large"}})
            return
        if self._protocol_version is None:
            try:
                first = json.loads(line)
            except json.JSONDecodeError:
                first = None
            # 首包必须是 JSON 对象；数组/null/数字是合法 JSON 但不是对象（P0-1）
            if isinstance(first, dict):
                try:
                    self._protocol_version = detect_protocol(
                        first, self.cfg.get("protocol_versions") or None)
                    self.dispatcher = self._build_dispatcher()
                    self._log_first_connect(first)
                except Exception:
                    pass
        if self.dispatcher is None:
            self._send({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32600,
                                  "message": "expected initialize"}})
            return
        # tools/call 交给线程：reader 必须随时可读，才能收到 cancelled。
        # status/job 走控制池，长同步 ask 占住推理池时它们仍能秒回。
        # 入池前先把 CancelFlag 预注册到 in-flight：排队期间收到
        # notifications/cancelled 也能生效（审计 P2-4）。
        route, request_id = self._tool_route(line)
        if route in ("control", "work"):
            flag = CancelFlag()
            if request_id is not None:
                with self._lock:
                    self._inflight[str(request_id)] = flag
            pool = (self._control_pool if route == "control"
                    else self._pool)
            pool.submit(self._handle_and_send, line, flag, request_id)
            return
        response = self.dispatcher.handle_message(line)
        if response is not None:
            self._send(response)

    def _tool_route(self, line: str) -> tuple[str | None, str | None]:
        """tools/call 分流：status/job → control，其余推理工具 → work。

        返回 (route, request_id)：request_id 用于入池前预注册 CancelFlag。
        """
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return None, None
        if not (isinstance(msg, dict) and msg.get("method") == "tools/call"
                and msg.get("id") is not None):
            return None, None
        params = msg.get("params")
        name = str(params.get("name", "")) if isinstance(params, dict) else ""
        prefix = self.cfg.get("tool_prefix", "")
        short = name[len(prefix):] if prefix and name.startswith(prefix) else name
        route = "control" if short in ("status", "job") else "work"
        return route, msg.get("id")

    def _handle_and_send(self, line: str, _flag: CancelFlag | None = None,
                         request_id=None) -> None:
        # _flag 由 reader 预注册；具体取消检查在 _tools_call/_run_as_job 中
        # 通过 request_id 从 self._inflight 复用，这里不直接使用。
        try:
            response = self.dispatcher.handle_message(line)
        except Exception as exc:              # 兜底：工作线程异常不能静默
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32603, "message": str(exc)[:300]}}
        # P2（task-8）：此前只在错误响应时清理预注册的 CancelFlag；status/job
        # 的成功响应走控制池直通，导致 _inflight 永久残留（job 轮询按次数增长）。
        # 仅当条目仍是当初预注册的 flag 时清理（Job 接管后身份不同，不能误删）。
        if request_id is not None and isinstance(response, dict):
            key = str(request_id)
            with self._lock:
                if self._inflight.get(key) is _flag:
                    self._inflight.pop(key, None)
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
