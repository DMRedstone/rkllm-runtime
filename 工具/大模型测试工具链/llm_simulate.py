#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实聊天客户端的「行为仿真」测试脚本。

不同于压力测试（只管灌请求量），本脚本复刻一个桌面聊天客户端（OpenWebUI /
WorkBuddy 风格）真实的使用模式，用来暴露压测发现不了的问题：

  1. 握手      —— 拉模型列表、健康探测、连通性探测、空输入误发
  2. 对话      —— 多轮对话，且每轮把完整历史回传，轮间带人类「思考停顿」
  3. 长文档    —— 用户逐级粘贴越来越长的文档（板端最容易出事的场景）
  4. 中断      —— 用户流式读到一半关闭窗口
  5. 切换会话  —— 重置上下文后开始新对话
  6. 后台标题  —— 用户还在打字时并发发起的「生成会话标题」请求（并发场景）
  7. 混合负载  —— 多个虚拟用户带 think-time 混发不同类型请求

每一步之后都做存活探测，因此一旦服务崩溃，可以精确定位到是哪一次请求导致的。

相对原版 tools/simulate_client.py 的改进
------------------------------------
1. 模型 id 从 /v1/models 动态读取，不再硬编码（换成任何部署都能跑）；
2. 新增第 7 段「多虚拟用户混合负载」，模拟小团队的并发使用；
3. 并发倾向的场景（后台标题、混合负载）统一放到最后，避免一旦崩溃就
   掩盖前面单流场景的结果；
4. 全部步骤记录到 JSON 时间线，供报告生成器消费。

用法
----
    python llm_simulate.py --host 192.0.2.10:8080 --key KEY
    python llm_simulate.py --host 192.0.2.10:8080 --key KEY --vu 3 --vu-seconds 45
    python llm_simulate.py --host 192.0.2.10:8080 --key KEY --skip-concurrent --skip-mixed
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

from board_api import auth_headers, build_base, join_url   # 共享 HTTP 层

# 强制 UTF-8 输出，避免 Windows 控制台中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 全局存活状态机：任一探测失败即置 alive=False 并记录死亡位置
STATE = {"alive": True, "died_at": None, "requests": 0}
# 逐步时间线，供报告渲染
TIMELINE = []


def record(scenario, label, ok, detail="", elapsed=None):
    """记录一条步骤结果到 TIMELINE。

    scenario -- 场景名
    label    -- 步骤名
    ok       -- 是否成功
    detail   -- 说明文本
    elapsed  -- 该步骤耗时（秒），可为 None
    """
    TIMELINE.append({"scenario": scenario, "label": label, "ok": ok,
                     "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
                     "detail": detail})


class Client(object):
    """模拟聊天客户端：维护会话历史，每轮把 system + 历史 + 新问题一起回传。"""

    def __init__(self, host, key, timeout=240, system=None):
        """
        host    -- "ip:port"
        key     -- Bearer 令牌
        timeout -- 默认超时（秒）
        system  -- system 提示词；缺省为一段本地助手设定
        """
        self.base = build_base(host)
        self.key = key
        self.timeout = timeout
        self.history = []
        self.model = None          # 握手时从 /v1/models 动态获取
        self.system = system or ("You are a helpful assistant running locally on a "
                                 "Rockchip NPU. Answer concisely.")

    # -- 底层管道 ---------------------------------------------------------
    def _open(self, path, payload=None, method="POST", timeout=None, stream=False):
        """构造并打开一个请求（返回 urlopen 的上下文管理器）。stream=True 时声明 SSE。"""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"
        headers.update(auth_headers(self.key))
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(join_url(self.base, path), data=data,
                                         method=method, headers=headers)
        return urllib.request.urlopen(request, timeout=timeout or self.timeout)

    def health(self):
        """读取 /health 并返回解析结果。"""
        with self._open("/health", method="GET", timeout=10) as response:
            return json.loads(response.read().decode())

    def alive(self, label):
        """存活探测：失败时更新状态机并返回 (False, None)。"""
        try:
            return True, self.health()
        except Exception:  # noqa: BLE001
            if STATE["alive"]:
                STATE["alive"] = False
                STATE["died_at"] = label
            return False, None

    def check(self, scenario, label):
        """带打印与记录的存活探测，返回是否存活。"""
        ok, info = self.alive(label)
        if ok:
            print("      service alive (requests=%s errors=%s)"
                  % (info["stats"]["requests"], info["stats"]["errors"]))
            record(scenario, "alive@" + label, True, "requests=%s" % info["stats"]["requests"])
        else:
            print("      *** SERVICE DOWN at: %s" % label)
            record(scenario, "alive@" + label, False, "service down")
        return ok

    def list_models(self):
        """GET /v1/models，返回模型 id 列表。"""
        with self._open("/v1/models", method="GET") as response:
            data = json.loads(response.read().decode())
        return [entry["id"] for entry in data.get("data", [])]

    def chat(self, text, max_tokens=128, stream=False, keep=True):
        """发一轮对话，返回回答文本。

        keep=True 时会把本轮问答追加进 history（模拟客户端保留上下文）。
        """
        messages = [{"role": "system", "content": self.system}]
        if keep:
            messages.extend(self.history)
        messages.append({"role": "user", "content": text})
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": 0.7, "stream": stream}
        if self.model:
            payload["model"] = self.model
        if not stream:
            with self._open("/v1/chat/completions", payload) as response:
                data = json.loads(response.read().decode())
            answer = data["choices"][0]["message"]["content"]
        else:
            answer = self._read_stream(payload)
        if keep:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": answer})
        return answer

    def _read_stream(self, payload, stop_after=None):
        """读取 SSE 流，拼接正文。stop_after 指定读到 N 个片段后主动断开（模拟用户关闭窗口）。"""
        pieces = []
        with self._open("/v1/chat/completions", payload, stream=True) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == b"[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                except Exception:  # noqa: BLE001
                    continue
                if delta:
                    pieces.append(delta)
                if stop_after is not None and len(pieces) >= stop_after:
                    break
        return "".join(pieces)

    def title_request(self, text):
        """模拟客户端后台「生成会话标题」请求（短输出、带专用 system 提示）。"""
        system = ("Generate a concise, sentence-case title (3-7 words) that captures "
                  "the main topic or goal of this session. Respond with EXACTLY one "
                  'JSON object and nothing else: {"isNewTopic": boolean, "title": string}')
        payload = {"messages": [{"role": "system", "content": system},
                                {"role": "user", "content": text}], "max_tokens": 48}
        if self.model:
            payload["model"] = self.model
        with self._open("/v1/chat/completions", payload, timeout=180) as response:
            return json.loads(response.read().decode())["choices"][0]["message"]["content"]

    def pause(self, low=0.6, high=2.2):
        """模拟真人阅读回答的停顿（随机区间）。"""
        time.sleep(random.uniform(low, high))


def step(number, text):
    """打印场景小标题。"""
    print("\n[%s] %s" % (number, text))


# ---------------------------------------------------------------------------
# 场景实现
# ---------------------------------------------------------------------------

def handshake(client):
    """场景 1：客户端首次配置流程（拉模型 / 健康探测 / 连通性 / 空输入误发）。"""
    step("1", "客户端首次配置：拉模型列表 / 健康探测 / 连通性测试 / 空输入误发")
    try:
        models = client.list_models()
        client.model = models[0] if models else None
        print("      GET /v1/models -> %s" % models)
        record("handshake", "list_models", bool(models), str(models))
    except Exception as exc:  # noqa: BLE001
        print("      GET /v1/models failed: %r" % exc)
        record("handshake", "list_models", False, repr(exc))
    if not client.check("handshake", "handshake"):
        return False

    # 仅 system 的探测请求：部分老版本会错误地返回 400
    try:
        with client._open("/v1/chat/completions",
                          {"messages": [{"role": "system", "content": "ping"}],
                           "max_tokens": 8}) as response:
            print("      连通性探测（仅 system）-> HTTP %s" % response.status)
            record("handshake", "system_only_probe", True, "HTTP %s" % response.status)
    except urllib.error.HTTPError as exc:
        print("      连通性探测（仅 system）-> HTTP %s  %s"
              % (exc.code, exc.read().decode()[:70]))
        record("handshake", "system_only_probe", False, "HTTP %s" % exc.code)
    if not client.check("handshake", "connectivity probe"):
        return False

    # 用户在空输入框直接点发送（content=None）
    try:
        with client._open("/v1/chat/completions",
                          {"messages": [{"role": "user", "content": None}],
                           "max_tokens": 8}) as response:
            print("      空输入误发 -> HTTP %s" % response.status)
            record("handshake", "empty_send", True, "HTTP %s" % response.status)
    except urllib.error.HTTPError as exc:
        print("      空输入误发 -> HTTP %s  %s" % (exc.code, exc.read().decode()[:70]))
        record("handshake", "empty_send", True, "HTTP %s (rejected cleanly)" % exc.code)
    return client.check("handshake", "empty send")


def conversation(client):
    """场景 2：多轮真实对话（每轮回传完整历史，轮间有停顿）。"""
    step("2", "真实对话：多轮 + 完整历史 + 人类节奏间隔")
    turns = ["你好，请简单介绍一下你自己。",
             "刚才你说的是什么模型？",
             "帮我把「今天天气不错」翻译成英文。",
             "再把它翻译成日文。"]
    for index, question in enumerate(turns, 1):
        started = time.time()
        try:
            answer = client.chat(question, max_tokens=96)
            elapsed = time.time() - started
            print("      第 %d 轮 (%.1fs, 历史 %d 条): %s"
                  % (index, elapsed, len(client.history), answer[:50].replace("\n", " ")))
            record("conversation", "turn %d" % index, True, answer[:60], elapsed)
        except Exception as exc:  # noqa: BLE001
            print("      第 %d 轮异常: %r" % (index, exc))
            record("conversation", "turn %d" % index, False, repr(exc))
        if not client.check("conversation", "conversation turn %d" % index):
            return False
        client.pause()
    return True


def background_title(client):
    """场景 6：后台标题请求与用户对话并发（客户端真实行为，也是并发压力点）。"""
    step("3", "客户端后台任务：用户还在打字时并发发起的『生成会话标题』")
    result = {}

    def worker():
        """后台线程：发起标题生成请求。"""
        try:
            result["title"] = client.title_request("帮我写一个 Python 快速排序")
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.4)              # 用户仍在打字，制造两请求重叠
    started = time.time()
    try:
        answer = client.chat("顺便问一下，Python 的列表和元组有什么区别？", max_tokens=96)
        print("      对话回答: %s" % answer[:50].replace("\n", " "))
        record("background", "chat_while_title", True, answer[:60], time.time() - started)
    except Exception as exc:  # noqa: BLE001
        print("      对话请求异常: %r" % exc)
        record("background", "chat_while_title", False, repr(exc))
    thread.join()
    print("      标题生成: %s" % (result.get("title", result.get("error", "?"))[:90]
                                .replace("\n", " ")))
    record("background", "title_request", "title" in result,
           (result.get("title") or result.get("error") or "")[:80])
    return client.check("background", "concurrent title generation")


def _bisect_reject(client, lo, hi):
    """在 (lo 通过, hi 被拒) 之间二分查找确切拒绝边界（字符数），返回边界值或 None。"""
    for _ in range(6):
        if hi - lo <= 256:
            return hi
        mid = (lo + hi) // 2
        document = mid * "a"
        try:
            client.chat("请用一句话概括这段文档：\n" + document, max_tokens=16, keep=False)
            lo = mid
        except urllib.error.HTTPError:
            hi = mid
        except Exception:  # noqa: BLE001
            return None
    return hi


def long_paste(client, sizes):
    """场景 3：用户逐级粘贴长文档并请求摘要（板端最易被拒/被压垮的场景）。

    返回被拒绝的 (尺寸, 状态码) 列表；若服务崩溃则返回空列表。
    若相邻档位跨越了拒绝线（前通过后被拒），自动二分查找确切拒绝边界。
    """
    step("4", "用户粘贴长文档（真实客户端最容易出事的场景）")
    paragraph = ("在嵌入式设备上部署大语言模型需要考虑内存占用、NPU 算力、量化精度"
                 "以及散热等多方面因素。本设备为 OrangePi 5 Plus，采用 RK3588 芯片，"
                 "配备 8GB 内存与三核 NPU。 ")
    rejected = []
    for size in sizes:
        document = (paragraph * (size // len(paragraph) + 1))[:size]
        print("      粘贴 %d 字符的文档，请求摘要…" % size)
        started = time.time()
        try:
            answer = client.chat("请用一句话概括这段文档：\n" + document, max_tokens=64,
                                 keep=False)
            elapsed = time.time() - started
            print("        -> %.1fs  %s" % (elapsed, answer[:45].replace("\n", " ")))
            record("paste", "%d chars" % size, True, answer[:50], elapsed)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:90].replace("\n", " ")
            print("        -> HTTP %s  %s" % (exc.code, detail))
            record("paste", "%d chars" % size, False, "HTTP %s %s" % (exc.code, detail),
                   time.time() - started)
            if not client.check("paste", "long paste %d chars" % size):
                return []                     # 服务崩溃：停止后续
            rejected.append((size, exc.code))  # 服务存活：请求级失败，继续尝试更短文档
            # F-3：档距可能跨过拒绝线——用已知的通过/拒绝邻点二分出确切边界
            _passed = [s for s in sizes if s < size and s not in [r[0] for r in rejected]]
            if _passed:
                _lo = max(_passed)
                boundary = _bisect_reject(client, _lo, size)
                if boundary:
                    print("        -> 二分拒绝边界 ≈ %d 字符" % boundary)
                    record("paste", "reject_boundary_chars", True, str(boundary))
            continue
        except Exception as exc:  # noqa: BLE001
            print("        -> 请求异常: %r" % exc)
            record("paste", "%d chars" % size, False, repr(exc), time.time() - started)
            if not client.check("paste", "long paste %d chars" % size):
                return []
            rejected.append((size, "exception"))
            continue
        if not client.check("paste", "long paste %d chars" % size):
            return []
        client.pause(0.4, 1.0)
    return rejected


def interruption(client):
    """场景 4：流式读到一半主动断开（模拟用户关闭窗口），随后服务应仍可用。"""
    step("5", "用户中途关闭窗口（流式读到一半断开）")
    try:
        partial = client._read_stream(
            {"messages": [{"role": "user", "content": "请写一篇 500 字的杭州游记。"}],
             "max_tokens": 512, "stream": True}, stop_after=3)
        print("      读到 %d 个片段后主动断开" % len(partial))
        record("interruption", "abort_mid_stream", True, "%d chunks" % len(partial))
    except Exception as exc:  # noqa: BLE001
        print("      流式异常: %r" % exc)
        record("interruption", "abort_mid_stream", False, repr(exc))
    # B-2：中断后立刻轮询 /health 的 busy，量化「服务端是否仍在空跑推理」（全库报告 P1-4）
    try:
        _t0 = time.time()
        _busy_tail = None
        while time.time() - _t0 < 180:
            _h = client.health() if hasattr(client, "health") else None
            _busy = bool(_h.get("busy")) if isinstance(_h, dict) else None
            if _busy is False:
                _busy_tail = time.time() - _t0
                break
            if _busy is None:
                break
            time.sleep(1.0)
        if _busy_tail is not None:
            print("      服务端 busy 恢复 False 用时 %.1fs（中断后仍在推理的时长）" % _busy_tail)
            record("interruption", "server_busy_tail_s", True, "%.1fs" % _busy_tail)
        else:
            record("interruption", "server_busy_tail_s", False, "180s 内未观察到 busy=False")
    except Exception as exc:  # noqa: BLE001
        print("      busy 轮询异常: %r" % exc)
    time.sleep(1.5)
    return client.check("interruption", "after interruption")


def session_switch(client):
    """场景 5：清空上下文并开始新会话。"""
    step("6", "切换会话：清空上下文后开始新对话")
    try:
        with client._open("/v1/reset", {"messages": []}) as response:
            print("      POST /v1/reset -> HTTP %s" % response.status)
            record("session", "reset", True, "HTTP %s" % response.status)
    except Exception as exc:  # noqa: BLE001
        print("      reset 异常: %r" % exc)
        record("session", "reset", False, repr(exc))
    client.history = []
    try:
        answer = client.chat("换个话题：推荐三个适合秋天去的地方。", max_tokens=96)
        print("      新会话回答: %s" % answer[:50].replace("\n", " "))
        record("session", "fresh_chat", True, answer[:60])
    except Exception as exc:  # noqa: BLE001
        record("session", "fresh_chat", False, repr(exc))
    return client.check("session", "session switch")


def mixed_workload(client, vu, seconds):
    """场景 7：多虚拟用户带 think-time 的混合负载（短问答 / 长摘要 / 标题生成）。"""
    step("7", "多虚拟用户混合负载（%d 个用户，持续约 %ds）" % (vu, seconds))
    stop_at = time.time() + seconds
    lock = threading.Lock()
    rows = []
    prompt = ("围绕嵌入式 NPU 部署大模型展开一段说明，包含内存、算力、量化、散热四方面。"
              * 6)

    def user_loop(uid):
        """单个虚拟用户循环：按概率选择请求类型，发完 sleep 一段 think-time。"""
        local = Client(client.base.replace("http://", ""), client.key, system=client.system)
        local.model = client.model
        while time.time() < stop_at:
            r = random.random()
            started = time.time()
            try:
                if r < 0.6:
                    ans = local.chat("用户%d：用一句话介绍杭州。" % uid, max_tokens=64, keep=False)
                    kind = "short_chat"
                elif r < 0.85:
                    ans = local.chat("请概括：\n" + prompt, max_tokens=64, keep=False)
                    kind = "summarise"
                else:
                    ans = local.title_request("用户%d 的会话" % uid)
                    kind = "title"
                ok = bool(ans)
                detail = str(ans)[:40].replace("\n", " ")
            except Exception as exc:  # noqa: BLE001
                kind, ok, detail = "error", False, repr(exc)[:60]
            with lock:
                rows.append({"kind": kind, "ok": ok, "elapsed_s": round(time.time() - started, 3),
                             "detail": detail})
            time.sleep(random.uniform(0.3, 1.5))     # think-time

    threads = [threading.Thread(target=user_loop, args=(i + 1,)) for i in range(vu)]
    started = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - started

    ok_rows = [r for r in rows if r["ok"]]
    lat = [r["elapsed_s"] for r in ok_rows]
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], {"n": 0, "ok": 0})
        by_kind[r["kind"]]["n"] += 1
        if r["ok"]:
            by_kind[r["kind"]]["ok"] += 1
    print("      完成 %d 个请求，成功 %d，用时 %.1fs" % (len(rows), len(ok_rows), wall))
    print("      分类: %s" % json.dumps(by_kind, ensure_ascii=False))
    if lat:
        print("      延迟 p50 %.2fs  p90 %.2fs  max %.2fs"
              % (statistics.median(lat), sorted(lat)[int(len(lat) * 0.9) - 1] if len(lat) > 1 else lat[0],
                 max(lat)))
    record("mixed", "vu=%d" % vu, len(ok_rows) == len(rows),
           "n=%d ok=%d wall=%.1fs" % (len(rows), len(ok_rows), wall), wall)
    for r in rows:
        TIMELINE.append({"scenario": "mixed", "label": r["kind"], "ok": r["ok"],
                         "elapsed_s": r["elapsed_s"], "detail": r["detail"]})
    return client.check("mixed", "after mixed workload")


# ---------------------------------------------------------------------------

def main():
    """命令行入口：按顺序跑各场景，记录时间线并落盘。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("RKLLM_HOST", "192.0.2.10:8080"),
                        help="host:port（可用 RKLLM_HOST 覆盖）")
    parser.add_argument("--key", default=os.environ.get("RKLLM_KEY") or os.environ.get("RKLLM_API_KEY", ""),
                        help="API key（可用 RKLLM_KEY 覆盖）")
    parser.add_argument("--paste-sizes", default="800,3000,8000,16000",
                        help="长文档场景的字符数，逗号分隔")
    parser.add_argument("--vu", type=int, default=3, help="混合负载的虚拟用户数")
    parser.add_argument("--vu-seconds", type=int, default=45, help="混合负载持续时间")
    parser.add_argument("--skip-paste", action="store_true", help="跳过长文档场景")
    parser.add_argument("--skip-mixed", action="store_true", help="跳过混合负载场景")
    parser.add_argument("--skip-concurrent", action="store_true",
                        help="跳过后台标题（并发）场景")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "results"),
                        help="结果 JSON 输出目录")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sizes = [int(part) for part in args.paste_sizes.split(",") if part.strip()]
    client = Client(args.host, args.key)
    random.seed(7)               # 固定随机种子，保证 think-time 可复现

    print("simulation target : http://%s" % args.host)
    ok, info = client.alive("startup")
    if not ok:
        print("服务不可达")
        return 1
    print("model             : %s" % info["model"])
    print("start state       : uptime %.0fs requests=%s errors=%s"
          % (info["uptime_s"], info["stats"]["requests"], info["stats"]["errors"]))

    # 并发倾向的场景放最后：一旦崩溃不至于掩盖前面单流场景的结果
    scenarios = [("握手", handshake), ("对话", conversation)]
    if not args.skip_paste:
        scenarios.append(("长文档", lambda c: long_paste(c, sizes)))
    scenarios.append(("中断", interruption))
    scenarios.append(("切换会话", session_switch))
    if not args.skip_concurrent:
        scenarios.append(("后台标题", background_title))
    if not args.skip_mixed:
        scenarios.append(("混合负载", lambda c: mixed_workload(c, args.vu, args.vu_seconds)))

    paste_failures = []
    started_at = time.time()
    for name, function in scenarios:
        result = function(client)
        if result is False:               # 服务不再应答
            print("\n=== 仿真在『%s』阶段终止" % name)
            if STATE["alive"]:
                print("    服务仍然存活：失败发生在请求层面，不是进程崩溃。")
            else:
                print("    服务在 %s 之后失去响应（进程崩溃）。" % STATE["died_at"])
            break
        if isinstance(result, list) and result:
            paste_failures.extend(result)

    print("\n=== 仿真结束")
    if STATE["alive"]:
        print("    服务全程存活")
    if paste_failures:
        print("    以下请求被拒绝（服务存活，属请求层面的失败）：")
        for size, code in paste_failures:
            print("      %d 字符的文档 -> HTTP %s" % (size, code))
    ok, info = client.alive("final")
    if info:
        print("    最终 requests=%s rejected=%s errors=%s"
              % (info["stats"]["requests"], info["stats"]["rejected"],
                 info["stats"]["errors"]))

    final = {
        "target": build_base(args.host),
        "model": info["model"] if info else None,
        "alive_at_end": STATE["alive"],
        "died_at": STATE["died_at"],
        "elapsed_s": round(time.time() - started_at, 1),
        "paste_failures": paste_failures,
        "final_stats": info["stats"] if info else None,
        "timeline": TIMELINE,
    }
    out = os.path.join(args.outdir, "simulate_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(final, fh, ensure_ascii=False, indent=2)
    print("\nresults written to %s" % out)
    return 0 if STATE["alive"] else 1


if __name__ == "__main__":
    sys.exit(main())
