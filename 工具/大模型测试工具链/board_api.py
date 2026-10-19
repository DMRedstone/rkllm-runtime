# -*- coding: utf-8 -*-
"""板端推理服务的共享 HTTP 客户端 —— 工具链与连接器共用的唯一实现。

为什么有这一个文件
------------------
改造前，同一件事被写了七遍：`01` 的 `request()`、`02` 的 `Board`、`03` 的
`Client`、`04` 的 `req()`、`08` 的 `post_chat()`、`00` 的 `wait_ready()`，
以及 `02` E2 段的内联 urlopen。签名与判读口径还互不一致：p90 一处用线性
插值、一处取索引；vLLM 的空 200 一处判"可达"、一处判"不可用"；认证只支持
`Authorization: Bearer`。同一个服务在两份报告里能得出不同结论。

现在所有脚本都 import 本模块。**脚本仍可 `python llm_stress.py` 单独
运行**——本文件与它们同目录。

双份副本
--------
`本地模型连接器/board_api.py` 是本文件的副本：连接器要能整目录拷到任意
机器（含 Windows/macOS），不能跨目录 import。`run_all.py` 的预检会比对
两份是否一致，所以改这里之后必须同步副本。

设计约定
--------
* 纯标准库，Python 3.8+；
* 一切判读只在这里做：2xx 但非 JSON 的 `/health` 视为**可达**（vLLM 就
  这么回），只有连不上或非 2xx 才算不可用；
* `ask()` 在正文为空且没有任何 token 计数时报错，而不是静默返回空串；
* `model` 只在调用方显式给出时才写进请求体（rkllm 忽略它，vLLM 用它路由，
  硬编码一个 RK3588 的模型名会让 vLLM 直接 404）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

VERSION = "1.0"

DEFAULT_TIMEOUT = 240.0
HEALTH_TIMEOUT = 10.0

# 只读 GET 时带上的头；POST 另外加 Content-Type
_BASE_HEADERS = {"Accept": "application/json"}


# ---------------------------------------------------------------------------
# URL 与认证：所有脚本都必须走这里，避免"一处保留 scheme、一处丢掉"的老问题
# ---------------------------------------------------------------------------

def split_base(raw):
    """把 `https://host:port/base` 拆成 (scheme, host_and_base)。

    协议必须显式保留：花生壳等内网穿透只通 HTTPS，丢掉 scheme 会退化成明文
    http，表现为 RemoteDisconnected。
    """
    value = str(raw or "").strip().rstrip("/")
    low = value.lower()
    if low.startswith("https://"):
        return "https", value[8:]
    if low.startswith("http://"):
        return "http", value[7:]
    return "http", value


def build_base(host, scheme=None):
    """`host` 允许自带协议前缀；显式给了 `scheme` 时以前者为准。"""
    auto_scheme, rest = split_base(host)
    return (scheme or auto_scheme) + "://" + rest


def join_url(base, path, base_path=""):
    """拼出完整 URL。

    `base_path` 用于只提供子路径的部署（内网穿透常在路径上加前缀
    `https://x.example/rkllm`，网关可能是 `/api`）。已有的 `/v1` 前缀不做
    去重——调用方明确给什么就发什么，猜错反而不如报 404 好查。
    """
    prefix = "/" + base_path.strip("/") if base_path and base_path.strip("/") else ""
    suffix = path if path.startswith("/") else "/" + path
    return base.rstrip("/") + prefix + suffix


def auth_headers(key, auth_header="Authorization", auth_prefix="Bearer "):
    """构造认证头。

    除了 OpenAI 的 `Authorization: Bearer`，还需要覆盖 Azure 的 `api-key`、
    Anthropic 兼容网关的 `x-api-key`，以及不需要前缀的裸值。
    """
    if not key:
        return {}
    return {auth_header: (auth_prefix or "") + key}


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

class Board(object):
    """一个目标的 HTTP 客户端。

    接口与改造前 `02_llm_stress.Board` 保持一致（`call` / `ask` / `stream` /
    `health` / `alive` / `last_health_problem` / `_request`），这样各脚本的
    调用点不用改签名，只换 import。
    """

    def __init__(self, host, key="", timeout=DEFAULT_TIMEOUT, scheme=None,
                 base_path="", auth_header="Authorization", auth_prefix="Bearer ",
                 chat_path="/v1/chat/completions", health_path="/health"):
        self.base = build_base(host, scheme)
        self.key = key
        self.timeout = float(timeout)
        self.base_path = base_path
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        self.chat_path = chat_path
        self.health_path = health_path
        self.last_health_problem = None

    # -- 底层 ---------------------------------------------------------

    def _headers(self, stream=False, extra=None):
        headers = dict(_BASE_HEADERS)
        if stream:
            headers["Accept"] = "text/event-stream"
        headers.update(auth_headers(self.key, self.auth_header, self.auth_prefix))
        if extra:
            headers.update(extra)
        return headers

    def _request(self, path, payload=None, method="POST", timeout=None):
        """构造 Request 对象（不发送）。保留给畸形请求体用例使用。"""
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
        elif payload is None:
            data = None
        else:
            data = json.dumps(payload).encode("utf-8")
        headers = self._headers()
        if data is not None:
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(
            join_url(self.base, path, self.base_path),
            data=data, method=method, headers=headers)

    def _open(self, path, payload=None, method="POST", timeout=None, stream=False):
        return urllib.request.urlopen(
            self._request(path, payload, method, timeout),
            timeout=timeout or self.timeout)

    # -- 高层 ---------------------------------------------------------

    def call(self, path=None, payload=None, method="POST", timeout=None):
        """发一次请求并读完响应 → (status, body_text, elapsed, usage|None)。

        status 为 HTTP 码，或字符串 "EXC" 表示客户端异常（连不上/超时）。
        """
        path = path or self.chat_path
        request = self._request(path, payload, method, timeout)
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                body = response.read().decode("utf-8", "replace")
                usage = None
                try:
                    usage = json.loads(body).get("usage")
                except Exception:                                   # noqa: BLE001
                    pass
                return response.status, body, time.time() - started, usage
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace"), \
                time.time() - started, None
        except Exception as exc:                                    # noqa: BLE001
            return "EXC", repr(exc), time.time() - started, None

    def ask(self, content, max_tokens=32, timeout=None, extra=None, model=None):
        """单轮对话 → (status, text, elapsed, usage)。

        与旧 `Board.ask` 同签名。`model` 只在显式给出时写进请求体；不传就
        让服务端用自己的默认——把某个 RK3588 量化模型名硬编码进去，接 vLLM
        时会 404。
        """
        payload = {"messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens}
        if model:
            payload["model"] = model
        if extra:
            payload.update(extra)
        status, body, elapsed, usage = self.call(payload=payload, timeout=timeout)
        if status == 200:
            try:
                return status, json.loads(body)["choices"][0]["message"]["content"], elapsed, usage
            except Exception:                                       # noqa: BLE001
                return "BAD", body[:80], elapsed, usage
        if status in ("EXC", "BAD"):
            return status, body[:200], elapsed, usage
        return status, body[:200], elapsed, usage

    def stream(self, payload, timeout=None):
        """SSE 请求 → (status, text, elapsed, ttft)。

        ttft 只算到第一个**有内容**的增量，跳过服务端先发的纯 role chunk。
        """
        payload = dict(payload, stream=True)
        request = self._request(self.chat_path, payload, timeout=timeout)
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                ttft, pieces = None, []
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == b"[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    except Exception:                               # noqa: BLE001
                        continue
                    if delta:
                        if ttft is None:
                            ttft = time.time() - started
                        pieces.append(delta)
                return response.status, "".join(pieces), time.time() - started, ttft
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace"), \
                time.time() - started, None
        except Exception as exc:                                    # noqa: BLE001
            return "EXC", repr(exc), time.time() - started, None

    def health(self, timeout=None):
        """读健康端点 → dict 或 None（失败原因在 last_health_problem）。

        2xx 但响应体不是 JSON 时返回 `{"status": "ok", ...}`：vLLM 的
        `/health` 就是 200 + 空体，把它判成"服务不可用"会让连接器误报。
        """
        timeout = timeout or HEALTH_TIMEOUT
        try:
            with self._open(self.health_path, method="GET", timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
                status = response.status
        except urllib.error.HTTPError as exc:
            self.last_health_problem = "HTTP %s" % exc.code
            return None
        except Exception as exc:                                    # noqa: BLE001
            self.last_health_problem = repr(exc)[:160]
            return None
        self.last_health_problem = None
        if not raw.strip():
            return {"status": "ok", "detail": "empty 200 body (treated as reachable)"}
        try:
            info = json.loads(raw)
        except Exception:                                           # noqa: BLE001
            return {"status": "ok", "detail": raw[:60]}
        if not isinstance(info, dict):
            return {"status": "ok", "detail": "non-object health body"}
        info.setdefault("status", "ok")
        return info

    def alive(self):
        """存活探测 → (是否存活, health dict 或 None)。"""
        info = self.health()
        return info is not None, info


# ---------------------------------------------------------------------------
# 模块级工具：各脚本的旧函数都收敛到这里
# ---------------------------------------------------------------------------

def request(base, key, path, payload=None, method="POST", timeout=10,
            send_key=True, auth_header="Authorization", auth_prefix="Bearer ",
            base_path=""):
    """一次请求 → (status, body_text, elapsed, error_repr)。

    与 `01_health_check.request()` 同签名；`send_key=False` 用于验证鉴权。
    """
    url = join_url(base, path, base_path)
    data = None
    if payload is not None:
        data = payload if isinstance(payload, (bytes, bytearray)) else \
            json.dumps(payload).encode("utf-8")
    headers = dict(_BASE_HEADERS)
    if data is not None:
        headers["Content-Type"] = "application/json"
    if send_key and key:
        headers.update(auth_headers(key, auth_header, auth_prefix))
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace"), \
                time.time() - started, None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), time.time() - started, None
    except Exception as exc:                                        # noqa: BLE001
        return "EXC", "", time.time() - started, repr(exc)


def req(base, key, payload=None, timeout=120, raw_method="POST",
        path="/v1/chat/completions"):
    """`04_crash_probe` 的三元组形态 → (status, elapsed, error)。"""
    status, _body, elapsed, err = request(base, key, path, payload, raw_method, timeout)
    return status, elapsed, err


def post_chat(base, key, payload, timeout=600, base_path=""):
    """`08_llm_realworld` 的 dict 形态（含 usage 与错误体）。"""
    status, body, elapsed, _usage = request(
        base, key, "/v1/chat/completions", payload, "POST", timeout, base_path=base_path)
    if status == "EXC":
        return {"ok": False, "dt": elapsed, "error": body or "connection error", "text": ""}
    if status != 200:
        return {"ok": False, "dt": elapsed, "error": "HTTP %d" % status,
                "body": body[:200], "text": ""}
    try:
        data = json.loads(body)
    except Exception as exc:                                        # noqa: BLE001
        return {"ok": False, "dt": elapsed, "error": "bad JSON: %r" % exc,
                "body": body[:200], "text": ""}
    usage = data.get("usage") or {}
    choice = (data.get("choices") or [{}])[0]
    return {"ok": True, "dt": elapsed,
            "text": (choice.get("message") or {}).get("content", "") or "",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason")}


def answer_of(body):
    """从 chat/completions 响应里取回答文本；解析失败返回 None。"""
    try:
        return json.loads(body)["choices"][0]["message"]["content"]
    except Exception:                                               # noqa: BLE001
        return None


def health_ok(base, key, timeout=5, base_path=""):
    """健康端点是否 2xx（不解析正文，vLLM 的空 200 也算）。"""
    status, _body, _elapsed, _err = request(
        base, key, "/health", None, "GET", timeout, base_path=base_path)
    return status == 200


def wait_ready(base, key, timeout=90, interval=1.0, base_path=""):
    """等到健康端点返回 2xx，返回等待秒数；超时返回 None。"""
    started = time.time()
    while time.time() - started < timeout:
        if health_ok(base, key, base_path=base_path):
            return time.time() - started
        time.sleep(interval)
    return None


def pct(values, p):
    """线性插值百分位（所有报告统一用这一个口径）。"""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def newest(pattern):
    """匹配 pattern 的最新文件（文件名含 `%Y%m%d_%H%M%S` 时间戳）。"""
    import glob
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None
