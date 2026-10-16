# -*- coding: utf-8 -*-
"""本地板端小模型（rkllm serve）的调用封装 —— 自包含 · 多目标 · 可选复用主项目工具链。

**独立性**：纯标准库（`urllib` / `json` / `threading` / `importlib`），不依赖任何项目目录、
不依赖第三方包。把 `local-rkllm/` 整个目录拷到任意能访问板端服务的机器即可工作，
Windows / Linux / macOS 均可。

**多目标**：支持同时配置**多个本地部署**（不同地址 / 令牌 / 模型），用目标名切换。
配置是本目录下的**纯文本** `targets.conf`（参考 `targets.example.conf`），随目录一起移植：

    # 一行一个目标：[名称 =] 地址 [密钥] [备注…] [model=模型名]
    board = 192.0.2.10:8080 <token>            # 只写地址和密钥就行
    192.0.2.50:8080 <token>                 # 名称省略时用地址作名称
    board-b = 192.0.2.10:8080 <token> model=qwen2.5-1.5b   # 需要时再指定模型

**配置来源优先级**：`RKLLM_CONFIG` 指定的文件 → 本目录 `targets.conf` → 本目录
`targets.json`（旧格式，仍兼容）→ 环境变量单目标（`RKLLM_HOST` / `RKLLM_API_KEY` / `RKLLM_MODEL`）。
**代码中不含任何具体地址**，未配置时直接报错并给出配置指引。

**可选的复用**：若同机存在主项目的测试工具链脚本（`02_llm_stress.py`，内含 `Board` 类），
可用环境变量 `RKLLM_BOARD_SCRIPT` 指定以复用其 HTTP 层；未指定或加载失败时自动降级到
内置客户端。**复用是增强，不是依赖。**

**默认阈值 = RK3588 特化值**（2026-10-15 报告 + 2026-10-16 实机校准，Qwen3.5-4B / OrangePi 5 Plus 8G / rkllm serve）：
  · 单流吞吐：2026-10-16 实机 6.3–6.9 tok/s（含 prefill 的端到端；旧报告 8–9 tok/s 为长文稳态）
    → 默认取保守下界 6.3，故 768 token 输出的超时约 178s
  · 并发无吞吐增益（NPU 单请求串行），QPS 恒 0.86–0.94
  · 上下文硬上限 4096 token
  · 输出超过约 1000 token 时生成退化：重复字符流 + 串入无关英文模板
  · 枚举类任务会提前停止（实测「数到 500」只输出到 23），别指望它做长枚举
  · max_tokens 不被严格遵守，实际输出可能超出请求
  · 长 prompt（1000–2000 字）偶发崩溃，由守护进程自动拉起

因此本模块强制：输入预算预检、输出预算封顶、客户端硬截断、
超时按吞吐推算、失败自动重试（等守护进程拉起）、同一目标内串行。

**这些特化值全部可按目标覆盖**（写在目标行里，`键=值`，空格分隔）。换平台 / 换模型时，
不必改代码，改配置即可：

    board-npu = 192.0.2.10:8080 <token> tok_per_sec=2.5 max_out=2048 context=8192 timeout_floor=300

**更省事的做法：用档位 `preset=`**（一行切换；写了档位后仍可用 `键=值` 覆盖单项）：

    board-14b = 192.0.2.60:8080 <token> preset=14b               # 按参数量选档
    board-auto = 192.0.2.61:8080 <token> preset=auto model=qwen2.5-14b   # 按模型名自动选档
    board-npu  = 192.0.2.10:8080 <token> preset=default             # RK3588 特化档（不写就是它）

    尺寸档：1b / 3b / 4b / 7b / 14b / 32b（32B 为本地部署实用上限），均为**跨平台保守估算**
    default 档：上表的 RK3588 + Qwen3.5-4B 实测特化值（板端专用，与 4b 档不同）
    auto：从模型名里的参数量自动挑最接近的档（认不出则回落到 default）

单项覆盖（写在目标行里，`键=值`）：

    max_in=2600        输入预算（token）        max_out=768     输出封顶（token）
    context=4096       上下文硬上限（token）    tok_per_sec=8.5 单流吞吐，用于推算超时
    timeout_floor=60   超时下限（秒）           retries=2       失败重试次数
    retry_wait=8       重试前等待（秒）         chars_per_token=1.6   1 token ≈ 字符数
    overrun=1.56       硬截断余量（× max_out）  degen_check=on  退化特征自检 on/off
    health=rkllm       健康检查：rkllm | none | 任意路径（openai → /v1/models）

环境变量：
    RKLLM_CONFIG       可选：配置文件路径（默认本目录 targets.conf，回退旧格式 targets.json）
    RKLLM_HOST         单目标快捷方式：板端地址（不含 http://）
    RKLLM_API_KEY      单目标快捷方式：令牌
    RKLLM_MODEL        单目标快捷方式：模型名
    RKLLM_BOARD_SCRIPT 可选：主项目工具链脚本路径，用于复用其 Board 类

命令行用法：
    python rkllm_client.py targets
    python rkllm_client.py health [--target board-b]
    python rkllm_client.py calibrate [--target board-b] [--tokens 128]    # 实机校准吞吐
    python rkllm_client.py ask "把下面这段改写得更简洁：……" [--target board-b]
"""
import importlib.util
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAMES = ("targets.conf", "targets.json")   # 依次查找；.conf 为主推格式
CONFIG_PATH = os.environ.get("RKLLM_CONFIG", "").strip() or os.path.join(HERE, CONFIG_NAMES[0])


def find_config():
    """定位配置文件：RKLLM_CONFIG → 本目录 targets.conf → 本目录 targets.json（旧格式）。"""
    p = os.environ.get("RKLLM_CONFIG", "").strip()
    if p:
        return p
    for name in CONFIG_NAMES:
        cand = os.path.join(HERE, name)
        if os.path.exists(cand):
            return cand
    return os.path.join(HERE, CONFIG_NAMES[0])    # 未创建时的默认位置（用于提示）

# 单目标快捷方式（配置文件缺失时使用）。**不含任何默认地址** —— 地址必须由配置提供。
ENV_HOST = os.environ.get("RKLLM_HOST", "").strip()
ENV_KEY = os.environ.get("RKLLM_API_KEY", "").strip()
ENV_MODEL = os.environ.get("RKLLM_MODEL", "").strip()
BOARD_SCRIPT = os.environ.get("RKLLM_BOARD_SCRIPT", "").strip()
ENV_PRESET = os.environ.get("RKLLM_PRESET", "").strip().lower()   # 单目标快捷方式的档位

# ---- 默认阈值：RK3588 + Qwen3.5-4B 实测特化值（依据见模块 docstring）。
#      所有项均可按目标覆盖（配置行里写 `键=值`），改平台/改模型不必改代码。----
DEFAULTS = {
    "max_in": 2600,          # 输入预算：给输出留空间，且避开长 prompt 崩溃高发区
    "max_out": 768,          # 输出封顶：越过约 1000 token 后生成质量会塌陷
    "context": 4096,         # 上下文硬上限
    "tok_per_sec": 6.3,      # 单流实测吞吐，用于推算超时（2026-10-16 实机：6.3–6.9 tok/s）
    "timeout_floor": 60,     # 超时下限（秒）
    "retries": 2,            # 失败重试次数（等守护进程拉起服务）
    "retry_wait": 8,         # 重试前等待（秒）
    "chars_per_token": 1.6,  # 中文为主文本，1 token ≈ 1.6 字符（保守）
    "overrun": 1.56,         # 硬截断余量：服务端不遵守 max_tokens，实测超出 25%–56%
    "degen_check": True,     # 退化特征自检（重复词流）
    "health": "rkllm",       # 健康检查方式：rkllm | none | 任意 HTTP 路径
}

# 配置行里可用的键名（含中文别名）→ DEFAULTS 的键
OPT_KEYS = {
    "max_in": ("max_in", "输入上限"), "max_out": ("max_out", "输出上限"),
    "context": ("context", "上下文"), "tok_per_sec": ("tok_per_sec", "tps", "吞吐"),
    "timeout_floor": ("timeout_floor", "timeout", "超时"), "retries": ("retries", "重试"),
    "retry_wait": ("retry_wait",), "chars_per_token": ("chars_per_token", "cpt"),
    "overrun": ("overrun",), "degen_check": ("degen_check", "degen", "退化自检"),
    "health": ("health", "健康检查"), "preset": ("preset", "档位", "模式", "通用选项"),
    "ability": ("ability", "能力", "能力说明"),
}
_INT_OPTS = ("max_in", "max_out", "context", "timeout_floor", "retries", "retry_wait")
_FLOAT_OPTS = ("tok_per_sec", "chars_per_token", "overrun")
_ZERO_OK = ("retries", "retry_wait")        # 这两项允许取 0（不重试 / 不等待）
_TRUE_WORDS = ("on", "true", "yes", "1", "是", "开")

_warned = set()


def _warn(msg):
    """告警去重：同一句只提醒一次（配置每次调用都会被重读，避免刷屏）。"""
    if msg not in _warned:
        _warned.add(msg)
        sys.stderr.write(msg)

# ---- 档位（preset）：按模型参数量给出的一组阈值，免去逐项写。
#      写了档位后，行内其它 `键=值` 仍可覆盖单项；`preset=auto` 按模型名自动选档。----
PRESETS = {
    # 「RK3588 + Qwen3.5-4B」实测特化档（本文件默认档，也是不写 preset 时的行为）
    "default": dict(DEFAULTS),
    # 以下尺寸档为**跨平台保守估算**（非实测）：吞吐故意取小，只用于把超时上限放宽；
    # 实际硬件更快不会变慢，更慢则请用 tok_per_sec= 覆盖。
    "1b": {"max_in": 3000, "max_out": 512, "context": 4096, "tok_per_sec": 60,
           "timeout_floor": 60, "retries": 2, "retry_wait": 5, "chars_per_token": 1.6,
           "overrun": 1.3, "degen_check": False, "health": "rkllm"},
    "3b": {"max_in": 5500, "max_out": 1024, "context": 8192, "tok_per_sec": 40,
           "timeout_floor": 120, "retries": 2, "retry_wait": 5, "chars_per_token": 1.6,
           "overrun": 1.3, "degen_check": False, "health": "rkllm"},
    "4b": {"max_in": 6000, "max_out": 1024, "context": 8192, "tok_per_sec": 30,
           "timeout_floor": 120, "retries": 2, "retry_wait": 5, "chars_per_token": 1.6,
           "overrun": 1.3, "degen_check": False, "health": "rkllm"},
    "7b": {"max_in": 6000, "max_out": 1536, "context": 8192, "tok_per_sec": 20,
           "timeout_floor": 180, "retries": 2, "retry_wait": 5, "chars_per_token": 1.6,
           "overrun": 1.25, "degen_check": False, "health": "rkllm"},
    "14b": {"max_in": 12000, "max_out": 2048, "context": 16384, "tok_per_sec": 10,
            "timeout_floor": 240, "retries": 2, "retry_wait": 5, "chars_per_token": 1.6,
            "overrun": 1.25, "degen_check": False, "health": "rkllm"},
    "32b": {"max_in": 24000, "max_out": 3072, "context": 32768, "tok_per_sec": 5,
            "timeout_floor": 300, "retries": 1, "retry_wait": 5, "chars_per_token": 1.6,
            "overrun": 1.2, "degen_check": False, "health": "rkllm"},
}
# 各档位的**能力边界**（给调用方模型看的提示词素材）：写清能干什么、不能干什么
TIER_ABILITY = {
    "default": "板端 4B（实测档）：短文本改写 / 摘要 / 分类 / 信息抽取 / 格式转换；"
               "输出上限 768 token。**不适合**：长文生成、代码生成、多步推理、需要精确控制长度。",
    "1b": "1B 档：短句纠错、关键词抽取、极短改写；输出上限 512 token。"
          "**不适合**：摘要成段、多步推理、长文、代码、翻译长句。",
    "3b": "3B 档：短文本改写、单段摘要、简单分类；输出上限 1024 token。"
          "**不适合**：长文生成、代码生成、复杂推理。",
    "4b": "4B 档：短文本改写 / 摘要 / 分类 / 信息抽取；输出上限 1024 token。"
          "**不适合**：长文生成、代码生成、多步推理。",
    "7b": "7B 档：中等长度摘要 / 改写 / 结构化抽取 / 短代码片段；输出上限 1536 token。"
          "**不适合**：长篇创作、复杂代码工程、高精度长链推理。",
    "14b": "14B 档：较长摘要 / 改写 / 多步问答 / 中等难度代码；输出上限 2048 token。"
           "复杂长文与高要求代码仍宜交给更大的模型。",
    "32b": "32B 档：接近通用能力，可做较长文本与较复杂推理；输出上限 3072 token。"
           "仍受本地上下文与并发限制，长任务请分段。",
}


def ability_of(preset):
    """取某档位的能力说明（未知档位回落到默认档说明）。"""
    return TIER_ABILITY.get(preset, TIER_ABILITY["default"])


# 档位名（含中文 / 常见写法）→ PRESETS 键；auto 表示按模型名自动选档
PRESET_ALIASES = {
    "default": "default", "4b-rk3588": "default", "rk3588": "default", "板端": "default", "默认": "default",
    "1b": "1b", "1.5b": "1b", "2b": "1b",
    "3b": "3b", "3.8b": "3b",
    "4b": "4b", "5b": "4b", "6b": "4b",
    "7b": "7b", "8b": "7b", "9b": "7b", "generic": "7b", "通用": "7b",
    "14b": "14b", "13b": "14b", "20b": "14b",
    "27b": "32b", "30b": "32b", "32b": "32b",
    "auto": "auto", "自动": "auto",
}
# 尺寸档位阶梯（用于 auto 识别；32B 为本地部署的实用上限），单位：十亿参数
SIZE_TIERS = (("1b", 1), ("3b", 3), ("4b", 4), ("7b", 7), ("14b", 14), ("32b", 32))
_SIZE_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.I)


def detect_preset(model):
    """从模型名里猜参数量并选最接近的尺寸档（如 qwen2.5-14b → 14b）；猜不出返回 None。"""
    nums = [float(m.group(1)) for m in _SIZE_RE.finditer(model or "")]
    nums = [n for n in nums if 0.3 <= n <= 400]
    if not nums:
        return None
    n = max(nums)
    return min(SIZE_TIERS, key=lambda kv: abs(kv[1] - n))[0]

HEALTH_TIMEOUT = 10          # 健康检查探测超时（秒）
FALLBACK_MODEL = "qwen3.5-4b_w8a8_rk3588"   # 仅显示与请求体用；实际以服务端返回为准

_boards = {}                 # 目标名 -> 客户端实例
_locks = {}                  # 目标名 -> 锁（不同目标可并行，同一目标内串行）


class LocalLLMError(Exception):
    """调用板端小模型时的可读错误。"""


class _SimpleBoard(object):
    """自包含的最小 HTTP 客户端 —— 主项目工具链不可用时的兜底实现。

    与工具链 `Board` 类保持相同的调用接口（`ask` / `health` / `last_health_problem`），
    只覆盖连接器实际用到的部分，纯标准库。
    """

    def __init__(self, host, key, timeout=240, scheme="http"):
        self.base = scheme + "://" + host
        self.key = key
        self.timeout = timeout
        self.last_health_problem = None

    def _open(self, path, payload=None, method="POST", timeout=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", "Bearer " + self.key)
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def ask(self, content, max_tokens=32, timeout=None, extra=None):
        """返回 (status, content, elapsed, usage)，与工具链 Board.ask 同签名。"""
        payload = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
        if extra:
            payload.update(extra)
        started = time.time()
        try:
            with self._open("/v1/chat/completions", payload, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
                data = json.loads(body)
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
                return resp.status, text, time.time() - started, data.get("usage")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")[:120], time.time() - started, None
        except Exception as exc:  # noqa: BLE001
            return "EXC", repr(exc), time.time() - started, None

    def health(self):
        """读取 /health。成功返回 dict，失败返回 None 并记录原因。"""
        try:
            with self._open("/health", method="GET", timeout=HEALTH_TIMEOUT) as resp:
                info = json.loads(resp.read().decode("utf-8"))
            self.last_health_problem = None
            return info
        except Exception as exc:  # noqa: BLE001
            self.last_health_problem = repr(exc)[:120]
            return None


# ---------------- 目标配置 ----------------

def _normalize_host(raw):
    """接受 `192.0.2.10:8080` 或 `http://192.0.2.10:8080`，统一为 host:port。"""
    h = str(raw).strip().rstrip("/")
    for prefix in ("http://", "https://"):
        if h.startswith(prefix):
            h = h[len(prefix):]
    return h


def _split_base(raw):
    """拆出 (scheme, host)。

    `http://` / `https://` 都认；无前缀按 http。协议必须显式保留——花生壳等
    内网穿透域名往往只通 HTTPS，若在这里丢弃 scheme，下游拼 URL 时会退化成
    明文 http 导致 RemoteDisconnected。
    """
    h = str(raw).strip().rstrip("/")
    low = h.lower()
    if low.startswith("https://"):
        return "https", h[8:]
    if low.startswith("http://"):
        return "http", h[7:]
    return "http", h


_HOST_HINT = re.compile(r"[.:/]")


def _looks_like_host(tok):
    """启发式判断字段是否为地址（含 `.` `:` `/`，或为 localhost）。"""
    return tok.lower() == "localhost" or bool(_HOST_HINT.search(tok))


def _check_key(name, key):
    """密钥只允许 ASCII —— HTTP 请求头承载不了非 ASCII，否则发请求时会抛 UnicodeEncodeError。"""
    if key and not key.isascii():
        _warn("[local-rkllm] 目标 %s 的密钥含非 ASCII 字符，无法写入请求头，"
              "已按无令牌处理（请检查配置）\n" % name)
        return ""
    return key


def _coerce_opt(raw_key, raw_val, lineno):
    """把配置里的 `键=值` 转成正确类型 → (DEFAULTS 键, 值, 是否被识别为选项)。

    非法值：告警并返回 (键, None, True)——保留默认，不让一个笔误废掉整行。
    """
    key = None
    for k, aliases in OPT_KEYS.items():
        if raw_key in aliases:
            key = k
            break
    if key is None:
        return None, None, False
    raw_val = raw_val.strip()
    if key == "preset":                               # 档位：归一为档位名
        preset = PRESET_ALIASES.get(raw_val.lower())
        if preset is None:
            _warn("[local-rkllm] 配置第 %s 行未知档位 %r，用默认档位（可选：%s）\n"
                  % (lineno, raw_val, "、".join(sorted(set(PRESET_ALIASES.values())))))
            return key, None, True
        return key, preset, True
    try:
        if key in _INT_OPTS:
            val = int(raw_val)
        elif key in _FLOAT_OPTS:
            val = float(raw_val)
        elif key == "degen_check":
            val = raw_val.lower() in _TRUE_WORDS
        else:
            val = raw_val
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if key in _ZERO_OK:
                if val < 0:
                    raise ValueError("不能为负数")
            elif val <= 0:
                raise ValueError("必须为正数")
        if key == "health" and val.lower() == "openai":
            val = "/v1/models"                        # openai 兼容服务的通用探测端点
    except Exception:  # noqa: BLE001
        _warn("[local-rkllm] 配置第 %s 行 %s=%s 不是有效值，用默认 %r\n"
              % (lineno, raw_key, raw_val, DEFAULTS[key]))
        return key, None, True
    return key, val, True


def _entry(name, parts, lineno):
    """把字段列表组装成目标配置。

    位置列只有两样：**地址**（第 1 列，必填）、**密钥**（第 2 列，可选）。
    其余内容：`model=` / `key=` 等具名项按名解析；`preset=` 选档位；可识别的阈值项
    （见 OPT_KEYS）覆盖该档位的对应值；剩下的文字一律当备注。
    """
    if not _looks_like_host(parts[0]):
        _warn("[local-rkllm] 配置第 %d 行地址 %r 看起来不像地址（缺 `.` 或 `:`），请确认\n"
              % (lineno, parts[0]))
    scheme, host = _split_base(parts[0])
    if not name:
        name = host                                   # 名称省略 → 用地址当名称
    rest = parts[1:]
    key = ""
    if rest and "=" not in rest[0]:
        key = rest.pop(0)                             # 第二列是密钥
    model, notes, opts, ability = FALLBACK_MODEL, [], {}, ""
    i = 0
    while i < len(rest):
        tok = rest[i]
        low = tok.lower()
        if low.startswith(("ability=", "能力=", "能力说明=")):
            ability = " ".join([tok.split("=", 1)[1]] + rest[i + 1:]).strip()
            break                                     # 其后所有文字都算能力说明
        if low.startswith(("model=", "模型=")):
            model = tok.split("=", 1)[1].strip() or FALLBACK_MODEL
        elif low.startswith(("key=", "token=", "apikey=", "api_key=", "密钥=")):
            key = tok.split("=", 1)[1].strip()
        elif "=" in tok:
            raw_k, raw_v = tok.split("=", 1)
            opt_key, val, known = _coerce_opt(raw_k.lower(), raw_v, lineno)
            if known:
                if val is not None:
                    opts[opt_key] = val
            else:
                notes.append(tok)                     # 不认识，按备注留着
        else:
            notes.append(tok)
        i += 1
    preset = opts.pop("preset", "default")            # 档位作为基准，行内其它项覆盖它
    if preset == "auto":
        preset = detect_preset(model) or "default"
        _warn("[local-rkllm] 配置第 %d 行 preset=auto → 模型名 %r 认作 %s 档\n"
              % (lineno, model, preset))
    entry = dict(PRESETS[preset])
    entry.update(opts)
    entry.update({"preset": preset, "host": host, "scheme": scheme,
                  "api_key": _check_key(name, key),
                  "model": model, "note": " ".join(notes),
                  "ability": ability or ability_of(preset)})
    return name, entry


def _parse_conf(text):
    """解析纯文本配置，返回 (targets, default)。

    格式（一行一个目标，空白分隔、多余空格自动折叠）：
        [名称 =] 地址 [密钥] [备注…] [model=模型名]
    名称省略时用地址作名称；密钥省略按无令牌；模型省略用 FALLBACK_MODEL（实际以服务端
    /health 返回为准）。独立一行 `default = 目标名` 指定默认目标（不写则取第一个）。
    整行以 `#` / `;` 开头为注释，行内 ` #` 之后也是注释。
    """
    targets, default = {}, None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        hit = re.search(r"(?:^|\s)[#;]", line)        # 行内注释
        if hit:
            line = line[:hit.start()].rstrip()
        if not line:
            continue
        if "=" in line:
            head, rest = line.split("=", 1)
            name = head.strip()
        else:
            name, rest = "", line
        parts = rest.split()
        if not parts:
            continue
        if name.lower() in ("default", "默认"):
            default = parts[0]                        # default = board / default board
            continue
        if not name and len(parts) > 1 and not _looks_like_host(parts[0]):
            name = parts.pop(0)                       # 也接受 `board 192.0.2.10:8080 KEY`
        if name in targets:
            _warn("[local-rkllm] 配置第 %d 行目标名重复（%s），后者覆盖前者\n" % (lineno, name))
        name, entry = _entry(name, parts, lineno)
        targets[name] = entry
    return targets, default


def _parse_json(text):
    """解析旧格式 targets.json（保留兼容，不再主推）。

    阈值项与 `.conf` 同键名，同样可覆盖默认值（如 `"tok_per_sec": 2.5`）；
    另可用 `"preset": "generic"` 选档位。
    """
    data = json.loads(text)
    targets, default = {}, data.get("default")
    for name, cfg in (data.get("targets") or {}).items():
        if not isinstance(cfg, dict) or not cfg.get("host"):
            continue
        preset = PRESET_ALIASES.get(str(cfg.get("preset", "default")).lower(), "default")
        model = cfg.get("model") or FALLBACK_MODEL
        if preset == "auto":
            preset = detect_preset(model) or "default"
        entry = dict(PRESETS[preset])
        for key in PRESETS[preset]:
            if key in cfg:
                _, val, _ = _coerce_opt(key, str(cfg[key]), "json")
                if val is not None:
                    entry[key] = val
        scheme, _host = _split_base(cfg["host"])
        entry.update({"preset": preset, "host": _host, "scheme": scheme,
                      "api_key": _check_key(name, cfg.get("api_key") or ""),
                      "model": model, "note": cfg.get("note") or "",
                      "ability": cfg.get("ability") or ability_of(preset)})
        targets[name] = entry
    return targets, default


def load_targets():
    """读取目标配置，返回 (targets, default_name)。

    优先级：配置文件（`RKLLM_CONFIG` → `targets.conf` → `targets.json`）→ 环境变量单目标。
    两者皆无时返回空字典（由 resolve 报错）。
    """
    global CONFIG_PATH
    path = find_config()
    CONFIG_PATH = path
    targets, default = {}, None
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            targets, default = (_parse_json(text) if path.lower().endswith(".json")
                                else _parse_conf(text))
        except Exception as exc:  # noqa: BLE001
            _warn("[local-rkllm] 配置文件解析失败（%s）：%s\n" % (path, exc))
    elif os.environ.get("RKLLM_CONFIG", "").strip():
        _warn("[local-rkllm] 指定的配置文件不存在，改用环境变量：%s\n" % path)
    if not targets and ENV_HOST:
        preset = PRESET_ALIASES.get(ENV_PRESET, "default")
        entry = dict(PRESETS[preset])
        scheme, _host = _split_base(ENV_HOST)
        entry.update({"preset": preset, "host": _host, "scheme": scheme,
                      "api_key": _check_key("env", ENV_KEY),
                      "model": ENV_MODEL or FALLBACK_MODEL, "note": "来自环境变量",
                      "ability": ability_of(preset)})
        targets["env"] = entry
        default = "env"
    return targets, default


def resolve(target=None):
    """解析目标名 → (name, cfg)。target 为空时取配置里的 default。"""
    targets, default = load_targets()
    if not targets:
        raise LocalLLMError(
            "未配置任何模型目标。任选其一：\n"
            "  ① 在 %s 里写一行即可（只写地址和密钥）：\n"
            "         board = 192.0.2.10:8080 <token>\n"
            "     可参考同目录 targets.example.conf；\n"
            "  ② 或设置环境变量 RKLLM_HOST（可选 RKLLM_API_KEY、RKLLM_MODEL）。\n"
            "  当前查找的配置文件：%s" % (os.path.join(HERE, "targets.conf"), CONFIG_PATH))
    name = target or default or next(iter(targets))
    if name not in targets:
        raise LocalLLMError("未知目标 %r；已配置：%s" % (name, "、".join(targets)))
    return name, targets[name]


def _board_for(name, cfg):
    """按目标惰性创建客户端实例（工具链 Board 或内置兜底，接口一致）。

    缓存键含地址 / 令牌 / 超时下限：改了配置会自动重建实例、立即生效，无需重启服务。
    """
    sig = (name, cfg["host"], cfg.get("api_key", ""), cfg.get("timeout_floor"),
           cfg.get("scheme", "http"))
    if sig not in _boards:
        cls = _load_board_class() or _SimpleBoard
        kw = {"timeout": int(cfg.get("timeout_floor", DEFAULTS["timeout_floor"]))}
        if cls is _SimpleBoard:                       # 工具链 Board 接口不含 scheme
            kw["scheme"] = cfg.get("scheme", "http")
        _boards[sig] = cls(cfg["host"], cfg.get("api_key", ""), **kw)
        sys.stderr.write("[local-rkllm] 目标 %s → %s://%s（客户端：%s）\n"
                         % (name, cfg.get("scheme", "http"), cfg["host"],
                            "工具链 Board" if cls is not _SimpleBoard else "内置兜底"))
    return _boards[sig]


def _probe_path(host, key, path, scheme="http"):
    """通用探测：任意 HTTP 路径，2xx 即视为可用；尽力从中解析出模型名。"""
    req = urllib.request.Request(scheme + "://" + host + path, method="GET")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return False, "", repr(exc)[:120]
    model = ""
    try:
        data = json.loads(body)
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list) and items and isinstance(items[0], dict):
            model = items[0].get("id") or items[0].get("name") or ""
        elif isinstance(data, dict):
            model = data.get("model") or ""
    except Exception:  # noqa: BLE001
        pass                                       # 非 JSON 也算可用，只是拿不到模型名
    return True, model, ""


def _probe(name, cfg):
    """按目标的 `health` 设置探测可用性 → (展示用字典, 原始 /health 字典或 None)。

    `rkllm`（默认）解析 rkllm serve 的 /health；`none` 不探测；其它值当作 HTTP 路径，
    2xx 即视为可用（`openai` 归一为 `/v1/models`）。
    """
    mode = cfg.get("health", "rkllm")
    if mode == "none":
        return {"status": "unchecked", "detail": "已配置为不探测（health=none）"}, None
    if mode == "rkllm":
        board = _board_for(name, cfg)
        raw = board.health()
        if not raw:
            return {"status": "unreachable", "detail": board.last_health_problem or ""}, None
        return ({"status": raw.get("status", "ok"), "model": raw.get("model"),
                 "busy": raw.get("busy"), "queue_length": raw.get("queue_length")}, raw)
    ok, model, detail = _probe_path(cfg["host"], cfg.get("api_key", ""), mode,
                                    cfg.get("scheme", "http"))
    if not ok:
        return {"status": "unreachable", "detail": detail}, None
    return {"status": "ok", "model": model, "mode": mode}, None


def _lock_for(name):
    if name not in _locks:
        _locks[name] = threading.Lock()
    return _locks[name]


def _load_board_class():
    """尝试加载主项目工具链的 Board 类。不可用则返回 None（不抛错，复用属可选增强）。

    脚本文件名以数字开头，不是合法 Python 标识符、无法直接 import，故走动态加载。
    加载期间关闭字节码缓存写入，避免在主项目目录里留下 `__pycache__`。
    """
    if not BOARD_SCRIPT or not os.path.exists(BOARD_SCRIPT):
        if BOARD_SCRIPT:
            _warn("[local-rkllm] 指定的工具链脚本不存在，改用内置客户端：%s\n" % BOARD_SCRIPT)
        return None
    try:
        spec = importlib.util.spec_from_file_location("rkllm_toolchain_stress", BOARD_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = prev
        cls = getattr(mod, "Board", None)
        if cls is None:
            _warn("[local-rkllm] 工具链脚本中未找到 Board 类，改用内置客户端\n")
        return cls
    except Exception as exc:  # noqa: BLE001
        _warn("[local-rkllm] 加载工具链脚本失败（%s），改用内置客户端\n" % exc)
        return None


# ---------------- 对外接口 ----------------

def list_targets():
    """列出所有已配置目标及其可用性（按各目标的 `health` 设置探测，不做推理）。"""
    targets, default = load_targets()
    default = default or (next(iter(targets)) if targets else None)
    rows = []
    for name, cfg in targets.items():
        row = {"target": name, "host": cfg["host"], "model": cfg["model"], "note": cfg.get("note", ""),
               "is_default": name == default, "status": "unknown", "detail": "",
               "ability": cfg.get("ability") or ability_of(cfg.get("preset", "default")),
               "budgets": {k: cfg[k] for k in ("preset", "context", "max_in", "max_out",
                                               "tok_per_sec", "timeout_floor", "retries", "health")}}
        try:
            info, _raw = _probe(name, cfg)
            if info.get("model"):
                row["model"] = info["model"]
            row.update(info)
            row["model"] = info.get("model") or row["model"]
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["detail"] = str(exc)[:160]
        rows.append(row)
    return {"config_path": CONFIG_PATH, "default": default, "count": len(rows), "targets": rows}


def health(target=None):
    """查询指定目标的健康状态。

    `health=rkllm`（默认）→ 返回 /health 原始字典；其它模式 → 返回探测结果。
    均附 `_target` / `_host` / `_health`（探测方式）/ `_budgets`（该目标生效的阈值）。
    """
    name, cfg = resolve(target)
    info, raw = _probe(name, cfg)
    if info.get("status") in ("unreachable", "error"):
        raise LocalLLMError("目标 %s（%s）不可用：%s"
                            % (name, cfg["host"], info.get("detail") or "未知原因"))
    out = dict(raw) if raw else {"status": info["status"], "model": info.get("model") or cfg["model"],
                                 "detail": info.get("detail", "")}
    out["_target"] = name
    out["_host"] = cfg["host"]
    out["_health"] = cfg.get("health", "rkllm")
    out["_budgets"] = {k: cfg[k] for k in ("preset", "context", "max_in", "max_out",
                                           "tok_per_sec", "timeout_floor", "retries")}
    out["_ability"] = cfg.get("ability") or ability_of(cfg.get("preset", "default"))
    return out


def _est_tokens(text, cfg=None):
    """按字符数保守估算 token 数（默认中文为主约 1 token ≈ 1.6 字符）。"""
    if not text:
        return 0
    cpt = (cfg or DEFAULTS)["chars_per_token"]
    return int(len(text) / cpt) + 1


def _timeout_for(max_tokens, cfg=None):
    """按目标的实测吞吐推算超时：生成耗时 + 固定余量。"""
    cfg = cfg or DEFAULTS
    return max(int(cfg["timeout_floor"]), int(20 + max_tokens / cfg["tok_per_sec"] * 1.3))


def ask(prompt, system=None, max_tokens=None, temperature=0.7,
        max_output_chars=None, timeout=None, retries=None, target=None):
    """单轮问答。返回 text / target / model / truncated / est_input_tokens / max_tokens / elapsed / retries / usage。

    防护行为（阈值取**该目标生效的配置**，默认即 RK3588 特化值）：
      · 输入超 `max_in` → 直接抛错并提示分段，不发请求（避开长 prompt 崩溃高发区）
      · max_tokens 封顶到 `max_out`（服务端不遵守该参数，只能客户端截）
      · 输出超长 → 硬截断并置 truncated=True；另做退化特征自检（`degen_check`）
      · 请求失败 → 等待 `retry_wait` 秒后重试，共 `retries` 次（等守护进程拉起服务）
    """
    if not prompt or not prompt.strip():
        raise LocalLLMError("prompt 不能为空")

    name, cfg = resolve(target)
    model = cfg["model"]
    safe_in, safe_out = cfg["max_in"], cfg["max_out"]
    if retries is None:
        retries = int(cfg["retries"])

    # 1) 输入预算预检
    est_in = _est_tokens(prompt, cfg) + _est_tokens(system, cfg)
    if est_in > safe_in:
        raise LocalLLMError(
            "输入约 %d token，超过目标 %s 的输入预算 %d。请先分段再逐段调用"
            "（该预算为实测值，可在配置里用 max_in= 调整）。" % (est_in, name, safe_in))

    # 2) 输出预算
    mt = safe_out if max_tokens is None else min(int(max_tokens), safe_out)
    mt = max(64, mt)
    mt = max(64, min(mt, cfg["context"] - est_in - 64))
    if max_output_chars is None:
        max_output_chars = int(mt * cfg["chars_per_token"] * cfg["overrun"])  # 留出 max_tokens 不遵守的余量

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    board = _board_for(name, cfg)
    if timeout is None:
        timeout = _timeout_for(mt, cfg)

    last_err = None
    for attempt in range(retries + 1):
        try:
            with _lock_for(name):            # 同一目标内串行；不同目标互不阻塞
                status, text, elapsed, usage = board.ask(
                    prompt, max_tokens=mt, timeout=timeout,
                    extra={"model": model, "temperature": temperature, "messages": messages})
            if status != 200:
                raise LocalLLMError("目标 %s 返回 HTTP %s：%s" % (name, status, str(text)[:120]))
            text = text or ""
            truncated = False
            if len(text) > max_output_chars:
                text = text[:max_output_chars]
                truncated = True
            # 退化特征自检：重复词流（实测越过约 1000 token 时的典型塌陷形态）
            if cfg.get("degen_check") and re.search(r"(\b\w{2,6}\b)(?:\s*\1){3,}", text[:600]):
                truncated = True
            return {
                "text": text,
                "target": name,
                "model": model,
                "truncated": truncated,
                "est_input_tokens": est_in,
                "max_tokens": mt,
                "timeout": timeout,
                "elapsed": round(elapsed, 2),
                "retries": attempt,
                "usage": usage or {},
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(cfg["retry_wait"])       # 等守护进程把服务拉起来
    raise LocalLLMError("目标 %s 调用失败（已重试 %d 次）：%s" % (name, retries, last_err)) from last_err


def calibrate(target=None, max_tokens=384):
    """**实机校准**：真跑一次较长生成，量出该目标的真实吞吐，并给出建议配置。

    为什么要这一步：档位里的吞吐/预算是跨平台估算值，**只有实机跑出来的才准**。
    说明：这里测的是「含 prefill 与首 token 开销」的端到端吞吐，因此是**保守下界**——
    拿它算超时不会偏小。输出过短（<60 token）时开销占比过高，会标注 `reliable=false`。
    校准用固定短任务（数数），不改动任何配置；把输出的建议行抄进 `targets.conf` 即可。

    返回 dict：目标 / 模型 / 实测 tok/s / 输出 token / 耗时 / 建议配置行 / 是否可靠。
    """
    name, cfg = resolve(target)
    # 探针必须能逼出长输出：枚举类任务（如「数到 500」）实测会被模型提前收尾，测不出真实吞吐
    probe = "请写一段约 300 字的说明文，主题是「秋天的早晨」，语言平实，不要分点。"
    t0 = time.time()
    r = ask(probe, max_tokens=max_tokens, temperature=0.2, retries=0, timeout=1800, target=target)
    elapsed = r["elapsed"] or (time.time() - t0)
    ct = int((r.get("usage") or {}).get("completion_tokens") or 0)
    if ct <= 0:                                        # 服务端没回 usage 时按字符估算
        ct = max(1, _est_tokens(r["text"], cfg))
    tps = round(ct / elapsed, 2) if elapsed else 0.0
    reliable = ct >= 60                                # 太短则开销占比过高，不足以代表稳态
    # 建议：实测值本身已是保守下界，不再打折；超时下限按该档位输出上限的预计耗时 + 余量
    suggest_floor = int(30 + cfg["max_out"] / max(0.1, tps) * 1.2)
    return {
        "target": name,
        "model": r["model"],
        "measured_tok_per_sec": tps,
        "completion_tokens": ct,
        "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
        "elapsed_s": elapsed,
        "reliable": reliable,
        "note": "" if reliable else "输出过短，含首 token 开销，实测值偏保守（建议用 --tokens 调大再测）",
        "text_head": r["text"][:60],
        "suggest": "preset=%s tok_per_sec=%s timeout_floor=%s"
                   % (cfg.get("preset", "default"), tps, suggest_floor),
    }


def _cli(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    cmd = argv[0]
    if cmd == "targets":
        print(json.dumps(list_targets(), ensure_ascii=False, indent=2))
        return 0

    target = None
    rest = []
    i = 1
    while i < len(argv):
        if argv[i] == "--target":
            target = argv[i + 1]; i += 2
        else:
            rest.append(argv[i]); i += 1

    if cmd == "health":
        print(json.dumps(health(target), ensure_ascii=False, indent=2))
        return 0

    if cmd == "calibrate":
        tokens = 128
        for j, a in enumerate(rest):
            if a == "--tokens" and j + 1 < len(rest):
                tokens = int(rest[j + 1])
        c = calibrate(target=target, max_tokens=tokens)
        sys.stderr.write("[calibrate] %s @ %s：实测 %s tok/s（%s token / %ss）\n"
                         % (c["model"], c["target"], c["measured_tok_per_sec"],
                            c["completion_tokens"], c["elapsed_s"]))
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return 0

    if cmd == "ask":
        prompt = system = None
        max_tokens = None
        i = 0
        while i < len(rest):
            a = rest[i]
            if a == "--file":
                prompt = open(rest[i + 1], encoding="utf-8").read(); i += 2
            elif a == "--system":
                system = rest[i + 1]; i += 2
            elif a == "--max-tokens":
                max_tokens = int(rest[i + 1]); i += 2
            else:
                prompt = a; i += 1
        if prompt is None:
            print("缺少 prompt", file=sys.stderr); return 2
        r = ask(prompt, system=system, max_tokens=max_tokens, target=target)
        sys.stderr.write("[%s @ %s] 输入≈%d tok 输出上限%d tok 耗时%ss 重试%d%s\n" % (
            r["model"], r["target"], r["est_input_tokens"], r["max_tokens"],
            r["elapsed"], r["retries"], " 【已截断】" if r["truncated"] else ""))
        print(r["text"])
        return 0

    print("未知子命令：%s" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.exit(_cli(sys.argv[1:]))
    except LocalLLMError as e:
        print("错误：%s" % e, file=sys.stderr)
        sys.exit(1)
