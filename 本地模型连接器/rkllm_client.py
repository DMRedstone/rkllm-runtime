# -*- coding: utf-8 -*-
"""本地板端小模型（rkllm serve）的调用封装 —— 自包含 · 多目标 · 可选复用主项目工具链。

**独立性**：纯标准库（`urllib` / `json` / `threading`），HTTP 客户端是同目录的
`board_api.py`（与工具链同一份），不依赖任何其它项目目录、
不依赖第三方包。把 `local-rkllm/` 整个目录拷到任意能访问板端服务的机器即可工作，
Windows / Linux / macOS 均可。

**多目标**：支持同时配置**多个本地部署**（不同地址 / 令牌 / 模型），用目标名切换。
配置是本目录下的**纯文本** `targets.conf`（参考 `targets.example.conf`），随目录一起移植：

    # 一行一个目标：[名称 =] 地址[@密钥] [preset=档位] [键=值…] [备注…]
    board = 192.0.2.10:8080@TOKEN             # 地址与密钥用 @ 连成一个字段，无歧义
    192.0.2.50:8080                           # 无密钥直接写地址（addr@ 表示空密钥）
    board-b = 192.0.2.10:8080@TOKEN model=qwen2.5-1.5b preset=1b   # 需要时再指定模型

**配置来源优先级**：`RKLLM_CONFIG` 指定的文件 → 本目录 `targets.conf` → 本目录
`targets.json`（旧格式，仍兼容）→ 环境变量单目标（`RKLLM_HOST` / `RKLLM_API_KEY` / `RKLLM_MODEL`）。
**代码中不含任何具体地址**，未配置时直接报错并给出配置指引。

**HTTP 层**：由同目录 `board_api.py` 提供（工具链目录里那份的副本，内容一致）。
旧版用 `RKLLM_BOARD_SCRIPT` 从工具链脚本里 exec 出 Board 类——脚本名以数字开头不是
合法模块名才需要那种 hack；现在两边共用同一模块，该变量已废弃。

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

命令行用法：
    python rkllm_client.py targets
    python rkllm_client.py health [--target board-b]
    python rkllm_client.py calibrate [--target board-b] [--tokens 128]    # 实机校准吞吐
    python rkllm_client.py ask "把下面这段改写得更简洁：……" [--target board-b]
"""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import board_api   # 与工具链同一份 HTTP 实现（本目录保留副本）

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
    "chars_per_token": 0,    # 0 = 按 CJK 占比自动（中文 1.6 / 拉丁 3.5）
    "prefill_tps": 0,        # 0 = 取 tok_per_sec × 8；长输入的超时要靠它
    "base_path": "",         # 网关 / 内网穿透的路径前缀（如 /rkllm）
    "auth_header": "Authorization",   # Azure 用 api-key、Anthropic 网关用 x-api-key
    "auth_prefix": "Bearer ",
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
    "prefill_tps": ("prefill_tps", "prefill"),
    "base_path": ("base_path", "路径前缀"),
    "auth_header": ("auth_header", "认证头"),
    "auth_prefix": ("auth_prefix", "认证前缀"),
    "overrun": ("overrun",), "degen_check": ("degen_check", "degen", "退化自检"),
    "health": ("health", "健康检查"), "preset": ("preset", "档位", "模式", "通用选项"),
    "ability": ("ability", "能力", "能力说明"),
}
_INT_OPTS = ("max_in", "max_out", "context", "timeout_floor", "retries", "retry_wait")
_FLOAT_OPTS = ("tok_per_sec", "chars_per_token", "overrun", "prefill_tps")
_ZERO_OK = ("retries", "retry_wait", "chars_per_token", "prefill_tps")
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
    "1b": {"max_in": 3000, "max_out": 512, "context": 4096, "tok_per_sec": 8,
           "timeout_floor": 60, "retries": 2, "retry_wait": 5, "chars_per_token": 0,
           "overrun": 1.3, "degen_check": False, "health": "rkllm"},
    "3b": {"max_in": 5500, "max_out": 1024, "context": 8192, "tok_per_sec": 5,
           "timeout_floor": 120, "retries": 2, "retry_wait": 5, "chars_per_token": 0,
           "overrun": 1.3, "degen_check": False, "health": "rkllm"},
    "4b": {"max_in": 6000, "max_out": 1024, "context": 8192, "tok_per_sec": 4,
           "timeout_floor": 120, "retries": 2, "retry_wait": 5, "chars_per_token": 0,
           "overrun": 1.3, "degen_check": False, "health": "rkllm"},
    "7b": {"max_in": 6000, "max_out": 1536, "context": 8192, "tok_per_sec": 3,
           "timeout_floor": 180, "retries": 2, "retry_wait": 5, "chars_per_token": 0,
           "overrun": 1.25, "degen_check": False, "health": "rkllm"},
    "14b": {"max_in": 12000, "max_out": 2048, "context": 16384, "tok_per_sec": 2,
            "timeout_floor": 240, "retries": 2, "retry_wait": 5, "chars_per_token": 0,
            "overrun": 1.25, "degen_check": False, "health": "rkllm"},
    "32b": {"max_in": 24000, "max_out": 3072, "context": 32768, "tok_per_sec": 1,
            "timeout_floor": 300, "retries": 1, "retry_wait": 5, "chars_per_token": 0,
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
_locks_guard = threading.Lock()   # 保护 _locks 的创建（check-then-act 竞态）
_targets_cache = {"key": None, "value": None}


class LocalLLMError(Exception):
    """调用板端小模型时的可读错误。

    `retryable=False` 表示重试不可能成功（4xx、空响应、输入越界），ask() 会
    立即失败，而不是把同一个错误再等三遍。
    """

    def __init__(self, message, retryable=True):
        Exception.__init__(self, message)
        self.retryable = retryable


# HTTP 客户端统一由同目录的 board_api.py 提供（与 工具/大模型测试工具链
# 里的那一份内容相同，本目录保留副本以便整目录移植）。改造前这里有一份
# _SimpleBoard，它在健康判定、认证头与 URL 拼接上与工具链的 Board 不一致，
# 同一个服务能得出两种结论。


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

    位置列只有一样：**地址**（第 1 列，必填）。地址可用 `@` 绑定密钥形成原子字段
    （`地址@密钥`，无歧义；`地址@` 表示空密钥）——推荐写法。兼容旧式：地址后跟独立
    密钥列（第 2 列裸词）。
    其余内容：`model=` / `key=` 等具名项按名解析；`preset=` 选档位；可识别的阈值项
    （见 OPT_KEYS）覆盖该档位的对应值；剩下的文字一律当备注。
    """
    addr = parts[0]
    key_inline = None                          # None=未用 @ 形式；""=显式空密钥
    if "@" in addr:
        addr, _, key_inline = addr.partition("@")
        key_inline = key_inline.strip()
    if not _looks_like_host(addr):
        _warn("[local-rkllm] 配置第 %d 行地址 %r 看起来不像地址（缺 `.` 或 `:`），请确认\n"
              % (lineno, addr))
    scheme, host = _split_base(addr)
    if not name:
        name = host                            # 名称省略 → 用地址当名称
    rest = parts[1:]
    key = "" if key_inline is not None else ""
    if key_inline is not None:
        key = key_inline
    elif rest and "=" not in rest[0]:
        key = rest.pop(0)                      # 旧式：第二列是密钥
    model, notes, opts, ability = FALLBACK_MODEL, [], {}, ""
    model_explicit = False
    i = 0
    while i < len(rest):
        tok = rest[i]
        low = tok.lower()
        if low.startswith(("ability=", "能力=", "能力说明=")):
            ability = " ".join([tok.split("=", 1)[1]] + rest[i + 1:]).strip()
            break                                     # 其后所有文字都算能力说明
        if low.startswith(("model=", "模型=")):
            explicit = tok.split("=", 1)[1].strip()
            model = explicit or FALLBACK_MODEL
            model_explicit = bool(explicit)
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
                  "model": model, "model_explicit": model_explicit,
                  "note": " ".join(notes),
                  "ability": ability or ability_of(preset)})
    return name, entry


def _parse_conf(text):
    """解析纯文本配置，返回 (targets, default)。

    目标区格式（一行一个目标，空白分隔、多余空格自动折叠）：
        [名称 =] 地址[@密钥] [preset=档位] [键=值…] [备注…]
    名称省略时用地址作名称；密钥用 `@` 绑定在地址上（推荐，无歧义），旧式独立密钥列
    仍兼容；模型省略用 FALLBACK_MODEL（实际以服务端 /health 返回为准）。独立一行
    `default = 目标名` 指定默认目标（不写则取第一个）。整行以 `#` / `;` 开头为注释，
    行内 ` #` 之后也是注释。

    `[ability]` 段：段内一行 `目标名 = 能力描述`，覆盖该目标给调用方模型看的能力边界
    （未写的回落到 preset 档位默认描述）。
    """
    targets, default = {}, None
    section = None                             # None=目标区；"ability"=能力段
    ability_map = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip().lower()
            if sec in ("ability", "能力", "能力说明"):
                section = "ability"
            else:
                section = None
                _warn("[local-rkllm] 配置第 %d 行未知配置段 [%s]，该段内容按目标区处理\n"
                      % (lineno, sec))
            continue
        if section == "ability":
            hit = re.search(r"(?:^|\s)[#;]", line)   # 行内注释（段内描述中避免用 ` #`）
            if hit:
                line = line[:hit.start()].rstrip()
            if "=" in line:
                n, _, v = line.partition("=")
                if n.strip():
                    ability_map[n.strip()] = v.strip()
            elif line:
                _warn("[local-rkllm] 配置第 %d 行 [ability] 段缺少 `目标名 =`，已忽略：%r\n"
                      % (lineno, line))
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
        if not name and len(parts) > 1 and not _looks_like_host(parts[0].split("@")[0]):
            name = parts.pop(0)                       # 也接受 `board 192.0.2.10:8080@KEY`
        if name in targets:
            _warn("[local-rkllm] 配置第 %d 行目标名重复（%s），后者覆盖前者\n" % (lineno, name))
        name, entry = _entry(name, parts, lineno)
        targets[name] = entry
    for name, entry in targets.items():           # [ability] 段后处理，位置无关
        if name in ability_map:                   # 优先于档位默认与行内 ability=
            entry["ability"] = ability_map[name]
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
        model_explicit = bool(cfg.get("model"))
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
                      "model": model, "model_explicit": model_explicit,
                      "note": cfg.get("note") or "",
                      "ability": cfg.get("ability") or ability_of(preset)})
        targets[name] = entry
    return targets, default


def load_targets():
    """读取目标配置，返回 (targets, default_name)。

    优先级：配置文件（`RKLLM_CONFIG` → `targets.conf` → `targets.json`）→ 环境变量单目标。
    结果按 (路径, mtime, 大小) 缓存：ask() 每次调用都走这里，配置没变就不重复
    解析；文件一改（mtime 变化）立即生效，保持「改配置不用重启」的既有行为。
    """
    global CONFIG_PATH
    path = find_config()
    CONFIG_PATH = path
    try:
        info = os.stat(path)
        cache_key = (path, info.st_mtime_ns, info.st_size)
    except OSError:
        cache_key = (path, None, None)
    if _targets_cache["key"] == cache_key and _targets_cache["value"] is not None:
        return _targets_cache["value"]

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
                      "model": ENV_MODEL or FALLBACK_MODEL,
                      "model_explicit": bool(ENV_MODEL),
                      "note": "来自环境变量", "ability": ability_of(preset)})
        targets["env"] = entry
        default = "env"
    _targets_cache.update({"key": cache_key, "value": (targets, default)})
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
    """按目标惰性创建 board_api.Board（与工具链共用同一实现）。

    缓存键含地址 / 令牌 / 超时下限 / 协议 / 路径前缀 / 认证头：改配置会自动
    重建实例、立即生效，无需重启服务。
    """
    sig = (name, cfg["host"], cfg.get("api_key", ""), cfg.get("timeout_floor"),
           cfg.get("scheme", "http"), cfg.get("base_path", ""),
           cfg.get("auth_header", "Authorization"), cfg.get("auth_prefix", "Bearer "))
    if sig not in _boards:
        _boards[sig] = board_api.Board(
            cfg["host"], cfg.get("api_key", ""),
            timeout=int(cfg.get("timeout_floor", DEFAULTS["timeout_floor"])),
            scheme=cfg.get("scheme", "http"),
            base_path=cfg.get("base_path", ""),
            auth_header=cfg.get("auth_header", "Authorization"),
            auth_prefix=cfg.get("auth_prefix", "Bearer "))
        prefix = "/" + cfg["base_path"].strip("/") if cfg.get("base_path") else ""
        sys.stderr.write("[local-rkllm] 目标 %s → %s://%s%s（board_api）\n"
                         % (name, cfg.get("scheme", "http"), cfg["host"], prefix))
    return _boards[sig]


def _probe_path(name, cfg, path):
    """通用探测：任意 HTTP 路径，2xx 即视为可用；尽力解析出模型名。

    走 board_api 的 URL/认证规则——旧实现自己拼 `scheme + "://" + host + path`，
    漏掉了路径前缀与自定义认证头。
    """
    status, body, _elapsed, err = board_api.request(
        board_api.build_base(cfg["host"], cfg.get("scheme", "http")),
        cfg.get("api_key", ""), path, None, "GET", HEALTH_TIMEOUT,
        auth_header=cfg.get("auth_header", "Authorization"),
        auth_prefix=cfg.get("auth_prefix", "Bearer "),
        base_path=cfg.get("base_path", ""))
    if status == "EXC":
        return False, "", (err or "connection error")[:120]
    if not (isinstance(status, int) and 200 <= status < 300):
        return False, "", "HTTP %s" % status
    model = ""
    try:
        data = json.loads(body)
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list) and items and isinstance(items[0], dict):
            model = items[0].get("id") or items[0].get("name") or ""
        elif isinstance(data, dict):
            model = data.get("model") or ""
    except Exception:  # noqa: BLE001
        pass                                   # 非 JSON 也算可用，只是拿不到模型名
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
    ok, model, detail = _probe_path(name, cfg, mode)
    if not ok:
        return {"status": "unreachable", "detail": detail}, None
    return {"status": "ok", "model": model, "mode": mode}, None


def _lock_for(name):
    """取目标锁。

    check-then-act 必须加锁：两个线程同时首访同一目标时各自创建一把锁并互相
    覆盖，之后各持不同的锁对象，串行化形同虚设——而 NPU 只接受一个请求。
    """
    with _locks_guard:
        if name not in _locks:
            _locks[name] = threading.Lock()
        return _locks[name]


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


def _cpt_for(text, cfg=None):
    """1 token 约等于多少字符；配置为 0/未配置时按 CJK 占比自动选。

    固定 1.6 是为中文定的：英文实际约 4 字符/token，用 1.6 会高估 2.5 倍，
    一段 12000 字符的英文会被算成 7500 token 而误拒。
    """
    cpt = (cfg or DEFAULTS).get("chars_per_token") or 0
    if cpt > 0:
        return cpt
    if not text:
        return 1.6
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    return 1.6 if cjk * 3 >= len(text) else 3.5


def _est_tokens(text, cfg=None):
    """按字符数保守估算 token 数（默认按语言自动选 1.6 / 3.5）。"""
    if not text:
        return 0
    return int(len(text) / _cpt_for(text, cfg)) + 1


def _timeout_for(max_tokens, est_in=0, cfg=None):
    """按吞吐推算超时：prefill + 生成 + 固定余量。

    旧公式只有生成项：小 NPU 上 2000 token 的 prefill 就要一两百秒，光算
    生成会把本来能成功的请求判成超时，再被 retries 放大三倍。
    """
    cfg = cfg or DEFAULTS
    tps = max(0.1, float(cfg["tok_per_sec"]))
    prefill_tps = float(cfg.get("prefill_tps") or 0) or tps * 8
    return max(int(cfg["timeout_floor"]),
               int(20 + est_in / max(0.1, prefill_tps) + max_tokens / tps * 1.3))


def ask(prompt, system=None, max_tokens=None, temperature=0.7,
        max_output_chars=None, timeout=None, retries=None, target=None):
    """单轮问答。返回 text / target / model / truncated / est_input_tokens / max_tokens / elapsed / retries / usage。

    防护行为（阈值取**该目标生效的配置**）：
      · 输入超 `max_in` → 直接抛错并提示分段，不发请求
      · 输出预算与输入共享上下文；装不下时拒绝，而不是把负数兜成 64 造成越界
      · `model` 只在配置里显式写了才进请求体（rkllm 忽略它，vLLM 用它路由）
      · 200 但正文为空 → 报错，而不是静默返回空串
      · 4xx（除 408/429）不重试；连接错误 / 5xx 才重试
    """
    if not prompt or not prompt.strip():
        raise LocalLLMError("prompt 不能为空", retryable=False)

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
            "（该预算可在配置里用 max_in= 调整）。" % (est_in, name, safe_in),
            retryable=False)

    # 2) 输出预算：与输入共享上下文，装不下就拒绝
    mt = safe_out if max_tokens is None else min(int(max_tokens), safe_out)
    mt = max(64, mt)
    room = cfg["context"] - est_in - 64
    if room < 64:
        raise LocalLLMError(
            "输入约 %d token 已占满 %s 的 %d 上下文，留不出输出空间；请分段。"
            % (est_in, name, cfg["context"]), retryable=False)
    mt = min(mt, room)
    cpt = _cpt_for(prompt, cfg)
    if max_output_chars is None:
        max_output_chars = int(mt * cpt * cfg["overrun"])   # 服务端不遵守 max_tokens

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra = {"temperature": temperature, "messages": messages}
    if cfg.get("model_explicit"):
        extra["model"] = model

    board = _board_for(name, cfg)
    if timeout is None:
        timeout = _timeout_for(mt, est_in, cfg)

    last_err = None
    attempt = 0
    for attempt in range(retries + 1):
        try:
            with _lock_for(name):            # 同一目标内串行；不同目标互不阻塞
                status, text, elapsed, usage = board.ask(
                    prompt, max_tokens=mt, timeout=timeout, extra=extra)
            if status != 200:
                retryable = not (isinstance(status, int) and 400 <= status < 500
                                 and status not in (408, 429))
                raise LocalLLMError(
                    "目标 %s 返回 HTTP %s：%s" % (name, status, str(text)[:120]),
                    retryable=retryable)
            text = text or ""
            if not text.strip() and not int((usage or {}).get("completion_tokens") or 0):
                raise LocalLLMError(
                    "目标 %s 返回 200 但正文为空——响应可能不是 OpenAI 兼容结构"
                    "（用 health= 指向 /v1/models 可先核对端点）" % name,
                    retryable=False)
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
            if getattr(exc, "retryable", True) is False:
                break
            if attempt < retries:
                time.sleep(cfg["retry_wait"])       # 等守护进程把服务拉起来
    raise LocalLLMError("目标 %s 调用失败（已重试 %d 次）：%s"
                        % (name, attempt, last_err)) from last_err


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
