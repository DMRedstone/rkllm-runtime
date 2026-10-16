# -*- coding: utf-8 -*-
"""本地模型连接器（local-rkllm）· MCP 服务层。

**职责**：stdio 上的 MCP 协议处理 + 4 个工具定义，推理调用全部委托给同目录的
`rkllm_client.py`（多目标解析 / 预算防护 / 可选复用主项目工具链 Board）。
本文件不含任何地址、令牌与阈值——那些都在 `rkllm_client.py` 与 `targets.conf` 里。

**独立性**：纯标准库，零第三方依赖，随目录整体移植。

**协议**：stdio 上按行分隔的 JSON-RPC 2.0（MCP stdio transport）。
stdout 只输出协议报文，一切日志与告警走 stderr，绝不污染 stdout。

暴露的工具：
    local_llm_ask      单轮问答，返回纯文本
    local_llm_batch    多段文本逐段串行处理（长文分段场景）
    local_llm_health   指定目标的健康状态
    local_llm_targets  列出全部目标的档位、生效预算、能力边界与可用性
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rkllm_client as rc  # noqa: E402  同目录，路径已插入 sys.path

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "local-rkllm", "version": "1.0.0"}

# 写进 ask / batch 的工具描述，用来约束调用方模型
ABILITY_NOTE = (
    "调用前先用 local_llm_targets 查看目标的档位与能力边界，不要把超出该目标能力"
    "的任务交给它——例如让 1B/4B 档写长文、写代码、做多步推理，或要求超过其输出上限"
    "的长度。单次调用耗时 3～150 秒，调用方需容忍长等待。"
)

TOOLS = [
    {
        "name": "local_llm_ask",
        "description": (
            "单轮问答，返回纯文本。适合短文本改写、摘要（输出<300 字）、分类、"
            "信息提取、格式转换。不适合长文生成、代码生成与多步推理。" + ABILITY_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "问题或待处理的文本（必填）"},
                "system": {"type": "string", "description": "可选的系统提示词"},
                "max_tokens": {
                    "type": "integer",
                    "description": "可选的输出 token 上限；实际按目标 max_out 封顶",
                },
                "temperature": {"type": "number", "description": "可选，默认 0.7"},
                "target": {
                    "type": "string",
                    "description": "可选的目标名；不传则用 targets.conf 里的 default",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "local_llm_batch",
        "description": (
            "多段文本逐段串行处理（长文分段场景）。同一目标内部串行、不同目标可并行，"
            "返回 JSON 数组（每段一条：index / ok / text）。" + ABILITY_NOTE
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "待处理的文本段列表（必填，逐段串行送入）",
                },
                "system": {"type": "string", "description": "可选的系统提示词，每段共用"},
                "max_tokens": {"type": "integer", "description": "可选的输出 token 上限"},
                "target": {"type": "string", "description": "可选的目标名"},
            },
            "required": ["segments"],
        },
    },
    {
        "name": "local_llm_health",
        "description": (
            "查询指定目标的健康状态：在线与忙碌情况、队列长度、运行时长、累计请求统计，"
            "并附该目标生效的预算阈值。只探测不推理，很快返回。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "可选的目标名；不传则用默认目标"},
            },
        },
    },
    {
        "name": "local_llm_targets",
        "description": (
            "列出全部已配置目标：档位、生效预算、模型、备注、能力边界及其可用性，并标出"
            "默认目标。首次调用或拿不准任务该不该交给它时，先调这个。"
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _text(text):
    return {"type": "text", "text": text}


def _json_text(obj):
    return _text(json.dumps(obj, ensure_ascii=False, indent=2))


def _ask_kwargs(args):
    kw = {"prompt": args.get("prompt")}
    for key in ("system", "max_tokens", "target"):
        if args.get(key) is not None:
            kw[key] = args[key]
    if args.get("temperature") is not None:
        kw["temperature"] = args["temperature"]
    return kw


def tool_ask(args):
    if not (args.get("prompt") or "").strip():
        raise ValueError("prompt 不能为空")
    r = rc.ask(**_ask_kwargs(args))
    blocks = [_text(r["text"])]
    if r.get("truncated") or r.get("retries"):
        blocks.append(_text(
            "[%s] 目标=%s 模型=%s 输入≈%s tok 输出上限%s tok 耗时%ss"
            % ("输出被截断" if r.get("truncated") else "已重试 %d 次" % r["retries"],
               r["target"], r["model"], r["est_input_tokens"],
               r["max_tokens"], r["elapsed"])))
    sys.stderr.write("[local-rkllm] ask 目标=%s 模型=%s 输入≈%s tok 耗时%ss 重试%d%s\n"
                     % (r["target"], r["model"], r["est_input_tokens"], r["elapsed"],
                        r["retries"], " 已截断" if r.get("truncated") else ""))
    return {"content": blocks}


def tool_batch(args):
    segments = args.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments 必须是非空字符串数组")
    results = []
    for i, seg in enumerate(segments):
        kw = _ask_kwargs({"prompt": seg, "system": args.get("system"),
                          "max_tokens": args.get("max_tokens"), "target": args.get("target")})
        try:
            r = rc.ask(**kw)
            results.append({"index": i, "ok": True, "text": r["text"],
                            "target": r["target"], "model": r["model"],
                            "truncated": r["truncated"], "elapsed": r["elapsed"]})
            sys.stderr.write("[local-rkllm] batch %d/%d ok 耗时%ss\n"
                             % (i + 1, len(segments), r["elapsed"]))
        except Exception as exc:  # noqa: BLE001  单段失败不中断整批
            results.append({"index": i, "ok": False, "error": str(exc)[:300]})
            sys.stderr.write("[local-rkllm] batch %d/%d 失败：%s\n" % (i + 1, len(segments), exc))
    return {"content": [_json_text({"count": len(results), "results": results})]}


def tool_health(args):
    return {"content": [_json_text(rc.health(args.get("target")))]}


def tool_targets(args):
    return {"content": [_json_text(rc.list_targets())]}


HANDLERS = {
    "local_llm_ask": tool_ask,
    "local_llm_batch": tool_batch,
    "local_llm_health": tool_health,
    "local_llm_targets": tool_targets,
}


def call_tool(name, args):
    fn = HANDLERS.get(name)
    if fn is None:
        return {"content": [_text("未知工具：%s（可用：%s）" % (name, "、".join(HANDLERS)))],
                "isError": True}
    try:
        result = fn(args or {})
    except Exception as exc:  # noqa: BLE001
        return {"content": [_text(str(exc))], "isError": True}
    result.setdefault("isError", False)
    return result


def handle(msg):
    """处理一条 JSON-RPC 报文，返回（是否需要响应, 待发送报文）。"""
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    if mid is None:                                   # 通知，不响应
        return False, None
    if method == "initialize":
        return True, {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }}
    if method == "tools/list":
        return True, {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        result = call_tool(params.get("name"), params.get("arguments") or {})
        return True, {"jsonrpc": "2.0", "id": mid, "result": result}
    if method == "ping":
        return True, {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method in ("resources/list", "resources/templates/list"):
        return True, {"jsonrpc": "2.0", "id": mid, "result": {"resources": []}}
    if method == "prompts/list":
        return True, {"jsonrpc": "2.0", "id": mid, "result": {"prompts": []}}
    return True, {"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "不支持的方法：%s" % method}}


def main():
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    sys.stderr.write("[local-rkllm] MCP 服务已启动（stdio）：%s\n" % "、".join(HANDLERS))
    sys.stderr.flush()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except Exception:  # noqa: BLE001
            sys.stdout.write(json.dumps(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "解析失败：不是合法 JSON"}},
                ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue
        try:
            send, payload = handle(msg)
        except Exception as exc:  # noqa: BLE001  兜底：任何异常都不能中断服务
            send, payload = True, {"jsonrpc": "2.0", "id": msg.get("id"),
                                   "error": {"code": -32603, "message": str(exc)[:300]}}
        if send and payload is not None:
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
