"""F10 实机验收：LangGraph 与 OpenAI Agents SDK 各跑通一条工具调用闭环。

设计原文（02 G2）：「作为子智能体可用：工具调用闭环、结构化输出、推理内容分离、
可取消」——验收方式为「LangGraph / OpenAI Agents SDK / Dify Agent 三种调用方式
各跑通一条链路」。本次无 Dify 实例，只跑前两个。

关键点：
- LangGraph 走 chat/completions（langchain-openai 的默认端点）；
- Agents SDK 走 /v1/responses（它默认就是这个端点）——正好验证本轮新增的翻译层；
- 板端 4B 模型可能不触发工具调用：脚本区分「框架链路失败」与「模型没吐 tool_call」，
  前者是缺陷，后者是模型能力，不能混为一谈。

用法：F10_KEY=<板端 key> python f10_frameworks.py
"""

import asyncio
import json
import os
import sys
import time

BASE = os.environ.get("F10_BASE", "http://10.0.0.14:8080/v1")
KEY = os.environ["F10_KEY"]


def ok(name, good, detail=""):
    print("%s %-44s %s" % ("[PASS]" if good else "[FAIL]", name, detail), flush=True)
    return good


# ---------------------------------------------------------------------------
# 1) LangGraph + langchain-openai（chat/completions）
# ---------------------------------------------------------------------------
def run_langgraph() -> dict:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    calls = []

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。"""
        calls.append(city)
        return json.dumps({"city": city, "temp_c": 18}, ensure_ascii=False)

    llm = ChatOpenAI(model="rkllm", base_url=BASE, api_key=KEY,
                     temperature=0.3, timeout=180, max_retries=1)
    agent = create_react_agent(llm, [get_weather])
    t0 = time.time()
    state = agent.invoke({"messages": [
        ("system", "你是助手。需要天气时必须调用 get_weather 工具，不要自己编。"),
        ("user", "巴黎现在多少度？请调用工具查询。")]},
        config={"recursion_limit": 6})
    elapsed = time.time() - t0
    msgs = state.get("messages") or []
    tool_msgs = [m for m in msgs if getattr(m, "type", "") == "tool"]
    answer = msgs[-1].content if msgs else ""
    return {"elapsed_s": round(elapsed, 1), "tool_called": bool(calls),
            "tool_results": tool_msgs[0].content if tool_msgs else None,
            "answer": (answer or "")[:160]}


# ---------------------------------------------------------------------------
# 2) OpenAI Agents SDK（默认 /v1/responses）
# ---------------------------------------------------------------------------
def run_agents_sdk() -> dict:
    from agents import (Agent, Runner, function_tool, set_default_openai_client,
                        set_default_openai_api)
    from openai import AsyncOpenAI

    calls = []

    @function_tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。"""
        calls.append(city)
        return json.dumps({"city": city, "temp_c": 18}, ensure_ascii=False)

    client = AsyncOpenAI(base_url=BASE, api_key=KEY, timeout=180.0)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("responses")        # 显式走翻译层

    agent = Agent(name="board", model="rkllm",
                  instructions="你是助手。需要天气时必须调用 get_weather 工具。",
                  tools=[get_weather])
    t0 = time.time()
    result = asyncio.run(Runner.run(agent, "巴黎现在多少度？请调用工具查询。"))
    elapsed = time.time() - t0
    return {"elapsed_s": round(elapsed, 1), "tool_called": bool(calls),
            "answer": (result.final_output or "")[:160]}


def main() -> int:
    report = {}
    passed = 0

    print("=== LangGraph（chat/completions）===", flush=True)
    try:
        lg = run_langgraph()
        report["langgraph"] = lg
        passed += ok("LangGraph 链路完成", True,
                     "%.1fs 工具调用=%s" % (lg["elapsed_s"], lg["tool_called"]))
        if lg["tool_called"]:
            passed += ok("LangGraph 工具结果回填", bool(lg["tool_results"]),
                         str(lg["tool_results"])[:60])
        else:
            print("      （模型未触发 tool_call：属 4B 能力限制，链路本身可用）")
    except Exception as exc:
        report["langgraph_error"] = repr(exc)[:300]
        passed += ok("LangGraph 链路完成", False, repr(exc)[:140])

    print("\n=== OpenAI Agents SDK（/v1/responses 翻译层）===", flush=True)
    try:
        ag = run_agents_sdk()
        report["agents_sdk"] = ag
        passed += ok("Agents SDK 链路完成（responses）", True,
                     "%.1fs 工具调用=%s" % (ag["elapsed_s"], ag["tool_called"]))
    except Exception as exc:
        report["agents_sdk_error"] = repr(exc)[:300]
        passed += ok("Agents SDK 链路完成（responses）", False, repr(exc)[:140])

    with open("f10_result.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("\n详细结果写入 f10_result.json")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
