# f10_frameworks 说明

## 定位

F10 tools 闭环真机验收：用两个第三方框架各跑通一条工具调用链路。

- **LangGraph** 走 `chat/completions`（langchain-openai 的默认端点）；
- **OpenAI Agents SDK** 走 `/v1/responses`——正好验证 Responses 翻译层。

设计原文要求三种调用方式，本次无 Dify 实例，只跑前两个。

## 关键设计

板端 4B 模型可能不触发工具调用。脚本区分两类结果：

- **框架链路失败** → 缺陷，计入失败；
- **模型没吐 `tool_call`** → 模型能力限制，链路本身可用，单独标注不计失败。

## 输出

- 终端逐框架结果；
- `f10_result.json`（写在当前目录）：两框架的耗时、是否触发工具调用、
  工具结果回填内容；异常时附 `repr(exc)[:300]`。

## 用法

```bash
python f10_frameworks.py      # 需板端服务就绪，且本机已装两个框架
```

## 注意

依赖 `langgraph` 与 `openai-agents`，未安装时对应框架报链路失败；
请对照输出中的错误类型判断是缺依赖还是真缺陷。
