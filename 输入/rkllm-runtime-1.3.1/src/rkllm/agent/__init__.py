"""agent 能力层（S3，改造设计 §6.4）。

- conversation.py  messages → prompt（tool 历史回填，修 D6）
- reasoning.py     思考段 → reasoning_content 分离
- tools.py         提示词注入 + 多模板流式解析（不碰 rkllm_set_function_tools）
- pipeline.py      推理管线：把上述能力串进 server 的 token 流
"""
