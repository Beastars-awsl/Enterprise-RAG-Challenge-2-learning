# 第 16 章 API 层与并发

> **本章目标**：理解三厂商适配与限速并发请求队列；体会"协议适配"与
> "并发调度"的分离。
> 分治策略：`APIProcessor` 只做选择，`send_message` 只做执行，
> 并发问题交给独立的并行处理器模块。

## 16.1 策略模式：三厂商适配

`src/api_requests.py:477` 的 `APIProcessor.__init__`：

```python
self.provider = provider.lower()
if self.provider == "openai":
    self.processor = BaseOpenaiProcessor()
elif self.provider == "ibm":
    self.processor = BaseIBMAPIProcessor()
elif self.provider == "gemini":
    self.processor = BaseGeminiProcessor()
```

`_build_rag_context_prompts`（第 552 行）按 schema 查 prompt 仓库（第 14 章），
决定用哪套 system prompt 与 response_format；ibm/gemini 走
`*_with_schema` 变体——把 pydantic 类源码文本嵌进 system prompt，
让模型"照着 schema 写 JSON"。

新增厂商/题型的改动面：processor 类 + `_build_rag_context_prompts`
的 if-elif 表。没有注册表/装饰器，**比赛代码选择直白而非优雅**。

## 16.2 并发模型

`src/api_request_parallel_processor.py`：asyncio 队列 + 限速，
单进程内高吞吐跑批（比赛跑分时靠它扫完整个题目集）。QuestionsProcessor 的
`parallel_requests` 开关决定单发或并发跑批。

与第 3 章解析的**多进程**对照：解析是 CPU/GPU 密集（docling 模型推理），
用多进程绕开 GIL；API 调用是 I/O 密集，用 asyncio——**按瓶颈类型选并发模型**。
另外第 10 章单块打分用了 `ThreadPoolExecutor`——一个项目里三种并发原语各有分工。

> **动手**
> 1. 观察 `send_message` 后 `response_data` 记录了什么（token 用量、原始响应），
>    思考 debug 文件里的 token 统计从哪来。
> 2. 思考：为什么并行处理器是独立模块而不塞进 api_requests.py？

> **自测（合并问题）**
> 1. 新增一个厂商（如 Anthropic）需要改哪几处？
> 2. asyncio 队列 + 限速 vs 简单 ThreadPoolExecutor——什么规模下值得上 asyncio 方案？
