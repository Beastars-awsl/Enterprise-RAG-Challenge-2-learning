# 第 14 章 Prompt 体系与结构化输出

> **模板族 + 双轨结构化输出**：本章拆解 `src/prompts.py` 的模板族设计
> （公共基座 + 题型继承）与 OpenAI 原生/with_schema 双轨制。
> 分治策略：把"让模型稳定输出机器可解析的答案"拆成 **行为指令（system prompt）**
> 与 **输出契约（schema）** 两个子问题，模板族再解决"五类行为指令高度重叠"的复用问题。

## 14.1 模板族的继承结构

`src/prompts.py` 的继承设计：

```text
AnswerWithRAGContextSharedPrompt        # 公共基座：CoT 指令 + {context}/{question} 占位符
├── AnswerWithRAGContextNamePrompt      # final_answer = 专名或 N/A
├── family 类，继承公共基座            # number 题型
...（names / boolean / comparative 同构）
```

公共基座承载两条全局行为指令：

1. **CoT**——先 `step_by_step_analysis`（≥5 步、≥150 词），再落 `final_answer`
2. **措辞警惕**——"上下文措辞可能与题目不同"、"题目自动生成，对某些公司可能
   无意义或不适用"——这是 NumberPrompt「看似有答案实则是相似值」条款、
   以及全族 N/A 偏向的总纲

## 14.2 双轨结构化输出

`_build_rag_context_prompts`（`src/api_requests.py:552`）：

| 厂商 | 结构化输出方式 | 使用的 prompt 变体 |
|---|---|---|
| OpenAI | 原生 response_format（beta.parse） | `system_prompt` |
| ibm / gemini | 无原生支持 → 把 pydantic 类源码文本嵌进 system prompt | `system_prompt_with_schema` |

`pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema))`——
把类源码去缩进后直接嵌进 prompt，让模型"照着 schema 写 JSON"。

新增题型必须先在 prompts 仓库注册，否则 `_build_rag_context_prompts` 抛
`ValueError`（"Unsupported schema"）——**prompt 注册表是题型体系的唯一入口**。

> **动手**
> 1. 数一数 system prompt 里每一句"防坑指令"，思考它对应什么失败模式。
> 2. 试试删掉 example（few-shot）重跑一道题，观察 step_by_step_analysis 与
>    final_answer 的稳定性变化。

> **自测（合并问题）**
> 1. instruction（行为指令）和 example（示范）各解决什么问题？删掉一个会怎样？
> 2. example 与字段描述互为校准——为什么改其中一处必须同步另一处？
> 3. with_schema 双轨制说明"结构化输出"其实由哪两个成分构成？
>    （输出格式约束 + 模型对格式的理解；前者是 API 能力，后者靠 prompt 补）
