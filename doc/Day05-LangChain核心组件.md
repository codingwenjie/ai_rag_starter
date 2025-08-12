
## 🎯 Day05 学习目标

深入掌握 LangChain 的核心组件，包括 Prompt 模板、Chain、Memory、Tool 等模块，并学会如何将它们组合成灵活的对话流程。

------

## 🧱 一、LangChain 核心模块概览

| 模块             | 作用说明                                                  |
| ---------------- | --------------------------------------------------------- |
| `PromptTemplate` | 模板化提示词，控制模型输出风格与结构                      |
| `LLM`            | 大语言模型调用接口，如 OpenAI、Anthropic、Azure 等        |
| `Chain`          | 串联多个组件形成处理流程，最核心的结构                    |
| `Memory`         | 保存上下文信息，支持多轮对话                              |
| `Tool`           | 外部工具或函数封装，使模型具备“行动能力”                  |
| `Agent`          | 模型 + 工具使用策略（例如 ReAct）组合，实现复杂任务自动化 |



------

## 🧪 二、PromptTemplate 使用详解

LangChain 提供了一种强大的提示词管理工具：`PromptTemplate`

### 示例：构建一个简单的提示模板

```python

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "你是一名资深程序员，请根据以下需求写一段 Python 代码：{需求}"
)

print(prompt.format(需求="打印前10个斐波那契数列"))
```

输出：

```
你是一名资深程序员，请根据以下需求写一段 Python 代码：打印前10个斐波那契数列
```

> ✅ 你可以将这个 PromptTemplate 作为大模型的输入，增强输出稳定性。

------

## 🔁 三、LLMChain：最常用的 Chain 结构

### 基本结构图：

```txt
PromptTemplate + LLM => LLMChain
```

### 示例代码：

```python

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 初始化模型
llm = OpenAI(temperature=0.7)

# 定义模板
template = "写一首关于{主题}的中文诗"
prompt = PromptTemplate(input_variables=["主题"], template=template)

# 构建 LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# 调用
response = chain.run("月亮")
print(response)
```

------

## 🧠 四、加入 Memory 支持上下文记忆

LangChain 中的 Memory 允许你保存对话历史，实现多轮对话。

### 示例：ConversationBufferMemory + LLMChain

```python

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0.7)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 可打印中间状态
)

print(conversation.run("你好"))
print(conversation.run("你还记得我刚才说什么吗？"))
```

> ✅ `ConversationBufferMemory` 会自动记录上下文历史，你无需手动管理。

------

## 🛠️ 五、自定义 Tool + Agent Preview（预告）

> 本节为 Day06 做准备，你今天可以简单了解：

```python

from langchain.agents import Tool

def get_weather(city):
    return f"{city} 当前温度是 30°C"

weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="获取天气信息，输入城市名"
)
```

------

## ✅ 实战练习：封装“写诗”接口到 FastAPI

你可以把 LLMChain 封装为 API 接口：

```python

# routers/poetry.py
from fastapi import APIRouter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

router = APIRouter()

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="请写一首关于{topic}的七言绝句"
)
chain = LLMChain(llm=llm, prompt=prompt)

@router.get("/poem")
def get_poem(topic: str):
    return {"poem": chain.run(topic)}
```

别忘了在 `main.py` 中注册路由：

```python

from routers import poetry
app.include_router(poetry.router, prefix="/api")
```

------

## 📚 总结

| 学习内容       | 说明                         |
| -------------- | ---------------------------- |
| PromptTemplate | 提示词模板，增强控制力       |
| LLMChain       | LLM + PromptTemplate 的组合  |
| Memory         | 保存对话历史，实现上下文记忆 |
| FastAPI 集成   | 将 LangChain 封装为 API 使用 |

