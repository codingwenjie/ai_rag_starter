## 🧠 第一步：LangChain 原理概述

### 什么是 LangChain？

LangChain 是一个用于构建语言模型驱动应用的框架。它将 LLM（如 OpenAI GPT、Claude、Llama）与工具链（如搜索、数据库、API）进行组合，帮助开发者更容易地构建 RAG、Agent 等复杂的应用。

### LangChain 核心组件

| 模块       | 功能                                                     |
| ---------- | -------------------------------------------------------- |
| **LLM**    | 调用语言模型，如 OpenAI、Anthropic、Claude、Local LLM 等 |
| **Prompt** | 提示词模板引擎，动态构建对话内容                         |
| **Chains** | 链式逻辑执行流（输入 → 处理 → 输出）                     |
| **Agents** | 智能体，具有决策能力，能调用多个工具来完成复杂任务       |
| **Memory** | 记忆模块，用于保存历史对话等上下文信息                   |
| **Tools**  | 接入外部能力：搜索、SQL 查询、向量库、Web API 等         |

## 🔧 第二步：实战演练（在你的项目中接入 LangChain）

### ✅ 目标：

1. 接入 LangChain + OpenAI 模型
2. 编写一个简单接口，用户提问 → 返回 LLM 回答
3. 使用 `.env` 管理 API 密钥
4. 使用 Pydantic 校验输入参数

------

## 📦 安装依赖

你当前项目环境已启用 `uvicorn + FastAPI`，只需安装：

```
bash


复制编辑
pip install langchain openai
```

------

## 📁 项目结构变更（简要）

```
bash


复制编辑
ai_rag_starter/
├── main.py
├── .env
├── services/
│   └── llm_service.py   👈 新增
├── schemas/
│   └── llm_schema.py    👈 新增
└── ...
```

------

## 🗂️ 第一步：配置 `.env` 和加载

`.env` 文件：

```
env


复制编辑
OPENAI_API_KEY=sk-xxx
```

加载环境变量（main.py 中加入）：

```
python


复制编辑
from dotenv import load_dotenv
load_dotenv()
```

------

## ✍️ 第二步：编写调用模型服务（services/llm_service.py）

```
python


复制编辑
from langchain.chat_models import ChatOpenAI
import os

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(question: str) -> str:
    return llm.predict(question)
```

------

## ✨ 第三步：定义请求/响应 Schema（schemas/llm_schema.py）

```
python


复制编辑
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
```

------

## 🚀 第四步：编写接口（main.py 中新增）

```
python


复制编辑
from fastapi import FastAPI
from schemas.llm_schema import QuestionRequest, AnswerResponse
from services.llm_service import ask_llm

app = FastAPI()

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(data: QuestionRequest):
    answer = ask_llm(data.question)
    return AnswerResponse(answer=answer)
```

------

## 🧪 测试

你可以用 curl、Postman 或 FastAPI 的 Swagger UI：

```
http


复制编辑
POST http://localhost:8000/ask
Content-Type: application/json

{
  "question": "你是谁？"
}
```

