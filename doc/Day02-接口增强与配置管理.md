🎯 学习目标

通过本日学习，掌握以下 FastAPI 的高级接口处理能力：

✅ 添加 全局异常处理器 与自定义异常响应格式
✅ 对 请求参数进行校验，避免脏数据传入逻辑层
✅ 添加 CORS 跨域配置，为前后端联调铺路
✅ 使用 .env 和 dotenv 模块进行 配置管理，保障安全和灵活性

📦 所需依赖

请确保项目已经安装以下依赖（使用 pip 安装）：
```bash
pip install python-dotenv
```
🛠 操作步骤详解

✅ 步骤 1：添加全局异常处理器 

用于捕捉未被处理的异常或 Pydantic 校验异常，并统一返回格式化 JSON 响应。

创建异常模块 app/exceptions.py

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# 处理请求参数校验异常
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "请求参数错误",
            "errors": exc.errors()
        },
    )

# 捕获所有其他未处理的异常
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"服务内部错误：{str(exc)}"},
    )

```
在 main.py 中添加异常处理器

```python
from fastapi import FastAPI
from core.exceptions import validation_exception_handler, general_exception_handler

app = FastAPI()

# 添加异常处理器
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
```

✅ 步骤 2：对请求参数进行校验
依赖 Pydantic，自动校验字段是否为空、格式是否符合要求。
示例模型定义（如 schemas/chat.py）：
```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200, description="用户提问内容")

```
✅ 步骤 3：配置 CORS 中间件
允许浏览器前端访问你的后端接口，防止被跨域拦截。
在 main.py 中引入 CORS：
```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境请替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

✅ 步骤 4：使用 .env 管理配置
将 API 密钥、端口、调试模式等配置移出代码，便于统一维护。
第一步：创建 .env 文件（在项目根目录）：
```bash
# .env
OPENAI_API_KEY=your_openai_key
EMBEDDING_MODEL=text-embedding-ada-002
```
第二步：使用 python-dotenv 加载配置
在 main.py 中引入 dotenv 模块：
```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))

```
    
第三步：在代码中使用配置：

```python
from core.config import API_KEY, DEBUG

print("当前使用 API_KEY 为：", API_KEY)

```