from pydantic import BaseModel, Field


# 解释这个类
# 这个类定义了一个 POST 请求的参数模型，包含一个 query 字段，类型为字符串。
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="用户提问内容")
