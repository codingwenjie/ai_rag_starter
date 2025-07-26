from pydantic import BaseModel

# 解释这个类
# 这个类定义了一个 POST 请求的参数模型，包含一个 query 字段，类型为字符串。
class ChatRequest(BaseModel):
    query: str
