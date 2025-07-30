from pydantic import BaseModel, Field
from typing import Optional, Any

# 请求体 schema
class ChatRequest(BaseModel):
    # 非必填 query
    query: Optional[str] = Field(None, description="想说的话", min_length=1)
    name: str = Field(..., description="物品名称", min_length=1)
    description: Optional[str] = Field(None, description="物品描述")

# 响应体 schema（统一结构）
class ResponseModel(BaseModel):
    code: int = 0
    msg: str = "success"
    data: Optional[Any] = None
