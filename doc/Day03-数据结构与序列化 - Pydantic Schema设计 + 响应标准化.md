🎯 今日目标：
1. 理解 Pydantic 的基础用法
2. 编写请求参数模型（Request Schema）
3. 编写响应数据模型（Response Schema）
4. 设计统一的 API 响应结构（如 code、msg、data）
5. 快速复用统一响应格式

🧠 理论部分（基础知识）
✅ Pydantic 是什么？
Pydantic 是 FastAPI 默认使用的数据验证和序列化库，基于 Python 的 dataclass 设计，支持：
1. 类型检查（int、str、bool 等）
2. 校验约束（最小值、最大长度、正则等）
3. 自动 JSON 序列化 / 反序列化

✅ 统一响应结构设计
建议 API 的返回结构统一格式，便于前后端协作与调试，如：
```json
{
    "code": 0,
    "msg": "success",
    "data": {
        "result": "你好"
    }
}
```

🛠️ 实操步骤
步骤 1：创建 schemas.py
在项目根目录下创建文件 schema/schemas.py：
```python
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

```


📌 今日学习小结：

✅ 使用 Pydantic 编写请求与响应结构
✅ 统一所有 API 响应格式
✅ 提升了代码规范性、可维护性、和前后端协作体验