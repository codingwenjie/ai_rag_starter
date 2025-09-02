from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY


# 自定义异常类
class EmbeddingError(Exception):
    """向量化相关异常"""
    pass


class VectorStoreError(Exception):
    """向量存储相关异常"""
    pass


# 处理请求参数检验异常
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"message": "请求参数校验失败", "errors": exc.errors()},
    )

# 捕获所有其他未处理异常
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"服务器内部错误:{str(exc)}"},
    )