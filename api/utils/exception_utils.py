"""异常处理工具函数

包含API异常处理相关的工具函数。
"""

import logging
from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from core.exceptions import EmbeddingError, VectorStoreError
from .response_utils import create_error_response

logger = logging.getLogger(__name__)


def handle_exception(exception: Exception, operation: str = "操作") -> JSONResponse:
    """统一异常处理函数
    
    Args:
        exception: 异常对象
        operation: 操作描述
        
    Returns:
        JSONResponse: 格式化的错误响应
    """
    # 记录异常日志
    logger.error(f"{operation}失败: {str(exception)}", exc_info=True)
    
    # 根据异常类型返回不同的错误响应
    if isinstance(exception, HTTPException):
        return create_error_response(
            message=exception.detail,
            error_code="HTTP_ERROR",
            status_code=exception.status_code
        )
    elif isinstance(exception, EmbeddingError):
        return create_error_response(
            message=f"向量化处理失败: {str(exception)}",
            error_code="EMBEDDING_ERROR",
            status_code=500
        )
    elif isinstance(exception, VectorStoreError):
        return create_error_response(
            message=f"向量存储操作失败: {str(exception)}",
            error_code="VECTOR_STORE_ERROR",
            status_code=500
        )
    elif isinstance(exception, ValueError):
        return create_error_response(
            message=f"参数错误: {str(exception)}",
            error_code="VALUE_ERROR",
            status_code=400
        )
    elif isinstance(exception, FileNotFoundError):
        return create_error_response(
            message=f"文件未找到: {str(exception)}",
            error_code="FILE_NOT_FOUND",
            status_code=404
        )
    elif isinstance(exception, PermissionError):
        return create_error_response(
            message=f"权限不足: {str(exception)}",
            error_code="PERMISSION_ERROR",
            status_code=403
        )
    else:
        # 通用异常处理
        return create_error_response(
            message=f"{operation}失败: {str(exception)}",
            error_code="INTERNAL_ERROR",
            details={
                "exception_type": type(exception).__name__,
                "operation": operation
            },
            status_code=500
        )