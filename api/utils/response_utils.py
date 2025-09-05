"""响应处理工具函数

包含API响应格式化相关的工具函数。
"""

import time
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse


def create_success_response(
    data: Any = None,
    message: str = "操作成功",
    processing_time: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_code: int = 200
) -> JSONResponse:
    """创建成功响应
    
    Args:
        data: 响应数据
        message: 响应消息
        processing_time: 处理时间（秒）
        metadata: 元数据
        status_code: HTTP状态码
        
    Returns:
        JSONResponse: 格式化的成功响应
    """
    response_content = {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": int(time.time())
    }
    
    if processing_time is not None:
        response_content["processing_time"] = round(processing_time, 4)
        
    if metadata:
        response_content["metadata"] = metadata
        
    return JSONResponse(
        status_code=status_code,
        content=response_content
    )


def create_error_response(
    message: str = "操作失败",
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> JSONResponse:
    """创建错误响应
    
    Args:
        message: 错误消息
        error_code: 错误代码
        details: 错误详情
        status_code: HTTP状态码
        
    Returns:
        JSONResponse: 格式化的错误响应
    """
    response_content = {
        "success": False,
        "message": message,
        "timestamp": int(time.time())
    }
    
    if error_code:
        response_content["error_code"] = error_code
        
    if details:
        response_content["details"] = details
        
    return JSONResponse(
        status_code=status_code,
        content=response_content
    )