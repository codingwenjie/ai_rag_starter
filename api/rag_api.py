"""RAG API主文件

重构后的RAG API主文件，只保留路由注册和基本配置。
所有具体的业务逻辑已迁移到相应的模块中。
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

# 导入重构后的模块
from .routes import qa_router, document_router

# 配置日志
logger = logging.getLogger(__name__)

# 创建主路由器
router = APIRouter(prefix="/rag", tags=["RAG"])

# 全局统计信息
api_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "start_time": datetime.now().isoformat()
}

# 全局RAG服务实例存储
rag_services: Dict[str, Any] = {}
qa_components: Dict[str, Any] = {}


@router.get("/", summary="RAG API根路径")
async def root() -> JSONResponse:
    """RAG API根路径接口
    
    提供API基本信息和状态。
    
    Returns:
        JSONResponse: API基本信息
    """
    try:
        api_info = {
            "name": "RAG API",
            "version": "2.0.0",
            "description": "基于检索增强生成(RAG)的API接口",
            "status": "running",
            "start_time": api_stats["start_time"],
            "current_time": datetime.now().isoformat(),
            "features": [
                "文档上传与处理",
                "智能问答查询",
                "对话式交互",
                "文档集合管理",
                "向量搜索"
            ],
            "endpoints": {
                "问答相关": "/rag/ask, /rag/conversation",
                "文档管理": "/rag/upload, /rag/collections, /rag/search"
            }
        }
        
        return JSONResponse(
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"获取API信息失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "获取API信息失败",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# 注册子路由
router.include_router(qa_router, prefix="", tags=["问答服务"])
router.include_router(document_router, prefix="", tags=["文档管理"])

# 导出路由器和全局变量
__all__ = ["router", "rag_services", "qa_components", "api_stats"]
