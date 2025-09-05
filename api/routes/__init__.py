"""API路由包

包含所有API路由模块的定义和注册。
"""

from .qa_routes import router as qa_router
from .document_routes import router as document_router

__all__ = [
    "qa_router",
    "document_router",
    "system_router"
]