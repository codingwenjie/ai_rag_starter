"""RAG API依赖注入函数

包含RAG服务实例获取、文件上传处理等依赖注入功能。"""

import os
import time
import logging
from typing import Optional, List
from fastapi import Depends, HTTPException, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


logger = logging.getLogger(__name__)

# 全局RAG服务实例
_rag_service = None

# 安全认证（可选）
security = HTTPBearer(auto_error=False)


def get_rag_service():
    """获取RAG服务实例
    
    Returns:
        RAGService: RAG服务实例
        
    Raises:
        HTTPException: 当服务未初始化时
    """
    global _rag_service
    
    if _rag_service is None:
        try:
            from services.rag.rag_service import RAGService
            _rag_service = RAGService()
            logger.info("RAG服务实例已创建")
        except Exception as e:
            logger.error(f"创建RAG服务实例失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG服务初始化失败: {str(e)}"
            )
    
    return _rag_service


def set_rag_service(service):
    """设置RAG服务实例（用于测试或自定义配置）
    
    Args:
        service: RAG服务实例
    """
    global _rag_service
    _rag_service = service
    logger.info("RAG服务实例已更新")


def validate_auth_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """验证认证令牌（可选功能）
    
    Args:
        credentials: HTTP认证凭据
        
    Returns:
        str: 用户ID或None
        
    Raises:
        HTTPException: 当令牌无效时
    """
    # 如果没有配置认证，直接返回
    auth_required = os.getenv("RAG_AUTH_REQUIRED", "false").lower() == "true"
    if not auth_required:
        return None
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="需要认证令牌"
        )
    
    # 这里可以添加实际的令牌验证逻辑
    # 例如：JWT验证、API密钥验证等
    token = credentials.credentials
    expected_token = os.getenv("RAG_API_TOKEN")
    
    if expected_token and token != expected_token:
        raise HTTPException(
            status_code=401,
            detail="无效的认证令牌"
        )
    
    return "authenticated_user"  # 返回用户ID


def validate_collection_param(collection_name: Optional[str] = None) -> str:
    """验证和清理集合名称参数
    
    Args:
        collection_name: 集合名称
        
    Returns:
        str: 清理后的集合名称
    """
    if not collection_name:
        return "default"


def validate_file_extension(filename: str) -> bool:
    """验证文件扩展名
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 是否为支持的文件类型
    """
    if not filename:
        return False
        
    # 支持的文件扩展名
    allowed_extensions = {
        '.txt', '.md', '.html', '.htm', '.pdf', 
        '.doc', '.docx', '.rtf', '.csv', '.json'
    }
    
    # 获取文件扩展名（转为小写）
    file_ext = os.path.splitext(filename.lower())[1]
    
    return file_ext in allowed_extensions


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """验证文件大小
    
    Args:
        file_size: 文件大小（字节）
        max_size_mb: 最大文件大小（MB）
        
    Returns:
        bool: 文件大小是否在允许范围内
    """
    max_size_bytes = max_size_mb * 1024 * 1024  # 转换为字节
    return file_size <= max_size_bytes
    
    return sanitize_collection_name(collection_name)


def validate_upload_files(files: List[UploadFile] = File(...)) -> List[UploadFile]:
    """验证上传的文件
    
    Args:
        files: 上传的文件列表
        
    Returns:
        List[UploadFile]: 验证通过的文件列表
        
    Raises:
        HTTPException: 当文件验证失败时
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="没有上传任何文件"
        )
    
    validated_files = []
    
    for file in files:
        # 检查文件名
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="文件名不能为空"
            )
        
        # 检查文件扩展名
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.filename}"
            )
        
        # 检查文件大小（如果可获取）
        if hasattr(file, 'size') and file.size is not None:
            if not validate_file_size(file.size):
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超出限制: {file.filename}"
                )
        
        validated_files.append(file)
    
    return validated_files


def get_processing_config(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    enable_summary: bool = False
) -> dict:
    """获取文档处理配置
    
    Args:
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        enable_summary: 是否启用摘要
        
    Returns:
        dict: 处理配置
        
    Raises:
        HTTPException: 当配置参数无效时
    """
    
    # 使用默认值
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = DEFAULT_CHUNK_OVERLAP
    
    # 验证参数
    if chunk_size <= 0 or chunk_size > 5000:
        raise HTTPException(
            status_code=400,
            detail="chunk_size必须在1-5000之间"
        )
    
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap必须在0到chunk_size之间"
        )
    
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "enable_summary": enable_summary
    }


def get_search_config(
    k: Optional[int] = None,
    score_threshold: Optional[float] = None
) -> dict:
    """获取搜索配置
    
    Args:
        k: 返回结果数量
        score_threshold: 相似度阈值
        
    Returns:
        dict: 搜索配置
        
    Raises:
        HTTPException: 当配置参数无效时
    """
    # 使用默认值
    if k is None:
        k = 4
    if score_threshold is None:
        score_threshold = 0.7
    
    # 验证参数
    if k <= 0 or k > 20:
        raise HTTPException(
            status_code=400,
            detail="k必须在1-20之间"
        )
    
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise HTTPException(
            status_code=400,
            detail="score_threshold必须在0.0-1.0之间"
        )
    
    return {
        "k": k,
        "score_threshold": score_threshold
    }


class RequestTimer:
    """请求计时器依赖"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """获取已用时间"""
        return time.time() - self.start_time


def get_request_timer() -> RequestTimer:
    """获取请求计时器实例
    
    Returns:
        RequestTimer: 计时器实例
    """
    return RequestTimer()


def validate_conversation_id(conversation_id: Optional[str] = None) -> Optional[str]:
    """验证对话ID
    
    Args:
        conversation_id: 对话ID
        
    Returns:
        Optional[str]: 验证后的对话ID
        
    Raises:
        HTTPException: 当对话ID格式无效时
    """
    if not conversation_id:
        return None
    
    # 验证对话ID格式
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', conversation_id):
        raise HTTPException(
            status_code=400,
            detail="对话ID只能包含字母、数字、下划线和连字符"
        )
    
    if len(conversation_id) > 100:
        raise HTTPException(
            status_code=400,
            detail="对话ID长度不能超过100个字符"
        )
    
    return conversation_id


def get_rate_limiter():
    """获取速率限制器（占位符实现）
    
    Returns:
        None: 当前未实现速率限制
    """
    # 这里可以添加速率限制逻辑
    # 例如：基于IP的请求频率限制
    return None


def validate_custom_prompt(custom_prompt: Optional[str] = None) -> Optional[str]:
    """验证自定义提示模板
    
    Args:
        custom_prompt: 自定义提示模板
        
    Returns:
        Optional[str]: 验证后的提示模板
        
    Raises:
        HTTPException: 当提示模板无效时
    """
    if not custom_prompt:
        return None
    
    # 检查长度
    if len(custom_prompt) > 2000:
        raise HTTPException(
            status_code=400,
            detail="自定义提示模板长度不能超过2000个字符"
        )
    
    # 检查是否包含必要的占位符
    required_placeholders = ['{context}', '{question}']
    missing_placeholders = [p for p in required_placeholders if p not in custom_prompt]
    
    if missing_placeholders:
        raise HTTPException(
            status_code=400,
            detail=f"自定义提示模板必须包含以下占位符: {', '.join(missing_placeholders)}"
        )
    
    return custom_prompt.strip()