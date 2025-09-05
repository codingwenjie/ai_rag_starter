"""文档管理相关API路由

包含文档上传、集合管理、搜索等功能的路由定义。
"""

import os
import time
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from api.models import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    CollectionRequest,
    CollectionInfo,
    SearchRequest
)
from api.dependencies import (
    get_rag_service,
    validate_auth_token,
    validate_collection_param,
    validate_upload_files,
    get_processing_config,
    get_search_config,
    get_request_timer
)
from api.utils import create_success_response, handle_exception, safe_filename


logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload", response_model=Dict[str, Any], summary="上传文档")
async def upload_documents(
    files: List[UploadFile] = Depends(validate_upload_files),
    collection_name: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    enable_summary: bool = False,
    rag_service=Depends(get_rag_service),
    timer=Depends(get_request_timer),
    user_id=Depends(validate_auth_token)
):
    """上传文档到指定集合
    
    Args:
        files: 上传的文件列表
        collection_name: 目标集合名称
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        enable_summary: 是否启用文档摘要
        rag_service: RAG服务实例
        timer: 请求计时器
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 上传结果
    """
    try:
        logger.info(f"开始上传 {len(files)} 个文档到集合: {collection_name}")
        
        # 验证集合名称
        target_collection = validate_collection_param(collection_name)
        
        # 获取处理配置
        config = get_processing_config(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_summary=enable_summary
        )
        
        # 处理文件上传
        uploaded_files = []
        total_chunks = 0
        
        for file in files:
            try:
                # 生成安全的文件名
                safe_name = safe_filename(file.filename)
                
                # 读取文件内容
                content = await file.read()
                
                # 获取文件信息
                file_info = {
                    "original_name": file.filename,
                    "safe_name": safe_name,
                    "size": len(content),
                    "content_type": file.content_type
                }
                
                # 处理文档
                result = await rag_service.add_document(
                    content=content.decode('utf-8'),
                    filename=safe_name,
                    collection_name=target_collection,
                    **config
                )
                
                file_info.update({
                    "chunks_created": result.get("chunks_created", 0),
                    "processing_status": "success"
                })
                
                total_chunks += result.get("chunks_created", 0)
                uploaded_files.append(file_info)
                
                logger.info(f"文档 {safe_name} 处理完成，创建 {result.get('chunks_created', 0)} 个文本块")
                
            except Exception as e:
                logger.error(f"处理文档 {file.filename} 失败: {e}")
                uploaded_files.append({
                    "original_name": file.filename,
                    "processing_status": "failed",
                    "error": str(e)
                })
        
        # 构建响应
        processing_time = timer.get_elapsed_time()
        successful_files = [f for f in uploaded_files if f.get("processing_status") == "success"]
        
        response_data = DocumentUploadResponse(
            success=len(successful_files) > 0,
            message=f"成功处理 {len(successful_files)}/{len(files)} 个文档",
            collection_name=target_collection,
            documents_processed=len(successful_files),
            chunks_created=total_chunks,
            processing_time=processing_time
        )
        
        logger.info(f"文档上传完成，耗时: {processing_time:.4f}秒")
        
        return create_success_response(
            data=response_data.dict(),
            message=response_data.message,
            processing_time=processing_time,
            metadata={
                "user_id": user_id,
                "files": uploaded_files,
                "config": config
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        return handle_exception(e, "文档上传")


@router.post("/collections", response_model=Dict[str, Any], summary="创建文档集合")
async def create_collection(
    request: CollectionRequest,
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """创建新的文档集合
    
    Args:
        request: 集合创建请求
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 创建结果
    """
    try:
        logger.info(f"创建文档集合: {request.collection_name}")
        
        # 创建集合
        result = await rag_service.create_collection(
            collection_name=request.collection_name,
            description=request.description
        )
        
        if result.get("success"):
            return create_success_response(
                data={
                    "collection_name": request.collection_name,
                    "description": request.description,
                    "created_at": result.get("created_at")
                },
                message="集合创建成功"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "集合创建失败")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        return handle_exception(e, "创建集合")


@router.get("/collections", response_model=Dict[str, Any], summary="获取集合列表")
async def list_collections(
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """获取所有文档集合列表
    
    Args:
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 集合列表
    """
    try:
        logger.info("获取文档集合列表")
        
        # 获取集合列表
        collections = rag_service.list_collections()
        
        return create_success_response(
            data={
                "collections": collections,
                "total": len(collections)
            },
            message="获取集合列表成功"
        )
        
    except Exception as e:
        logger.error(f"获取集合列表失败: {e}")
        return handle_exception(e, "获取集合列表")


@router.get("/collections/{collection_name}", response_model=Dict[str, Any], summary="获取集合信息")
async def get_collection_info(
    collection_name: str,
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """获取指定集合的详细信息
    
    Args:
        collection_name: 集合名称
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 集合信息
    """
    try:
        # 验证集合名称
        target_collection = validate_collection_param(collection_name)
        
        logger.info(f"获取集合信息: {target_collection}")
        
        # 获取集合信息
        info = await rag_service.get_collection_info(target_collection)
        
        if info:
            return create_success_response(
                data=info,
                message="获取集合信息成功"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="集合不存在"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取集合信息失败: {e}")
        return handle_exception(e, "获取集合信息")


@router.delete("/collections/{collection_name}", response_model=Dict[str, Any], summary="删除集合")
async def delete_collection(
    collection_name: str,
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """删除指定的文档集合
    
    Args:
        collection_name: 集合名称
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 删除结果
    """
    try:
        # 验证集合名称
        target_collection = validate_collection_param(collection_name)
        
        logger.info(f"删除集合: {target_collection}")
        
        # 删除集合
        result = await rag_service.delete_collection(target_collection)
        
        if result.get("success"):
            return create_success_response(
                data={"collection_name": target_collection},
                message="集合删除成功"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="集合不存在或删除失败"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除集合失败: {e}")
        return handle_exception(e, "删除集合")


@router.post("/search", response_model=Dict[str, Any], summary="搜索文档")
async def search_documents(
    request: SearchRequest,
    rag_service=Depends(get_rag_service),
    timer=Depends(get_request_timer),
    user_id=Depends(validate_auth_token)
):
    """在指定集合中搜索相关文档
    
    Args:
        request: 搜索请求
        rag_service: RAG服务实例
        timer: 请求计时器
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 搜索结果
    """
    try:
        logger.info(f"搜索文档: {request.query[:50]}...")
        
        # 验证集合名称
        target_collection = validate_collection_param(request.collection_name)
        
        # 获取搜索配置
        config = get_search_config(k=request.k)
        
        # 执行搜索
        results = await rag_service.search_documents(
            query=request.query,
            collection_name=target_collection,
            **config
        )
        
        # 构建响应
        processing_time = timer.get_elapsed_time()
        
        return create_success_response(
            data={
                "query": request.query,
                "collection_name": target_collection,
                "results": results,
                "total_results": len(results),
                "processing_time": processing_time
            },
            message="搜索完成",
            processing_time=processing_time,
            metadata={
                "user_id": user_id,
                "config": config
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档搜索失败: {e}")
        return handle_exception(e, "文档搜索")


@router.get("/collections/{collection_name}/documents", summary="获取集合中的文档列表")
async def list_documents_in_collection(
    collection_name: str,
    limit: int = 10,
    offset: int = 0,
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """获取指定集合中的文档列表
    
    Args:
        collection_name: 集合名称
        limit: 返回数量限制
        offset: 偏移量
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 文档列表
    """
    try:
        # 验证集合名称
        target_collection = validate_collection_param(collection_name)
        
        # 验证参数
        if limit <= 0 or limit > 100:
            raise HTTPException(
                status_code=400,
                detail="limit必须在1-100之间"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=400,
                detail="offset不能为负数"
            )
        
        logger.info(f"获取集合 {target_collection} 中的文档列表")
        
        # 获取文档列表
        documents = await rag_service.list_documents(
            collection_name=target_collection,
            limit=limit,
            offset=offset
        )
        
        return create_success_response(
            data={
                "collection_name": target_collection,
                "documents": documents,
                "limit": limit,
                "offset": offset,
                "total": len(documents)
            },
            message="获取文档列表成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return handle_exception(e, "获取文档列表")