"""问答相关API路由

包含简单问答、检索问答、对话问答等功能的路由定义。
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from api.models import (
    QuestionRequest,
    QuestionResponse,
    ConversationRequest,
    ConversationResponse
)
from api.dependencies import (
    get_rag_service,
    validate_auth_token,
    validate_collection_param,
    get_request_timer,
    validate_conversation_id,
    validate_custom_prompt
)
from api.utils import create_success_response, handle_exception

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/qa", tags=["问答"])


@router.post("/ask", response_model=Dict[str, Any], summary="简单问答")
async def ask_question(
    request: QuestionRequest,
    rag_service=Depends(get_rag_service),
    timer=Depends(get_request_timer),
    user_id=Depends(validate_auth_token),
    collection_name: str = Depends(validate_collection_param)
):
    """处理用户问题并返回答案
    
    Args:
        request: 问题请求
        rag_service: RAG服务实例
        timer: 请求计时器
        user_id: 用户ID（可选）
        collection_name: 集合名称
        
    Returns:
        Dict[str, Any]: 问答响应
    """
    try:
        logger.info(f"收到问答请求: {request.question[:50]}...")
        
        # 使用请求中的集合名称（如果提供）
        target_collection = request.collection_name or collection_name
        
        # 验证自定义提示模板
        custom_prompt = validate_custom_prompt(request.custom_prompt)
        
        # 根据问答类型选择处理方式
        if request.qa_type == "simple":
            result = await rag_service.simple_qa(
                question=request.question,
                collection_name=target_collection
            )
        elif request.qa_type == "retrieval":
            result = await rag_service.retrieval_qa(
                question=request.question,
                collection_name=target_collection,
                k=request.k,
                custom_prompt=custom_prompt
            )
        elif request.qa_type == "conversational":
            result = await rag_service.conversational_qa(
                question=request.question,
                collection_name=target_collection,
                k=request.k
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的问答类型: {request.qa_type}"
            )
        
        # 构建响应
        processing_time = timer.get_elapsed_time()
        
        response_data = QuestionResponse(
            question=request.question,
            answer=result.get("answer", ""),
            source_documents=result.get("source_documents", []),
            collection_name=target_collection,
            qa_type=request.qa_type,
            processing_time=processing_time,
            metadata={
                "user_id": user_id,
                "k": request.k,
                "use_conversational": request.use_conversational,
                "custom_prompt_used": custom_prompt is not None
            }
        )
        
        logger.info(f"问答处理完成，耗时: {processing_time:.4f}秒")
        
        return create_success_response(
            data=response_data.dict(),
            message="问答处理成功",
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        return handle_exception(e, "问答处理")


@router.post("/conversation", response_model=Dict[str, Any], summary="对话问答")
async def conversation_qa(
    request: ConversationRequest,
    rag_service=Depends(get_rag_service),
    timer=Depends(get_request_timer),
    user_id=Depends(validate_auth_token),
    collection_name: str = Depends(validate_collection_param)
):
    """处理对话式问答
    
    Args:
        request: 对话请求
        rag_service: RAG服务实例
        timer: 请求计时器
        user_id: 用户ID（可选）
        collection_name: 集合名称
        
    Returns:
        Dict[str, Any]: 对话响应
    """
    try:
        logger.info(f"收到对话请求: {request.question[:50]}...")
        
        # 使用请求中的集合名称（如果提供）
        target_collection = request.collection_name or collection_name
        
        # 验证对话ID
        conversation_id = validate_conversation_id(request.conversation_id)
        
        # 处理对话
        result = await rag_service.conversational_qa(
            question=request.question,
            collection_name=target_collection,
            conversation_id=conversation_id,
            k=request.k
        )
        
        # 构建响应
        processing_time = timer.get_elapsed_time()
        
        response_data = ConversationResponse(
            question=request.question,
            answer=result.get("answer", ""),
            conversation_id=result.get("conversation_id", conversation_id or "new"),
            turn_number=result.get("turn_number", 1),
            source_documents=result.get("source_documents", []),
            chat_history=result.get("chat_history", []),
            processing_time=processing_time
        )
        
        logger.info(f"对话处理完成，耗时: {processing_time:.4f}秒")
        
        return create_success_response(
            data=response_data.dict(),
            message="对话处理成功",
            processing_time=processing_time,
            metadata={
                "user_id": user_id,
                "collection_name": target_collection
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话处理失败: {e}")
        return handle_exception(e, "对话处理")


@router.get("/conversation/{conversation_id}/history", summary="获取对话历史")
async def get_conversation_history(
    conversation_id: str,
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """获取指定对话的历史记录
    
    Args:
        conversation_id: 对话ID
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 对话历史
    """
    try:
        # 验证对话ID
        validated_id = validate_conversation_id(conversation_id)
        if not validated_id:
            raise HTTPException(
                status_code=400,
                detail="无效的对话ID"
            )
        
        # 获取对话历史
        history = await rag_service.get_conversation_history(validated_id)
        
        return create_success_response(
            data={
                "conversation_id": validated_id,
                "history": history,
                "total_turns": len(history)
            },
            message="获取对话历史成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        return handle_exception(e, "获取对话历史")


@router.delete("/conversation/{conversation_id}", summary="删除对话")
async def delete_conversation(
    conversation_id: str,
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """删除指定的对话
    
    Args:
        conversation_id: 对话ID
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 删除结果
    """
    try:
        # 验证对话ID
        validated_id = validate_conversation_id(conversation_id)
        if not validated_id:
            raise HTTPException(
                status_code=400,
                detail="无效的对话ID"
            )
        
        # 删除对话
        success = await rag_service.delete_conversation(validated_id)
        
        if success:
            return create_success_response(
                data={"conversation_id": validated_id},
                message="对话删除成功"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="对话不存在或删除失败"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话失败: {e}")
        return handle_exception(e, "删除对话")


@router.get("/conversations", summary="获取对话列表")
async def list_conversations(
    limit: int = 10,
    offset: int = 0,
    rag_service=Depends(get_rag_service),
    user_id=Depends(validate_auth_token)
):
    """获取对话列表
    
    Args:
        limit: 返回数量限制
        offset: 偏移量
        rag_service: RAG服务实例
        user_id: 用户ID（可选）
        
    Returns:
        Dict[str, Any]: 对话列表
    """
    try:
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
        
        # 获取对话列表
        conversations = await rag_service.list_conversations(
            limit=limit,
            offset=offset,
            user_id=user_id
        )
        
        return create_success_response(
            data={
                "conversations": conversations,
                "limit": limit,
                "offset": offset,
                "total": len(conversations)
            },
            message="获取对话列表成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话列表失败: {e}")
        return handle_exception(e, "获取对话列表")