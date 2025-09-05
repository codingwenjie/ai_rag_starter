"""RAG API数据模型定义

包含所有RAG相关的Pydantic模型定义，用于API请求和响应的数据验证。
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class RAGConfigModel(BaseModel):
    """RAG配置模型"""
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="文本块大小")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="文本块重叠大小")
    embedding_model: str = Field(default="text-embedding-ada-002", description="嵌入模型")
    llm_model: str = Field(default="gpt-3.5-turbo", description="语言模型")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=1000, ge=100, le=4000, description="最大生成token数")
    retrieval_k: int = Field(default=4, ge=1, le=20, description="检索文档数量")
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """验证chunk_overlap不能大于chunk_size"""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap必须小于chunk_size')
        return v
    
    class Config:
        """Pydantic配置"""
        json_schema_extra = {
            "example": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embedding_model": "text-embedding-ada-002",
                "llm_model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 1000,
                "retrieval_k": 4,
                "score_threshold": 0.7
            }
        }


class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    collection_name: str = Field(default="default", pattern=r'^[a-zA-Z0-9_-]+$', description="文档集合名称")
    use_conversational: bool = Field(default=False, description="是否使用对话模式")
    qa_type: str = Field(default="simple", description="问答类型: simple, retrieval, conversational")
    custom_prompt: Optional[str] = Field(None, max_length=2000, description="自定义提示模板")
    k: int = Field(default=4, ge=1, le=20, description="检索文档数量")
    
    @validator('qa_type')
    def validate_qa_type(cls, v):
        """验证问答类型"""
        allowed_types = {'simple', 'retrieval', 'conversational'}
        if v not in allowed_types:
            raise ValueError(f'qa_type必须是以下之一: {", ".join(allowed_types)}')
        return v
    
    @validator('question')
    def validate_question(cls, v):
        """验证问题内容"""
        if not v.strip():
            raise ValueError('问题不能为空或只包含空白字符')
        return v.strip()
    
    class Config:
        """Pydantic配置"""
        json_schema_extra = {
            "example": {
                "question": "什么是Python?",
                "collection_name": "default",
                "qa_type": "simple",
                "k": 4
            }
        }


class QuestionResponse(BaseModel):
    """问答响应模型"""
    question: str
    answer: str
    source_documents: List[Dict[str, Any]]
    collection_name: str
    qa_type: str
    processing_time: float
    metadata: Dict[str, Any]


class ConversationRequest(BaseModel):
    """对话请求模型"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    collection_name: str = Field(default="default", pattern=r'^[a-zA-Z0-9_-]+$', description="文档集合名称")
    conversation_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9_-]+$', description="对话ID")
    k: int = Field(default=4, ge=1, le=20, description="检索文档数量")
    
    @validator('question')
    def validate_question(cls, v):
        """验证问题内容"""
        if not v.strip():
            raise ValueError('问题不能为空或只包含空白字符')
        return v.strip()


class ConversationResponse(BaseModel):
    """对话响应模型"""
    question: str
    answer: str
    conversation_id: str
    turn_number: int
    source_documents: List[Dict[str, Any]]
    chat_history: List[Dict[str, str]]
    processing_time: float


class DocumentUploadRequest(BaseModel):
    """文档上传请求模型"""
    collection_name: str = Field(default="default", pattern=r'^[a-zA-Z0-9_-]+$', description="文档集合名称")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="文本块大小")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="文本块重叠大小")
    enable_summary: bool = Field(default=False, description="是否启用文档摘要")
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """验证chunk_overlap不能大于chunk_size"""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap必须小于chunk_size')
        return v


class DocumentUploadResponse(BaseModel):
    """文档上传响应模型"""
    success: bool
    message: str
    collection_name: Optional[str] = "default"
    documents_processed: int
    chunks_created: int
    processing_time: float


class CollectionRequest(BaseModel):
    """集合请求模型"""
    collection_name: str = Field(..., min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$', description="集合名称")
    description: Optional[str] = Field(None, max_length=500, description="集合描述")
    
    @validator('collection_name')
    def validate_collection_name(cls, v):
        """验证集合名称"""
        if not v.strip():
            raise ValueError('集合名称不能为空或只包含空白字符')
        return v.strip().lower()  # 统一转为小写


class CollectionInfo(BaseModel):
    """文档集合信息模型"""
    name: str
    document_count: int
    chunk_count: int
    created_at: str
    last_updated: str
    config: Dict[str, Any]


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., min_length=1, max_length=1000, description="搜索查询")
    collection_name: str = Field(default="default", pattern=r'^[a-zA-Z0-9_-]+$', description="集合名称")
    k: int = Field(default=4, ge=1, le=20, description="返回结果数量")
    
    @validator('query')
    def validate_query(cls, v):
        """验证查询内容"""
        if not v.strip():
            raise ValueError('查询不能为空或只包含空白字符')
        return v.strip()


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    version: str
    uptime: float
    components: Dict[str, Any]
    system_info: Dict[str, Any]


class MetricsResponse(BaseModel):
    """性能指标响应模型"""
    timestamp: str
    system: Dict[str, Any]
    services: Dict[str, Any]
    memory: Dict[str, Any]


class StatsResponse(BaseModel):
    """统计信息响应模型"""
    requests: Dict[str, Any]
    uptime: Dict[str, Any]
    services: Dict[str, Any]