"""向量数据库检索API的数据模型定义

本模块定义了向量检索相关的请求和响应数据结构，
包括参数验证、类型检查和序列化规则。
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime


class SearchType(str, Enum):
    """搜索类型枚举"""
    SIMILARITY = "similarity"  # 相似度搜索
    HYBRID = "hybrid"  # 混合搜索
    SEMANTIC = "semantic"  # 语义搜索
    KEYWORD = "keyword"  # 关键词搜索
    FUZZY = "fuzzy"  # 模糊搜索


class SortOrder(str, Enum):
    """排序方式枚举"""
    ASC = "asc"  # 升序
    DESC = "desc"  # 降序


class SortBy(str, Enum):
    """排序字段枚举"""
    SCORE = "score"  # 按相似度分数
    TIMESTAMP = "timestamp"  # 按时间戳
    RELEVANCE = "relevance"  # 按相关性


class FilterOperator(str, Enum):
    """过滤操作符枚举"""
    EQ = "eq"  # 等于
    NE = "ne"  # 不等于
    GT = "gt"  # 大于
    GTE = "gte"  # 大于等于
    LT = "lt"  # 小于
    LTE = "lte"  # 小于等于
    IN = "in"  # 包含
    NOT_IN = "not_in"  # 不包含
    CONTAINS = "contains"  # 字符串包含
    STARTS_WITH = "starts_with"  # 以...开始
    ENDS_WITH = "ends_with"  # 以...结束


class MetadataFilter(BaseModel):
    """元数据过滤条件"""
    field: str = Field(..., description="过滤字段名")
    operator: FilterOperator = Field(..., description="过滤操作符")
    value: Union[str, int, float, bool, List[Any]] = Field(..., description="过滤值")
    
    @field_validator('value')
    @classmethod
    def validate_value_type(cls, v, info):
        """验证过滤值类型是否与操作符匹配"""
        # 在Pydantic V2中，需要通过model_validator来访问其他字段
        return v
    
    @model_validator(mode='after')
    def validate_operator_value_match(self):
        """验证操作符和值的匹配性"""
        if self.operator in [FilterOperator.IN, FilterOperator.NOT_IN] and not isinstance(self.value, list):
            raise ValueError(f"操作符 {self.operator} 需要列表类型的值")
        return self


class VectorSearchRequest(BaseModel):
    """向量搜索请求模型"""
    query: str = Field(..., min_length=1, max_length=2000, description="搜索查询文本")
    search_type: SearchType = Field(default=SearchType.SIMILARITY, description="搜索类型")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    
    # 过滤条件
    metadata_filters: Optional[List[MetadataFilter]] = Field(default=None, description="元数据过滤条件")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    include_content: bool = Field(default=True, description="是否包含文档内容")
    
    # 排序选项
    sort_by: SortBy = Field(default=SortBy.SCORE, description="排序字段")
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="排序方式")
    
    # 分页参数
    offset: int = Field(default=0, ge=0, description="偏移量")
    
    # 高级选项
    rerank: bool = Field(default=False, description="是否重新排序")
    explain: bool = Field(default=False, description="是否返回解释信息")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """验证查询文本"""
        if not v.strip():
            raise ValueError("查询文本不能为空")
        return v.strip()


class BatchSearchRequest(BaseModel):
    """批量搜索请求模型"""
    queries: List[str] = Field(..., min_items=1, max_items=50, description="批量查询文本列表")
    search_type: SearchType = Field(default=SearchType.SIMILARITY, description="搜索类型")
    top_k: int = Field(default=10, ge=1, le=50, description="每个查询返回结果数量")
    threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    
    # 过滤和排序选项（应用于所有查询）
    metadata_filters: Optional[List[MetadataFilter]] = Field(default=None, description="元数据过滤条件")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    include_content: bool = Field(default=True, description="是否包含文档内容")
    
    @field_validator('queries')
    @classmethod
    def validate_queries(cls, v):
        """验证查询列表"""
        cleaned_queries = []
        for query in v:
            if not query.strip():
                raise ValueError("查询文本不能为空")
            cleaned_queries.append(query.strip())
        return cleaned_queries


class HybridSearchRequest(BaseModel):
    """混合搜索请求模型"""
    query: str = Field(..., min_length=1, max_length=2000, description="搜索查询文本")
    keywords: Optional[List[str]] = Field(default=None, description="关键词列表")
    
    # 权重配置
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="语义搜索权重")
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="关键词搜索权重")
    
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0, description="综合相似度阈值")
    
    # 其他选项
    metadata_filters: Optional[List[MetadataFilter]] = Field(default=None, description="元数据过滤条件")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    include_content: bool = Field(default=True, description="是否包含文档内容")
    
    @model_validator(mode='after')
    def validate_weights(self):
        """验证权重总和"""
        if abs(self.semantic_weight + self.keyword_weight - 1.0) > 0.001:
            raise ValueError("语义搜索权重和关键词搜索权重之和必须等于1.0")
        return self


class DocumentMatch(BaseModel):
    """文档匹配结果"""
    id: str = Field(..., description="文档ID")
    content: Optional[str] = Field(default=None, description="文档内容")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="文档元数据")
    score: float = Field(..., ge=0.0, le=1.0, description="相似度分数")
    
    # 详细信息
    rank: int = Field(..., ge=1, description="排名")
    distance: Optional[float] = Field(default=None, description="向量距离")
    
    # 解释信息
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="匹配解释")
    highlights: Optional[List[str]] = Field(default=None, description="高亮片段")


class SearchStats(BaseModel):
    """搜索统计信息"""
    total_documents: int = Field(..., ge=0, description="总文档数")
    searched_documents: int = Field(..., ge=0, description="搜索的文档数")
    matched_documents: int = Field(..., ge=0, description="匹配的文档数")
    search_time_ms: float = Field(..., ge=0, description="搜索耗时（毫秒）")
    
    # 性能指标
    vector_search_time_ms: Optional[float] = Field(default=None, description="向量搜索耗时")
    filter_time_ms: Optional[float] = Field(default=None, description="过滤耗时")
    rerank_time_ms: Optional[float] = Field(default=None, description="重排序耗时")
    
    # 缓存信息
    cache_hit: bool = Field(default=False, description="是否命中缓存")
    cache_key: Optional[str] = Field(default=None, description="缓存键")


class VectorSearchResponse(BaseModel):
    """向量搜索响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(default="搜索完成", description="响应消息")
    
    # 搜索结果
    matches: List[DocumentMatch] = Field(default=[], description="匹配结果列表")
    total_matches: int = Field(..., ge=0, description="总匹配数")
    
    # 分页信息
    offset: int = Field(default=0, ge=0, description="当前偏移量")
    limit: int = Field(..., ge=1, description="返回数量限制")
    has_more: bool = Field(default=False, description="是否有更多结果")
    
    # 统计信息
    stats: SearchStats = Field(..., description="搜索统计")
    
    # 请求信息
    query: str = Field(..., description="原始查询")
    search_type: SearchType = Field(..., description="搜索类型")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


class BatchSearchResponse(BaseModel):
    """批量搜索响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(default="批量搜索完成", description="响应消息")
    
    # 批量结果
    results: List[VectorSearchResponse] = Field(default=[], description="每个查询的搜索结果")
    total_queries: int = Field(..., ge=0, description="总查询数")
    successful_queries: int = Field(..., ge=0, description="成功查询数")
    failed_queries: int = Field(..., ge=0, description="失败查询数")
    
    # 整体统计
    total_search_time_ms: float = Field(..., ge=0, description="总搜索耗时")
    average_search_time_ms: float = Field(..., ge=0, description="平均搜索耗时")
    
    # 请求信息
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


class VectorRetrievalError(BaseModel):
    """向量检索错误模型"""
    error_code: str = Field(..., description="错误代码")
    error_type: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(default=None, description="错误详情")
    
    # 调试信息
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    trace_id: Optional[str] = Field(default=None, description="追踪ID")
    
    # 建议信息
    suggestions: Optional[List[str]] = Field(default=None, description="解决建议")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="API版本")
    
    # 组件状态
    vector_store_status: str = Field(..., description="向量存储状态")
    embedding_service_status: str = Field(..., description="嵌入服务状态")
    
    # 性能指标
    total_documents: int = Field(..., ge=0, description="总文档数")
    index_size_mb: float = Field(..., ge=0, description="索引大小（MB）")
    
    # 系统信息
    uptime_seconds: float = Field(..., ge=0, description="运行时间（秒）")
    memory_usage_mb: float = Field(..., ge=0, description="内存使用（MB）")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间戳")


# 常用的错误代码定义
class ErrorCodes:
    """错误代码常量"""
    # 请求错误 (4xx)
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_QUERY = "INVALID_QUERY"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    TOO_MANY_RESULTS = "TOO_MANY_RESULTS"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # 服务错误 (5xx)
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    EMBEDDING_SERVICE_ERROR = "EMBEDDING_SERVICE_ERROR"
    SEARCH_TIMEOUT = "SEARCH_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # 数据错误
    NO_DOCUMENTS_FOUND = "NO_DOCUMENTS_FOUND"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    CORRUPTED_INDEX = "CORRUPTED_INDEX"


# 响应示例（用于API文档）
class ResponseExamples:
    """响应示例"""
    
    SEARCH_SUCCESS = {
        "success": True,
        "message": "搜索完成",
        "matches": [
            {
                "id": "doc_001",
                "content": "Python是一种高级编程语言...",
                "metadata": {"category": "编程", "language": "中文"},
                "score": 0.95,
                "rank": 1,
                "distance": 0.05
            }
        ],
        "total_matches": 1,
        "offset": 0,
        "limit": 10,
        "has_more": False,
        "stats": {
            "total_documents": 1000,
            "searched_documents": 1000,
            "matched_documents": 1,
            "search_time_ms": 45.2,
            "cache_hit": False
        },
        "query": "什么是Python",
        "search_type": "similarity"
    }
    
    SEARCH_ERROR = {
        "error_code": "INVALID_QUERY",
        "error_type": "ValidationError",
        "message": "查询文本不能为空",
        "details": {"field": "query", "value": ""},
        "suggestions": ["请提供有效的查询文本"]
    }