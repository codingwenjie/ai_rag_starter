"""向量存储服务模块

基于FAISS实现向量存储、检索和相似度搜索功能
"""

import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .embedding_service import EmbeddingService
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStoreService:
    """向量存储服务类
    
    提供基于FAISS的向量存储、检索和管理功能
    """
    
    def __init__(self, embedding_service: EmbeddingService, index_path: str = "./data/vector_index"):
        """
        初始化向量存储服务
        
        Args:
            embedding_service: embedding服务实例
            index_path: 索引文件存储路径
        """
        self.embedding_service = embedding_service
        self.index_path = index_path
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []
        self.metadata: Dict[str, Any] = {}
        
        # 确保存储目录存在
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        logger.info(f"初始化向量存储服务，索引路径: {index_path}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
            
        Returns:
            文档ID列表
        """
        try:
            if not documents:
                return []
            
            logger.info(f"开始添加 {len(documents)} 个文档到向量存储")
            
            # 提取文档文本
            texts = [doc.page_content for doc in documents]
            
            # 生成向量
            vectors = await self.embedding_service.embed_texts(texts)
            
            # 创建或更新向量存储
            if self.vector_store is None:
                # 首次创建
                self.vector_store = FAISS.from_documents(
                    documents, 
                    self.embedding_service.langchain_embeddings
                )
                logger.info("创建新的FAISS向量存储")
            else:
                # 添加到现有存储
                self.vector_store.add_documents(documents)
                logger.info(f"向现有向量存储添加 {len(documents)} 个文档")
            
            # 更新文档列表
            self.documents.extend(documents)
            
            # 生成文档ID
            doc_ids = [f"doc_{len(self.documents) - len(documents) + i}" for i in range(len(documents))]
            
            # 更新元数据
            self.metadata.update({
                "total_documents": len(self.documents),
                "last_updated": datetime.now().isoformat(),
                "vector_dimension": len(vectors[0]) if vectors else 0
            })
            
            logger.info(f"成功添加文档，当前总数: {len(self.documents)}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            (文档, 相似度分数)元组列表
        """
        try:
            if self.vector_store is None:
                logger.warning("向量存储为空，无法进行搜索")
                return []
            
            logger.info(f"执行相似度搜索，查询: '{query[:50]}...', Top-{k}")
            
            # 执行相似度搜索
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # 过滤低于阈值的结果
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            
            logger.info(f"搜索完成，返回 {len(filtered_results)} 个结果")
            return filtered_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []
    
    async def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        语义搜索，返回格式化结果
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            results = await self.similarity_search(query, k)
            
            formatted_results = []
            for i, (doc, score) in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "relevance": "高" if score > 0.8 else "中" if score > 0.6 else "低"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"语义搜索失败: {str(e)}")
            return []
    
    def save_index(self, path: Optional[str] = None) -> bool:
        """
        保存向量索引到磁盘
        
        Args:
            path: 保存路径，默认使用初始化时的路径
            
        Returns:
            是否保存成功
        """
        try:
            if self.vector_store is None:
                logger.warning("没有向量存储可保存")
                return False
            
            save_path = path or self.index_path
            
            # 保存FAISS索引
            self.vector_store.save_local(save_path)
            
            # 保存元数据
            metadata_path = f"{save_path}/metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"向量索引已保存到: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存向量索引失败: {str(e)}")
            return False
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """
        从磁盘加载向量索引
        
        Args:
            path: 加载路径，默认使用初始化时的路径
            
        Returns:
            是否加载成功
        """
        try:
            load_path = path or self.index_path
            
            if not os.path.exists(load_path):
                logger.warning(f"索引路径不存在: {load_path}")
                return False
            
            # 加载FAISS索引
            self.vector_store = FAISS.load_local(
                load_path, 
                self.embedding_service.langchain_embeddings,
                allow_dangerous_deserialization=True
            )
            
            # 加载元数据
            metadata_path = f"{load_path}/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"向量索引已从 {load_path} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载向量索引失败: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_documents": len(self.documents),
            "has_index": self.vector_store is not None,
            "index_path": self.index_path,
            **self.metadata
        }
        
        if self.vector_store is not None:
            stats["index_size"] = self.vector_store.index.ntotal
        
        return stats
    
    def clear(self) -> None:
        """
        清空向量存储
        """
        self.vector_store = None
        self.documents = []
        self.metadata = {}
        logger.info("向量存储已清空")