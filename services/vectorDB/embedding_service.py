"""Embedding服务模块

提供文本向量化功能，支持OpenAI Embedding API调用
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
from core.exceptions import EmbeddingError
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding服务类
    
    提供文本向量化功能，支持批量处理和错误重试
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        初始化Embedding服务
        
        Args:
            api_key: OpenAI API密钥，如果不提供则从环境变量获取
            model: 使用的embedding模型
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.langchain_embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=self.model
        )
        logger.info(f"初始化Embedding服务，使用模型: {model}")
    
    async def embed_text(self, text: str) -> List[float]:
        """
        对单个文本进行向量化
        
        Args:
            text: 待向量化的文本
            
        Returns:
            文本的向量表示
            
        Raises:
            EmbeddingError: 向量化失败时抛出
        """
        try:
            if not text or not text.strip():
                raise ValueError("文本不能为空")
            
            # 使用LangChain的异步方法
            vector = await self.langchain_embeddings.aembed_query(text.strip())
            logger.debug(f"成功向量化文本，向量维度: {len(vector)}")
            return vector
            
        except Exception as e:
            logger.error(f"文本向量化失败: {str(e)}")
            raise EmbeddingError(f"文本向量化失败: {str(e)}")
    
    async def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        批量文本向量化
        
        Args:
            texts: 待向量化的文本列表
            batch_size: 批处理大小
            
        Returns:
            文本向量列表
            
        Raises:
            EmbeddingError: 向量化失败时抛出
        """
        try:
            if not texts:
                return []
            
            # 过滤空文本
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                raise ValueError("没有有效的文本进行向量化")
            
            logger.info(f"开始批量向量化，文本数量: {len(valid_texts)}，批大小: {batch_size}")
            
            # 分批处理
            all_vectors = []
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                batch_vectors = await self.langchain_embeddings.aembed_documents(batch_texts)
                all_vectors.extend(batch_vectors)
                
                logger.debug(f"完成批次 {i//batch_size + 1}/{(len(valid_texts)-1)//batch_size + 1}")
                
                # 避免API限流，添加延迟
                if i + batch_size < len(valid_texts):
                    await asyncio.sleep(0.1)
            
            logger.info(f"批量向量化完成，共生成 {len(all_vectors)} 个向量")
            return all_vectors
            
        except Exception as e:
            logger.error(f"批量文本向量化失败: {str(e)}")
            raise EmbeddingError(f"批量文本向量化失败: {str(e)}")
    
    def calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            余弦相似度值 (0-1之间)
        """
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 计算余弦相似度
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            return 0.0
    
    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            余弦相似度值 (-1到1之间)
        """
        # 转换为numpy数组
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        similarity = dot_product / (norm_v1 * norm_v2)
        return float(similarity)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前使用的模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model,
            "provider": "OpenAI",
            "embedding_dimension": 1536 if "ada-002" in self.model else 1536
        }


# 全局embedding服务实例
embedding_service = EmbeddingService()


async def get_embedding_service() -> EmbeddingService:
    """
    获取embedding服务实例
    
    Returns:
        EmbeddingService实例
    """
    return embedding_service


if __name__ == "__main__":
    # 测试代码
    async def test_embedding():
        service = EmbeddingService()
        
        # 测试单个文本向量化
        text = "这是一个测试文本"
        vector = await service.embed_text(text)
        print(f"文本: {text}")
        print(f"向量维度: {len(vector)}")
        print(f"向量前5个值: {vector[:5]}")
        
        # 测试批量向量化
        texts = ["文本1", "文本2", "文本3"]
        vectors = await service.embed_texts(texts)
        print(f"\n批量向量化结果: {len(vectors)} 个向量")
        
        # 测试相似度计算
        similarity = service.calculate_similarity(vectors[0], vectors[1])
        print(f"向量1和向量2的相似度: {similarity:.4f}")
    
    # 运行测试
    asyncio.run(test_embedding())