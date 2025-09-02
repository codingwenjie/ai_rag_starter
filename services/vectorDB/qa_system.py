"""问答系统服务模块

整合LangChain + FAISS构建简单问答系统
"""

from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
import logging

logger = logging.getLogger(__name__)


class QASystem:
    """问答系统类
    
    基于向量检索的智能问答系统
    """
    
    def __init__(self, embedding_service: EmbeddingService, vector_store_service: VectorStoreService):
        """
        初始化问答系统
        
        Args:
            embedding_service: embedding服务实例
            vector_store_service: 向量存储服务实例
        """
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500
        )
        self.qa_chain: Optional[RetrievalQA] = None
        
        logger.info("问答系统初始化完成")
    
    def setup_qa_chain(self) -> bool:
        """
        设置问答链
        
        Returns:
            是否设置成功
        """
        try:
            if self.vector_store_service.vector_store is None:
                logger.error("向量存储为空，无法设置问答链")
                return False
            
            # 创建检索器
            retriever = self.vector_store_service.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # 创建问答链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("问答链设置成功")
            return True
            
        except Exception as e:
            logger.error(f"设置问答链失败: {str(e)}")
            return False
    
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        提问并获取答案
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案和相关信息的字典
        """
        try:
            if self.qa_chain is None:
                if not self.setup_qa_chain():
                    return {
                        "answer": "抱歉，问答系统尚未准备就绪。",
                        "sources": [],
                        "error": "问答链未设置"
                    }
            
            logger.info(f"处理问题: {question}")
            
            # 执行问答
            result = await self.qa_chain.ainvoke({"query": question})
            
            # 格式化结果
            response = {
                "question": question,
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ],
                "confidence": "高" if len(result.get("source_documents", [])) >= 2 else "中"
            }
            
            logger.info(f"问答完成，答案长度: {len(response['answer'])}")
            return response
            
        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
            return {
                "question": question,
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    async def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        批量问答
        
        Args:
            questions: 问题列表
            
        Returns:
            答案列表
        """
        results = []
        for question in questions:
            result = await self.ask(question)
            results.append(result)
        
        return results
    
    async def get_relevant_docs(self, query: str, k: int = 3) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            相关文档列表
        """
        try:
            # 使用向量存储服务进行相似度搜索
            search_results = await self.vector_store_service.similarity_search(query, k=k)
            
            # 提取文档对象（忽略分数）
            documents = [doc for doc, score in search_results]
            
            logger.info(f"获取到 {len(documents)} 个相关文档")
            return documents
            
        except Exception as e:
            logger.error(f"获取相关文档失败: {str(e)}")
            return []
    
    async def answer_question(self, question: str) -> str:
        """
        简化的问答接口，只返回答案文本
        
        Args:
            question: 用户问题
            
        Returns:
            答案文本
        """
        result = await self.ask(question)
        return result.get("answer", "抱歉，无法回答您的问题。")