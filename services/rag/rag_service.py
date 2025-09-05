"""RAG服务模块

实现检索增强生成(RAG)的核心服务，整合文档加载、文本切分、向量化和检索功能。
提供完整的RAG工作流程，支持文档问答和对话式检索。
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

try:
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # 提供基础Document类
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# 导入本地模块
from .document_loader import DocumentLoader
from .text_splitter import SmartTextSplitter
from ..vectorDB.vector_store_service import VectorStoreService
from ..vectorDB.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class RAGConfig:
    """RAG配置类"""
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 retrieval_k: int = 4,
                 score_threshold: float = 0.7):
        """初始化RAG配置
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            embedding_model: 嵌入模型名称
            llm_model: 语言模型名称
            temperature: 生成温度
            max_tokens: 最大生成token数
            retrieval_k: 检索返回的文档数量
            score_threshold: 相似度阈值
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_k = retrieval_k
        self.score_threshold = score_threshold


class RAGService:
    """RAG服务类
    
    提供完整的RAG功能，包括文档处理、向量化、检索和生成。
    """
    
    def __init__(self, config: RAGConfig = None):
        """初始化RAG服务
        
        Args:
            config: RAG配置对象
        """
        self.config = config or RAGConfig()
        
        # 初始化组件
        self.document_loader = DocumentLoader()
        self.text_splitter = SmartTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # 初始化嵌入服务
        self.embedding_service = EmbeddingService()
        
        # 向量存储服务
        self.vector_store_service = VectorStoreService(self.embedding_service)
        self.vector_store = None
        
        # LangChain组件（如果可用）
        self.embeddings = None
        self.llm = None
        self.qa_chain = None
        self.conversational_chain = None
        self.memory = None
        
        # 初始化LangChain组件
        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain_components()
        
        # 尝试加载现有的向量索引
        self._load_existing_vector_store()
        
        logger.info("RAG服务初始化完成")

    def _load_existing_vector_store(self):
        """加载现有的向量存储索引
        
        尝试加载默认路径下的向量索引，如果存在则自动加载
        """
        try:
            # 尝试加载向量存储服务的索引
            if self.vector_store_service.load_index():
                logger.info("成功加载现有向量存储索引")
            else:
                logger.info("未找到现有向量存储索引，将在首次使用时创建")
        except Exception as e:
            logger.warning(f"加载向量存储索引时出现异常: {e}")

    def _initialize_langchain_components(self):
        """初始化LangChain组件"""
        try:
            # 初始化嵌入模型
            if "openai" in self.config.embedding_model.lower():
                self.embeddings = OpenAIEmbeddings(
                    model=self.config.embedding_model
                )
            else:
                # 使用HuggingFace嵌入模型作为备选
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
            # 初始化语言模型
            if "gpt" in self.config.llm_model.lower():
                self.llm = ChatOpenAI(
                    model_name=self.config.llm_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            else:
                self.llm = OpenAI(
                    model_name=self.config.llm_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
            # 初始化对话记忆
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            logger.info("LangChain组件初始化成功")
            
        except Exception as e:
            logger.warning(f"LangChain组件初始化失败: {e}")
            
    def load_documents(self, 
                      source: Union[str, Path, List[str]], 
                      **kwargs) -> List[Document]:
        """加载文档
        
        Args:
            source: 文档源（文件路径、目录路径或路径列表）
            **kwargs: 其他参数
            
        Returns:
            List[Document]: 加载的文档列表
        """
        try:
            documents = []
            
            if isinstance(source, (list, tuple)):
                # 处理多个文件路径
                for path in source:
                    path_obj = Path(path)
                    if path_obj.is_file():
                        documents.extend(self.document_loader.load_document(path))
                    elif path_obj.is_dir():
                        documents.extend(self.document_loader.load_directory(path, **kwargs))
            else:
                # 处理单个路径
                path_obj = Path(source)
                if path_obj.is_file():
                    documents = self.document_loader.load_document(source)
                elif path_obj.is_dir():
                    documents = self.document_loader.load_directory(source, **kwargs)
                else:
                    raise ValueError(f"路径不存在或无效: {source}")
            
            logger.info(f"成功加载 {len(documents)} 个文档")
            return documents
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            raise
            
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档
        
        Args:
            documents: 文档列表
            
        Returns:
            List[Document]: 切分后的文档列表
        """
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
            return split_docs
        except Exception as e:
            logger.error(f"文档切分失败: {e}")
            raise
            
    def create_vector_store(self, 
                           documents: List[Document], 
                           store_name: str = "default",
                           persist: bool = True) -> Any:
        """创建向量存储
        
        Args:
            documents: 文档列表
            store_name: 存储名称
            persist: 是否持久化
            
        Returns:
            向量存储对象
        """
        try:
            if LANGCHAIN_AVAILABLE and self.embeddings:
                # 使用LangChain的FAISS
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                
                if persist:
                    # 保存向量存储
                    store_path = f"vector_stores/{store_name}"
                    os.makedirs(store_path, exist_ok=True)
                    self.vector_store.save_local(store_path)
                    
                logger.info(f"向量存储创建成功: {len(documents)} 个文档")
                
            else:
                # 使用本地向量存储服务
                self.vector_store_service.create_collection(store_name)
                
                for doc in documents:
                    self.vector_store_service.add_document(
                        collection_name=store_name,
                        content=doc.page_content,
                        metadata=doc.metadata
                    )
                    
                self.vector_store = store_name
                logger.info(f"本地向量存储创建成功: {len(documents)} 个文档")
                
            return self.vector_store
            
        except Exception as e:
            logger.error(f"向量存储创建失败: {e}")
            raise
            
    async def add_document(self, 
                          content: str, 
                          metadata: Dict[str, Any] = None,
                          collection_name: str = "default",
                          filename: str = None,
                          **kwargs) -> Dict[str, Any]:
        """添加单个文档到向量存储
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            collection_name: 集合名称
            filename: 文件名
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 准备元数据
            doc_metadata = metadata or {}
            if filename:
                doc_metadata['filename'] = filename
                doc_metadata['source'] = filename
                doc_metadata['collection'] = collection_name
            
            # 切分文档内容
            from langchain.schema import Document
            doc = Document(page_content=content, metadata=doc_metadata)
            split_docs = self.split_documents([doc])
            
            # 添加文档块到向量存储
            chunks_created = len(split_docs)
            if chunks_created > 0:
                # 使用VectorStoreService的add_documents方法
                await self.vector_store_service.add_documents(split_docs)
            
            logger.info(f"文档添加成功到集合: {collection_name}，创建 {chunks_created} 个文本块")
            return {
                "success": True,
                "chunks_created": chunks_created,
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"文档添加失败: {e}")
            return {
                "success": False,
                "chunks_created": 0,
                "error": str(e)
            }
            
    def load_vector_store(self, store_name: str) -> Any:
        """加载向量存储
        
        Args:
            store_name: 存储名称
            
        Returns:
            向量存储对象
        """
        try:
            if LANGCHAIN_AVAILABLE and self.embeddings:
                store_path = f"vector_stores/{store_name}"
                if os.path.exists(store_path):
                    self.vector_store = FAISS.load_local(
                        store_path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"向量存储加载成功: {store_name}")
                else:
                    raise FileNotFoundError(f"向量存储不存在: {store_path}")
            else:
                # 使用本地向量存储服务
                collections = self.vector_store_service.list_collections()
                if store_name in collections:
                    self.vector_store = store_name
                    logger.info(f"本地向量存储加载成功: {store_name}")
                else:
                    raise ValueError(f"向量存储不存在: {store_name}")
                    
            return self.vector_store
            
        except Exception as e:
            logger.error(f"向量存储加载失败: {e}")
            raise
            
    def setup_qa_chain(self, 
                      chain_type: str = "stuff",
                      custom_prompt: str = None) -> Any:
        """设置问答链
        
        Args:
            chain_type: 链类型 ("stuff", "map_reduce", "refine", "map_rerank")
            custom_prompt: 自定义提示模板
            
        Returns:
            问答链对象
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain不可用，无法创建问答链")
            return None
            
        if not self.vector_store or not self.llm:
            raise ValueError("向量存储和语言模型必须先初始化")
            
        try:
            # 创建检索器
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.retrieval_k}
            )
            
            # 自定义提示模板
            if custom_prompt:
                prompt_template = PromptTemplate(
                    template=custom_prompt,
                    input_variables=["context", "question"]
                )
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type=chain_type,
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template},
                    return_source_documents=True
                )
            else:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type=chain_type,
                    retriever=retriever,
                    return_source_documents=True
                )
                
            logger.info(f"问答链设置成功: {chain_type}")
            return self.qa_chain
            
        except Exception as e:
            logger.error(f"问答链设置失败: {e}")
            raise
            
    def setup_conversational_chain(self, custom_prompt: str = None) -> Any:
        """设置对话式问答链
        
        Args:
            custom_prompt: 自定义提示模板
            
        Returns:
            对话式问答链对象
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain不可用，无法创建对话式问答链")
            return None
            
        if not self.vector_store or not self.llm:
            raise ValueError("向量存储和语言模型必须先初始化")
            
        try:
            # 创建检索器
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.retrieval_k}
            )
            
            # 创建对话式检索链
            if custom_prompt:
                self.conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    combine_docs_chain_kwargs={"prompt": PromptTemplate(
                        template=custom_prompt,
                        input_variables=["context", "question", "chat_history"]
                    )}
                )
            else:
                self.conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True
                )
                
            logger.info("对话式问答链设置成功")
            return self.conversational_chain
            
        except Exception as e:
            logger.error(f"对话式问答链设置失败: {e}")
            raise
            
    async def ask_question(self, 
                    question: str, 
                    use_conversational: bool = False) -> Dict[str, Any]:
        """提问
        
        Args:
            question: 问题
            use_conversational: 是否使用对话式问答
            
        Returns:
            Dict[str, Any]: 包含答案和源文档的字典
        """
        try:
            if LANGCHAIN_AVAILABLE:
                if use_conversational and self.conversational_chain:
                    result = self.conversational_chain({"question": question})
                elif self.qa_chain:
                    result = self.qa_chain({"query": question})
                else:
                    # 使用基础检索
                    return await self._basic_retrieval_qa(question)
                    
                return {
                    "answer": result.get("answer", ""),
                    "source_documents": result.get("source_documents", []),
                    "question": question
                }
            else:
                # 使用基础检索问答
                return await self._basic_retrieval_qa(question)
                
        except Exception as e:
            logger.error(f"问答失败: {e}")
            return {
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "source_documents": [],
                "question": question
            }
            
    async def _basic_retrieval_qa(self, question: str) -> Dict[str, Any]:
        """基础检索问答（不依赖LangChain）
        
        Args:
            question: 问题
            
        Returns:
            Dict[str, Any]: 包含答案和源文档的字典
        """
        try:
            if isinstance(self.vector_store, str):
                # 使用本地向量存储服务
                results = await self.vector_store_service.similarity_search(
                    query=question,
                    k=self.config.retrieval_k
                )
                
                # 构建上下文
                context_parts = []
                source_docs = []
                
                for doc, score in results:
                    context_parts.append(doc.page_content)
                    source_docs.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(score)
                    })
                    
                context = "\n\n".join(context_parts)
                
                # 简单的基于规则的回答生成
                answer = self._generate_simple_answer(question, context)
                
                return {
                    "answer": answer,
                    "source_documents": source_docs,
                    "question": question
                }
            else:
                return {
                    "answer": "向量存储未正确初始化",
                    "source_documents": [],
                    "question": question
                }
                
        except Exception as e:
            logger.error(f"基础检索问答失败: {e}")
            return {
                "answer": f"检索失败: {e}",
                "source_documents": [],
                "question": question
            }
            
    def _generate_simple_answer(self, question: str, context: str) -> str:
        """生成简单答案（基于规则）
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            str: 生成的答案
        """
        if not context.strip():
            return "抱歉，没有找到相关信息来回答您的问题。"
            
        # 简单的答案生成逻辑
        answer_parts = [
            "根据相关文档，我找到了以下信息：\n",
            context[:500] + "..." if len(context) > 500 else context,
            "\n\n如需更详细的信息，请查看源文档。"
        ]
        
        return "".join(answer_parts)
        
    def similarity_search(self, 
                         query: str, 
                         k: int = None) -> List[Dict[str, Any]]:
        """相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        k = k or self.config.retrieval_k
        
        try:
            if LANGCHAIN_AVAILABLE and hasattr(self.vector_store, 'similarity_search'):
                docs = self.vector_store.similarity_search(query, k=k)
                return [{
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 1.0  # FAISS不直接返回分数
                } for doc in docs]
            elif isinstance(self.vector_store, str):
                # 使用本地向量存储服务
                return self.vector_store_service.search(
                    collection_name=self.vector_store,
                    query=query,
                    limit=k
                )
            else:
                logger.warning("向量存储未正确初始化")
                return []
                
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
            
    def clear_memory(self):
        """清除对话记忆"""
        if self.memory:
            self.memory.clear()
            logger.info("对话记忆已清除")
            
    def get_stats(self) -> Dict[str, Any]:
        """获取RAG服务统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "embedding_model": self.config.embedding_model,
                "llm_model": self.config.llm_model,
                "retrieval_k": self.config.retrieval_k
            },
            "components": {
                "langchain_available": LANGCHAIN_AVAILABLE,
                "vector_store_initialized": self.vector_store is not None,
                "qa_chain_initialized": self.qa_chain is not None,
                "conversational_chain_initialized": self.conversational_chain is not None
            }
        }
        
        # 添加向量存储统计
        if isinstance(self.vector_store, str):
            try:
                vector_stats = self.vector_store_service.get_stats()
                stats["vector_store"] = {
                    "type": "local",
                    "current_collection": self.vector_store,
                    **vector_stats
                }
            except Exception as e:
                stats["vector_store"] = {"error": str(e)}
        elif LANGCHAIN_AVAILABLE and self.vector_store:
            stats["vector_store"] = {
                "type": "langchain_faiss",
                "initialized": True
            }
            
        return stats
    
    def list_collections(self) -> List[str]:
        """获取集合列表
        
        Returns:
            List[str]: 集合名称列表
        """
        try:
            if hasattr(self.vector_store_service, 'list_collections'):
                return self.vector_store_service.list_collections()
            else:
                # 如果vector_store_service没有list_collections方法，
                # 返回基于文件系统的集合列表
                vector_stores_dir = "vector_stores"
                if os.path.exists(vector_stores_dir):
                    collections = []
                    for item in os.listdir(vector_stores_dir):
                        item_path = os.path.join(vector_stores_dir, item)
                        if os.path.isdir(item_path):
                            collections.append(item)
                    return collections
                else:
                    return []
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            return []
    
    async def simple_qa(self, question: str, collection_name: str = "default") -> Dict[str, Any]:
        """简单问答
        
        Args:
            question: 用户问题
            collection_name: 集合名称
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        try:
            logger.info(f"开始简单问答: {question[:50]}...")
            
            # 检索相关文档
            if self.vector_store_service and self.vector_store_service.vector_store:
                results = self.vector_store_service.similarity_search(
                    query=question,
                    k=self.config.retrieval_k
                )
                docs = [doc for doc, score in results]
            elif self.vector_store:
                # 使用LangChain向量存储
                docs = self.vector_store.similarity_search(question, k=self.config.retrieval_k)
            else:
                # 没有可用的向量存储，返回默认答案
                return {
                    "answer": "抱歉，当前没有可用的知识库，无法回答您的问题。请先上传文档。",
                    "source_documents": []
                }
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 生成简单答案
            answer = self._generate_simple_answer(question, context)
            
            return {
                "answer": answer,
                "source_documents": [{
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs]
            }
            
        except Exception as e:
            logger.error(f"简单问答失败: {e}")
            return {
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "source_documents": []
            }
    
    async def retrieval_qa(self, question: str, collection_name: str = "default", 
                          k: int = 3, custom_prompt: str = None) -> Dict[str, Any]:
        """检索问答
        
        Args:
            question: 用户问题
            collection_name: 集合名称
            k: 检索文档数量
            custom_prompt: 自定义提示模板
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        try:
            logger.info(f"开始检索问答: {question[:50]}...")
            
            # 检索相关文档
            if self.vector_store_service and self.vector_store_service.vector_store:
                results = self.vector_store_service.similarity_search(
                    query=question,
                    k=k
                )
                docs = [doc for doc, score in results]
            elif self.vector_store:
                # 使用LangChain向量存储
                docs = self.vector_store.similarity_search(question, k=k)
            else:
                # 没有可用的向量存储，返回默认答案
                return {
                    "answer": "抱歉，当前没有可用的知识库，无法回答您的问题。请先上传文档。",
                    "source_documents": []
                }
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 使用自定义提示或默认提示
            if custom_prompt:
                prompt = custom_prompt.format(context=context, question=question)
            else:
                prompt = f"""请基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的信息中找到答案。

上下文信息：
{context}

问题：{question}

请提供详细和准确的答案："""
            
            # 生成答案
            answer = self._generate_simple_answer(question, context)
            
            return {
                "answer": answer,
                "source_documents": [{
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs]
            }
            
        except Exception as e:
            logger.error(f"检索问答失败: {e}")
            return {
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "source_documents": []
            }
    
    async def hr_qa(self, question: str, collection_name: str = "default", k: int = 3) -> Dict[str, Any]:
        """HR场景专用问答
        
        Args:
            question: 候选人问题
            collection_name: 集合名称
            k: 检索文档数量
            
        Returns:
            Dict[str, Any]: HR问答结果
        """
        try:
            logger.info(f"开始HR问答处理: {question[:50]}...")
            
            # HR场景关键词映射，提高检索准确性
            hr_keywords = {
                "薪资": ["薪资", "工资", "待遇", "收入", "底薪", "绩效"],
                "工作时间": ["工作时间", "上班时间", "班次", "排班", "倒班"],
                "休息": ["休息", "公休", "假期", "休假", "法定"],
                "要求": ["要求", "条件", "学历", "年龄", "技能"],
                "地点": ["地点", "位置", "地址", "工作地", "办公地"]
            }
            
            # 增强查询词，提高检索相关性
            enhanced_query = question
            for category, keywords in hr_keywords.items():
                if any(keyword in question for keyword in keywords):
                    enhanced_query += f" {category}"
            
            # 检索相关文档
            if self.vector_store_service and self.vector_store_service.vector_store:
                results = self.vector_store_service.similarity_search(
                    query=enhanced_query,
                    k=k
                )
                docs = [doc for doc, score in results]
            elif self.vector_store:
                docs = self.vector_store.similarity_search(enhanced_query, k=k)
            else:
                return {
                    "answer": "抱歉，当前HR知识库不可用，请联系技术支持。",
                    "source_documents": [],
                    "sources": []
                }
            
            # 构建HR专用上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # HR场景专用提示模板
            hr_prompt = f"""
            你是一位专业的HR招聘顾问，正在与候选人进行招聘咨询。请基于以下招聘信息准确回答候选人的问题：
            
            招聘信息：
            {context}
            
            候选人问题：{question}
            
            回答要求：
            1. 语调友好、专业，体现HR的亲和力
            2. 信息准确，直接引用招聘信息中的具体内容
            3. 如果问题涉及薪资、工作时间等关键信息，请详细说明
            4. 如果招聘信息中没有相关内容，请诚实告知并建议进一步沟通
            5. 适当表达对候选人的关注和欢迎
            
            请回答：
            """
            
            # 生成答案
            if self.llm:
                response = await self.llm.ainvoke(hr_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            else:
                answer = "抱歉，AI服务暂时不可用，请稍后再试或直接联系HR。"
            
            # 构建源文档信息
            sources = []
            for doc in docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            logger.info(f"HR问答处理完成，检索到 {len(docs)} 个相关文档")
            
            return {
                "answer": answer,
                "source_documents": docs,
                "sources": sources,
                "enhanced_query": enhanced_query
            }
            
        except Exception as e:
            logger.error(f"HR问答处理失败: {str(e)}")
            return {
                "answer": "抱歉，处理您的问题时出现了技术问题，请稍后再试或直接联系HR。",
                "source_documents": [],
                "sources": [],
                "error": str(e)
            }

    async def conversational_qa(self, question: str, collection_name: str = "default",
                               conversation_id: str = None, k: int = 3) -> Dict[str, Any]:
        """对话问答
        
        Args:
            question: 用户问题
            collection_name: 集合名称
            conversation_id: 对话ID
            k: 检索文档数量
            
        Returns:
            Dict[str, Any]: 对话结果
        """
        try:
            logger.info(f"开始对话问答: {question[:50]}...")
            
            # 检索相关文档
            if self.vector_store_service and self.vector_store_service.vector_store:
                results = self.vector_store_service.similarity_search(
                    query=question,
                    k=k
                )
                docs = [doc for doc, score in results]
            elif self.vector_store:
                # 使用LangChain向量存储
                docs = self.vector_store.similarity_search(question, k=k)
            else:
                 # 没有可用的向量存储，返回默认答案
                 return {
                     "answer": "抱歉，当前没有可用的知识库，无法回答您的问题。请先上传文档。",
                     "source_documents": []
                 }
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 获取对话历史（简化实现）
            chat_history = []  # 这里可以从数据库或缓存中获取历史对话
            
            # 构建对话提示
            history_text = "\n".join([f"用户: {h.get('question', '')}\n助手: {h.get('answer', '')}" 
                                    for h in chat_history[-3:]])  # 只保留最近3轮对话
            
            prompt = f"""你是一个智能助手，请基于提供的上下文信息和对话历史回答用户问题。

上下文信息：
{context}

对话历史：
{history_text}

当前问题：{question}

请提供准确和有帮助的答案："""
            
            # 生成答案
            answer = self._generate_simple_answer(question, context)
            
            # 生成或使用提供的对话ID
            if not conversation_id:
                import time
                conversation_id = f"conv_{int(time.time())}"
            
            return {
                "answer": answer,
                "conversation_id": conversation_id,
                "turn_number": len(chat_history) + 1,
                "source_documents": [{
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs],
                "chat_history": chat_history
            }
            
        except Exception as e:
            logger.error(f"对话问答失败: {e}")
            return {
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "conversation_id": conversation_id or "unknown",
                "turn_number": 1,
                "source_documents": [],
                "chat_history": []
            }


# 使用示例
if __name__ == "__main__":
    # 创建RAG服务
    config = RAGConfig(
        chunk_size=500,
        chunk_overlap=100,
        retrieval_k=3
    )
    
    rag_service = RAGService(config)
    
    # 示例文档
    sample_docs = [
        Document(
            page_content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            metadata={"source": "ai_intro.txt", "topic": "AI基础"}
        ),
        Document(
            page_content="机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
            metadata={"source": "ml_intro.txt", "topic": "机器学习"}
        )
    ]
    
    try:
        # 创建向量存储
        rag_service.create_vector_store(sample_docs, "demo_store")
        
        # 设置问答链
        if LANGCHAIN_AVAILABLE:
            rag_service.setup_qa_chain()
            
        # 提问
        result = rag_service.ask_question("什么是人工智能？")
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"源文档数量: {len(result['source_documents'])}")
        
        # 获取统计信息
        stats = rag_service.get_stats()
        print(f"\n服务统计: {stats}")
        
    except Exception as e:
        print(f"示例运行失败: {e}")