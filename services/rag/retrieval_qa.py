"""检索问答组件模块

实现基于检索的问答功能，包括RetrievalQA和ConversationalRetrievalQA组件。
支持多种问答策略和自定义提示模板。
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from datetime import datetime

try:
    from langchain.schema import Document, BaseRetriever
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.chains.question_answering import load_qa_chain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.prompts import PromptTemplate
    from langchain.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # 提供基础类定义
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class BaseRetriever(ABC):
        @abstractmethod
        def get_relevant_documents(self, query: str) -> List[Document]:
            pass

logger = logging.getLogger(__name__)

class QAPrompts:
    """问答提示模板集合"""
    
    # 基础问答提示
    BASIC_QA_PROMPT = """
    请基于以下上下文信息回答问题。如果上下文中没有相关信息，请诚实地说"我不知道"。
    
    上下文信息:
    {context}
    
    问题: {question}
    
    请提供准确、简洁的答案:
    """
    
    # 详细问答提示
    DETAILED_QA_PROMPT = """
    你是一个专业的AI助手。请仔细阅读以下上下文信息，并基于这些信息回答用户的问题。
    
    要求:
    1. 答案必须基于提供的上下文信息
    2. 如果上下文中没有足够信息回答问题，请明确说明
    3. 提供具体、准确的答案，避免模糊表述
    4. 如果可能，引用相关的具体信息
    
    上下文信息:
    {context}
    
    问题: {question}
    
    答案:
    """
    
    # 对话式问答提示
    CONVERSATIONAL_QA_PROMPT = """
    你是一个友好的AI助手，正在与用户进行对话。请基于提供的上下文信息和对话历史来回答问题。
    
    对话历史:
    {chat_history}
    
    上下文信息:
    {context}
    
    当前问题: {question}
    
    请提供有帮助的回答，保持对话的连贯性:
    """
    
    # 分析型问答提示
    ANALYTICAL_QA_PROMPT = """
    请作为一个分析专家，基于提供的上下文信息深入分析并回答问题。
    
    分析要求:
    1. 提供详细的分析过程
    2. 列出关键要点
    3. 如果有多个角度，请分别说明
    4. 给出结论和建议
    
    上下文信息:
    {context}
    
    分析问题: {question}
    
    详细分析:
    """


class BaseQAComponent(ABC):
    """问答组件基类"""
    
    def __init__(self, 
                 retriever: Any,
                 llm: Any = None,
                 prompt_template: str = None):
        """初始化问答组件
        
        Args:
            retriever: 检索器对象
            llm: 语言模型对象
            prompt_template: 提示模板
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or QAPrompts.BASIC_QA_PROMPT
        self.qa_history = []
        
    @abstractmethod
    async def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """提问接口
        
        Args:
            question: 问题
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        pass
        
    def _log_qa_interaction(self, question: str, answer: str, metadata: Dict[str, Any] = None):
        """记录问答交互
        
        Args:
            question: 问题
            answer: 答案
            metadata: 元数据
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        }
        self.qa_history.append(interaction)
        
    def get_history(self) -> List[Dict[str, Any]]:
        """获取问答历史
        
        Returns:
            List[Dict[str, Any]]: 问答历史列表
        """
        return self.qa_history.copy()
        
    def clear_history(self):
        """清除问答历史"""
        self.qa_history.clear()


class SimpleRetrievalQA(BaseQAComponent):
    """简单检索问答组件
    
    不依赖LangChain的基础实现。
    """
    
    def __init__(self, 
                 retriever: Any,
                 answer_generator: Callable[[str, str], str] = None,
                 **kwargs):
        """初始化简单检索问答
        
        Args:
            retriever: 检索器对象
            answer_generator: 答案生成函数
            **kwargs: 其他参数
        """
        super().__init__(retriever, **kwargs)
        self.answer_generator = answer_generator or self._default_answer_generator
        
    async def ask(self, question: str, k: int = 4, **kwargs) -> Dict[str, Any]:
        """提问
        
        Args:
            question: 问题
            k: 检索文档数量
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        try:
            # 检索相关文档
            if hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(question)
                # 确保docs是Document对象列表
                if docs and not isinstance(docs[0], Document):
                    # 如果返回的是元组列表，提取Document对象
                    docs = [doc if isinstance(doc, Document) else doc[0] for doc in docs]
            elif hasattr(self.retriever, 'similarity_search'):
                # 假设是向量存储服务
                search_results = self.retriever.similarity_search(
                    query=question,
                    k=k
                )
                docs = []
                for result in search_results:
                    if isinstance(result, tuple) and len(result) == 2:
                        doc, score = result
                        docs.append(Document(
                            page_content=doc.page_content,
                            metadata=doc.metadata
                        ))
                    elif isinstance(result, Document):
                        docs.append(result)
            else:
                docs = []
                
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 生成答案
            answer = self.answer_generator(question, context)
            
            # 构建结果
            result = {
                "question": question,
                "answer": answer,
                "source_documents": docs,
                "context_length": len(context),
                "num_sources": len(docs)
            }
            
            # 记录交互
            self._log_qa_interaction(
                question, 
                answer, 
                {"num_sources": len(docs), "context_length": len(context)}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"简单检索问答失败: {e}")
            error_result = {
                "question": question,
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "source_documents": [],
                "error": str(e)
            }
            return error_result
            
    def _default_answer_generator(self, question: str, context: str) -> str:
        """默认答案生成器
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            str: 生成的答案
        """
        if not context.strip():
            return "抱歉，我没有找到相关信息来回答您的问题。"
            
        # 简单的基于规则的答案生成
        if len(context) > 1000:
            context = context[:1000] + "..."
            
        answer_parts = [
            "根据相关文档，我找到了以下信息：\n\n",
            context,
            "\n\n这些信息应该能够回答您的问题。如需更详细的信息，请查看完整的源文档。"
        ]
        
        return "".join(answer_parts)


class LangChainRetrievalQA(BaseQAComponent):
    """基于LangChain的检索问答组件"""
    
    def __init__(self, 
                 retriever: Any,
                 llm: Any,
                 chain_type: str = "stuff",
                 prompt_template: str = None,
                 **kwargs):
        """初始化LangChain检索问答
        
        Args:
            retriever: 检索器对象
            llm: 语言模型对象
            chain_type: 链类型
            prompt_template: 提示模板
            **kwargs: 其他参数
        """
        super().__init__(retriever, llm, prompt_template)
        self.chain_type = chain_type
        self.qa_chain = None
        
        if LANGCHAIN_AVAILABLE:
            self._setup_qa_chain()
        else:
            logger.warning("LangChain不可用，将使用简单实现")
            
    def _setup_qa_chain(self):
        """设置问答链"""
        try:
            if self.prompt_template != QAPrompts.BASIC_QA_PROMPT:
                # 使用自定义提示模板
                prompt = PromptTemplate(
                    template=self.prompt_template,
                    input_variables=["context", "question"]
                )
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type=self.chain_type,
                    retriever=self.retriever,
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
            else:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type=self.chain_type,
                    retriever=self.retriever,
                    return_source_documents=True
                )
                
            logger.info(f"LangChain问答链设置成功: {self.chain_type}")
            
        except Exception as e:
            logger.error(f"LangChain问答链设置失败: {e}")
            self.qa_chain = None
            
    async def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """提问
        
        Args:
            question: 问题
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        if not self.qa_chain:
            # 降级到简单实现
            simple_qa = SimpleRetrievalQA(self.retriever)
            return await simple_qa.ask(question, **kwargs)
            
        try:
            # 使用LangChain问答链
            with get_openai_callback() as cb:
                result = self.qa_chain({"query": question})
                
            # 构建结果
            qa_result = {
                "question": question,
                "answer": result.get("result", ""),
                "source_documents": result.get("source_documents", []),
                "token_usage": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
            }
            
            # 记录交互
            self._log_qa_interaction(
                question, 
                qa_result["answer"], 
                {
                    "num_sources": len(qa_result["source_documents"]),
                    "token_usage": qa_result["token_usage"]
                }
            )
            
            return qa_result
            
        except Exception as e:
            logger.error(f"LangChain检索问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "source_documents": [],
                "error": str(e)
            }


class ConversationalRetrievalQA(BaseQAComponent):
    """对话式检索问答组件"""
    
    def __init__(self, 
                 retriever: Any,
                 llm: Any = None,
                 memory_type: str = "buffer",
                 prompt_template: str = None,
                 **kwargs):
        """初始化对话式检索问答
        
        Args:
            retriever: 检索器对象
            llm: 语言模型对象
            memory_type: 记忆类型 ("buffer", "summary")
            prompt_template: 提示模板
            **kwargs: 其他参数
        """
        super().__init__(retriever, llm, prompt_template or QAPrompts.CONVERSATIONAL_QA_PROMPT)
        self.memory_type = memory_type
        self.memory = None
        self.conversational_chain = None
        self.conversation_history = []
        
        if LANGCHAIN_AVAILABLE and llm:
            self._setup_conversational_chain()
        else:
            logger.warning("LangChain不可用或LLM未提供，将使用简单对话实现")
            
    def _setup_conversational_chain(self):
        """设置对话式问答链"""
        try:
            # 初始化记忆
            if self.memory_type == "summary":
                self.memory = ConversationSummaryMemory(
                    llm=self.llm,
                    memory_key="chat_history",
                    return_messages=True
                )
            else:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                
            # 创建对话式检索链
            if self.prompt_template != QAPrompts.CONVERSATIONAL_QA_PROMPT:
                # 使用自定义提示模板
                self.conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    combine_docs_chain_kwargs={"prompt": PromptTemplate(
                        template=self.prompt_template,
                        input_variables=["context", "question", "chat_history"]
                    )}
                )
            else:
                self.conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True
                )
                
            logger.info(f"对话式问答链设置成功: {self.memory_type} memory")
            
        except Exception as e:
            logger.error(f"对话式问答链设置失败: {e}")
            self.conversational_chain = None
            
    async def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """对话式提问
        
        Args:
            question: 问题
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        if self.conversational_chain:
            return self._langchain_conversational_ask(question, **kwargs)
        else:
            return self._simple_conversational_ask(question, **kwargs)
            
    def _langchain_conversational_ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """使用LangChain的对话式问答
        
        Args:
            question: 问题
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        try:
            with get_openai_callback() as cb:
                result = self.conversational_chain({"question": question})
                
            qa_result = {
                "question": question,
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", []),
                "chat_history": self._get_chat_history_summary(),
                "token_usage": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
            }
            
            # 记录交互
            self._log_qa_interaction(
                question, 
                qa_result["answer"], 
                {
                    "conversation_turn": len(self.conversation_history) + 1,
                    "token_usage": qa_result["token_usage"]
                }
            )
            
            return qa_result
            
        except Exception as e:
            logger.error(f"LangChain对话式问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "source_documents": [],
                "error": str(e)
            }
            
    def _simple_conversational_ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """简单对话式问答实现
        
        Args:
            question: 问题
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        try:
            # 检索相关文档
            if hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(question)
            elif hasattr(self.retriever, 'search'):
                search_results = self.retriever.search(
                    collection_name=kwargs.get('collection_name', 'default'),
                    query=question,
                    limit=kwargs.get('k', 4)
                )
                docs = [Document(
                    page_content=result['content'],
                    metadata=result.get('metadata', {})
                ) for result in search_results]
            else:
                docs = []
                
            # 构建上下文（包含对话历史）
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 生成考虑对话历史的答案
            answer = self._generate_conversational_answer(question, context)
            
            # 更新对话历史
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            result = {
                "question": question,
                "answer": answer,
                "source_documents": docs,
                "chat_history": self._get_chat_history_summary(),
                "conversation_turn": len(self.conversation_history)
            }
            
            # 记录交互
            self._log_qa_interaction(
                question, 
                answer, 
                {"conversation_turn": len(self.conversation_history)}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"简单对话式问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理问题时出现错误: {e}",
                "source_documents": [],
                "error": str(e)
            }
            
    def _generate_conversational_answer(self, question: str, context: str) -> str:
        """生成考虑对话历史的答案
        
        Args:
            question: 当前问题
            context: 检索到的上下文
            
        Returns:
            str: 生成的答案
        """
        if not context.strip():
            if self.conversation_history:
                return "基于我们之前的对话，我没有找到相关信息来回答这个新问题。您能提供更多背景信息吗？"
            else:
                return "抱歉，我没有找到相关信息来回答您的问题。"
                
        # 考虑对话历史的简单实现
        answer_parts = []
        
        if self.conversation_history:
            answer_parts.append("基于我们之前的对话和相关文档：\n\n")
        else:
            answer_parts.append("根据相关文档：\n\n")
            
        # 限制上下文长度
        if len(context) > 800:
            context = context[:800] + "..."
            
        answer_parts.append(context)
        answer_parts.append("\n\n如果您需要更多信息或有后续问题，请随时告诉我。")
        
        return "".join(answer_parts)
        
    def _get_chat_history_summary(self) -> List[Dict[str, str]]:
        """获取对话历史摘要
        
        Returns:
            List[Dict[str, str]]: 对话历史摘要
        """
        if self.memory and hasattr(self.memory, 'chat_memory'):
            # 使用LangChain记忆
            messages = self.memory.chat_memory.messages
            history = []
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        "question": messages[i].content,
                        "answer": messages[i + 1].content
                    })
            return history
        else:
            # 使用简单对话历史
            return [{
                "question": item["question"],
                "answer": item["answer"]
            } for item in self.conversation_history[-5:]]  # 只返回最近5轮对话
            
    def clear_conversation(self):
        """清除对话历史"""
        if self.memory:
            self.memory.clear()
        self.conversation_history.clear()
        logger.info("对话历史已清除")
        
    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息
        
        Returns:
            Dict[str, Any]: 对话统计信息
        """
        return {
            "total_turns": len(self.conversation_history),
            "memory_type": self.memory_type,
            "langchain_available": self.conversational_chain is not None,
            "recent_questions": [item["question"] for item in self.conversation_history[-3:]]
        }


# 工厂函数
def create_qa_component(qa_type: str, 
                       retriever: Any, 
                       llm: Any = None, 
                       **kwargs) -> BaseQAComponent:
    """创建问答组件的工厂函数
    
    Args:
        qa_type: 问答类型 ("simple", "retrieval", "conversational")
        retriever: 检索器对象
        llm: 语言模型对象
        **kwargs: 其他参数
        
    Returns:
        BaseQAComponent: 问答组件实例
    """
    if qa_type == "simple":
        return SimpleRetrievalQA(retriever, **kwargs)
    elif qa_type == "retrieval":
        if LANGCHAIN_AVAILABLE and llm:
            return LangChainRetrievalQA(retriever, llm, **kwargs)
        else:
            logger.warning("LangChain不可用或LLM未提供，使用简单检索问答")
            return SimpleRetrievalQA(retriever, **kwargs)
    elif qa_type == "conversational":
        return ConversationalRetrievalQA(retriever, llm, **kwargs)
    else:
        raise ValueError(f"不支持的问答类型: {qa_type}")


# 使用示例
if __name__ == "__main__":
    # 模拟检索器
    class MockRetriever:
        def get_relevant_documents(self, query: str) -> List[Document]:
            return [
                Document(
                    page_content="人工智能是计算机科学的一个分支。",
                    metadata={"source": "ai_doc.txt"}
                ),
                Document(
                    page_content="机器学习是AI的重要组成部分。",
                    metadata={"source": "ml_doc.txt"}
                )
            ]
    
    # 创建检索器
    retriever = MockRetriever()
    
    # 测试简单检索问答
    print("=== 简单检索问答 ===")
    simple_qa = create_qa_component("simple", retriever)
    result = simple_qa.ask("什么是人工智能？")
    print(f"问题: {result['question']}")
    print(f"答案: {result['answer'][:100]}...")
    print(f"源文档数量: {result['num_sources']}")
    
    # 测试对话式问答
    print("\n=== 对话式问答 ===")
    conv_qa = create_qa_component("conversational", retriever)
    
    # 第一轮对话
    result1 = conv_qa.ask("什么是人工智能？")
    print(f"Q1: {result1['question']}")
    print(f"A1: {result1['answer'][:100]}...")
    
    # 第二轮对话
    result2 = conv_qa.ask("它有什么应用？")
    print(f"Q2: {result2['question']}")
    print(f"A2: {result2['answer'][:100]}...")
    
    # 对话统计
    stats = conv_qa.get_conversation_stats()
    print(f"对话统计: {stats}")