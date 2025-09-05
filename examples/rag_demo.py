#!/usr/bin/env python3
"""RAG系统演示脚本

展示如何使用RAG系统进行文档问答，包括：
1. 文档上传和处理
2. 基础问答查询
3. 对话式问答
4. 相似度搜索

使用方法:
    python examples/rag_demo.py
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from services.rag.rag_service import RAGService, RAGConfig
    from services.rag.retrieval_qa import create_qa_component
    from services.rag.document_loader import DocumentLoader
    from services.rag.text_splitter import SmartTextSplitter
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需依赖包")
    sys.exit(1)

class RAGDemo:
    """RAG系统演示类"""
    
    def __init__(self):
        """初始化演示环境"""
        self.rag_service = None
        self.qa_components = {}
        self.collection_name = "demo_collection"
        
    def setup_rag_service(self):
        """设置RAG服务"""
        print("🔧 初始化RAG服务...")
        
        # 创建RAG配置
        config = RAGConfig(
            chunk_size=800,
            chunk_overlap=100,
            embedding_model="text-embedding-ada-002",
            llm_model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000,
            retrieval_k=4
        )
        
        # 初始化RAG服务
        self.rag_service = RAGService(config)
        print("✅ RAG服务初始化完成")
        
    def create_sample_documents(self) -> List[str]:
        """创建示例文档
        
        Returns:
            List[str]: 示例文档文件路径列表
        """
        print("📝 创建示例文档...")
        
        # 示例文档内容
        documents = {
            "ai_introduction.md": """
# 人工智能简介

人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

## 主要领域

### 机器学习
机器学习是AI的核心技术之一，通过算法让计算机从数据中学习模式，而无需明确编程。

### 深度学习
深度学习是机器学习的子集，使用多层神经网络来模拟人脑的工作方式。

### 自然语言处理
自然语言处理（NLP）使计算机能够理解、解释和生成人类语言。

## 应用场景
- 图像识别
- 语音识别
- 推荐系统
- 自动驾驶
- 医疗诊断
""",
            "rag_technology.md": """
# RAG技术详解

检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合信息检索和文本生成的AI技术。

## RAG工作原理

1. **文档索引**: 将知识库文档转换为向量表示并存储
2. **查询检索**: 根据用户问题检索相关文档片段
3. **上下文增强**: 将检索到的信息作为上下文提供给语言模型
4. **答案生成**: 基于上下文生成准确的回答

## 技术优势

- **知识更新**: 可以实时更新知识库，无需重新训练模型
- **可解释性**: 可以追溯答案来源，提高可信度
- **领域适应**: 容易适应特定领域的知识
- **成本效益**: 相比训练大模型，成本更低

## 核心组件

### 文档加载器
负责从各种格式（PDF、Word、HTML等）中提取文本内容。

### 文本切分器
将长文档切分为适合检索的文本块，保持语义完整性。

### 向量存储
使用嵌入模型将文本转换为向量，并建立高效的检索索引。

### 检索器
根据查询向量找到最相关的文档片段。

### 生成器
基于检索到的上下文生成最终答案。
""",
            "langchain_guide.md": """
# LangChain使用指南

LangChain是一个用于构建基于语言模型应用的框架，特别适合开发RAG系统。

## 核心概念

### 链（Chains）
链是LangChain的核心概念，用于组合多个组件来完成复杂任务。

### 文档加载器（Document Loaders）
LangChain提供了多种文档加载器，支持PDF、CSV、HTML等格式。

### 文本分割器（Text Splitters）
用于将长文档分割为较小的块，便于向量化和检索。

### 向量存储（Vector Stores）
支持多种向量数据库，如FAISS、Chroma、Pinecone等。

## RAG实现示例

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

# 创建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# 执行问答
result = qa_chain.run("什么是RAG技术？")
```

## 最佳实践

1. **合理的文本分块**: 选择适当的块大小和重叠
2. **高质量嵌入**: 使用合适的嵌入模型
3. **检索优化**: 调整检索参数和相似度阈值
4. **提示工程**: 设计有效的提示模板
"""
        }
        
        # 创建临时文件
        temp_files = []
        for filename, content in documents.items():
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.md', 
                delete=False,
                encoding='utf-8'
            )
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
            print(f"  ✅ 创建文档: {filename}")
            
        print(f"📄 共创建 {len(temp_files)} 个示例文档")
        return temp_files
        
    def load_and_process_documents(self, file_paths: List[str]):
        """加载和处理文档
        
        Args:
            file_paths: 文档文件路径列表
        """
        print("\n📚 加载和处理文档...")
        
        try:
            # 加载文档
            documents = self.rag_service.load_documents(file_paths)
            print(f"  ✅ 成功加载 {len(documents)} 个文档")
            
            # 切分文档
            split_documents = self.rag_service.split_documents(documents)
            print(f"  ✅ 文档切分完成，共 {len(split_documents)} 个文本块")
            
            # 创建向量存储
            self.rag_service.create_vector_store(split_documents, self.collection_name)
            print(f"  ✅ 向量存储创建完成: {self.collection_name}")
            
            # 显示文档统计
            stats = self.rag_service.get_stats()
            print(f"\n📊 文档处理统计:")
            print(f"  - 原始文档数: {len(documents)}")
            print(f"  - 文本块数: {len(split_documents)}")
            print(f"  - 平均块大小: {stats.get('avg_chunk_size', 'N/A')}")
            
        except Exception as e:
            print(f"❌ 文档处理失败: {e}")
            raise
            
    def setup_qa_components(self):
        """设置问答组件"""
        print("\n🤖 设置问答组件...")
        
        try:
            # 加载向量存储
            self.rag_service.load_vector_store(self.collection_name)
            
            # 创建不同类型的问答组件
            qa_types = ["simple", "retrieval", "conversational"]
            
            for qa_type in qa_types:
                qa_component = create_qa_component(
                    qa_type=qa_type,
                    retriever=self.rag_service.vector_store,
                    llm=self.rag_service.llm
                )
                self.qa_components[qa_type] = qa_component
                print(f"  ✅ {qa_type} 问答组件创建完成")
                
        except Exception as e:
            print(f"❌ 问答组件设置失败: {e}")
            raise
            
    def demo_basic_qa(self):
        """演示基础问答"""
        print("\n" + "="*50)
        print("🎯 基础问答演示")
        print("="*50)
        
        questions = [
            "什么是人工智能？",
            "RAG技术的工作原理是什么？",
            "LangChain的核心概念有哪些？",
            "深度学习和机器学习的关系是什么？"
        ]
        
        qa_component = self.qa_components["simple"]
        
        for i, question in enumerate(questions, 1):
            print(f"\n❓ 问题 {i}: {question}")
            
            try:
                result = qa_component.ask(
                    question=question,
                    k=3,
                    collection_name=self.collection_name
                )
                
                print(f"💡 回答: {result['answer']}")
                
                # 显示源文档
                if result.get('source_documents'):
                    print(f"\n📖 参考来源 ({len(result['source_documents'])} 个):")
                    for j, doc in enumerate(result['source_documents'][:2], 1):
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"  {j}. {preview}")
                        
            except Exception as e:
                print(f"❌ 问答失败: {e}")
                
    def demo_conversational_qa(self):
        """演示对话式问答"""
        print("\n" + "="*50)
        print("💬 对话式问答演示")
        print("="*50)
        
        conversation = [
            "什么是RAG技术？",
            "它有什么优势？",
            "在LangChain中如何实现？",
            "能给个具体的代码示例吗？"
        ]
        
        qa_component = self.qa_components["conversational"]
        
        for i, question in enumerate(conversation, 1):
            print(f"\n👤 用户 {i}: {question}")
            
            try:
                result = qa_component.ask(
                    question=question,
                    k=3,
                    collection_name=self.collection_name
                )
                
                print(f"🤖 助手: {result['answer']}")
                
                # 显示对话历史长度
                if hasattr(qa_component, 'get_conversation_stats'):
                    stats = qa_component.get_conversation_stats()
                    print(f"   (对话轮次: {stats.get('total_turns', i)})")
                    
            except Exception as e:
                print(f"❌ 对话失败: {e}")
                
    def demo_similarity_search(self):
        """演示相似度搜索"""
        print("\n" + "="*50)
        print("🔍 相似度搜索演示")
        print("="*50)
        
        queries = [
            "机器学习算法",
            "向量存储技术",
            "文档处理方法"
        ]
        
        for query in queries:
            print(f"\n🔎 搜索: {query}")
            
            try:
                results = self.rag_service.similarity_search(query, k=3)
                
                print(f"📋 找到 {len(results)} 个相关结果:")
                for i, result in enumerate(results, 1):
                    content = result.get('content', str(result))
                    score = result.get('score', 'N/A')
                    preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"  {i}. [相似度: {score}] {preview}")
                    
            except Exception as e:
                print(f"❌ 搜索失败: {e}")
                
    def demo_interactive_mode(self):
        """交互式问答模式"""
        print("\n" + "="*50)
        print("🎮 交互式问答模式")
        print("="*50)
        print("输入问题进行问答，输入 'quit' 退出")
        
        qa_component = self.qa_components["conversational"]
        
        while True:
            try:
                question = input("\n👤 您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 再见！")
                    break
                    
                if not question:
                    continue
                    
                result = qa_component.ask(
                    question=question,
                    k=4,
                    collection_name=self.collection_name
                )
                
                print(f"🤖 回答: {result['answer']}")
                
                # 显示源文档数量
                source_count = len(result.get('source_documents', []))
                if source_count > 0:
                    print(f"   (基于 {source_count} 个相关文档片段)")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                
    def cleanup_temp_files(self, file_paths: List[str]):
        """清理临时文件
        
        Args:
            file_paths: 临时文件路径列表
        """
        print("\n🧹 清理临时文件...")
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"  ✅ 删除: {file_path}")
            except Exception as e:
                print(f"  ⚠️ 删除失败: {file_path}, {e}")
                
    def run_demo(self, interactive: bool = False):
        """运行完整演示
        
        Args:
            interactive: 是否启用交互模式
        """
        print("🚀 RAG系统演示开始")
        print("="*60)
        
        temp_files = []
        
        try:
            # 1. 设置RAG服务
            self.setup_rag_service()
            
            # 2. 创建示例文档
            temp_files = self.create_sample_documents()
            
            # 3. 加载和处理文档
            self.load_and_process_documents(temp_files)
            
            # 4. 设置问答组件
            self.setup_qa_components()
            
            # 5. 运行演示
            self.demo_basic_qa()
            self.demo_conversational_qa()
            self.demo_similarity_search()
            
            # 6. 交互模式（可选）
            if interactive:
                self.demo_interactive_mode()
                
            print("\n" + "="*60)
            print("🎉 RAG系统演示完成！")
            print("\n📝 演示总结:")
            print("  ✅ 文档加载和处理")
            print("  ✅ 基础问答查询")
            print("  ✅ 对话式问答")
            print("  ✅ 相似度搜索")
            if interactive:
                print("  ✅ 交互式问答")
                
        except Exception as e:
            print(f"\n❌ 演示运行失败: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 清理临时文件
            if temp_files:
                self.cleanup_temp_files(temp_files)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG系统演示")
    parser.add_argument(
        "--interactive", 
        "-i", 
        action="store_true", 
        help="启用交互式问答模式"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="显示详细日志"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
        
    # 运行演示
    demo = RAGDemo()
    demo.run_demo(interactive=args.interactive)

if __name__ == "__main__":
    main()