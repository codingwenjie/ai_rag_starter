"""Day6 向量数据库学习示例

演示如何使用embedding、向量存储和问答系统
"""

import asyncio
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.qa_system import QASystem


async def main():
    """主函数：演示完整的向量数据库使用流程"""
    print("🚀 Day6 向量数据库学习示例")
    print("=" * 50)
    
    # 1. 初始化服务
    print("\n📝 步骤1: 初始化服务")
    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService(embedding_service)
    qa_system = QASystem(embedding_service, vector_store_service)
    
    # 2. 准备测试文档
    print("\n📚 步骤2: 准备测试文档")
    documents = [
        Document(
            page_content="Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。它广泛应用于Web开发、数据科学、人工智能和自动化等领域。",
            metadata={"source": "python_intro.txt", "category": "编程语言"}
        ),
        Document(
            page_content="机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。常见的机器学习算法包括线性回归、决策树、神经网络等。",
            metadata={"source": "ml_basics.txt", "category": "机器学习"}
        ),
        Document(
            page_content="向量数据库是专门设计用来存储和查询高维向量数据的数据库系统。它们在相似性搜索、推荐系统和语义搜索等应用中发挥重要作用。",
            metadata={"source": "vector_db.txt", "category": "数据库"}
        ),
        Document(
            page_content="FAISS（Facebook AI Similarity Search）是一个用于高效相似性搜索和密集向量聚类的库。它可以处理任意大小的向量集合，甚至是内存无法容纳的向量集合。",
            metadata={"source": "faiss_intro.txt", "category": "工具库"}
        ),
        Document(
            page_content="LangChain是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件，使开发者能够轻松构建复杂的AI应用。",
            metadata={"source": "langchain_intro.txt", "category": "AI框架"}
        )
    ]
    
    print(f"准备了 {len(documents)} 个测试文档")
    
    # 3. 添加文档到向量存储
    print("\n🔄 步骤3: 向量化并存储文档")
    doc_ids = await vector_store_service.add_documents(documents)
    print(f"成功添加文档，ID: {doc_ids}")
    
    # 4. 测试相似度搜索
    print("\n🔍 步骤4: 测试相似度搜索")
    search_queries = [
        "什么是Python？",
        "机器学习的应用有哪些？",
        "向量数据库的作用是什么？"
    ]
    
    for query in search_queries:
        print(f"\n查询: {query}")
        results = await vector_store_service.semantic_search(query, k=2)
        
        for result in results:
            print(f"  📄 排名 {result['rank']}: {result['content'][:60]}...")
            print(f"     相似度: {result['similarity_score']:.4f} | 相关性: {result['relevance']}")
    
    # 5. 测试问答系统
    print("\n💬 步骤5: 测试问答系统")
    qa_questions = [
        "Python有什么特点？",
        "FAISS是什么？",
        "LangChain的主要用途是什么？"
    ]
    
    for question in qa_questions:
        print(f"\n❓ 问题: {question}")
        answer = await qa_system.ask(question)
        
        print(f"🤖 答案: {answer['answer']}")
        print(f"📊 置信度: {answer['confidence']}")
        
        if answer['sources']:
            print("📚 参考来源:")
            for i, source in enumerate(answer['sources'][:2], 1):
                print(f"  {i}. {source['content'][:50]}...")
    
    # 6. 保存向量索引
    print("\n💾 步骤6: 保存向量索引")
    success = vector_store_service.save_index()
    if success:
        print("✅ 向量索引保存成功")
    else:
        print("❌ 向量索引保存失败")
    
    # 7. 显示统计信息
    print("\n📊 步骤7: 系统统计信息")
    stats = vector_store_service.get_stats()
    print(f"文档总数: {stats['total_documents']}")
    print(f"向量维度: {stats.get('vector_dimension', 'N/A')}")
    print(f"最后更新: {stats.get('last_updated', 'N/A')}")
    
    print("\n🎉 Day6 学习示例完成！")
    print("\n📖 学习要点总结:")
    print("1. ✅ 掌握了文本向量化的基本原理")
    print("2. ✅ 学会了使用FAISS进行向量存储和检索")
    print("3. ✅ 理解了相似度搜索的工作机制")
    print("4. ✅ 构建了基于向量检索的问答系统")
    print("5. ✅ 学会了向量索引的持久化存储")


if __name__ == "__main__":
    # 确保环境变量已设置
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请先设置 OPENAI_API_KEY 环境变量")
        exit(1)
    
    # 运行示例
    asyncio.run(main())