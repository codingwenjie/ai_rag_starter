#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day6 向量数据库功能测试用例

本文件包含了Day6学习内容的完整测试用例，包括:
1. EmbeddingService 文本向量化测试
2. VectorStoreService FAISS存储测试
3. QASystem 问答系统测试
4. 端到端集成测试

运行方式:
    python -m pytest tests/test_day6_vector_db.py -v
    或者
    python tests/test_day6_vector_db.py
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.qa_system import QASystem


class TestEmbeddingService:
    """EmbeddingService 测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 检查是否有OpenAI API密钥
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  警告: 未设置OPENAI_API_KEY环境变量，跳过需要API调用的测试")
            self.skip_api_tests = True
        else:
            self.skip_api_tests = False
            self.embedding_service = EmbeddingService()
    
    def test_embedding_service_init(self):
        """测试EmbeddingService初始化"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_embedding_service_init")
            return
            
        assert self.embedding_service is not None
        assert self.embedding_service.model == "text-embedding-3-small"
        print("✅ EmbeddingService初始化测试通过")
    
    async def test_embed_single_text(self):
        """测试单个文本向量化"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_embed_single_text")
            return
            
        text = "这是一个测试文本"
        vector = await self.embedding_service.embed_text(text)
        
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(x, float) for x in vector)
        print(f"✅ 单文本向量化测试通过，向量维度: {len(vector)}")
    
    async def test_embed_multiple_texts(self):
        """测试批量文本向量化"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_embed_multiple_texts")
            return
            
        texts = ["文本1", "文本2", "文本3"]
        vectors = await self.embedding_service.embed_texts(texts)
        
        assert isinstance(vectors, list)
        assert len(vectors) == len(texts)
        assert all(isinstance(v, list) for v in vectors)
        print(f"✅ 批量文本向量化测试通过，处理了{len(texts)}个文本")
    
    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        # 使用模拟向量进行测试
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [0.0, 1.0, 0.0]
        vector3 = [1.0, 0.0, 0.0]
        
        if not self.skip_api_tests:
            similarity1 = self.embedding_service.cosine_similarity(vector1, vector2)
            similarity2 = self.embedding_service.cosine_similarity(vector1, vector3)
            
            assert abs(similarity1 - 0.0) < 1e-6  # 垂直向量相似度为0
            assert abs(similarity2 - 1.0) < 1e-6  # 相同向量相似度为1
            print("✅ 余弦相似度计算测试通过")
        else:
            print("⏭️  跳过API测试: test_cosine_similarity")
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_get_model_info")
            return
            
        info = self.embedding_service.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "provider" in info
        assert info["provider"] == "OpenAI"
        print(f"✅ 模型信息获取测试通过: {info}")


class TestVectorStoreService:
    """VectorStoreService 测试类"""
    
    def setup_method(self):
        """测试前准备"""
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  警告: 未设置OPENAI_API_KEY环境变量，跳过需要API调用的测试")
            self.skip_api_tests = True
        else:
            self.skip_api_tests = False
            self.embedding_service = EmbeddingService()
            self.vector_store_service = VectorStoreService(self.embedding_service)
            
            # 创建临时目录用于测试
            self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    async def test_add_documents(self):
        """测试添加文档"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_add_documents")
            return
            
        documents = [
            Document(page_content="测试文档1", metadata={"source": "test1"}),
            Document(page_content="测试文档2", metadata={"source": "test2"})
        ]
        
        doc_ids = await self.vector_store_service.add_documents(documents)
        
        assert isinstance(doc_ids, list)
        assert len(doc_ids) == len(documents)
        print(f"✅ 文档添加测试通过，添加了{len(documents)}个文档")
    
    async def test_similarity_search(self):
        """测试相似度搜索"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_similarity_search")
            return
            
        # 先添加文档
        documents = [
            Document(page_content="Python是一种编程语言", metadata={"source": "doc1"}),
            Document(page_content="Java是另一种编程语言", metadata={"source": "doc2"}),
            Document(page_content="苹果是一种水果", metadata={"source": "doc3"})
        ]
        
        await self.vector_store_service.add_documents(documents)
        
        # 搜索相关文档
        results = await self.vector_store_service.similarity_search("编程语言", k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        # 检查返回结果的类型
        if results:
            print(f"   搜索结果类型: {type(results[0])}")
            if isinstance(results[0], tuple) and len(results[0]) == 2:
                doc, score = results[0]
                print(f"   文档类型: {type(doc)}, 分数类型: {type(score)}")
                if hasattr(doc, 'page_content'):
                    print(f"   第一个结果内容: {doc.page_content[:50]}...")
        
        # 检查每个结果是否为 (Document, float) 元组
        for result in results:
            assert isinstance(result, tuple), f"结果应该是元组: {type(result)}"
            assert len(result) == 2, "元组应该包含文档和分数两个元素"
            
            doc, score = result
            assert hasattr(doc, 'page_content'), f"文档对象缺少 page_content 属性: {type(doc)}"
            import numpy as np
            assert isinstance(score, (int, float, np.floating)), f"分数应该是数值类型: {type(score)}"
        print(f"✅ 相似度搜索测试通过，找到{len(results)}个相关文档")
    
    async def test_save_and_load_index(self):
        """测试保存和加载索引"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_save_and_load_index")
            return
            
        # 添加文档
        documents = [
            Document(page_content="测试保存文档", metadata={"source": "save_test"})
        ]
        await self.vector_store_service.add_documents(documents)
        
        # 保存索引
        save_path = os.path.join(self.temp_dir, "test_index")
        success = self.vector_store_service.save_index(save_path)
        
        assert success
        assert os.path.exists(save_path)
        print(f"✅ 索引保存测试通过，保存到: {save_path}")
        
        # 创建新的服务实例并加载索引
        new_service = VectorStoreService(self.embedding_service)
        load_success = new_service.load_index(save_path)
        assert load_success
        
        # 验证加载的索引可以搜索
        results = await new_service.similarity_search("测试", k=1)
        assert len(results) > 0
        # 验证结果格式
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2
        doc, score = results[0]
        assert hasattr(doc, 'page_content')
        import numpy as np
        assert isinstance(score, (int, float, np.floating))
        print("✅ 索引加载测试通过")
    
    def test_get_stats(self):
        """测试获取统计信息"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_get_stats")
            return
            
        stats = self.vector_store_service.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_documents" in stats
        assert "has_index" in stats
        assert "index_path" in stats
        print(f"✅ 统计信息获取测试通过: {stats}")


class TestQASystem:
    """QASystem 测试类"""
    
    def setup_method(self):
        """测试前准备"""
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  警告: 未设置OPENAI_API_KEY环境变量，跳过需要API调用的测试")
            self.skip_api_tests = True
        else:
            self.skip_api_tests = False
            self.embedding_service = EmbeddingService()
            self.vector_store_service = VectorStoreService(self.embedding_service)
            self.qa_system = QASystem(self.embedding_service, self.vector_store_service)
    
    async def test_qa_system_answer(self):
        """测试问答系统回答"""
        if self.skip_api_tests:
            print("⏭️  跳过API测试: test_qa_system_answer")
            return
            
        # 添加知识文档
        documents = [
            Document(
                page_content="Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。",
                metadata={"source": "python_intro"}
            ),
            Document(
                page_content="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
                metadata={"source": "ml_intro"}
            )
        ]
        
        await self.vector_store_service.add_documents(documents)
        
        # 测试问答功能
        question = "什么是Python？"
        answer = await self.qa_system.answer_question(question)
        
        # 验证答案
        assert isinstance(answer, str)
        assert len(answer) > 0
        print(f"问答测试通过，答案长度: {len(answer)}")
        
        # 测试获取相关文档
        docs = await self.qa_system.get_relevant_docs("Python编程", k=2)
        assert isinstance(docs, list)
        assert len(docs) > 0
        
        # QA系统的 get_relevant_docs 返回 Document 对象列表
        for doc in docs:
            assert hasattr(doc, 'page_content')
            assert isinstance(doc.page_content, str)
        
        print(f"✅ 问答系统测试通过")
        print(f"   问题: {question}")
        print(f"   答案: {answer[:100]}...")
        print(f"   相关文档数量: {len(docs)}")


async def run_all_tests():
    """运行所有测试"""
    print("🧪 开始运行Day6向量数据库测试用例")
    print("=" * 50)
    
    # 测试EmbeddingService
    print("\n📝 测试 EmbeddingService")
    print("-" * 30)
    embedding_test = TestEmbeddingService()
    embedding_test.setup_method()
    
    embedding_test.test_embedding_service_init()
    await embedding_test.test_embed_single_text()
    await embedding_test.test_embed_multiple_texts()
    embedding_test.test_cosine_similarity()
    embedding_test.test_get_model_info()
    
    # 测试VectorStoreService
    print("\n🗄️  测试 VectorStoreService")
    print("-" * 30)
    vector_test = TestVectorStoreService()
    vector_test.setup_method()
    
    await vector_test.test_add_documents()
    await vector_test.test_similarity_search()
    await vector_test.test_save_and_load_index()
    vector_test.test_get_stats()
    
    vector_test.teardown_method()
    
    # 测试QASystem
    print("\n🤖 测试 QASystem")
    print("-" * 30)
    qa_test = TestQASystem()
    qa_test.setup_method()
    
    await qa_test.test_qa_system_answer()
    
    print("\n🎉 所有测试完成！")
    print("=" * 50)
    
    # 输出学习总结
    print("\n📚 Day6 学习成果总结:")
    print("1. ✅ 掌握了OpenAI Embedding API的使用")
    print("2. ✅ 学会了FAISS向量数据库的基本操作")
    print("3. ✅ 理解了向量相似度搜索的原理")
    print("4. ✅ 构建了完整的RAG问答系统")
    print("5. ✅ 学会了向量索引的持久化存储")
    print("6. ✅ 编写了完整的测试用例")
    
    if os.getenv("OPENAI_API_KEY"):
        print("\n💡 提示: 所有功能测试正常，可以开始实际项目开发！")
    else:
        print("\n⚠️  提示: 请设置OPENAI_API_KEY环境变量以运行完整测试")


if __name__ == "__main__":
    # 直接运行测试
    asyncio.run(run_all_tests())