#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量检索演示脚本

本脚本演示如何：
1. 验证文档是否已存储到向量数据库
2. 执行相似度搜索
3. 进行问答查询
4. 查看存储统计信息
"""

import asyncio
import requests
import json
from typing import Dict, Any, List

# API基础URL
BASE_URL = "http://localhost:8000"

class VectorRetrievalDemo:
    """向量检索演示类"""
    
    def __init__(self, base_url: str = BASE_URL):
        """初始化演示类
        
        Args:
            base_url: API服务基础URL
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_server_status(self) -> bool:
        """检查服务器状态
        
        Returns:
            bool: 服务器是否可用
        """
        try:
            response = self.session.get(f"{self.base_url}/docs")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False
    
    def get_vector_store_stats(self, collection_name: str = "default") -> Dict[str, Any]:
        """获取向量存储统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            # 通过搜索接口间接获取存储状态
            response = self.session.post(
                f"{self.base_url}/api/rag/search",
                data={
                    "query": "test",
                    "collection_name": collection_name,
                    "k": 1
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "collection_exists": True,
                    "collection_name": collection_name,
                    "test_search_success": result.get("success", False),
                    "available_results": result.get("total_results", 0)
                }
            else:
                return {
                    "collection_exists": False,
                    "error": response.text
                }
        except Exception as e:
            return {
                "collection_exists": False,
                "error": str(e)
            }
    
    def similarity_search(self, query: str, collection_name: str = "default", k: int = 3) -> Dict[str, Any]:
        """执行相似度搜索
        
        Args:
            query: 搜索查询
            collection_name: 集合名称
            k: 返回结果数量
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/rag/search",
                json={
                    "query": query,
                    "collection_name": collection_name,
                    "k": k
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def ask_question(self, question: str, collection_name: str = "default", qa_type: str = "basic", k: int = 3) -> Dict[str, Any]:
        """执行问答查询
        
        Args:
            question: 问题
            collection_name: 集合名称
            qa_type: 问答类型 (basic/langchain)
            k: 检索文档数量
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/rag/ask",
                json={
                    "question": question,
                    "collection_name": collection_name,
                    "qa_type": qa_type,
                    "k": k
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def print_search_results(self, results: Dict[str, Any]) -> None:
        """打印搜索结果
        
        Args:
            results: 搜索结果
        """
        if not results.get("success"):
            print(f"❌ 搜索失败: {results.get('error')}")
            return
        
        print(f"\n🔍 搜索查询: {results.get('query')}")
        print(f"📊 集合名称: {results.get('collection_name')}")
        print(f"📈 结果数量: {results.get('total_results')}")
        
        for i, result in enumerate(results.get('results', []), 1):
            print(f"\n--- 结果 {i} ---")
            content = result.get('content', result.get('page_content', ''))
            print(f"内容: {content[:200]}{'...' if len(content) > 200 else ''}")
            
            metadata = result.get('metadata', {})
            if metadata:
                print(f"元数据: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
            
            if 'score' in result:
                print(f"相似度得分: {result['score']:.4f}")
    
    def print_qa_results(self, results: Dict[str, Any]) -> None:
        """打印问答结果
        
        Args:
            results: 问答结果
        """
        if not results.get("success", True):
            print(f"❌ 问答失败: {results.get('error')}")
            return
        
        print(f"\n❓ 问题: {results.get('question')}")
        print(f"✅ 答案: {results.get('answer')}")
        print(f"📚 问答类型: {results.get('qa_type')}")
        print(f"⏱️ 处理时间: {results.get('processing_time', 0):.2f}秒")
        
        source_docs = results.get('source_documents', [])
        if source_docs:
            print(f"\n📖 参考文档 ({len(source_docs)}个):")
            for i, doc in enumerate(source_docs, 1):
                content = doc.get('content', '')
                print(f"\n  {i}. {content[:150]}{'...' if len(content) > 150 else ''}")
                
                metadata = doc.get('metadata', {})
                if metadata:
                    print(f"     元数据: {json.dumps(metadata, ensure_ascii=False)}")
    
    def run_demo(self) -> None:
        """运行完整演示"""
        print("🚀 向量检索演示开始")
        print("=" * 50)
        
        # 1. 检查服务器状态
        print("\n1️⃣ 检查服务器状态...")
        if not self.check_server_status():
            print("❌ 服务器不可用，请确保RAG API服务正在运行")
            return
        print("✅ 服务器运行正常")
        
        # 2. 检查向量存储状态
        print("\n2️⃣ 检查向量存储状态...")
        stats = self.get_vector_store_stats()
        print(f"📊 存储状态: {json.dumps(stats, ensure_ascii=False, indent=2)}")
        
        if not stats.get("collection_exists"):
            print("❌ 向量数据库中没有数据，请先上传文档")
            print("💡 提示: 使用 /api/rag/upload 接口上传文档")
            return
        
        # 3. 执行相似度搜索
        print("\n3️⃣ 执行相似度搜索...")
        search_queries = [
            "Python编程",
            "机器学习算法",
            "数据科学"
        ]
        
        for query in search_queries:
            print(f"\n🔍 搜索: {query}")
            search_results = self.similarity_search(query, k=2)
            self.print_search_results(search_results)
        
        # 4. 执行问答查询
        print("\n4️⃣ 执行问答查询...")
        questions = [
            "什么是Python？",
            "机器学习有哪些常见算法？",
            "如何开始学习数据科学？"
        ]
        
        for question in questions:
            print(f"\n❓ 问题: {question}")
            qa_results = self.ask_question(question, qa_type="basic", k=2)
            self.print_qa_results(qa_results)
        
        print("\n🎉 演示完成！")
        print("=" * 50)

def main():
    """主函数"""
    demo = VectorRetrievalDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()