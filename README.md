# AI RAG 大模型开发入门实战项目 🚀

> 从零开始学习 AI RAG（检索增强生成）和大模型应用开发的完整教程项目

## 🎯 项目简介

这是一个专为 **AI 初学者** 设计的实战学习项目，通过构建一个完整的 RAG 知识问答系统，帮助你掌握现代 AI 应用开发的核心技能。

**适合人群**：
- 🔰 AI 开发零基础的程序员
- 🎓 想要学习大模型应用开发的学生
- 💼 希望将 AI 技术应用到实际业务的开发者
- 🚀 对 RAG 技术感兴趣的技术爱好者

**你将学到什么**：
- ✅ RAG（检索增强生成）的核心原理和实现
- ✅ 如何使用 LangChain 构建 AI 应用
- ✅ 向量数据库的使用和优化
- ✅ 大模型 API 的集成和调用
- ✅ FastAPI 后端开发最佳实践
- ✅ 从简单问答到复杂 AI 系统的进阶路径

---

## 🛣️ 学习路线图

本项目采用 **渐进式学习** 的方式，从基础概念到高级应用，每个阶段都有对应的文档和代码示例：

### 📚 第一阶段：基础搭建（Day 1-2）
- [Day01: FastAPI 项目搭建与测试运行](docs/Day01-FastAPI%20项目搭建与测试运行.md)
- [Day02: 接口增强与配置管理](docs/Day02-接口增强与配置管理.md)

### 🏗️ 第二阶段：数据建模（Day 3）
- [Day03: 数据结构与序列化](docs/Day03-数据结构与序列化%20-%20Pydantic%20Schema设计%20+%20响应标准化.md)

### 🤖 第三阶段：AI 集成（Day 4-5）
- [Day04: LangChain集成+大模型调用](docs/Day04-LangChain集成+大模型调用.md)
- [Day05: LangChain核心组件](docs/Day05-LangChain核心组件.md)

### 🔍 第四阶段：向量检索（Day 6）
- [Day06: 向量数据库入门与实践](docs/Day06-向量数据库入门与实践.md)

### 🎯 第五阶段：RAG 实现（Day 7）
- [Day07: RAG检索增强生成实现](docs/Day07-RAG检索增强生成实现.md)

### 🚀 第六阶段：实战应用（Day 8）
- [Day08: 智能HR问答系统实战](docs/Day08-智能HR问答系统实战.md)

---

## 🏗️ 项目架构

```
📦 ai_rag_starter/
├── 📁 api/                    # FastAPI 接口层
│   ├── 📁 routes/             # 路由定义
│   │   ├── document_routes.py # 文档管理接口
│   │   └── qa_routes.py       # 问答接口
│   ├── 📁 models/             # 请求响应模型
│   └── 📁 utils/              # 接口工具类
├── 📁 services/               # 业务逻辑层
│   ├── 📁 rag/                # RAG 核心服务
│   │   ├── rag_service.py     # RAG 主服务
│   │   ├── document_loader.py # 文档加载器
│   │   └── text_splitter.py   # 文本分割器
│   ├── 📁 vectorDB/           # 向量数据库服务
│   │   ├── vector_store_service.py # 向量存储
│   │   └── embedding_service.py    # 向量化服务
│   └── 📁 langchain/          # LangChain 组件
├── 📁 schemas/                # 数据模型定义
├── 📁 core/                   # 核心配置
├── 📁 docs/                   # 学习文档
├── 📁 test_documents/         # 测试文档
├── 📁 examples/               # 示例代码
└── 📁 tests/                  # 单元测试
```

---

## 🚀 快速开始

### 环境准备

**系统要求**：
- Python 3.8+
- 稳定的网络连接（用于下载模型）

### 1️⃣ 克隆项目
```bash
git clone <your-repo-url>
cd ai_rag_starter
```

### 2️⃣ 创建虚拟环境
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 4️⃣ 配置环境变量
复制 `.env.example` 为 `.env` 并配置：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```bash
# OpenAI 配置（推荐）
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# 或者使用其他模型服务
# ZHIPU_API_KEY=your_zhipu_api_key
```

### 5️⃣ 启动服务
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6️⃣ 验证安装
访问以下地址验证服务正常运行：
- 📖 API 文档：http://localhost:8000/docs
- 🔍 健康检查：http://localhost:8000/health

---

## 💡 核心功能演示

### 📄 文档上传与处理
```bash
# 上传文档到向量数据库
curl -X POST "http://localhost:8000/api/rag/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_documents/QA.txt" \
     -F "collection_name=default"
```

### 🤖 智能问答
```bash
# HR 智能问答示例
curl -X POST "http://localhost:8000/api/rag/qa/hr" \
     -H "Content-Type: application/json" \
     -d '{"question": "薪资待遇怎么样？"}'
```

### 📊 文档管理
```bash
# 查看已上传的文档集合
curl "http://localhost:8000/api/rag/documents/collections"
```

---

## 🧱 技术栈详解

| 技术组件 | 作用 | 学习重点 |
|---------|------|----------|
| **FastAPI** | Web 框架 | 异步编程、API 设计、文档自动生成 |
| **LangChain** | AI 应用框架 | 链式调用、提示工程、模型集成 |
| **FAISS** | 向量数据库 | 向量检索、相似度计算、索引优化 |
| **OpenAI API** | 大语言模型 | API 调用、参数调优、成本控制 |
| **Pydantic** | 数据验证 | 类型检查、数据序列化、模型定义 |
| **Uvicorn** | ASGI 服务器 | 异步服务、性能优化、部署配置 |

---

## 📖 学习建议

### 🔰 零基础学习路径

**第1周：基础准备**
1. 熟悉 Python 基础语法（如果还不会）
2. 了解 HTTP 协议和 RESTful API 概念
3. 学习 FastAPI 基础用法

**第2周：AI 概念理解**
1. 理解什么是大语言模型（LLM）
2. 学习向量和嵌入（Embedding）的概念
3. 了解 RAG 的工作原理

**第3周：动手实践**
1. 跟着文档一步步搭建项目
2. 尝试上传不同类型的文档
3. 测试问答功能，观察效果

**第4周：深入优化**
1. 调整参数，观察效果变化
2. 尝试集成不同的模型
3. 思考如何应用到实际场景

### 💡 学习技巧

- **边学边做**：不要只看文档，一定要动手实践
- **记录问题**：遇到问题及时记录，便于后续查阅
- **参考示例**：多看 `examples/` 目录下的示例代码
- **阅读日志**：学会看控制台输出，理解程序运行过程
- **小步快跑**：每完成一个小功能就测试一下

---

## 🎯 项目亮点

### 🚀 已实现功能

- ✅ **文档智能处理**：支持多种格式文档的自动解析和分块
- ✅ **向量化存储**：文档内容自动转换为向量并存储
- ✅ **语义检索**：基于语义相似度的智能文档检索
- ✅ **智能问答**：结合检索结果生成准确回答
- ✅ **HR 问答系统**：专门针对 HR 场景优化的问答功能
- ✅ **API 接口**：完整的 RESTful API，支持前端集成
- ✅ **批量处理**：支持批量文档上传和处理
- ✅ **多模型支持**：可切换不同的 LLM 和 Embedding 模型

### 🔮 扩展方向

- 🔄 **Agent 系统**：基于当前架构扩展为智能 Agent
- 📊 **数据分析**：添加使用统计和效果分析功能
- 🎨 **前端界面**：开发 Web 前端界面
- 🔐 **用户系统**：添加用户认证和权限管理
- 📱 **移动端**：开发移动端应用
- 🌐 **多语言**：支持多语言文档和问答

---

## ❓ 常见问题

### Q1: 启动时提示 "OpenAI API key not found"
**A**: 请检查 `.env` 文件是否正确配置了 `OPENAI_API_KEY`，确保 API key 有效且有足够余额。

### Q2: 文档上传后问答效果不好
**A**: 可能的原因：
- 文档内容质量不高，建议使用结构化、内容丰富的文档
- 文档分块大小不合适，可以调整 `chunk_size` 参数
- 问题表述不够清晰，尝试换个问法

### Q3: 如何使用其他大模型？
**A**: 项目支持多种模型，在 `.env` 文件中配置相应的 API key 即可：
```bash
# 智谱 AI
ZHIPU_API_KEY=your_zhipu_key

# 其他模型...
```

### Q4: 向量数据库数据丢失怎么办？
**A**: 默认使用 FAISS 本地存储，数据保存在 `vector_stores/` 目录。如需持久化，建议：
- 定期备份 `vector_stores/` 目录
- 考虑使用云端向量数据库（如 Pinecone、Qdrant）

### Q5: 如何提高问答准确性？
**A**: 几个优化建议：
- 提供高质量的知识文档
- 优化文档的结构和格式
- 调整检索参数（top_k、score_threshold）
- 优化提示词模板
- 使用更强大的模型

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！**

**🎯 开始你的 AI 开发之旅吧！**

