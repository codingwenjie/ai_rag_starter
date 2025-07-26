# AI RAG FastAPI 应用模板 🚀

> 基于 FastAPI + LangChain 构建的 RAG（Retrieval-Augmented Generation）知识问答系统

本项目是一个 AI 应用开发的快速入门模板，结合现代 AI 大模型开发最佳实践，适用于构建企业内部智能问答系统、智能客服、私有化知识库检索等场景。

---

## ✨ 项目特色

- ✅ 基于 FastAPI 构建现代化后端接口
- 🔗 集成 LangChain 实现 RAG 能力
- 🧠 可接入 OpenAI / ZhipuAI 等 LLM 模型
- 🧪 支持本地文档上传、切分、向量化、存储
- 🔍 支持向量数据库（默认使用 FAISS）
- ⚙️ 环境变量配置支持（`.env`）
- 🌍 支持跨域请求（CORS）
- 📦 模块化结构，方便后续扩展为 Agent 系统

---

## 🧱 技术栈

| 层级       | 技术               |
|------------|--------------------|
| 后端框架   | FastAPI            |
| 模型框架   | LangChain          |
| 向量数据库 | FAISS（可替换为 Milvus / Qdrant） |
| LLM 接口   | OpenAI / 通义千问（可拓展） |
| 数据切分   | LangChain 文档加载器 + 分块器 |

---

## 📂 项目结构

```bash
ai-rag-fastapi/
├── app/                    # 主应用代码
│   ├── api/                # FastAPI 路由
│   ├── core/               # 配置与中间件
│   ├── rag/                # RAG 相关逻辑（文档加载、分块、向量化、查询）
│   ├── models/             # 请求响应模型
│   └── utils/              # 工具类（日志等）
├── docs/                   # 项目文档
├── tests/                  # 单元测试
├── .env                    # 环境变量
├── requirements.txt        # Python 依赖
└── main.py                 # 入口文件
