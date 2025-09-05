import os
import logging
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from core.exceptions import validation_exception_handler, general_exception_handler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

# 禁用 LangSmith 追踪以避免 Pydantic 兼容性问题
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 初始化 FastAPI 应用
app = FastAPI(
    title="AI RAG Starter API",
    version="1.0.0"
)

# 注册异常处理函数
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可改为指定前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 尝试导入 chat 路由，如果失败则使用维护模式
try:
    from api import chat
    app.include_router(chat.router, prefix="/api")
    print("✅ Chat routes loaded successfully")
except Exception as e:
    print(f"⚠️  Chat routes failed to load: {e}")
    print("🔧 Loading maintenance mode endpoints")
    

# 导入RAG API路由
try:
    from api import rag_api
    app.include_router(rag_api.router, prefix="/api")
    print("✅ RAG API routes loaded successfully")
except Exception as e:
    print(f"⚠️  RAG API routes failed to load: {e}")

# # 尝试导入向量检索API路由（如果存在）
# try:
#     from api import vector_retrieval
#     app.include_router(vector_retrieval.router, prefix="/api")
#     print("✅ Vector retrieval routes loaded successfully")
# except Exception as e:
#     print(f"⚠️  Vector retrieval routes failed to load: {e}")
#     print("ℹ️  Vector retrieval functionality will be unavailable")

print("🚀 Application startup complete")
