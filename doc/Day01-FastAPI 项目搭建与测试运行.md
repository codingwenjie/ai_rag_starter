🧠 Day 1 - FastAPI 项目搭建与测试运行

🧾 今日任务清单：
   1. 安装依赖（FastAPI、Uvicorn）。Uvicorn是一个ASGI服务器
   2. 创建 FastAPI 项目结构
   3. 编写第一个 /chat POST 接口
   4. 使用 Pydantic 处理请求参数
   5. 启动服务并用 curl 或 Postman 测试

操作步骤：
 1. 安装依赖（FastAPI、Uvicorn）。Uvicorn是一个AIGC服务器，用于运行FastAPI应用。
    ```
    pip install fastapi uvicorn
    pip install pydantic python-dotenv
    ```
 2. 创建 FastAPI 项目结构。
       ai-rag-starter/
       │
       ├── app/
       │   ├── main.py  # 应用实例，类似Spring Boot中的Application类
       │   ├── api/
       │   │   └── chat.py  # 定义接口，类似Java中的controller
       │   └── models/
       │       └── request_model.py  # 定义请求参数模型，类似Java中的DTO
       ├── requirements.txt  # 定义项目依赖，类似Java中的pom.xml
       ├── README.md  # 项目说明文档
 3. 编写requirements.txt文件。
    ```
    fastapi  # FastAPI 框架
    uvicorn  # ASGI 服务器
    pydantic  # 数据验证库
    python-dotenv  # 环境变量加载库
    ```
 4. 编写第一个 /chat POST 接口。
    app/main.py 中添加以下代码：
    ```
    from fastapi import FastAPI
    from app.api import chat
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    ```
    
 5. 使用 Pydantic 处理请求参数。
 6. 启动服务并用 curl 或 Postman 测试。
    uvicorn app.main:app --reload
    ```shell
    # 解释这个命令
     uvicorn app.main:app --reload
    # app.main:app 表示 FastAPI 应用实例
    # --reload 表示在代码有变化时自动重启服务器
    ```
总结：
 1. 项目主要包含，FastAPI 应用实例、路由、请求参数模型、响应模型。
 2. 项目使用 Pydantic 处理请求参数，确保参数类型和格式正确。
 3. 项目使用 Uvicorn 启动服务，支持热重载。
 

