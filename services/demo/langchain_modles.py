from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

# 加载环境变量
load_dotenv()

# 初始化ChatModel（以ChatGPT为例）
chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    openai_api_base=os.getenv("OPENAI_API_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 调用ChatModel（输入是消息列表）
messages = [
    SystemMessage(content="你是一个AI助手，用简洁的语言回答问题"),
    HumanMessage(content="LangChain的核心价值是什么？")
]
chat_response = chat_model(messages)
print("\nChatModel输出:", chat_response.content)


if __name__ == '__main__':
    print(chat_response.content)
