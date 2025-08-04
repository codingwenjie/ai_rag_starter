from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

"""
这几个参数分别是什么意思：
    openai_api_key  openai的api key
    model_name  模型名称
    temperature  温度参数, 控制模型的创造性
    max_tokens  最大token数，控制模型的输出长度
    top_p  核采样参数, 控制模型的创造性
    frequency_penalty  频率惩罚参数, 控制模型的创造性
    presence_penalty  存在惩罚参数, 控制模型的创造性
"""
llm = ChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

def ask_llm(question: str) -> str:
    return llm.predict(question)
