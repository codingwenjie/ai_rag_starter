from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser
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

# 获取OpenAI API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def ask_llm(question: str) -> str:
    """
    使用LangChain调用大模型回答问题
    
    Args:
        question: 用户问题
        
    Returns:
        str: 模型回答
    """
    # 创建提示模板
    prompt = PromptTemplate(
        input_variables=["question"],
        template="请回答以下问题：{question}"
    )
    
    # 创建LLM实例
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )
    
    # 创建输出解析器
    output_parser = StrOutputParser()
    
    # 使用新的RunnableSequence语法创建链
    chain = prompt | llm | output_parser
    
    # 执行链
    response = chain.invoke({"question": question})
    
    return response




