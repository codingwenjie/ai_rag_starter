from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import os

llm = OpenAI(
        model_name="gpt-3.5-turbo"
        , openai_api_key=os.getenv("OPENAI_API_KEY")
        , openai_api_base=os.getenv("OPENAI_API_BASE")
        , temperature=0)

# 1. 基础LLMChain
prompt = PromptTemplate(
    input_variables=["topic"],
    template="请用一句话概括{topic}的核心思想"
)
chain = LLMChain(llm=llm, prompt=prompt)
print("LLMChain输出:", chain.run("人工智能"))

# 2. SequentialChain：多步骤串联
# 第一步：生成产品名称
prompt1 = PromptTemplate(
    input_variables=["category"],
    template="为{category}领域生成一个创新产品的名称"
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="product_name")

# 第二步：生成产品功能
prompt2 = PromptTemplate(
    input_variables=["product_name"],
    template="为产品{product_name}设计3个核心功能"
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="features")

# 串联两个链
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["category"],
    output_variables=["product_name", "features"],  # 输出所有中间结果
    verbose=True  # 打印执行过程
)

result = overall_chain.run("健康管理")
print("\nSequentialChain结果:")
print("产品名称:", result["product_name"])
print("核心功能:", result["features"])
