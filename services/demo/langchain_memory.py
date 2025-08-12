from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
import os

llm = OpenAI(
        model_name="gpt-3.5-turbo"
        , openai_api_key=os.getenv("OPENAI_API_KEY")
        , openai_api_base=os.getenv("OPENAI_API_BASE")
        , temperature=0)

# 1. 基础记忆：保存完整历史
buffer_memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=buffer_memory, verbose=True)

print("=== 完整记忆 ===")
chain.run("我叫小明，喜欢打篮球")
chain.run("我刚才说我喜欢什么运动？")  # 模型应记住“打篮球”

# 2. 窗口记忆：只保存最近2轮
window_memory = ConversationBufferWindowMemory(k=2)  # k=2表示保留2轮
chain = ConversationChain(llm=llm, memory=window_memory, verbose=True)

print("\n=== 窗口记忆 ===")
chain.run("我叫小红，喜欢画画")
chain.run("我喜欢的颜色是蓝色")
chain.run("我刚才提到的爱好是什么？")  # 能记住（在2轮内）
chain.run("我叫什么名字？")  # 记不住（超过2轮）

# 3. 总结记忆：用模型总结历史
summary_memory = ConversationSummaryMemory(llm=llm)  # 需要LLM来生成总结
chain = ConversationChain(llm=llm, memory=summary_memory, verbose=True)

print("\n=== 总结记忆 ===")
chain.run("我是一名学生，在北京大学学习计算机科学")
chain.run("我平时喜欢编程和打羽毛球，周末会去图书馆")
print("\n对话总结:", summary_memory.load_memory_variables({})["history"])  # 查看总结内容
chain.run("我在哪所大学上学？")  # 模型通过总结回忆信息

