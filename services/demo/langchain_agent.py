from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool  # 注意这里从langchain.tools导入
from langchain.chains import LLMMathChain
import os



def fun():
    load_dotenv()
    # 1. 初始化模型和工具
    # 确保已设置环境变量OPENAI_API_KEY，或直接在这里填入
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        temperature=0,
        model_name="gpt-3.5-turbo"  # 明确指定文本模型，避免混淆
    )

    # 内置数学计算工具
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

    # 自定义工具：获取当前时间
    def get_current_time():
        """返回当前的日期和时间，格式为YYYY-MM-DD HH:MM:SS"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 定义工具列表 - 使用最新的Tool初始化方式
    tools = [
        Tool.from_function(
            func=llm_math_chain.run,
            name="Calculator",
            description="用于执行数学计算，输入应为数学表达式（如'3+5*2'）"
        ),
        Tool.from_function(
            func=get_current_time,
            name="CurrentTime",
            description="用于获取当前的日期和时间"
        )
    ]

    # 2. 初始化代理
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # 3. 使用代理执行任务
    print("=== 数学计算任务 ===")
    result = agent.invoke("3的平方加上5的立方等于多少？")
    print("计算结果:", result["output"])

    print("\n=== 时间查询任务 ===")
    result = agent.invoke("现在是什么时间？")
    print("当前时间:", result["output"])

    print("\n=== 复杂任务 ===")
    result = agent.invoke("假设现在是下午3点，3小时45分钟后是什么时间？")
    print("复杂任务计算结果:", result["output"])


if __name__ == '__main__':
    fun()
