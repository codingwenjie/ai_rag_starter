import dotenv
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
import os

dotenv.load_dotenv()

if __name__ == '__main__':
    llm = OpenAI(
        model_name="gpt-3.5-turbo"
        , openai_api_key=os.getenv("OPENAI_API_KEY")
        , openai_api_base=os.getenv("OPENAI_API_BASE")
        , temperature=0)

    # 1. 基础模板：简单变量替换
    basic_template = """
    请为产品"{product_name}"生成一句宣传语，突出其"{feature}"特点。
    """
    basic_prompt = PromptTemplate(
        input_variables=["product_name", "feature"],  # 定义变量
        template=basic_template
    )

    # 生成提示词并调用模型
    prompt_text = basic_prompt.format(product_name="智能手表", feature="超长续航")
    print("基础提示词:\n", prompt_text)
    # print("模型输出:", llm(prompt_text))

    # 2. Few-shot模板：注入示例提升效果
    # 定义示例
    examples = [
        {"input": "苹果", "output": "红色、圆形、甜脆的水果"},
        {"input": "香蕉", "output": "黄色、长条形、软糯的水果"}
    ]

    # 示例模板
    example_template = "输入: {input}\n输出: {output}"
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    # 构建Few-shot模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="按照以下格式描述物品：",  # 提示词前缀
        suffix="输入: {input}\n输出:",  # 提示词后缀（包含用户输入）
        input_variables=["input"]  # 最终变量
    )

    # 使用Few-shot模板
    prompt_text = few_shot_prompt.format(input="西瓜")
    print("\nFew-shot提示词:\n", prompt_text)
    print("模型输出:", llm(prompt_text))
