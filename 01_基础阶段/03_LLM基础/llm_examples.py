import os
import openai
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("=== LLM 基础示例 ===")
print("=" * 50)

# 检查API密钥
def check_api_keys():
    """检查API密钥是否设置"""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
    }
    
    print("API密钥状态:")
    for key_name, key_value in keys.items():
        if key_value:
            print(f"✓ {key_name}: 已设置")
        else:
            print(f"✗ {key_name}: 未设置")
    print("=" * 50)
    return keys

# 1. OpenAI API 示例
def openai_example():
    """OpenAI API 使用示例"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API 密钥未设置，跳过此示例")
        return
    
    print("1. OpenAI API 示例:")
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # 聊天补全
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个 helpful 的智能助手。"},
                {"role": "user", "content": "什么是大语言模型？"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        print("OpenAI 响应:")
        print(response.choices[0].message["content"])
    except Exception as e:
        print(f"错误: {e}")
    print("=" * 50)

# 2. Anthropic API 示例
def anthropic_example():
    """Anthropic API 使用示例"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Anthropic API 密钥未设置，跳过此示例")
        return
    
    print("2. Anthropic API 示例:")
    try:
        from anthropic import Anthropic
        
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # 聊天补全
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            messages=[
                {"role": "user", "content": "什么是大语言模型？"}
            ]
        )
        
        print("Anthropic 响应:")
        print(response.content[0].text)
    except ImportError:
        print("错误: 未安装 anthropic 库，请运行 'pip install anthropic'")
    except Exception as e:
        print(f"错误: {e}")
    print("=" * 50)

# 3. Prompt 工程示例
def prompt_engineering_examples():
    """Prompt 工程示例"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API 密钥未设置，跳过 Prompt 工程示例")
        return
    
    print("3. Prompt 工程示例:")
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # 3.1 零样本学习
    print("3.1 零样本学习:")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "请将以下文本翻译成英文：\"大语言模型正在改变世界。\""}
            ],
            temperature=0.7,
            max_tokens=100
        )
        print("翻译结果:")
        print(response.choices[0].message["content"])
    except Exception as e:
        print(f"错误: {e}")
    
    # 3.2 少样本学习
    print("\n3.2 少样本学习:")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "请将以下中文翻译成英文：\n中文：你好\n英文：Hello\n\n中文：谢谢\n英文：Thank you\n\n中文：大语言模型正在改变世界。\n英文："}
            ],
            temperature=0.7,
            max_tokens=100
        )
        print("翻译结果:")
        print(response.choices[0].message["content"])
    except Exception as e:
        print(f"错误: {e}")
    
    # 3.3 思维链
    print("\n3.3 思维链:")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "请解决以下数学问题，并详细说明你的思考过程：\n小明有5个苹果，小红给了他3个，然后他吃了2个，请问小明现在有多少个苹果？"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        print("解题过程:")
        print(response.choices[0].message["content"])
    except Exception as e:
        print(f"错误: {e}")
    
    # 3.4 角色设定
    print("\n3.4 角色设定:")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "你是一位专业的美食评论家，请评价以下菜品：\n麻婆豆腐："}
            ],
            temperature=0.7,
            max_tokens=200
        )
        print("美食评论:")
        print(response.choices[0].message["content"])
    except Exception as e:
        print(f"错误: {e}")
    
    # 3.5 结构化输出
    print("\n3.5 结构化输出:")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "请分析以下文本的情感，并以JSON格式输出：\n\"今天天气真好，我心情非常愉快！\"\n\n输出格式：\n{\n  \"text\": \"\",\n  \"sentiment\": \"\",\n  \"confidence\": 0.0\n}\n"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        print("情感分析结果:")
        print(response.choices[0].message["content"])
    except Exception as e:
        print(f"错误: {e}")
    
    print("=" * 50)

# 主函数
def main():
    """主函数"""
    # 检查API密钥
    keys = check_api_keys()
    
    # 运行示例
    openai_example()
    anthropic_example()
    prompt_engineering_examples()
    
    print("\n所有示例运行完成！")
    print("提示：")
    print("1. 如需运行OpenAI示例，请在.env文件中设置OPENAI_API_KEY")
    print("2. 如需运行Anthropic示例，请在.env文件中设置ANTHROPIC_API_KEY")
    print("3. 您可以使用以下命令安装依赖：pip install openai anthropic python-dotenv")

if __name__ == "__main__":
    main()
