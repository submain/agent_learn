import os
import requests
from dotenv import load_dotenv
from langchain_deepseek import DeepSeekChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

class SimpleChatAgent:
    def __init__(self, name="智能助手", role="你是一个 helpful 的智能助手"):
        self.name = name
        self.role = role
        self.history = []
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
    
    def add_message(self, role, content):
        """添加消息到对话历史"""
        self.history.append({"role": role, "content": content})
    
    def generate_response(self, user_input):
        """生成响应"""
        # 添加用户输入到历史
        self.add_message("user", user_input)
        
        # 构建完整的对话历史
        messages = [{"role": "system", "content": self.role}]
        messages.extend(self.history)
        
        # 调用DeepSeek API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()  # 检查请求是否成功
        
        # 获取并添加助手响应到历史
        assistant_response = response.json()["choices"][0]["message"]["content"].strip()
        self.add_message("assistant", assistant_response)
        
        return assistant_response
    
    def chat(self):
        """启动聊天会话"""
        print(f"{self.name}: 你好！我是{self.name}，有什么可以帮助你的吗？")
        
        while True:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "再见", "bye"]:
                print(f"{self.name}: 再见！")
                break
            
            try:
                response = self.generate_response(user_input)
                print(f"{self.name}: {response}")
            except Exception as e:
                print(f"{self.name}: 抱歉，出现了错误: {str(e)}")

if __name__ == "__main__":
    print("=== LangChain + DeepSeek 示例 ===")
    
    # 检查API密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请在.env文件中设置DEEPSEEK_API_KEY")
        exit(1)
    
    # 执行LangChain示例
    try:
        llm = DeepSeekChat(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.7
        )
        
        template = "你是一个{role}，请{task}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"role": "教师", "task": "解释什么是智能体"})
        print("LangChain + DeepSeek 结果:")
        print(result)
        print("-" * 50)
    except Exception as e:
        print(f"LangChain 示例错误: {str(e)}")
    
    print("\n=== 简单聊天智能体示例 ===")
    # 执行聊天智能体示例
    try:
        agent = SimpleChatAgent()
        agent.chat()
    except Exception as e:
        print(f"聊天智能体错误: {str(e)}")
