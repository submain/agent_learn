# LLM基础示例

本目录包含《LLM基础》文档中示例代码的可直接运行版本。

## 文件说明

- `llm_examples.py` - 完整的示例代码，包含：
  - OpenAI API 使用示例
  - Anthropic API 使用示例
  - Prompt 工程示例（零样本学习、少样本学习、思维链、角色设定、结构化输出）

- `.env.example` - 环境变量配置示例文件

- `LLM基础.md` - 原始文档

## 运行步骤

1. **配置环境变量**
   - 复制 `.env.example` 文件并重命名为 `.env`
   - 在 `.env` 文件中填写您的 API 密钥

2. **安装依赖**
   ```bash
   # 确保已激活虚拟环境
   venv\Scripts\Activate.ps1
   
   # 安装所需依赖
   pip install openai anthropic python-dotenv
   ```

3. **运行示例**
   ```bash
   python llm_examples.py
   ```

## API密钥获取

- **OpenAI API Key**：访问 [OpenAI 平台](https://platform.openai.com/account/api-keys)
- **Anthropic API Key**：访问 [Anthropic 控制台](https://console.anthropic.com/account/keys)
- **DeepSeek API Key**：访问 [DeepSeek 平台](https://platform.deepseek.com/)

## 注意事项

- 运行示例需要有效的 API 密钥
- 不同的 API 可能会产生费用，请合理使用
- 示例代码已包含错误处理，即使某些 API 密钥未设置也能正常运行其他示例

## 示例输出

运行示例后，您将看到类似以下的输出：

```
=== LLM 基础示例 ===
==================================================
API密钥状态:
✓ OPENAI_API_KEY: 已设置
✗ ANTHROPIC_API_KEY: 未设置
==================================================
1. OpenAI API 示例:
OpenAI 响应:
大语言模型（Large Language Model，LLM）是一种基于深度学习的人工智能模型，通过训练大量文本数据来理解和生成人类语言。这类模型通常具有数十亿甚至数千亿个参数，能够捕捉语言的复杂模式和语义关系。

大语言模型的主要特点包括：
1. **语言理解能力**：能够理解上下文、识别意图、处理歧义
2. **生成能力**：能够生成连贯、相关且符合语境的文本
3. **多任务处理**：可以执行翻译、摘要、问答、创作等多种语言任务
4. **少样本学习**：只需少量示例即可适应新任务

==================================================
```
