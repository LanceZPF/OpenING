import os
from openai import AzureOpenAI

## An example for using LLM API based on Azure OpenAI service

# 配置API连接信息
API_KEY = "XXX"  # 请替换为实际的API密钥
AZURE_ENDPOINT = "https://guohe-apim.azure-api.net"
API_VERSION = "2025-01-01-preview"

def test_text_completion():
    """测试文本对话功能"""
    try:
        # 创建Azure OpenAI客户端
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION
        )
        
        # 发送文本对话请求
        response = client.chat.completions.create(
            model="gpt-4o",  # 使用支持的模型
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": "Who were the founders of Microsoft?"}
            ]
        )
        
        print("=== 文本对话测试 ===")
        print(f"回复: {response.choices[0].message.content}")
        print(f"使用模型: {response.model}")
        print(f"Token使用: {response.usage}")
        
    except Exception as e:
        print(f"文本对话测试失败: {e}")

def test_image_analysis():
    """测试图片分析功能"""
    try:
        # 创建Azure OpenAI客户端
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION
        )
        
        # 这里需要替换为实际的base64编码图片内容
        # 示例：将图片转换为base64格式
        import base64
        
        # 读取图片文件并转换为base64（示例）
        with open("assets\\overview_opening.jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Assistant is a large language model trained by OpenAI."
                        }
                    ]
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What is in the picture?"
                        }
                    ]
                }
            ]
        )
        
        print("=== 图片分析测试 ===")
        print(f"回复: {response.choices[0].message.content}")
        print(f"使用模型: {response.model}")
        
    except Exception as e:
        print(f"图片分析测试失败: {e}")

def test_different_models():
    """测试不同模型"""
    models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]
    
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION
    )
    
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            )
            print(f"模型 {model} 测试成功: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            print(f"模型 {model} 测试失败: {e}")

def main():
    """主函数"""
    print("开始测试Azure OpenAI API...")
    print("=" * 50)
    
    # 测试文本对话
    # test_text_completion()
    # print()
    
    # 测试图片分析（需要提供实际的图片数据）
    test_image_analysis()
    print()
    
    # 测试不同模型
    # print("=== 测试不同模型 ===")
    # test_different_models()
    
    print("=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    main()
