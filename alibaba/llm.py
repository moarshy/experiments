import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ALIBABA_API_KEY"), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-max", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'}],
    )
    
print(completion.model_dump_json())