from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class OpenAILLM:
    def __init__(self, api_key: str, model_base_url: str, model_name: str):
        self.api_key = api_key or os.getenv("api_key")
        self.base_url = model_base_url or os.getenv("model_base_url")
        self.model_name = model_name or os.getenv("model_name")
        self.max_token = os.getenv("max_token")
        self.token = 0 # 当前消耗的token
        
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            model=self.model_name
        )
        
    def chat(self, message: dict):
        new_token = len(message["content"].encode("utf-8"))
        self.token += new_token
        if self.token > self.max_token:
            self.messages.append({
                "role": "user",
                "content": f"现在消上下文对话太长了，"
            })
            message["content"] = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=self.messages
                                )
            new_token = len(message["content"].encode("utf-8"))
        self.messages.append(message)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )
        content = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": content
        })
        return content
    
    def delete_messages(self, index: int):
        self.messages.pop(index)
        return True
    
    def get_messages(self):
        return self.messages
    
    def clear_messages(self):
        self.messages = []
        return True
    
    def get_message(self, index: int):
        return self.messages[index]