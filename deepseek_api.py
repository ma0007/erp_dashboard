import requests
from types import SimpleNamespace

class DeepSeekClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or "sk-0ba9207ae8e94772ace86a5c727452a1"
        self.base_url = "https://api.deepseek.com/v1"
        self.chat = self.Chat(self)
        
    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)
            
        class Completions:
            def __init__(self, client):
                self.client = client
                
            def create(self, *, model, messages, temperature, max_tokens):
                headers = {
                    "Authorization": f"Bearer {self.client.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = requests.post(
                    f"{self.client.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"DeepSeek API Error: {response.text}")
                    
                response_data = response.json()
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=response_data["choices"][0]["message"]["content"]
                            )
                        )
                    ]
                )