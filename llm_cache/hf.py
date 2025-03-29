# # Other models:
# # meta-llama/Llama-2-7b-chat-hf
# # mistralai/Mistral-7B-v0.1
# # mistralai/Mistral-7B-Instruct
# # meta-llama/Meta-Llama-3-8B

import os
import requests

class HuggingFaceChat:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B"):
        self.api_key = os.getenv("HUGGING_FACE_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        self.model_name = model_name

    def get_response(self, message, max_new_tokens=200):
        data = {
            "inputs": message,
            "parameters": {"max_new_tokens": max_new_tokens}
        }
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model_name}",
            headers=self.headers,
            json=data
        )
        return response.json()
        # return response