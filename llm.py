import torch
import sys
import os
import soundfile as sf
from transformers import pipeline
from typing import Tuple, Optional
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLM:
    """Base class for LLM implementations."""
    def new_conversation(self, user_query: str) -> list[dict]:
        """Start a new conversation."""
        return [
            {"role": "system", "content": "You are a helpful assistant. You are going to talk to a user."},
            {"role": "user", "content": user_query},
        ]

    def add_user_query(self, messages: list[dict], user_query: str) -> list[dict]:
        """Add user query to conversation."""
        messages.append({"role": "user", "content": user_query})
        return messages
    
    def add_assistant_reply(self, messages: list[dict], assistant_reply: str) -> list[dict]:
        """Add assistant reply to conversation."""
        messages.append({"role": "assistant", "content": assistant_reply})
        return messages
    

class LLMModel(LLM):
    """Subclass for using a local model."""
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe = pipeline("text-generation", model=model_name, device=self.device)
    
    def generate(self, messages: list[dict]) -> str:
        """Generate text from conversation using the local model."""
        try:
            prompt = messages[-1]["content"]
            return self.pipe(prompt)[0]['generated_text']
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

class LLMAPI(LLM):
    """Subclass for using an API endpoint."""
    def __init__(self, base_url: str, api_key: str = 'your-api-key', model: str = 'llama-3.2-3b-instruct'):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key, 
        )
        
    def set_model(self, model: str):
        """Set the model to use for generating text."""
        self.model = model
    
    def generate(self, messages: list[dict]) -> list[str]:
        """Generate text from conversation using an API endpoint."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            response = completion.choices[0].message.content
            return self.add_assistant_reply(messages, response)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
        

if __name__ == "__main__":
    type = 'api' # or 'local'
    if type == 'local':
        # Example usage with LLMModel
        local_llm = LLMModel()
        messages = local_llm.new_conversation("Hello, what is the capital of France?")
        reply = local_llm.generate(messages)
        print(f"Local Model Reply: {reply}")
        
    if type == 'api':
        # Example usage with LLMAPI 
        api_llm = LLMAPI(base_url="http://127.0.0.1:1234", api_key='your-api-key', model='llama-3.2-3b-instruct')
        messages = api_llm.new_conversation("Hello, what is the capital of France?")
        reply = api_llm.generate(messages)
        print(f"API Reply: {reply}")