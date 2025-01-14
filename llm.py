import torch
import sys
import os
import soundfile as sf
from transformers import pipeline
from typing import Tuple, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, model_name: Optional[str] = "meta-llama/Llama-3.2-1B-Instruct"):
        """
        Initialize Text to Audio converter.
        
        Args:
            model_dir (str): Path to Kokoro model directory
            sample_rate (int): Audio sample rate
        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_name
        self.pipe = pipeline("text-generation", model=model_name, device=self.device)
        
    
    def load_model(self, model_name: str) -> None:
        """Load model by name."""
        self.model = model_name
        self.pipe = pipeline("text-generation", model=model_name, device=self.device)
        
    
    def generate(self, messages: list[dict]) -> list[dict]:
        """
        Generate text from conversation.
        
        Args:
            List[dict]: List of text/conversation
        Returns:
            list[dict]: Conversation list with generated text
        """ 
        try:
            return self.pipe(messages)[0]['generated_text']
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
        
    def new_conversation(self, user_query) -> list[dict]:
        """Start a new conversation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. You are going to talk to a user."},
            {"role": "user", "content": user_query},
        ]
        return messages
    
    def add_user_query(self, messages, user_query) -> list[dict]:
        """Add user query to conversation."""
        messages.append({"role": "user", "content": user_query})
        return messages
        
if __name__ == "__main__":
    llm = LLM()
    messages = llm.new_conversation("Hello, my name is Amir. What is your name?")
    messages = llm.generate(messages)
    messages = llm.add_user_query(messages, "Nice to meet you, can I call you John?")
    messages = llm.generate(messages)
    print(messages)