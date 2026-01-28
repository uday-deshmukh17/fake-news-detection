import os
from dotenv import load_dotenv

class Config:
    """Application configuration"""
    def __init__(self):
        # Load environment variables
        load_dotenv(override=True)
        
        # Model Configuration
        self.MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-2-7b-chat-hf")
        self.HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        self.DEVICE = os.getenv("DEVICE", "auto")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        # API Configuration
        self.NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
        self.API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/classify")

# Create global config instance
app_config = Config()