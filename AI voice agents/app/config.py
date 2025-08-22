import os
from dotenv import load_dotenv

load_dotenv()
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
GEMINI_API_KEY = os.getenv("G_API_KEY")
HOST = "127.0.0.1"
PORT = 8000

if not ASSEMBLY_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing required API keys: ASSEMBLY_API_KEY, G_API_KEY")