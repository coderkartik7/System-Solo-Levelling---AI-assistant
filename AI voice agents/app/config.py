import os
from dotenv import load_dotenv

load_dotenv()
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
GEMINI_API_KEY = os.getenv("G_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT",8000))

# if not ASSEMBLY_API_KEY:
#     raise ValueError("Missing ASSEMBLY_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("Missing GEMINI_API_KEY")
# if not MURF_API_KEY:
#     raise ValueError("Missing MURF_API_KEY")
# if not SEARCH_API_KEY:
#     raise ValueError("Missing SEARCH_API_KEY")