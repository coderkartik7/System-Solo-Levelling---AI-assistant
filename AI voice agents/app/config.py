import os
from dotenv import load_dotenv

load_dotenv()
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT",8000))