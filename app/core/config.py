import os

from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("APP_KEY")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))
