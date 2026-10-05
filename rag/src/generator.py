import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

from src.config import LLM_MODEL_NAME


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set.")

llm = ChatOpenRouter(
    model=LLM_MODEL_NAME,
    api_key=OPENROUTER_API_KEY
)
