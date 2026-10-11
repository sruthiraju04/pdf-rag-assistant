import os
from pathlib import Path

from dotenv import load_dotenv

# Keep noisy tokenizer/runtime warnings out of the Streamlit UI without hiding
# every Python warning from the project.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
TEMP_DIR = BASE_DIR / "temp"

DB_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

APP_TITLE = "LangChain PDF RAG Chatbot"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL",
    "openai/text-embedding-3-small",
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
DEFAULT_RETRIEVAL_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "4"))


def is_configured(api_key: str = None) -> bool:
    """Check whether an OpenRouter API key is available."""
    key = api_key or OPENROUTER_API_KEY
    return bool(key and key.strip())
