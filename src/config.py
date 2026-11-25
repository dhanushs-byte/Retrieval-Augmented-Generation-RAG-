from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
QDRANT_PATH = os.getenv("QDRANT_PATH", str(ROOT_DIR / "qdrant_data"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "week7_documents")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
TOP_K = int(os.getenv("TOP_K", "4"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
