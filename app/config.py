"""全局配置：路径、LLM、Embedding 等，全部从环境变量读取。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DB_PATH = BASE_DIR / "contracts.db"

# ===== LLM（DeepSeek，OpenAI 兼容）=====
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ===== Embedding =====
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()  # local | api
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ===== RAG 检索参数 =====
TOP_K = int(os.getenv("TOP_K", "3"))
