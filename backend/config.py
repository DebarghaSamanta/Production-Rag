from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DATA_DIR  = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR    = DATA_DIR / "chroma_db"

# ── OpenRouter — main generation ───────────────────────────────────────────────
# Currently pointing at Groq because OpenRouter rate limit hit
# When OpenRouter resets, change these back:
#   OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
#   OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
#   LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_API_KEY: str  = os.getenv("GROQ_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://api.groq.com/openai/v1"
LLM_MODEL: str           = "llama-3.1-8b-instant"
LLM_TEMPERATURE: float   = 0.0
LLM_MAX_TOKENS: int      = 400    # keep low to prevent 8b model looping

# ── Groq — critique calls ──────────────────────────────────────────────────────
GROQ_API_KEY: str        = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str       = "https://api.groq.com/openai/v1"
GROQ_MODEL: str          = "llama-3.3-70b-versatile"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int     = 500
CHUNK_OVERLAP: int  = 100
MIN_CHUNK_SIZE: int = 50

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str      = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS: int = 384
EMBEDDING_BATCH_SIZE: int = 64

# ── Vector Store ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "rag_documents"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K: int                  = 8
SIMILARITY_THRESHOLD: float = 0.45

# ── Hybrid Search ──────────────────────────────────────────────────────────────
USE_HYBRID_SEARCH: bool = True

# ── Reranker ───────────────────────────────────────────────────────────────────
USE_RERANKER: bool       = True
RERANKER_MODEL: str      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_CANDIDATES: int = 15   # retrieve this many, rerank down to TOP_K
RERANKER_TOP_K: int      = 5    # final chunks passed to LLM after reranking

# ── Self-RAG ───────────────────────────────────────────────────────────────────
USE_SELF_RAG: bool  = True
MAX_ITERATIONS: int = 2

# ── Citation ───────────────────────────────────────────────────────────────────
MAX_CITATION_CHUNKS: int = 3

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = """You are a precise document assistant.
Answer in 3-5 sentences maximum. Do not repeat yourself.
Use ONLY the provided chunks. Cite each sentence with [Chunk N].
Stop writing once the question is fully answered."""

PROMPT_TEMPLATE: str = """
Context chunks (use these to answer):
{context}

Question: {question}

Instructions:
- Answer using ONLY the chunks above
- After EVERY sentence write [Chunk N] where N is the exact number shown in the context
- The chunk number MUST match the label shown above — do not guess or default to [Chunk 1]
- Use regular square brackets only: [Chunk 1] not (Chunk 1)

Answer:
"""