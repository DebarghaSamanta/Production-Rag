# core/config.py
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "chroma_db", "bm25_index.pkl")
MONITOR_DB_PATH = os.path.join(BASE_DIR, "monitoring", "rag_monitor.db")
# Embedding Configuration
# bge-small-en-v1.5 is an excellent, fast model for retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking Configuration
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200

CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50