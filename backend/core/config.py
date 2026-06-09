# core/config.py
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "chroma_db", "bm25_index.pkl")

# Embedding Configuration
# bge-small-en-v1.5 is an excellent, fast model for retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200