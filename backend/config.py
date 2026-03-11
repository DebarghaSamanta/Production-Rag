"""
If a number or string appears more than once in the codebase change here.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv() 
#--------OPen router from en----------
OPENROUTER_API_KEY: str  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

#----------paths----------------
PROJECT_ROOT     = Path(__file__).parent
DATA_DIR         = PROJECT_ROOT / "data"
RAW_DATA_DIR     = DATA_DIR / "raw"         
PROCESSED_DIR    = DATA_DIR / "processed"   
CHROMA_DIR       = DATA_DIR / "chroma_db"   

#Chunking
CHUNK_SIZE: int    = 500    
CHUNK_OVERLAP: int = 100    
MIN_CHUNK_SIZE: int = 50    

#Embedding
EMBEDDING_MODEL: str       = "BAAI/bge-small-en-v1.5"  
EMBEDDING_DIMENSIONS: int  = 384                  
EMBEDDING_BATCH_SIZE: int  = 64                  
 #Vector store
CHROMA_COLLECTION_NAME: str = "rag_documents"  
#Retrieval
TOP_K: int = 8                 
SIMILARITY_THRESHOLD: float = 0.0   
#Generation
LLM_MODEL: str         = "meta-llama/llama-3.3-70b-instruct:free" 
LLM_TEMPERATURE: float = 0.0   
LLM_MAX_TOKENS: int    = 1024 
MAX_CITATION_CHUNKS: int = 3   
#self Rag part
GROQ_API_KEY: str        = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str       = "https://api.groq.com/openai/v1"
GROQ_MODEL: str          = "llama-3.3-70b-versatile"
#Self rag
USE_SELF_RAG: bool   = True
MAX_ITERATIONS: int  = 2
#Citation
MAX_CITATION_CHUNKS: int = 3

SYSTEM_PROMPT: str = """You are a precise research assistant. Answer the user's question
using ONLY the context chunks provided below. Do not use prior knowledge.

Rules:
- If the answer is in the context, answer clearly and cite the source.
- If the answer is NOT in the context, say: "I could not find this in the provided documents."
- Never speculate or hallucinate facts.
- Keep your answer concise and factual.
"""

PROMPT_TEMPLATE: str = """
Context chunks (use these to answer):
{context}

Question: {question}

Answer (you MUST cite sources using exactly this format: [Chunk 1], [Chunk 2] — use regular square brackets only):
"""