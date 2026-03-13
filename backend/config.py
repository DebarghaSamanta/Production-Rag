from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv() 
#--------GROQ API----------
OPENROUTER_API_KEY: str  = os.getenv("GROQ_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://api.groq.com/openai/v1"

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
TOP_K: int = 12            
SIMILARITY_THRESHOLD: float = 0.45 
#Generation
LLM_MODEL: str         = "llama-3.1-8b-instant" 
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

USE_HYBRID_SEARCH: bool = True

SYSTEM_PROMPT: str = """You are a document analysis assistant.
You MUST only use information from the numbered chunks provided.
If the chunks do not contain the answer, you MUST respond with exactly:
"I could not find this in the provided documents."
Quoting from memory or general knowledge is strictly forbidden.
Every sentence must end with [Chunk N]."""

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