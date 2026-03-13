from ingestion.embedder import embed_query
from store.vector_store import query as vector_query
from config import TOP_K, SIMILARITY_THRESHOLD, USE_HYBRID_SEARCH
from retreiver.hybrid import hybrid_search

def retrieve(user_query: str, top_k: int = TOP_K) -> list[dict]:
    if USE_HYBRID_SEARCH:
        
        return hybrid_search(user_query, top_k=top_k)

    
    embedding = embed_query(user_query)
    hits      = vector_query(embedding, top_k)
    return [h for h in hits if h["score"] >= SIMILARITY_THRESHOLD]