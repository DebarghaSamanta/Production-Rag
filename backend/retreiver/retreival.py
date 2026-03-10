from ingestion.embedder import embed_query
from store.vector_store import query
from config import TOP_K, SIMILARITY_THRESHOLD


def retrieve(user_query: str, top_k: int = TOP_K) -> list[dict]:
    query_embedding = embed_query(user_query)
    hits = query(query_embedding, top_k)
    return [h for h in hits if h["score"] >= SIMILARITY_THRESHOLD]