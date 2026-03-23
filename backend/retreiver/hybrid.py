from retreiver.bm_25 import bm25_search
from store.vector_store import query as vector_query
from ingestion.embedder import embed_query
from config import TOP_K

RRF_K = 60  # standard constant — dampens impact of rank position

INSTRUCTION_WORDS = {
    "simplify", "explain", "summarize", "describe", "list",
    "what", "how", "why", "when", "who", "is", "are", "the",
    "tell", "me", "about", "define", "give", "show", "mean", "do", "you"
}

def _clean_for_bm25(query: str) -> str:
    tokens = query.lower().split()
    cleaned = [t for t in tokens if t not in INSTRUCTION_WORDS]
    return " ".join(cleaned) if cleaned else query


def _rrf_score(bm25_rank: int | None, vector_rank: int | None) -> float:
    score = 0.0
    if bm25_rank is not None:
        score += 1 / (RRF_K + bm25_rank)
    if vector_rank is not None:
        score += 1 / (RRF_K + vector_rank)
    return score


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    # --- BM25 results ---
    bm25_query = _clean_for_bm25(query)
    print(f"  [BM25 query]   → '{bm25_query}'")
    bm25_results = bm25_search(bm25_query, top_k=top_k)  
    
    query_embedding = embed_query(query)                  
    vector_results  = vector_query(query_embedding, top_k=top_k)

    for rank, chunk in enumerate(vector_results):
        chunk["vector_rank"] = rank + 1

    # --- Build lookup maps keyed by (source, chunk_index) ---
    bm25_map    = {(c["source"], c["chunk_index"]): (i + 1, c) for i, c in enumerate(bm25_results)}
    vector_map  = {(c["source"], c["chunk_index"]): (i + 1, c) for i, c in enumerate(vector_results)}

    all_keys = set(bm25_map.keys()) | set(vector_map.keys())

    # --- RRF fusion ---
    fused = []
    for key in all_keys:
        bm25_rank   = bm25_map[key][0]   if key in bm25_map   else None
        vector_rank = vector_map[key][0] if key in vector_map else None

        # prefer vector chunk dict (has score), fall back to bm25
        chunk = (vector_map[key][1] if key in vector_map else bm25_map[key][1]).copy()

        chunk["bm25_rank"]   = bm25_rank
        chunk["vector_rank"] = vector_rank
        chunk["rrf_score"]   = round(_rrf_score(bm25_rank, vector_rank), 6)

        fused.append(chunk)

    fused.sort(key=lambda c: c["rrf_score"], reverse=True)

    return fused[:top_k]