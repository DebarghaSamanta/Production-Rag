from ingestion.embedder import embed_query
from store.vector_store import query as vector_query
from config import TOP_K, SIMILARITY_THRESHOLD, USE_HYBRID_SEARCH, USE_RERANKER, RERANKER_CANDIDATES, RERANKER_TOP_K


def _deduplicate(chunks: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for chunk in chunks:
        fingerprint = chunk["text"][:80].strip()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(chunk)
    return unique


def retrieve(user_query: str, top_k: int = TOP_K) -> list[dict]:
    print(f"  [USE_HYBRID]   → {USE_HYBRID_SEARCH}")
    print(f"  [USE_RERANKER] → {USE_RERANKER}")

    # Step 1 — get candidates
    # fetch more candidates when reranking so it has room to filter
    candidate_k = RERANKER_CANDIDATES if USE_RERANKER else top_k

    if USE_HYBRID_SEARCH:
        from retreiver.hybrid import hybrid_search
        chunks = hybrid_search(user_query, top_k=candidate_k)
    else:
        embedding = embed_query(user_query)
        chunks    = vector_query(embedding, candidate_k)
        chunks    = [h for h in chunks if h["score"] >= SIMILARITY_THRESHOLD]

    # Step 2 — deduplicate overlapping chunks
    chunks = _deduplicate(chunks)

    # Step 3 — rerank candidates down to RERANKER_TOP_K
    if USE_RERANKER:
        from retreiver.reranker import rerank
        chunks = rerank(user_query, chunks, top_k=RERANKER_TOP_K)

    return chunks