from config import TOP_K
RERANKER = "none"   # "none" | "cross_encoder"


def rerank(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    if RERANKER == "cross_encoder":
        return _cross_encoder_rerank(query, chunks, top_k)
    return chunks[:top_k]


def _cross_encoder_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]