from sentence_transformers import CrossEncoder
from config import USE_RERANKER, RERANKER_MODEL, RERANKER_TOP_K

_model = None   # lazy load — only downloads when first used


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"  [Reranker] Loading {RERANKER_MODEL}...")
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = RERANKER_TOP_K) -> list[dict]:
    if not USE_RERANKER or not chunks:
        return chunks[:top_k]

    model  = _get_model()
    pairs  = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

    print(f"  [Reranker] {len(chunks)} → {top_k} chunks | "
          f"top score: {ranked[0]['rerank_score']} | "
          f"bottom score: {ranked[top_k-1]['rerank_score']}")

    return ranked[:top_k]