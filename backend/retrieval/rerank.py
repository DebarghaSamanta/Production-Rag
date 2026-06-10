from sentence_transformers import CrossEncoder

# bge-reranker-base is a strong, CPU-friendly cross-encoder from the same
# BAAI family as your bge-small-en-v1.5 embedder.
# It outputs a raw logit score — higher = more relevant. No fixed scale.
# Alternatives if this is too slow on CPU: cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_MODEL = "BAAI/bge-reranker-base"

# How many parent paragraphs come in from fusion
INPUT_FROM_FUSION = 10

# How many to pass to the generation layer
TOP_K_FINAL = 3


class Reranker:
    def __init__(self):
        print(f"[Reranker] Loading cross-encoder: {RERANKER_MODEL}...")
        # CrossEncoder reads query + document TOGETHER in one forward pass.
        # This is fundamentally more accurate than bi-encoders (which embed
        # query and document separately) but too slow to run on 50+ candidates.
        # That's why we only run it on the top 10 from fusion, not on raw results.
        self.model = CrossEncoder(RERANKER_MODEL)
        print("[Reranker] Cross-encoder loaded.")

    def rerank(self, raw_query: str, parent_chunks: list[dict]) -> list[dict]:
        """
        Takes the top 10 parent paragraphs from fusion and re-scores each one
        by reading the query and paragraph TOGETHER through a cross-encoder.

        Why cross-encoder here and not earlier:
        - ChromaDB/BM25 never "read" query+document together — they just compare
          pre-computed representations. Fast but imprecise.
        - A cross-encoder does a full attention pass over both texts jointly,
          catching subtle relevance signals that bi-encoders miss entirely.
        - But cross-encoders are O(n) inference calls — 1 per candidate.
          Running this on 60 raw chunks would be too slow for real-time API use.
          Running on 10 post-fusion candidates takes ~1–2 seconds on CPU.

        Args:
            raw_query:     The ORIGINAL user question (not the rewritten variants).
                           Cross-encoders work best with natural language questions.
            parent_chunks: Output list from HybridFusion.fuse()

        Returns:
            Top TOP_K_FINAL chunks re-sorted by cross-encoder score, with a
            "rerank_score" field added to each dict.
        """
        if not parent_chunks:
            return []

        print(f"\n[Reranker] Scoring {len(parent_chunks)} candidates with cross-encoder...")

        # Build (query, document) pairs — one per candidate
        pairs = [(raw_query, chunk["text"]) for chunk in parent_chunks]

        # Single batched inference call — more efficient than calling predict() in a loop
        scores = self.model.predict(pairs)

        # Attach score to each chunk
        scored_chunks = []
        for chunk, score in zip(parent_chunks, scores):
            chunk = chunk.copy()
            chunk["rerank_score"] = float(score)
            scored_chunks.append(chunk)

        # Sort descending — higher cross-encoder score = more relevant
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        top = scored_chunks[:TOP_K_FINAL]

        print(f"[Reranker] Cut {len(parent_chunks)} → {len(top)} final context chunks.")
        for i, c in enumerate(top, 1):
            print(f"  [{i}] Score: {c['rerank_score']:.4f} | "
                  f"Parent: {c['metadata'].get('parent_id')} | "
                  f"Section: {c['metadata'].get('section_id')}")

        return top


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrieval.rewrite import QueryRewriter
    from retrieval.search_dense import DenseRetriever
    from retrieval.search_sparse import SparseRetriever
    from retrieval.fusion import HybridFusion

    raw_query = "what loss function is used for regularization"

    queries       = QueryRewriter().generate_search_queries(raw_query)
    dense_results = DenseRetriever().search(queries)
    sparse_results = SparseRetriever().search(queries)
    top_parents   = HybridFusion().fuse(dense_results, sparse_results)

    reranker      = Reranker()
    final_context = reranker.rerank(raw_query, top_parents)

    print("\n--- FINAL CONTEXT FOR GENERATION ---")
    for i, chunk in enumerate(final_context, 1):
        print(f"\n[{i}] Rerank Score: {chunk['rerank_score']:.4f}")
        print(f"     Source:  {chunk['metadata'].get('source')}")
        print(f"     Section: {chunk['metadata'].get('section_id')} | "
              f"Parent: {chunk['metadata'].get('parent_id')}")
        print(f"     Text:\n{chunk['text'][:500]}")