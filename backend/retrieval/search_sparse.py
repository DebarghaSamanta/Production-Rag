import pickle
from core.config import BM25_INDEX_PATH

RESULTS_PER_QUERY = 20  # 3 queries × 20 = up to 60 raw candidates


class SparseRetriever:
    def __init__(self):
        print(f"[SparseRetriever] Loading BM25 index from {BM25_INDEX_PATH}...")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)

        # Unpack the serialized BM25 bundle written by DocumentIndexer.
        # Structure matches exactly what indexer.py saves:
        #   { "bm25_instance", "documents", "metadatas", "ids" }
        self.bm25      = bm25_data["bm25_instance"]
        self.documents = bm25_data["documents"]   # list[str]
        self.metadatas = bm25_data["metadatas"]   # list[dict]
        self.ids       = bm25_data["ids"]         # list[str]
        print(f"[SparseRetriever] BM25 index loaded. Corpus size: {len(self.documents)} child chunks.")

    def _tokenize(self, text: str) -> list[str]:
        """
        Must match the tokenizer used at index time in DocumentIndexer._tokenize_text.
        Both use simple lowercase whitespace splitting — no stemming, no stopwords —
        to ensure query tokens map correctly onto the indexed corpus tokens.
        """
        return text.lower().split()

    def search(self, queries: list[str], n_results: int = RESULTS_PER_QUERY) -> list[dict]:
        """
        Tokenizes each rewritten query and scores the entire BM25 corpus against it.
        Returns a flat, deduplicated list of the top-scoring raw child chunk dicts.

        Each returned dict has the shape:
            {
                "id":       str,   # e.g. "paper.pdf_S2_P1_C3"
                "text":     str,   # the child chunk text
                "metadata": dict,  # source, parent_id, section_id, chunk_id, type
                "score":    float  # BM25 relevance score (higher = more relevant)
            }
        """
        if not queries:
            return []

        raw_results: list[dict] = []
        seen_ids: set[str] = set()

        for i, query in enumerate(queries):
            print(f"  → BM25 scoring query {i + 1}/{len(queries)}: '{query[:60]}...'")
            tokenized_query = self._tokenize(query)

            # get_scores returns a numpy array of shape (corpus_size,).
            # Each element is the BM25 Okapi score for that corpus document.
            scores = self.bm25.get_scores(tokenized_query)

            # argsort is ascending; we reverse with [::-1] to get top-scoring first.
            ranked_indices = scores.argsort()[::-1][:n_results]

            for idx in ranked_indices:
                chunk_id = self.ids[idx]
                bm25_score = float(scores[idx])

                # Skip chunks with zero relevance — they matched no query tokens at all.
                if bm25_score <= 0.0:
                    continue

                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    raw_results.append({
                        "id":       chunk_id,
                        "text":     self.documents[idx],
                        "metadata": self.metadatas[idx],
                        "score":    bm25_score
                    })

        print(f"[SparseRetriever] Returned {len(raw_results)} unique child chunks from BM25.")
        return raw_results


# Quick smoke test
if __name__ == "__main__":
    retriever = SparseRetriever()

    test_queries = [
        "what loss function is used for regularization",
        "regularization techniques penalty terms machine learning optimization",
        "L1 L2 weight decay dropout cross-entropy loss"
    ]

    results = retriever.search(test_queries)

    print("\n--- SPARSE RETRIEVAL SAMPLE ---")
    for r in results[:5]:
        print(f"\nID:    {r['id']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Meta:  {r['metadata']}")
        print(f"Text:  {r['text'][:200]}")