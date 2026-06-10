import chromadb
from core.config import CHROMA_DB_DIR
from ingestion.embedder import BGEEmbedder

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

COLLECTION_NAME = "arxiv_papers_collection"
RESULTS_PER_QUERY = 15  # 3 queries × 15 = up to 45 raw candidates


class DenseRetriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        self.embedder = BGEEmbedder()

    def _embed_queries(self, queries: list[str]) -> list[list[float]]:
        """
        Prepend the BGE query prefix to each query before embedding.
        This is critical for retrieval quality with bge-small-en-v1.5 —
        document embeddings are stored without the prefix, but query
        embeddings must use it to land in the right part of the vector space.
        """
        prefixed = [BGE_QUERY_PREFIX + q for q in queries]
        return self.embedder.embed_batch(prefixed)

    def search(self, queries: list[str], n_results: int = RESULTS_PER_QUERY) -> list[dict]:
        """
        Embeds all rewritten queries and fires a separate ChromaDB query for each.
        Returns a flat, deduplicated list of raw child chunk dicts.

        Each returned dict has the shape:
            {
                "id":       str,   # e.g. "paper.pdf_S2_P1_C3"
                "text":     str,   # the child chunk text
                "metadata": dict,  # source, parent_id, section_id, chunk_id, type
                "score":    float  # cosine distance (lower = more similar)
            }
        """
        if not queries:
            return []

        print(f"\n[DenseRetriever] Embedding {len(queries)} rewritten queries...")
        query_embeddings = self._embed_queries(queries)

        raw_results: list[dict] = []
        seen_ids: set[str] = set()

        for i, embedding in enumerate(query_embeddings):
            print(f"  → Querying ChromaDB with vector {i + 1}/{len(queries)}...")
            response = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # ChromaDB returns nested lists (one sub-list per query embedding).
            # We sent one embedding at a time, so we always take index [0].
            ids        = response["ids"][0]
            documents  = response["documents"][0]
            metadatas  = response["metadatas"][0]
            distances  = response["distances"][0]

            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    raw_results.append({
                        "id":       chunk_id,
                        "text":     text,
                        "metadata": metadata,
                        "score":    distance  # cosine distance; lower = better match
                    })

        print(f"[DenseRetriever] Returned {len(raw_results)} unique child chunks from ChromaDB.")
        return raw_results


# Quick smoke test
if __name__ == "__main__":
    retriever = DenseRetriever()

    test_queries = [
        "what loss function is used for regularization",
        "regularization techniques penalty terms machine learning optimization",
        "L1 L2 weight decay dropout cross-entropy loss"
    ]

    results = retriever.search(test_queries)

    print("\n--- DENSE RETRIEVAL SAMPLE ---")
    for r in results[:5]:
        print(f"\nID:    {r['id']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Meta:  {r['metadata']}")
        print(f"Text:  {r['text']}")
    