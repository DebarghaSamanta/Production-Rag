import os
import pickle
from core.config import CHROMA_DB_DIR

PARENT_MAP_PATH = os.path.join(CHROMA_DB_DIR, "parent_chunks_map.pkl")

# RRF constant — 60 is the standard value from the original RRF paper.
# It dampens the impact of very high ranks so a single dominant result
# doesn't drown out everything else.
RRF_K = 60

# How many post-RRF results to consider when counting section density.
# We look at the top 20 to find clusters, not the full 50+ raw list.
DENSITY_WINDOW = 20

# How much each additional chunk from the same section boosts the score.
# e.g. 3 chunks from section 4 → multiplier = 1.0 + (0.15 × 3) = 1.45
DENSITY_BOOST_FACTOR = 0.15

# Final number of parent paragraphs passed to the reranker.
TOP_PARENTS = 10


class HybridFusion:
    def __init__(self):
        print(f"[HybridFusion] Loading parent map from {PARENT_MAP_PATH}...")
        with open(PARENT_MAP_PATH, "rb") as f:
            self.parent_map: dict[str, str] = pickle.load(f)
        print(f"[HybridFusion] Parent map loaded. {len(self.parent_map)} parent chunks available.")

    # ── Step 1: Reciprocal Rank Fusion ──────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict]
    ) -> list[dict]:
        """
        Merges two ranked lists (dense + sparse) into one normalized score.

        Why RRF instead of weighted sum:
        Dense scores are cosine distances (0.0–2.0, lower = better).
        BM25 scores are Okapi TF-IDF values (unbounded, higher = better).
        These scales are completely incompatible — you cannot add them directly.

        RRF sidesteps this entirely by only caring about RANK, not raw score.
        Formula: RRF(chunk) = Σ 1 / (k + rank_in_list)
        A chunk ranked #1 in both lists gets: 1/61 + 1/61 = 0.0328
        A chunk ranked #1 in dense but #40 in sparse gets: 1/61 + 1/100 = 0.026
        This naturally rewards consistent performance across both retrievers.
        """
        rrf_scores: dict[str, float] = {}
        chunk_registry: dict[str, dict] = {}  # id → full chunk dict for later lookup

        # Dense list: rank by ascending distance (lower distance = better = rank 1)
        sorted_dense = sorted(dense_results, key=lambda x: x["score"])
        for rank, chunk in enumerate(sorted_dense, start=1):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            chunk_registry[cid] = chunk

        # Sparse list: rank by descending BM25 score (higher = better = rank 1)
        sorted_sparse = sorted(sparse_results, key=lambda x: x["score"], reverse=True)
        for rank, chunk in enumerate(sorted_sparse, start=1):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            chunk_registry[cid] = chunk

        # Build merged list sorted by RRF score descending
        merged = []
        for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = chunk_registry[cid].copy()
            chunk["rrf_score"] = rrf_score
            merged.append(chunk)

        print(f"[Fusion] RRF merged {len(dense_results)} dense + "
              f"{len(sparse_results)} sparse → {len(merged)} unique chunks.")
        return merged

    # ── Step 2: Section-Density Boost ───────────────────────────────────────

    def _apply_density_boost(self, merged_chunks: list[dict]) -> list[dict]:
        """
        Boosts chunks that belong to sections with multiple hits in the top results.

        The intuition: if 4 different child chunks from Section 7 all ranked
        highly across both dense and sparse retrieval, Section 7 is genuinely
        relevant — not a fluke keyword match. We reward cohesion.

        We only count density within the top DENSITY_WINDOW (20) results to
        avoid letting a large section game the system just by having more chunks.

        Multiplier formula:
            multiplier = 1.0 + (DENSITY_BOOST_FACTOR × section_chunk_count)
        e.g. section with 4 chunks in top 20 → 1.0 + (0.15 × 4) = 1.60×
        """
        # Count section_id occurrences within the top DENSITY_WINDOW results only
        window = merged_chunks[:DENSITY_WINDOW]
        section_counts: dict[int, int] = {}
        for chunk in window:
            sec_id = chunk["metadata"].get("section_id", -1)
            section_counts[sec_id] = section_counts.get(sec_id, 0) + 1

        print(f"[Fusion] Section density map (top {DENSITY_WINDOW}): {section_counts}")

        # Apply multiplier to ALL chunks (not just the window), so boosted
        # sections can pull lower-ranked chunks up into the final top 10
        boosted = []
        for chunk in merged_chunks:
            sec_id = chunk["metadata"].get("section_id", -1)
            count = section_counts.get(sec_id, 0)
            multiplier = 1.0 + (DENSITY_BOOST_FACTOR * count)
            chunk = chunk.copy()
            chunk["final_score"] = chunk["rrf_score"] * multiplier
            chunk["density_multiplier"] = multiplier
            boosted.append(chunk)

        # Re-sort by final boosted score
        boosted.sort(key=lambda x: x["final_score"], reverse=True)
        return boosted

    # ── Step 3: Deduplicate to parents ──────────────────────────────────────

    def _deduplicate_to_parents(self, boosted_chunks: list[dict]) -> list[dict]:
        """
        Multiple child chunks often point to the same parent paragraph.
        Example: child _S7_P2_C0, _S7_P2_C1, _S7_P2_C3 all map to parent _S7_P2.

        We collapse these by keeping only the highest-scoring child as the
        representative for each parent_id. This prevents the LLM from receiving
        the same paragraph 3 times and wasting its context window.
        """
        seen_parents: dict[str, dict] = {}

        for chunk in boosted_chunks:
            parent_id = chunk["metadata"].get("parent_id")
            if parent_id not in seen_parents:
                seen_parents[parent_id] = chunk  # first occurrence = highest score

        unique = list(seen_parents.values())
        print(f"[Fusion] Deduplicated {len(boosted_chunks)} child chunks → "
              f"{len(unique)} unique parent references.")
        return unique

    # ── Step 4: Swap child text → full parent paragraph ─────────────────────

    def _swap_to_parent_text(self, deduplicated: list[dict]) -> list[dict]:
        """
        The child chunks are tiny (250 chars) — useful for retrieval precision
        but too small to give an LLM meaningful context.

        We swap each child's text for its full parent paragraph (1500 chars)
        using the parent_chunks_map.pkl lookup table built during ingestion.

        If a parent_id isn't found in the map (shouldn't happen, but defensive),
        we keep the child text rather than crashing.
        """
        swapped = []
        missing = 0

        for chunk in deduplicated:
            parent_id = chunk["metadata"].get("parent_id")
            parent_text = self.parent_map.get(parent_id)

            if parent_text:
                chunk = chunk.copy()
                chunk["text"] = parent_text
                chunk["retrieved_as"] = "parent"
            else:
                missing += 1
                chunk["retrieved_as"] = "child_fallback"

            swapped.append(chunk)

        if missing:
            print(f"[Fusion] WARNING: {missing} parent_ids not found in map. "
                  f"Child text used as fallback.")

        return swapped

    # ── Public entry point ───────────────────────────────────────────────────

    def fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict]
    ) -> list[dict]:
        """
        Full fusion pipeline. Call this with the raw outputs of DenseRetriever
        and SparseRetriever. Returns top TOP_PARENTS parent paragraphs, ready
        for the reranker.

        Returns list of dicts with shape:
            {
                "id":                 str,   # child chunk id (for traceability)
                "text":               str,   # FULL parent paragraph text (swapped)
                "metadata":           dict,  # source, section_id, parent_id, etc.
                "rrf_score":          float,
                "final_score":        float, # rrf_score × density_multiplier
                "density_multiplier": float,
                "retrieved_as":       str    # "parent" or "child_fallback"
            }
        """
        print(f"\n[HybridFusion] Starting fusion pipeline...")

        # Step 1: RRF merge
        merged = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Step 2: Section-density boost
        boosted = self._apply_density_boost(merged)

        # Step 3: Deduplicate to unique parents
        deduplicated = self._deduplicate_to_parents(boosted)

        # Step 4: Swap child text → full parent context
        final = self._swap_to_parent_text(deduplicated)

        # Slice to top TOP_PARENTS
        top = final[:TOP_PARENTS]

        print(f"[HybridFusion] Pipeline complete. Returning top {len(top)} parent paragraphs.\n")
        return top


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrieval.search_dense import DenseRetriever
    from retrieval.search_sparse import SparseRetriever

    query = "what loss function is used for regularization"

    from retrieval.rewrite import QueryRewriter
    queries = QueryRewriter().generate_search_queries(query)
    print(f"Rewritten queries: {queries}")

    dense_results  = DenseRetriever().search(queries)
    sparse_results = SparseRetriever().search(queries)

    fusion = HybridFusion()
    top_parents = fusion.fuse(dense_results, sparse_results)

    print("\n--- FUSION OUTPUT (Top Parents) ---")
    for i, p in enumerate(top_parents, 1):
        print(f"\n[{i}] ID: {p['id']}")
        print(f"     Final Score: {p['final_score']:.5f}  "
              f"(RRF: {p['rrf_score']:.5f} × {p['density_multiplier']:.2f}x density)")
        print(f"     Section: {p['metadata'].get('section_id')}  |  "
              f"Parent: {p['metadata'].get('parent_id')}")
        print(f"     Text preview: {p['text'][:300]}")