# backend/api/router_query.py
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from retrieval.rewrite import QueryRewriter
from retrieval.search_dense import DenseRetriever
from retrieval.search_sparse import SparseRetriever
from retrieval.fusion import HybridFusion
from retrieval.rerank import Reranker

router = APIRouter()

# All four components are loaded once at startup when FastAPI imports this router.
# Loading them per-request would reload the BGE cross-encoder and BM25 index
# on every single query — adding 5–10 seconds of overhead each time.
print("[QueryRouter] Loading retrieval pipeline components...")
rewriter       = QueryRewriter()
dense_retriever  = DenseRetriever()
sparse_retriever = SparseRetriever()
fusion           = HybridFusion()
reranker         = Reranker()
print("[QueryRouter] All components ready.")


class QueryRequest(BaseModel):
    query: str


class ChunkContext(BaseModel):
    """
    Represents one final parent paragraph that will be passed to the LLM.
    This is the exact payload shape your generation layer will consume.
    """
    parent_id:        str
    source:           str
    section_id:       int
    text:             str
    rerank_score:     float
    final_rrf_score:  float


class QueryResponse(BaseModel):
    original_query:   str
    rewritten_queries: list[str]
    context_chunks:   list[ChunkContext]
    # Pipeline telemetry — useful for debugging retrieval quality
    telemetry: dict


@router.post("/query", response_model=QueryResponse)
async def run_retrieval_pipeline(payload: QueryRequest):
    """
    Full retrieval pipeline endpoint. Accepts a raw user question,
    runs all 4 phases, and returns the final context chunks ready
    for your generation layer.

    Phases:
      1. Query rewriting  (Groq / Llama 3)
      2. Parallel retrieval (ChromaDB dense + BM25 sparse)
      3. Fusion           (RRF + section-density boost + parent swap)
      4. Reranking        (BGE cross-encoder)
    """
    raw_query = payload.query.strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        print(f"\n{'='*60}")
        print(f"[QueryRouter] New query: '{raw_query}'")

        # ── Phase 1: Query Rewriting ─────────────────────────────────
        rewritten_queries = rewriter.generate_search_queries(raw_query)
        print(f"[QueryRouter] Rewritten into {len(rewritten_queries)} queries.")

        # ── Phase 2: Parallel Retrieval ──────────────────────────────
        # Run dense and sparse retrieval concurrently in a thread pool.
        # Both are CPU/IO bound (model inference + disk reads), so we
        # use asyncio.to_thread to avoid blocking the FastAPI event loop.
        dense_results, sparse_results = await asyncio.gather(
            asyncio.to_thread(dense_retriever.search, rewritten_queries),
            asyncio.to_thread(sparse_retriever.search, rewritten_queries),
        )
        print(f"[QueryRouter] Dense: {len(dense_results)} chunks | "
              f"Sparse: {len(sparse_results)} chunks")

        # ── Phase 3: Fusion ──────────────────────────────────────────
        top_parents = await asyncio.to_thread(
            fusion.fuse, dense_results, sparse_results
        )

        # ── Phase 4: Reranking ───────────────────────────────────────
        final_context = await asyncio.to_thread(
            reranker.rerank, raw_query, top_parents
        )

        # ── Build response ───────────────────────────────────────────
        context_chunks = []
        for chunk in final_context:
            meta = chunk["metadata"]
            context_chunks.append(ChunkContext(
                parent_id       = meta.get("parent_id", ""),
                source          = meta.get("source", ""),
                section_id      = meta.get("section_id", -1),
                text            = chunk["text"],
                rerank_score    = round(chunk["rerank_score"], 5),
                final_rrf_score = round(chunk.get("final_score", 0.0), 5),
            ))

        return QueryResponse(
            original_query    = raw_query,
            rewritten_queries = rewritten_queries,
            context_chunks    = context_chunks,
            telemetry={
                "dense_raw_count":   len(dense_results),
                "sparse_raw_count":  len(sparse_results),
                "post_fusion_count": len(top_parents),
                "post_rerank_count": len(final_context),
            }
        )

    except Exception as e:
        print(f"[QueryRouter] ERROR: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval pipeline failed: {str(e)}"
        )