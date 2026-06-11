import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from retrieval.rewrite import QueryRewriter
from retrieval.search_dense import DenseRetriever
from retrieval.search_sparse import SparseRetriever
from retrieval.fusion import HybridFusion
from retrieval.rerank import Reranker
from generation.generator import AnswerGenerator
from generation.critic import AnswerCritic

router = APIRouter()

# ── Load all components once at startup ─────────────────────────────────────
print("[GenerationRouter] Loading pipeline components...")
rewriter         = QueryRewriter()
dense_retriever  = DenseRetriever()
sparse_retriever = SparseRetriever()
fusion           = HybridFusion()
reranker         = Reranker()
generator        = AnswerGenerator()
critic           = AnswerCritic()
print("[GenerationRouter] All components ready.")

MAX_SELF_RAG_ITERATIONS = 2   # re-retrieve at most twice before giving up


class GenerationRequest(BaseModel):
    query: str


class CitationEntry(BaseModel):
    source:     str
    section_id: int
    parent_id:  str


class GenerationResponse(BaseModel):
    original_query:    str
    answer:            str
    citation_map:      dict[str, CitationEntry]
    prompt_version:    str
    self_rag_iterations: int         # how many re-retrieval loops were needed
    critique:          dict          # final critique scores
    telemetry:         dict


# ── Internal helpers ─────────────────────────────────────────────────────────

async def _retrieve(queries: list[str]) -> list[dict]:
    """Runs parallel dense+sparse retrieval → fusion → rerank. Returns final context."""
    dense_results, sparse_results = await asyncio.gather(
        asyncio.to_thread(dense_retriever.search, queries),
        asyncio.to_thread(sparse_retriever.search, queries),
    )
    top_parents   = await asyncio.to_thread(fusion.fuse, dense_results, sparse_results)
    final_context = await asyncio.to_thread(reranker.rerank, queries[0], top_parents)
    return final_context


def _broaden_query(query: str, iteration: int) -> list[str]:
    """
    When self-RAG critique fails, we re-retrieve with a broadened query
    instead of re-running the rewriter. Each iteration loosens specificity.

    Iteration 1: add "overview explanation background"
    Iteration 2: strip to core noun phrase only
    This is simple and deterministic — no extra LLM call needed.
    """
    if iteration == 1:
        return [query + " overview explanation background"]
    else:
        # Take just the first 4 words as a bare keyword fallback
        core = " ".join(query.split()[:4])
        return [core]


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerationResponse)
async def run_generation_pipeline(payload: GenerationRequest):
    """
    Full RAG + Self-RAG pipeline.

    Flow:
      1. Rewrite query → retrieve → rerank  (initial retrieval)
      2. Generate answer from top context chunks
      3. Critique: is the answer faithful and complete?
         - PASS → return answer
         - FAIL → broaden query → re-retrieve (no rewriter, just broadened string)
                → regenerate → critique again
         Repeat up to MAX_SELF_RAG_ITERATIONS times.
      4. Return best answer found (even if final critique is FAIL, we return
         with the critique scores so the caller knows quality was low)
    """
    raw_query = payload.query.strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        print(f"\n{'='*60}")
        print(f"[GenerationRouter] Query: '{raw_query}'")

        # ── Phase 1: Initial retrieval ───────────────────────────────
        rewritten_queries = await asyncio.to_thread(
            rewriter.generate_search_queries, raw_query
        )
        context_chunks = await _retrieve(rewritten_queries)

        # ── Self-RAG loop ────────────────────────────────────────────
        iteration     = 0
        best_answer   = None
        best_critique = None
        best_gen_result = None

        while iteration <= MAX_SELF_RAG_ITERATIONS:
            print(f"\n[GenerationRouter] Generation attempt {iteration + 1}...")

            # Generate
            gen_result = await asyncio.to_thread(
                generator.generate, raw_query, context_chunks
            )

            # Critique
            critique_result = await asyncio.to_thread(
                critic.critique,
                raw_query,
                gen_result["answer"],
                context_chunks,
                gen_result["context_block"],
            )

            print(f"[GenerationRouter] Critique → "
                  f"Faithfulness: {critique_result['faithfulness']}/5 | "
                  f"Completeness: {critique_result['completeness']}/5 | "
                  f"Verdict: {critique_result['verdict']}")
            print(f"  Reason: {critique_result['reason']}")

            best_answer     = gen_result
            best_critique   = critique_result
            best_gen_result = gen_result

            if critique_result["verdict"] == "PASS":
                print(f"[GenerationRouter] PASS on iteration {iteration + 1}. Done.")
                break

            # FAIL → re-retrieve with broadened query (no rewriter call)
            if iteration < MAX_SELF_RAG_ITERATIONS:
                print(f"[GenerationRouter] FAIL. Re-retrieving with broadened query "
                      f"(iteration {iteration + 1}/{MAX_SELF_RAG_ITERATIONS})...")
                broadened = _broaden_query(raw_query, iteration + 1)
                context_chunks = await _retrieve(broadened)

            iteration += 1

        # ── Build response ───────────────────────────────────────────
        citation_map = {
            str(k): CitationEntry(**v)
            for k, v in best_gen_result["citation_map"].items()
        }

        return GenerationResponse(
            original_query       = raw_query,
            answer               = best_gen_result["answer"],
            citation_map         = citation_map,
            prompt_version       = best_gen_result["prompt_version"],
            self_rag_iterations  = iteration,
            critique             = best_critique,
            telemetry={
                "rewritten_queries":    rewritten_queries,
                "context_chunks_used":  len(context_chunks),
                "final_verdict":        best_critique["verdict"],
                "faithfulness_score":   best_critique["faithfulness"],
                "completeness_score":   best_critique["completeness"],
            }
        )

    except Exception as e:
        print(f"[GenerationRouter] ERROR: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation pipeline failed: {str(e)}"
        )