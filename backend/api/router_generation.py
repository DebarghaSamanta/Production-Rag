import time
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
from gaurdrails.input_gaurd import InputGuard
from gaurdrails.output_gaurd import OutputGuard
from monitoring.logger import log_query

router = APIRouter()

print("[GenerationRouter] Loading pipeline components...")
rewriter         = QueryRewriter()
dense_retriever  = DenseRetriever()
sparse_retriever = SparseRetriever()
fusion           = HybridFusion()
reranker         = Reranker()
generator        = AnswerGenerator()
critic           = AnswerCritic()
input_guard      = InputGuard()
output_guard     = OutputGuard()
print("[GenerationRouter] All components ready.")

MAX_SELF_RAG_ITERATIONS = 2


class GenerationRequest(BaseModel):
    query: str


class CitationEntry(BaseModel):
    source:     str
    section_id: int
    parent_id:  str

class RetrievedChunk(BaseModel):
    source:     str
    section_id: int
    parent_id:  str
    text:       str

class GenerationResponse(BaseModel):
    original_query:      str
    answer:              str
    citation_map:        dict[str, CitationEntry]
    context_chunks:      list[RetrievedChunk] 
    prompt_version:      str
    self_rag_iterations: int
    critique:            dict
    input_guard:         dict
    output_guard:        dict
    telemetry:           dict


async def _retrieve(queries: list[str]):
    dense_results, sparse_results = await asyncio.gather(
        asyncio.to_thread(dense_retriever.search, queries),
        asyncio.to_thread(sparse_retriever.search, queries),
    )
    top_parents   = await asyncio.to_thread(fusion.fuse, dense_results, sparse_results)
    final_context = await asyncio.to_thread(reranker.rerank, queries[0], top_parents)
    return final_context, len(dense_results), len(sparse_results)


def _broaden_query(query: str, iteration: int) -> list[str]:
    if iteration == 1:
        return [query + " overview explanation background"]
    return [" ".join(query.split()[:4])]


@router.post("/generate", response_model=GenerationResponse)
async def run_generation_pipeline(payload: GenerationRequest):
    start_ms  = time.time()
    raw_query = payload.query.strip()

    # ── Input guardrail ──────────────────────────────────────────
    in_guard = input_guard.check(raw_query)
    if not in_guard.passed:
        log_query(
            query=raw_query, answer="", context_chunks=[],
            self_rag_iterations=0,
            critique={"faithfulness": 0, "completeness": 0, "verdict": "BLOCKED"},
            input_guard_passed=False, output_guard_passed=False,
            guard_reason=in_guard.reason, telemetry={},
            prompt_version="N/A", latency_ms=0,
        )
        raise HTTPException(status_code=400, detail=in_guard.reason)

    try:
        # ── Initial retrieval ────────────────────────────────────
        rewritten_queries = await asyncio.to_thread(
            rewriter.generate_search_queries, raw_query
        )
        context_chunks, dense_count, sparse_count = await _retrieve(rewritten_queries)

        # ── Self-RAG loop ────────────────────────────────────────
        iteration        = 0
        best_gen_result  = None
        best_critique    = None
        out_guard_result = None

        while iteration <= MAX_SELF_RAG_ITERATIONS:
            gen_result = await asyncio.to_thread(
                generator.generate, raw_query, context_chunks
            )
            critique_result = await asyncio.to_thread(
                critic.critique,
                raw_query, gen_result["answer"],
                context_chunks, gen_result["context_block"],
            )
            out_guard_result = output_guard.check(
                gen_result["answer"], critique_result, context_chunks
            )

            best_gen_result = gen_result
            best_critique   = critique_result

            if critique_result["verdict"] == "PASS" and out_guard_result.passed:
                break

            if iteration < MAX_SELF_RAG_ITERATIONS and out_guard_result.should_retry:
                broadened = _broaden_query(raw_query, iteration + 1)
                context_chunks, dense_count, sparse_count = await _retrieve(broadened)

            iteration += 1

        # ── Telemetry ────────────────────────────────────────────
        top_rerank = max(
            (c.get("rerank_score", 0.0) for c in context_chunks), default=0.0
        )
        telemetry = {
            "rewritten_queries":   rewritten_queries,
            "context_chunks_used": len(context_chunks),
            "dense_raw_count":     dense_count,
            "sparse_raw_count":    sparse_count,
            "top_rerank_score":    round(top_rerank, 4),
            "final_verdict":       best_critique["verdict"],
        }
        latency_ms = int((time.time() - start_ms) * 1000)

        # ── Log ──────────────────────────────────────────────────
        log_query(
            query=raw_query,
            answer=best_gen_result["answer"],
            context_chunks=context_chunks,          # ← real chunks now
            self_rag_iterations=iteration,
            critique=best_critique,
            input_guard_passed=True,
            output_guard_passed=out_guard_result.passed,
            guard_reason=out_guard_result.reason,
            telemetry=telemetry,
            prompt_version=best_gen_result["prompt_version"],
            latency_ms=latency_ms,
        )

        citation_map = {
            str(k): CitationEntry(**v)
            for k, v in best_gen_result["citation_map"].items()
        }

        return GenerationResponse(
            original_query      = raw_query,
            answer              = best_gen_result["answer"],
            citation_map        = citation_map,
            context_chunks      = [                        
                RetrievedChunk(
                    source     = c["metadata"].get("source", ""),
                    section_id = c["metadata"].get("section_id", -1),
                    parent_id  = c["metadata"].get("parent_id", ""),
                    text       = c.get("text", ""),
                )
                for c in context_chunks
            ],
            prompt_version      = best_gen_result["prompt_version"],
            self_rag_iterations = iteration,
            critique            = best_critique,
            input_guard         = {"passed": True, "reason": ""},
            output_guard        = {
                "passed": out_guard_result.passed,
                "reason": out_guard_result.reason,
            },
            telemetry=telemetry,
        )

    except Exception as e:
        print(f"[GenerationRouter] ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")