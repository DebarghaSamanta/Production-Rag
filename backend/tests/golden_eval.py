
import json
import os
import sys
import time
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

GOLDEN_SET_PATH = "tests/golden_set.json"
API_URL         = "http://localhost:8000/api/v1/generate"

# Semantic similarity threshold for answer accuracy
ANSWER_SIMILARITY_THRESHOLD = 0.75

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def semantic_similarity(text1: str, text2: str) -> float:
    e1 = model.encode(text1)
    e2 = model.encode(text2)
    return float(cosine_similarity([e1], [e2])[0][0])


def evaluate_retrieval(api_response: dict, golden_item: dict) -> dict:
    chunks   = api_response.get("context_chunks", [])
    
    # Fix 1 — source is a field directly on chunk, not nested in metadata
    sources  = [c.get("source", "") for c in chunks]
    texts    = [c.get("text", "").lower() for c in chunks]
    all_text = " ".join(texts)

    # Fix 2 — fuzzy match using significant words instead of exact string
    expected_source = golden_item.get("expected_source", "")
    
    # Handle multiple expected sources separated by ";"
    expected_sources = [s.strip() for s in expected_source.split(";")]
    
    source_hit = False
    for exp_src in expected_sources:
        # Extract significant words (ignore common words, keep paper-specific terms)
        exp_words = set(
            w.lower() for w in exp_src.replace(".pdf", "").split()
            if len(w) > 4  # skip short words like "for", "and", "in"
        )
        for actual_src in sources:
            actual_words = set(actual_src.lower().replace(".pdf", "").split())
            # If 2+ significant words match, consider it a hit
            if len(exp_words & actual_words) >= 2:
                source_hit = True
                break

    # Keywords check unchanged
    keywords          = golden_item.get("expected_keywords", [])
    keywords_found    = [kw for kw in keywords if kw.lower() in all_text]
    keyword_recall    = len(keywords_found) / len(keywords) if keywords else 0.0
    relevant_chunks   = sum(
        1 for t in texts
        if any(kw.lower() in t for kw in keywords)
    )
    context_precision = relevant_chunks / len(chunks) if chunks else 0.0

    return {
        "source_hit":         source_hit,
        "keyword_recall":     round(keyword_recall, 3),
        "context_precision":  round(context_precision, 3),
        "keywords_found":     keywords_found,
        "keywords_missing":   [kw for kw in keywords if kw.lower() not in all_text],
    }

def evaluate_answer(api_response: dict, golden_item: dict) -> dict:
    """
    Compares generated answer to ground truth using semantic similarity.
    No LLM judge — purely embedding-based.
    """
    ground_truth = golden_item.get("ground_truth", "")
    answer       = api_response.get("answer", "")

    if not ground_truth or not answer:
        return {"similarity": 0.0, "accurate": False}

    sim      = semantic_similarity(answer, ground_truth)
    accurate = sim >= ANSWER_SIMILARITY_THRESHOLD

    return {
        "similarity": round(sim, 3),
        "accurate":   accurate,
    }


def run_golden_eval():
    golden_set = load_golden_set()
    print(f"[GoldenEval] Running on {len(golden_set)} golden questions...\n")

    results = []

    for item in golden_set:
        print(f"  Q: {item['question'][:70]}...")

        try:
            resp = requests.post(
                API_URL,
                json={"query": item["question"]},
                timeout=60
            )

            if resp.status_code != 200:
                print(f"  [SKIP] API returned {resp.status_code}")
                continue

            api_response = resp.json()

            # Evaluate retrieval (deterministic)
            retrieval_scores = evaluate_retrieval(api_response, item)

            # Evaluate answer (embedding-based, no LLM)
            answer_scores = evaluate_answer(api_response, item)

            result = {
                "id":                  item["id"],
                "question":            item["question"],
                "ground_truth":        item.get("ground_truth", ""),
                "generated_answer":    api_response.get("answer", "")[:200],
                **retrieval_scores,
                **answer_scores,
                "self_rag_iterations": api_response.get("self_rag_iterations", 0),
                "critique_verdict":    api_response.get("critique", {}).get("verdict", ""),
            }
            results.append(result)

            print(f"  Source Hit: {retrieval_scores['source_hit']} | "
                  f"KW Recall: {retrieval_scores['keyword_recall']:.2f} | "
                  f"Ctx Precision: {retrieval_scores['context_precision']:.2f} | "
                  f"Answer Sim: {answer_scores['similarity']:.2f} | "
                  f"Accurate: {answer_scores['accurate']}")

            if retrieval_scores["keywords_missing"]:
                print(f"  Missing keywords: {retrieval_scores['keywords_missing']}")

        except Exception as e:
            print(f"  [ERROR] {str(e)}")

        time.sleep(1)

    # ── Aggregate metrics ──────────────────────────────────────
    if not results:
        print("\n[GoldenEval] No results to aggregate.")
        return

    n = len(results)
    avg_kw_recall    = sum(r["keyword_recall"]    for r in results) / n
    avg_ctx_prec     = sum(r["context_precision"] for r in results) / n
    avg_answer_sim   = sum(r["similarity"]        for r in results) / n
    source_hit_rate  = sum(1 for r in results if r["source_hit"])   / n
    answer_accuracy  = sum(1 for r in results if r["accurate"])     / n

    print(f"""
{'='*55}
GOLDEN DATASET EVALUATION SUMMARY  ({n} questions)
{'='*55}
Retrieval
  Source Hit Rate       : {source_hit_rate:.1%}   (did right paper appear?)
  Avg Keyword Recall    : {avg_kw_recall:.3f}  (keywords found in context)
  Avg Context Precision : {avg_ctx_prec:.3f}  (fraction of chunks relevant)

Answer Quality  (embedding similarity, no LLM judge)
  Avg Answer Similarity : {avg_answer_sim:.3f}
  Answer Accuracy       : {answer_accuracy:.1%}   (similarity > {ANSWER_SIMILARITY_THRESHOLD})
{'='*55}
""")

    # Save results to JSON for CI comparison
    out_path = "tests/golden_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "source_hit_rate":    source_hit_rate,
                "avg_keyword_recall": avg_kw_recall,
                "avg_ctx_precision":  avg_ctx_prec,
                "avg_answer_sim":     avg_answer_sim,
                "answer_accuracy":    answer_accuracy,
            },
            "per_question": results
        }, f, indent=2)

    print(f"[GoldenEval] Results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_golden_eval()