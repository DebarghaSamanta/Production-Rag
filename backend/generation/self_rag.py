from generation.llm import critique, generate
from generation.prompt_builder import build_prompt
from generation.citation import format_response
from retreiver.retreival import retrieve
from config import TOP_K, MAX_ITERATIONS


def needs_retrieval(query: str) -> bool:
    prompt = f"""Does answering this question require searching through documents?
Question: {query}
Reply with only 'yes' or 'no'."""
    result = critique(prompt)
    print(f"  [RETRIEVAL decision] → {result}")
    return "yes" in result


def is_answer_supported(answer: str, chunks: list[dict]) -> bool:
    context = "\n\n".join(f"Chunk {i+1}: {c['text'][:300]}" for i, c in enumerate(chunks))
    prompt = f"""Is every claim in the answer directly supported by the context below?
Context:
{context}

Answer: {answer}

Reply with only 'yes', 'no', or 'partial'."""
    result = critique(prompt)
    print(f"  [ISSUP decision]     → {result}")
    return "yes" in result


def is_answer_complete(query: str, answer: str) -> bool:
    prompt = f"""Does this answer fully and completely address the question?
Question: {query}
Answer: {answer}
Reply with only 'yes' or 'no'."""
    result = critique(prompt)
    print(f"  [ISUSE decision]     → {result}")
    return "yes" in result


def refine_query(original: str, iteration: int) -> str:
    prompt = f"""This search query did not return a complete answer (attempt {iteration}).
Rephrase it to be more specific for document retrieval.
Original query: {original}
Rephrased query (return only the rephrased query, nothing else):"""
    refined = critique(prompt)
    print(f"  [Query refined]      → {refined}")
    return refined


def self_rag_query(user_query: str) -> dict:
    print(f"\n[Self-RAG] '{user_query}'")

    # Decision 1 — does this need retrieval?
    if not needs_retrieval(user_query):
        print("[Self-RAG] Answering directly without retrieval...")
        answer = generate([
            {"role": "system", "content": "Answer the question directly and concisely."},
            {"role": "user",   "content": user_query},
        ])
        return {"query": user_query, "answer": answer, "citations": [], "iterations": 0}

    current_query = user_query
    best_response = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[Self-RAG] Iteration {iteration}/{MAX_ITERATIONS}")

        chunks = retrieve(current_query, top_k=TOP_K)
        print(f"  [Retrieval]          → {len(chunks)} chunks")

        if not chunks:
            print("  No chunks retrieved, stopping.")
            break

        messages = build_prompt(user_query, chunks)
        answer   = generate(messages)

        # Decision 2 — is answer grounded in chunks?
        supported = is_answer_supported(answer, chunks)

        # Decision 3 — is answer complete?
        complete  = is_answer_complete(user_query, answer)

        best_response = format_response(user_query, answer, chunks)
        best_response["iterations"] = iteration

        if supported and complete:
            print(f"[Self-RAG] Answer accepted at iteration {iteration}")
            return best_response

        if iteration < MAX_ITERATIONS:
            print(f"[Self-RAG] Answer not sufficient, refining query...")
            current_query = refine_query(user_query, iteration)

    print(f"[Self-RAG] Max iterations reached, returning best answer.")
    return best_response