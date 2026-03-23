from generation.llm import critique, generate
from generation.prompt_builder import build_prompt
from generation.citation import format_response
from retreiver.retreival import retrieve
from config import TOP_K, MAX_ITERATIONS


def needs_retrieval(query: str) -> bool:
    prompt = f"""You are working with a document question-answering system.
The user is asking a question that should be answered using the provided documents.

Should we search the documents to answer this question?
Answer 'yes' for ALL of these cases:
- Questions about specific terms, clauses, or sections in a document
- Requests to simplify, explain, or summarize something from a document
- Questions about people, companies, agreements, or contracts
- Any legal or technical term that may be defined in the document

Answer 'no' ONLY for completely general questions with no document context needed:
- Pure math ("what is 2+2")
- General world knowledge ("what is the capital of France")

Question: {query}
Reply with only 'yes' or 'no'."""
    result = critique(prompt)
    print(f"  [RETRIEVAL decision] → {result}")
    return "yes" in result


def is_answer_supported(answer: str, chunks: list[dict]) -> bool:
    prompt = f"""Does this answer contain at least one [Chunk N] citation?
Answer: {answer}
Reply 'yes' if you see any [Chunk N] pattern in the answer.
Reply 'no' only if there are zero citations.
Reply with only 'yes' or 'no'."""
    result = critique(prompt)
    print(f"  [ISSUP decision]     → {result}")
    return "yes" in result                         


def is_answer_complete(query: str, answer: str) -> bool:
    prompt = f"""Does this answer make a reasonable attempt to address the question using document content?
Question: {query}
Answer: {answer}
Reply 'yes' if the answer is relevant and substantive.
Reply 'no' only if the answer is empty, refuses to answer, or is completely off-topic.
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