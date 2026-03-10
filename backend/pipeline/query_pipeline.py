import sys
from pathlib import Path
from retreiver.retreival import retrieve
from retreiver.reranker import rerank
from generation.prompt_builder import build_prompt
from generation.llm import generate
from generation.citation import format_response, render
from config import TOP_K


def query(user_query: str, top_k: int = TOP_K, reranking: bool = False) -> dict:
    print(f"[1/4] Retrieving top {top_k} chunks...")
    chunks = retrieve(user_query, top_k=top_k)
    print(f"      → {len(chunks)} chunks retrieved")

    if reranking:
        print(f"[2/4] Reranking...")
        chunks = rerank(user_query, chunks)
    else:
        print(f"[2/4] Reranking skipped")

    print(f"[3/4] Generating answer...")
    messages = build_prompt(user_query, chunks)
    answer = generate(messages)

    print(f"[4/4] Building citations...")
    response = format_response(user_query, answer, chunks)

    return response


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your question: ")
    response = query(user_query)
    print("\n" + "="*60)
    print(render(response))