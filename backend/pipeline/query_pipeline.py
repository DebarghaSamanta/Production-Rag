import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from config import USE_SELF_RAG, TOP_K
from generation.citation import render


def query(user_query: str, top_k: int = TOP_K) -> dict:
    if USE_SELF_RAG:
        from generation.self_rag import self_rag_query
        return self_rag_query(user_query)

    # linear pipeline — untouched fallback
    from retreiver.retreival import retrieve
    from retreiver.reranker import rerank
    from generation.prompt_builder import build_prompt
    from generation.llm import generate
    from generation.citation import format_response

    chunks   = retrieve(user_query, top_k=top_k)
    messages = build_prompt(user_query, chunks)
    answer   = generate(messages)
    return format_response(user_query, answer, chunks)


if __name__ == "__main__":
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    response = query(user_query)
    print("\n" + "=" * 60)
    print(render(response))