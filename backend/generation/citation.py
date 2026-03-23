import re


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    cited_indices = set(int(n) - 1 for n in re.findall(r"\[Chunk (\d+)\]", answer))

    citations = []
    for i in cited_indices:
        if 0 <= i < len(chunks):
            citations.append({
                "chunk_number": i + 1,
                "source":       chunks[i]["source"],
                "chunk_index":  chunks[i]["chunk_index"],
                "score":        chunks[i].get("score"),
                "text":         chunks[i]["text"],
            })

    return sorted(citations, key=lambda c: c["chunk_number"])


def render(response: dict) -> str:
    lines = [
        f"Question: {response['query']}",
        f"\nAnswer:\n{response['answer']}",
        "\nSources:",
    ]
    for c in response["citations"]:
        lines.append(f"\n  [Chunk {c['chunk_number']}] {c['source']} (chunk {c['chunk_index']})")
        lines.append(f"  Score: {c['score']}")
        lines.append(f"  Excerpt: {c['text'][:300]}...")

    return "\n".join(lines)

def _remove_repetition(text: str) -> str:
    sentences = text.replace("\n", " ").split(". ")
    seen, cleaned = set(), []
    for s in sentences:
        fingerprint = s.strip().lower()[:80]
        if fingerprint not in seen:
            seen.add(fingerprint)
            cleaned.append(s)
    return ". ".join(cleaned)

def format_response(query: str, answer: str, chunks: list[dict]) -> dict:
    answer = _remove_repetition(answer)  
    return {
        "query":     query,
        "answer":    answer,
        "citations": extract_citations(answer, chunks),
    }