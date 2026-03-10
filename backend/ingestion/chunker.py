import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

_enc = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> list[int]:
    return _enc.encode(text)


def _decode(tokens: list[int]) -> str:
    return _enc.decode(tokens)


def chunk_text(text: str, source: str) -> list[dict]:
    tokens = _tokenize(text)
    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) >= MIN_CHUNK_SIZE:
            chunks.append({
                "text":        _decode(chunk_tokens),
                "source":      source,
                "chunk_index": index,
                "token_count": len(chunk_tokens),
                "start_token": start,
            })
            index += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_document(doc: dict) -> list[dict]:
    return chunk_text(doc["text"], doc["source"])


def chunk_documents(docs: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks