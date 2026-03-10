from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


def embed_query(query: str) -> list[float]:
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return _model.encode(prefixed, convert_to_numpy=True).tolist()