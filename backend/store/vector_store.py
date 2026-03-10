import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_DIMENSIONS


_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(chunks: list[dict]) -> None:
    _collection.add(
        ids        = [f"{c['source']}__chunk_{c['chunk_index']}" for c in chunks],
        embeddings = [c["embedding"] for c in chunks],
        documents  = [c["text"] for c in chunks],
        metadatas  = [{
            "source":      c["source"],
            "chunk_index": c["chunk_index"],
            "token_count": c["token_count"],
            "start_token": c["start_token"],
        } for c in chunks],
    )


def query(embedding: list[float], top_k: int) -> list[dict]:
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":        text,
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "token_count": meta["token_count"],
            "start_token": meta["start_token"],
            "score":       round(1 - distance, 4),  # cosine distance → similarity
        })

    return hits


def delete_collection() -> None:
    _client.delete_collection(CHROMA_COLLECTION_NAME)


def count() -> int:
    return _collection.count()