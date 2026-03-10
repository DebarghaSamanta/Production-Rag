import sys
from pathlib import Path
print(sys.path)
from ingestion.loader import load_document, load_directory
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_chunks
from store.vector_store import add_chunks, count


def ingest_file(path: str | Path) -> dict:
    print(f"[1/4] Loading {path}...")
    doc = load_document(path)

    print(f"[2/4] Chunking...")
    chunks = chunk_documents([doc])
    print(f"      → {len(chunks)} chunks")

    print(f"[3/4] Embedding...")
    chunks = embed_chunks(chunks)

    print(f"[4/4] Storing in vector DB...")
    add_chunks(chunks)
    print(f"      → Total chunks in DB: {count()}")

    return {"source": doc["source"], "chunks_added": len(chunks)}


def ingest_directory(dir_path: str | Path) -> list[dict]:
    print(f"[1/4] Loading directory {dir_path}...")
    docs = load_directory(dir_path)
    print(f"      → {len(docs)} documents found")

    print(f"[2/4] Chunking...")
    chunks = chunk_documents(docs)
    print(f"      → {len(chunks)} total chunks")

    print(f"[3/4] Embedding...")
    chunks = embed_chunks(chunks)

    print(f"[4/4] Storing in vector DB...")
    add_chunks(chunks)
    print(f"      → Total chunks in DB: {count()}")

    return [{"source": d["source"]} for d in docs]


if __name__ == "__main__":
    import sys
    from config import RAW_DATA_DIR

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DATA_DIR

    if target.is_file():
        result = ingest_file(target)
        print(f"\nDone: {result}")
    else:
        results = ingest_directory(target)
        print(f"\nDone: ingested {len(results)} documents")