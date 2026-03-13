
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from retreiver.retreival import retrieve

chunks = retrieve("simplify Indemnification")
for i, c in enumerate(chunks):
    print(f"Chunk {i+1} | score: {c.get('score')} | bm25_rank: {c.get('bm25_rank')} | vector_rank: {c.get('vector_rank')}")
    print(c['text'][:200])
    print("---")

from store.vector_store import _collection

all_chunks = _collection.get(include=["documents", "metadatas"])
for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
    if "indemnif" in doc.lower():
        print(f"chunk_index: {meta['chunk_index']}")
        print(doc[:500])
        print("---")