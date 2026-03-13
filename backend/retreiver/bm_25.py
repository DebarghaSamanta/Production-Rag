import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from config import PROJECT_ROOT

BM25_INDEX_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_index(chunks: list[dict]) -> None:
    tokenized = [_tokenize(c["text"]) for c in chunks]
    index = BM25Okapi(tokenized)

    payload = {"index": index, "chunks": chunks}
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[BM25] Index built with {len(chunks)} chunks → {BM25_INDEX_PATH}")


def _load_index() -> tuple[BM25Okapi, list[dict]]:
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. Run ingest_pipeline first."
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["index"], payload["chunks"]


def bm25_search(query: str, top_k: int) -> list[dict]:
    index, chunks = _load_index()
    tokens = _tokenize(query)
    scores = index.get_scores(tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []
    for rank, (idx, score) in enumerate(ranked):
        chunk = chunks[idx].copy()
        chunk["bm25_score"] = round(float(score), 4)
        chunk["bm25_rank"]  = rank + 1
        results.append(chunk)

    return results