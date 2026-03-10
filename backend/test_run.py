
from pathlib import Path
from ingestion.loader import load_document
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_chunks, embed_query
from config import RAW_DATA_DIR
from store.vector_store import add_chunks,  count
from retreiver.retreival import retrieve
from generation.prompt_builder import build_prompt
from generation.llm import generate
from generation.citation import format_response, render

doc = load_document(f"{RAW_DATA_DIR}/sample.pdf") 
"""
print("=== LOADER ===")
print(f"Source:     {doc['source']}")
print(f"Type:       {doc['file_type']}")
print(f"Text length:{len(doc['text'])} chars")
print(f"First 300 chars:\n{doc['text'][:300]}")"""

chunks = chunk_document(doc)
"""
print("\n=== CHUNKER ===")
print(f"Total chunks: {len(chunks)}")
print(f"Chunk 0 tokens: {chunks[0]['token_count']}")
print(f"Chunk 1 tokens: {chunks[1]['token_count']}")
print(f"Chunk 0 text:\n{chunks[0]['text'][:300]}")
print(f"\nChunk 1 text (should overlap with end of chunk 0):\n{chunks[1]['text'][:300]}")"""

sample = chunks[:3]   # just embed 3 chunks, not the whole doc
embedded = embed_chunks(sample)
"""
print("\n=== EMBEDDER ===")
print(f"Embedding dimensions: {len(embedded[0]['embedding'])}")
print(f"First 5 values: {embedded[0]['embedding'][:5]}")
"""
add_chunks(embedded)

query = "what is the termination condition?"
chunks  = retrieve(query)              
messages = build_prompt(query, chunks)
answer   = generate(messages)
response = format_response(query, answer, chunks)

print(render(response))