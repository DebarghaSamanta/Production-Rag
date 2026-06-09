# backend/api/router_ingest.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ingestion.fetch_paper import fetch_arxiv_papers
from ingestion.chunker import process_all_pdfs_hierarchical  # Ensure correct import
from ingestion.indexer import DocumentIndexer

router = APIRouter()
indexer = DocumentIndexer()

class IngestRequest(BaseModel):
    topic: str
    max_results: int = 3

@router.post("/ingest")
async def trigger_ingestion_pipeline(payload: IngestRequest):
    try:
        print(f"--- Starting Ingestion Pipeline for topic: {payload.topic} ---")
        
        # 1. Fetch from arXiv
        fetch_arxiv_papers(
            query=payload.topic, 
            max_results=payload.max_results, 
            output_dir="raw_data"
        )
        
        # 2. Extract and chunk with your updated Section-Aware split architecture
        parents, children = process_all_pdfs_hierarchical()
        
        # 3. Store both sets using the updated flat-list pipeline
        if children:
            indexer.index_hierarchical_data(parents, children)
        
        return {
            "status": "success",
            "topic": payload.topic,
            "papers_requested": payload.max_results,
            "total_parents_indexed": len(parents),
            "total_children_indexed": len(children),
            "message": "Papers parsed by logical section, chunked, and dual-indexed successfully."
        }
        
    except Exception as e:
        print(f"Error occurred during ingestion: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Ingestion pipeline failed: {str(e)}"
        )