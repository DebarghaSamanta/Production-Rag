# In your ingestion entry point / main script
from ingestion.chunker import process_all_pdfs_hierarchical
from ingestion.indexer import DocumentIndexer

parents, children = process_all_pdfs_hierarchical()  # now uses pymupdf4llm

indexer = DocumentIndexer()
indexer.index_hierarchical_data(parents, children)