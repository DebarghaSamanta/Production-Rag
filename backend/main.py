# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router_ingest import router as ingest_router
from api.query_router import router as query_router
from api.router_generation import router as generation_router
app = FastAPI(
    title="arXiv RAG Backend Engine",
    description="FastAPI backend serving the hybrid retrieval and generation layers",
    version="0.1.0"
)

# Configure CORS Middleware
# This ensures that when you build your React frontend, it won't block API requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development testing
    allow_credentials=True,
    allow_methods=["*"],  # Allows all standard HTTP methods (POST, GET, etc.)
    allow_headers=["*"],
)

# Include the ingestion router under a structured API version prefix
app.include_router(ingest_router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query_router,  prefix="/api/v1", tags=["Retrieval"])
app.include_router(generation_router, prefix="/api/v1", tags=["Generation"])
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "FastAPI RAG Backend is running smoothly."
    }