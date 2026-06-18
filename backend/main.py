from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router_ingest import router as ingest_router
from api.query_router import router as query_router
from api.router_generation import router as generation_router
from monitoring.logger import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()          # runs on startup
    yield              # app runs here
                       # anything after yield runs on shutdown

app = FastAPI(
    title="arXiv RAG Backend Engine",
    version="0.1.0",
    lifespan=lifespan  # ← pass it here
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(ingest_router,     prefix="/api/v1", tags=["Ingestion"])
app.include_router(query_router,      prefix="/api/v1", tags=["Retrieval"])
app.include_router(generation_router, prefix="/api/v1", tags=["Generation"])

@app.get("/")
async def root():
    return {"status": "online"}