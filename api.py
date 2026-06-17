"""FastAPI application for the Medical RAG system.

Provides REST API endpoints for health checks, question answering,
document retrieval, and PDF ingestion.

Usage:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.health import health_check

app = FastAPI(
    title="Medical RAG API",
    description="面向医疗系统的企业级 RAG 问答系统",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response schemas ──


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: str | None = Field(default=None, pattern=r"^(medical_qa|drug_query|diagnosis)$")
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    session_id: str | None = None
    intent: str | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class RetrieveResult(BaseModel):
    content: str
    source: str
    page: int
    section: str
    chunk_id: str
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]
    query: str


class HealthResponse(BaseModel):
    status: str
    components: dict


# ── Endpoints ──


@app.get("/health", response_model=HealthResponse)
async def health():
    """Aggregate health check for all backend services."""
    components = health_check()
    all_ok = all(c.get("status") == "ok" for c in components.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        components=components,
    )


@app.post("/rag/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Answer a medical question using the full RAG pipeline."""
    from src.conversation import create_conversational_chain

    chain = create_conversational_chain(mode=req.mode)
    try:
        result = chain.invoke(
            {"question": req.question},
            config={"configurable": {"session_id": req.session_id or "api"}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {e}") from e
    return AskResponse(answer=result, session_id=req.session_id)


@app.post("/rag/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Retrieve relevant documents without generating an answer."""
    from src.vector_store import get_vector_store

    store = get_vector_store()
    try:
        docs = store.hybrid_search(req.query, k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}") from e

    results = [
        RetrieveResult(
            content=doc.page_content,
            source=doc.metadata.get("source", ""),
            page=doc.metadata.get("page", 0),
            section=doc.metadata.get("section", ""),
            chunk_id=doc.metadata.get("chunk_id", ""),
            score=doc.metadata.get("score", 0),
        )
        for doc in docs
    ]
    return RetrieveResponse(results=results, query=req.query)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest a PDF into the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    max_bytes = 50 * 1024 * 1024  # 50MB
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large (max {max_bytes // 1024 // 1024}MB)")

    tmp_path = Path(f"/tmp/rag_ingest_{file.filename}")
    try:
        tmp_path.write_bytes(content)
        from src.document_loader import load_pdf
        from src.vector_store import add_documents_to_store

        docs = load_pdf(str(tmp_path))
        if not docs:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        add_documents_to_store(docs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return {"status": "ok", "filename": file.filename, "size_bytes": len(content)}
