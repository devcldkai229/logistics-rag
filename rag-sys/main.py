import importlib
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from data.database import Base, engine

import data.models  # noqa: F401 — đăng ký FeedbackChunk, PolicyChunk vào Base.metadata

# Tuỳ chọn: warmup embedding/reranker khi đã có pipeline.
try:
    from pipeline.embedder import warmup_embedding_model, warmup_reranker
except ImportError:
    warmup_embedding_model = None
    warmup_reranker = None

def _optional_router(module: str):
    try:
        m = importlib.import_module(module)
        return getattr(m, "router", None)
    except ImportError:
        return None


chat_router = _optional_router("api.chat")
dashboard_router = _optional_router("api.dashboard")
document_router = _optional_router("api.document")

try:
    from api.ingest import router as ingest_router
except ImportError:
    ingest_router = None

# Đảm bảo console UTF-8 trên Windows (tránh lỗi print Unicode).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    if warmup_embedding_model is not None:
        try:
            warmup_embedding_model()
            print("BGE-M3 embedding model is ready!")
        except Exception as e:
            print(f"Warning: unable to warm up BGE-M3 at startup: {e}")
    if warmup_reranker is not None:
        try:
            warmup_reranker()
            print("BGE reranker is ready!")
        except Exception as e:
            print(f"Warning: unable to warm up reranker at startup: {e}")

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        print("PostgreSQL: extension 'vector' OK; ORM tables synced (create_all).")
    except Exception as e:
        print(f"Warning: database init failed (check DATABASE_URL, PostgreSQL, pgvector): {e}")

    yield

    await engine.dispose()


app = FastAPI(
    title="MACA — Marketing & Customer Assistant",
    description=(
        "RAG backend: customer feedback (CSV) và chính sách công ty (policy chunks), "
        "PostgreSQL + pgvector."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if chat_router is not None:
    app.include_router(chat_router, prefix="/chat", tags=["Chat"])
if dashboard_router is not None:
    app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
if document_router is not None:
    app.include_router(document_router, prefix="/documents", tags=["Documents"])
if ingest_router is not None:
    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])


@app.get("/")
async def root():
    return {
        "service": "MACA",
        "message": "Marketing & Customer Assistant API",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
