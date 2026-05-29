"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import router as threads_router
from core.logging import configure_logging, get_logger
from infra.database import create_tables, dispose_engine

configure_logging()
logger = get_logger(__name__)

PAPERS_DIR = Path("/app/papers")
STATIC_DIR = Path("/app/static")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("application_starting")
    await create_tables()
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("application_ready")
    yield
    logger.info("application_shutting_down")
    await dispose_engine()


app = FastAPI(
    title="ML Paper Analysis Assistant",
    description="Multi-agent system for ML paper analysis.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(threads_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/papers/upload", tags=["papers"])
async def upload_paper(file: UploadFile = File(...)) -> dict:
    """Upload a PDF, ingest it into ChromaDB, return chunk count."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dest = PAPERS_DIR / file.filename
    dest.write_bytes(content)
    logger.info("pdf_uploaded", filename=file.filename, size_bytes=len(content))

    try:
        import hashlib
        import tiktoken
        from scripts.ingest_papers import _chunk_text, _extract_text_with_sections
        from infra.llm_client import embed_documents
        from infra.vector_store import get_vector_store
        from core.settings import get_settings

        settings = get_settings()
        store = get_vector_store()
        encoding = tiktoken.get_encoding("cl100k_base")

        title = file.filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()
        arxiv_id = file.filename.replace(".pdf", "").replace(" ", "_")[:30]

        blocks = _extract_text_with_sections(dest)
        section_map = []
        for text, section in blocks:
            for chunk in _chunk_text(text, settings.chunk_size, settings.chunk_overlap, encoding):
                if chunk.strip():
                    section_map.append((chunk, section))

        if not section_map:
            raise HTTPException(status_code=422, detail="No text could be extracted from the PDF.")

        texts = [t for t, _ in section_map]
        embeddings = await embed_documents(texts)

        ids, metadatas = [], []
        for i, (chunk_text, section) in enumerate(section_map):
            h = hashlib.md5(f"{arxiv_id}-{i}-{chunk_text[:50]}".encode()).hexdigest()[:8]
            ids.append(f"{arxiv_id}-{i}-{h}")
            metadatas.append({
                "arxiv_id": arxiv_id,
                "paper_title": title,
                "section": section or "body",
                "chunk_index": i,
            })

        await store.upsert_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("upload_ingested", title=title, n_chunks=len(ids))
        return {"status": "ok", "title": title, "chunks": len(ids)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("upload_ingest_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}") from exc


# Serve frontend — mounted AFTER API routes so /api/docs still works
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
