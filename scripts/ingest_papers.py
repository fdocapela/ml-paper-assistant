#!/usr/bin/env python3
"""
Ingest all papers into ChromaDB.

Pipeline:
  1. Parse PDFs with pypdf
  2. Detect section headers
  3. Chunk text with overlap using tiktoken
  4. Embed chunks with Google embedding model
  5. Upsert into ChromaDB
"""

import asyncio
import hashlib
import re
import sys
from pathlib import Path

import tiktoken
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging import configure_logging, get_logger
from core.models import PAPER_REGISTRY
from core.settings import get_settings
from infra.llm_client import embed_documents
from infra.vector_store import get_vector_store

configure_logging()
logger = get_logger(__name__)

PAPERS_DIR = Path(__file__).parent.parent / "papers"

_SECTION_RE = re.compile(
    r"^(?:\d+\.?\s+)?("
    r"Abstract|Introduction|Related Work|Background|Methodology|Method|"
    r"Approach|Model|Architecture|Experiments?|Evaluation|Results?|"
    r"Discussion|Conclusion|Limitations?|Future Work|References|Appendix"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)


def _detect_section(text: str) -> str | None:
    """Return the lowercase section name if text starts with a known header."""
    match = _SECTION_RE.match(text.strip())
    return match.group(1).lower() if match else None


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    """Chunk text by token count with overlap."""
    tokens = encoding.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _extract_text_with_sections(pdf_path: Path) -> list[tuple[str, str | None]]:
    """
    Extract (text_block, section_name) pairs from a PDF.
    Returns list of (page_text, current_section) tuples.
    """
    reader = PdfReader(str(pdf_path))
    current_section: str | None = "abstract"
    blocks: list[tuple[str, str | None]] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        for line in page_text.split("\n"):
            detected = _detect_section(line)
            if detected:
                current_section = detected
        blocks.append((page_text, current_section))

    return blocks


async def ingest_paper(
    arxiv_id: str,
    meta: dict[str, str],
    store,
    settings,
    encoding: tiktoken.Encoding,
) -> int:
    """Ingest a single paper. Returns number of chunks upserted."""
    pdf_path = PAPERS_DIR / meta["filename"]
    if not pdf_path.exists():
        logger.error("pdf_not_found", arxiv_id=arxiv_id, path=str(pdf_path))
        return 0

    logger.info("ingesting_paper", arxiv_id=arxiv_id, title=meta["title"])

    blocks = _extract_text_with_sections(pdf_path)

    section_map: list[tuple[str, str | None]] = []
    for text, section in blocks:
        chunks = _chunk_text(text, settings.chunk_size, settings.chunk_overlap, encoding)
        for chunk in chunks:
            if chunk.strip():
                section_map.append((chunk, section))

    if not section_map:
        logger.warning("no_chunks_extracted", arxiv_id=arxiv_id)
        return 0

    texts = [t for t, _ in section_map]
    logger.info("embedding_chunks", arxiv_id=arxiv_id, n_chunks=len(texts))
    embeddings = await embed_documents(texts)

    ids: list[str] = []
    metadatas: list[dict] = []
    for i, (chunk_text, section) in enumerate(section_map):
        chunk_hash = hashlib.md5(
            f"{arxiv_id}-{i}-{chunk_text[:50]}".encode()
        ).hexdigest()[:8]
        ids.append(f"{arxiv_id}-{i}-{chunk_hash}")
        metadatas.append(
            {
                "arxiv_id": arxiv_id,
                "paper_title": meta["title"],
                "section": section or "body",
                "chunk_index": i,
            }
        )

    await store.upsert_chunks(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info("paper_ingested", arxiv_id=arxiv_id, n_chunks=len(ids))
    return len(ids)


async def main(reset: bool = False) -> None:
    settings = get_settings()
    store = get_vector_store()
    encoding = tiktoken.get_encoding("cl100k_base")

    if reset:
        logger.info("resetting_collection")
        await store.reset_collection()

    print(f"\n🔢 Ingesting {len(PAPER_REGISTRY)} papers into ChromaDB...\n")

    total = 0
    for arxiv_id, meta in PAPER_REGISTRY.items():
        count = await ingest_paper(arxiv_id, meta, store, settings, encoding)
        total += count
        print(f"  ✓ {meta['title']}: {count} chunks")

    final_count = await store.count()
    print(f"\n✅ Done. Total chunks in store: {final_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate collection before ingesting",
    )
    args = parser.parse_args()
    asyncio.run(main(reset=args.reset))
