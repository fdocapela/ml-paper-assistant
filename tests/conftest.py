"""Shared pytest fixtures."""

import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set dummy env vars before importing the app
os.environ.setdefault("GOOGLE_API_KEY", "test-key-xxx")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8001")
os.environ.setdefault("SQLITE_DB_PATH", "/tmp/test_threads.db")

from api.main import app
from infra.database import Base


# ── In-memory SQLite for tests ────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine(TEST_DB_URL, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


# ── Mock Vector Store ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.query = AsyncMock(
        return_value={
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [
                [
                    "Mock text about attention mechanism.",
                    "More content about the paper.",
                ]
            ],
            "metadatas": [
                [
                    {
                        "arxiv_id": "1706.03762",
                        "paper_title": "Attention Is All You Need",
                        "section": "introduction",
                    },
                    {
                        "arxiv_id": "1706.03762",
                        "paper_title": "Attention Is All You Need",
                        "section": "methodology",
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
    )
    store.upsert_chunks = AsyncMock()
    store.count = AsyncMock(return_value=42)
    store.reset_collection = AsyncMock()
    store.get_collection = AsyncMock()
    return store


# ── Mock LLM response ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_response():
    response = MagicMock()
    response.text = "This is a mock LLM response for testing purposes."
    candidate = MagicMock()
    part = MagicMock()
    part.function_call = MagicMock()
    part.function_call.name = ""
    part.text = "This is a mock LLM response for testing purposes."
    candidate.content.parts = [part]
    response.candidates = [candidate]
    response.usage_metadata = None
    return response


# ── HTTP Test Client ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def async_client(test_engine) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient with overridden DB session for API tests."""
    from api.dependencies import get_db_session

    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    app.dependency_overrides.clear()
