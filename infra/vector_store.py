import asyncio
from typing import Any

import httpx

from core.logging import get_logger
from core.settings import get_settings

logger = get_logger(__name__)

_TENANT = "default_tenant"
_DATABASE = "default_database"


class VectorStore:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._base_url = (
            f"http://{self._settings.chroma_host}"
            f":{self._settings.chroma_port}/api/v1"
        )
        self._collection_id: str | None = None
        self._lock = asyncio.Lock()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    async def _get_collection_id(self) -> str:
        if self._collection_id is not None:
            return self._collection_id
        async with self._lock:
            if self._collection_id is not None:
                return self._collection_id
            name = self._settings.chroma_collection
            params = {"tenant": _TENANT, "database": _DATABASE}
            async with self._client() as client:
                resp = await client.get(f"/collections/{name}", params=params)
                if resp.status_code == 200:
                    self._collection_id = resp.json()["id"]
                else:
                    resp = await client.post(
                        "/collections",
                        params=params,
                        json={"name": name, "metadata": {"hnsw:space": "cosine"}},
                    )
                    resp.raise_for_status()
                    self._collection_id = resp.json()["id"]
            logger.info("collection_ready", name=name, id=self._collection_id)
            return self._collection_id

    async def upsert_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        coll_id = await self._get_collection_id()
        async with self._client() as client:
            resp = await client.post(
                f"/collections/{coll_id}/upsert",
                json={"ids": ids, "embeddings": embeddings,
                      "documents": documents, "metadatas": metadatas},
            )
            resp.raise_for_status()
        logger.info("chunks_upserted", count=len(ids))

    async def query(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        coll_id = await self._get_collection_id()
        payload: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            payload["where"] = where
        async with self._client() as client:
            resp = await client.post(f"/collections/{coll_id}/query", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def count(self) -> int:
        coll_id = await self._get_collection_id()
        async with self._client() as client:
            resp = await client.get(f"/collections/{coll_id}/count")
            resp.raise_for_status()
            return resp.json()

    async def reset_collection(self) -> None:
        name = self._settings.chroma_collection
        params = {"tenant": _TENANT, "database": _DATABASE}
        self._collection_id = None
        async with self._client() as client:
            await client.delete(f"/collections/{name}", params=params)
            resp = await client.post(
                "/collections",
                params=params,
                json={"name": name, "metadata": {"hnsw:space": "cosine"}},
            )
            resp.raise_for_status()
            self._collection_id = resp.json()["id"]
        logger.info("collection_reset", name=name)


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store