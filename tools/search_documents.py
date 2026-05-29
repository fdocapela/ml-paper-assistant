"""Tool: search_documents — semantic search over the ML paper vector store."""

from typing import Any

from core.logging import get_logger
from core.models import DocumentChunk, SearchResult, ToolResult
from core.settings import get_settings
from infra.llm_client import embed_text
from infra.vector_store import VectorStore, get_vector_store
from tools.base import BaseTool

logger = get_logger(__name__)


class SearchDocumentsTool(BaseTool):
    """
    Performs semantic search over the ChromaDB vector store.

    Given a natural-language query, embeds it and retrieves the top-k
    most relevant chunks. Optionally filtered by arxiv_id.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store or get_vector_store()
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Semantic search over the ML paper corpus. "
            "Returns the most relevant text chunks for a given query. "
            "Use this to find specific information across papers."
        )

    async def run(
        self,
        query: str,
        arxiv_ids: list[str] | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Args:
            query: Natural language search query.
            arxiv_ids: Optional list of paper IDs to restrict search to.
            top_k: Number of results to return.
        """
        k = top_k or self._settings.retrieval_top_k

        try:
            query_embedding = await embed_text(query)

            where: dict[str, Any] | None = None
            if arxiv_ids and len(arxiv_ids) == 1:
                where = {"arxiv_id": {"$eq": arxiv_ids[0]}}
            elif arxiv_ids and len(arxiv_ids) > 1:
                where = {"arxiv_id": {"$in": arxiv_ids}}

            raw = await self._store.query(
                query_embedding=query_embedding,
                n_results=k,
                where=where,
            )

            chunks: list[DocumentChunk] = []
            for i, (doc, meta, dist) in enumerate(
                zip(
                    raw["documents"][0],
                    raw["metadatas"][0],
                    raw["distances"][0],
                )
            ):
                chunks.append(
                    DocumentChunk(
                        chunk_id=raw["ids"][0][i],
                        arxiv_id=meta.get("arxiv_id", ""),
                        paper_title=meta.get("paper_title", ""),
                        section=meta.get("section"),
                        text=doc,
                        score=1.0 - dist,
                    )
                )

            result = SearchResult(chunks=chunks, query=query)
            logger.debug("search_complete", query=query, n_results=len(chunks))
            return ToolResult.success(self.name, result.model_dump())

        except Exception as exc:
            logger.error("search_error", query=query, error=str(exc))
            return ToolResult.failure(self.name, str(exc))
