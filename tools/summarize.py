"""Tool: summarize — retrieves broad context from a paper to enable summarization."""

from typing import Any

from core.logging import get_logger
from core.models import ToolResult
from core.settings import get_settings
from infra.llm_client import embed_text
from infra.vector_store import VectorStore, get_vector_store
from tools.base import BaseTool

logger = get_logger(__name__)

_SUMMARY_QUERIES = [
    "main contribution and key idea",
    "methodology and approach",
    "experimental results and evaluation",
    "limitations and future work",
]


class SummarizeTool(BaseTool):
    """
    Retrieves a balanced set of chunks from a paper to support summarization.

    Runs multiple queries and deduplicates results, giving the AnalystAgent
    a comprehensive view of the paper.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store or get_vector_store()
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return "summarize"

    @property
    def description(self) -> str:
        return (
            "Retrieves a representative set of passages from a specific paper "
            "to support structured summarization."
        )

    async def run(
        self,
        arxiv_id: str,
        chunks_per_aspect: int = 2,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Args:
            arxiv_id: arXiv ID of the paper to summarize.
            chunks_per_aspect: Chunks retrieved per summary query.
        """
        try:
            seen_ids: set[str] = set()
            all_chunks: list[dict[str, Any]] = []
            title: str = ""

            for query_text in _SUMMARY_QUERIES:
                query_embedding = await embed_text(query_text)
                raw = await self._store.query(
                    query_embedding=query_embedding,
                    n_results=chunks_per_aspect,
                    where={"arxiv_id": {"$eq": arxiv_id}},
                )
                for i, (doc, meta, chunk_id) in enumerate(
                    zip(
                        raw["documents"][0],
                        raw["metadatas"][0],
                        raw["ids"][0],
                    )
                ):
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        if not title:
                            title = meta.get("paper_title", "")
                        all_chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "section": meta.get("section"),
                                "text": doc,
                                "query_aspect": query_text,
                            }
                        )

            logger.debug(
                "summarize_context_ready",
                arxiv_id=arxiv_id,
                n_chunks=len(all_chunks),
            )
            return ToolResult.success(
                self.name,
                {"arxiv_id": arxiv_id, "title": title, "chunks": all_chunks},
            )

        except Exception as exc:
            logger.error("summarize_error", arxiv_id=arxiv_id, error=str(exc))
            return ToolResult.failure(self.name, str(exc))
