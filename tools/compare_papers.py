"""Tool: compare_papers — retrieves evidence for comparing multiple papers on an aspect."""

import asyncio
from typing import Any

from core.logging import get_logger
from core.models import ToolResult
from core.settings import get_settings
from infra.llm_client import embed_text
from infra.vector_store import VectorStore, get_vector_store
from tools.base import BaseTool

logger = get_logger(__name__)


class ComparePapersTool(BaseTool):
    """
    Retrieves relevant context for each paper on a given aspect.

    Does NOT call an LLM — it returns structured context that the
    AnalystAgent will synthesise with its own LLM call.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store or get_vector_store()
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return "compare_papers"

    @property
    def description(self) -> str:
        return (
            "Retrieves and structures the most relevant passages from multiple papers "
            "on a specific aspect or dimension for comparison."
        )

    async def run(
        self,
        arxiv_ids: list[str],
        aspect: str,
        chunks_per_paper: int = 3,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Args:
            arxiv_ids: List of paper arXiv IDs to compare.
            aspect: The dimension to compare (e.g. 'tool use mechanism').
            chunks_per_paper: Number of chunks to retrieve per paper.
        """
        try:
            aspect_embedding = await embed_text(aspect)

            async def _fetch_paper(arxiv_id: str) -> dict[str, Any]:
                raw = await self._store.query(
                    query_embedding=aspect_embedding,
                    n_results=chunks_per_paper,
                    where={"arxiv_id": {"$eq": arxiv_id}},
                )
                chunks_text = (
                    " ".join(raw["documents"][0]) if raw["documents"][0] else ""
                )
                title = (
                    raw["metadatas"][0][0].get("paper_title", "")
                    if raw["metadatas"][0]
                    else ""
                )
                return {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "relevant_context": chunks_text,
                }

            papers_context = await asyncio.gather(
                *[_fetch_paper(aid) for aid in arxiv_ids]
            )

            logger.debug(
                "compare_papers_context_ready",
                aspect=aspect,
                n_papers=len(arxiv_ids),
            )
            return ToolResult.success(
                self.name,
                {"aspect": aspect, "papers": list(papers_context)},
            )

        except Exception as exc:
            logger.error("compare_papers_error", aspect=aspect, error=str(exc))
            return ToolResult.failure(self.name, str(exc))
