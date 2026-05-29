"""Tool: rank_papers — retrieves evidence to rank papers by a given criterion."""

import asyncio
from typing import Any

from core.logging import get_logger
from core.models import PAPER_REGISTRY, ToolResult
from core.settings import get_settings
from infra.llm_client import embed_text
from infra.vector_store import VectorStore, get_vector_store
from tools.base import BaseTool

logger = get_logger(__name__)


class RankPapersTool(BaseTool):
    """
    Retrieves criterion-relevant passages from every paper to support ranking.

    Returns structured context that the AnalystAgent uses to produce
    a ranked list with per-paper justifications.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store or get_vector_store()
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return "rank_papers"

    @property
    def description(self) -> str:
        return (
            "Retrieves evidence from all papers relevant to a ranking criterion. "
            "Returns structured context per paper to support a justified ranking."
        )

    async def run(
        self,
        criterion: str,
        arxiv_ids: list[str] | None = None,
        chunks_per_paper: int = 3,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Args:
            criterion: The ranking criterion description.
            arxiv_ids: Papers to rank (defaults to all 5).
            chunks_per_paper: Evidence chunks per paper.
        """
        ids_to_rank = arxiv_ids or list(PAPER_REGISTRY.keys())

        try:
            criterion_embedding = await embed_text(criterion)

            async def _fetch(arxiv_id: str) -> dict[str, Any]:
                raw = await self._store.query(
                    query_embedding=criterion_embedding,
                    n_results=chunks_per_paper,
                    where={"arxiv_id": {"$eq": arxiv_id}},
                )
                context = (
                    " ".join(raw["documents"][0]) if raw["documents"][0] else ""
                )
                title = (
                    raw["metadatas"][0][0].get("paper_title", "")
                    if raw["metadatas"][0]
                    else PAPER_REGISTRY.get(arxiv_id, {}).get("title", arxiv_id)
                )
                return {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "evidence": context,
                }

            papers_evidence = await asyncio.gather(
                *[_fetch(aid) for aid in ids_to_rank]
            )

            logger.debug(
                "rank_papers_evidence_ready",
                criterion=criterion,
                n_papers=len(ids_to_rank),
            )
            return ToolResult.success(
                self.name,
                {"criterion": criterion, "papers": list(papers_evidence)},
            )

        except Exception as exc:
            logger.error("rank_papers_error", criterion=criterion, error=str(exc))
            return ToolResult.failure(self.name, str(exc))
