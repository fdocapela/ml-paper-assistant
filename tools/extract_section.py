"""Tool: extract_section — retrieve a named section from a specific paper."""

from typing import Any

from core.logging import get_logger
from core.models import DocumentChunk, ToolResult
from core.settings import get_settings
from infra.llm_client import embed_text
from infra.vector_store import VectorStore, get_vector_store
from tools.base import BaseTool

logger = get_logger(__name__)


class ExtractSectionTool(BaseTool):
    """
    Retrieves the most relevant chunks from a specific section of a paper.

    Uses metadata filtering on `section` plus a semantic query.
    Falls back to paper-wide search if the section is not found.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store or get_vector_store()
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return "extract_section"

    @property
    def description(self) -> str:
        return (
            "Extracts content from a specific section (e.g. abstract, introduction, "
            "conclusion, methodology) of a given paper identified by its arXiv ID."
        )

    async def run(
        self,
        arxiv_id: str,
        section: str,
        top_k: int = 4,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Args:
            arxiv_id: The paper's arXiv ID (e.g. '1706.03762').
            section: Section name to extract (e.g. 'abstract', 'conclusion').
            top_k: Number of chunks to return.
        """
        section_lower = section.lower().strip()

        try:
            query = f"{section_lower} of the paper"
            query_embedding = await embed_text(query)

            where: dict[str, Any] = {
                "$and": [
                    {"arxiv_id": {"$eq": arxiv_id}},
                    {"section": {"$eq": section_lower}},
                ]
            }

            raw = await self._store.query(
                query_embedding=query_embedding,
                n_results=top_k,
                where=where,
            )

            chunks: list[DocumentChunk] = []
            if raw["documents"][0]:
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
            else:
                # Fallback: search the paper broadly
                logger.warning(
                    "section_not_found_fallback",
                    arxiv_id=arxiv_id,
                    section=section_lower,
                )
                where_fallback: dict[str, Any] = {"arxiv_id": {"$eq": arxiv_id}}
                raw2 = await self._store.query(
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where=where_fallback,
                )
                for i, (doc, meta, dist) in enumerate(
                    zip(
                        raw2["documents"][0],
                        raw2["metadatas"][0],
                        raw2["distances"][0],
                    )
                ):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=raw2["ids"][0][i],
                            arxiv_id=meta.get("arxiv_id", ""),
                            paper_title=meta.get("paper_title", ""),
                            section=meta.get("section"),
                            text=doc,
                            score=1.0 - dist,
                        )
                    )

            logger.debug(
                "section_extracted",
                arxiv_id=arxiv_id,
                section=section_lower,
                n_chunks=len(chunks),
            )
            return ToolResult.success(
                self.name,
                {
                    "arxiv_id": arxiv_id,
                    "section": section_lower,
                    "chunks": [c.model_dump() for c in chunks],
                },
            )

        except Exception as exc:
            logger.error(
                "extract_section_error",
                arxiv_id=arxiv_id,
                section=section,
                error=str(exc),
            )
            return ToolResult.failure(self.name, str(exc))
