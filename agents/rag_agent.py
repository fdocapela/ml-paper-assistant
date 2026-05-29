"""RAGAgent — responsible for all vector-store retrieval operations."""

from typing import Any

from core.logging import get_logger
from core.models import ToolResult
from infra.vector_store import VectorStore, get_vector_store
from tools.extract_section import ExtractSectionTool
from tools.search_documents import SearchDocumentsTool

logger = get_logger(__name__)


class RAGAgent:
    """
    Retrieval-Augmented Generation agent.

    Owns the two retrieval tools and exposes a clean async interface
    to the OrchestratorAgent. Does NOT call an LLM — it fetches context
    and returns structured ToolResults.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        vs = vector_store or get_vector_store()
        self._search = SearchDocumentsTool(vs)
        self._extract = ExtractSectionTool(vs)
        logger.info("rag_agent_initialized")

    async def search_documents(
        self,
        query: str,
        arxiv_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ToolResult:
        """Semantic search across the corpus."""
        return await self._search.run(query=query, arxiv_ids=arxiv_ids, top_k=top_k)

    async def extract_section(
        self,
        arxiv_id: str,
        section: str,
        top_k: int = 4,
    ) -> ToolResult:
        """Extract a named section from a specific paper."""
        return await self._extract.run(
            arxiv_id=arxiv_id, section=section, top_k=top_k
        )

    async def dispatch(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Generic dispatch used by the orchestrator's function-calling loop."""
        match tool_name:
            case "search_documents":
                return await self.search_documents(**params)
            case "extract_section":
                return await self.extract_section(**params)
            case _:
                return ToolResult.failure(
                    tool_name, f"RAGAgent does not own tool '{tool_name}'"
                )
