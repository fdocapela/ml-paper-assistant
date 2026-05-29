"""AnalystAgent — performs comparative analysis, summaries, and rankings."""

from typing import Any

from core.logging import get_logger
from core.models import ToolResult
from infra.llm_client import generate_content
from infra.vector_store import VectorStore, get_vector_store
from tools.compare_papers import ComparePapersTool
from tools.rank_papers import RankPapersTool
from tools.summarize import SummarizeTool

logger = get_logger(__name__)

_ANALYST_SYSTEM = """You are an expert Machine Learning researcher with deep knowledge of NLP,
deep learning architectures, and AI systems. You receive structured context extracted from
academic papers and produce rigorous, insightful analysis.

Guidelines:
- Be precise and technical; cite specific details from the provided context.
- When comparing, highlight both similarities AND key differences.
- When summarising, be concise but comprehensive.
- When ranking, provide clear, evidence-based justifications.
- Structure your responses clearly with headers or bullet points as appropriate.
- Respond in the same language as the user's question."""


class AnalystAgent:
    """
    Analysis agent.

    Owns the three analysis tools (compare_papers, summarize, rank_papers).
    Each method: (1) calls the tool to fetch evidence, (2) calls Gemini to
    synthesise a structured response from that evidence.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        vs = vector_store or get_vector_store()
        self._compare = ComparePapersTool(vs)
        self._summarize = SummarizeTool(vs)
        self._rank = RankPapersTool(vs)
        logger.info("analyst_agent_initialized")

    async def compare_papers(
        self,
        arxiv_ids: list[str],
        aspect: str,
        question_context: str = "",
    ) -> ToolResult:
        """Fetch evidence then synthesise a structured comparison via LLM."""
        evidence_result = await self._compare.run(arxiv_ids=arxiv_ids, aspect=aspect)
        if evidence_result.status == "error":
            return evidence_result

        data = evidence_result.data
        papers_context = "\n\n".join(
            f"### {p['title']} ({p['arxiv_id']})\n{p['relevant_context']}"
            for p in data["papers"]
        )

        prompt = (
            f"Compare the following papers on the aspect: **{aspect}**\n\n"
            f"{papers_context}\n\n"
            f"{'Additional context: ' + question_context if question_context else ''}\n\n"
            "Provide a structured, rigorous comparison highlighting key differences "
            "and similarities."
        )

        response = await generate_content(prompt, system_instruction=_ANALYST_SYSTEM)
        synthesis = response.text or ""

        return ToolResult.success(
            "compare_papers",
            {"aspect": aspect, "comparison": synthesis},
        )

    async def summarize(
        self,
        arxiv_id: str,
        bullet_points: int = 5,
    ) -> ToolResult:
        """Fetch evidence then produce a structured paper summary via LLM."""
        evidence_result = await self._summarize.run(arxiv_id=arxiv_id)
        if evidence_result.status == "error":
            return evidence_result

        data = evidence_result.data
        chunks_text = "\n\n".join(
            f"[{c['query_aspect']}] {c['text']}" for c in data["chunks"]
        )

        prompt = (
            f"Based on the following excerpts from **{data['title']}**, "
            f"produce a structured executive summary in exactly {bullet_points} "
            f"bullet points.\n\n{chunks_text}\n\n"
            "Each bullet point should be concise (1-2 sentences) and capture a "
            "distinct, important aspect of the paper."
        )

        response = await generate_content(prompt, system_instruction=_ANALYST_SYSTEM)
        summary = response.text or ""

        return ToolResult.success(
            "summarize",
            {"arxiv_id": arxiv_id, "title": data["title"], "summary": summary},
        )

    async def rank_papers(
        self,
        criterion: str,
        arxiv_ids: list[str] | None = None,
        question_context: str = "",
    ) -> ToolResult:
        """Fetch evidence then produce a ranked list with justifications via LLM."""
        evidence_result = await self._rank.run(
            criterion=criterion, arxiv_ids=arxiv_ids
        )
        if evidence_result.status == "error":
            return evidence_result

        data = evidence_result.data
        papers_evidence = "\n\n".join(
            f"### {p['title']} ({p['arxiv_id']})\n{p['evidence']}"
            for p in data["papers"]
        )

        prompt = (
            f"Rank the following papers according to: **{criterion}**\n\n"
            f"{papers_evidence}\n\n"
            f"{'Additional context: ' + question_context if question_context else ''}\n\n"
            "Produce a ranked list (1 = most relevant) with a clear, evidence-based "
            "justification for each paper's position."
        )

        response = await generate_content(prompt, system_instruction=_ANALYST_SYSTEM)
        ranking = response.text or ""

        return ToolResult.success(
            "rank_papers",
            {"criterion": criterion, "ranking": ranking},
        )

    async def dispatch(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Generic dispatch for the orchestrator's function-calling loop."""
        match tool_name:
            case "compare_papers":
                return await self.compare_papers(**params)
            case "summarize":
                return await self.summarize(**params)
            case "rank_papers":
                return await self.rank_papers(**params)
            case _:
                return ToolResult.failure(
                    tool_name, f"AnalystAgent does not own tool '{tool_name}'"
                )
