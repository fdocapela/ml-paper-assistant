"""Unit tests for AnalystAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.analyst_agent import AnalystAgent
from core.models import ToolStatus


def _make_mock_llm_response(text: str = "Mock LLM synthesis") -> MagicMock:
    response = MagicMock()
    response.text = text
    return response


class TestAnalystAgent:
    @pytest.mark.asyncio
    async def test_compare_papers_success(self, mock_vector_store):
        with (
            patch(
                "tools.compare_papers.embed_text", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "agents.analyst_agent.generate_content", new_callable=AsyncMock
            ) as mock_llm,
        ):
            mock_embed.return_value = [0.1] * 768
            mock_llm.return_value = _make_mock_llm_response("Comparison result")

            agent = AnalystAgent(mock_vector_store)
            result = await agent.compare_papers(
                arxiv_ids=["2210.03629", "2302.04761"],
                aspect="tool use",
            )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["comparison"] == "Comparison result"
        assert result.data["aspect"] == "tool use"

    @pytest.mark.asyncio
    async def test_summarize_success(self, mock_vector_store):
        with (
            patch(
                "tools.summarize.embed_text", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "agents.analyst_agent.generate_content", new_callable=AsyncMock
            ) as mock_llm,
        ):
            mock_embed.return_value = [0.1] * 768
            mock_llm.return_value = _make_mock_llm_response(
                "• Bullet 1\n• Bullet 2\n• Bullet 3"
            )

            agent = AnalystAgent(mock_vector_store)
            result = await agent.summarize(arxiv_id="1706.03762")

        assert result.status == ToolStatus.SUCCESS
        assert "• Bullet 1" in result.data["summary"]
        assert result.data["arxiv_id"] == "1706.03762"

    @pytest.mark.asyncio
    async def test_rank_papers_success(self, mock_vector_store):
        with (
            patch(
                "tools.rank_papers.embed_text", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "agents.analyst_agent.generate_content", new_callable=AsyncMock
            ) as mock_llm,
        ):
            mock_embed.return_value = [0.1] * 768
            mock_llm.return_value = _make_mock_llm_response(
                "1. ReAct\n2. Toolformer\n3. RAG"
            )

            agent = AnalystAgent(mock_vector_store)
            result = await agent.rank_papers(criterion="agent tool use")

        assert result.status == ToolStatus.SUCCESS
        assert "ReAct" in result.data["ranking"]

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_returns_error(self, mock_vector_store):
        agent = AnalystAgent(mock_vector_store)
        result = await agent.dispatch("unknown_tool", {})
        assert result.status == ToolStatus.ERROR
        assert "AnalystAgent does not own" in result.error

    @pytest.mark.asyncio
    async def test_compare_propagates_tool_error(self, mock_vector_store):
        with patch(
            "tools.compare_papers.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("embed failed")
            agent = AnalystAgent(mock_vector_store)
            result = await agent.compare_papers(
                arxiv_ids=["1706.03762"],
                aspect="test",
            )
        assert result.status == ToolStatus.ERROR

    @pytest.mark.asyncio
    async def test_summarize_propagates_tool_error(self, mock_vector_store):
        with patch(
            "tools.summarize.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("embed failed")
            agent = AnalystAgent(mock_vector_store)
            result = await agent.summarize(arxiv_id="1706.03762")
        assert result.status == ToolStatus.ERROR
