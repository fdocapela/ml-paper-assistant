"""Unit tests for AnalystAgent tools (compare_papers, summarize, rank_papers)."""

import pytest
from unittest.mock import AsyncMock, patch

from core.models import ToolStatus
from tools.compare_papers import ComparePapersTool
from tools.summarize import SummarizeTool
from tools.rank_papers import RankPapersTool


# ── ComparePapersTool ─────────────────────────────────────────────────────────

class TestComparePapersTool:
    def test_tool_name(self, mock_vector_store):
        assert ComparePapersTool(mock_vector_store).name == "compare_papers"

    def test_tool_description_not_empty(self, mock_vector_store):
        assert len(ComparePapersTool(mock_vector_store).description) > 10

    @pytest.mark.asyncio
    async def test_run_success(self, mock_vector_store):
        with patch(
            "tools.compare_papers.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = ComparePapersTool(mock_vector_store)
            result = await tool.run(
                arxiv_ids=["2210.03629", "2302.04761"],
                aspect="tool use mechanism",
            )

        assert result.status == ToolStatus.SUCCESS
        assert result.data["aspect"] == "tool use mechanism"
        assert len(result.data["papers"]) == 2

    @pytest.mark.asyncio
    async def test_run_fetches_each_paper(self, mock_vector_store):
        """Should query ChromaDB once per paper."""
        with patch(
            "tools.compare_papers.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = ComparePapersTool(mock_vector_store)
            await tool.run(
                arxiv_ids=["1706.03762", "1810.04805", "2005.11401"],
                aspect="architecture",
            )

        assert mock_vector_store.query.call_count == 3

    @pytest.mark.asyncio
    async def test_run_error_propagation(self, mock_vector_store):
        with patch(
            "tools.compare_papers.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = ValueError("bad input")
            tool = ComparePapersTool(mock_vector_store)
            result = await tool.run(arxiv_ids=["1706.03762"], aspect="test")

        assert result.status == ToolStatus.ERROR


# ── SummarizeTool ─────────────────────────────────────────────────────────────

class TestSummarizeTool:
    def test_tool_name(self, mock_vector_store):
        assert SummarizeTool(mock_vector_store).name == "summarize"

    def test_tool_description_not_empty(self, mock_vector_store):
        assert len(SummarizeTool(mock_vector_store).description) > 10

    @pytest.mark.asyncio
    async def test_run_deduplicates_chunks(self, mock_vector_store):
        """Repeated chunk IDs across multiple queries must be deduplicated."""
        same_response = {
            "ids": [["chunk-same"]],
            "documents": [["Repeated content."]],
            "metadatas": [
                [
                    {
                        "arxiv_id": "1706.03762",
                        "paper_title": "Test Paper",
                        "section": "intro",
                    }
                ]
            ],
            "distances": [[0.1]],
        }
        mock_vector_store.query = AsyncMock(return_value=same_response)

        with patch(
            "tools.summarize.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = SummarizeTool(mock_vector_store)
            result = await tool.run(arxiv_id="1706.03762")

        assert result.status == ToolStatus.SUCCESS
        # 4 queries but same chunk → deduplicated to 1
        assert len(result.data["chunks"]) == 1

    @pytest.mark.asyncio
    async def test_run_error_handling(self, mock_vector_store):
        with patch(
            "tools.summarize.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("embed error")
            tool = SummarizeTool(mock_vector_store)
            result = await tool.run(arxiv_id="1706.03762")

        assert result.status == ToolStatus.ERROR


# ── RankPapersTool ────────────────────────────────────────────────────────────

class TestRankPapersTool:
    def test_tool_name(self, mock_vector_store):
        assert RankPapersTool(mock_vector_store).name == "rank_papers"

    def test_tool_description_not_empty(self, mock_vector_store):
        assert len(RankPapersTool(mock_vector_store).description) > 10

    @pytest.mark.asyncio
    async def test_run_defaults_to_all_papers(self, mock_vector_store):
        with patch(
            "tools.rank_papers.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = RankPapersTool(mock_vector_store)
            result = await tool.run(criterion="relevance for building agents")

        assert result.status == ToolStatus.SUCCESS
        # Should have queried for all 5 papers
        assert mock_vector_store.query.call_count == 5

    @pytest.mark.asyncio
    async def test_run_with_specific_papers(self, mock_vector_store):
        with patch(
            "tools.rank_papers.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = RankPapersTool(mock_vector_store)
            result = await tool.run(
                criterion="tool use",
                arxiv_ids=["2210.03629", "2302.04761"],
            )

        assert result.status == ToolStatus.SUCCESS
        assert mock_vector_store.query.call_count == 2

    @pytest.mark.asyncio
    async def test_run_error_handling(self, mock_vector_store):
        with patch(
            "tools.rank_papers.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("connection error")
            tool = RankPapersTool(mock_vector_store)
            result = await tool.run(criterion="test criterion")

        assert result.status == ToolStatus.ERROR
