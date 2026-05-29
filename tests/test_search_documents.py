"""Unit tests for SearchDocumentsTool."""

import pytest
from unittest.mock import AsyncMock, patch

from core.models import ToolStatus
from tools.search_documents import SearchDocumentsTool


class TestSearchDocumentsTool:
    def test_tool_name(self, mock_vector_store):
        tool = SearchDocumentsTool(mock_vector_store)
        assert tool.name == "search_documents"

    def test_tool_description_not_empty(self, mock_vector_store):
        tool = SearchDocumentsTool(mock_vector_store)
        assert len(tool.description) > 10

    @pytest.mark.asyncio
    async def test_run_success(self, mock_vector_store):
        with patch(
            "tools.search_documents.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = SearchDocumentsTool(mock_vector_store)
            result = await tool.run(query="attention mechanism")

        assert result.status == ToolStatus.SUCCESS
        assert result.tool_name == "search_documents"
        assert result.data is not None
        assert "chunks" in result.data
        assert len(result.data["chunks"]) == 2

    @pytest.mark.asyncio
    async def test_run_with_single_arxiv_filter(self, mock_vector_store):
        with patch(
            "tools.search_documents.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = SearchDocumentsTool(mock_vector_store)
            await tool.run(query="self-attention", arxiv_ids=["1706.03762"])

        call_kwargs = mock_vector_store.query.call_args.kwargs
        assert call_kwargs["where"] == {"arxiv_id": {"$eq": "1706.03762"}}

    @pytest.mark.asyncio
    async def test_run_with_multiple_arxiv_filter(self, mock_vector_store):
        with patch(
            "tools.search_documents.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = SearchDocumentsTool(mock_vector_store)
            await tool.run(
                query="tool use", arxiv_ids=["2210.03629", "2302.04761"]
            )

        call_kwargs = mock_vector_store.query.call_args.kwargs
        assert call_kwargs["where"] == {
            "arxiv_id": {"$in": ["2210.03629", "2302.04761"]}
        }

    @pytest.mark.asyncio
    async def test_run_score_conversion(self, mock_vector_store):
        """Distance 0.1 should become score 0.9."""
        with patch(
            "tools.search_documents.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = SearchDocumentsTool(mock_vector_store)
            result = await tool.run(query="attention")

        first_chunk = result.data["chunks"][0]
        assert abs(first_chunk["score"] - 0.9) < 1e-6

    @pytest.mark.asyncio
    async def test_run_error_handling(self, mock_vector_store):
        with patch(
            "tools.search_documents.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("Network error")
            tool = SearchDocumentsTool(mock_vector_store)
            result = await tool.run(query="test")

        assert result.status == ToolStatus.ERROR
        assert "Network error" in result.error
