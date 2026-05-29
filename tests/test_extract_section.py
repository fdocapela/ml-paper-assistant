"""Unit tests for ExtractSectionTool."""

import pytest
from unittest.mock import AsyncMock, patch

from core.models import ToolStatus
from tools.extract_section import ExtractSectionTool


class TestExtractSectionTool:
    def test_tool_name(self, mock_vector_store):
        tool = ExtractSectionTool(mock_vector_store)
        assert tool.name == "extract_section"

    def test_tool_description_not_empty(self, mock_vector_store):
        tool = ExtractSectionTool(mock_vector_store)
        assert len(tool.description) > 10

    @pytest.mark.asyncio
    async def test_run_success(self, mock_vector_store):
        with patch(
            "tools.extract_section.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = ExtractSectionTool(mock_vector_store)
            result = await tool.run(arxiv_id="1706.03762", section="abstract")

        assert result.status == ToolStatus.SUCCESS
        assert result.data["arxiv_id"] == "1706.03762"
        assert result.data["section"] == "abstract"
        assert isinstance(result.data["chunks"], list)

    @pytest.mark.asyncio
    async def test_run_fallback_when_no_section_results(self, mock_vector_store):
        """When section filter returns no results, fallback to paper-only filter."""
        empty_response = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        full_response = {
            "ids": [["chunk-1"]],
            "documents": [["Full paper content."]],
            "metadatas": [
                [
                    {
                        "arxiv_id": "1706.03762",
                        "paper_title": "Test",
                        "section": "body",
                    }
                ]
            ],
            "distances": [[0.15]],
        }
        mock_vector_store.query = AsyncMock(
            side_effect=[empty_response, full_response]
        )

        with patch(
            "tools.extract_section.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            tool = ExtractSectionTool(mock_vector_store)
            result = await tool.run(
                arxiv_id="1706.03762", section="nonexistent_section"
            )

        assert result.status == ToolStatus.SUCCESS
        assert len(result.data["chunks"]) == 1

    @pytest.mark.asyncio
    async def test_run_error_handling(self, mock_vector_store):
        with patch(
            "tools.extract_section.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = RuntimeError("Embed failed")
            tool = ExtractSectionTool(mock_vector_store)
            result = await tool.run(arxiv_id="1706.03762", section="abstract")

        assert result.status == ToolStatus.ERROR
