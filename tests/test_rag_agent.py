"""Unit tests for RAGAgent."""

import pytest
from unittest.mock import AsyncMock, patch

from agents.rag_agent import RAGAgent
from core.models import ToolStatus


class TestRAGAgent:
    @pytest.mark.asyncio
    async def test_dispatch_search_documents(self, mock_vector_store):
        with patch(
            "tools.search_documents.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            agent = RAGAgent(mock_vector_store)
            result = await agent.dispatch(
                "search_documents", {"query": "self-attention"}
            )
        assert result.status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_dispatch_extract_section(self, mock_vector_store):
        with patch(
            "tools.extract_section.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 768
            agent = RAGAgent(mock_vector_store)
            result = await agent.dispatch(
                "extract_section",
                {"arxiv_id": "1706.03762", "section": "abstract"},
            )
        assert result.status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_returns_error(self, mock_vector_store):
        agent = RAGAgent(mock_vector_store)
        result = await agent.dispatch("nonexistent_tool", {})
        assert result.status == ToolStatus.ERROR
        assert "RAGAgent does not own" in result.error

    @pytest.mark.asyncio
    async def test_search_documents_direct(self, mock_vector_store):
        with patch(
            "tools.search_documents.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.0] * 768
            agent = RAGAgent(mock_vector_store)
            result = await agent.search_documents(query="BERT pre-training")
        assert result.status == ToolStatus.SUCCESS
        assert result.data["query"] == "BERT pre-training"

    @pytest.mark.asyncio
    async def test_extract_section_direct(self, mock_vector_store):
        with patch(
            "tools.extract_section.embed_text", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.0] * 768
            agent = RAGAgent(mock_vector_store)
            result = await agent.extract_section(
                arxiv_id="2005.11401", section="conclusion"
            )
        assert result.status == ToolStatus.SUCCESS
