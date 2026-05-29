"""Integration tests: API endpoints + Orchestrator → Agent → Tool flow."""

import pytest
from unittest.mock import AsyncMock, patch


class TestThreadEndpoints:
    """Tests for thread management API endpoints."""

    @pytest.mark.asyncio
    async def test_create_thread_returns_201(self, async_client):
        response = await async_client.post("/threads")
        assert response.status_code == 201
        data = response.json()
        assert "thread_id" in data
        assert len(data["thread_id"]) == 36  # valid UUID

    @pytest.mark.asyncio
    async def test_list_threads_empty_on_start(self, async_client):
        response = await async_client.get("/threads")
        assert response.status_code == 200
        assert response.json()["threads"] == []

    @pytest.mark.asyncio
    async def test_list_threads_after_creation(self, async_client):
        await async_client.post("/threads")
        await async_client.post("/threads")

        response = await async_client.get("/threads")
        assert response.status_code == 200
        assert len(response.json()["threads"]) == 2

    @pytest.mark.asyncio
    async def test_get_messages_thread_not_found(self, async_client):
        response = await async_client.get(
            "/threads/00000000-0000-0000-0000-000000000000/messages"
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_send_message_thread_not_found(self, async_client):
        response = await async_client.post(
            "/threads/00000000-0000-0000-0000-000000000000/messages",
            json={"content": "Hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_send_message_empty_content_returns_422(self, async_client):
        create_resp = await async_client.post("/threads")
        thread_id = create_resp.json()["thread_id"]

        response = await async_client.post(
            f"/threads/{thread_id}/messages",
            json={"content": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_messages_empty_thread(self, async_client):
        create_resp = await async_client.post("/threads")
        thread_id = create_resp.json()["thread_id"]

        response = await async_client.get(f"/threads/{thread_id}/messages")
        assert response.status_code == 200
        assert response.json()["messages"] == []

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestOrchestratorIntegration:
    """
    Integration tests: full request flow from API → Orchestrator → Agent → Tool.
    Uses mocked LLM and vector store.
    """

    @pytest.mark.asyncio
    async def test_send_message_full_flow(self, async_client):
        """
        POST /threads/{id}/messages triggers the orchestrator,
        response is returned and both messages are persisted.
        """
        create_resp = await async_client.post("/threads")
        assert create_resp.status_code == 201
        thread_id = create_resp.json()["thread_id"]

        mock_answer = "The Transformer uses self-attention instead of RNNs."

        with patch(
            "agents.orchestrator.OrchestratorAgent.answer",
            new_callable=AsyncMock,
            return_value=mock_answer,
        ):
            response = await async_client.post(
                f"/threads/{thread_id}/messages",
                json={"content": "What is the attention mechanism?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == thread_id
        assert data["response"] == mock_answer

        # Verify both messages were persisted
        messages_resp = await async_client.get(
            f"/threads/{thread_id}/messages"
        )
        messages = messages_resp.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the attention mechanism?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == mock_answer

    @pytest.mark.asyncio
    async def test_multi_turn_history_passed_correctly(self, async_client):
        """History from previous messages must be passed to the orchestrator."""
        create_resp = await async_client.post("/threads")
        thread_id = create_resp.json()["thread_id"]

        captured_histories: list[list] = []

        async def mock_answer(question, history=None):
            captured_histories.append(history or [])
            return f"Answer to: {question}"

        with patch(
            "agents.orchestrator.OrchestratorAgent.answer",
            side_effect=mock_answer,
        ):
            # First message — no history
            await async_client.post(
                f"/threads/{thread_id}/messages",
                json={"content": "Question 1"},
            )
            # Second message — should include Q1 and A1 in history
            await async_client.post(
                f"/threads/{thread_id}/messages",
                json={"content": "Question 2"},
            )

        assert len(captured_histories) == 2
        # First call: empty history
        assert captured_histories[0] == []
        # Second call: history has 2 messages (Q1 + A1)
        assert len(captured_histories[1]) == 2
        assert captured_histories[1][0]["role"] == "user"
        assert captured_histories[1][0]["content"] == "Question 1"
        assert captured_histories[1][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_threads_do_not_share_history(self, async_client):
        """Two separate threads must have completely isolated histories."""
        t1_resp = await async_client.post("/threads")
        t2_resp = await async_client.post("/threads")
        t1_id = t1_resp.json()["thread_id"]
        t2_id = t2_resp.json()["thread_id"]

        with patch(
            "agents.orchestrator.OrchestratorAgent.answer",
            new_callable=AsyncMock,
            return_value="Some answer",
        ):
            await async_client.post(
                f"/threads/{t1_id}/messages",
                json={"content": "Thread 1 question"},
            )
            await async_client.post(
                f"/threads/{t2_id}/messages",
                json={"content": "Thread 2 question"},
            )

        msgs1 = (
            await async_client.get(f"/threads/{t1_id}/messages")
        ).json()["messages"]
        msgs2 = (
            await async_client.get(f"/threads/{t2_id}/messages")
        ).json()["messages"]

        assert len(msgs1) == 2
        assert len(msgs2) == 2
        assert msgs1[0]["content"] == "Thread 1 question"
        assert msgs2[0]["content"] == "Thread 2 question"
