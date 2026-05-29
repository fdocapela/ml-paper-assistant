"""Unit tests for ThreadRepository (database layer)."""

import pytest

from infra.database import ThreadRepository


class TestThreadRepository:
    @pytest.mark.asyncio
    async def test_create_thread(self, test_session):
        repo = ThreadRepository(test_session)
        thread = await repo.create_thread()
        assert thread.id is not None
        assert len(thread.id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_get_thread_exists(self, test_session):
        repo = ThreadRepository(test_session)
        created = await repo.create_thread()
        fetched = await repo.get_thread(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    @pytest.mark.asyncio
    async def test_get_thread_not_found(self, test_session):
        repo = ThreadRepository(test_session)
        result = await repo.get_thread("00000000-0000-0000-0000-000000000000")
        assert result is None

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, test_session):
        repo = ThreadRepository(test_session)
        thread = await repo.create_thread()

        await repo.add_message(thread.id, "user", "Hello!")
        await repo.add_message(thread.id, "assistant", "Hi there!")

        messages = await repo.get_messages(thread.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self, test_session):
        repo = ThreadRepository(test_session)
        thread = await repo.create_thread()

        for i in range(10):
            await repo.add_message(thread.id, "user", f"Message {i}")

        messages = await repo.get_messages(thread.id, limit=5)
        assert len(messages) == 5
        # Should return the LAST 5 in chronological order
        assert messages[-1].content == "Message 9"

    @pytest.mark.asyncio
    async def test_count_messages(self, test_session):
        repo = ThreadRepository(test_session)
        thread = await repo.create_thread()

        assert await repo.count_messages(thread.id) == 0
        await repo.add_message(thread.id, "user", "q1")
        await repo.add_message(thread.id, "assistant", "a1")
        assert await repo.count_messages(thread.id) == 2

    @pytest.mark.asyncio
    async def test_list_threads_ordered_most_recent_first(self, test_session):
        repo = ThreadRepository(test_session)
        t1 = await repo.create_thread()
        t2 = await repo.create_thread()

        threads = await repo.list_threads()
        # Most recent first
        assert threads[0].id == t2.id
        assert threads[1].id == t1.id

    @pytest.mark.asyncio
    async def test_threads_isolation(self, test_session):
        """Messages from different threads must not mix."""
        repo = ThreadRepository(test_session)
        t1 = await repo.create_thread()
        t2 = await repo.create_thread()

        await repo.add_message(t1.id, "user", "Thread 1 message")
        await repo.add_message(t2.id, "user", "Thread 2 message")

        msgs1 = await repo.get_messages(t1.id)
        msgs2 = await repo.get_messages(t2.id)

        assert len(msgs1) == 1
        assert msgs1[0].content == "Thread 1 message"
        assert len(msgs2) == 1
        assert msgs2[0].content == "Thread 2 message"
