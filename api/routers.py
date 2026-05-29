"""Thread management endpoints."""

import uuid
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import OrchestratorAgent
from api.dependencies import get_db_session, get_orchestrator
from core.logging import get_logger
from core.models import (
    CreateThreadResponse,
    Message,
    MessageListResponse,
    MessageRole,
    SendMessageRequest,
    SendMessageResponse,
    Thread,
    ThreadListResponse,
)
from core.settings import get_settings
from infra.database import ThreadRepository

logger = get_logger(__name__)

router = APIRouter(prefix="/threads", tags=["threads"])


def _make_repo(session: AsyncSession) -> ThreadRepository:
    return ThreadRepository(session)


@router.post(
    "",
    response_model=CreateThreadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation thread",
)
async def create_thread(
    session: AsyncSession = Depends(get_db_session),
) -> CreateThreadResponse:
    """Creates a new isolated conversation thread and returns its UUID."""
    repo = _make_repo(session)
    thread_orm = await repo.create_thread()
    return CreateThreadResponse(thread_id=uuid.UUID(thread_orm.id))


@router.post(
    "/{thread_id}/messages",
    response_model=SendMessageResponse,
    summary="Send a message to the orchestrator within a thread",
)
async def send_message(
    thread_id: str,
    body: SendMessageRequest,
    session: AsyncSession = Depends(get_db_session),
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
) -> SendMessageResponse:
    """
    Sends a user message to the OrchestratorAgent.
    The full thread history is passed as context for multi-turn conversations.
    """
    repo = _make_repo(session)

    thread = await repo.get_thread(thread_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found.",
        )

    await repo.add_message(thread_id, MessageRole.USER, body.content)

    settings = get_settings()
    history_orm = await repo.get_messages(
        thread_id, limit=settings.max_thread_history * 2
    )
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_orm[:-1]
    ]

    logger.info(
        "message_received",
        thread_id=thread_id,
        question_preview=body.content[:80],
        history_len=len(history),
    )

    try:
        response_text = await orchestrator.answer(
            question=body.content,
            history=history,
        )
    except Exception as exc:
        logger.error("orchestrator_error", thread_id=thread_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestrator error: {str(exc)}",
        ) from exc

    await repo.add_message(thread_id, MessageRole.ASSISTANT, response_text)

    return SendMessageResponse(
        thread_id=uuid.UUID(thread_id),
        response=response_text,
    )


@router.get(
    "/{thread_id}/messages",
    response_model=MessageListResponse,
    summary="Get all messages in a thread",
)
async def get_messages(
    thread_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> MessageListResponse:
    """Returns the full ordered message history for the given thread."""
    repo = _make_repo(session)

    thread = await repo.get_thread(thread_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found.",
        )

    messages_orm = await repo.get_messages(thread_id)
    messages = [
        Message(
            id=uuid.UUID(m.id),
            thread_id=uuid.UUID(m.thread_id),
            role=MessageRole(m.role),
            content=m.content,
            created_at=m.created_at.replace(tzinfo=timezone.utc)
            if m.created_at.tzinfo is None
            else m.created_at,
        )
        for m in messages_orm
    ]
    return MessageListResponse(
        thread_id=uuid.UUID(thread_id),
        messages=messages,
    )


@router.get(
    "",
    response_model=ThreadListResponse,
    summary="List all threads",
)
async def list_threads(
    session: AsyncSession = Depends(get_db_session),
) -> ThreadListResponse:
    """Returns all conversation threads ordered by creation date."""
    repo = _make_repo(session)
    threads_orm = await repo.list_threads()

    threads = []
    for t in threads_orm:
        count = await repo.count_messages(t.id)
        threads.append(
            Thread(
                id=uuid.UUID(t.id),
                created_at=t.created_at.replace(tzinfo=timezone.utc)
                if t.created_at.tzinfo is None
                else t.created_at,
                message_count=count,
            )
        )
    return ThreadListResponse(threads=threads)
