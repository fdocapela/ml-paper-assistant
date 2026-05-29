"""Async SQLAlchemy database engine, session factory, and ORM models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.logging import get_logger
from core.settings import get_settings

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class ThreadORM(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MessageORM(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


# ── Engine / Session Factory ──────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.sqlite_url,
            echo=False,
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def create_tables() -> None:
    """Create all tables on startup (idempotent)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_ready")


async def dispose_engine() -> None:
    engine = get_engine()
    await engine.dispose()


# ── Repository ────────────────────────────────────────────────────────────────

class ThreadRepository:
    """Data-access layer for threads and messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_thread(self) -> ThreadORM:
        thread = ThreadORM(id=str(uuid.uuid4()))
        self._session.add(thread)
        await self._session.commit()
        await self._session.refresh(thread)
        logger.info("thread_created", thread_id=thread.id)
        return thread

    async def get_thread(self, thread_id: str) -> ThreadORM | None:
        result = await self._session.execute(
            select(ThreadORM).where(ThreadORM.id == thread_id)
        )
        return result.scalar_one_or_none()

    async def list_threads(self, limit: int = 50) -> list[ThreadORM]:
        from sqlalchemy import text as sa_text
        query = (
            select(ThreadORM)
            .order_by(ThreadORM.created_at.desc(), sa_text("rowid desc"))
            .limit(limit)
        )
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def add_message(
        self, thread_id: str, role: str, content: str
    ) -> MessageORM:
        message = MessageORM(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            role=role,
            content=content,
        )
        self._session.add(message)
        await self._session.commit()
        await self._session.refresh(message)
        return message

    async def get_messages(
        self, thread_id: str, limit: int | None = None
    ) -> list[MessageORM]:
        from sqlalchemy import text as sa_text
        query = (
            select(MessageORM)
            .where(MessageORM.thread_id == thread_id)
            .order_by(MessageORM.created_at.asc(), sa_text("rowid"))
        )
        result = await self._session.execute(query)
        messages = list(result.scalars().all())
        if limit:
            messages = messages[-limit:]
        return messages

    async def count_messages(self, thread_id: str) -> int:
        result = await self._session.execute(
            select(func.count()).where(MessageORM.thread_id == thread_id)
        )
        return result.scalar_one()
