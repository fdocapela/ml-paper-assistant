"""Domain models shared across the application."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ── Paper Registry ────────────────────────────────────────────────────────────

PAPER_REGISTRY: dict[str, dict[str, str]] = {
    "1706.03762": {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "filename": "attention_is_all_you_need.pdf",
    },
    "1810.04805": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "arxiv_id": "1810.04805",
        "filename": "bert.pdf",
    },
    "2005.11401": {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "arxiv_id": "2005.11401",
        "filename": "rag.pdf",
    },
    "2210.03629": {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "arxiv_id": "2210.03629",
        "filename": "react.pdf",
    },
    "2302.04761": {
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "arxiv_id": "2302.04761",
        "filename": "toolformer.pdf",
    },
}

PAPER_IDS = list(PAPER_REGISTRY.keys())


# ── Tool Models ───────────────────────────────────────────────────────────────

class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class ToolResult(BaseModel):
    """Standardised return type for every tool."""

    status: ToolStatus
    tool_name: str
    data: Any = None
    error: str | None = None

    @classmethod
    def success(cls, tool_name: str, data: Any) -> "ToolResult":
        return cls(status=ToolStatus.SUCCESS, tool_name=tool_name, data=data)

    @classmethod
    def failure(cls, tool_name: str, error: str) -> "ToolResult":
        return cls(status=ToolStatus.ERROR, tool_name=tool_name, error=error)


# ── RAG Models ────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single text chunk with its metadata."""

    chunk_id: str
    arxiv_id: str
    paper_title: str
    section: str | None = None
    text: str
    score: float | None = None


class SearchResult(BaseModel):
    chunks: list[DocumentChunk]
    query: str


# ── Thread / Conversation Models ─────────────────────────────────────────────

class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    id: UUID
    thread_id: UUID
    role: MessageRole
    content: str
    created_at: datetime


class Thread(BaseModel):
    id: UUID
    created_at: datetime
    message_count: int = 0


# ── API Request / Response Models ─────────────────────────────────────────────

class CreateThreadResponse(BaseModel):
    thread_id: UUID


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4096)


class SendMessageResponse(BaseModel):
    thread_id: UUID
    response: str


class ThreadListResponse(BaseModel):
    threads: list[Thread]


class MessageListResponse(BaseModel):
    thread_id: UUID
    messages: list[Message]
