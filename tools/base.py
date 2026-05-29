"""Abstract base class for all tools."""

from abc import ABC, abstractmethod
from typing import Any

from core.models import ToolResult


class BaseTool(ABC):
    """
    All tools extend this class.

    A tool is:
      - Stateless (no memory, no LLM calls by default)
      - Async
      - Single-purpose
      - Always returns a ToolResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique machine-readable name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for function calling schema."""
        ...

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters."""
        ...
