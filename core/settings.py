from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ── Google AI ─────────────────────────────────────────────────────────────
    google_api_key: str = Field(..., description="Google AI Studio API key")
    gemini_model: str = Field(default="gemini-2.0-flash")
    embedding_model: str = Field(default="models/embedding-001")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)
    chroma_collection: str = Field(default="ml_papers")

    # ── SQLite ────────────────────────────────────────────────────────────────
    sqlite_db_path: str = Field(default="/data/threads.db")

    @property
    def sqlite_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.sqlite_db_path}"

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)
    retrieval_top_k: int = Field(default=6)

    # ── Conversation ──────────────────────────────────────────────────────────
    max_thread_history: int = Field(default=10)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance (singleton)."""
    return Settings()
