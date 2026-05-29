"""Async wrapper for Google Generative AI using the google-genai SDK."""

import asyncio
from typing import Any

from google import genai
from google.genai import types

from core.logging import get_logger
from core.settings import get_settings

logger = get_logger(__name__)

# Two clients: embeddings need v1, generation (with tools) needs v1beta
_llm_client: genai.Client | None = None
_embed_client: genai.Client | None = None


def _get_llm_client() -> genai.Client:
    global _llm_client
    if _llm_client is None:
        settings = get_settings()
        _llm_client = genai.Client(
            api_key=settings.google_api_key,
            http_options={"api_version": "v1beta"},
        )
    return _llm_client


def _get_embed_client() -> genai.Client:
    """Client for embed_content — uses v1 (required for gemini-embedding-001)."""
    global _embed_client
    if _embed_client is None:
        settings = get_settings()
        _embed_client = genai.Client(
            api_key=settings.google_api_key,
            http_options={"api_version": "v1"},
        )
    return _embed_client


async def generate_content(
    prompt: str,
    system_instruction: str | None = None,
    history: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> Any:
    settings = get_settings()
    client = _get_llm_client()

    contents: list[Any] = []
    if history:
        for msg in history:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=prompt)])
    )

    config = types.GenerateContentConfig(
        temperature=0.2,
        system_instruction=system_instruction,
    )

    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=config,
    )

    logger.debug("llm_response", model=settings.gemini_model)
    return response


async def embed_text(text: str) -> list[float]:
    settings = get_settings()
    client = _get_embed_client()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
        ),
    )
    return result.embeddings[0].values


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed sequentially with automatic retry on rate limit."""
    settings = get_settings()
    client = _get_embed_client()

    async def _embed_one(text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: client.models.embed_content(
                model=settings.embedding_model,
                contents=text,
            ),
        )
        return result.embeddings[0].values

    async def _embed_with_retry(text: str) -> list[float]:
        wait = 5.0
        for attempt in range(8):
            try:
                return await _embed_one(text)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning("rate_limit_hit",
                                   attempt=attempt + 1,
                                   wait_seconds=wait)
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, 60.0)
                else:
                    raise
        raise RuntimeError(f"Failed after 8 attempts: {text[:50]}")

    results = []
    for text in texts:
        embedding = await _embed_with_retry(text)
        results.append(embedding)
        await asyncio.sleep(1.0)
    return results