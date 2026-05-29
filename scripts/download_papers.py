#!/usr/bin/env python3
"""Download the 5 ML papers from arXiv."""

import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import PAPER_REGISTRY

PAPERS_DIR = Path(__file__).parent.parent / "papers"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"


async def download_paper(
    client: httpx.AsyncClient,
    arxiv_id: str,
    meta: dict[str, str],
) -> None:
    dest = PAPERS_DIR / meta["filename"]
    if dest.exists():
        print(f"  ✓ {meta['title']} — already downloaded")
        return

    url = f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf"
    print(f"  ↓ {meta['title']} ({arxiv_id})...")
    try:
        response = await client.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
        dest.write_bytes(response.content)
        print(f"    → saved to {dest.name} ({len(response.content) // 1024} KB)")
    except httpx.HTTPError as exc:
        print(f"    ✗ FAILED: {exc}")
        sys.exit(1)


async def main() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    print("📥 Downloading papers...\n")

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (research/ml-paper-assistant)"}
    ) as client:
        for arxiv_id, meta in PAPER_REGISTRY.items():
            await download_paper(client, arxiv_id, meta)

    print("\n✅ All papers downloaded.")


if __name__ == "__main__":
    asyncio.run(main())
