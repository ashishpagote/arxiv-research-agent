"""Shared arXiv client with proper rate limiting."""

from __future__ import annotations

import arxiv

from arxiv_agent.config import ARXIV_DELAY_SECONDS, ARXIV_NUM_RETRIES

_client: arxiv.Client | None = None


def get_client() -> arxiv.Client:
    """Return a singleton arxiv.Client configured for polite rate limiting."""
    global _client
    if _client is None:
        _client = arxiv.Client(
            page_size=100,
            delay_seconds=ARXIV_DELAY_SECONDS,
            num_retries=ARXIV_NUM_RETRIES,
        )
    return _client
