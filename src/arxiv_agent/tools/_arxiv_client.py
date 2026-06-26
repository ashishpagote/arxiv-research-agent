"""Shared arXiv client with proper rate limiting."""

from __future__ import annotations

import threading

import arxiv

from arxiv_agent.config import ARXIV_DELAY_SECONDS, ARXIV_NUM_RETRIES

_client: arxiv.Client | None = None

# arXiv throttles per IP, and arxiv.Client only enforces ``delay_seconds``
# between requests made through the *same* client instance sequentially. When
# the agent emits several arXiv tool calls in one turn, LangGraph's ToolNode
# runs them concurrently on separate threads, so two requests can hit arXiv at
# once and trigger HTTP 429s. This lock serializes every fetch through the
# shared client so the configured delay is actually honored.
_fetch_lock = threading.Lock()


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


def fetch_results(search: arxiv.Search) -> list[arxiv.Result]:
    """Run a search through the shared client, serialized across threads.

    All arXiv network access should go through this helper rather than calling
    ``client.results()`` directly, so concurrent tool calls can't defeat the
    client's rate limiting (see ``_fetch_lock`` above).
    """
    client = get_client()
    with _fetch_lock:
        return list(client.results(search))
