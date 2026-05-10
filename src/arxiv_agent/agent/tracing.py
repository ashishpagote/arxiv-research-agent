"""Langfuse tracing integration for the agent.

If Langfuse keys are configured in the environment, calls to the agent are
traced automatically. If not configured, this module is a no-op and the
agent runs normally without tracing.
"""
from __future__ import annotations

import sys
from typing import Any

from arxiv_agent.config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)


def is_configured() -> bool:
    """Return True if Langfuse credentials are present."""
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def get_callback_handler() -> Any | None:
    """Return a Langfuse callback handler, or None if not configured.

    The handler can be passed to LangChain/LangGraph invocations via
    ``config={"callbacks": [handler]}``.
    """
    if not is_configured():
        return None

    try:
        from langfuse.langchain import CallbackHandler  # type: ignore
    except ImportError:
        print(
            "[tracing] langfuse.langchain not available; skipping tracing",
            file=sys.stderr,
            flush=True,
        )
        return None

    return CallbackHandler()


def flush() -> None:
    """Flush any pending traces. Call before the program exits."""
    if not is_configured():
        return
    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
        client.flush()
    except Exception as exc:  # noqa: BLE001
        print(f"[tracing] flush failed: {exc}", file=sys.stderr, flush=True)