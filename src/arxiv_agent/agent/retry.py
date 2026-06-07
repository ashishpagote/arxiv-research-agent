"""Tenacity-based retry decorator for LLM calls."""

from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from arxiv_agent.config import (
    RETRY_BASE_WAIT_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_WAIT_SECONDS,
)

_logger = logging.getLogger(__name__)


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True for any 429 / rate-limit error, regardless of how it's wrapped.

    LangChain may re-raise anthropic SDK errors under a different type, preserving
    the original as __cause__ or __context__. We walk the chain so both the raw SDK
    error and any wrapper are caught.
    """
    from anthropic import APIStatusError, RateLimitError

    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
        return True

    # Walk the exception chain (LangChain wrapper → original anthropic error)
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _is_rate_limit(cause)

    # Last resort: string-match covers future wrappers that don't preserve the chain
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg


with_llm_retry = retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(
        multiplier=RETRY_BASE_WAIT_SECONDS,
        min=RETRY_BASE_WAIT_SECONDS,
        max=RETRY_MAX_WAIT_SECONDS,
    ),
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
    reraise=True,
)
