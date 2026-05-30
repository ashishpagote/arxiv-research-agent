"""Tests for arxiv_agent.agent.retry."""

from __future__ import annotations

import httpx
import pytest
from anthropic import APIStatusError, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_none

from arxiv_agent.agent.retry import _is_rate_limit

# ---------------------------------------------------------------------------
# Helpers — build real anthropic exceptions without network calls
# ---------------------------------------------------------------------------


def _make_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code, request=request)


def _rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limited", response=_make_response(429), body={})


def _api_status_error(status_code: int) -> APIStatusError:
    return APIStatusError("error", response=_make_response(status_code), body={})


# Fast retry for unit tests — zero wait, 3 attempts max
_fast_retry = retry(
    retry=retry_if_exception(_is_rate_limit),
    stop=stop_after_attempt(3),
    wait=wait_none(),
    reraise=True,
)


# ---------------------------------------------------------------------------
# _is_rate_limit
# ---------------------------------------------------------------------------


def test_detects_rate_limit_error():
    assert _is_rate_limit(_rate_limit_error()) is True


def test_detects_api_status_429():
    assert _is_rate_limit(_api_status_error(429)) is True


def test_ignores_api_status_500():
    assert _is_rate_limit(_api_status_error(500)) is False


def test_detects_langchain_wrapped_429():
    """LangChain may wrap the anthropic error as __cause__."""
    inner = _api_status_error(429)
    outer = RuntimeError("LangChain wrapper")
    outer.__cause__ = inner
    assert _is_rate_limit(outer) is True


def test_detects_deeply_nested_rate_limit():
    inner = _rate_limit_error()
    mid = ValueError("mid")
    mid.__cause__ = inner
    outer = RuntimeError("outer")
    outer.__cause__ = mid
    assert _is_rate_limit(outer) is True


def test_ignores_unrelated_error():
    assert _is_rate_limit(ValueError("something else")) is False


# ---------------------------------------------------------------------------
# Retry behaviour (uses _fast_retry with wait_none() so tests are instant)
# ---------------------------------------------------------------------------


def test_retries_on_rate_limit_then_succeeds():
    call_count = 0

    @_fast_retry
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _rate_limit_error()
        return "ok"

    assert flaky() == "ok"
    assert call_count == 3


def test_raises_after_max_attempts():
    @_fast_retry
    def always_limited():
        raise _rate_limit_error()

    with pytest.raises(RateLimitError):
        always_limited()


def test_does_not_retry_non_rate_limit():
    call_count = 0

    @_fast_retry
    def boom():
        nonlocal call_count
        call_count += 1
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError, match="not a rate limit"):
        boom()

    assert call_count == 1  # no retries
