"""Search tool for finding papers on arXiv."""

from __future__ import annotations

import json
import sys

import arxiv

from arxiv_agent.tools._arxiv_client import get_client
from arxiv_agent.tools.cache import cache_get_json, cache_set_json
from arxiv_agent.tools.metadata import _to_metadata
from arxiv_agent.tools.schemas import PaperMetadata, SearchResult

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _build_query(query: str, year_range: tuple[int, int] | None) -> str:
    """Combine the user's query with an optional year range filter.

    arXiv supports submittedDate filters in the form:
        submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]

    We only filter on year boundaries to keep things simple.
    """
    base = query.strip()
    if not base:
        raise ValueError("Query string must not be empty.")

    if year_range is None:
        return base

    start_year, end_year = year_range
    if start_year > end_year:
        raise ValueError(f"Invalid year range: start {start_year} > end {end_year}")
    date_filter = f"submittedDate:[{start_year}01010000 TO {end_year}12312359]"
    return f"{base} AND {date_filter}"


def _cache_key(query: str, max_results: int, year_range: tuple[int, int] | None) -> str:
    """Stable cache key for a search invocation."""
    payload = {
        "q": query.strip().lower(),
        "n": max_results,
        "y": list(year_range) if year_range else None,
    }
    return json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def search_arxiv(
    query: str,
    max_results: int = 10,
    year_range: tuple[int, int] | None = None,
) -> SearchResult:
    """Search arXiv for papers matching a query.

    Args:
        query: Search string. Supports arXiv's query syntax (e.g. ``ti:``,
            ``au:``, ``cat:``, ``AND``/``OR``). Plain free-text also works.
        max_results: Maximum number of papers to return (1-100).
        year_range: Optional (start_year, end_year) inclusive filter on
            submission date.

    Returns:
        A SearchResult whose ``.papers`` is a list of PaperMetadata
        (each including its abstract). On error, ``.success`` is False
        and ``.error`` describes the issue.
    """
    # Input validation
    if not isinstance(query, str) or not query.strip():
        return SearchResult(
            success=False,
            query=query if isinstance(query, str) else "",
            error="Query must be a non-empty string.",
        )

    if max_results < 1 or max_results > 100:
        return SearchResult(
            success=False,
            query=query,
            error=f"max_results must be between 1 and 100, got {max_results}.",
        )

    try:
        full_query = _build_query(query, year_range)
    except ValueError as exc:
        return SearchResult(success=False, query=query, error=str(exc))

    # Cache lookup
    cache_key = _cache_key(query, max_results, year_range)
    cached = cache_get_json("search", cache_key)
    if cached is not None:
        print(
            f"[search_arxiv] Cache hit for query={query!r}",
            file=sys.stderr,
            flush=True,
        )
        return SearchResult.model_validate(cached)

    # API call
    print(
        f"[search_arxiv] Querying arXiv: {full_query!r} (max={max_results})",
        file=sys.stderr,
        flush=True,
    )
    try:
        client = get_client()
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        raw_results = list(client.results(search))
        print(
            f"[search_arxiv] Got {len(raw_results)} results",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        return SearchResult(
            success=False,
            query=query,
            error=f"arXiv API error: {exc}",
        )

    papers: list[PaperMetadata] = [_to_metadata(r) for r in raw_results]
    # arXiv's submittedDate filter is approximate at the boundaries, so we
    # do a strict client-side filter to honor our tool's contract.
    if year_range is not None:
        start_year, end_year = year_range
        papers = [p for p in papers if start_year <= p.published_date.year <= end_year]
    result = SearchResult(
        success=True,
        query=query,
        papers=papers,
        total_returned=len(papers),
    )
    cache_set_json("search", cache_key, result.model_dump(mode="json"))
    return result
