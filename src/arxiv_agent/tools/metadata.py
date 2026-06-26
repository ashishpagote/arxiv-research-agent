"""Tools for fetching paper metadata and verifying arXiv IDs."""

from __future__ import annotations

import re
import sys
from datetime import date

import arxiv

from arxiv_agent.tools._arxiv_client import fetch_results
from arxiv_agent.tools.cache import cache_get_json, cache_set_json
from arxiv_agent.tools.schemas import (
    MetadataResult,
    PaperMetadata,
    VerifyResult,
)

# ---------------------------------------------------------------------------
# arXiv ID validation
# ---------------------------------------------------------------------------

# arXiv IDs come in two formats:
#   - new style: YYMM.NNNNN  (e.g. 2106.09685)
#   - old style: archive.subject/YYMMNNN  (e.g. cs.LG/0612077)
#
# We only support new-style IDs for V1. Old-style IDs predate 2007 and are
# rare in modern ML/AI work. Optional version suffix like "v2" is allowed.
_ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def is_valid_arxiv_id_format(arxiv_id: str) -> bool:
    """Return True if the string looks like a well-formed arXiv ID.

    This is a *format* check, not an existence check. Use verify_arxiv_id
    to check whether a paper actually exists.
    """
    if not isinstance(arxiv_id, str):
        return False
    return bool(_ARXIV_ID_PATTERN.match(arxiv_id.strip()))


def _normalize_arxiv_id(arxiv_id: str) -> str:
    """Strip whitespace and a leading 'arxiv:' prefix if present."""
    cleaned = arxiv_id.strip()
    if cleaned.lower().startswith("arxiv:"):
        cleaned = cleaned[len("arxiv:") :].strip()
    return cleaned


# ---------------------------------------------------------------------------
# Paper metadata extraction
# ---------------------------------------------------------------------------


def _to_metadata(paper: arxiv.Result) -> PaperMetadata:
    """Convert an arxiv.Result into our PaperMetadata schema."""
    # arxiv.Result.entry_id looks like "http://arxiv.org/abs/2106.09685v2"
    raw_id = paper.entry_id.rsplit("/", 1)[-1]
    return PaperMetadata(
        arxiv_id=raw_id,
        title=paper.title.strip(),
        authors=[a.name for a in paper.authors],
        abstract=paper.summary.strip(),
        published_date=paper.published.date() if paper.published else date.today(),
        updated_date=paper.updated.date() if paper.updated else None,
        categories=list(paper.categories),
        pdf_url=paper.pdf_url,
        primary_category=paper.primary_category,
    )


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def verify_arxiv_id(arxiv_id: str) -> VerifyResult:
    """Check whether an arXiv ID exists.

    Args:
        arxiv_id: The arXiv ID to check (e.g. "2106.09685").

    Returns:
        A VerifyResult. On success, .exists indicates whether the paper
        was found. On error (e.g. malformed ID, network failure),
        .success is False and .error contains a description.
    """
    cleaned = _normalize_arxiv_id(arxiv_id)

    # Fast-path: format check rejects malformed IDs before hitting the API.
    if not is_valid_arxiv_id_format(cleaned):
        return VerifyResult(
            success=True,  # we successfully determined the answer
            arxiv_id=cleaned,
            exists=False,
        )

    # Cache: if we've seen this ID before, return the cached answer.
    cached = cache_get_json("verify", cleaned)
    if cached is not None:
        print(
            f"[verify_arxiv_id] Cache hit for {cleaned}",
            file=sys.stderr,
            flush=True,
        )
        return VerifyResult.model_validate(cached)

    # API call.
    print(
        f"[verify_arxiv_id] Querying arXiv for ID: {cleaned}",
        file=sys.stderr,
        flush=True,
    )
    try:
        search = arxiv.Search(id_list=[cleaned])
        results = fetch_results(search)
    except Exception as exc:  # noqa: BLE001 - we want to surface the error string
        return VerifyResult(
            success=False,
            arxiv_id=cleaned,
            exists=False,
            error=f"arXiv API error: {exc}",
        )

    if not results:
        result = VerifyResult(success=True, arxiv_id=cleaned, exists=False)
    else:
        paper = results[0]
        result = VerifyResult(
            success=True,
            arxiv_id=cleaned,
            exists=True,
            title=paper.title.strip(),
        )

    # Cache the result.
    cache_set_json("verify", cleaned, result.model_dump(mode="json"))
    return result


def get_paper_metadata(arxiv_id: str) -> MetadataResult:
    """Fetch full metadata for a given arXiv ID.

    Args:
        arxiv_id: The arXiv ID to fetch (e.g. "2106.09685").

    Returns:
        A MetadataResult. On success, .paper contains the full metadata.
    """
    cleaned = _normalize_arxiv_id(arxiv_id)

    if not is_valid_arxiv_id_format(cleaned):
        return MetadataResult(
            success=False,
            error=f"Malformed arXiv ID: {arxiv_id!r}",
        )

    cached = cache_get_json("metadata", cleaned)
    if cached is not None:
        print(
            f"[get_paper_metadata] Cache hit for {cleaned}",
            file=sys.stderr,
            flush=True,
        )
        return MetadataResult.model_validate(cached)

    print(
        f"[get_paper_metadata] Fetching metadata for: {cleaned}",
        file=sys.stderr,
        flush=True,
    )
    try:
        search = arxiv.Search(id_list=[cleaned])
        results = fetch_results(search)
    except Exception as exc:  # noqa: BLE001
        return MetadataResult(
            success=False,
            error=f"arXiv API error: {exc}",
        )

    if not results:
        return MetadataResult(
            success=False,
            error=f"No paper found with arXiv ID {cleaned!r}",
        )

    metadata = _to_metadata(results[0])
    result = MetadataResult(success=True, paper=metadata)
    cache_set_json("metadata", cleaned, result.model_dump(mode="json"))
    return result
