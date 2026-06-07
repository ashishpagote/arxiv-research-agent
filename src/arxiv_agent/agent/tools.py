"""LangChain tool wrappers around our arXiv tool functions.

The underlying tools (in arxiv_agent.tools.*) return Pydantic models.
The LLM sees these wrappers, which:
  - Have docstrings the LLM reads as tool descriptions
  - Return strings (not Pydantic objects)
  - Format errors and edge cases clearly so the LLM can react sensibly
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from arxiv_agent.tools.content import get_paper_full_text as _get_full_text
from arxiv_agent.tools.metadata import get_paper_metadata as _get_metadata
from arxiv_agent.tools.metadata import verify_arxiv_id as _verify
from arxiv_agent.tools.search import search_arxiv as _search

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_paper_brief(paper) -> dict:
    """Format a PaperMetadata into a brief dict for tool output."""
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors[:5] + (["..."] if len(paper.authors) > 5 else []),
        "published_date": str(paper.published_date),
        "categories": paper.categories,
        "abstract": paper.abstract,
    }


# ---------------------------------------------------------------------------
# Tools exposed to the LLM
# ---------------------------------------------------------------------------


@tool
def search_arxiv(
    query: str,
    max_results: int = 10,
    year_start: int | None = None,
    year_end: int | None = None,
) -> str:
    """Search arXiv for papers matching a query.

    Use this to find papers about a topic. Returns a JSON list of papers,
    each with arxiv_id, title, authors, publication date, categories, and
    abstract. Read the abstracts to decide which papers to fetch in full.

    Args:
        query: Search string. Free text works; arXiv syntax also works
            (e.g. "ti:LoRA", "au:hinton", "cat:cs.LG", "RLHF AND calibration").
        max_results: How many papers to return (1-100). Default 10.
        year_start: Optional. Only return papers from this year onward.
        year_end: Optional. Only return papers up to this year (inclusive).

    Returns:
        A JSON string. On success: {"papers": [...], "total": N}.
        On error: {"error": "..."}.
    """
    year_range = None
    if year_start is not None and year_end is not None:
        year_range = (year_start, year_end)
    elif year_start is not None or year_end is not None:
        return json.dumps(
            {
                "error": "Both year_start and year_end must be provided together, or neither."
            }
        )

    result = _search(query=query, max_results=max_results, year_range=year_range)
    if not result.success:
        return json.dumps({"error": result.error})

    return json.dumps(
        {
            "papers": [_format_paper_brief(p) for p in result.papers],
            "total": result.total_returned,
        }
    )


@tool
def verify_arxiv_id(arxiv_id: str) -> str:
    """Check whether an arXiv paper exists. Use this BEFORE discussing a paper a user mentions.

    This is the primary tool for trap detection. If the user references a
    paper like "the OpenAI o5 paper" or "arxiv:2401.99999", verify it exists
    before answering. If it doesn't, refuse with question_type='cannot_answer'.

    Args:
        arxiv_id: The arXiv ID to check (e.g. "2106.09685" or "arxiv:2106.09685").

    Returns:
        A JSON string with "exists" (bool) and optionally "title".
    """
    result = _verify(arxiv_id)
    if not result.success:
        return json.dumps({"error": result.error or "Verification failed."})

    return json.dumps(
        {
            "arxiv_id": result.arxiv_id,
            "exists": result.exists,
            "title": result.title,
        }
    )


@tool
def get_paper_metadata(arxiv_id: str) -> str:
    """Fetch full metadata for an arXiv paper (title, authors, abstract, categories).

    Use this when you need the abstract or other metadata for a specific paper
    without reading the full text. Cheaper than get_paper_full_text.

    Args:
        arxiv_id: The arXiv ID (e.g. "2106.09685").

    Returns:
        A JSON string with paper metadata, or {"error": "..."} on failure.
    """
    result = _get_metadata(arxiv_id)
    if not result.success or result.paper is None:
        return json.dumps({"error": result.error or "Metadata fetch failed."})

    return json.dumps(_format_paper_brief(result.paper))


@tool
def get_paper_full_text(arxiv_id: str) -> str:
    """Fetch the full text of an arXiv paper.

    Use this when you need to read the methodology, results, or specific
    content of a paper. Returns the entire paper text (can be 20k-100k chars).
    Use sparingly — reading 5 papers fully is expensive.

    Args:
        arxiv_id: The arXiv ID (e.g. "2106.09685").

    Returns:
        A JSON string with the full text, or {"error": "..."} on failure.
    """
    result = _get_full_text(arxiv_id)
    if not result.success:
        return json.dumps({"error": result.error or "Full text fetch failed."})

    return json.dumps(
        {
            "arxiv_id": result.arxiv_id,
            "title": result.title,
            "num_pages": result.num_pages,
            "char_count": result.char_count,
            "text": result.text,
        }
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_arxiv,
    verify_arxiv_id,
    get_paper_metadata,
    get_paper_full_text,
]
