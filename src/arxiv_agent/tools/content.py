"""Tool for fetching the full text of arXiv papers."""

from __future__ import annotations

import sys
from io import BytesIO

import requests

from arxiv_agent.tools.cache import (
    cache_get_bytes,
    cache_get_json,
    cache_set_bytes,
    cache_set_json,
)
from arxiv_agent.tools.metadata import (
    _normalize_arxiv_id,
    get_paper_metadata,
    is_valid_arxiv_id_format,
)
from arxiv_agent.tools.schemas import FullTextResult

# ---------------------------------------------------------------------------
# PDF download + parsing
# ---------------------------------------------------------------------------


def _download_pdf(arxiv_id: str, pdf_url: str) -> bytes | None:
    """Download a PDF, using the byte-cache when possible."""
    cached = cache_get_bytes("pdf", arxiv_id)
    if cached is not None:
        print(
            f"[get_paper_full_text] PDF cache hit for {arxiv_id}",
            file=sys.stderr,
            flush=True,
        )
        return cached

    print(
        f"[get_paper_full_text] Downloading PDF: {pdf_url}",
        file=sys.stderr,
        flush=True,
    )
    try:
        # 60s timeout — arXiv PDFs are usually small but can be slow
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(
            f"[get_paper_full_text] PDF download failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None

    pdf_bytes = response.content
    cache_set_bytes("pdf", arxiv_id, pdf_bytes)
    return pdf_bytes


def _extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Return (full_text, num_pages) extracted from PDF bytes.

    Raises a ValueError if the PDF is unreadable.
    """
    # Import lazily so test discovery doesn't require pymupdf
    import pymupdf  # type: ignore

    try:
        doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    pages_text: list[str] = []
    try:
        for page in doc:
            pages_text.append(page.get_text())
    finally:
        doc.close()

    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        raise ValueError("PDF contained no extractable text (likely scanned images).")

    return full_text, len(pages_text)


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def get_paper_full_text(arxiv_id: str) -> FullTextResult:
    """Fetch and parse the full text of an arXiv paper.

    Args:
        arxiv_id: The arXiv ID to fetch (e.g. "2106.09685").

    Returns:
        A FullTextResult. On success, .text contains the extracted text.
        On error (malformed ID, missing paper, PDF parsing failure),
        .success is False and .error describes the issue.
    """
    cleaned = _normalize_arxiv_id(arxiv_id)

    if not is_valid_arxiv_id_format(cleaned):
        return FullTextResult(
            success=False,
            arxiv_id=cleaned,
            error=f"Malformed arXiv ID: {arxiv_id!r}",
        )

    # Cache check on extracted text
    cached = cache_get_json("fulltext", cleaned)
    if cached is not None:
        print(
            f"[get_paper_full_text] Text cache hit for {cleaned}",
            file=sys.stderr,
            flush=True,
        )
        return FullTextResult.model_validate(cached)

    # We need metadata to know the PDF URL and title
    metadata_result = get_paper_metadata(cleaned)
    if not metadata_result.success or metadata_result.paper is None:
        return FullTextResult(
            success=False,
            arxiv_id=cleaned,
            error=(metadata_result.error or f"Could not fetch metadata for {cleaned}"),
        )

    paper = metadata_result.paper

    # Download the PDF
    pdf_bytes = _download_pdf(paper.arxiv_id, paper.pdf_url)
    if pdf_bytes is None:
        return FullTextResult(
            success=False,
            arxiv_id=cleaned,
            title=paper.title,
            error="Failed to download PDF.",
        )

    # Extract text
    try:
        text, num_pages = _extract_text(pdf_bytes)
    except ValueError as exc:
        return FullTextResult(
            success=False,
            arxiv_id=cleaned,
            title=paper.title,
            error=str(exc),
        )

    result = FullTextResult(
        success=True,
        arxiv_id=cleaned,
        title=paper.title,
        text=text,
        num_pages=num_pages,
        char_count=len(text),
        truncated=False,
    )
    cache_set_json("fulltext", cleaned, result.model_dump(mode="json"))
    return result
