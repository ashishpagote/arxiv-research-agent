"""Pydantic schemas for tool inputs and outputs."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# === Paper representations ===


class PaperMetadata(BaseModel):
    """Lightweight paper metadata, returned by search and metadata tools."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: date
    updated_date: date | None = None
    categories: list[str] = Field(default_factory=list)
    pdf_url: str
    primary_category: str | None = None


# === Tool result wrappers ===


class ToolResult(BaseModel):
    """Base class for tool results. Each tool returns success or error info."""

    success: bool
    error: str | None = None


class VerifyResult(ToolResult):
    """Result of verify_arxiv_id."""

    arxiv_id: str
    exists: bool = False
    title: str | None = None  # populated if exists


class MetadataResult(ToolResult):
    """Result of get_paper_metadata."""

    paper: PaperMetadata | None = None


class SearchResult(ToolResult):
    """Result of search_arxiv."""

    query: str
    papers: list[PaperMetadata] = Field(default_factory=list)
    total_returned: int = 0


class FullTextResult(ToolResult):
    """Result of get_paper_full_text."""

    arxiv_id: str
    title: str | None = None
    text: str | None = None
    num_pages: int | None = None
    truncated: bool = False
    char_count: int | None = None
