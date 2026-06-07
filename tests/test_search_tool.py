"""Tests for search_arxiv."""

import pytest

from arxiv_agent.tools.search import _build_query, search_arxiv

# ---------------------------------------------------------------------------
# Query construction (no network)
# ---------------------------------------------------------------------------


class TestBuildQuery:
    def test_simple_query_passthrough(self):
        assert _build_query("speculative decoding", None) == "speculative decoding"

    def test_query_is_stripped(self):
        assert _build_query("  LoRA  ", None) == "LoRA"

    def test_year_range_appended(self):
        result = _build_query("RLHF", (2023, 2024))
        assert "RLHF" in result
        assert "submittedDate:" in result
        assert "20230101" in result
        assert "20241231" in result

    def test_invalid_year_range_raises(self):
        with pytest.raises(ValueError):
            _build_query("test", (2025, 2020))

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            _build_query("   ", None)


# ---------------------------------------------------------------------------
# Input validation (no network)
# ---------------------------------------------------------------------------


class TestSearchInputValidation:
    def test_empty_query(self):
        result = search_arxiv("")
        assert not result.success
        assert "non-empty" in result.error.lower()

    def test_whitespace_query(self):
        result = search_arxiv("   ")
        assert not result.success

    def test_max_results_too_high(self):
        result = search_arxiv("test", max_results=500)
        assert not result.success
        assert "max_results" in result.error.lower()

    def test_max_results_zero(self):
        result = search_arxiv("test", max_results=0)
        assert not result.success

    def test_invalid_year_range_returns_error(self):
        result = search_arxiv("test", year_range=(2025, 2020))
        assert not result.success


# ---------------------------------------------------------------------------
# Real searches (cached after first run)
# ---------------------------------------------------------------------------


class TestSearchArxiv:
    def test_lora_search_returns_results(self):
        result = search_arxiv("LoRA low-rank adaptation", max_results=5)
        assert result.success
        assert result.total_returned > 0
        assert len(result.papers) <= 5

        text_blob = " ".join(f"{p.title} {p.abstract}".lower() for p in result.papers)
        assert "lora" in text_blob or "low-rank" in text_blob

    def test_papers_have_required_fields(self):
        result = search_arxiv("transformer attention", max_results=3)
        assert result.success
        assert len(result.papers) > 0
        for p in result.papers:
            assert p.arxiv_id
            assert p.title
            assert p.abstract
            assert p.pdf_url.startswith("http")
            assert p.published_date is not None

    def test_year_range_filter(self):
        result = search_arxiv("language models", max_results=5, year_range=(2024, 2024))
        assert result.success
        for p in result.papers:
            assert p.published_date.year == 2024
