"""Tests for the metadata tools."""

from arxiv_agent.tools.metadata import (
    get_paper_metadata,
    is_valid_arxiv_id_format,
    verify_arxiv_id,
)

# ---------------------------------------------------------------------------
# Format validation (no network calls)
# ---------------------------------------------------------------------------


class TestArxivIdFormat:
    def test_valid_new_style(self):
        assert is_valid_arxiv_id_format("2106.09685")

    def test_valid_with_version(self):
        assert is_valid_arxiv_id_format("2106.09685v2")

    def test_valid_4_digit_suffix(self):
        # Older new-style IDs sometimes have 4-digit suffixes
        assert is_valid_arxiv_id_format("0712.1234")

    def test_invalid_garbage(self):
        assert not is_valid_arxiv_id_format("not-an-id")

    def test_invalid_empty(self):
        assert not is_valid_arxiv_id_format("")

    def test_invalid_letters_in_id(self):
        assert not is_valid_arxiv_id_format("abcd.efghi")

    def test_invalid_non_string(self):
        assert not is_valid_arxiv_id_format(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# verify_arxiv_id (network calls, but cached after first run)
# ---------------------------------------------------------------------------


class TestVerifyArxivId:
    def test_real_paper_exists(self):
        result = verify_arxiv_id("2106.09685")  # LoRA
        assert result.success
        assert result.exists
        assert result.title is not None
        assert "LoRA" in result.title or "Low-Rank" in result.title

    def test_arxiv_prefix_handled(self):
        result = verify_arxiv_id("arxiv:2106.09685")
        assert result.success
        assert result.exists

    def test_malformed_id_returns_not_exists(self):
        result = verify_arxiv_id("not-an-id")
        assert result.success  # we determined the answer
        assert not result.exists

    def test_fake_but_valid_format(self):
        # 9999.99999 is format-valid but doesn't refer to a real paper.
        # This is the key behavior for trap detection.
        result = verify_arxiv_id("9999.99999")
        assert result.success
        assert not result.exists


# ---------------------------------------------------------------------------
# get_paper_metadata
# ---------------------------------------------------------------------------


class TestGetPaperMetadata:
    def test_real_paper_full_metadata(self):
        result = get_paper_metadata("2106.09685")  # LoRA
        assert result.success
        assert result.paper is not None
        p = result.paper
        assert "LoRA" in p.title or "Low-Rank" in p.title
        assert len(p.authors) > 0
        assert len(p.abstract) > 100
        assert "cs.CL" in p.categories or "cs.LG" in p.categories
        assert p.pdf_url.startswith("http")

    def test_malformed_id(self):
        result = get_paper_metadata("not-an-id")
        assert not result.success
        assert result.error is not None

    def test_nonexistent_id(self):
        result = get_paper_metadata("9999.99999")
        assert not result.success
        assert result.error is not None
