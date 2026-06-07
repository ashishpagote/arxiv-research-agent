"""Tests for get_paper_full_text."""

from arxiv_agent.tools.content import get_paper_full_text


class TestGetPaperFullText:
    def test_real_paper_extracts_text(self):
        """Fetch the LoRA paper and verify text extraction."""
        result = get_paper_full_text("2106.09685")
        assert result.success
        assert result.text is not None
        assert result.title is not None
        assert "LoRA" in result.title or "Low-Rank" in result.title
        # The paper has substantial text
        assert result.char_count is not None
        assert result.char_count > 10000
        assert result.num_pages is not None
        assert result.num_pages > 5

    def test_text_contains_expected_content(self):
        """The extracted text should contain content from the LoRA paper."""
        result = get_paper_full_text("2106.09685")
        assert result.success
        text_lower = result.text.lower()
        # Things that should appear in any reasonable extraction of LoRA
        assert "low-rank" in text_lower or "lora" in text_lower
        # The paper discusses adaptation and parameters
        assert "adapt" in text_lower
        assert "parameter" in text_lower

    def test_arxiv_prefix_handled(self):
        """The 'arxiv:' prefix should be stripped before lookup."""
        result = get_paper_full_text("arxiv:2106.09685")
        assert result.success

    def test_malformed_id(self):
        """Malformed IDs should fail fast without an API call."""
        result = get_paper_full_text("not-an-id")
        assert not result.success
        assert result.error is not None
        assert "malformed" in result.error.lower()

    def test_nonexistent_id_fails_cleanly(self):
        """Format-valid but nonexistent IDs should fail with a clear error."""
        result = get_paper_full_text("9999.99999")
        assert not result.success
        assert result.error is not None

    def test_cache_works(self):
        """Second fetch of the same paper should be near-instant."""
        import time

        # Prime the cache
        get_paper_full_text("2106.09685")

        # Second call should hit cache
        t0 = time.time()
        result = get_paper_full_text("2106.09685")
        elapsed = time.time() - t0

        assert result.success
        assert elapsed < 0.5  # cache hit should be fast
