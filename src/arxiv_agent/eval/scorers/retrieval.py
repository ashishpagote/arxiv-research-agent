"""Deterministic retrieval scorer.

For each question, checks whether the agent retrieved the papers it
should have. Uses substring matching against paper titles + abstracts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from arxiv_agent.eval.loader import EvalQuestion
from arxiv_agent.tools.metadata import get_paper_metadata

# ---------------------------------------------------------------------------
# Score schema
# ---------------------------------------------------------------------------


class TopicCheck(BaseModel):
    """Per-topic retrieval check result."""

    topic: str
    found: bool
    matching_papers: list[str] = Field(default_factory=list)  # arxiv_ids


class RetrievalScore(BaseModel):
    """Result of the retrieval check for one question."""

    question_id: int
    retrieval_type: str
    skipped: bool = False  # true when retrieval_type='none'
    passed: bool = False
    topics: list[TopicCheck] = Field(default_factory=list)
    required_paper_check: dict | None = None  # populated for specific_paper type
    notes: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_searchable_text(arxiv_ids: list[str]) -> dict[str, str]:
    """For each arxiv_id, return its title+abstract as one lowercase string.

    Uses cached metadata so this is fast after first run.
    Skips IDs whose metadata can't be fetched.
    """
    text_by_id: dict[str, str] = {}
    for aid in arxiv_ids:
        result = get_paper_metadata(aid)
        if not result.success or result.paper is None:
            continue
        p = result.paper
        combined = f"{p.title}\n{p.abstract}".lower()
        text_by_id[aid] = combined
    return text_by_id


def _check_topic(topic: str, text_by_id: dict[str, str]) -> TopicCheck:
    """Check whether any paper covers a given topic."""
    needle = topic.lower().strip()
    matching = [aid for aid, text in text_by_id.items() if needle in text]
    return TopicCheck(
        topic=topic,
        found=len(matching) > 0,
        matching_papers=matching,
    )


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------


def score_retrieval(
    question: EvalQuestion,
    papers_consulted: list[str],
) -> RetrievalScore:
    """Score the retrieval performance for a single question.

    Args:
        question: The EvalQuestion (defines requirements).
        papers_consulted: The arxiv_ids the agent fetched.

    Returns:
        A RetrievalScore with per-topic and overall pass/fail.
    """
    reqs = question.retrieval_requirements

    # Type 'none': no retrieval expected; treat as N/A.
    if reqs.type == "none":
        return RetrievalScore(
            question_id=question.id,
            retrieval_type="none",
            skipped=True,
            passed=True,  # vacuously passes
            notes="No retrieval required for this question.",
        )

    # Type 'specific_paper': must include the required paper.
    if reqs.type == "specific_paper":
        required = reqs.required_paper
        found = required in papers_consulted if required else False
        return RetrievalScore(
            question_id=question.id,
            retrieval_type="specific_paper",
            passed=found,
            required_paper_check={
                "required_paper": required,
                "found": found,
                "papers_consulted": papers_consulted,
            },
            notes=(
                None if found else f"Required paper {required} not in papers_consulted."
            ),
        )

    # Type 'topical': check that each required topic is covered.
    text_by_id = _build_searchable_text(papers_consulted)
    topic_checks = [_check_topic(t, text_by_id) for t in reqs.required_topics]

    topics_found_count = sum(1 for tc in topic_checks if tc.found)
    topics_required = len(reqs.required_topics)

    # Pass condition: every required topic must have at least one matching paper,
    # AND total paper count meets minimum_total_papers (if specified).
    all_topics_found = topics_found_count == topics_required
    meets_min_total = len(papers_consulted) >= reqs.minimum_total_papers
    passed = all_topics_found and meets_min_total

    notes = None
    if not all_topics_found:
        missing = [tc.topic for tc in topic_checks if not tc.found]
        notes = f"Missing topics: {missing}"
    elif not meets_min_total:
        notes = (
            f"Retrieved only {len(papers_consulted)} papers; "
            f"required at least {reqs.minimum_total_papers}."
        )

    return RetrievalScore(
        question_id=question.id,
        retrieval_type="topical",
        passed=passed,
        topics=topic_checks,
        notes=notes,
    )
