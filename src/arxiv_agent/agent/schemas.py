"""Pydantic schemas for the agent's structured output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

QuestionType = Literal[
    "literature_review",
    "comparison",
    "specific_result",
    "explain_paper",
    "cannot_answer",
]

ConfidenceLevel = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single citation linking a claim to a specific paper."""

    arxiv_id: str = Field(
        description="The arXiv ID of the cited paper (e.g. '2106.09685')."
    )
    title: str = Field(
        description="The title of the cited paper."
    )
    supports_claim: str = Field(
        description=(
            "A brief description of the specific claim this citation supports. "
            "Should be a paraphrase of the claim, not a quote."
        )
    )


# ---------------------------------------------------------------------------
# Top-level agent output
# ---------------------------------------------------------------------------

class AgentAnswer(BaseModel):
    """The structured output produced by the agent for every query."""

    question: str = Field(
        default="",
        description="The user's original question (set by the runner, not the LLM).",
    )

    question_type: QuestionType = Field(
        description=(
            "The agent's classification of the question type. "
            "Use 'cannot_answer' for traps, edge cases, or any question "
            "the agent should refuse to answer."
        )
    )

    answer: str = Field(
        description=(
            "The agent's answer in markdown format. For refusals, this is "
            "the refusal message explaining why the question cannot be answered."
        )
    )

    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Claim-level citations. Each citation links a specific claim "
            "in the answer to a specific paper. May be empty for refusals."
        ),
    )

    confidence: ConfidenceLevel = Field(
        description=(
            "The agent's overall confidence in the answer. "
            "'high' = strong evidence from canonical sources. "
            "'medium' = reasonable evidence but some gaps. "
            "'low' = limited evidence or contested claims. "
            "For refusals, use 'high' (high confidence that the question cannot be answered)."
        )
    )

    confidence_reason: str = Field(
        description="A brief explanation of why this confidence level was chosen."
    )

    papers_consulted: list[str] = Field(
        default_factory=list,
        description=(
            "The arXiv IDs of all papers the agent fetched and read while "
            "researching the answer. May be a superset of cited papers."
        ),
    )

    iterations_used: int = Field(
        default=0,
        description=(
            "The number of agent reasoning steps used. "
            "Populated by the agent loop, not by the LLM itself."
        ),
    )