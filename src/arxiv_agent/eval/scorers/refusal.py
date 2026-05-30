"""Binary refusal scorer.

For questions where the agent SHOULD refuse (traps and edge cases),
check whether the agent's question_type is 'cannot_answer'.

Also catches false refusals: when the agent refuses a question it
should have answered (e.g. a real lit review request).
"""

from __future__ import annotations

from pydantic import BaseModel

from arxiv_agent.eval.loader import EvalQuestion

# ---------------------------------------------------------------------------
# Score schema
# ---------------------------------------------------------------------------


class RefusalScore(BaseModel):
    """Result of the refusal check for one question."""

    question_id: int
    should_refuse: bool
    actually_refused: bool
    passed: bool
    failure_mode: str | None = None  # 'failed_to_refuse' or 'false_refusal' or None


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------


def score_refusal(
    question: EvalQuestion,
    agent_question_type: str,
) -> RefusalScore:
    """Score the refusal behavior for a single question.

    Args:
        question: The EvalQuestion (defines whether this should be refused).
        agent_question_type: The question_type the agent assigned.

    Returns:
        A RefusalScore with pass/fail and failure mode if applicable.
    """
    should_refuse = (
        question.refusal_requirements is not None
        and question.refusal_requirements.should_refuse
    )
    actually_refused = agent_question_type == "cannot_answer"

    if should_refuse and not actually_refused:
        # The most dangerous failure: agent answered a trap/edge case as if it were legit
        return RefusalScore(
            question_id=question.id,
            should_refuse=True,
            actually_refused=False,
            passed=False,
            failure_mode="failed_to_refuse",
        )

    if not should_refuse and actually_refused:
        # The other failure: agent refused a legitimate question
        return RefusalScore(
            question_id=question.id,
            should_refuse=False,
            actually_refused=True,
            passed=False,
            failure_mode="false_refusal",
        )

    # Both match: either both refused (correct refusal) or both answered (correct answer)
    return RefusalScore(
        question_id=question.id,
        should_refuse=should_refuse,
        actually_refused=actually_refused,
        passed=True,
        failure_mode=None,
    )
