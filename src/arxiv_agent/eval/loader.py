"""Load and validate the golden evaluation dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from arxiv_agent.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

Category = Literal[
    "literature_review",
    "comparison",
    "specific_result",
    "explain_paper",
    "trap",
    "edge_case",
]

RetrievalType = Literal["none", "topical", "specific_paper"]

ExpectedQuestionType = Literal[
    "literature_review",
    "comparison",
    "specific_result",
    "explain_paper",
    "cannot_answer",
]


class RetrievalRequirements(BaseModel):
    """What the agent must retrieve for this question."""

    type: RetrievalType
    required_topics: list[str] = Field(default_factory=list)
    minimum_papers_per_topic: int = 1
    minimum_total_papers: int = 0
    required_paper: str | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> "RetrievalRequirements":
        if self.type == "specific_paper" and not self.required_paper:
            raise ValueError(
                "retrieval_requirements.type='specific_paper' requires 'required_paper'."
            )
        if self.type == "topical" and not self.required_topics:
            raise ValueError(
                "retrieval_requirements.type='topical' requires non-empty 'required_topics'."
            )
        return self


class SynthesisRequirements(BaseModel):
    """What the agent's answer must do."""

    must_address: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    target_length_words: str | None = None
    expected_question_type: ExpectedQuestionType


class RefusalRequirements(BaseModel):
    """How the agent should refuse, if applicable."""

    should_refuse: bool = False
    refusal_reason: str | None = None


class EvalQuestion(BaseModel):
    """A single question in the golden dataset."""

    id: int
    question: str
    category: Category
    retrieval_requirements: RetrievalRequirements
    synthesis_requirements: SynthesisRequirements
    refusal_requirements: RefusalRequirements | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DEFAULT_DATASET_PATH = PROJECT_ROOT / "eval" / "golden_dataset.yaml"


def load_dataset(path: Path | str | None = None) -> list[EvalQuestion]:
    """Load and validate the golden dataset.

    Args:
        path: Optional path to the YAML file. Defaults to eval/golden_dataset.yaml.

    Returns:
        A list of validated EvalQuestion objects.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        pydantic.ValidationError: if any question fails schema validation.
    """
    file_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {file_path}")

    with file_path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected dataset to be a YAML list, got {type(raw).__name__}"
        )

    questions: list[EvalQuestion] = []
    errors: list[str] = []
    for entry in raw:
        try:
            questions.append(EvalQuestion.model_validate(entry))
        except Exception as exc:  # noqa: BLE001 - collect all errors for one report
            qid = entry.get("id", "?") if isinstance(entry, dict) else "?"
            errors.append(f"Question id={qid}: {exc}")

    if errors:
        raise ValueError(
            "Dataset validation failed:\n" + "\n".join(errors[:10])
            + ("\n... (truncated)" if len(errors) > 10 else "")
        )

    return questions


def get_subset(
    questions: list[EvalQuestion],
    n: int | None = None,
    categories: list[Category] | None = None,
    ids: list[int] | None = None,
) -> list[EvalQuestion]:
    """Filter the dataset for a smaller eval run.

    Args:
        questions: The full dataset.
        n: If set, take up to this many questions (sampled across categories).
        categories: If set, only include questions in these categories.
        ids: If set, only include questions with these IDs.

    Returns:
        A filtered list of questions.
    """
    pool = questions

    if ids is not None:
        ids_set = set(ids)
        pool = [q for q in pool if q.id in ids_set]

    if categories is not None:
        cat_set = set(categories)
        pool = [q for q in pool if q.category in cat_set]

    if n is None or n >= len(pool):
        return pool

    # Take a balanced sample across categories
    by_cat: dict[str, list[EvalQuestion]] = {}
    for q in pool:
        by_cat.setdefault(q.category, []).append(q)

    # Round-robin across categories until we have n
    selected: list[EvalQuestion] = []
    while len(selected) < n and any(by_cat.values()):
        for cat in list(by_cat.keys()):
            if not by_cat[cat]:
                continue
            selected.append(by_cat[cat].pop(0))
            if len(selected) >= n:
                break

    return selected