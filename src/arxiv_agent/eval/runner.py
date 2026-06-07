"""Run the agent against the eval dataset and save raw answers to disk."""

from __future__ import annotations

import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from arxiv_agent.agent.graph import run_agent
from arxiv_agent.agent.schemas import AgentAnswer
from arxiv_agent.config import PRIMARY_MODEL, PROJECT_ROOT
from arxiv_agent.eval.loader import EvalQuestion

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = PROJECT_ROOT / "eval_results"


def run_dir(run_name: str) -> Path:
    """Return the directory for a named eval run, creating it if needed."""
    d = RESULTS_ROOT / run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "answers").mkdir(exist_ok=True)
    (d / "scores").mkdir(exist_ok=True)
    return d


def answer_path(run_name: str, question_id: int) -> Path:
    return run_dir(run_name) / "answers" / f"{question_id:03d}.json"


def metadata_path(run_name: str) -> Path:
    return run_dir(run_name) / "run_metadata.json"


# ---------------------------------------------------------------------------
# Saving and loading answers
# ---------------------------------------------------------------------------


def save_answer(
    run_name: str,
    question: EvalQuestion,
    answer: AgentAnswer,
    *,
    error: str | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Persist a single question's answer to disk."""
    path = answer_path(run_name, question.id)
    payload: dict[str, Any] = {
        "question_id": question.id,
        "category": question.category,
        "question": question.question,
        "agent_answer": answer.model_dump(mode="json"),
        "error": error,
        "elapsed_seconds": elapsed_seconds,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_answer(run_name: str, question_id: int) -> dict | None:
    """Return the saved payload for a question, or None if not yet run."""
    path = answer_path(run_name, question_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def has_answer(run_name: str, question_id: int) -> bool:
    """Return True if this question already has a saved answer."""
    return answer_path(run_name, question_id).exists()


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def save_run_metadata(run_name: str, question_ids: list[int]) -> None:
    """Persist information about this run for later analysis."""
    meta = {
        "run_name": run_name,
        "started_at": datetime.now(UTC).isoformat(),
        "model": PRIMARY_MODEL,
        "question_ids": sorted(question_ids),
        "num_questions": len(question_ids),
    }
    metadata_path(run_name).write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_eval(
    questions: list[EvalQuestion],
    run_name: str,
    *,
    skip_existing: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the agent against a list of questions, saving each answer to disk.

    Args:
        questions: The questions to run.
        run_name: Identifier for this run. Determines the output directory.
        skip_existing: If True, skip questions that already have a saved answer.
            Set to False to force a fresh run.
        show_progress: If True, render a progress bar to stderr.

    Returns:
        A dict with counters: {ran, skipped, errored}.
    """
    console = Console(stderr=True)
    save_run_metadata(run_name, [q.id for q in questions])

    console.print(
        f"[bold]Starting eval run:[/bold] {run_name}    "
        f"questions: {len(questions)}    "
        f"output: {run_dir(run_name)}"
    )

    counters = {"ran": 0, "skipped": 0, "errored": 0}

    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ]

    progress = Progress(*progress_columns, console=console) if show_progress else None
    task_id = None
    if progress is not None:
        progress.start()
        task_id = progress.add_task("Running eval", total=len(questions))

    try:
        for i, question in enumerate(questions, start=1):
            label = f"[{i}/{len(questions)}] id={question.id} [{question.category}]"

            if skip_existing and has_answer(run_name, question.id):
                console.print(f"{label} [yellow]SKIP[/yellow] (already saved)")
                counters["skipped"] += 1
                if progress is not None and task_id is not None:
                    progress.advance(task_id)
                continue

            console.print(f"{label} {question.question[:80]!r}")

            t0 = time.time()
            try:
                answer = run_agent(question.question)
                elapsed = time.time() - t0
                save_answer(
                    run_name,
                    question,
                    answer,
                    error=None,
                    elapsed_seconds=elapsed,
                )
                console.print(
                    f"   [green]done[/green] in {elapsed:.1f}s "
                    f"({answer.question_type}, conf={answer.confidence}, "
                    f"iters={answer.iterations_used})"
                )
                counters["ran"] += 1
            except Exception as exc:  # noqa: BLE001 - we want to capture and continue
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                console.print(f"   [red]ERROR[/red]: {exc}")
                # Save a placeholder answer so we don't retry next time
                placeholder = AgentAnswer(
                    question=question.question,
                    question_type="cannot_answer",
                    answer=f"Agent run failed: {exc}",
                    confidence="low",
                    confidence_reason="Agent threw an exception.",
                )
                save_answer(
                    run_name,
                    question,
                    placeholder,
                    error=tb,
                    elapsed_seconds=elapsed,
                )
                counters["errored"] += 1

            if progress is not None and task_id is not None:
                progress.advance(task_id)
    finally:
        if progress is not None:
            progress.stop()

    console.print(
        f"\n[bold]Done.[/bold] ran={counters['ran']}, "
        f"skipped={counters['skipped']}, errored={counters['errored']}"
    )
    return counters
