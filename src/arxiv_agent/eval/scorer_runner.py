"""Score all answers in an eval run.

For each saved AgentAnswer, applies the retrieval and refusal scorers,
saves the result, and prints a summary.

LLM-as-judge synthesis scoring is added in a separate module later.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from arxiv_agent.config import PROJECT_ROOT
from arxiv_agent.eval.loader import EvalQuestion, load_dataset
from arxiv_agent.eval.runner import load_answer, run_dir
from arxiv_agent.eval.scorers.refusal import RefusalScore, score_refusal
from arxiv_agent.eval.scorers.retrieval import RetrievalScore, score_retrieval


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def score_path(run_name: str, question_id: int) -> Path:
    return run_dir(run_name) / "scores" / f"{question_id:03d}.json"


# ---------------------------------------------------------------------------
# Score one question
# ---------------------------------------------------------------------------

def score_question(run_name: str, question: EvalQuestion) -> dict[str, Any] | None:
    """Score a single question. Returns the score payload, or None if no answer exists."""
    saved = load_answer(run_name, question.id)
    if saved is None:
        return None

    ans = saved["agent_answer"]
    papers_consulted: list[str] = ans.get("papers_consulted", []) or []
    agent_qtype: str = ans.get("question_type", "")

    is_exception = saved.get("error") is not None
    is_unparseable = "failed to produce" in (ans.get("answer", "") or "")

    retrieval = score_retrieval(question, papers_consulted)
    refusal = score_refusal(question, agent_qtype)

    payload = {
        "question_id": question.id,
        "category": question.category,
        "agent_question_type": agent_qtype,
        "agent_confidence": ans.get("confidence"),
        "iterations_used": ans.get("iterations_used", 0),
        "is_exception": is_exception,
        "is_unparseable": is_unparseable,
        "retrieval": retrieval.model_dump(mode="json"),
        "refusal": refusal.model_dump(mode="json"),
    }

    # Save per-question score file
    path = score_path(run_name, question.id)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

    return payload


# ---------------------------------------------------------------------------
# Score the whole run
# ---------------------------------------------------------------------------

def score_run(run_name: str) -> dict[str, Any]:
    """Score every answered question in the run and print a summary."""
    console = Console()
    questions = load_dataset()
    by_id = {q.id: q for q in questions}

    # Find which questions have answers in this run
    answers_dir = run_dir(run_name) / "answers"
    answered_ids = sorted(
        int(f.stem) for f in answers_dir.glob("*.json")
    )

    scored: list[dict[str, Any]] = []
    for qid in answered_ids:
        if qid not in by_id:
            console.print(f"[yellow]Warning:[/yellow] qid={qid} in answers but not in dataset")
            continue
        payload = score_question(run_name, by_id[qid])
        if payload is not None:
            scored.append(payload)

    _print_summary(console, run_name, scored)
    return {"scored": scored, "count": len(scored)}


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(console: Console, run_name: str, scored: list[dict[str, Any]]) -> None:
    if not scored:
        console.print("[red]No questions scored.[/red]")
        return

    n = len(scored)

    # Per-question table
    table = Table(
        title=f"Scorecard: {run_name}  ({n} questions)",
        show_lines=False,
    )
    table.add_column("ID", justify="right")
    table.add_column("Category")
    table.add_column("Agent Type")
    table.add_column("Iters", justify="right")
    table.add_column("Retrieval")
    table.add_column("Refusal")
    table.add_column("Notes")

    for s in scored:
        ret = s["retrieval"]
        ref = s["refusal"]

        ret_str = "[green]PASS[/green]" if ret["passed"] else "[red]FAIL[/red]"
        if ret.get("skipped"):
            ret_str = "[dim]n/a[/dim]"

        ref_str = "[green]PASS[/green]" if ref["passed"] else "[red]FAIL[/red]"
        if ref.get("failure_mode"):
            ref_str = f"[red]{ref['failure_mode']}[/red]"

        notes = []
        if s["is_exception"]:
            notes.append("[red]exception[/red]")
        if s["is_unparseable"]:
            notes.append("[red]unparseable[/red]")
        if ret.get("notes"):
            notes.append(ret["notes"][:40])

        table.add_row(
            str(s["question_id"]),
            s["category"],
            s["agent_question_type"],
            str(s["iterations_used"]),
            ret_str,
            ref_str,
            " | ".join(notes),
        )

    console.print(table)

    # Aggregate metrics
    console.print()
    console.print(Table.grid().add_row(f"[bold]Aggregate metrics ({n} questions)[/bold]"))

    # Retrieval pass rate (excluding skipped)
    ret_evaluable = [s for s in scored if not s["retrieval"].get("skipped")]
    ret_pass = sum(1 for s in ret_evaluable if s["retrieval"]["passed"])
    if ret_evaluable:
        console.print(
            f"  Retrieval pass: {ret_pass}/{len(ret_evaluable)} "
            f"= {100 * ret_pass / len(ret_evaluable):.1f}%  "
            f"(skipped: {n - len(ret_evaluable)})"
        )

    # Refusal pass rate (overall, separated by should_refuse)
    ref_pass = sum(1 for s in scored if s["refusal"]["passed"])
    console.print(f"  Refusal pass: {ref_pass}/{n} = {100 * ref_pass / n:.1f}%")

    should_refuse = [s for s in scored if s["refusal"]["should_refuse"]]
    if should_refuse:
        trap_pass = sum(1 for s in should_refuse if s["refusal"]["passed"])
        console.print(
            f"    Trap-refusal rate: {trap_pass}/{len(should_refuse)} "
            f"= {100 * trap_pass / len(should_refuse):.1f}%"
        )
    should_not_refuse = [s for s in scored if not s["refusal"]["should_refuse"]]
    if should_not_refuse:
        false_refusals = sum(
            1 for s in should_not_refuse if s["refusal"]["failure_mode"] == "false_refusal"
        )
        console.print(
            f"    False-refusal rate: {false_refusals}/{len(should_not_refuse)} "
            f"= {100 * false_refusals / len(should_not_refuse):.1f}%"
        )

    # Health
    exceptions = sum(1 for s in scored if s["is_exception"])
    unparseable = sum(1 for s in scored if s["is_unparseable"])
    iter_cap_hits = sum(1 for s in scored if s["iterations_used"] >= 7)
    console.print(f"  Exceptions: {exceptions}")
    console.print(f"  Unparseable answers: {unparseable}")
    console.print(f"  Hit iteration cap (>=7): {iter_cap_hits}")

    # Per-category breakdown
    console.print()
    by_cat: dict[str, list] = defaultdict(list)
    for s in scored:
        by_cat[s["category"]].append(s)

    cat_table = Table(title="Per-category breakdown", show_lines=False)
    cat_table.add_column("Category")
    cat_table.add_column("N", justify="right")
    cat_table.add_column("Retrieval pass", justify="right")
    cat_table.add_column("Refusal pass", justify="right")
    cat_table.add_column("Exceptions", justify="right")
    cat_table.add_column("Unparseable", justify="right")

    for cat, items in sorted(by_cat.items()):
        ret_eval = [s for s in items if not s["retrieval"].get("skipped")]
        ret_pass = sum(1 for s in ret_eval if s["retrieval"]["passed"])
        ret_cell = (
            f"{ret_pass}/{len(ret_eval)}" if ret_eval else "n/a"
        )
        ref_pass = sum(1 for s in items if s["refusal"]["passed"])
        ex = sum(1 for s in items if s["is_exception"])
        up = sum(1 for s in items if s["is_unparseable"])
        cat_table.add_row(
            cat,
            str(len(items)),
            ret_cell,
            f"{ref_pass}/{len(items)}",
            str(ex),
            str(up),
        )
    console.print(cat_table)