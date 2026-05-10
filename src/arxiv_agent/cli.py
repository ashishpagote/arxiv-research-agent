"""Command-line interface for the arXiv research agent."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from arxiv_agent.agent.graph import run_agent

app = typer.Typer(
    help="Ask the arXiv research agent a question.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The research question to ask."),
    save: Path | None = typer.Option(
        None,
        "--save",
        "-s",
        help="Save the full structured answer as JSON to this path.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress the agent's progress messages on stderr.",
    ),
) -> None:
    """Ask the agent a research question."""
    console = Console()

    if quiet:
        # Redirect stderr to /dev/null for the duration of the run
        import os
        sys.stderr = open(os.devnull, "w")

    console.print(Panel(question, title="Question", border_style="cyan"))

    answer = run_agent(question)

    # Header summary
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Type:[/bold] {answer.question_type}    "
            f"[bold]Confidence:[/bold] {answer.confidence}    "
            f"[bold]Iterations:[/bold] {answer.iterations_used}    "
            f"[bold]Papers:[/bold] {len(answer.papers_consulted)}",
            border_style="green",
        )
    )

    # Answer body (rendered as markdown)
    console.print()
    console.print(Markdown(answer.answer))

    # Citations
    if answer.citations:
        console.print()
        console.print(f"[bold]Citations ({len(answer.citations)}):[/bold]")
        for c in answer.citations:
            console.print(f"  • [cyan]{c.arxiv_id}[/cyan]: {c.title}")
            console.print(f"    [dim]→ {c.supports_claim}[/dim]")

    # Footer
    console.print()
    console.print(f"[dim]Confidence reason: {answer.confidence_reason}[/dim]")
    console.print(f"[dim]Papers consulted: {answer.papers_consulted}[/dim]")

    # Optionally save full structured answer
    if save is not None:
        save.write_text(answer.model_dump_json(indent=2))
        console.print(f"\n[green]Saved structured answer to:[/green] {save}")


def main() -> None:
    """Entry point used by the console_scripts hook in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()