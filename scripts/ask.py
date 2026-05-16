"""Ask the agent a single question from the command line.

Usage:
    uv run python scripts/ask.py "Compare LoRA and QLoRA"
"""
import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from arxiv_agent.agent.graph import run_agent


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python scripts/ask.py "<your question>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    console = Console()
    console.print(Panel(question, title="Question", border_style="cyan"))

    answer = run_agent(question)

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold]Type:[/bold] {answer.question_type}    "
        f"[bold]Confidence:[/bold] {answer.confidence}    "
        f"[bold]Iterations:[/bold] {answer.iterations_used}    "
        f"[bold]Papers:[/bold] {len(answer.papers_consulted)}",
        border_style="green",
    ))

    # Answer body
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


if __name__ == "__main__":
    main()