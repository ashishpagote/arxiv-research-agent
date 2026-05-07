"""
Sanity check script: verifies arXiv API + Anthropic API both work.

Run with:
    uv run python scripts/hello_world.py
"""
import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()


def check_env():
    """Verify required env vars are set."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("sk-ant-xxxxx"):
        console.print("[red]ANTHROPIC_API_KEY not set in .env[/red]")
        sys.exit(1)
    console.print("[green]✓[/green] ANTHROPIC_API_KEY found")
    return api_key


def test_arxiv():
    """Fetch a known paper from arXiv to verify the API works."""
    import arxiv

    console.print("\n[bold]Testing arXiv API...[/bold]")

    # Fetch the LoRA paper as a known reference
    client = arxiv.Client()
    search = arxiv.Search(id_list=["2106.09685"])
    results = list(client.results(search))

    if not results:
        console.print("[red]✗ No results returned from arXiv[/red]")
        sys.exit(1)

    paper = results[0]
    console.print(f"[green]✓[/green] Fetched paper: [bold]{paper.title}[/bold]")
    console.print(f"  Authors: {', '.join(a.name for a in paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
    console.print(f"  Published: {paper.published.date()}")
    console.print(f"  Categories: {', '.join(paper.categories)}")
    return paper


def test_anthropic(api_key: str, paper):
    """Send a small request to Claude to verify the API works."""
    from anthropic import Anthropic

    console.print("\n[bold]Testing Anthropic API...[/bold]")

    client = Anthropic(api_key=api_key)
    model = os.getenv("PRIMARY_MODEL", "claude-sonnet-4-6")

    message = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    f"In one sentence, summarize what this paper is about based on its title:\n\n"
                    f"Title: {paper.title}"
                ),
            }
        ],
    )

    response_text = message.content[0].text
    console.print(f"[green]✓[/green] Claude responded with model [bold]{model}[/bold]")
    console.print(Panel(response_text, title="Claude's response", border_style="cyan"))
    console.print(f"  Tokens used: {message.usage.input_tokens} input, {message.usage.output_tokens} output")


def main():
    console.print(Panel.fit("arXiv Research Agent - Environment Sanity Check", style="bold blue"))

    api_key = check_env()
    paper = test_arxiv()
    test_anthropic(api_key, paper)

    console.print("\n[bold green]All checks passed.[/bold green] Environment is ready.")


if __name__ == "__main__":
    main()