# arXiv Research Agent

An LLM-driven agent that takes a research question, searches arXiv, reads papers, and produces a structured, cited synthesis with confidence assessments and rigorous evaluation.

Built as a foundation track project to demonstrate agent design + evaluation infrastructure.

## What it does

Given a research question (e.g., *"Compare LoRA and QLoRA — when should I use each?"*), the agent:

1. Searches arXiv for relevant papers
2. Filters and reads the most relevant ones
3. Synthesizes a structured answer with citations
4. Assesses its own confidence in the answer
5. Refuses to answer when the question is unanswerable, subjective, or based on a fabricated premise

## Project status

🚧 In development. See [progress tracker](#progress) below.

## Architecture

- **Single agent (Pattern A)** with open-ended tool use loop (YOLO).
- **4 tools:** `search_arxiv`, `get_paper_metadata`, `get_paper_full_text`, `verify_arxiv_id`.
- **Output:** Structured JSON with question type, answer, citations, confidence, and audit trail.

Future ablations: routed agent (Pattern B), structured workflow, RAG-over-paper.

## Evaluation

- Golden dataset of 60 questions across 6 categories: literature review, comparison, specific result lookup, explain a paper, trap, edge case.
- Two-axis evaluation: deterministic retrieval check + LLM-as-judge synthesis check.
- Trap-refusal rate is the headline metric.

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd arxiv-research-agent
uv sync
cp .env.example .env
# Add your Anthropic API key to .env
```

## Usage

(Coming soon — agent CLI not built yet.)

## Progress

- [x] Project scope locked
- [x] Tools designed (4 functions)
- [x] Golden dataset built (60 questions)
- [x] Architecture: single agent, YOLO loop
- [x] Output schema locked
- [x] Repo setup
- [x] Tools implemented (33 tests passing)
- [x] Agent built (single-agent YOLO loop, Langfuse-traced)
- [x] CLI entry point (`uv run arxiv-agent "..."`)
- [ ] Eval harness
- [ ] Ablations + writeup

## License

MIT