"""Run the agent against a subset of the golden dataset.

Usage:
    uv run python scripts/run_eval.py <run_name> [n_questions]

Examples:
    uv run python scripts/run_eval.py v1_first_pass 20
    uv run python scripts/run_eval.py v1_full
"""
from __future__ import annotations

import sys

from arxiv_agent.eval.loader import get_subset, load_dataset
from arxiv_agent.eval.runner import run_eval


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python scripts/run_eval.py <run_name> [n_questions]')
        sys.exit(1)

    run_name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None

    questions = load_dataset()
    subset = get_subset(questions, n=n)

    print(f"Run name: {run_name}")
    print(f"Questions: {len(subset)}")
    print(f"Categories: {sorted({q.category for q in subset})}")

    run_eval(subset, run_name=run_name, skip_existing=True)


if __name__ == "__main__":
    main()
