"""Score a saved eval run.

Usage:
    uv run python scripts/score_eval.py <run_name>
"""

import sys

from arxiv_agent.eval.scorer_runner import score_run


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/score_eval.py <run_name>")
        sys.exit(1)
    score_run(sys.argv[1])


if __name__ == "__main__":
    main()
