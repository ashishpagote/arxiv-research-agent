"""Central configuration for the arxiv-agent package."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# === API keys ===
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")


# === Models ===
PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "claude-sonnet-4-6")
SMALL_MODEL: str = os.getenv("SMALL_MODEL", "claude-haiku-4-5")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-opus-4-7")


# === Behavior ===
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
MAX_PAPERS_PER_SEARCH: int = int(os.getenv("MAX_PAPERS_PER_SEARCH", "10"))


# === Paths ===
PROJECT_ROOT: Path = _PROJECT_ROOT
CACHE_DIR: Path = PROJECT_ROOT / ".paper_cache"
CACHE_DIR.mkdir(exist_ok=True)


# === arXiv ===
# arXiv asks for max 1 request per 3 seconds; we use a generous buffer
ARXIV_DELAY_SECONDS: float = 3.5
ARXIV_NUM_RETRIES: int = 5