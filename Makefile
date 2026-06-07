SOURCES = src tests scripts

.PHONY: format lint install-hooks run-hooks

format:
	uv run isort $(SOURCES)
	uv run black $(SOURCES)

lint:
	uv run ruff check $(SOURCES)

install-hooks:
	uv sync
	uv run pre-commit install

run-hooks:
	uv run pre-commit run --all-files
