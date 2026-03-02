# Run all tests
test:
    uv run pytest tests/ -v

# Lint and check code
lint:
    uv run ruff check

# Format code
format:
    uv run ruff format

# Lint + test (run before committing)
check: lint test