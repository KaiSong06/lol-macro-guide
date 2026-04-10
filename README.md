# lol-macro-guide

Real-time vision-grounded jungle coach for League of Legends. Free, local, open-source.
Windows-only. Voice-only MVP. See [`docs/plans/2026-04-10-001-feat-jungle-coach-mvp-plan.md`](docs/plans/2026-04-10-001-feat-jungle-coach-mvp-plan.md) for the full implementation plan and [`TODOS.md`](TODOS.md) for the post-MVP backlog.

> **Status:** pre-alpha, under active development. No release cut yet.

## Requirements

- Windows 10 or 11
- Python 3.11 or 3.12
- 12–16 GB VRAM GPU (shared with League)
- [Ollama](https://ollama.com/download) with a vision model (final tag TBD after Week 1 benchmark)
- League of Legends client (active game required for runtime)

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest

# Lint
ruff check src tests
```

Full install, usage, and distribution docs land in Unit 12 of the implementation plan.
