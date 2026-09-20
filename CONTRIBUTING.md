# Contributing

## Getting started

1. Fork the repo and clone your fork.
2. `uv sync --dev` to install dependencies including pytest.
3. Copy `.env.example` to `.env` if you need API-backed features locally.

## Before you push

- Run the test suite: `uv run pytest` (and `python -m py_compile` on anything
  you touched if you want to be quick).
- Keep secrets out of the repo: never commit `.env`, API keys, tokens, or
  credentials. Tests must use fake values only.
- Keep generated artifacts out: `state/`, `runs/`, `data/projects/`,
  `models/`, and downloaded media are gitignored. Do not force-add them.
- No em dashes in user-facing content. Use hyphens, commas, or colons.

## Pull requests

- Small, focused PRs with a clear description of what changed and why.
- Add or update tests for behavior changes (see `tests/` for the existing
  style: fast, mocked, no network).
- Update `README.md` if you add a user-facing feature or config option.

## Reporting issues

Include the command you ran, the relevant log tail, your OS, Python version,
and FFmpeg version. Redact any API keys from logs before pasting.
