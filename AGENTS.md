# Agent Instructions (Codex / coding agents)

## Role
You are the implementer/tester for this repo.
- Prefer small, testable changes with clear diffs.
- If asked for architecture or design review, ask for guidance or route to Claude Code (see CLAUDE.md).

## Repository purpose
Build a Kaggle competition automation CLI:
- download data via Kaggle CLI
- train a robust baseline (MVP: tabular CSV)
- generate a valid submission.csv (must match sample_submission.csv)
- optionally submit via Kaggle CLI with strong guardrails

## Hard constraints (must follow)
- Do NOT automate accepting rules / joining competitions in the browser.
- Do NOT scrape Kaggle pages.
- Do NOT bypass limits, spam submissions, or encourage multi-account behavior.
- Do NOT write or commit secrets:
  - API credentials, tokens
- Do NOT commit large datasets or artifacts.

## Operational safety defaults
- Default to DRY RUN for end-to-end command.
- Submissions require an explicit flag (e.g., `--submit`) AND a human-readable message.
- Require `--force` to allow side effects beyond local validation/ledgers.
- Implement duplicate submission detection (hash + local history).
- Implement strict submission validation:
  - identical columns to sample_submission.csv
  - matching row count
  - align by id column when present

## Development workflow
1) Before large changes, produce a short plan and list touched files.
2) Keep changes minimal and well-tested.
3) Run unit tests (`uv run pytest -q`) and linters (`uv run ruff check .`, `uv run ruff format .`) before concluding.
4) Update docs (README/CLAUDE.md) if behavior changes.

## Tooling
- Use `uv` for dependency management and command execution only (no pip/poetry).
- Use `uv add/remove` for dependencies.
- Keep `uv.lock` committed.
- Standard test run: `uv sync` then `uv run pytest -q`.

## Coding standards
- Python 3.11+ recommended
- Use Kaggle CLI via subprocess; authenticate via `~/.kaggle/kaggle.json` or env vars
- Clear exceptions + actionable error messages
- Deterministic runs (seed control) when feasible

## Notes on Kaggle CLI integration
- Use Kaggle CLI: `kaggle competitions download -c <slug>`, `kaggle competitions submit -c <slug> ...`
- If a CLI call fails due to missing rule acceptance, print the Rules URL and exit.

## What “done” looks like for MVP
- `kagglebot run <slug>` downloads, trains, and produces a valid submission.csv in artifacts/
- `kagglebot run <slug> --submit` submits once (with guardrails)
- Works on a common tabular competition (e.g., Titanic-like structure)
