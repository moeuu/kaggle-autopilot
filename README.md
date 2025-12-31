# Kaggle Autopilot

A CLI tool that automates Kaggle competition workflows with safety guardrails.

## Features

- Download competition data via Kaggle Python API
- Build baseline models (MVP: tabular CSV competitions)
- Generate valid submission.csv matching sample_submission.csv
- Submit to Kaggle with safety guardrails (dry-run by default)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd kaggle-autopilot

# Install with uv
uv sync
```

## Prerequisites

1. Kaggle API authentication:
   - Uses OAuth token from `~/.kaggle/access_token` (automatic)
   - If not authenticated, run any Kaggle CLI command to trigger OAuth flow
2. Manually accept competition rules in your browser (required once per competition)

## Usage

### Bootstrap workspace (no network actions)

```bash
kagglebot bootstrap titanic
```

### Validate a submission (dry-run default)

```bash
kagglebot run titanic --submission path/to/submission.csv
```

By default, `--sample` resolves to `data/<slug>/raw/sample_submission.csv`.

### Record a submission (requires explicit force)

```bash
kagglebot run titanic --submission path/to/submission.csv --no-dry-run --force --message "baseline"
```

## Safety Features

- **Dry-run by default**: Use `--no-dry-run --force` to allow side effects
- **Duplicate detection**: Prevents recording identical submissions by hash
- **Strict validation**: Validates submission format against sample_submission.csv
- **No automated rule acceptance**: Users must manually accept rules in browser
- **Run ledger**: Records runs in `artifacts/<slug>/runs/<run_id>/metadata.json`
- **Submission ledger**: Records submissions in `artifacts/<slug>/submissions/ledger.jsonl`

## Project Structure

```
kaggle-autopilot/
├── src/
│   └── kagglebot/
│       ├── cli.py              # CLI entry point
│       ├── bootstrap.py        # Data download & setup
│       ├── tabular_baseline.py # Baseline model training
│       ├── validation.py       # Submission validation
│       ├── history.py          # Submission tracking
│       ├── kaggle_cli.py       # Kaggle API wrapper
│       ├── paths.py            # Path management
│       └── hashing.py          # File hashing utilities
├── data/                   # Downloaded datasets (gitignored)
├── artifacts/              # Models & submissions (gitignored)
└── tests/                  # Test suite
```

## Development

```bash
# Sync deps
uv sync

# Run tests
uv run pytest -q

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run pyright
```

## Limitations (MVP)

- Only supports tabular competitions with train.csv, test.csv, and sample_submission.csv
- Only supports single-target competitions (multi-target will be supported later)
- Basic baseline models only (Ridge regression or Logistic regression)

## License

MIT
