# Kaggle Autopilot

A CLI tool that automates Kaggle competition workflows with safety guardrails.

## Features

- Download competition data via Kaggle CLI
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

1. Install Kaggle CLI: `pip install kaggle`
2. Set up Kaggle API credentials:
   - Download `kaggle.json` from your Kaggle account settings
   - Place it at `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Manually accept competition rules in your browser (required once per competition)

## Usage

### End-to-end workflow (DRY RUN - default)

```bash
# Download data, train baseline, generate submission (no actual submission)
kagglebot run titanic
```

### Submit to Kaggle

```bash
# Add --submit flag to actually submit
kagglebot run titanic --submit --message "first baseline"
```

### Bootstrap only (download data)

```bash
kagglebot bootstrap titanic
```

## Safety Features

- **Dry-run by default**: Must explicitly use `--submit` flag
- **Duplicate detection**: Prevents submitting identical files (use `--force-submit` to override)
- **Strict validation**: Validates submission format against sample_submission.csv
- **No automated rule acceptance**: Users must manually accept rules in browser
- **Submission history**: Tracks all submissions in `artifacts/<slug>/submissions/history.jsonl`

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
│       ├── kaggle_cli.py       # Kaggle CLI wrapper
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
