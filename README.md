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

1. Kaggle CLI installed and authenticated:
   - Ensure `kaggle` is on your PATH
   - Credentials in `~/.kaggle/kaggle.json` or via env vars
2. Manually accept competition rules in your browser (required once per competition)

## Usage

### Download data

```bash
kagglebot download https://www.kaggle.com/competitions/titanic --force
```

### Train and predict (baseline)

```bash
kagglebot train titanic
kagglebot predict titanic
```

### Submit (requires explicit force + message)

```bash
kagglebot submit titanic --message "baseline v1" --force
```

### End-to-end run (dry-run by default)

```bash
kagglebot run https://www.kaggle.com/competitions/titanic --submit --message "baseline v1" --no-dry-run --force
```

## Safety Features

- **Dry-run by default**: Use `--no-dry-run --force` to allow network actions
- **Duplicate detection**: Prevents recording identical submissions by hash
- **Strict validation**: Validates submission format against sample_submission.csv
- **No automated rule acceptance**: Users must manually accept rules in browser
- **Run ledger**: Records runs in `artifacts/<slug>/runs/<run_id>/metadata.json`
- **Submission ledger**: Records submissions in `artifacts/<slug>/submissions/history.jsonl`

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
│       ├── competition.py      # URL/slug parsing helpers
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
