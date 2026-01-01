# Kaggle Autopilot

A CLI tool that automates Kaggle competition workflows with safety guardrails and compute switching (local CPU/GPU or Kaggle notebook GPU/TPU).

## Features

- **End-to-end pipeline**: bootstrap → implement (agent) → train → submit
- **Compute switching**: local CPU/GPU or Kaggle notebook GPU/TPU
- **GPU auto-detection**: detects CUDA/MPS for local GPU runs
- **Guardrails**: explicit submission flags, dedupe ledger, strict validation
- **Non-interactive**: no prompts by default
- **Artifacts & logs**: structured run metadata and agent transcripts

## Prerequisites

- **Python**: 3.11+
- **uv**: https://astral.sh/uv
- **Kaggle CLI**: `kaggle --version`
- **Kaggle auth**: `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`
- **Rules acceptance**: manually accept rules in the browser

## Setup

```bash
uv sync
```

## Commands

### Bootstrap
Creates artifacts, prompts, and plan.json. Optionally downloads data.

```bash
uv run kagglebot bootstrap titanic --rules-source url
uv run kagglebot bootstrap titanic --download --rules-source url --force
```

### Implement (agent)
Runs Codex or Claude on a clean git worktree, verifies, and optionally commits.

```bash
uv run kagglebot implement titanic --agent codex --no-commit
```

### Train
Trains locally or in a Kaggle notebook. Local runs write to `artifacts/<slug>/submissions/<runid>_submission.csv`.

```bash
uv run kagglebot train titanic --compute local_cpu
uv run kagglebot train titanic --compute local_gpu --strict-accelerator
uv run kagglebot train titanic --compute kaggle_gpu --kaggle-username <user> --force
```

### Submit
Validates submission, checks dedupe ledger, then submits via Kaggle CLI.

```bash
uv run kagglebot submit titanic \
  -f artifacts/titanic/submissions/<runid>_submission.csv \
  -m "baseline v1" \
  --force
```

### Run (orchestrated)
Bootstrap → implement → train → optional submit.

```bash
uv run kagglebot run titanic \
  --agent codex \
  --compute local_cpu \
  --submit \
  --message "baseline v1" \
  --force
```

## Compute Modes

- **local_cpu**: Train locally on CPU (default)
- **local_gpu**: Train locally on GPU (CUDA/MPS if available)
- **kaggle_gpu**: Train in a Kaggle Notebook with GPU
- **kaggle_tpu**: Train in a Kaggle Notebook with TPU

Optional flags:
```bash
--strict-accelerator   # Fail if local GPU not available
--enable-internet      # Kaggle kernel internet access (only if rules allow)
--kaggle-username      # Required for kaggle_* if not in env/config
--dry-run              # Skip external commands
```

## Safety Guardrails

- **Rules acceptance**: Never automated; exits with rules URL if missing
- **Submission validation**: Columns + row count + ID alignment
- **Deduplication**: SHA256 ledger under `artifacts/<slug>/submissions/history.jsonl`
- **Rate limiting**: Prevents rapid repeated submissions
- **Explicit submission**: Requires `--submit` and `--message`
- **Force flags**: `--force` for Kaggle CLI side effects, `--force-duplicate` to bypass dedupe

## Artifacts Layout

```
artifacts/<slug>/
├── context/            # meta.json, plan.json, rules
├── prompts/            # codex.md, claude.md
├── data/               # downloaded CSV/ZIP files
├── runs/
│   └── <run_id>/
│       ├── kernel/     # kernel package (kaggle_* compute)
│       ├── output/     # kernel outputs
│       └── <agent>/    # transcripts + last_message
└── submissions/
    └── history.jsonl   # submission ledger
```

## Testing

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -q
```
