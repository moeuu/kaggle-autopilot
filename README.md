# Kaggle Autopilot

A CLI tool that automates Kaggle competition workflows with safety guardrails and flexible compute options (local CPU/GPU or Kaggle cloud GPU/TPU).

## Features

- **Automated pipeline**: Download → analyze → train → predict → validate → submit
- **Compute switching**: Train locally (CPU/GPU) or on Kaggle cloud (GPU/TPU)
- **GPU auto-detection**: Automatically detects CUDA/MPS, falls back to CPU
- **Production models**: Gradient boosting (LightGBM, CatBoost, XGBoost) with cross-validation
- **Safety guardrails**: Dry-run by default, duplicate detection, strict validation
- **Non-interactive**: All decisions via CLI flags/config, zero prompts
- **Submission tracking**: Local ledger with deduplication and audit trail

---

## Quickstart / Commands

### Prerequisites

**Python**: 3.11 or later

**uv package manager**: Install if not already available:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Kaggle CLI**: Ensure `kaggle` command is on your PATH:
```bash
# Test Kaggle CLI is installed
kaggle --version
```

**Kaggle authentication**: Use OAuth tokens (recommended):
- Authenticate via `kaggle config` or ensure `~/.kaggle/access_token` exists
- **No need for kaggle.json** - OAuth tokens are preferred

**Competition rules**: You **must manually accept rules** in your browser once per competition:
1. Visit: `https://www.kaggle.com/competitions/<competition-slug>/rules`
2. Click "I Understand and Accept"
3. This tool will **never automate** this step (security requirement)

---

### Setup

**Clone and install dependencies**:
```bash
git clone <your-repo-url>
cd kaggle-autopilot
uv sync
```

**Add a new dependency**:
```bash
uv add pandas  # Runtime dependency
uv add --dev pytest  # Development dependency
```

**Remove a dependency**:
```bash
uv remove <package-name>
```

---

### Basic Usage

**Get help**:
```bash
uv run kagglebot --help
uv run kagglebot run --help
```

**Dry-run** (preview without submitting):
```bash
uv run kagglebot run https://www.kaggle.com/competitions/titanic --dry-run
```

**Full run with submission**:
```bash
uv run kagglebot run https://www.kaggle.com/competitions/titanic \
  --submit \
  --message "baseline v1" \
  --no-dry-run \
  --force
```

**Competition slug** (alternative to URL):
```bash
uv run kagglebot run titanic --submit --message "baseline v1" --no-dry-run --force
```

---

### Compute Modes

Kagglebot supports **4 compute modes** via the `--compute` flag:

#### 1. Local CPU (default, works everywhere)
```bash
uv run kagglebot run titanic --compute local_cpu --submit --message "CPU baseline"
```
- **Default**: No `--compute` flag needed
- **Use case**: Testing, small datasets, or no GPU available

#### 2. Local GPU (auto-detect CUDA/MPS)
```bash
uv run kagglebot run titanic --compute local_gpu --submit --message "GPU baseline"
```
- **Auto-detection**: Automatically detects NVIDIA CUDA or Apple Silicon MPS
- **Fallback**: Falls back to CPU if GPU not available (unless `--strict-accelerator`)
- **Strict mode**: Fail if GPU not found:
  ```bash
  uv run kagglebot run titanic --compute local_gpu --strict-accelerator
  ```

#### 3. Kaggle GPU kernel (free cloud GPU)
```bash
uv run kagglebot run titanic \
  --compute kaggle_gpu \
  --kaggle-username your-kaggle-username \
  --submit \
  --message "kernel GPU baseline"
```
- **Required**: `--kaggle-username` (or auto-detected from Kaggle config)
- **What happens**: Generates kernel package → pushes to Kaggle → polls until complete → downloads outputs → validates → submits locally
- **Submission**: Always happens **locally** (not from kernel) for safety

#### 4. Kaggle TPU kernel (free cloud TPU)
```bash
uv run kagglebot run titanic \
  --compute kaggle_tpu \
  --kaggle-username your-kaggle-username \
  --enable-internet \
  --submit \
  --message "kernel TPU baseline"
```
- **Internet access**: Use `--enable-internet` only if competition allows external data
- **Warning**: Enabling internet logs a security warning

**Optional flags for Kaggle kernels**:
```bash
--dry-run  # Preview kernel package without pushing
```

---

### Testing and Quality

**Run all tests**:
```bash
uv run pytest -q
```

**Run specific test file**:
```bash
uv run pytest tests/test_validation.py -v
```

**Test coverage**:
```bash
uv run pytest --cov=kagglebot --cov-report=term-missing
```

**Lint code**:
```bash
uv run ruff check .
```

**Auto-format code**:
```bash
uv run ruff format .
```

**Type checking** (optional):
```bash
uv run pyright
```

---

### Troubleshooting

#### Competition rules not accepted
**Symptom**: Exit code 2, message: "Competition rules not accepted"

**Solution**:
1. Visit: `https://www.kaggle.com/competitions/<slug>/rules`
2. Manually click "I Understand and Accept" in browser
3. Retry command

**Why manual?**: Automating rule acceptance violates Kaggle terms of service

---

#### Kaggle authentication errors
**Symptom**: "Kaggle credentials not found" or "401 Unauthorized"

**Solution**:
```bash
# Run Kaggle config to authenticate
kaggle config

# Credentials stored in:
# - OAuth tokens: ~/.kaggle/access_token (recommended)
# - Legacy: ~/.kaggle/kaggle.json (still works)
```

**Note**: This tool uses Kaggle Python API with OAuth tokens (no need for kaggle.json)

---

#### GPU not detected
**Symptom**: Fails with `--strict-accelerator`, or falls back to CPU

**Check GPU availability**:
```python
# CUDA (NVIDIA)
python -c "import torch; print(torch.cuda.is_available())"

# MPS (Apple Silicon)
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Solutions**:
- Use `--compute local_cpu` to run on CPU
- Use `--compute kaggle_gpu` to run on Kaggle's free GPU
- Omit `--strict-accelerator` to allow fallback to CPU

---

#### Where are artifacts stored?

**Directory structure**:
```
kaggle-autopilot/
├── data/                           # Downloaded datasets (gitignored)
│   └── <slug>/
│       ├── raw/                    # Original ZIP files
│       └── extracted/              # Extracted CSVs
├── artifacts/                      # All outputs (gitignored)
│   └── <slug>/
│       ├── runs/
│       │   └── <run_id>/
│       │       ├── plan.json       # Modeling strategy
│       │       ├── summary.json    # CV scores, model info
│       │       ├── submission.csv  # Final submission
│       │       └── kernel/         # Kernel package (if --compute kaggle_*)
│       ├── submissions/
│       │   └── history.jsonl       # Submission ledger
│       └── reports/
│           └── analysis.json       # Competition analysis
```

**View run history**:
```bash
cat artifacts/titanic/submissions/history.jsonl
```

## Safety Features

- **Dry-run by default**: Use `--no-dry-run --force` to allow network actions
- **Duplicate detection**: Prevents recording identical submissions by hash
- **Strict validation**: Validates submission format against sample_submission.csv
- **No automated rule acceptance**: Users must manually accept rules in browser
- **Run ledger**: Records runs in `artifacts/<slug>/runs/<run_id>/metadata.json`
- **Submission ledger**: Records submissions in `artifacts/<slug>/submissions/history.jsonl`
- **Reports**: Analysis and training reports in `artifacts/<slug>/reports/`
- **Notebook runs**: Kaggle kernel artifacts stored in `artifacts/<slug>/<run_id>/`

## Project Structure

```
kaggle-autopilot/
├── src/kagglebot/
│   ├── cli.py              # CLI entry point (Typer)
│   ├── compute.py          # Compute mapping & GPU detection
│   ├── runners/            # Execution backends
│   │   ├── base.py         # Runner interface (ABC)
│   │   ├── local.py        # LocalRunner (CPU/GPU)
│   │   └── kaggle_notebook.py  # KaggleNotebookRunner
│   ├── analyzer/           # Competition analysis
│   ├── training/           # Model training (tabular, text, image)
│   ├── validation.py       # Submission validation
│   ├── history.py          # Submission ledger
│   ├── kaggle_cli.py       # Kaggle API wrapper
│   └── paths.py            # Path management
├── data/                   # Downloaded datasets (gitignored)
├── artifacts/              # All outputs (gitignored)
├── tests/                  # Test suite (pytest)
└── config/                 # Config templates
```

## Development

See **Testing and Quality** section above for common commands.

**Additional development setup**:
```bash
# Sync dependencies after pulling changes
uv sync

# Install pre-commit hooks (optional)
uv run pre-commit install
```

## Current Status

**Implemented**:
- ✅ Data download and validation
- ✅ Competition analysis (tabular)
- ✅ Tabular training (Ridge/LogReg/HistGB/CatBoost)
- ✅ Submission validation and ledger
- ✅ Compute switching (local CPU/GPU + Kaggle GPU/TPU)

**In Progress** (see PLAN_COMPUTE.md):
- 🚧 Production models (LightGBM, XGBoost)

**Roadmap**:
- Competition types: Text, image, timeseries
- Advanced strategies: Stacking, ensembles, feature engineering
- Auto-tuning: Hyperparameter optimization

**Limitations (MVP)**:
- Only tabular competitions (train.csv, test.csv, sample_submission.csv)
- Single-target competitions only (multi-target planned)
- Basic models only (production models in progress)

## Documentation

- **SPEC_COMPUTE.md**: CLI flags, exit codes, artifact layout
- **ARCHITECTURE_COMPUTE.md**: Module design, runner interface
- **PLAN_COMPUTE.md**: 7-week implementation roadmap
- **CLAUDE.md**: Development guidelines for contributors
- **AGENTS.md**: Instructions for implementer agents

## License

MIT
