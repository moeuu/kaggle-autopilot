# Specification: Compute Switching

## Overview

This spec defines how kagglebot supports multiple compute backends (local CPU/GPU and Kaggle cloud GPU/TPU) through a single `--compute` flag.

**Design Philosophy**: Simple user interface, complex implementation hidden behind the scenes.

---

## CLI Interface

### Primary Command

```bash
kagglebot run <competition> [OPTIONS]
```

### Compute Flag (NEW)

```bash
--compute {local_cpu,local_gpu,kaggle_gpu,kaggle_tpu}
    Compute backend for training (default: local_cpu)

    local_cpu:    Train on local machine using CPU
    local_gpu:    Train on local machine using available GPU (CUDA/MPS)
    kaggle_gpu:   Train on Kaggle Notebooks with GPU
    kaggle_tpu:   Train on Kaggle Notebooks with TPU
```

### Additional Flags

```bash
--strict-accelerator
    Fail if requested accelerator not available (default: false)
    Example: --compute local_gpu --strict-accelerator
    → fails if no GPU found (instead of CPU fallback)

--kaggle-username TEXT
    Kaggle username for kernel ownership
    Optional override; auto-detected from KAGGLE_USERNAME or ~/.kaggle/kaggle.json

--enable-internet
    Allow internet in Kaggle Notebooks (default: false)
    SECURITY: Only enable if competition explicitly allows external data

--max-kernel-runtime MINUTES
    Max execution time for Kaggle kernels (default: 120, max: 540)

--kernel-slug TEXT
    Custom kernel slug (default: auto-generated kb-<slug>-<run_id>)
```

### Existing Flags (Keep)

```bash
--dry-run
    Show plan without execution (default: false)
    Safe: no downloads, no kernel push, no submission

--submit
    Submit to Kaggle after validation (default: false)

--message TEXT
    Submission message (default: auto-generated)

--force-submit
    Bypass deduplication and rate limits (use with caution)

--time-budget-min MINUTES
    Time budget for local training (default: 60)
    Note: Ignored for kaggle_* compute (use --max-kernel-runtime)
```

---

## Usage Examples

### Local CPU (Default)
```bash
# Simple: train locally on CPU
kagglebot run titanic

# With submission
kagglebot run titanic --submit --message "initial model v1"

# Dry-run first
kagglebot run titanic --dry-run
```

### Local GPU
```bash
# Use local GPU (auto-detect CUDA/MPS)
kagglebot run titanic --compute local_gpu --submit

# Strict mode: fail if no GPU
kagglebot run titanic --compute local_gpu --strict-accelerator

# Dry-run shows GPU detection
kagglebot run titanic --compute local_gpu --dry-run
```

### Kaggle GPU
```bash
# Train on Kaggle GPU, submit locally
kagglebot run titanic --compute kaggle_gpu --submit

# With custom runtime limit
kagglebot run titanic \
  --compute kaggle_gpu \
  --max-kernel-runtime 240 \
  --submit

# Dry-run shows kernel metadata
kagglebot run titanic --compute kaggle_gpu --dry-run
```

### Kaggle TPU
```bash
# Train on Kaggle TPU
kagglebot run titanic --compute kaggle_tpu --submit

# With internet (if competition allows external data)
kagglebot run nlp-competition \
  --compute kaggle_gpu \
  --enable-internet \
  --submit
```

---

## Compute Mapping

Internal mapping from `--compute` to runner configuration:

| --compute   | Runner           | Accelerator | Notes |
|-------------|------------------|-------------|-------|
| local_cpu   | LocalRunner      | cpu         | Default, always works |
| local_gpu   | LocalRunner      | gpu         | Auto-detect CUDA/MPS, fallback to CPU |
| kaggle_gpu  | KaggleNotebookRunner | gpu     | Requires Kaggle credentials |
| kaggle_tpu  | KaggleNotebookRunner | tpu     | Requires Kaggle credentials |

### ComputePlan

Internally represented as:

```python
@dataclass
class ComputePlan:
    """Derived from --compute flag."""
    compute: Literal["local_cpu", "local_gpu", "kaggle_gpu", "kaggle_tpu"]
    runner: Literal["local", "kaggle_notebook"]
    accelerator: Literal["cpu", "gpu", "tpu"]
    strict: bool  # From --strict-accelerator

# Mapping
COMPUTE_PLANS = {
    "local_cpu": ComputePlan("local_cpu", "local", "cpu", False),
    "local_gpu": ComputePlan("local_gpu", "local", "gpu", False),
    "kaggle_gpu": ComputePlan("kaggle_gpu", "kaggle_notebook", "gpu", False),
    "kaggle_tpu": ComputePlan("kaggle_tpu", "kaggle_notebook", "tpu", False),
}
```

---

## Exit Codes

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | Pipeline completed successfully |
| 1 | General failure | Unhandled exception |
| 2 | **Rules not accepted** | User must join competition in browser |
| 3 | Invalid competition URL/slug | Malformed input |
| 4 | Data download failed | Network error, credentials invalid |
| 5 | Training failed | Model training error |
| 6 | Submission validation failed | Format mismatch |
| 7 | Submission upload failed | Kaggle API error |
| 8 | Duplicate submission | Already submitted (use --force-submit) |
| 9 | Rate limit exceeded | Too many recent submissions |
| 10 | **GPU not available** | local_gpu requested but no GPU found (with --strict-accelerator) |
| 11 | **Kernel timeout** | Kaggle kernel exceeded max runtime |
| 12 | **Kernel failed** | Kaggle kernel error or cancelled |

---

## Artifact Layout

```
artifacts/<slug>/<run_id>/
├── plan.json                    # ComputePlan + ModelingStrategy
├── config_snapshot.json         # Full config used for run
├── logs/
│   ├── main.log                 # Main pipeline log
│   ├── training.log             # Training details
│   └── kernel.log               # Kaggle kernel logs (if applicable)
├── models/                      # For local runs
│   ├── preprocessor.pkl
│   ├── lgbm_fold_0.pkl
│   ├── catboost_fold_0.pkl
│   └── final.pkl
├── kernel/                      # For kaggle_* runs
│   ├── kernel-metadata.json
│   ├── main.py
│   └── plan.json
├── output/                      # Downloaded from Kaggle
│   ├── submission.csv
│   ├── metrics.json
│   └── *.log
├── submission.csv               # Final validated submission
├── submission_hash.txt          # SHA256 for deduplication
└── summary.json                 # Run summary
```

### plan.json

```json
{
  "run_id": "7f8e9d2a",
  "slug": "titanic",
  "compute": {
    "requested": "kaggle_gpu",
    "runner": "kaggle_notebook",
    "accelerator": "gpu",
    "detected_gpu": "Tesla T4",
    "strict": false
  },
  "strategy": {
    "type": "tabular",
    "task": "classification",
    "models": ["lgbm", "catboost"],
    "cv_folds": 5,
    "preprocessing": ["impute_median", "onehot"]
  },
  "execution": {
    "started_at": "2026-01-01T12:00:00Z",
    "completed_at": "2026-01-01T12:45:00Z",
    "duration_seconds": 2700
  }
}
```

### summary.json

```json
{
  "run_id": "7f8e9d2a",
  "slug": "titanic",
  "compute": "kaggle_gpu",
  "success": true,
  "cv_score": 0.854,
  "submission_hash": "a1b2c3d4e5f6...",
  "submitted": true,
  "kaggle_submission_id": "12345678",
  "kaggle_score": 0.77511,
  "kernel_id": "moritaeiji/kb-titanic-7f8e9d2a",
  "kernel_url": "https://www.kaggle.com/code/moritaeiji/kb-titanic-7f8e9d2a"
}
```

### Submission Ledger

```jsonl
{"timestamp": "2026-01-01T12:45:00Z", "run_id": "7f8e9d2a", "compute": "kaggle_gpu", "kernel_id": "moritaeiji/kb-titanic-7f8e9d2a", "hash": "a1b2c3d4", "kaggle_id": "12345678", "score": 0.77511, "message": "GPU initial model"}
{"timestamp": "2026-01-01T14:20:00Z", "run_id": "8g9f0e1b", "compute": "local_cpu", "kernel_id": null, "hash": "b2c3d4e5", "kaggle_id": "12345679", "score": 0.76076, "message": "CPU initial model"}
```

---

## Dry-Run Behavior

When `--dry-run` is set:

```bash
kagglebot run titanic --compute kaggle_gpu --submit --dry-run
```

Output:
```
[DRY RUN] Execution plan:

Competition: titanic
Compute: kaggle_gpu (runner=kaggle_notebook, accelerator=gpu)

Steps:
  1. Check rules acceptance (no API call in dry-run)
  2. Download data to data/titanic/raw/ (SKIPPED)
  3. Analyze competition → tabular binary classification
  4. Generate modeling strategy → lgbm + catboost, 5-fold CV
  5. Generate kernel package:
     - kernel-metadata.json (preview below)
     - main.py (~150 lines, template: tabular_script.py.j2)
     - plan.json
  6. Push kernel to Kaggle (SKIPPED)
  7. Poll kernel status (SKIPPED)
  8. Download outputs (SKIPPED)
  9. Validate submission.csv
  10. Submit to Kaggle (WOULD SUBMIT: --submit flag set)

Kernel Metadata Preview:
{
  "id": "moritaeiji/kb-titanic-7f8e9d2a",
  "title": "kagglebot: titanic (7f8e9d2a)",
  "code_file": "main.py",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "competition_sources": ["titanic"]
}

Generated files (dry-run, not saved):
  artifacts/titanic/7f8e9d2a/kernel/kernel-metadata.json
  artifacts/titanic/7f8e9d2a/kernel/main.py
  artifacts/titanic/7f8e9d2a/kernel/plan.json

No side effects performed (dry-run mode).
Re-run without --dry-run to execute.
```

For local_gpu:
```bash
kagglebot run titanic --compute local_gpu --dry-run
```

Output:
```
[DRY RUN] Execution plan:

Competition: titanic
Compute: local_gpu (runner=local, accelerator=gpu)

GPU Detection:
  CUDA available: Yes
  CUDA version: 12.1
  GPU: NVIDIA RTX 3090 (24GB)
  MPS available: No

Models selected:
  - LightGBM (GPU mode: cuda)
  - CatBoost (GPU mode: GPU)

Steps:
  1. Check rules acceptance
  2. Download data (SKIPPED)
  3. Analyze competition
  4. Train models on local GPU
  5. Generate predictions
  6. Validate submission
  7. Submit (WOULD SUBMIT: --submit flag set)

No side effects performed (dry-run mode).
```

---

## Error Messages

### Exit Code 2: Rules Not Accepted

```
❌ Competition rules not accepted

You must manually join this competition:
  https://www.kaggle.com/competitions/titanic/rules

Steps:
  1. Visit the URL above
  2. Click "I Understand and Accept"
  3. Re-run: kagglebot run titanic --compute kaggle_gpu --submit

This is required once per competition.
```

### Exit Code 10: GPU Not Available (Strict Mode)

```
❌ GPU not available

Requested: --compute local_gpu --strict-accelerator
Detected: No CUDA or MPS GPU found

Available options:
  1. Install CUDA drivers (NVIDIA GPU)
  2. Use MPS (Apple Silicon Mac)
  3. Remove --strict-accelerator (fallback to CPU with warning)
  4. Use --compute local_cpu explicitly

GPU detection details:
  torch.cuda.is_available(): False
  torch.backends.mps.is_available(): False
  nvidia-smi: command not found
```

### Exit Code 11: Kernel Timeout

```
❌ Kernel timeout (120 minutes exceeded)

Kernel is still running on Kaggle:
  https://www.kaggle.com/code/moritaeiji/kb-titanic-7f8e9d2a

Options:
  1. Wait for completion and check manually
  2. Increase timeout: --max-kernel-runtime 240
  3. Cancel kernel via Kaggle web UI
  4. Check kernel logs for issues

Partial outputs may be available in:
  artifacts/titanic/7f8e9d2a/output/
```

### Exit Code 8: Duplicate Submission

```
❌ Duplicate submission detected

This exact submission was already made:
  Hash:      a1b2c3d4e5f6
  Submitted: 2026-01-01 12:45:00
  Score:     0.77511
  Run ID:    7f8e9d2a

Ledger: artifacts/titanic/submissions/history.jsonl

To submit anyway: --force-submit (not recommended)
```

---

## Configuration

### Config Schema (config/default.toml)

```toml
[compute]
default = "local_cpu"  # Default compute backend

[compute.local]
time_budget_minutes = 60
detect_gpu = true  # Auto-detect GPU availability
fallback_to_cpu = true  # Fallback if GPU unavailable (when not --strict-accelerator)
n_jobs = -1  # Parallel jobs (-1 = all cores)

[compute.local.gpu]
# GPU detection order: CUDA first, then MPS
prefer_cuda = true
cuda_device_id = 0  # Which GPU to use if multiple
# Model-specific GPU settings
lightgbm_device = "cuda"  # or "gpu"
catboost_task_type = "GPU"
xgboost_tree_method = "gpu_hist"

[compute.kaggle_notebook]
max_kernel_runtime_minutes = 120
max_kernel_runtime_gpu_minutes = 540  # Kaggle limit: 9 hours
max_kernel_runtime_tpu_minutes = 180
poll_interval_seconds = 30
enable_internet_default = false
kernel_is_private = true
kernel_slug_include_run_id = true

[submission]
# Safety guardrails
check_duplicates = true
hash_algorithm = "sha256"
max_submissions_per_day = 5
min_hours_between_submissions = 1.0
strict_validation = true

[logging]
level = "INFO"
format = "json"
log_to_file = true
log_to_console = true
```

---

## Precedence Rules

Configuration hierarchy (highest to lowest priority):

1. **CLI flags**: `--compute kaggle_gpu`
2. **Competition config**: `config/titanic.toml`
3. **Global config**: `config/default.toml`
4. **Hardcoded defaults**: In code

Example:
```bash
# CLI overrides config
kagglebot run titanic --compute kaggle_gpu
# Uses kaggle_gpu even if config says local_cpu

# Config overrides defaults
# In config/titanic.toml:
[compute]
default = "local_gpu"
# Now: kagglebot run titanic uses local_gpu
```

---

## Validation Rules

### Submission Validation (Strict)

Before any submission:

1. **Column check**: Columns match sample_submission.csv exactly (names + order)
2. **Row count**: Number of rows matches sample_submission.csv exactly
3. **ID alignment**: If ID column exists, all IDs present and aligned
4. **Value ranges**:
   - Probabilities in [0, 1]
   - Integers are actual integers (no floats)
   - No NaN or Inf values
5. **Format**: CSV format valid (parseable by pandas)

### Deduplication

```python
def compute_submission_hash(file_path: Path) -> str:
    """SHA256 hash of submission CSV content (excluding timestamps)."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def is_duplicate(hash: str, ledger: Path) -> bool:
    """Check if submission hash already in ledger."""
    if not ledger.exists():
        return False

    with open(ledger) as f:
        for line in f:
            entry = json.loads(line)
            if entry["hash"] == hash:
                return True
    return False
```

### Rate Limiting

```python
def check_rate_limit(ledger: Path, config: Config) -> tuple[bool, str]:
    """
    Check if submission allowed based on rate limits.

    Returns:
        (allowed: bool, reason: str)
    """
    recent = get_submissions_last_24h(ledger)

    # Daily limit
    if len(recent) >= config.max_submissions_per_day:
        return False, f"Daily limit: {config.max_submissions_per_day}"

    # Time since last
    if recent:
        last = recent[-1]
        elapsed_hours = (datetime.now() - last.timestamp).total_seconds() / 3600
        if elapsed_hours < config.min_hours_between_submissions:
            wait = int((config.min_hours_between_submissions - elapsed_hours) * 60)
            return False, f"Wait {wait} minutes"

    return True, ""
```

---

## Security Checklist

Before every commit involving compute switching:

- [ ] No Kaggle API keys in kernel code
- [ ] No credentials in kernel-metadata.json
- [ ] enable_internet defaults to false
- [ ] Dry-run performs no side effects
- [ ] Rules acceptance check works (no automation)
- [ ] Kernel slug includes run_id (no overwrites)
- [ ] Timeout enforced on kernel polling
- [ ] All subprocess calls use list args (no shell=True)
- [ ] Local GPU detection doesn't crash if no GPU
- [ ] Fallback to CPU logged clearly
- [ ] Submission validation before any upload
- [ ] Deduplication and rate limiting enforced

---

## Backward Compatibility

### Migration Path

Existing users (no `--compute` flag):
- **Default behavior**: `--compute local_cpu` (same as before)
- **No breaking changes**: All existing commands work
- **Opt-in**: New compute modes require explicit `--compute` flag

### Deprecated (None)

No flags or features are being deprecated.

---

## Future Enhancements

After MVP:

1. **Auto-detection**: `--compute auto` (detect based on availability)
2. **Hybrid**: Train locally, then push to Kaggle for final run
3. **Distributed**: Multi-GPU local training
4. **Cloud providers**: AWS/GCP/Azure support
5. **Cost tracking**: Estimate and track compute costs
6. **Kernel caching**: Reuse kernels if code unchanged

---

## Success Criteria

After implementation:

```bash
# Local CPU (existing functionality)
✓ kagglebot run titanic --submit

# Local GPU (new)
✓ kagglebot run titanic --compute local_gpu --submit
✓ GPU detected and used
✓ Fallback to CPU if no GPU (without --strict-accelerator)
✓ Clear error if no GPU (with --strict-accelerator)

# Kaggle GPU (new)
✓ kagglebot run titanic --compute kaggle_gpu --submit
✓ Kernel pushed and executed
✓ Outputs downloaded
✓ Submission validated locally
✓ Submitted from local machine

# Kaggle TPU (new)
✓ kagglebot run titanic --compute kaggle_tpu --submit
✓ TPU kernel works

# Safety
✓ Dry-run shows plan, no side effects
✓ Rules check works (exit 2 if not accepted)
✓ No secrets in kernel
✓ Deduplication works
✓ Rate limiting works
✓ All error messages clear and actionable
```
