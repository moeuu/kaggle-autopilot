# Kagglebot Architecture (Final Review)

**Purpose**: Comprehensive architecture overview for kagglebot CLI automation tool

**Last Updated**: 2026-01-02

**Status**: Production-ready with autopilot feature

**Note**: Git integration has been removed from the implementation; any references to git operations are historical.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Data Flow](#data-flow)
4. [Core Components](#core-components)
5. [Safety Gates](#safety-gates)
6. [Kaggle Notebook Runner](#kaggle-notebook-runner)
7. [Autopilot Mode](#autopilot-mode)
8. [Security Model](#security-model)
9. [Failure Modes](#failure-modes)
10. [Future Enhancements](#future-enhancements)

---

## Overview

Kagglebot is a Python CLI tool (using Typer + uv) that automates Kaggle competition workflows with safety-first design:

**Core Workflows:**
1. **Bootstrap**: Download data, create artifacts, generate prompts
2. **Implement**: Run AI agents (Codex/Claude) on clean git worktree
3. **Train**: Local (CPU/GPU) or Kaggle notebook (GPU/TPU) training
4. **Submit**: Validate + deduplicate + submit to Kaggle
5. **Run**: Orchestrated end-to-end (bootstrap → implement → train → submit)
6. **Autopilot**: Iterative offline improvement with Top1 heuristic gating

**Key Principles:**
- **Safe by default**: No submissions without explicit flags
- **No secrets**: Never log or commit credentials
- **No automation of rule acceptance**: User must manually accept in browser
- **Validation first**: Schema + row count + ID alignment checks
- **Deduplication**: SHA256-based submission ledger

---

## Directory Structure

```
kaggle-autopilot/
├── src/kagglebot/           # Main package
│   ├── agents/              # Agent runners (codex, claude)
│   ├── analyzer/            # Data profiling
│   ├── runners/             # Training runners (local, kaggle)
│   ├── solver/              # ML initial model implementations
│   ├── training/            # Training utilities
│   ├── autopilot.py         # Autopilot core logic
│   ├── autopilot_runner.py  # Autopilot main loop
│   ├── bootstrap.py         # Bootstrap logic
│   ├── cli.py               # Typer CLI commands
│   ├── compute.py           # Compute mode selection
│   ├── exceptions.py        # Custom exceptions
│   ├── hashing.py           # SHA256 hashing
│   ├── history.py           # Run/submission ledgers
│   ├── kaggle_api.py        # Kaggle Python API wrapper
│   ├── kaggle_cli.py        # Kaggle CLI wrapper (subprocess)
│   ├── kernel_runner.py     # Kaggle notebook execution
│   ├── paths.py             # Path management
│   ├── types.py             # Type definitions
│   ├── validation.py        # Submission validation
│   └── validators.py        # Input validators
│
├── tests/                   # pytest tests (72 tests)
│   ├── test_autopilot.py    # Autopilot tests (28 tests)
│   ├── test_*.py            # Component tests
│
├── artifacts/<slug>/        # Per-competition artifacts
│   ├── context/             # meta.json, plan.json, rules.txt, top1_public.json
│   ├── data/                # Downloaded CSV/ZIP files
│   ├── prompts/             # codex.md, claude.md
│   ├── models/              # Saved models
│   ├── runs/                # Run-specific outputs
│   │   └── <run_id>/
│   │       ├── iter-<N>/    # Autopilot iterations
│   │       │   ├── metrics.json
│   │       │   ├── diagnostics.md
│   │       │   └── submission.csv
│   │       ├── kernel/      # Kaggle kernel package
│   │       ├── output/      # Kernel outputs
│   │       └── codex/       # Agent transcripts
│   └── submissions/
│       ├── history.jsonl    # Submission ledger (dedupe)
│       └── <run_id>_submission.csv
│
├── docs/                         # Design documentation
│   ├── README.md                 # Documentation index
│   ├── spec_autopilot.md         # Autopilot spec
│   ├── architecture.md           # Autopilot control flow
│   ├── architecture_final.md     # Full architecture reference
│   ├── guardrails_checklist.md   # Safety checklist
│   ├── autopilot.md              # Autopilot walkthrough
│   ├── AUTOPILOT_SINGLE_SUBMIT.md
│   ├── AUTOPILOT_SUMMARY.md
│   ├── safety/
│   │   ├── submission_checklist.md
│   │   └── failure_modes.md
│   ├── compute/
│   │   ├── architecture.md
│   │   ├── spec.md
│   │   ├── plan.md
│   │   └── tasks.md
│   ├── notebook_runner/
│   │   ├── design.md
│   │   └── tasks.md
│   └── agents/
│       ├── codex_prompt.md
│       └── codex_implementation_plan.md
│
├── AGENTS.md                     # Agent instructions (Codex)
├── CLAUDE.md                     # Claude Code working rules
├── README.md                     # User documentation
├── SECURITY.md                   # Security guidelines
└── pyproject.toml                # uv project config
```

---

## Data Flow

### 1. Bootstrap Flow

```
User: uv run kagglebot bootstrap titanic --download --force

1. Parse competition slug (URL or slug)
2. Create artifacts/<slug>/ directory structure
3. Generate meta.json (competition metadata)
4. Capture rules (URL, fetch, or file)
5. [Optional] Download data via Kaggle CLI
   └─> Extract ZIPs to data/ directory
6. Generate prompts (codex.md, claude.md) from templates
7. Output: artifacts/<slug>/context/meta.json
```

**Artifacts Created:**
- `artifacts/<slug>/context/meta.json`: Competition metadata
- `artifacts/<slug>/context/rules.txt`: Competition rules
- `artifacts/<slug>/data/`: Downloaded CSV files
- `artifacts/<slug>/prompts/codex.md`: Agent prompt

**Safety Gates:**
- Requires `--force` for Kaggle CLI download
- Never automates rules acceptance
- Prints rules URL if not accepted

### 2. Implement Flow (Agent Execution)

```
User: uv run kagglebot implement titanic --agent codex --commit

1. Bootstrap (if not already done)
2. Create run record in ledger
3. Create clean git branch (bot/<slug>/<run_id>)
4. Run agent (codex exec or claude print mode)
   └─> Agent reads prompt, writes code to src/
5. Run verification command (default: uv run pytest -q)
6. [Optional] Commit changes if --commit
7. Output: Code changes, agent transcript
```

**Artifacts Created:**
- `artifacts/<slug>/runs/<run_id>/codex/`: Agent transcript
- Git branch: `bot/<slug>/<run_id>`
- Code changes in src/ (if agent successful)

**Safety Gates:**
- Clean git worktree required
- Branch must not exist
- Verification must pass before commit

### 3. Train Flow

```
User: uv run kagglebot train titanic --compute local_gpu --force

1. Bootstrap (if not already done)
2. Create run record
3. Select runner based on --compute:
   - local_cpu/local_gpu: LocalRunner
   - kaggle_gpu/kaggle_tpu: KaggleNotebookRunner
4. Run training:
   LocalRunner:
     └─> train_and_predict() locally
   KaggleNotebookRunner:
     └─> Generate kernel package
     └─> Push to Kaggle via CLI
     └─> Poll for completion
     └─> Download outputs
5. Validate submission.csv
6. Output: submission.csv in artifacts/<slug>/submissions/
```

**Artifacts Created:**
- `artifacts/<slug>/runs/<run_id>/metrics.json`: Training metrics
- `artifacts/<slug>/submissions/<run_id>_submission.csv`: Submission file
- [Kaggle] `artifacts/<slug>/runs/<run_id>/kernel/`: Kernel package
- [Kaggle] `artifacts/<slug>/runs/<run_id>/output/`: Kernel outputs

**Safety Gates:**
- Requires `--force` for Kaggle kernel execution
- GPU detection (local_gpu with --strict-accelerator)
- Kernel timeout enforcement (default: 120 min)
- Submission validation (schema + row count)

### 4. Submit Flow

```
User: uv run kagglebot submit titanic -f sub.csv -m "initial model v1" --force

1. Load submission file
2. Validate against sample_submission.csv:
   └─> Column names match
   └─> Row count matches
   └─> ID alignment (if id column exists)
3. Check rate limit (submissions/hour)
4. Check duplicate (SHA256 hash in ledger)
5. Check rules accepted (Kaggle API)
6. Submit via Kaggle API
7. Record in ledger (history.jsonl)
```

**Artifacts Created:**
- `artifacts/<slug>/submissions/history.jsonl`: Ledger entry

**Safety Gates:**
- Requires `--force` flag
- Requires `--message` flag
- Validation before submission
- Rate limiting (default: 1 submission/10 minutes)
- Deduplication (SHA256 hash)
- Rules acceptance check

### 5. Autopilot Flow

```
User: uv run kagglebot autopilot titanic --agent codex --force

1. Bootstrap
2. Fetch Kaggle Top1 public score (via leaderboard CSV)
3. For iteration 1..max_iterations (default: 5):
   a. Train model (local or Kaggle)
   b. Evaluate offline (holdout or CV)
   c. Compare offline to Top1 using heuristic:
      - Maximize: offline >= top1 * (1 - margin_rel) - margin_abs
      - Minimize: offline <= top1 * (1 + margin_rel) + margin_abs
   d. If heuristic met and --submit-on-heuristic:
      └─> Submit and stop
   e. Else continue to next iteration
4. After max iterations:
   └─> Select best iteration (by offline score)
   └─> Submit if --submit-at-final (default: true)
```

**Artifacts Created:**
- `artifacts/<slug>/context/top1_public.json`: Cached Top1 score
- `artifacts/<slug>/runs/<run_id>/iter-<N>/metrics.json`: Per-iteration metrics
- `artifacts/<slug>/runs/<run_id>/iter-<N>/diagnostics.md`: Diagnostics
- `artifacts/<slug>/submissions/<run_id>_iter<N>_submission.csv`: Iteration submissions

**Safety Gates:**
- MAX_SUBMISSIONS = 1 (hard-coded)
- Heuristic-based submission gating
- Full validation + deduplication
- 60-minute Top1 cache (avoid excessive API calls)

---

## Core Components

### CLI Commands (src/kagglebot/cli.py)

| Command | Purpose | Safety Level |
|---------|---------|--------------|
| `bootstrap` | Setup artifacts, download data | Medium (--force for download) |
| `implement` | Run AI agent on clean branch | High (git required) |
| `train` | Train model locally or on Kaggle | High (--force for Kaggle) |
| `submit` | Submit to Kaggle with validation | Very High (--force + --message) |
| `run` | Orchestrated end-to-end | Very High (combines above) |
| `autopilot` | Iterative offline improvement | Very High (heuristic gating) |

### Compute Modes (src/kagglebot/compute.py)

```python
class Compute(str, Enum):
    local_cpu = "local_cpu"      # Train locally on CPU
    local_gpu = "local_gpu"      # Train locally on GPU (CUDA/MPS)
    kaggle_gpu = "kaggle_gpu"    # Train in Kaggle notebook with GPU
    kaggle_tpu = "kaggle_tpu"    # Train in Kaggle notebook with TPU
```

**GPU Detection:**
- CUDA: `torch.cuda.is_available()`
- MPS (Apple Silicon): `torch.backends.mps.is_available()`
- Fallback to CPU if GPU unavailable (unless --strict-accelerator)

### Runners (src/kagglebot/runners/)

**LocalRunner:**
- Trains on local machine (CPU/GPU)
- Uses existing train_and_predict() function
- Outputs to local artifacts/

**KaggleNotebookRunner:**
- Generates kernel package (kernel-metadata.json + script.py)
- Pushes to Kaggle via `kaggle kernels push`
- Polls for completion (default: 120 min timeout)
- Downloads outputs via `kaggle kernels output`
- Outputs to artifacts/<slug>/runs/<run_id>/output/

### Validation (src/kagglebot/validation.py)

**Submission Validation:**
```python
def validate_submission_file(sample_path, submission_path):
    # 1. Column names match exactly
    # 2. Row count matches
    # 3. ID alignment (if id column exists)
    # Raises ValidationError if any check fails
```

**Deduplication:**
```python
def ensure_not_duplicate_submission(ledger, submission_path):
    # Compute SHA256 hash
    # Check if hash exists in ledger
    # Raises DuplicateSubmissionError if duplicate
```

**Rate Limiting:**
```python
def ensure_submission_rate_limit(ledger, window_minutes=10):
    # Check last submission time
    # Raises SubmissionRateLimitError if too frequent
```

### History Ledgers (src/kagglebot/history.py)

**RunLedger:**
- Tracks all runs (bootstrap, train, autopilot)
- JSONL format: `artifacts/<slug>/runs/ledger.jsonl`
- Fields: run_id, timestamp, command, argv, extra

**SubmissionLedger:**
- Tracks all submissions
- JSONL format: `artifacts/<slug>/submissions/history.jsonl`
- Fields: timestamp, submission_path, sha256, message, run_id, slug, autopilot_run_id

---

## Safety Gates

### 1. Rules Acceptance (Manual Only)

**Check:**
```python
if not check_rules_accepted(slug):
    print(f"Visit: https://www.kaggle.com/competitions/{slug}/rules")
    raise RulesNotAcceptedError()
```

**Why Manual:**
- Kaggle ToS requires user consent
- Automated acceptance violates ethical guidelines
- Prevents accidental joining of competitions with restrictive rules

### 2. Secret Scanning

**Patterns:**
```python
SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",  # OpenAI API keys
    r"kaggle_key_[a-zA-Z0-9]{16,}",
    r"[0-9a-f]{64}",  # SHA256 hashes (potential tokens)
    r"-----BEGIN [A-Z ]+ KEY-----",  # PEM keys
]
```

**Applied:**
- Pre-submission scan
- Kernel package generation
- Git commit hooks (future)

### 3. Submission Validation

**Three-Stage Validation:**

1. **Schema Validation**:
   - Column names match sample_submission.csv (case-sensitive, order-sensitive)
   - Example: `['id', 'target']` vs `['id', 'prediction']` → ERROR

2. **Row Count Validation**:
   - Submission row count == test.csv row count
   - Example: 28000 expected, 27999 actual → ERROR

3. **ID Alignment** (if id column exists):
   - All test IDs present in submission
   - No duplicate IDs
   - IDs match exactly

### 4. Deduplication

**SHA256-Based Ledger:**
```jsonl
{"timestamp": "2026-01-02T12:00:00", "sha256": "abc123...", "message": "initial model v1", "run_id": "20260102_120000", "slug": "titanic"}
```

**Bypass Options:**
- `--force-duplicate`: Allow duplicate within same autopilot run
- Context: Autopilot iterations may produce identical CSVs (acceptable)

### 5. Rate Limiting

**Default Policy:**
- Max 1 submission per 10 minutes
- Prevents accidental submission spam
- Configurable via ledger logic

**Autopilot Special Case:**
- MAX_SUBMISSIONS = 1 (hard-coded)
- Single submission per autopilot run
- Early submit OR final submit (not both)

### 6. Dry-Run Mode

**All Commands Support `--dry-run`:**
- Bootstrap: Skip Kaggle CLI download
- Train: Skip actual training (mock outputs)
- Submit: Skip Kaggle API call
- Autopilot: Skip external commands

**Purpose:**
- Test workflow without side effects
- Preview generated artifacts
- Debug submission validation

---

## Kaggle Notebook Runner

### End-to-End Flow

```
1. Generate Kernel Package
   ├─> kernel-metadata.json (accelerator, competition_sources, etc.)
   └─> script.py (training script with Kaggle paths)

2. Validate Package
   ├─> Secret scanning
   ├─> Metadata schema validation
   └─> Accelerator conflict check (gpu XOR tpu)

3. Push to Kaggle
   └─> kaggle kernels push -p artifacts/<slug>/runs/<run_id>/kernel/

4. Poll for Completion
   ├─> kaggle kernels status <kernel_id>
   ├─> Backoff: 10s → 30s → 60s
   └─> Timeout: 120 minutes (configurable)

5. Download Outputs
   └─> kaggle kernels output <kernel_id> -p artifacts/<slug>/runs/<run_id>/output/

6. Extract Submission
   └─> Find submission.csv in outputs/
   └─> Copy to artifacts/<slug>/submissions/<run_id>_submission.csv

7. Validate Locally
   └─> validate_submission_file(sample_path, submission_path)

8. Submit from Local Machine
   └─> submit_competition(slug, submission_path, message)
```

### Kernel Metadata Schema

```json
{
  "id": "username/kernel-slug-runid",
  "title": "kagglebot-titanic-20260102_120000",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": ["titanic"],
  "kernel_sources": []
}
```

**Critical Fields:**
- `competition_sources`: Must NOT include "c/" prefix (use slug only)
- `enable_gpu` / `enable_tpu`: Only one can be true
- `enable_internet`: Default false (--enable-internet to override)
- `is_private`: Always true (prevent public kernels)

### Kernel Paths

**Kaggle Environment:**
```
/kaggle/input/<slug>/train.csv          # Competition data
/kaggle/input/<slug>/test.csv
/kaggle/input/<slug>/sample_submission.csv
/kaggle/working/                         # Output directory
/kaggle/working/submission.csv           # Must write here
```

**Template Approach:**
```python
# Bad (local paths)
train = pd.read_csv("data/train.csv")  # ❌ Doesn't exist in kernel

# Good (Kaggle paths)
train = pd.read_csv("/kaggle/input/{slug}/train.csv")  # ✅ Works in kernel
submission.to_csv("/kaggle/working/submission.csv", index=False)  # ✅ Correct output
```

### Polling Strategy

**Backoff Schedule:**
```python
attempts = 0
while True:
    status = check_kernel_status(kernel_id)
    if status in ["complete", "error", "cancelled"]:
        break
    if elapsed > timeout:
        raise KernelTimeoutError()

    # Exponential backoff
    if attempts < 5:
        sleep(10)  # 10 seconds
    elif attempts < 10:
        sleep(30)  # 30 seconds
    else:
        sleep(60)  # 60 seconds

    attempts += 1
```

**Timeout Limits:**
- Tool default: 120 minutes
- Kaggle GPU limit: 120 minutes (hard limit)
- Kaggle CPU limit: 540 minutes (9 hours)
- Kaggle TPU limit: 180 minutes (3 hours)

**Exit Conditions:**
- Status "complete": Success, download outputs
- Status "error": Failure, print kernel URL for debugging
- Status "cancelled": User cancelled, exit gracefully
- Timeout exceeded: Exit with KernelTimeoutError

---

## Autopilot Mode

### Architecture

**Components:**
1. **AutopilotState** (src/kagglebot/autopilot.py):
   - Tracks iteration history
   - Enforces MAX_SUBMISSIONS = 1
   - Selects best candidate

2. **Top1 Fetcher** (src/kagglebot/autopilot.py):
   - Downloads leaderboard CSV via Kaggle CLI
   - Parses Top1 public score
   - Caches for 60 minutes

3. **Heuristic Evaluator** (src/kagglebot/autopilot.py):
   - Compares offline score to Top1
   - Dual-margin logic (absolute + relative)
   - Direction-aware (minimize/maximize)

4. **Autopilot Runner** (src/kagglebot/autopilot_runner.py):
   - Main iteration loop (1-5)
   - Training + evaluation
   - Submission gating

### Iteration Loop

```python
for iteration in range(1, max_iterations + 1):
    # 1. Train model
    result = train_model(iteration)

    # 2. Evaluate offline (holdout or CV)
    offline_score = evaluate_offline(result)

    # 3. Compare to Top1
    meets_heuristic = evaluate_top1_heuristic(
        offline_score, top1_score, direction, margins
    )

    # 4. Record metrics
    metrics = {
        "offline_score": {...},
        "top1_comparison": {...},
    }

    # 5. Early submit check
    if meets_heuristic and submit_on_heuristic:
        submit(result.submission_path)
        return  # Stop loop

    # 6. Continue to next iteration
```

### Heuristic Logic

**Maximize (accuracy, AUC, F1):**
```python
abs_threshold = top1_score - margin_abs
rel_threshold = top1_score * (1 - margin_rel)
meets = (offline >= abs_threshold) AND (offline >= rel_threshold)
```

**Minimize (RMSE, MAE, loss):**
```python
abs_threshold = top1_score + margin_abs
rel_threshold = top1_score * (1 + margin_rel)
meets = (offline <= abs_threshold) AND (offline <= rel_threshold)
```

**Example (Maximize):**
- Top1: 0.95, Offline: 0.9567
- margin_abs: 0.05, margin_rel: 0.02
- abs_threshold: 0.90, rel_threshold: 0.931
- Result: 0.9567 >= 0.90 ✓ AND 0.9567 >= 0.931 ✓ → MEETS

### Submission Triggers

**Two Modes:**

1. **Early Submit** (`--submit-on-heuristic`, default: true):
   - Submit at first iteration where heuristic met
   - Stop loop immediately
   - Use case: Get submission ASAP

2. **Final Submit** (`--submit-at-final`, default: true):
   - Complete all 5 iterations
   - Select best iteration (by offline score)
   - Submit once at end
   - Use case: Explore full iteration space

**Exploration Mode** (`--no-submit-at-final`):
- Complete all 5 iterations
- Do NOT submit
- Use case: Offline tuning, KB building

---

## Security Model

### Credential Handling

**Supported Methods:**
1. `~/.kaggle/kaggle.json`:
   ```json
   {"username": "user", "key": "api_key"}
   ```

2. Environment variables:
   ```bash
   export KAGGLE_USERNAME="user"
   export KAGGLE_KEY="api_key"
   ```

**Never Logged:**
- API keys
- Access tokens
- Credential file paths

**Kernel Security:**
- No credentials in kernel code
- Submission from local machine (not kernel)
- Private kernels only (`is_private: true`)

### Git Operations

**Branch Strategy:**
- Dedicated branch per implement run: `bot/<slug>/<run_id>`
- Clean worktree required before implement
- Optional commit after verification

**Never Committed:**
- `.kaggle/kaggle.json`
- Large datasets (data/)
- Model artifacts (models/)
- Submissions (submissions/)

**Gitignore:**
```
.kaggle/
kaggle.json
artifacts/
data/
*.csv
*.zip
*.pkl
*.joblib
```

### Subprocess Safety

**Always Use List Args:**
```python
# Good
subprocess.run(["kaggle", "competitions", "download", "-c", slug])

# Bad (NEVER do this)
subprocess.run(f"kaggle competitions download -c {slug}", shell=True)
```

**Why:**
- Prevents command injection
- Proper argument escaping
- Safer error handling

---

## Failure Modes

### 1. Kaggle Authentication Failures

**Symptom**: 401 Unauthorized
**Cause**: Invalid/expired API key
**Remediation**:
1. Visit https://www.kaggle.com/<username>/account
2. Create New API Token
3. Replace ~/.kaggle/kaggle.json
4. Retry

**Tool Behavior**:
- Exit code 2
- Print remediation URL
- Never log credentials

### 2. Rules Not Accepted

**Symptom**: 403 Forbidden (competition access)
**Cause**: User hasn't manually accepted rules
**Remediation**:
1. Visit https://www.kaggle.com/competitions/<slug>/rules
2. Click "I Understand and Accept"
3. Retry command

**Tool Behavior**:
- Detect via Kaggle API
- Print rules URL
- Exit code 2 (user action required)
- NEVER automate clicking

### 3. Kernel Failures

**Push Rejected:**
- Invalid metadata schema
- Both `enable_gpu` and `enable_tpu` true
- Wrong `competition_sources` format (includes "c/")
- Missing required fields

**Execution Error:**
- Python exception in kernel code
- Out of memory (OOM)
- Missing dependencies
- Invalid file paths

**Timeout:**
- Exceeded tool timeout (default: 120 min)
- Exceeded Kaggle hard limit (GPU: 120, TPU: 180, CPU: 540)

**Remediation**:
- Visit kernel URL: https://www.kaggle.com/code/<kernel_id>
- Check logs for error message
- Fix code and regenerate package

### 4. Missing sample_submission.csv

**Symptom**: Validation error (file not found)
**Cause**: Competition doesn't provide sample_submission.csv
**Remediation**:
- Check competition data files
- Create sample_submission.csv manually based on competition description
- Some competitions only specify format in description

**Tool Behavior**:
- Skip validation if sample_submission.csv missing
- Warn user
- Proceed with submission (at user's risk)

### 5. Schema Mismatch

**Symptom**: Validation error (columns don't match)
**Cause**: Submission columns != sample_submission columns
**Remediation**:
- Check sample_submission.csv format
- Ensure exact column names (case-sensitive)
- Ensure correct column order

**Tool Behavior**:
- Print detailed diff (expected vs actual)
- Exit code 6 (validation error)
- Do not submit

### 6. Leaderboard Fetch Failure

**Symptom**: Top1 fetch error (autopilot mode)
**Cause**: Kaggle CLI timeout, CSV format change, no leaderboard
**Remediation**:
- Retry with `--force` (bypass cache)
- Check Kaggle API status
- Manually verify leaderboard exists

**Tool Behavior**:
- Continue autopilot without Top1 comparison
- Use offline scores only
- Warn user

---

## Future Enhancements

### Phase 1 (Completed)

- ✅ Core commands (bootstrap, implement, train, submit, run)
- ✅ Kaggle notebook runner (GPU/TPU)
- ✅ Autopilot with Top1 heuristic
- ✅ Comprehensive tests (72 tests)
- ✅ Documentation (README, architecture, failure modes)

### Phase 2 (Planned)

**Knowledge Base Integration:**
- Store improvement patterns in SQLite
- Retrieve similar competitions by tags
- Inject KB into Codex prompts
- Track offline vs public LB deltas

**Codex Improvement Prompts:**
- Auto-generate improvement prompts between iterations
- Include diagnostics + KB patterns
- Invoke Codex automatically
- Commit improvements on verification pass

**Advanced Metrics:**
- Feature importance tracking
- Error analysis by prediction quantile
- Overfitting detection (train/val gap)
- Auto-generated diagnostics.md

### Phase 3 (Future)

**Multi-Model Ensembling:**
- Train multiple models per iteration
- Weighted averaging / stacking
- Cross-validation for ensemble weights

**Hyperparameter Tuning:**
- Optuna/GridSearch integration
- Budget-aware tuning (time/compute)
- Parallel trial execution

**Distributed Training:**
- Multi-GPU local training
- Distributed Kaggle kernels
- Ray/Dask integration

**Advanced Autopilot:**
- Bayesian optimization of margins
- Adaptive iteration count
- Multi-objective optimization (score + speed)

---

## References

**Internal Documentation:**
- [AGENTS.md](../AGENTS.md) - Agent instructions
- [safety/submission_checklist.md](./safety/submission_checklist.md) - Pre-submission checklist
- [safety/failure_modes.md](./safety/failure_modes.md) - Known failure modes
- [AUTOPILOT_SINGLE_SUBMIT.md](./AUTOPILOT_SINGLE_SUBMIT.md) - Autopilot spec
- [AUTOPILOT_SUMMARY.md](./AUTOPILOT_SUMMARY.md) - Autopilot summary

**External References:**
- [Kaggle API Docs](https://www.kaggle.com/docs/api)
- [Kaggle CLI GitHub](https://github.com/Kaggle/kaggle-api)
- [Kaggle Kernels Docs](https://www.kaggle.com/docs/kernels)

---

**Last Reviewed**: 2026-01-02
**Status**: Production-ready
**Test Coverage**: 72/72 tests passing
