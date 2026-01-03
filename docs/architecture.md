# Architecture & Control Flow

This document describes kagglebot's autopilot control flow and safety gates.

## Overview

Kagglebot autopilot runs a score-gated improvement loop that **never submits unless the target score is met**. Evaluation happens offline using pseudo-test data (holdout or CV), and the public leaderboard Top1 score is fetched only as context.

## Autopilot Control Flow

```
┌─────────────────────────────────────────────────┐
│ 1. Bootstrap & Plan Generation                 │
│    - Download competition data                  │
│    - Agent creates plan.json with targets       │
│    - Generate initial model solution                 │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 2. Iteration Loop (max_iterations, max_total)  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │ a) Verify: Run tests (pytest)            │  │
│  └───────────────┬──────────────────────────┘  │
│                  │                              │
│                  ▼                              │
│  ┌──────────────────────────────────────────┐  │
│  │ b) Train: Run solution.py with compute   │  │
│  │    - Uses train.csv only                 │  │
│  │    - Generates submission.csv            │  │
│  └───────────────┬──────────────────────────┘  │
│                  │                              │
│                  ▼                              │
│  ┌──────────────────────────────────────────┐  │
│  │ c) Evaluate: Offline scoring              │  │
│  │    - Holdout split (default)             │  │
│  │    - OR CV (if plan specifies)           │  │
│  │    - OR test.csv (if labeled)            │  │
│  │    - Save metrics.json                   │  │
│  └───────────────┬──────────────────────────┘  │
│                  │                              │
│                  ▼                              │
│  ┌──────────────────────────────────────────┐  │
│  │ d) Check Target (Safety Gate #1)         │  │
│  │    - Compare offline score to target     │  │
│  │    - Direction-aware (minimize/maximize) │  │
│  │    met_target = _meets_target(score)     │  │
│  └───────────────┬──────────────────────────┘  │
│                  │                              │
│         ┌────────┴────────┐                     │
│         ▼                 ▼                     │
│  ┌──────────┐      ┌─────────────┐             │
│  │ TARGET   │      │ TARGET NOT  │             │
│  │ MET      │      │ MET         │             │
│  └────┬─────┘      └──────┬──────┘             │
│       │                   │                     │
│       │                   ▼                     │
│       │            ┌─────────────────────────┐  │
│       │            │ e) Diagnose & Improve   │  │
│       │            │    - Generate diag.md   │  │
│       │            │    - Call Codex improve │  │
│       │            │    - Update solution.py │  │
│       │            └──────┬──────────────────┘  │
│       │                   │                     │
│       │                   ▼                     │
│       │            ┌─────────────────────────┐  │
│       │            │ f) Patience Check       │  │
│       │            │    (Safety Gate #2)     │  │
│       │            │    - Track improvements │  │
│       │            │    - Stop if no progress│  │
│       │            └──────┬──────────────────┘  │
│       │                   │                     │
│       │                   │ Continue            │
│       │                   └─────────────────────┤
│       │                                         │
│       ▼                                         │
│  ┌──────────────────────────────────────────┐  │
│  │ 3. Attempt Submit (Safety Gate #3)       │  │
│  │    - Only if --submit flag set           │  │
│  │    - Only if met_target=true             │  │
│  │    - Check rules accepted                │  │
│  │    - Check deduplication ledger          │  │
│  │    - Check rate limit (cooldown)         │  │
│  │    - Submit via Kaggle CLI               │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Safety Gates

### 1. Score-Based Gating (`_meets_target`)

**Location**: `src/kagglebot/autopilot.py:_meets_target()`

```python
def _meets_target(score: float, target: float, direction: str) -> bool:
    """Direction-aware target comparison."""
    if direction == "minimize":
        return score <= target
    else:  # maximize
        return score >= target
```

- **Purpose**: Ensure submission only when offline performance meets agent-defined target
- **Input**: Current iteration score, target score from plan.json, metric direction
- **Output**: Boolean met_target flag
- **Safety**: No submission unless this returns True

### 2. Patience & Improvement Tracking

**Location**: `src/kagglebot/autopilot.py:run_autopilot()`

```python
patience_counter = 0
best_score = None

for iteration in range(max_iterations):
    outcome = train_evaluate_and_predict(...)

    if _is_improvement(outcome.score, best_score, direction, min_improvement):
        best_score = outcome.score
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break  # Stop iteration
```

- **Purpose**: Prevent infinite loops without progress
- **Parameters**: `patience` (default 2), `min_improvement` (default 0.0)
- **Safety**: Stops autopilot if N consecutive iterations show no improvement

### 3. Submission Guardrails

**Location**: `src/kagglebot/autopilot.py:_attempt_submit()`

Checks performed before submission:
1. **Flag check**: `--submit` must be explicitly set
2. **Target check**: `met_target` must be True
3. **Rules check**: Competition rules must be accepted (via `check_rules_accepted()`)
4. **Deduplication**: SHA256 hash must not exist in ledger.jsonl
5. **Rate limit**: Cooldown period enforced (default 5 min between submissions)
6. **Max submissions**: Hard cap per autopilot run (default 5)

### 4. Hard Caps

**Location**: `src/kagglebot/autopilot.py:AutopilotConfig`

```python
@dataclass
class AutopilotConfig:
    max_iterations: int = 5      # Max improvement cycles
    max_total_min: int = 120      # Max wall-clock time (2 hours)
    patience: int = 2             # Early stopping
    max_submissions: int = 5      # Submission quota per run
```

- **Purpose**: Prevent runaway resource consumption
- **Enforcement**: Loop breaks if any limit exceeded
- **Safety**: No infinite compute or API abuse

### 5. Non-Interactive Design

**Key Principle**: Zero human intervention required during autopilot run

- No prompts for user input
- No automated rule acceptance (manual browser action required once)
- All decisions encoded in `plan.json` or CLI flags
- Errors exit with actionable messages and non-zero exit codes

### 6. Secret Protection

**Enforced in**:
- `.gitignore`: Excludes `kaggle.json`, `*.env`, `data/`, `artifacts/`
- Knowledge Base: No competition data or predictions stored, only metadata
- Logs: No credential echoing in subprocess output

## Evaluation Strategy

### Pseudo-Test Scoring

Autopilot uses **offline evaluation** on `train.csv` only (unless labeled test exists):

1. **Holdout** (default):
   - Split train.csv into train/val (e.g., 80/20)
   - Stratified for classification
   - Fixed seed for reproducibility

2. **Cross-Validation**:
   - KFold or StratifiedKFold
   - Configurable n_splits (default 5)
   - Reports mean and std across folds

3. **Labeled Test** (rare):
   - Only if `test.csv` contains target column
   - Errors if not available and `--score-source test` specified

### Top1 Public Leaderboard

**Location**: `src/kagglebot/autopilot.py:leaderboard_top1()`

- **Fetch**: Via `kaggle competitions leaderboard --csv`
- **Usage**: Recorded in `run.json` as context only
- **NOT used for**: Submission decisions (only `met_target` gates submission)
- **Purpose**: Provide agent with public benchmark for plan.json target setting

## Artifacts Layout

```
artifacts/<slug>/
  meta.json                    # Competition metadata
  plan.json                    # Agent-defined targets

  context/
    rules_url.txt
    dataset_profile.json       # Dataset stats + file format summary
    sample_submission.csv
    top1_public.json           # Top1 score (context only)

  prompts/
    codex_plan_and_implement.md # Initial plan + implementation template
    codex_improve.md           # Improvement template

  runs/<run-id>/
    run.json                   # Run config + status

    iter-<k>/
      metrics.json             # Offline scores
      diagnostics.md           # Agent-readable eval summary
      submission.csv
      agent/
        prompt.md              # Codex input
        codex_last_message.txt # Codex output

  submissions/
    ledger.jsonl               # Deduplication log
```

### Key Files

**plan.json** (agent-created, user-editable):
```json
{
  "target_metric": "rmse",
  "target_score": 0.13,
  "target_direction": "minimize",
  "score_source": "holdout",
  "holdout_frac": 0.2,
  "seed": 42
}
```

**metrics.json** (per iteration):
```json
{
  "score_source": "holdout",
  "metric": "rmse",
  "direction": "minimize",
  "value": 0.145,
  "train_score": 0.120,
  "val_score": 0.145,
  "target": 0.13,
  "met_target": false,
  "top1_public": 0.11234
}
```

## Knowledge Base Integration

**Location**: `src/kagglebot/knowledge/`

- **Schema**: SQLite database at `knowledge/kagglebot.db`
- **Tables**: competitions, tags, competition_tags, runs, iterations, improvements
- **Purpose**: Store cross-competition learnings (what worked, what didn't)
- **Privacy**: No raw data or predictions stored, only metadata and summaries

**Tagging**:
- Controlled vocabulary in `knowledge/taxonomy.yml`
- Tags inferred from dataset profiling (e.g., `n_rows_large`, `high_cardinality_cats`)
- Used for similarity search: "Show me past runs on similar tabular competitions"

## Error Handling

### Exit Codes

Defined in `src/kagglebot/exceptions.py`:

| Code | Error                        | Meaning                               |
|------|------------------------------|---------------------------------------|
| 1    | KaggleBotError               | Generic error                         |
| 2    | RulesNotAcceptedError        | Rules not manually accepted in browser|
| 4    | KaggleCliError               | Kaggle CLI returned non-zero          |
| 6    | ValidationError              | submission.csv invalid                |
| 8    | DuplicateSubmissionError     | Hash already in ledger                |
| 9    | SubmissionRateLimitError     | Cooldown period not elapsed           |
| 10   | GPUNotAvailableError         | Requested GPU not available           |
| 11   | KernelTimeoutError           | Kaggle kernel timed out               |
| 12   | KernelFailedError            | Kaggle kernel failed                  |
| 14   | MaxSubmissionsError          | Quota exceeded for autopilot run      |

### Actionable Messages

All errors include:
- **What went wrong**: Clear description
- **Why it matters**: Impact on workflow
- **How to fix**: Concrete next steps

Example:
```
RulesNotAcceptedError: Competition rules not accepted
→ Visit https://www.kaggle.com/c/titanic/rules
→ Click "I Understand and Accept" in your browser
→ Re-run: uv run kagglebot autopilot titanic
```

## CLI Ergonomics

### Consistent Defaults

- `--dry-run`: Default for `submit` command (require `--no-dry-run` to actually submit)
- `--compute local_cpu`: Safe default (no GPU assumptions)
- `--score-source auto`: Use holdout unless test is labeled
- `--max-iterations 5`: Reasonable cap
- `--patience 2`: Stop if 2 iterations show no improvement

### Help Text

Every command has:
- Clear description
- Required vs optional arguments
- Examples
- Safety warnings (e.g., "--submit will submit to Kaggle")

```bash
uv run kagglebot autopilot --help
```

### Progress Indicators

- Rich library for progress bars during downloads/training
- Clear iteration logging: `[iter-1/5] Training... RMSE=0.145 (target: 0.13)`
- Final summary: `Autopilot complete: 3 iterations, best RMSE=0.132, met_target=true, submitted=true`

## Testing Strategy

**Location**: `tests/`

### Unit Tests
- `test_met_target_logic()`: Verify direction-aware comparison
- `test_deduplication()`: Ensure hash collision prevention
- `test_validation()`: Check submission.csv format enforcement

### Integration Tests
- `test_autopilot_uses_plan_from_agent()`: Plan.json flows through correctly
- `test_autopilot_no_submit_when_below_target()`: Gating works
- `test_autopilot_submit_when_target_met()`: Submission on success
- `test_autopilot_patience_stops()`: Early stopping logic

### Safety Tests
- No test should submit to Kaggle (all use monkeypatching)
- Dry-run mode verified in CI
- Secret leakage tests (check logs for kaggle.json patterns)

## Future Enhancements

- **Multi-objective optimization**: Support Pareto frontiers (e.g., accuracy + runtime)
- **Ensemble submission**: Auto-blend top-K iterations
- **Model persistence**: Save trained models for reuse
- **Distributed compute**: Support Ray/Dask for large datasets
- **Streaming evaluation**: Handle datasets that don't fit in memory
