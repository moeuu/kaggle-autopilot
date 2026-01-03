# Autopilot: Single-Submit with Top1 Heuristic

> Note: References to "initial model" in this document should be read as "strong initial model" (web‑researched, not a simplistic initial model).

**Method 1 Implementation Specification**

## Overview

This document specifies the "single-submit autopilot" approach (Method 1) for kagglebot. The core principle: iterate improvements offline using holdout or cross-validation evaluation, fetch Kaggle's public Top1 score without submitting, compare offline score to Top1 heuristically, and submit only once when the heuristic is met.

**Key Characteristics:**
- **Max 5 improvement iterations** (hard-coded)
- **No submissions during loop** (offline evaluation only)
- **Single submit at end** (unless early-submit triggered)
- **Top1 heuristic** (margin-based comparison)
- **Knowledge Base integration** (capture what worked)

**User Request Summary:**
1. Define exact autopilot policy (single-submit)
2. Define robust compare rule (minimize/maximize + margins)
3. Leaderboard Top1 fetch (prefer CSV download and parse)
4. Metrics.json contract updates
5. Guardrails (max-submissions=1, dedupe, timeouts, secrets)
6. Knowledge Base integration (unchanged but capture top1/offline)
7. Prompt templates for Codex (initial model + improvement with KB)

---

## Table of Contents

1. [Autopilot Policy](#1-autopilot-policy)
2. [Compare Rule (Top1 Heuristic)](#2-compare-rule-top1-heuristic)
3. [Leaderboard Top1 Fetch](#3-leaderboard-top1-fetch)
4. [Metrics.json Contract](#4-metricsjson-contract)
5. [Guardrails](#5-guardrails)
6. [Knowledge Base Integration](#6-knowledge-base-integration)
7. [Codex Prompt Templates](#7-codex-prompt-templates)
8. [Implementation Phases](#8-implementation-phases)

---

## 1. Autopilot Policy

### 1.1 Core Loop

```python
# Pseudo-code for autopilot loop
MAX_ITERATIONS = 5  # Hard-coded

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"[Iteration {iteration}/{MAX_ITERATIONS}]")

    # Step 1: Train model
    train_result = train_model(iteration=iteration)

    # Step 2: Evaluate offline (holdout or CV)
    offline_score = evaluate_offline(
        train_result=train_result,
        score_source=config.score_source,  # auto/holdout/cv
        random_seed=config.seed,
    )

    # Step 3: Fetch Top1 from Kaggle leaderboard
    top1_info = fetch_top1_score(slug=config.slug)

    # Step 4: Evaluate heuristic
    meets_heuristic = evaluate_top1_heuristic(
        offline_score=offline_score.value,
        top1_score=top1_info["score"],
        direction=infer_direction(config.target_metric),
        margin_abs=config.top1_margin_abs,
        margin_rel=config.top1_margin_rel,
    )

    # Step 5: Record metrics
    record_metrics(
        iteration=iteration,
        offline_score=offline_score,
        top1_score=top1_info["score"],
        meets_heuristic=meets_heuristic,
    )

    # Step 6: Early submit (optional)
    if meets_heuristic and config.early_submit:
        submit(train_result.submission_path, reason="early_submit")
        return  # Stop loop

    # Step 7: Diagnose and generate improvement prompt
    diagnostics = run_diagnostics(train_result, offline_score)

    # Step 8: Generate improvement prompt (if not last iteration)
    if iteration < MAX_ITERATIONS:
        improvement_prompt = generate_improvement_prompt(
            iteration=iteration,
            diagnostics=diagnostics,
            kb_entries=retrieve_kb_entries(config.kb_tags),
        )
        # Next iteration will use this prompt

# Final submit (unless disabled)
if config.final_submit:
    best_iteration = select_best_iteration(metrics_history)
    submit(best_iteration.submission_path, reason="final_best")
```

### 1.2 Submission Triggers

**Two submission modes:**

1. **Early Submit** (optional, requires `--early-submit` flag):
   - If `meets_heuristic == True` at ANY iteration
   - Submit immediately and stop loop
   - Use case: User wants fastest possible valid submission

2. **Final Submit** (default, disable with `--no-final-submit`):
   - After all iterations complete
   - Select best iteration by offline score
   - Submit regardless of heuristic (user decision)
   - Use case: Standard workflow (iterate 5 times, submit best)

**Max submissions enforced:** `MAX_SUBMISSIONS = 1` (hard-coded in AutopilotState)

### 1.3 CLI Interface

```bash
# Extend existing `kagglebot run` command
uv run kagglebot run titanic \
  --agent codex \
  --compute local_cpu \
  --autopilot \
  --target-metric accuracy \
  --target-direction maximize \
  --top1-margin-abs 0.05 \
  --top1-margin-rel 0.02 \
  --score-source auto \
  --early-submit \
  --force
```

**New Flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--autopilot` | bool | False | Enable autopilot mode (5 iterations) |
| `--target-metric` | str | auto | Metric name (e.g., accuracy, rmse) |
| `--target-direction` | str | auto | "maximize" or "minimize" (auto-infer from metric) |
| `--top1-margin-abs` | float | 0.05 | Absolute margin for Top1 heuristic |
| `--top1-margin-rel` | float | 0.02 | Relative margin for Top1 heuristic (2%) |
| `--score-source` | str | auto | Offline evaluation: auto/holdout/cv |
| `--early-submit` | bool | False | Submit immediately when heuristic met |
| `--no-final-submit` | bool | False | Skip final submit (exploration mode) |
| `--kb-tags` | str | auto | Comma-separated tags for KB retrieval |

**Auto-inference:**
- `--target-metric auto`: Infer from plan.json or sample_submission columns
- `--target-direction auto`: Infer from metric name (accuracy/auc → maximize, rmse/mae → minimize)
- `--score-source auto`: Use holdout if small (<1000 rows), CV if larger
- `--kb-tags auto`: Extract from competition (e.g., "tabular,classification,small-dataset")

### 1.4 Exit Conditions

The loop stops when:
1. **Max iterations reached** (5 iterations always)
2. **Early submit triggered** (if `--early-submit` and heuristic met)
3. **User interrupt** (Ctrl+C, save state)
4. **Time limit exceeded** (if `--max-time` set)
5. **Consecutive failures** (3 training failures = abort)

**Exit codes:**
- 0: Success (submitted or completed iterations)
- 1: General error
- 11: Timeout (exceeded `--max-time`)
- 5: Training failure (model crashed)

---

## 2. Compare Rule (Top1 Heuristic)

### 2.1 Heuristic Function

```python
def evaluate_top1_heuristic(
    offline_score: float,
    top1_score: float,
    direction: str,  # "maximize" or "minimize"
    margin_abs: float = 0.05,
    margin_rel: float = 0.02,
) -> bool:
    """
    Check if offline score meets Top1 heuristic.

    Uses BOTH absolute and relative margins (AND condition).
    Stricter margin wins (more conservative).

    Args:
        offline_score: Offline evaluation score (holdout or CV)
        top1_score: Kaggle public Top1 score
        direction: "maximize" (accuracy, AUC) or "minimize" (RMSE, MAE)
        margin_abs: Absolute margin (e.g., 0.05 means within 0.05 of Top1)
        margin_rel: Relative margin (e.g., 0.02 means within 2% of Top1)

    Returns:
        True if offline score meets BOTH margin thresholds

    Examples:
        >>> # Maximize (accuracy)
        >>> evaluate_top1_heuristic(0.9567, 0.95, "maximize", 0.05, 0.02)
        True  # 0.9567 >= 0.90 (abs) AND >= 0.931 (rel)

        >>> # Minimize (RMSE)
        >>> evaluate_top1_heuristic(0.21, 0.20, "minimize", 0.05, 0.02)
        True  # 0.21 <= 0.25 (abs) AND <= 0.204 (rel)
    """
    if direction == "maximize":
        # Higher is better
        abs_threshold = top1_score - margin_abs
        rel_threshold = top1_score * (1 - margin_rel)

        # Must meet BOTH thresholds (AND condition)
        meets_abs = offline_score >= abs_threshold
        meets_rel = offline_score >= rel_threshold

        return meets_abs and meets_rel

    elif direction == "minimize":
        # Lower is better
        abs_threshold = top1_score + margin_abs
        rel_threshold = top1_score * (1 + margin_rel)

        # Must meet BOTH thresholds (AND condition)
        meets_abs = offline_score <= abs_threshold
        meets_rel = offline_score <= rel_threshold

        return meets_abs and meets_rel

    else:
        raise ValueError(f"Invalid direction: {direction}")
```

### 2.2 Direction Auto-Inference

```python
def infer_direction(metric_name: str) -> str:
    """
    Infer metric direction from name.

    Maximize: accuracy, auc, roc_auc, f1, precision, recall
    Minimize: rmse, mae, mse, error, loss

    Args:
        metric_name: Metric name (case-insensitive)

    Returns:
        "maximize" or "minimize"

    Raises:
        ValueError: If metric direction cannot be inferred
    """
    metric_lower = metric_name.lower()

    # Maximize metrics
    if any(keyword in metric_lower for keyword in [
        "accuracy", "auc", "roc", "f1", "precision", "recall", "score"
    ]):
        return "maximize"

    # Minimize metrics
    if any(keyword in metric_lower for keyword in [
        "rmse", "mae", "mse", "error", "loss"
    ]):
        return "minimize"

    # Unknown - require explicit flag
    raise ValueError(
        f"Cannot infer direction for metric '{metric_name}'. "
        f"Use --target-direction {{maximize,minimize}}."
    )
```

### 2.3 Recommended Defaults

| Metric Type | margin_abs | margin_rel | Rationale |
|-------------|------------|------------|-----------|
| Accuracy (0-1) | 0.05 | 0.02 (2%) | 5% absolute drop acceptable |
| RMSE (0-∞) | 0.05 | 0.02 (2%) | 5% absolute increase acceptable |
| AUC (0-1) | 0.03 | 0.01 (1%) | AUC more sensitive, tighter margins |
| MAE (0-∞) | 0.10 | 0.05 (5%) | MAE less sensitive, wider margins |

**User override:**
```bash
# Tighter margins for competitive leaderboards
uv run kagglebot run <slug> --autopilot \
  --top1-margin-abs 0.02 \
  --top1-margin-rel 0.01

# Looser margins for exploration
uv run kagglebot run <slug> --autopilot \
  --top1-margin-abs 0.10 \
  --top1-margin-rel 0.05
```

### 2.4 Example Calculations

**Example 1: Maximize (Accuracy)**
- Top1 score: 0.95
- Offline score: 0.9567
- margin_abs: 0.05
- margin_rel: 0.02

```python
abs_threshold = 0.95 - 0.05 = 0.90
rel_threshold = 0.95 * (1 - 0.02) = 0.931

offline_score >= abs_threshold?  0.9567 >= 0.90  ✓
offline_score >= rel_threshold?  0.9567 >= 0.931 ✓

Result: True (meets heuristic)
```

**Example 2: Minimize (RMSE)**
- Top1 score: 0.20
- Offline score: 0.21
- margin_abs: 0.05
- margin_rel: 0.02

```python
abs_threshold = 0.20 + 0.05 = 0.25
rel_threshold = 0.20 * (1 + 0.02) = 0.204

offline_score <= abs_threshold?  0.21 <= 0.25  ✓
offline_score <= rel_threshold?  0.21 <= 0.204 ✗

Result: False (does not meet heuristic - rel margin too strict)
```

**Example 3: Edge Case (Very High Top1)**
- Top1 score: 0.99 (very high)
- Offline score: 0.9567
- margin_abs: 0.05
- margin_rel: 0.02

```python
abs_threshold = 0.99 - 0.05 = 0.94
rel_threshold = 0.99 * (1 - 0.02) = 0.9702

offline_score >= abs_threshold?  0.9567 >= 0.94  ✓
offline_score >= rel_threshold?  0.9567 >= 0.9702 ✗

Result: False (relative margin stricter when Top1 is high)
```

---

## 3. Leaderboard Top1 Fetch

### 3.1 Implementation Strategy

**Preferred approach:** Use Kaggle CLI to download leaderboard CSV and parse.

**Why not web scraping?**
- Violates Kaggle ToS
- Fragile (breaks on HTML changes)
- Rate limited unpredictably

**Why not Kaggle Python API?**
- `KaggleApi.competition_leaderboard_view()` exists but may require auth
- Returns list of dicts (same data as CSV)
- CLI approach simpler (no API client changes)

**Chosen: Kaggle CLI CSV download**
```bash
kaggle competitions leaderboard <slug> --download --path <cache_dir>
```

Output: `<cache_dir>/<slug>.csv`

### 3.2 CSV Format

**Example: titanic.csv**
```csv
teamId,teamName,submissionDate,score
4567890,user123,2026-01-02 10:15:00,0.95
4567891,user456,2026-01-02 09:30:00,0.9467
...
```

**Columns:**
- `teamId`: Team ID (int)
- `teamName`: Team name (str)
- `submissionDate`: Submission timestamp (str, ISO format)
- `score`: Public leaderboard score (float or str with commas)

**Important:** Some leaderboards use comma separators in scores (e.g., "1,234.56"). Strip before parsing.

### 3.3 Fetch Implementation

```python
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

def fetch_top1_score(
    competition_slug: str,
    cache_dir: Path,
    cache_ttl_minutes: int = 60,
) -> dict:
    """
    Fetch Kaggle public Top1 score via leaderboard CSV.

    Caches result to avoid excessive API calls (60-minute TTL).

    Args:
        competition_slug: Competition slug (e.g., "titanic")
        cache_dir: Directory for cache and CSV downloads
        cache_ttl_minutes: Cache expiration time (default: 60 min)

    Returns:
        dict with keys:
            - score: float (Top1 public score)
            - team_name: str
            - team_id: int
            - submission_date: str (ISO format)
            - timestamp: str (fetch timestamp)
            - cached: bool (True if from cache)

    Raises:
        KaggleCliError: If CSV download fails
        ValueError: If CSV is empty or malformed
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "leaderboard_cache.json"

    # Check cache
    if cache_file.exists():
        cache_data = json.loads(cache_file.read_text())
        cache_time = datetime.fromisoformat(cache_data["timestamp"])
        age_minutes = (datetime.now() - cache_time).total_seconds() / 60

        if age_minutes < cache_ttl_minutes:
            print(f"[Cache hit] Top1: {cache_data['score']} (age: {age_minutes:.1f} min)")
            cache_data["cached"] = True
            return cache_data

    # Download leaderboard CSV
    csv_path = cache_dir / f"{competition_slug}.csv"
    try:
        subprocess.run(
            [
                "kaggle", "competitions", "leaderboard",
                competition_slug,
                "--download",
                "--path", str(cache_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise KaggleCliError(
            message=f"Leaderboard download failed: {exc.stderr}",
            command=exc.cmd,
            exit_code=exc.returncode,
            output=exc.stderr,
        ) from exc

    # Parse CSV
    if not csv_path.exists():
        raise ValueError(f"Leaderboard CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("Leaderboard CSV is empty")

    # Extract Top1 (first row)
    top1_row = df.iloc[0]

    # Parse score (handle comma separators)
    score_str = str(top1_row["score"]).replace(",", "")
    score = float(score_str)

    # Build result
    result = {
        "score": score,
        "team_name": str(top1_row["teamName"]),
        "team_id": int(top1_row["teamId"]),
        "submission_date": str(top1_row["submissionDate"]),
        "timestamp": datetime.now().isoformat(),
        "cached": False,
    }

    # Write cache
    cache_file.write_text(json.dumps(result, indent=2))

    print(f"[Leaderboard] Top1: {score} by {result['team_name']}")
    return result
```

### 3.4 Caching Strategy

**Cache location:** `artifacts/<slug>/leaderboard_cache.json`

**Cache TTL:** 60 minutes (configurable via `--leaderboard-cache-ttl`)

**Rationale:**
- Kaggle leaderboards update slowly (minutes to hours)
- 5 autopilot iterations = 5 fetches (without cache = excessive)
- 60-minute TTL balances freshness and API courtesy

**Cache invalidation:**
- Manual: Delete `leaderboard_cache.json`
- Auto: TTL expiration
- Flag: `--no-leaderboard-cache` (bypass cache)

**Cache schema:**
```json
{
  "score": 0.95,
  "team_name": "user123",
  "team_id": 4567890,
  "submission_date": "2026-01-02 10:15:00",
  "timestamp": "2026-01-02T11:30:00",
  "cached": false
}
```

---

## 4. Metrics.json Contract

### 4.1 Schema Overview

Extend existing `metrics.json` with three new blocks:
1. **offline_score**: Offline evaluation details (holdout or CV)
2. **top1_comparison**: Top1 heuristic evaluation
3. **chosen_best**: Best iteration selection and submission status

### 4.2 Full Schema

```json
{
  "iteration": 3,
  "model": "LogisticRegression",
  "features": ["Pclass", "Sex", "Age", "Fare"],
  "hyperparameters": {
    "C": 1.0,
    "max_iter": 2000
  },

  "offline_score": {
    "source": "holdout",
    "metric": "accuracy",
    "value": 0.9567,
    "std": null,
    "n_train": 800,
    "n_eval": 200,
    "holdout_fraction": 0.2,
    "cv_folds": null,
    "random_seed": 42
  },

  "top1_comparison": {
    "top1_score": 0.95,
    "top1_team_name": "user123",
    "direction": "maximize",
    "margin_abs": 0.05,
    "margin_rel": 0.02,
    "threshold_abs": 0.90,
    "threshold_rel": 0.931,
    "threshold_used": 0.931,
    "offline_meets_heuristic": true
  },

  "chosen_best": {
    "iteration": 3,
    "reason": "best_offline_score",
    "submitted": true,
    "submission_path": "artifacts/titanic/submissions/20260102_113000_submission.csv"
  },

  "training": {
    "duration_seconds": 2.3,
    "timestamp": "2026-01-02T11:30:00"
  }
}
```

### 4.3 Field Definitions

**offline_score block:**

| Field | Type | Description |
|-------|------|-------------|
| `source` | str | "holdout", "cv", or "test" (if labels available) |
| `metric` | str | Metric name (e.g., "accuracy", "rmse") |
| `value` | float | Offline score value |
| `std` | float\|null | Standard deviation (CV only, null for holdout) |
| `n_train` | int | Number of training samples |
| `n_eval` | int | Number of evaluation samples (holdout or CV total) |
| `holdout_fraction` | float\|null | Holdout fraction (e.g., 0.2 = 20%), null for CV |
| `cv_folds` | int\|null | Number of CV folds, null for holdout |
| `random_seed` | int | Random seed for reproducibility |

**top1_comparison block:**

| Field | Type | Description |
|-------|------|-------------|
| `top1_score` | float | Kaggle public Top1 score |
| `top1_team_name` | str | Top1 team name |
| `direction` | str | "maximize" or "minimize" |
| `margin_abs` | float | Absolute margin (user-configured) |
| `margin_rel` | float | Relative margin (user-configured) |
| `threshold_abs` | float | Calculated absolute threshold |
| `threshold_rel` | float | Calculated relative threshold |
| `threshold_used` | float | Stricter threshold (min for max, max for min) |
| `offline_meets_heuristic` | bool | True if offline meets BOTH margins |

**chosen_best block:**

| Field | Type | Description |
|-------|------|-------------|
| `iteration` | int | Best iteration number (1-5) |
| `reason` | str | Selection reason: "best_offline_score", "early_submit", "only_valid" |
| `submitted` | bool | True if submission occurred |
| `submission_path` | str\|null | Path to submitted file, null if not submitted |

### 4.4 Example: CV Evaluation

```json
{
  "iteration": 2,
  "offline_score": {
    "source": "cv",
    "metric": "rmse",
    "value": 0.21,
    "std": 0.015,
    "n_train": 900,
    "n_eval": 1000,
    "holdout_fraction": null,
    "cv_folds": 5,
    "random_seed": 42
  },
  "top1_comparison": {
    "top1_score": 0.20,
    "direction": "minimize",
    "margin_abs": 0.05,
    "margin_rel": 0.02,
    "threshold_abs": 0.25,
    "threshold_rel": 0.204,
    "threshold_used": 0.204,
    "offline_meets_heuristic": false
  },
  "chosen_best": {
    "iteration": 2,
    "reason": "best_offline_score",
    "submitted": false,
    "submission_path": null
  }
}
```

---

## 5. Guardrails

### 5.1 Max Submissions Enforcement

```python
class AutopilotState:
    """
    Autopilot state tracker with submission enforcement.

    Hard-coded MAX_SUBMISSIONS = 1 (cannot be overridden).
    """
    MAX_SUBMISSIONS = 1  # Hard-coded

    def __init__(self, slug: str, run_id: str):
        self.slug = slug
        self.run_id = run_id
        self.submissions_made = 0
        self.iteration_history = []

    def submit(self, submission_path: Path, reason: str):
        """
        Submit to Kaggle with MAX_SUBMISSIONS enforcement.

        Args:
            submission_path: Path to submission CSV
            reason: Submission reason ("early_submit" or "final_best")

        Raises:
            MaxSubmissionsError: If already submitted in this autopilot run
        """
        if self.submissions_made >= self.MAX_SUBMISSIONS:
            raise MaxSubmissionsError(
                f"Max submissions ({self.MAX_SUBMISSIONS}) already made in autopilot run. "
                f"Cannot submit again (reason: {reason})."
            )

        # Submit via existing submit_competition()
        submit_competition(self.slug, submission_path, f"Autopilot {reason}")

        # Record submission
        self.submissions_made += 1
        print(f"[Submitted] {reason} ({self.submissions_made}/{self.MAX_SUBMISSIONS})")
```

### 5.2 Enhanced Deduplication

**Existing ledger:** `artifacts/<slug>/submissions/history.jsonl`

**Enhancement:** Add `autopilot_run_id` field to distinguish autopilot runs.

```jsonl
{"timestamp": "2026-01-02T11:30:00", "submission_path": "...", "sha256": "abc123", "message": "Autopilot early_submit", "run_id": "20260102_113000", "autopilot_run_id": "auto_20260102_110000", "slug": "titanic"}
```

**Deduplication check:**
1. Compute SHA256 of submission file
2. Check if hash exists in ledger
3. If exists in SAME autopilot run → allow (iteration improvement)
4. If exists in DIFFERENT autopilot run → warn but allow (user decision)
5. If exists in MANUAL submission → error (duplicate submission)

**Rationale:** Autopilot iterations may generate identical submissions (e.g., no improvement). Allow duplicates within same run, but warn across runs.

### 5.3 Iteration Cap

**Hard-coded:** `MAX_ITERATIONS = 5`

**Why not configurable?**
- Prevents runaway loops
- Aligns with submission limits (many competitions cap at 5/day)
- Forces users to focus on quality over quantity

**User override (future):** `--max-iterations N` (requires justification)

### 5.4 Time Cap

**Default:** 8 hours (480 minutes)

**Override:** `--max-time <minutes>`

**Implementation:**
```python
import time

def run_autopilot_with_timeout(max_time_minutes: int):
    start_time = time.time()

    for iteration in range(1, MAX_ITERATIONS + 1):
        elapsed_minutes = (time.time() - start_time) / 60

        if elapsed_minutes > max_time_minutes:
            print(f"[Timeout] Exceeded {max_time_minutes} minutes")
            raise TimeoutError(f"Autopilot exceeded {max_time_minutes} minutes")

        # Run iteration...
        print(f"[Iteration {iteration}] Elapsed: {elapsed_minutes:.1f} min")
```

**Exit code:** 11 (timeout)

### 5.5 Secret Scanning

**Pre-submission check:** Scan submission CSV for patterns resembling secrets.

**Patterns:**
```python
SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",  # OpenAI API keys
    r"kaggle_key_[a-zA-Z0-9]{16,}",  # Kaggle API keys (hypothetical)
    r"[0-9a-f]{64}",  # SHA256 hashes (potential tokens)
    r"-----BEGIN [A-Z ]+ KEY-----",  # PEM keys
]

def scan_for_secrets(file_path: Path) -> bool:
    """
    Scan file for secret patterns.

    Returns:
        True if secrets detected, False otherwise
    """
    content = file_path.read_text()

    for pattern in SECRET_PATTERNS:
        if re.search(pattern, content):
            return True

    return False

def validate_submission_safety(file_path: Path):
    """Pre-submission safety check."""
    if scan_for_secrets(file_path):
        raise SecurityError(
            f"Potential secrets detected in {file_path}. "
            f"Refusing to submit. Review file contents."
        )
```

**Exit code:** 13 (security error)

### 5.6 Consecutive Failure Cap

**Policy:** Abort after 3 consecutive training failures.

**Rationale:** If training fails 3 times in a row, likely a persistent issue (OOM, missing deps, etc.). Don't waste time/quota.

```python
consecutive_failures = 0

for iteration in range(1, MAX_ITERATIONS + 1):
    try:
        train_result = train_model(iteration)
        consecutive_failures = 0  # Reset on success
    except Exception as exc:
        consecutive_failures += 1
        print(f"[Training failed] Iteration {iteration}: {exc}")

        if consecutive_failures >= 3:
            raise AutopilotAbortError(
                f"Aborting autopilot: {consecutive_failures} consecutive training failures."
            )
```

**Exit code:** 5 (training error)

---

## 6. Knowledge Base Integration

### 6.1 Schema Updates

**Add to existing KB entry schema:**

```json
{
  "id": "kb_20260102_153000_abc123",
  "competition_slug": "titanic",
  "tags": ["tabular", "classification", "small-dataset"],
  "created_at": "2026-01-02T15:30:00",

  "improvement_record": {
    "offline_score_before": 0.93,
    "offline_score_after": 0.9567,
    "delta": 0.0267,
    "iterations_taken": 3,
    "top1_score": 0.95,
    "meets_heuristic": true,
    "metric": "accuracy",
    "direction": "maximize"
  },

  "what_changed": {
    "features_added": ["age_pclass_interaction", "cabin_null_flag"],
    "features_removed": [],
    "hyperparameters_changed": {
      "C": {"old": 1.0, "new": 0.5},
      "max_iter": {"old": 1000, "new": 2000}
    },
    "model_changed": false
  },

  "why_it_worked": {
    "hypothesis": "Age and Pclass interaction captures survival patterns (1st class children prioritized)",
    "evidence": "Feature importance increased for age_pclass_interaction (0.15 → 0.23)",
    "diagnostics_summary": "Previous errors: 60% misclassified young 3rd class passengers",
    "rationale": "Interaction feature directly addresses largest error category"
  },

  "diagnostics_file": "artifacts/titanic/runs/<run_id>/diagnostics.md",
  "submission_path": "artifacts/titanic/submissions/<run_id>_submission.csv"
}
```

### 6.2 Retrieval Ranking

**Objective:** Given current competition tags, rank KB entries by relevance.

**Algorithm: Jaccard Similarity**

```python
def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Compute Jaccard similarity: |A ∩ B| / |A ∪ B|

    Returns:
        Float in [0, 1], higher = more similar
    """
    if len(set_a | set_b) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def rank_kb_entries(query_tags: set[str], kb_entries: list[dict]) -> list[dict]:
    """
    Rank KB entries by Jaccard similarity to query tags.

    Args:
        query_tags: Current competition tags (e.g., {"tabular", "classification"})
        kb_entries: List of KB entry dicts

    Returns:
        Sorted list of dicts with keys:
            - entry: KB entry dict
            - jaccard: Jaccard score
            - rank: 1-indexed rank
    """
    scored = []

    for entry in kb_entries:
        entry_tags = set(entry["tags"])
        jaccard = jaccard_similarity(query_tags, entry_tags)
        scored.append({
            "entry": entry,
            "jaccard": jaccard,
        })

    # Sort by Jaccard (descending)
    scored.sort(key=lambda x: x["jaccard"], reverse=True)

    # Add rank
    for i, item in enumerate(scored, start=1):
        item["rank"] = i

    return scored
```

**Retrieval in prompt:**
```python
# In initial model prompt generation
kb_entries = load_kb_entries()  # From artifacts/kb/*.json
query_tags = {"tabular", "classification", "small-dataset"}
ranked_entries = rank_kb_entries(query_tags, kb_entries)

# Use top 3
top3 = ranked_entries[:3]
for item in top3:
    print(f"[KB {item['rank']}] Jaccard: {item['jaccard']:.2f}")
    print(f"  What worked: {item['entry']['why_it_worked']['hypothesis']}")
```

### 6.3 Improvement Record Template

**Codex prompt includes:**
```markdown
## Retrieved Knowledge (What Worked Before)

### KB Entry #1 (Jaccard: 0.85)
**Competition:** titanic (similar tags: tabular, classification)
**Improvement:** 0.93 → 0.9567 (+0.0267) in 3 iterations
**Top1 at time:** 0.95 (met heuristic)

**What Changed:**
- Added features: `age_pclass_interaction`, `cabin_null_flag`
- Tuned C: 1.0 → 0.5

**Why It Worked:**
- Hypothesis: Age and Pclass interaction captures survival patterns
- Evidence: Feature importance increased 0.15 → 0.23
- Previous errors: 60% misclassified young 3rd class passengers
- Rationale: Interaction directly addresses error category

**Takeaway:** Look for class-specific patterns in age/fare distributions
```

### 6.4 Capture Trigger

**When to create KB entry:**
1. Autopilot completes (5 iterations or early submit)
2. Improvement observed (any `delta > 0` for maximize, `delta < 0` for minimize)
3. Diagnostics.md exists (required for why_it_worked)

**Manual KB creation:**
```bash
uv run kagglebot kb create \
  --slug titanic \
  --run-id 20260102_113000 \
  --tags tabular,classification,small-dataset
```

**Auto KB creation (at autopilot end):**
```python
if improvement_observed and diagnostics_exists:
    create_kb_entry(
        slug=slug,
        run_id=run_id,
        tags=auto_infer_tags(slug, meta),
    )
```

---

## 7. Codex Prompt Templates

### 7.1 Initial model Prompt

**File:** `artifacts/<slug>/prompts/codex_autopilot_initial.md`

**Length:** ~250 lines

**Structure:**
1. Task description
2. Competition info (rules, Top1, target)
3. Retrieved KB entries (top 3)
4. Dataset profile
5. Constraints and requirements
6. Acceptance criteria
7. Template variables

**Template:**

```markdown
# Task: Build Initial model Model (Autopilot Iteration 1)

You are Codex, a coding agent tasked with building a initial model machine learning model for a Kaggle competition.

## Competition: {{competition_slug}}

**Objective:** {{objective_summary}}
**Evaluation Metric:** {{target_metric}} ({{direction}})
**Kaggle Top1 Public Score:** {{top1_score}}
**Target:** Within {{margin_abs_display}} or {{margin_rel_display}} of Top1

## Retrieved Knowledge (What Worked Before)

{{#if kb_entries}}
{{#each kb_entries}}
### KB Entry #{{rank}} (Jaccard: {{jaccard}})

**Competition:** {{entry.competition_slug}} (similar tags: {{entry.tags}})
**Improvement:** {{entry.improvement_record.offline_score_before}} → {{entry.improvement_record.offline_score_after}} (+{{entry.improvement_record.delta}})
**Top1 at time:** {{entry.improvement_record.top1_score}} ({{#if entry.improvement_record.meets_heuristic}}met heuristic{{else}}did not meet{{/if}})

**What Changed:**
- Features added: {{entry.what_changed.features_added}}
- Features removed: {{entry.what_changed.features_removed}}
- Hyperparameters: {{json entry.what_changed.hyperparameters_changed}}

**Why It Worked:**
- Hypothesis: {{entry.why_it_worked.hypothesis}}
- Evidence: {{entry.why_it_worked.evidence}}
- Previous errors: {{entry.why_it_worked.diagnostics_summary}}
- Rationale: {{entry.why_it_worked.rationale}}

**Takeaway:** {{entry.why_it_worked.takeaway}}

{{/each}}
{{else}}
No previous knowledge base entries found. You're starting fresh!
{{/if}}

## Dataset Profile

**Location:** `{{data_dir}}`

**Files:**
- train.csv: {{n_train}} rows, {{n_train_cols}} columns
- test.csv: {{n_test}} rows, {{n_test_cols}} columns
- sample_submission.csv: {{n_sample}} rows, {{sample_cols}} columns

**Target Column:** {{target_column}} (inferred from: {{target_inference_method}})

**Feature Summary:**
{{feature_summary}}

**Missing Data:**
{{missing_summary}}

## Rules and Constraints

{{competition_rules}}

**Code Requirements:**
- Read data from `{{data_dir}}/train.csv` and `{{data_dir}}/test.csv`
- Output submission to `/kaggle/working/submission.csv` (Kaggle kernel paths)
- Use scikit-learn or similar (no deep learning for initial model)
- Set random_seed={{random_seed}} for reproducibility
- Handle missing data appropriately
- Match sample_submission.csv format exactly

**Offline Evaluation:**
- Source: {{score_source}} ({{#if is_holdout}}{{holdout_fraction}} holdout{{else}}{{cv_folds}}-fold CV{{/if}})
- Metric: {{target_metric}}
- Report score in metrics.json

**Submission Criteria:**
- Offline score must meet Top1 heuristic (see target above)
- Columns match sample_submission.csv
- Row count matches test.csv
- No missing values in submission

## Acceptance Criteria

Your initial model is acceptable if:
1. ✅ Code runs without errors
2. ✅ Produces valid submission.csv
3. ✅ Offline score reported in metrics.json
4. ✅ Submission validates against sample_submission.csv
5. ✅ Offline score within {{margin_abs}} (abs) OR {{margin_rel}} (rel) of Top1

**Success = Submission created, ready for Kaggle upload**

## Output Requirements

1. **Code:** Write `train.py` in `/kaggle/working/`
2. **Metrics:** Write `metrics.json` with schema:
   ```json
   {
     "iteration": 1,
     "model": "LogisticRegression",
     "features": ["Pclass", "Sex", "Age"],
     "offline_score": {
       "source": "{{score_source}}",
       "metric": "{{target_metric}}",
       "value": 0.XXX,
       "std": null,
       "n_train": XXX,
       "n_eval": XXX,
       "random_seed": {{random_seed}}
     }
   }
   ```
3. **Submission:** Write `submission.csv` matching sample_submission.csv

## Template Variables

- `{{competition_slug}}`: Competition slug (e.g., "titanic")
- `{{objective_summary}}`: One-sentence objective (from rules or meta.json)
- `{{target_metric}}`: Metric name (e.g., "accuracy")
- `{{direction}}`: "maximize" or "minimize"
- `{{top1_score}}`: Kaggle public Top1 score (float)
- `{{margin_abs}}`: Absolute margin (float)
- `{{margin_rel}}`: Relative margin (float, e.g., 0.02 = 2%)
- `{{margin_abs_display}}`: Human-readable (e.g., "0.05 points")
- `{{margin_rel_display}}`: Human-readable (e.g., "2%")
- `{{kb_entries}}`: List of retrieved KB entries (top 3)
- `{{data_dir}}`: Path to data directory
- `{{n_train}}`, `{{n_test}}`, `{{n_sample}}`: Row counts
- `{{target_column}}`: Target column name
- `{{feature_summary}}`: Feature types and distributions
- `{{missing_summary}}`: Missing data summary
- `{{competition_rules}}`: Rules summary (from rules.txt or URL)
- `{{random_seed}}`: Random seed (default: 42)
- `{{score_source}}`: "holdout" or "cv"
- `{{holdout_fraction}}`: Holdout fraction (e.g., 0.2)
- `{{cv_folds}}`: Number of CV folds (e.g., 5)

## Notes

- This is iteration 1 (initial model). Keep it simple.
- Focus on data quality (missing values, encoding) over complex models.
- Use LogisticRegression or RandomForest for tabular data.
- Learn from KB entries but don't blindly copy (different competitions may differ).
- If offline score doesn't meet heuristic, that's okay—diagnostics will guide iteration 2.

Good luck!
```

### 7.2 Improvement Prompt

**File:** `artifacts/<slug>/prompts/codex_autopilot_improvement.md`

**Length:** ~200 lines

**Structure:**
1. Task description (iteration N)
2. Previous results (iteration N-1)
3. Diagnostics summary
4. Retrieved KB (similar improvements)
5. Overfitting warning
6. Acceptance criteria
7. Template variables

**Template:**

```markdown
# Task: Improve Model (Autopilot Iteration {{iteration}})

You are Codex, tasked with improving the model from iteration {{prev_iteration}}.

## Competition: {{competition_slug}}

**Target:** {{target_metric}} ({{direction}})
**Kaggle Top1:** {{top1_score}}
**Target heuristic:** Within {{margin_abs}} or {{margin_rel}} of Top1

## Previous Results (Iteration {{prev_iteration}})

**Offline Score:** {{prev_offline_score}} ({{score_source}})
**Meets Heuristic:** {{prev_meets_heuristic}}
**Gap to Top1:** {{gap_to_top1}} ({{gap_direction}})

**Model:**
- Type: {{prev_model}}
- Features: {{prev_features}}
- Hyperparameters: {{prev_hyperparameters}}

**Training Time:** {{prev_training_time}} seconds

## Diagnostics Summary

{{diagnostics_summary}}

**Key Issues:**
{{#each diagnostics_issues}}
- {{this}}
{{/each}}

**Error Analysis:**
{{error_analysis}}

**Feature Importance (Top 5):**
{{#each top_features}}
{{rank}}. {{name}}: {{importance}}
{{/each}}

**Recommendations from Diagnostics:**
{{diagnostics_recommendations}}

## Retrieved Knowledge (Similar Improvements)

{{#if kb_similar}}
{{#each kb_similar}}
### KB Entry #{{rank}} (Jaccard: {{jaccard}})

**Problem:** {{entry.why_it_worked.diagnostics_summary}}
**Solution:** {{entry.what_changed.features_added}}
**Result:** {{entry.improvement_record.offline_score_before}} → {{entry.improvement_record.offline_score_after}} (+{{entry.improvement_record.delta}})
**Why:** {{entry.why_it_worked.hypothesis}}

{{/each}}
{{else}}
No similar improvement patterns found in KB.
{{/if}}

## Overfitting Warning

⚠️ **Important:** You are evaluating on {{score_source}}.

{{#if is_holdout}}
**Holdout Risk:** Avoid overfitting to the {{holdout_fraction}} holdout set.
- Don't tune hyperparameters excessively based on holdout score.
- If offline score becomes unstable (iteration-to-iteration variance > 0.01), switch to CV.
{{/if}}

{{#if is_cv}}
**CV Risk:** {{cv_folds}}-fold CV is more robust but slower.
- Monitor train/val gap (if train >> val, you're overfitting).
- Regularize (increase C for LogisticRegression, max_depth for trees).
{{/if}}

**General:**
- Prioritize features that generalize (based on domain knowledge).
- Avoid creating too many interaction features (leads to overfitting).
- Keep model complexity in check (fewer features is better if score is similar).

## Constraints

**Code:**
- Modify `train.py` from iteration {{prev_iteration}}
- Output to `/kaggle/working/submission.csv`
- Update `metrics.json` with iteration {{iteration}}

**Evaluation:**
- Same {{score_source}} as iteration {{prev_iteration}}
- Same random_seed={{random_seed}}

**Acceptance Criteria:**
1. ✅ Offline score improves OR meets heuristic
2. ✅ Valid submission.csv
3. ✅ Metrics.json updated
4. ✅ Training completes in <5 minutes (avoid excessive grid search)

## Output Requirements

1. **Code:** Updated `train.py`
2. **Metrics:** Updated `metrics.json` (increment iteration)
3. **Submission:** Updated `submission.csv`
4. **Diagnostics:** Updated `diagnostics.md` (if error patterns change)

## Suggested Improvements (Based on Diagnostics)

{{suggested_improvements}}

## Template Variables

- `{{iteration}}`: Current iteration (2-5)
- `{{prev_iteration}}`: Previous iteration (1-4)
- `{{prev_offline_score}}`: Previous offline score
- `{{prev_meets_heuristic}}`: True/False
- `{{gap_to_top1}}`: Absolute gap to Top1
- `{{gap_direction}}`: "above" or "below"
- `{{diagnostics_summary}}`: Summary from diagnostics.md
- `{{diagnostics_issues}}`: List of key issues
- `{{error_analysis}}`: Error breakdown
- `{{top_features}}`: Top 5 feature importances
- `{{diagnostics_recommendations}}`: Recommendations from diagnostics
- `{{kb_similar}}`: KB entries with similar improvement patterns
- `{{is_holdout}}`, `{{is_cv}}`: Booleans for score source
- `{{suggested_improvements}}`: Auto-generated improvement suggestions

## Notes

- Focus on addressing the largest error categories first.
- Don't change everything at once—isolate improvements.
- If offline score plateaus (no improvement in 2 iterations), consider:
  - Switching models (LogisticRegression → RandomForest)
  - Switching score source (holdout → CV for stability)
  - Simplifying (remove low-importance features)

Good luck with iteration {{iteration}}!
```

### 7.3 Template Variable Population

**Initial model template variables (35+):**
```python
initial_vars = {
    # Competition info
    "competition_slug": slug,
    "objective_summary": meta["objective"],
    "target_metric": config.target_metric,
    "direction": config.direction,
    "top1_score": top1_info["score"],
    "margin_abs": config.margin_abs,
    "margin_rel": config.margin_rel,
    "margin_abs_display": f"{config.margin_abs} points",
    "margin_rel_display": f"{config.margin_rel * 100:.0f}%",

    # KB entries
    "kb_entries": ranked_kb_entries[:3],

    # Dataset
    "data_dir": str(paths.data_dir),
    "n_train": len(train_df),
    "n_test": len(test_df),
    "n_sample": len(sample_df),
    "n_train_cols": len(train_df.columns),
    "n_test_cols": len(test_df.columns),
    "sample_cols": list(sample_df.columns),
    "target_column": infer_target_column(...),
    "target_inference_method": "sample_submission",
    "feature_summary": generate_feature_summary(train_df),
    "missing_summary": generate_missing_summary(train_df),

    # Rules
    "competition_rules": load_rules(paths.context_dir / "rules.txt"),

    # Evaluation
    "random_seed": config.seed,
    "score_source": config.score_source,
    "is_holdout": config.score_source == "holdout",
    "is_cv": config.score_source == "cv",
    "holdout_fraction": 0.2 if config.score_source == "holdout" else None,
    "cv_folds": 5 if config.score_source == "cv" else None,
}
```

**Improvement template variables (25+):**
```python
improvement_vars = {
    # Iteration
    "iteration": iteration,
    "prev_iteration": iteration - 1,

    # Previous results
    "prev_offline_score": prev_metrics["offline_score"]["value"],
    "prev_meets_heuristic": prev_metrics["top1_comparison"]["offline_meets_heuristic"],
    "gap_to_top1": abs(prev_metrics["offline_score"]["value"] - top1_score),
    "gap_direction": "above" if prev_score > top1_score else "below",
    "prev_model": prev_metrics["model"],
    "prev_features": prev_metrics["features"],
    "prev_hyperparameters": prev_metrics["hyperparameters"],
    "prev_training_time": prev_metrics["training"]["duration_seconds"],

    # Diagnostics
    "diagnostics_summary": load_diagnostics_summary(...),
    "diagnostics_issues": extract_issues(...),
    "error_analysis": extract_error_analysis(...),
    "top_features": extract_top_features(...),
    "diagnostics_recommendations": extract_recommendations(...),

    # KB
    "kb_similar": ranked_kb_entries[:3],

    # Overfitting warnings
    "is_holdout": config.score_source == "holdout",
    "is_cv": config.score_source == "cv",
    "holdout_fraction": 0.2,
    "cv_folds": 5,

    # Suggestions
    "suggested_improvements": generate_suggestions(diagnostics, prev_metrics),
}
```

---

## 8. Implementation Phases

### Phase A: Core Autopilot Loop

**A1: Autopilot State Management**
- [x] Create `src/kagglebot/autopilot.py`
- [ ] Implement `AutopilotState` class
- [ ] Add `MAX_SUBMISSIONS = 1` enforcement
- [ ] Add `MAX_ITERATIONS = 5` hard-coded
- [ ] Add iteration history tracking

**A2: Offline Evaluation**
- [ ] Extend `src/kagglebot/solver/initial_model.py`
- [ ] Add holdout evaluation (train_test_split)
- [ ] Add CV evaluation (cross_val_score)
- [ ] Auto-select source based on dataset size
- [ ] Update metrics.json with offline_score block

**A3: Top1 Heuristic**
- [ ] Implement `fetch_top1_score()` in `src/kagglebot/leaderboard.py`
- [ ] Add CSV parsing with comma handling
- [ ] Add 60-minute caching (`leaderboard_cache.json`)
- [ ] Implement `evaluate_top1_heuristic()`
- [ ] Implement `infer_direction()` from metric name
- [ ] Update metrics.json with top1_comparison block

**A4: Autopilot Loop**
- [ ] Implement main autopilot loop (5 iterations)
- [ ] Add early-submit trigger (`--early-submit` flag)
- [ ] Add final-submit trigger (default, `--no-final-submit` to disable)
- [ ] Add best-iteration selection (`select_best_iteration()`)
- [ ] Update metrics.json with chosen_best block

### Phase B: Guardrails

**B1: Max Submissions**
- [ ] Integrate `AutopilotState.submit()` with existing `submit_competition()`
- [ ] Raise `MaxSubmissionsError` if already submitted
- [ ] Test: Attempt second submission (should fail)

**B2: Enhanced Deduplication**
- [ ] Add `autopilot_run_id` to ledger schema
- [ ] Update `SubmissionLedger.record()` to accept autopilot_run_id
- [ ] Warn (don't error) on duplicate within same autopilot run
- [ ] Error on duplicate across different runs/manual

**B3: Time Cap**
- [ ] Add `--max-time` flag (default: 480 minutes)
- [ ] Track elapsed time in loop
- [ ] Raise `TimeoutError` if exceeded (exit code 11)

**B4: Consecutive Failure Cap**
- [ ] Track consecutive training failures
- [ ] Abort after 3 consecutive (exit code 5)

**B5: Secret Scanning**
- [ ] Implement `scan_for_secrets()` with regex patterns
- [ ] Add pre-submission check
- [ ] Raise `SecurityError` if secrets detected (exit code 13)

### Phase C: Knowledge Base

**C1: Schema Extension**
- [ ] Add `improvement_record` to KB entry schema
- [ ] Add `what_changed` and `why_it_worked` blocks
- [ ] Update `src/kagglebot/kb.py` (create if needed)

**C2: Retrieval**
- [ ] Implement `jaccard_similarity()`
- [ ] Implement `rank_kb_entries()`
- [ ] Test retrieval with sample KB entries

**C3: Capture**
- [ ] Implement `create_kb_entry()` (manual)
- [ ] Auto-create KB entry at autopilot end (if improvement observed)
- [ ] Store in `artifacts/<slug>/kb/<run_id>.json`

### Phase D: Prompt Templates

**D1: Initial model Prompt**
- [ ] Create `codex_autopilot_initial.md` template
- [ ] Implement template variable population
- [ ] Integrate KB retrieval (top 3 entries)
- [ ] Test rendering with real competition data

**D2: Improvement Prompt**
- [ ] Create `codex_autopilot_improvement.md` template
- [ ] Implement diagnostics parsing
- [ ] Implement suggested improvements generation
- [ ] Test rendering with iteration 2+ data

**D3: Diagnostics Generation**
- [ ] Extend `train_and_predict()` to generate diagnostics.md
- [ ] Include error analysis, feature importance, recommendations
- [ ] Store in `artifacts/<slug>/runs/<run_id>/diagnostics.md`

### Phase E: CLI Integration

**E1: Extend `run` Command**
- [ ] Add `--autopilot` flag
- [ ] Add `--target-metric`, `--target-direction`, etc.
- [ ] Route to `run_autopilot()` if `--autopilot` set

**E2: Autopilot Entrypoint**
- [ ] Implement `run_autopilot()` in `src/kagglebot/autopilot.py`
- [ ] Call bootstrap, fetch Top1, run loop
- [ ] Handle early/final submit logic

**E3: Testing**
- [ ] Add `tests/test_autopilot.py`
- [ ] Test single iteration (no submit)
- [ ] Test early submit (heuristic met at iteration 2)
- [ ] Test final submit (best of 5)
- [ ] Test max submissions enforcement
- [ ] Test time cap

### Phase F: Documentation

**F1: Update README**
- [ ] Add autopilot section with example CLI
- [ ] Link to `docs/AUTOPILOT_SINGLE_SUBMIT.md`

**F2: Update CLAUDE.md**
- [ ] Add autopilot design principles
- [ ] Add single-submit policy

**F3: Create User Guide**
- [ ] Create `docs/AUTOPILOT_USER_GUIDE.md`
- [ ] Explain Top1 heuristic with examples
- [ ] Explain early vs final submit
- [ ] Explain Knowledge Base usage

---

## Appendix A: Exit Codes

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | Autopilot completed, submitted |
| 1 | General error | Invalid flags, missing files |
| 2 | Rules not accepted | Must accept rules in browser |
| 3 | Invalid competition | Slug malformed |
| 5 | Training error | Model crashed, OOM |
| 6 | Validation error | Submission format mismatch |
| 7 | Missing submission | submission.csv not generated |
| 8 | Network error | Kaggle API timeout |
| 11 | Timeout | Exceeded `--max-time` |
| 12 | Kernel failed | Kaggle kernel error state |
| 13 | Security error | Secrets detected in submission |

---

## Appendix B: Example CLI Sessions

### B1: Basic Autopilot (5 iterations, final submit)

```bash
$ uv run kagglebot run titanic \
  --agent codex \
  --compute local_cpu \
  --autopilot \
  --target-metric accuracy \
  --force

[Bootstrap] Creating artifacts/titanic...
[Leaderboard] Top1: 0.95 by user123
[Iteration 1/5] Training initial model...
[Offline] Holdout accuracy: 0.9567
[Heuristic] 0.9567 >= 0.931 (rel threshold) ✓
[Iteration 2/5] Generating improvement prompt...
[Offline] Holdout accuracy: 0.9601
[Iteration 3/5] ...
[Iteration 4/5] ...
[Iteration 5/5] ...
[Best] Iteration 3 (accuracy: 0.9601)
[Submitting] artifacts/titanic/submissions/20260102_113000_submission.csv
[Success] Submission recorded
```

### B2: Early Submit

```bash
$ uv run kagglebot run titanic \
  --autopilot \
  --early-submit \
  --force

[Iteration 1/5] Offline: 0.93 (does not meet heuristic)
[Iteration 2/5] Offline: 0.9567 (meets heuristic) ✓
[Early Submit] Submitting iteration 2...
[Success] Autopilot stopped early (2/5 iterations)
```

### B3: No Final Submit (Exploration)

```bash
$ uv run kagglebot run titanic \
  --autopilot \
  --no-final-submit

[Iteration 1/5] Offline: 0.93
[Iteration 2/5] Offline: 0.9567
[Iteration 3/5] Offline: 0.9601
[Iteration 4/5] Offline: 0.9589
[Iteration 5/5] Offline: 0.9612
[Best] Iteration 5 (accuracy: 0.9612)
[No Submit] Exploration mode (use --submit for final submission)
```

---

## Appendix C: Open Questions

1. **Knowledge Base Location:** Store KB in artifacts/<slug>/kb/ or global ~/.kagglebot/kb/?
   - **Recommendation:** Per-competition (artifacts/<slug>/kb/) for privacy, optional sync to global

2. **Diagnostics Automation:** How detailed should auto-generated diagnostics be?
   - **Recommendation:** Start simple (error analysis by prediction quantiles), add complexity later

3. **Hyperparameter Tuning:** Should autopilot support grid search?
   - **Recommendation:** No for MVP (manual tuning in improvement prompts), consider for Phase 2

4. **Multi-Model Ensembling:** Should autopilot try multiple models?
   - **Recommendation:** No for MVP (single model per iteration), user can manually ensemble

5. **Leaderboard Privacy:** Should we cache team names?
   - **Recommendation:** Yes (for diagnostics), but sanitize in logs (no PII)

---

## Appendix D: Recommended Workflow

**Day 1: Setup**
```bash
# Bootstrap competition
uv run kagglebot bootstrap titanic --download --force
```

**Day 2: First Autopilot Run**
```bash
# Run autopilot (5 iterations, final submit)
uv run kagglebot run titanic \
  --agent codex \
  --autopilot \
  --target-metric accuracy \
  --force
```

**Day 3: Review KB and Manual Improvements**
```bash
# Check KB entries
cat artifacts/titanic/kb/*.json

# Manual improvements (if needed)
# Edit src/titanic_train.py
# Test locally
uv run python src/titanic_train.py

# Submit manually
uv run kagglebot submit titanic \
  -f artifacts/titanic/submissions/manual_submission.csv \
  -m "manual feature engineering" \
  --force
```

**Day 4+: Iterate**
```bash
# Another autopilot run (different approach)
uv run kagglebot run titanic \
  --autopilot \
  --score-source cv \
  --early-submit \
  --force
```

---

**End of Specification**

**Summary:**
- ✅ Autopilot policy defined (single-submit, max 5 iterations)
- ✅ Compare rule defined (both margins, AND condition, defaults)
- ✅ Top1 fetch specified (CSV download, 60-min cache)
- ✅ Metrics.json contract updated (3 new blocks)
- ✅ Guardrails specified (MAX_SUBMISSIONS=1, dedup, time cap, secrets)
- ✅ KB integration specified (schema, retrieval, capture)
- ✅ Prompt templates created (initial model + improvement)
- ✅ Implementation phases outlined (A-F, ~40 tasks)

**Ready for implementation by Codex or manual coding.**
