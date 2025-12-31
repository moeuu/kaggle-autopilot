# Specification

## CLI Commands

### `kagglebot run`

**Purpose**: Execute full automated pipeline for a competition.

**Synopsis**:
```bash
kagglebot run <competition> [OPTIONS]
```

**Arguments**:
- `competition` (required): Kaggle competition URL or slug
  - Accepted formats:
    - `https://www.kaggle.com/competitions/titanic`
    - `https://www.kaggle.com/c/titanic`
    - `titanic` (slug only)

**Options**:
- `--submit / --no-submit`: Submit predictions to Kaggle after validation (default: `--no-submit`)
- `--time-budget MINUTES`: Maximum training time in minutes (default: `60`)
- `--config PATH`: Path to custom config file (default: `config/<slug>.toml`)
- `--resume RUN_ID`: Resume from a previous run checkpoint
- `--models MODEL1,MODEL2`: Comma-separated list of models to train (default: from config)
- `--cv-folds N`: Number of cross-validation folds (default: `5`)
- `--no-stacking`: Disable ensemble stacking
- `--message TEXT`: Submission message (default: auto-generated)
- `--force`: Skip safety checks (duplicate detection, rate limits) - **use with caution**
- `--dry-run`: Execute pipeline but skip actual submission

**Examples**:
```bash
# Analyze, train, predict, validate (no submission)
kagglebot run titanic

# Full pipeline with submission
kagglebot run https://www.kaggle.com/c/titanic --submit

# Custom time budget and models
kagglebot run titanic --time-budget 120 --models lgbm,catboost --submit

# Resume interrupted run
kagglebot run titanic --resume 7f8e9d2a --submit

# Dry run to test pipeline
kagglebot run titanic --submit --dry-run
```

**Exit Codes**:
- `0`: Success (pipeline completed, submission valid)
- `1`: General failure (unhandled exception)
- `2`: Rules not accepted - **user must join competition in browser**
- `3`: Invalid competition URL or slug
- `4`: Data download failed (network, permissions)
- `5`: Training failed (errors during model training)
- `6`: Submission validation failed (format mismatch)
- `7`: Submission upload failed (API error, network)
- `8`: Duplicate submission detected (already in ledger)
- `9`: Rate limit exceeded (too many recent submissions)

### `kagglebot analyze`

**Purpose**: Analyze competition without training (useful for planning).

**Synopsis**:
```bash
kagglebot analyze <competition> [OPTIONS]
```

**Options**:
- `--output PATH`: Save analysis to JSON file (default: stdout)

**Example**:
```bash
kagglebot analyze titanic --output titanic_analysis.json
```

**Output** (JSON):
```json
{
  "slug": "titanic",
  "type": "tabular",
  "task": "classification",
  "metric": "accuracy",
  "schema": {
    "train_rows": 891,
    "test_rows": 418,
    "id_column": "PassengerId",
    "target_columns": ["Survived"],
    "feature_columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
  },
  "strategy": {
    "preprocessing": ["impute_median", "onehot_categorical"],
    "models": ["ridge", "lgbm", "catboost"],
    "cv_folds": 5,
    "use_stacking": true
  }
}
```

### `kagglebot list-runs`

**Purpose**: List all runs for a competition.

**Synopsis**:
```bash
kagglebot list-runs <competition> [OPTIONS]
```

**Options**:
- `--limit N`: Show only last N runs (default: `10`)
- `--format {table,json}`: Output format (default: `table`)

**Example**:
```bash
kagglebot list-runs titanic --limit 5
```

### `kagglebot show-run`

**Purpose**: Display details of a specific run.

**Synopsis**:
```bash
kagglebot show-run <competition> <run_id>
```

**Example**:
```bash
kagglebot show-run titanic 7f8e9d2a
```

### `kagglebot list-submissions`

**Purpose**: Show submission history from local ledger.

**Synopsis**:
```bash
kagglebot list-submissions <competition> [OPTIONS]
```

**Options**:
- `--limit N`: Show only last N submissions (default: `20`)

**Example**:
```bash
kagglebot list-submissions titanic
```

### `kagglebot clean`

**Purpose**: Clean up old artifacts and cache.

**Synopsis**:
```bash
kagglebot clean <competition> [OPTIONS]
```

**Options**:
- `--keep-runs N`: Keep last N runs (default: `5`)
- `--keep-days N`: Keep runs from last N days (default: `30`)
- `--dry-run`: Show what would be deleted without deleting

**Example**:
```bash
kagglebot clean titanic --keep-runs 3
```

---

## Configuration

### Config File Format

Configs are TOML files. Three levels:
1. **Global**: `config/default.toml` (shipped with package)
2. **Competition**: `config/<slug>.toml` (user-created overrides)
3. **CLI flags**: Override everything

### Schema (`config/default.toml`)

```toml
[paths]
data_root = "data"
artifacts_root = "artifacts"
cache_root = ".cache"

[training]
default_time_budget_minutes = 60
cv_folds = 5
cv_shuffle = true
random_seed = 42
n_jobs = -1  # CPU cores for parallel training (-1 = all)

# Early stopping for GBDT models
early_stopping_rounds = 50

[training.tabular]
# Models to train (in order)
models = ["ridge", "lgbm", "catboost"]

# Stacking ensemble
use_stacking = true
stacking_meta_model = "ridge"

# Preprocessing
impute_strategy_numeric = "median"
impute_strategy_categorical = "most_frequent"
scale_features = false  # Usually not needed for tree models

# Feature engineering
create_interactions = false  # Only for small feature sets
polynomial_degree = 1  # Set to 2 for polynomial features

[training.text]
# Future: Text competition settings
models = []
max_features = 10000
vectorizer = "tfidf"

[training.image]
# Future: Image competition settings
models = []
image_size = [224, 224]
augmentation = true

[submission]
# Safety limits
max_submissions_per_day = 5
min_hours_between_submissions = 1.0

# Deduplication
check_duplicates = true
hash_algorithm = "sha256"

# Validation
strict_validation = true  # Fail on any schema mismatch

[submission.rate_limit]
# Exponential backoff if rate-limited
initial_retry_seconds = 60
max_retry_seconds = 3600
backoff_multiplier = 2.0

[logging]
level = "INFO"  # DEBUG, INFO, WARNING, ERROR
format = "json"  # json or text
log_to_file = true
log_to_console = true

[logging.telemetry]
# Optional: Send anonymous usage stats
enabled = false
endpoint = ""

[resources]
# Resource limits for training
max_memory_gb = 16
max_training_time_minutes = 240
gpu_enabled = false  # Use GPU if available
```

### Competition-Specific Config

Create `config/titanic.toml` to override defaults:

```toml
[training]
cv_folds = 10  # More folds for small dataset

[training.tabular]
models = ["logreg", "lgbm"]  # Skip CatBoost
use_stacking = false  # Simple dataset, no stacking needed

[submission]
max_submissions_per_day = 10  # Playground competition, more lenient
```

---

## Artifact Layout

### Directory Structure

```
artifacts/
└── <slug>/
    ├── runs/
    │   └── <run_id>/
    │       ├── metadata.json          # Run metadata
    │       ├── config.json             # Config snapshot
    │       ├── pipeline_state.json     # Resumable state
    │       ├── competition_analysis.json
    │       ├── logs/
    │       │   ├── pipeline.log        # Main log
    │       │   ├── training.log        # Training details
    │       │   └── errors.log          # Errors only
    │       ├── models/
    │       │   ├── preprocessor.pkl    # Feature transformer
    │       │   ├── ridge/
    │       │   │   ├── fold_0.pkl
    │       │   │   ├── fold_1.pkl
    │       │   │   ├── ...
    │       │   │   └── final.pkl       # Trained on full data
    │       │   ├── lgbm/
    │       │   │   └── ...
    │       │   ├── catboost/
    │       │   │   └── ...
    │       │   └── stacked/
    │       │       └── final.pkl       # Ensemble
    │       ├── cv_results.json         # Cross-validation scores
    │       ├── predictions/
    │       │   ├── ridge_train.npy     # OOF predictions for stacking
    │       │   ├── ridge_test.npy
    │       │   ├── lgbm_train.npy
    │       │   ├── lgbm_test.npy
    │       │   └── ...
    │       └── submission.csv          # Final submission
    └── submissions/
        └── ledger.jsonl                # All submissions log
```

### Metadata Files

#### `metadata.json`

```json
{
  "run_id": "7f8e9d2a-1b3c-4d5e-6f7a-8b9c0d1e2f3a",
  "slug": "titanic",
  "started_at": "2026-01-01T12:00:00Z",
  "completed_at": "2026-01-01T12:45:30Z",
  "duration_seconds": 2730,
  "success": true,
  "submitted": true,
  "kaggle_submission_id": "12345678",
  "command": "kagglebot run titanic --submit",
  "user": "eiji",
  "hostname": "macbook",
  "git_commit": "a1b2c3d",
  "python_version": "3.13.0",
  "kagglebot_version": "0.2.0"
}
```

#### `competition_analysis.json`

```json
{
  "slug": "titanic",
  "type": "tabular",
  "task": "classification",
  "metric": "accuracy",
  "metric_direction": "maximize",
  "schema": {
    "train_path": "data/titanic/raw/train.csv",
    "train_rows": 891,
    "train_columns": 12,
    "test_path": "data/titanic/raw/test.csv",
    "test_rows": 418,
    "test_columns": 11,
    "id_column": "PassengerId",
    "target_columns": ["Survived"],
    "feature_columns": ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
    "numeric_columns": ["Age", "SibSp", "Parch", "Fare"],
    "categorical_columns": ["Pclass", "Sex", "Ticket", "Cabin", "Embarked"],
    "datetime_columns": []
  },
  "constraints": {
    "allows_external_data": false,
    "allows_pretrained_models": false,
    "allows_manual_labels": false
  },
  "strategy": {
    "preprocessing": ["impute_median_numeric", "impute_mode_categorical", "onehot_encode"],
    "models": ["ridge", "lgbm", "catboost"],
    "cv_folds": 5,
    "use_stacking": true,
    "time_budget_minutes": 60
  }
}
```

#### `cv_results.json`

```json
{
  "metric": "accuracy",
  "cv_folds": 5,
  "models": {
    "ridge": {
      "fold_scores": [0.80, 0.82, 0.79, 0.81, 0.80],
      "mean": 0.804,
      "std": 0.010,
      "training_time_seconds": 1.2
    },
    "lgbm": {
      "fold_scores": [0.83, 0.85, 0.82, 0.84, 0.83],
      "mean": 0.834,
      "std": 0.011,
      "training_time_seconds": 5.8
    },
    "catboost": {
      "fold_scores": [0.84, 0.86, 0.83, 0.85, 0.84],
      "mean": 0.844,
      "std": 0.011,
      "training_time_seconds": 12.3
    },
    "stacked": {
      "fold_scores": [0.85, 0.87, 0.84, 0.86, 0.85],
      "mean": 0.854,
      "std": 0.011,
      "training_time_seconds": 0.5
    }
  },
  "best_model": "stacked",
  "best_score": 0.854
}
```

#### `ledger.jsonl` (one JSON per line)

```jsonl
{"timestamp": "2026-01-01T12:45:30Z", "run_id": "7f8e9d2a", "submission_hash": "a1b2c3d4e5f6", "file_path": "artifacts/titanic/runs/7f8e9d2a/submission.csv", "message": "Auto-generated: stacked ensemble (CV: 0.854)", "kaggle_submission_id": "12345678", "kaggle_score": 0.77511, "submitted_at": "2026-01-01T12:45:35Z"}
{"timestamp": "2026-01-01T14:20:10Z", "run_id": "8g9f0e1b", "submission_hash": "b2c3d4e5f6g7", "file_path": "artifacts/titanic/runs/8g9f0e1b/submission.csv", "message": "Auto-generated: lgbm only (CV: 0.834)", "kaggle_submission_id": "12345679", "kaggle_score": 0.76076, "submitted_at": "2026-01-01T14:20:15Z"}
```

---

## Error Messages and Exit Codes

### Exit Code 2: Rules Not Accepted

**Message**:
```
❌ Competition rules not accepted

You must manually join this competition:
  https://www.kaggle.com/competitions/titanic/rules

Steps:
  1. Visit the URL above
  2. Click "I Understand and Accept"
  3. Re-run: kagglebot run titanic --submit

Note: This is required once per competition. Kagglebot cannot automate
      this step due to Kaggle's Terms of Service.
```

### Exit Code 6: Submission Validation Failed

**Message**:
```
❌ Submission validation failed

Issues found:
  - Column mismatch: expected ['PassengerId', 'Survived'], got ['PassengerId', 'Prediction']
  - Row count mismatch: expected 418, got 419

Sample submission: data/titanic/raw/sample_submission.csv
Your submission:   artifacts/titanic/runs/7f8e9d2a/submission.csv

Fix the issues and re-run.
```

### Exit Code 8: Duplicate Submission

**Message**:
```
⚠️  Duplicate submission detected

This exact submission was already made:
  Hash:      a1b2c3d4e5f6
  Submitted: 2026-01-01 12:45:30
  Score:     0.77511
  Run ID:    7f8e9d2a

Skipping duplicate submission. Use --force to override (not recommended).
```

### Exit Code 9: Rate Limit Exceeded

**Message**:
```
⚠️  Rate limit exceeded

Current limits:
  - Max submissions per day: 5
  - Min hours between submissions: 1.0

Your recent submissions (last 24h):
  12:00 - Run 1a2b3c - Score: 0.775
  12:45 - Run 7f8e9d - Score: 0.777
  13:30 - Run 8g9f0e - Score: 0.780
  14:15 - Run 9h0g1f - Score: 0.783
  14:50 - Run 0i1h2g - Score: 0.785

Next submission allowed after: 2026-01-01 15:50

Wait 15 minutes or use --force to override (not recommended).
```

---

## Pipeline Stages and Checkpointing

Each stage writes a checkpoint before proceeding. This enables resume with `--resume`.

### Stage Checkpoints

1. **INIT** → `pipeline_state.json` created with `run_id`
2. **FETCHING** → Data downloaded to `data/<slug>/raw/`
3. **ANALYZING** → `competition_analysis.json` saved
4. **TRAINING** → Models saved to `models/`, `cv_results.json` written
5. **PREDICTING** → `submission.csv` generated
6. **VALIDATING** → Hash computed, validation results logged
7. **SUBMITTING** → Ledger updated with Kaggle submission ID
8. **DONE** → `metadata.json` marked as complete

### Resume Behavior

```bash
kagglebot run titanic --submit --resume 7f8e9d2a
```

- Loads `pipeline_state.json` from `artifacts/titanic/runs/7f8e9d2a/`
- Skips completed stages
- Continues from last checkpoint
- Reuses downloaded data and trained models

---

## Model Registry

### Built-in Models

| Model ID | Class | Use Case | Training Time | Typical Performance |
|----------|-------|----------|---------------|---------------------|
| `ridge` | Ridge Regression | Regression, fast baseline | Very fast | Baseline |
| `logreg` | Logistic Regression | Classification, fast baseline | Very fast | Baseline |
| `lgbm` | LightGBM | Tabular (general) | Fast | Strong |
| `catboost` | CatBoost | Tabular (categorical-heavy) | Medium | Strong |
| `xgboost` | XGBoost | Tabular (general) | Medium | Strong |
| `rf` | Random Forest | Tabular (robust) | Slow | Moderate |

### Model Selection Logic

Based on `competition_analysis.json`:

- **Task = classification**:
  - `task_type = "binary"` → `logreg`, `lgbm`, `catboost`
  - `task_type = "multiclass"` → `lgbm`, `catboost`

- **Task = regression**:
  - Default: `ridge`, `lgbm`, `catboost`

- **Dataset size**:
  - `rows < 1000` → Skip complex models, use `logreg`/`ridge` + `lgbm`
  - `rows > 100k` → Prioritize `lgbm` (faster than CatBoost)

- **Feature types**:
  - Many categorical → Prefer `catboost` (handles categories natively)
  - Mostly numeric → All GBDT models work well

- **Time budget**:
  - `< 15 min` → `ridge`/`logreg` + `lgbm` only
  - `15-60 min` → Add `catboost`
  - `> 60 min` → Full ensemble with stacking

---

## API (for programmatic use)

Users can import kagglebot as a library:

```python
from kagglebot import Pipeline, Config

# Programmatic execution
config = Config.from_file("config/titanic.toml")
pipeline = Pipeline(slug="titanic", config=config)

result = pipeline.execute(submit=True)

print(f"Run ID: {result.run_id}")
print(f"CV Score: {result.cv_score}")
print(f"Submission: {result.submission_path}")
```

This enables:
- Integration into larger workflows
- Custom orchestration
- Batch processing of multiple competitions

---

## Monitoring and Observability

### Real-time Progress

Terminal output with Rich progress bars:

```
╭─────────────────────────────────────────────────────────────╮
│ Kagglebot Pipeline: titanic                                 │
├─────────────────────────────────────────────────────────────┤
│ Run ID: 7f8e9d2a                                            │
│ Started: 2026-01-01 12:00:00                                │
╰─────────────────────────────────────────────────────────────╯

[1/7] Checking rules acceptance... ✓
[2/7] Downloading data... ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 3.2MB
[3/7] Analyzing competition... ✓ (tabular, binary classification)
[4/7] Training models...
  ├─ Ridge regression... ━━━━━━━━━━━━━━━━━━━━ 100% CV: 0.804
  ├─ LightGBM... ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% CV: 0.834
  ├─ CatBoost... ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75%  CV: 0.829 (ongoing)
  └─ Stacking ensemble... (pending)
[5/7] Generating predictions... (pending)
[6/7] Validating submission... (pending)
[7/7] Submitting to Kaggle... (pending)
```

### Logs

Structured JSON logs for parsing:

```json
{"timestamp": "2026-01-01T12:05:30Z", "level": "INFO", "stage": "training", "model": "lgbm", "fold": 0, "score": 0.835, "metric": "accuracy"}
{"timestamp": "2026-01-01T12:10:15Z", "level": "INFO", "stage": "training", "model": "lgbm", "fold": 1, "score": 0.842, "metric": "accuracy"}
```

Human-readable text logs also available (`format = "text"` in config).
