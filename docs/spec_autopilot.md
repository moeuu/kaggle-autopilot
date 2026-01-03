# Kagglebot Autopilot Specification

**Version**: 1.0
**Status**: Production
**Last Updated**: 2026-01-02

---

> Note: Git integration has been removed from the implementation. Any sections describing git stashing, branch enforcement, or diff capture are historical and no longer apply.
> Note: Any "initial model" references in this spec should be read as "strong initial model" (web‑researched, not a simplistic initial model).

## Executive Summary

Kagglebot autopilot is a fully automated Kaggle competition workflow system that implements a 5-iteration improvement loop with score-gated submission. The system operates non-interactively with minimal user arguments, leveraging a Knowledge Base for cross-competition learning and supporting multiple compute backends (local GPU, Kaggle GPU/TPU).

**Core Principle**: Accuracy-first, no training time limits, submit when Top1 heuristic met OR after 5 iterations.

---

## 1. CLI Contract

### 1.1 Minimal Required Command

```bash
uv run kagglebot autopilot <competition_url> \
  --agent codex \
  --compute <local_gpu|kaggle_gpu|kaggle_tpu> \
  --submit
```

**Required Arguments**:
- `<competition_url>`: Full Kaggle competition URL or slug
- `--agent codex`: Agent backend (only codex supported in v1)
- `--compute <mode>`: Training compute backend
- `--submit`: Enable submission (required for safety)

**No Other Flags Required**: All other parameters have safe defaults or are auto-detected.

### 1.2 Optional Flags (Advanced)

```bash
--max-iterations N        # Default: 5
--margin-abs FLOAT        # Top1 comparison absolute margin, default: 0.0
--margin-rel FLOAT        # Top1 comparison relative margin, default: 0.0
--verify-cmd "CMD"        # Test command, default: "uv run pytest -q"
--dry-run                 # Skip external API calls
--kaggle-username USER    # For kernel runner
```

### 1.3 Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Success: Submitted successfully |
| 1    | Generic error |
| 2    | Rules not accepted (user must accept manually) |
| 4    | Kaggle CLI error |
| 6    | Validation error |
| 8    | Duplicate submission |
| 9    | Rate limit exceeded |
| 10   | GPU not available |
| 11   | Kernel timeout |
| 12   | Kernel failed |
| 14   | Max submissions exceeded |

---

## 2. End-to-End Workflow

### 2.1 Preconditions

1. ✅ User has manually accepted competition rules on Kaggle website
2. ✅ Kaggle CLI installed and configured (`~/.kaggle/kaggle.json`)
3. ✅ Competition is active and accepting submissions
4. ✅ Git repository on `main` branch (auto-switched if not)

### 2.2 Workflow Steps

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: BOOTSTRAP & CONTEXT GATHERING                      │
│  - Parse competition slug from URL                          │
│  - Download competition data (train, test, sample_sub)      │
│  - Fetch rules page (best-effort, always store URL)         │
│  - Fetch public leaderboard (Top1 snapshot)                 │
│  - Profile dataset (detect task, metric, distributions)     │
│  - Query Knowledge Base for similar competitions            │
│  - Create context pack for agent                            │
│  Duration: ~2-5 minutes                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: PLAN & INITIAL MODEL (Iteration 0)                 │
│  - Send context pack to Codex agent                         │
│  - Agent decides: approach, model, features, eval strategy  │
│  - Agent implements initial model solution in kagglebot/solver/  │
│  - Agent generates plan.json with:                          │
│    * target_metric (auto-detected or inferred)              │
│    * target_score (based on Top1 + margin)                  │
│    * eval_strategy (holdout or CV)                          │
│    * compute_hints (GPU/TPU optimization notes)             │
│  - Verify tests pass: uv run pytest -q                      │
│  Duration: ~5-15 minutes (agent implementation time)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: ITERATION LOOP (Iterations 1-5)                    │
│                                                             │
│  FOR iteration IN [1..5]:                                   │
│                                                             │
│    3a) TRAIN                                                │
│        - Execute solver with specified compute backend      │
│        - If local_gpu: train_local()                        │
│        - If kaggle_gpu/tpu: run_kernel() → poll status      │
│        - Maximize GPU/TPU utilization                       │
│        - No time limit (accuracy-first)                     │
│        Duration: Variable (5 min - 2 hours+)                │
│                                                             │
│    3b) EVALUATE (Pseudo-Test)                               │
│        - Split train.csv into train/val or use CV          │
│        - Compute offline score on validation set            │
│        - NEVER use actual test.csv for evaluation           │
│        - Record metrics.json with train/val scores          │
│        Duration: <1 minute (already done during training)   │
│                                                             │
│    3c) COMPARE (Top1 Heuristic)                             │
│        - Fetch current Top1 public leaderboard score        │
│        - Apply direction-aware comparison:                  │
│          * minimize: offline <= top1*(1+margin_rel)+margin_abs│
│          * maximize: offline >= top1*(1-rel)-margin_abs     │
│        - Set met_heuristic = True/False                     │
│        Duration: <10 seconds                                │
│                                                             │
│    3d) DECISION                                             │
│        IF met_heuristic:                                    │
│          → Submit submission.csv                            │
│          → STOP (early exit)                                │
│        ELIF iteration == 5:                                 │
│          → Submit best submission.csv from all iterations   │
│          → STOP (final submit)                              │
│        ELSE:                                                │
│          → Continue to 3e                                   │
│                                                             │
│    3e) DIAGNOSE & IMPROVE                                   │
│        - Generate diagnostics.md (agent-readable analysis)  │
│        - Identify likely causes (underfit, overfit, etc)    │
│        - Send improve_iteration.md prompt to Codex          │
│        - Agent modifies solver code                         │
│        - Verify tests pass                                  │
│        - Loop back to 3a                                    │
│        Duration: ~3-10 minutes (agent time)                 │
│                                                             │
│  END FOR                                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: SUBMISSION & POSTMORTEM                             │
│  - Submit to Kaggle via CLI (with dedup + rate limit)      │
│  - Generate postmortem summary                              │
│  - Update Knowledge Base with learnings                     │
│  - Save git diff to artifacts/diffs/<run_id>.diff          │
│  - Restore git stash if created                             │
│  - Ensure on main branch                                    │
│  Duration: <1 minute                                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Stop Conditions

The loop stops when ANY of these is true:

1. ✅ **Heuristic met early** (iteration 1-4): offline score meets Top1 heuristic → submit and stop
2. ✅ **Max iterations** (iteration 5): submit best candidate and stop
3. ❌ **Error conditions**:
   - Rules not accepted → exit code 2
   - Kernel failed 3 times → exit code 12
   - Tests fail after 3 retry attempts → exit code 1

### 2.4 Submission Rule: "Top1 OR 5 Loops"

```python
for iteration in range(1, 6):  # 1..5
    train()
    evaluate()  # offline score on validation
    top1 = fetch_top1()

    if meets_heuristic(offline_score, top1, direction, margins):
        submit(submission_path)
        break  # Early exit

    if iteration == 5:
        submit(best_submission_path)  # Final submit
        break

    # Else: improve and continue
    diagnose_and_improve()
```

**Guarantees**:
- Exactly **1 submission per autopilot run**
- Submit happens when Top1 met OR after iteration 5
- No submissions if rules not accepted or errors occur

---

## 3. Artifacts Contract

### 3.1 Directory Structure

```
artifacts/<slug>/
  meta.json                          # Competition metadata
  plan.json                          # Agent-generated plan (iteration 0)

  context/
    rules_url.txt                    # Always present
    rules.html                       # Best-effort fetch
    dataset_profile.json             # Dataset stats + file format summary
    sample_submission.csv            # First 10 rows preview
    top1_snapshot.json               # Top1 at bootstrap time
    kb_hints.json                    # Similar competitions from KB

  runs/<run_id>/
    run.json                         # Run configuration

    iter-0/                          # Bootstrap iteration
      agent/
        prompt.md                    # initial_plan_and_implement.md
        response.txt                 # Agent output

    iter-{1..5}/
      submission.csv                 # Predictions for this iteration
      metrics.json                   # Offline + Top1 scores
      diagnostics.md                 # Agent-readable analysis
      logs/                          # Training logs (if local)
      agent/
        prompt.md                    # improve_iteration.md
        response.txt                 # Agent output

    postmortem/
      summary.md                     # What worked, what didn't
      kb_update.json                 # Data to insert into KB

  submissions/
    ledger.jsonl                     # Deduplication log

  diffs/
    <run_id>.diff                    # Git diff of code changes

knowledge/
  kagglebot.db                       # SQLite Knowledge Base
  taxonomy.yml                       # Tag vocabulary
```

### 3.2 File Schemas

#### meta.json
```json
{
  "slug": "house-prices-advanced-regression-techniques",
  "url": "https://www.kaggle.com/competitions/house-prices...",
  "title": "House Prices - Advanced Regression Techniques",
  "created_at": "2026-01-02T12:00:00Z",
  "tags": ["tabular", "regression", "n_rows_small", "missingness_high"]
}
```

#### plan.json (generated by agent in iteration 0)
```json
{
  "target_metric": "rmse",
  "target_score": 0.12,
  "target_direction": "minimize",
  "eval_strategy": "holdout",
  "holdout_frac": 0.2,
  "cv_folds": null,
  "seed": 42,
  "compute_hints": {
    "use_gpu": true,
    "batch_size": 256,
    "precision": "float32"
  },
  "model_type": "xgboost",
  "features_engineered": ["log_transform", "polynomial_2", "target_encoding"]
}
```

#### metrics.json (per iteration)
```json
{
  "iteration": 3,
  "timestamp": "2026-01-02T14:30:00Z",
  "offline_score": 0.125,
  "offline_std": 0.003,
  "train_score": 0.110,
  "val_score": 0.125,
  "eval_strategy": "holdout",
  "top1_public": 0.11234,
  "met_heuristic": false,
  "margin_abs": 0.0,
  "margin_rel": 0.0,
  "gpu_utilization": 0.95,
  "training_time_sec": 245
}
```

#### diagnostics.md (generated by bot for agent)
```markdown
# Diagnostics: Iteration 3

## Current Performance
- **Offline Score**: 0.125 (RMSE)
- **Target**: <= 0.11234 (Top1 public)
- **Gap**: 0.01266 (10.1% worse than Top1)
- **Met Heuristic**: No

## Analysis
- **Train/Val Gap**: 0.015 → Moderate overfitting
- **Trend**: Improving (prev: 0.130 → current: 0.125)
- **GPU Utilization**: 95% (good)

## Likely Causes
1. Insufficient regularization (XGBoost max_depth too high)
2. Missing important features (no interaction terms)
3. Target leakage not addressed

## Recommended Actions (Ranked)
1. Add L2 regularization, reduce max_depth from 8 to 6
2. Create interaction features (e.g., LotArea * OverallQual)
3. Implement CV instead of holdout for more robust validation
4. Try stacking with Ridge as meta-model

## Previous Attempts
- Iter 1: Initial model linear model (RMSE=0.145)
- Iter 2: Added XGBoost (RMSE=0.130, improved by 0.015)
- Iter 3: Tuned hyperparameters (RMSE=0.125, improved by 0.005)
```

---

## 4. Metric + Direction Inference

### 4.1 Detection Strategy (Priority Order)

```python
def infer_metric_and_direction(context):
    # 1. Explicit from Kaggle metadata (if available)
    if context.evaluation_metric:
        return (context.evaluation_metric, infer_direction(context.evaluation_metric))

    # 2. Web search for competition details
    if web_search_available():
        results = search(f"{slug} kaggle evaluation metric")
        metric = extract_metric_from_search(results)
        if metric:
            return (metric, infer_direction(metric))

    # 3. Leaderboard ordering analysis
    if len(leaderboard) >= 2:
        direction = "maximize" if leaderboard[0].score > leaderboard[1].score else "minimize"
        metric = guess_metric_from_task(task_type)
        return (metric, direction)

    # 4. Fallback heuristics based on task
    task = context.dataset_profile.task
    target_stats = context.dataset_profile.target_statistics

    if task == "regression":
        if target_stats.all_positive and target_stats.skewness > 1.5:
            return ("rmsle", "minimize")  # Positive and skewed
        else:
            return ("rmse", "minimize")    # General regression

    elif task == "binary":
        return ("logloss", "minimize")     # Binary classification

    elif task == "multiclass":
        return ("logloss", "minimize")     # Multi-class

    else:
        return ("rmse", "minimize")        # Ultimate fallback
```

### 4.2 Direction Inference Rules

```python
def infer_direction(metric_name: str) -> str:
    """Infer optimization direction from metric name."""
    metric_lower = metric_name.lower()

    # Minimize these
    if any(keyword in metric_lower for keyword in [
        "loss", "error", "rmse", "rmsle", "mae", "mse", "mape"
    ]):
        return "minimize"

    # Maximize these
    if any(keyword in metric_lower for keyword in [
        "accuracy", "auc", "roc", "f1", "precision", "recall", "r2", "score"
    ]):
        return "maximize"

    # Ambiguous: use leaderboard ordering if available
    # Else default to minimize
    return "minimize"
```

### 4.3 Metric Computation Mapping

| Metric | Computation | Library |
|--------|-------------|---------|
| RMSE | `sqrt(mean_squared_error(y, pred))` | sklearn |
| RMSLE | `sqrt(mean_squared_error(log1p(y), log1p(pred)))` | sklearn |
| MAE | `mean_absolute_error(y, pred)` | sklearn |
| LogLoss | `log_loss(y, pred_proba)` | sklearn |
| AUC-ROC | `roc_auc_score(y, pred_proba)` | sklearn |
| Accuracy | `accuracy_score(y, pred)` | sklearn |
| F1 | `f1_score(y, pred, average='macro')` | sklearn |

---

## 5. Top1 Heuristic Compare Rule

### 5.1 Mathematical Definition

Given:
- `offline`: Offline validation score
- `top1`: Public leaderboard Top1 score
- `margin_abs`: Absolute margin (default: 0.0)
- `margin_rel`: Relative margin (default: 0.0)
- `direction`: "minimize" or "maximize"

**Heuristic Met** when:

```python
def meets_heuristic(offline, top1, direction, margin_abs=0.0, margin_rel=0.0):
    if direction == "minimize":
        threshold = top1 * (1 + margin_rel) + margin_abs
        return offline <= threshold

    else:  # maximize
        threshold = top1 * (1 - margin_rel) - margin_abs
        return offline >= threshold
```

### 5.2 Examples

**Minimize (RMSE)**:
- Top1 = 0.11234
- margin_abs = 0.0, margin_rel = 0.0
- Threshold = 0.11234 * 1.0 + 0.0 = 0.11234
- Met if: offline <= 0.11234

**Maximize (AUC)**:
- Top1 = 0.95
- margin_abs = 0.0, margin_rel = 0.0
- Threshold = 0.95 * 1.0 - 0.0 = 0.95
- Met if: offline >= 0.95

**With Margins (Conservative)**:
- Top1 = 0.11234 (RMSE, minimize)
- margin_abs = 0.01, margin_rel = 0.05
- Threshold = 0.11234 * 1.05 + 0.01 = 0.128957
- Met if: offline <= 0.128957 (more lenient)

### 5.3 Disclaimers

**Must log prominently**:

```
⚠️  Top1 Heuristic is a PROXY, not a guarantee:
- Offline validation uses train.csv split (holdout or CV)
- Kaggle public leaderboard uses test.csv (different distribution)
- Top1 may use different preprocessing, ensembles, or seeds
- Meeting heuristic does NOT guarantee public/private LB ranking
- Use as a stopping criterion, not a performance guarantee
```

**Recommendation**: Set `margin_rel=0.0, margin_abs=0.0` (exact match) for conservative stopping.

---

## 6. Compute Modes

### 6.1 Supported Backends

| Mode | Description | When to Use | GPU Support |
|------|-------------|-------------|-------------|
| `local_gpu` | Local GPU (CUDA/MPS) | Fast iterations, large datasets | ✅ Required |
| `kaggle_gpu` | Kaggle Notebook GPU (T4) | No local GPU, GPU needed | ✅ T4 (16GB) |
| `kaggle_tpu` | Kaggle Notebook TPU (v3-8) | TF/JAX workloads, huge batches | ✅ TPU v3-8 |

### 6.2 Runner Interface

All runners implement:

```python
class Runner(Protocol):
    def train_and_evaluate(
        self,
        data_dir: Path,
        output_path: Path,
        plan: PlanConfig,
        seed: int,
    ) -> TrainingOutcome:
        """Train model and return predictions + metrics."""
        ...

class TrainingOutcome:
    submission_path: Path       # Path to submission.csv
    metrics: dict               # Offline scores
    model_summary: dict         # Model metadata
    logs_path: Path | None      # Training logs
```

### 6.3 Local GPU Runner

**Requirements**:
- CUDA 11.8+ or Metal (MPS)
- PyTorch / XGBoost / LightGBM with GPU support

**Implementation**:
```python
def train_local_gpu(data_dir, output_path, plan, seed):
    # 1. Set device
    device = "cuda" if torch.cuda.is_available() else "mps"

    # 2. Load data
    train, test = load_competition_data(data_dir)

    # 3. Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        train, stratify=(y if plan.task == "classification" else None), random_state=seed
    )

    # 4. Train model (GPU-accelerated)
    model = build_model(plan, device=device)
    model.fit(X_train, y_train)

    # 5. Evaluate
    val_pred = model.predict(X_val)
    offline_score = compute_metric(plan.target_metric, y_val, val_pred)

    # 6. Predict on test
    test_pred = model.predict(test)
    save_submission(output_path, test_pred)

    return TrainingOutcome(
        submission_path=output_path,
        metrics={"offline": offline_score, "train": ..., "val": ...},
        model_summary={"type": plan.model_type, "gpu_used": True},
    )
```

### 6.4 Kaggle GPU/TPU Runner (Kernel Mode)

**Requirements**:
- Kaggle CLI configured
- Kernel metadata template
- Status polling logic

**Implementation**:
```python
def train_kaggle_kernel(slug, plan, compute_mode, seed):
    # 1. Generate kernel metadata
    kernel_meta = {
        "id": f"{username}/kagglebot-{slug}-{run_id}",
        "title": f"Kagglebot {slug} Training",
        "code_file": "solver.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": (compute_mode == "kaggle_gpu"),
        "enable_tpu": (compute_mode == "kaggle_tpu"),
        "enable_internet": False,  # Default OFF
        "dataset_sources": [slug],
    }

    # 2. Write solver.py (self-contained training script)
    write_kernel_script(plan, seed)

    # 3. Push kernel
    run(["kaggle", "kernels", "push", "-p", kernel_dir])

    # 4. Poll status (timeout: 2 hours)
    for _ in range(240):  # Poll every 30s for 2 hours
        status = get_kernel_status(kernel_id)
        if status == "complete":
            break
        elif status in ["error", "cancelled"]:
            raise KernelFailedError()
        time.sleep(30)

    # 5. Download outputs
    run(["kaggle", "kernels", "output", kernel_id, "-p", output_dir])

    # 6. Parse metrics.json and submission.csv
    metrics = json.load(open(output_dir / "metrics.json"))
    submission = output_dir / "submission.csv"

    return TrainingOutcome(
        submission_path=submission,
        metrics=metrics,
        model_summary={"kernel_id": kernel_id, "compute": compute_mode},
    )
```

### 6.5 GPU/TPU Utilization Guidelines

**Agent must be told**:
- Use `device='cuda'` or `device='mps'` for PyTorch
- Use `tree_method='gpu_hist'` for XGBoost
- Use `device='gpu'` for LightGBM
- Batch size should be large enough to saturate GPU (e.g., 256-1024)
- Use mixed precision (`torch.cuda.amp`) for faster training
- For TPU: use TensorFlow/JAX with TPU strategy

---

## 7. Knowledge Base Design

### 7.1 SQLite Schema

```sql
-- Competitions table
CREATE TABLE competitions (
    slug TEXT PRIMARY KEY,
    url TEXT,
    title TEXT,
    task TEXT,  -- regression, binary, multiclass
    metric TEXT,
    created_at INTEGER
);

-- Tags table (controlled vocabulary from taxonomy.yml)
CREATE TABLE tags (
    tag TEXT PRIMARY KEY
);

-- Competition-Tag mapping
CREATE TABLE competition_tags (
    slug TEXT,
    tag TEXT,
    PRIMARY KEY (slug, tag),
    FOREIGN KEY (slug) REFERENCES competitions(slug),
    FOREIGN KEY (tag) REFERENCES tags(tag)
);

-- Runs table
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    slug TEXT,
    started_at INTEGER,
    compute TEXT,
    final_iteration INTEGER,
    best_offline_score REAL,
    submitted BOOLEAN,
    FOREIGN KEY (slug) REFERENCES competitions(slug)
);

-- Iterations table
CREATE TABLE iterations (
    run_id TEXT,
    iteration INTEGER,
    offline_score REAL,
    top1_public REAL,
    met_heuristic BOOLEAN,
    training_time_sec REAL,
    created_at INTEGER,
    PRIMARY KEY (run_id, iteration),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Improvements table (what changed and impact)
CREATE TABLE improvements (
    run_id TEXT,
    iteration INTEGER,
    change_summary TEXT,  -- Agent-written summary
    delta_offline REAL,   -- Improvement from previous iteration
    worked BOOLEAN,       -- Whether it helped
    created_at INTEGER,
    PRIMARY KEY (run_id, iteration),
    FOREIGN KEY (run_id, iteration) REFERENCES iterations(run_id, iteration)
);
```

### 7.2 Taxonomy (taxonomy.yml)

```yaml
# Controlled tag vocabulary
data_modality:
  - tabular
  - text
  - image
  - timeseries
  - multi_modal

problem_type:
  - regression
  - binary
  - multiclass
  - ranking
  - forecasting

dataset_scale:
  - n_rows_tiny      # < 1K
  - n_rows_small     # 1K-10K
  - n_rows_medium    # 10K-100K
  - n_rows_large     # 100K-1M
  - n_rows_huge      # > 1M

characteristics:
  - missingness_low    # < 5%
  - missingness_high   # > 20%
  - high_cardinality_cats
  - imbalanced_classes
  - skewed_target
  - temporal_structure
  - hierarchical_data

aliases:
  binary_classification: binary
  multiclass_classification: multiclass
  nlp: text
  cv: image
```

### 7.3 Tagging Rules (Auto-Applied During Bootstrap)

```python
def generate_tags(dataset_profile):
    tags = []

    # Modality
    if dataset_profile.has_images:
        tags.append("image")
    elif dataset_profile.avg_text_length > 50:
        tags.append("text")
    else:
        tags.append("tabular")

    # Task
    tags.append(dataset_profile.task)  # regression, binary, multiclass

    # Scale
    n_rows = dataset_profile.n_rows
    if n_rows < 1000:
        tags.append("n_rows_tiny")
    elif n_rows < 10000:
        tags.append("n_rows_small")
    elif n_rows < 100000:
        tags.append("n_rows_medium")
    elif n_rows < 1000000:
        tags.append("n_rows_large")
    else:
        tags.append("n_rows_huge")

    # Characteristics
    if dataset_profile.missing_pct > 0.2:
        tags.append("missingness_high")

    if any(card > 100 for card in dataset_profile.cardinalities):
        tags.append("high_cardinality_cats")

    if dataset_profile.class_balance_ratio > 5:
        tags.append("imbalanced_classes")

    if dataset_profile.target_skewness > 1.5:
        tags.append("skewed_target")

    return tags
```

### 7.4 Retrieval Strategy

```python
def find_similar_competitions(current_tags, limit=3):
    """Find competitions with maximum tag overlap."""
    query = """
        SELECT
            c.slug,
            c.title,
            COUNT(*) as tag_overlap,
            GROUP_CONCAT(ct.tag) as matching_tags
        FROM competitions c
        JOIN competition_tags ct ON c.slug = ct.slug
        WHERE ct.tag IN ({placeholders})
        GROUP BY c.slug
        ORDER BY tag_overlap DESC, c.created_at DESC
        LIMIT ?
    """.format(placeholders=','.join('?' * len(current_tags)))

    results = db.execute(query, [*current_tags, limit]).fetchall()

    # For each similar competition, get successful improvements
    for comp in results:
        improvements = get_successful_improvements(comp.slug)
        comp.kb_hints = improvements

    return results

def get_successful_improvements(slug):
    """Get improvements that actually helped."""
    query = """
        SELECT i.change_summary, i.delta_offline
        FROM improvements i
        JOIN runs r ON i.run_id = r.run_id
        WHERE r.slug = ?
        AND i.worked = 1
        AND i.delta_offline IS NOT NULL
        ORDER BY ABS(i.delta_offline) DESC
        LIMIT 5
    """
    return db.execute(query, [slug]).fetchall()
```

### 7.5 KB Update (Post-Run)

After autopilot completes, update KB with:

```python
def update_kb(run_id, slug, iterations):
    # 1. Insert run
    db.execute("""
        INSERT INTO runs (run_id, slug, started_at, compute, final_iteration, best_offline_score, submitted)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (run_id, slug, timestamp, compute, max_iter, best_score, True))

    # 2. Insert iterations
    for iter_data in iterations:
        db.execute("""
            INSERT INTO iterations (run_id, iteration, offline_score, top1_public, met_heuristic, training_time_sec)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, iter_data.iteration, iter_data.offline, iter_data.top1, iter_data.met, iter_data.time))

    # 3. Insert improvements (agent-generated summaries)
    for improvement in postmortem.improvements:
        db.execute("""
            INSERT INTO improvements (run_id, iteration, change_summary, delta_offline, worked)
            VALUES (?, ?, ?, ?, ?)
        """, (run_id, improvement.iteration, improvement.summary, improvement.delta, improvement.worked))

    db.commit()
```

---

## 8. Version Control

Git integration has been removed from the implementation. Autopilot does not run any git commands, does not stash,
and does not capture diffs. Users manage version control manually.

---

## 9. Terminal Progress UX

### 9.1 Rich-Based Progress Display

Use `rich` library for:
- Progress bars (downloads, training)
- Live status tables (iteration summary)
- Color-coded messages (success, warnings, errors)
- Syntax highlighting (code snippets, JSON)

### 9.2 Iteration Summary Table

After each iteration, display:

```
╭──────────────── Iteration 3 Summary ────────────────╮
│ Offline Score   │ 0.125 RMSE                        │
│ Top1 Public     │ 0.11234 RMSE                      │
│ Gap             │ +0.01266 (10.1% worse)            │
│ Met Heuristic   │ ✗ No                              │
│ Training Time   │ 4m 12s                            │
│ GPU Util        │ 95%                               │
│ Action          │ → Improve (iteration 4)           │
╰─────────────────────────────────────────────────────╯
```

### 9.3 Kernel Status Polling (Kaggle Compute)

When using `kaggle_gpu` or `kaggle_tpu`:

```
[cyan]Kernel Status[/cyan]: https://www.kaggle.com/user/kernel-id

╭─ Polling Kernel Status ─╮
│ Status: running          │
│ Elapsed: 2m 35s          │
│ Timeout: 120m            │
│ Progress: [████──────] 15% │
╰──────────────────────────╯

Logs (last 5 lines):
  > Loading data... (1.2s)
  > Training XGBoost... (GPU detected)
  > Iteration 100/1000, RMSE: 0.134
  > Iteration 200/1000, RMSE: 0.128
  > Iteration 300/1000, RMSE: 0.126
```

### 9.4 Artifact Paths

Print where outputs are written:

```
[green]✓ Artifacts saved:[/green]
  Submission:   artifacts/house-prices.../runs/run-abc123/iter-3/submission.csv
  Metrics:      artifacts/house-prices.../runs/run-abc123/iter-3/metrics.json
  Diagnostics:  artifacts/house-prices.../runs/run-abc123/iter-3/diagnostics.md
  Diff:         artifacts/house-prices.../diffs/run-abc123.diff
```

### 9.5 Final Summary

```
╭────────────────────── Autopilot Complete ──────────────────────╮
│ Competition       │ house-prices-advanced-regression-techniques│
│ Run ID            │ run-abc123                                  │
│ Iterations        │ 3 / 5                                       │
│ Best Offline      │ 0.125 RMSE (iteration 3)                    │
│ Top1 Public       │ 0.11234 RMSE                                │
│ Met Heuristic     │ ✗ No (gap: +10.1%)                         │
│ Submitted         │ ✓ Yes (iteration 3 submission)              │
│ Submission URL    │ https://www.kaggle.com/competitions/.../... │
│ Training Time     │ 14m 27s                                     │
│ Knowledge Base    │ Updated with 3 improvements                 │
╰────────────────────────────────────────────────────────────────╯

[blue]Next steps:[/blue]
1. Review diff: artifacts/house-prices.../diffs/run-abc123.diff
2. Check submission: https://www.kaggle.com/competitions/.../submissions
3. Query KB: uv run kagglebot knowledge show house-prices...
```

---

## 10. Error Handling & Resilience

### 10.1 Retry Strategies

| Operation | Max Retries | Backoff | Fallback |
|-----------|-------------|---------|----------|
| Kaggle API download | 3 | Exponential (1s, 2s, 4s) | Fail |
| Rules page fetch | 2 | Linear (1s, 2s) | Store URL only |
| Leaderboard fetch | 3 | Exponential | Cache last value |
| Kernel status poll | 240 | Fixed 30s | Timeout error |
| Test verification | 3 | None | Revert changes |

### 10.2 Graceful Degradation

**Rules fetch failure**:
- Store `rules_url.txt`
- Log warning: "Could not fetch rules HTML, stored URL only"
- Continue (agent can use URL)

**Top1 fetch failure**:
- Use cached value from bootstrap
- Log warning: "Using cached Top1 from bootstrap"
- Continue

**Web search unavailable**:
- Skip web search enhancement
- Use fallback metric inference
- Log info: "Web search not available, using fallbacks"

### 10.3 User-Actionable Errors

All errors must include:
- What went wrong
- Why it matters
- How to fix it

Example:
```
[red]✗ Error: Rules not accepted[/red]

Competition rules have not been accepted for 'house-prices-advanced-regression-techniques'.

Action required:
1. Visit: https://www.kaggle.com/competitions/house-prices.../rules
2. Click "I Understand and Accept"
3. Re-run: uv run kagglebot autopilot house-prices... --agent codex --compute local_gpu --submit

Exit code: 2
```

---

## 11. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Bootstrap time | < 5 min | Data download + profiling + KB query |
| Agent response (initial model) | < 15 min | Codex implementation of initial model |
| Agent response (improve) | < 10 min | Codex iteration improvement |
| Local GPU training | Variable | Depends on dataset size, no limit |
| Kaggle kernel | < 2 hours | Timeout enforced |
| Total autopilot (5 iter) | < 3 hours | Typical case (local GPU) |

---

## 12. Security & Privacy

### 12.1 Secrets Management

**Never Include in Prompts**:
- Kaggle API credentials (`kaggle.json`)
- API keys
- Personal identifiable information (PII)
- Private leaderboard scores (only public Top1)

**Allowed in Prompts**:
- Public competition URLs
- Public leaderboard scores
- Dataset statistics (aggregated)
- Competition rules (public)

### 12.2 Web Search Safety

If web search is enabled:
- Only search public Kaggle forums, kernels, discussions
- Never search for private/leaked data
- Log all search queries for audit
- Respect Kaggle Terms of Service

### 12.3 Submission Safety

- Deduplicate via SHA256 hash
- Rate limit: 5 min cooldown between submissions
- Max submissions per run: 1 (by design)
- Validate submission format before uploading

---

## 13. Acceptance Criteria

Before deployment, verify:

### 13.1 Functional Requirements

- [ ] Minimal CLI works: `autopilot <url> --agent codex --compute <mode> --submit`
- [ ] Bootstrap downloads data, rules, leaderboard
- [ ] Agent generates plan.json in iteration 0
- [ ] Iteration loop runs 1-5 times
- [ ] Top1 heuristic comparison is direction-aware
- [ ] Submits when heuristic met OR at iteration 5
- [ ] Exactly 1 submission per run
- [ ] Knowledge Base queries and updates work
- [ ] Git stays on main, auto-stashes, saves diffs
- [ ] Tests verified after each agent change

### 13.2 Compute Modes

- [ ] `local_gpu` trains locally with GPU utilization
- [ ] `kaggle_gpu` pushes kernel, polls status, downloads outputs
- [ ] `kaggle_tpu` works with TPU-compatible code (TF/JAX)

### 13.3 Documentation

- [ ] README is minimal (single command example)
- [ ] spec_autopilot.md is complete and unambiguous
- [ ] Prompt templates are actionable
- [ ] Guardrails checklist is comprehensive

### 13.4 Safety

- [ ] No secrets in prompts or logs
- [ ] No branches created
- [ ] No auto-commits (diffs saved)
- [ ] Rules acceptance verified (manual)
- [ ] Deduplication and rate limiting enforced

### 13.5 Performance

- [ ] Bootstrap < 5 min
- [ ] Agent initial model < 15 min
- [ ] Agent improve < 10 min
- [ ] GPU/TPU utilization > 80%

---

## Appendix A: Decision Log

| Decision | Rationale |
|----------|-----------|
| 5 iterations max | Balance exploration vs time; most improvements plateau by 5 |
| No training time limit | Accuracy-first; user can interrupt if needed |
| Submit when Top1 met OR iter 5 | Guarantees 1 submission; early exit when heuristic satisfied |
| Main-only git | Simplicity; avoid branch sprawl; user commits manually |
| No auto-commit | Let user review diffs before committing |
| Holdout default, CV optional | Faster; agent can choose CV in plan.json if needed |
| Offline validation only | Never use test.csv; prevent leakage |
| Top1 margins default 0.0 | Conservative; user can tune if needed |
| KB auto-tagging | Reduce manual work; deterministic |
| Rich terminal UI | Better UX than plain text; progress visibility |

---

## Appendix B: Future Enhancements

- Multi-agent collaboration (plan, code, review agents)
- Ensemble submission (blend top-K iterations)
- Hyperparameter tuning with Optuna
- Auto-detect private LB shift patterns
- Team collaboration (shared KB)
- Competition-specific plugins (time series, NLP, CV)

---

**End of Specification**
