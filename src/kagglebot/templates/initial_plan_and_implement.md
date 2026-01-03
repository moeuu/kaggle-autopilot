# Kagglebot Codex: Plan + Initial Model Implementation

## Competition Overview

**Slug**: `{slug}`
**URL**: {competition_url}
**Rules**: {rules_url}
**Task**: {task_type}
**Metric (confirm via rules)**: {suggested_metric}
**Dataset**: {n_rows:,} rows × {n_cols} columns
**Tags**: {tags}

## Compute Environment

**Mode**: `{compute_mode}`
**Constraints**:
{compute_constraints}

**Resource Targets**:
- GPU Utilization: Aim for >80% during training (check with `nvidia-smi` or kernel logs)
- TPU Utilization: Aim for >70% MXU usage (for kaggle_tpu mode)
- Memory: Stay within available limits ({memory_limit})
- Internet: {internet_status} (web search allowed for documentation, never for secrets)

## Available Context Files

- `{dataset_profile_path}` - Full dataset statistics (missingness, cardinality, distributions)
- `{sample_submission_path}` - Required submission format (MUST match exactly)
- `{top1_public_path}` - Current leaderboard leader score (context only, NOT submission criterion)
- `{rules_url_path}` - Competition rules reference
- `{rules_html_path}` - Rules HTML (if present)
- `artifacts/{slug}/context/overview.md` - Competition overview (if present)
- `artifacts/{slug}/context/data.md` - Data description (if present)
- `{submission_format_path}` - Submission format details (if present)
{kb_hints_section}

## Knowledge Base: Similar Competitions

{kb_similar_competitions}

---

## Your Task

### Step 1: Create Competition Plan

Update `{plan_path}` with the following fields:

```json
{{
  "target_metric": "{suggested_metric}",
  "target_score": {suggested_target_score},
  "target_direction": "{suggested_direction}",
  "score_source": "{default_score_source}",
  "holdout_frac": 0.2,
  "cv_folds": 5,
  "seed": 42,
  "submit_policy": "on_target_only"
}}
```

**Plan Guidelines**:
- **target_metric**: Derive from rules.html and dataset (do not assume defaults)
- **Context**: Read overview.md/data.md for problem framing and data caveats
- **target_score**: Set based on Top1 public leaderboard ({top1_score}) and dataset complexity
  - For small datasets (<5K rows): Add 10-20% margin to Top1
  - For medium datasets (5K-100K rows): Add 5-10% margin
  - For large datasets (>100K rows): Match or slightly exceed Top1
  - Direction-aware: minimize means higher value is more lenient, maximize means lower value
- **score_source**: Use `holdout` by default; only use `cv` if you need robust estimates (small datasets, high variance)
- **submit_policy**: Keep as `on_target_only` (only submit when target met OR at final iteration)

### Step 2: Implement Strong Initial Solution

Implement a strong initial model based on overview/data/rules + web research.

**Where to implement**:
- `local_cpu` / `local_gpu`: update `kagglebot/solver/initial_model.py` (use the same best‑model flow).
- `kaggle_gpu` / `kaggle_tpu`: create `artifacts/{slug}/kernel.py` from scratch.
- If the data is non‑tabular (images/text/FASTA/etc.), create `artifacts/{slug}/kernel.py` with a `custom_main()` entrypoint.

Your implementation should:

#### 2a) Data Loading and Preprocessing

**Load data**:
```python
import pandas as pd
train = pd.read_csv("{train_path}")
test = pd.read_csv("{test_path}")
```

**Handle missing values**:
- Numeric: Impute with median or mean + add missingness indicator
- Categorical: Impute with mode or add "missing" category
- Check dataset_profile.json for columns with high missingness

**Feature engineering**:
- Create interaction features for important variables
- Log-transform skewed numeric features (check skewness in dataset_profile)
- Extract date components if temporal columns exist
- One-hot encode low-cardinality categoricals (<10 unique values)
- Target/frequency encode high-cardinality categoricals (>10 unique values)
- Scale numeric features (StandardScaler or RobustScaler)

**Split for offline evaluation**:
```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size={default_holdout_frac},
    random_state=42,
    stratify=y if {is_classification} else None
)
```

#### 2b) Model Selection and Training

Use web search to identify the strongest known approach for this competition and task.
Prefer GPU-accelerated supervised models when available.

**Recommended families** (choose based on research + data traits):
- Gradient-boosted trees with native categorical handling (CatBoost) for mixed tabular data
- GPU-accelerated GBDT (XGBoost / LightGBM) for large/tabular datasets
- Deep models only if the competition trend or data type supports it

**Classification/Regression notes**:
- Pick loss/metric consistent with rules and sample_submission
- Address class imbalance (class weights / focal loss) if indicated
- Set enough iterations/trees to fully use GPU (avoid tiny training jobs)

**GPU/TPU Utilization**:
- For `local_gpu` or `kaggle_gpu`: Use XGBoost/LightGBM with `tree_method='gpu_hist'`
- For `kaggle_tpu`: Use TensorFlow/JAX models that support TPU
- Monitor utilization: GPU >80%, TPU MXU >70%
- If utilization is low: Increase batch size, use larger models, or parallelize CV folds

#### 2c) Offline Evaluation

**Evaluate on validation set**:
```python
from kagglebot.solver.metrics import compute_metric, infer_direction

# For regression and binary classification
y_pred = model.predict(X_val)

# For metrics requiring probabilities (logloss, auc)
if metric_requires_proba("{suggested_metric}"):
    y_pred = model.predict_proba(X_val)[:, 1]  # Binary: positive class
    # y_pred = model.predict_proba(X_val)  # Multiclass: all classes

val_score = compute_metric("{suggested_metric}", y_val, y_pred)
```

**Save metrics**:
```python
metrics = {{
    "iteration": 0,
    "score_source": "{default_score_source}",
    "metric": "{suggested_metric}",
    "direction": "{suggested_direction}",
    "train_score": train_score,
    "val_score": val_score,
    "value": val_score,
    "target": {suggested_target_score},
    "met_target": _meets_target(val_score, {suggested_target_score}, "{suggested_direction}"),
    "top1_public": {top1_score},
    "gpu_utilization": gpu_util,  # From nvidia-smi or kernel logs
}}

with open("{metrics_path}", "w") as f:
    json.dump(metrics, f, indent=2)
```

#### 2d) Generate Submission

**Predict on test set**:
```python
# Use same preprocessing as training data
X_test_processed = preprocess(test)

# Generate predictions
if metric_requires_proba("{suggested_metric}"):
    y_test_pred = model.predict_proba(X_test_processed)[:, 1]  # or all classes
else:
    y_test_pred = model.predict(X_test_processed)
```

**Create submission.csv**:
```python
import pandas as pd

# Load sample submission to get exact format
sample = pd.read_csv("{sample_submission_path}")

# CRITICAL: Match column names and row count exactly
submission = pd.DataFrame({{
    sample.columns[0]: sample.iloc[:, 0],  # ID column (preserve order)
    sample.columns[1]: y_test_pred          # Prediction column
}})

# Validate format
assert submission.shape == sample.shape, "Shape mismatch"
assert list(submission.columns) == list(sample.columns), "Column mismatch"

submission.to_csv("{submission_path}", index=False)
```

### Step 3: Safety and Quality Checks

**CRITICAL SAFETY RULES**:
- ❌ NEVER automate Kaggle rules acceptance (users must accept manually in browser)
- ❌ NEVER commit secrets (kaggle.json, API keys, tokens, passwords)
- ❌ NEVER add interactive prompts (CLI must be non-interactive for automation)
- ❌ NEVER hardcode competition-specific logic in solver/ (keep it generic)
- ❌ NEVER use test.csv for validation (offline eval uses train.csv splits only)
- ❌ NEVER bypass safety guardrails (duplicate checks, rate limits, validation)
- ✅ DO validate submission format matches sample_submission.csv exactly
- ✅ DO handle edge cases (empty files, missing columns, type mismatches)
- ✅ DO set random seeds for reproducibility (42 everywhere)
- ✅ DO use web search for documentation (scikit-learn, XGBoost, pandas)
- ✅ DO maximize GPU/TPU utilization (monitor and adjust)

**Quality Checklist**:
- [ ] plan.json created with all required fields and realistic target_score
- [ ] Initial model runs end-to-end without errors
- [ ] submission.csv format matches sample_submission.csv exactly (columns, rows, types)
- [ ] Offline evaluation produces a numeric score in metrics.json
- [ ] GPU/TPU utilization >80% (GPU) or >70% (TPU MXU) during training
- [ ] Tests pass: `uv run pytest -q`
- [ ] No secrets in code or logs

### Step 4: Run Verification

Before finishing, verify:

```bash
# Run tests (must pass)
uv run pytest -q

# Train and evaluate initial model
uv run kagglebot train {slug} --compute {compute_mode}

# Check metrics
cat {metrics_path}

# Verify submission format
head {submission_path}
diff <(head -1 {sample_submission_path}) <(head -1 {submission_path})
```

**Expected Output**:
- metrics.json exists with `value` field containing offline score
- submission.csv exists and matches sample_submission.csv format
- GPU/TPU utilization logged in metrics.json

---

## Acceptance Criteria

Your initial model will be accepted if:

1. ✅ **Tests pass**: `uv run pytest -q` returns 0 exit code
2. ✅ **Offline score computed**: metrics.json has valid `value` field
3. ✅ **Submission valid**: submission.csv matches sample_submission.csv format
4. ✅ **GPU/TPU utilized**: Utilization >80% (GPU) or >70% (TPU) logged
5. ✅ **No errors**: Training completes without exceptions
6. ✅ **Reproducible**: Same seed produces same results

If any criterion fails, the autopilot will retry or abort.

---

## Tips for Success

**Dataset Analysis**:
- Check dataset_profile.json for missingness patterns, outliers, skewness
- Look for high-cardinality categoricals (may need target encoding)
- Check target distribution (imbalanced? skewed?)

**Model Selection**:
- Small data (<5K rows): Regularized linear models (Ridge, Logistic)
- Medium data (5K-100K rows): Tree ensembles (Random Forest, XGBoost)
- Large data (>100K rows): Gradient boosting (XGBoost, LightGBM, CatBoost)
- GPU available: XGBoost/LightGBM with gpu_hist, or neural networks
- TPU available: TensorFlow/JAX models

**Common Pitfalls**:
- Don't overfit: Large models on small datasets fail
- Don't underfit: Too simple models miss patterns
- Don't leak: Never use test.csv for validation
- Don't ignore format: submission.csv must match sample exactly
- Don't waste resources: Monitor and maximize GPU/TPU usage

**Debugging**:
- If score is "too good": Check for data leakage
- If GPU utilization is low: Increase batch size or model complexity
- If submission fails validation: Check column names and row count
- If tests fail: Read error messages carefully

Good luck with the initial model! 🚀
