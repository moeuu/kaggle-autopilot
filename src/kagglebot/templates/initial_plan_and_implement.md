# Kagglebot {implementation_agent_name}: Plan + Initial Model Implementation

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
  "submit_policy": "always"
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
- **score_source**: Use `cv` by default for robust model ranking; use `holdout` only when CV is infeasible
- **submit_policy**: Keep as `always` unless rules clearly limit submission counts
- **candidate selection guard**: If multiple pipelines are evaluated, do not select the final submission by CV alone. Log every candidate's primary CV score, holdout/validation score, and submission/test prediction distribution. Prefer the candidate with the best competition-faithful validation signal when CV and holdout disagree; reject candidates whose holdout/validation score is materially worse than another candidate or whose test prediction distribution collapses to an implausibly sparse/constant output.

### Step 2: Implement Strong Initial Solution

Implement a strong initial model based on overview/data/rules + web research.

**Where to implement**:
- For **all compute modes** (`local_gpu`, `kaggle_gpu`, `kaggle_tpu`), implement in `artifacts/{slug}/kernel/kernel.py`.
- `local_gpu` and `kaggle_gpu` must use the **same algorithm/pipeline**; only execution location differs.
- If the data is non-tabular (image/video/audio/text/document/medical-imaging/point-cloud/3D/geospatial/bio/sequence/graph/signal/annotation/array/model-artifact), implement a `custom_main()` entrypoint in `kernel.py`.

Your implementation should:

#### 2a) Data Loading and Preprocessing

**Load data**:
```python
from pathlib import Path
from kagglebot.solver.io import read_table

train = read_table(Path("{train_path}"))
test = read_table(Path("{test_path}"))
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

**Anti-leakage policy for external data**:
- External data may support pretraining, feature learning, or additional supervised training only when rules allow it.
- Never use external labels to directly assign competition test predictions through exact or near-duplicate matching on row IDs, filenames, hashes, timestamps, bounding boxes, or image/audio/text content.
- If a public/external dataset appears to contain labeled copies of the competition test rows, exclude that overlap from prediction logic, report it in diagnostics, and build a supervised model that must generalize from allowed training data.
- Do not select pipelines named or behaving like `external_overlap_mapping`, `official_solution_mapping`, `label_transfer`, or `test_label_lookup`.

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
- Pretrained transfer models (vision/NLP/audio/video) when competition type and rules allow it

**High-accuracy tabular policy**:
- If the dataset is tabular binary with large row count and mixed/high-cardinality categoricals, do not stop at one family.
- Make the plan explicitly include CatBoost raw categorical, XGBoost with leak-safe target/stat encodings, and LightGBM or a second CatBoost/XGBoost variant.
- Add at least one OOF blend candidate (weighted/rank/logit blend) to the shortlist.
- Set a winner-mode objective in the plan (`target_medal=winner`, `target_rank_percentile=0.001`) so later iterations do not collapse into minor tuning before reaching a near-first-place rank band.

**Classification/Regression notes**:
- Pick loss/metric consistent with rules and sample_submission
- Address class imbalance (class weights / focal loss) if indicated
- Set enough iterations/trees to fully use GPU (avoid tiny training jobs)

**GPU/TPU Utilization**:
- For `local_gpu` or `kaggle_gpu`: Use XGBoost/LightGBM with `tree_method='gpu_hist'`
- For `kaggle_tpu`: Use TensorFlow/JAX models that support TPU
- Monitor utilization: GPU >80%, TPU MXU >70%
- If utilization is low: Increase batch size, use larger models, or parallelize CV folds

**Pretrained model policy**:
- If stronger results are likely from pretrained checkpoints, implement download logic in `kernel.py`.
- Prefer official hubs/sources (e.g., Hugging Face, official model repos) and cache under working/output dirs.
- Respect competition rules and internet settings; when internet is off, fail gracefully with a clear fallback model.

#### 2c) Offline Evaluation

**Evaluate on validation set**:
```python
from kagglebot.solver.metrics import compute_metric, infer_direction

# For regression and classification
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

**Create submission artifact**:
```python
from pathlib import Path
import pandas as pd
from kagglebot.solver.io import read_table, write_table

# Load sample submission to get exact tabular format when the competition provides one.
sample = read_table(Path("{sample_submission_path}"))

# CRITICAL: Match column names and row count exactly
submission = pd.DataFrame({{
    sample.columns[0]: sample.iloc[:, 0],  # ID column (preserve order)
    sample.columns[1]: y_test_pred          # Prediction column
}})

# Validate format
assert submission.shape == sample.shape, "Shape mismatch"
assert list(submission.columns) == list(sample.columns), "Column mismatch"

# Preserve the required suffix/format instead of forcing CSV.
write_table(submission, Path("{submission_path}"))
```

### Step 3: Safety and Quality Checks

**CRITICAL SAFETY RULES**:
- ❌ NEVER automate Kaggle rules acceptance (users must accept manually in browser)
- ❌ NEVER commit secrets (kaggle.json, API keys, tokens, passwords)
- ❌ NEVER add interactive prompts (CLI must be non-interactive for automation)
- ❌ NEVER hardcode competition-specific logic in generic src runtime modules; keep it in `kernel.py`
- ❌ NEVER use test files for validation (offline eval uses train data splits only)
- ❌ NEVER bypass safety guardrails (duplicate checks, rate limits, validation)
- ✅ DO validate submission format matches the required sample/format exactly
- ✅ DO handle edge cases (empty files, missing columns, type mismatches)
- ✅ DO set random seeds for reproducibility (42 everywhere)
- ✅ DO use web search for documentation (scikit-learn, XGBoost, pandas)
- ✅ DO maximize GPU/TPU utilization (monitor and adjust)

**Quality Checklist**:
- [ ] plan.json created with all required fields and realistic target_score
- [ ] Initial model runs end-to-end without errors
- [ ] Submission artifact format matches the required sample/format exactly (columns, rows, types when tabular)
- [ ] Offline evaluation produces a numeric score in metrics.json
- [ ] The primary score comes from a real trained candidate and train-data CV/holdout validation; no placeholder, proxy, public-anchor, packaging-only, identity/noop, or unscored metrics
- [ ] If multiple candidates exist, final selection is justified by CV + holdout/validation + prediction distribution, not by CV alone
- [ ] GPU/TPU utilization >80% (GPU) or >70% (TPU MXU) during training
- [ ] Implementation is in `artifacts/{slug}/kernel/kernel.py` (not src local trainer code)
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

# Verify submission artifact
ls -lh {submission_path}

# For tabular submissions, verify columns against the required sample/format.
# For non-tabular submissions, inspect submission_manifest.json or validate archive/single-file contents.
uv run python - <<'PY'
from pathlib import Path
from kagglebot.solver.io import read_table
sample_path = Path("{sample_submission_path}")
submission_path = Path("{submission_path}")
try:
    sample = read_table(sample_path)
    submission = read_table(submission_path)
except Exception as exc:
    print(f"non-tabular or manifest-based submission check required: {exc}")
else:
    assert list(submission.columns) == list(sample.columns)
    print("tabular submission columns match sample")
PY
```

**Expected Output**:
- metrics.json exists with `value` field containing offline score
- Submission artifact exists and matches the required sample/format
- GPU/TPU utilization logged in metrics.json

---

## Acceptance Criteria

Your initial model will be accepted if:

1. ✅ **Tests pass**: `uv run pytest -q` returns 0 exit code
2. ✅ **Offline score computed**: metrics.json has valid `value` field
3. ✅ **Submission valid**: submission artifact matches the required format
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
- Don't leak: Never use test files for validation
- Don't transfer labels: Never copy hidden/test labels from external labeled overlaps, exact file-hash matches, row-id mappings, or solution-like artifacts
- Don't ignore format: the submission artifact must match the required sample/format exactly
- Don't waste resources: Monitor and maximize GPU/TPU usage

**Debugging**:
- If score is "too good": Check for data leakage
- If GPU utilization is low: Increase batch size or model complexity
- If submission fails validation: Check column names and row count
- If tests fail: Read error messages carefully

Good luck with the initial model! 🚀
