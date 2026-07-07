# Kagglebot {implementation_agent_name}: Improvement Iteration

## Context

**Competition**: `{slug}`
**Iteration**: {iteration} / {max_iterations}
**Goal**: Improve loop-decision score to meet target ({target_score} {target_direction})
**Current Score Signal**: {current_score} ({current_direction})
**Target Met**: {met_target}

## Compute Environment

**Mode**: `{compute_mode}`
**Constraints**:
{compute_constraints}

**Resource Targets**:
- GPU Utilization: Aim for >80% during training
- TPU Utilization: Aim for >70% MXU usage (for kaggle_tpu mode)
- Memory: Stay within available limits ({memory_limit})
- Internet: {internet_status} (web search allowed for documentation)

## Input Files

**Plan and Configuration**:
- `{plan_path}` - Target metric/score/direction, evaluation strategy
- `{run_path}` - Run settings, iteration history, and outcomes

**Current Iteration Artifacts**:
- `{current_metrics_path}` - Latest loop-decision + offline-by-source results (iteration {current_iteration})
- `{current_diagnostics_path}` - Agent-readable performance analysis
- `{current_submission_path}` - Current submission artifact
- `{logs_dir}` - Training logs and error messages (if any)

**Previous Iterations** (learn from history):
{previous_iterations_list}

**Context Files**:
- `{dataset_profile_path}` - Dataset statistics
- `{sample_submission_path}` - Required submission format
- `{top1_public_path}` - Top1 leaderboard score (context only)
- `artifacts/<slug>/context/code.md` - Competition code snapshot with Required Reference Notebook baseline
- `artifacts/<slug>/context/code_notebooks_index.json` - Structured notebook metadata (kernel_id/title/score/source_file)
- `{rules_url_path}` - Competition rules URL
- `{rules_html_path}` - Rules HTML (if present)
- `{submission_format_path}` - Submission format details (if present)
- `{kernel_main_path}` - Kernel entrypoint for all compute modes (authoritative file)
{kb_hints_section}

## Performance Analysis

### Current Metrics Summary

```json
{current_metrics_json}
```

### Diagnostics

{diagnostics_summary}

### Iteration History

| Iter | Offline Score | Met Target | Change Summary |
|------|--------------|------------|----------------|
{iteration_history_table}

---

## Your Task

### Step 1: Diagnose Root Cause

Read diagnostics and metrics to identify why the current score doesn't meet target:

**Common Issues**:

1. **Underfitting** (high train + val error):
   - Symptoms: Both train and val scores are poor
   - Causes: Model too simple, insufficient features, poor preprocessing
   - Solutions: Add features, use more complex model, reduce regularization

2. **Overfitting** (low train error, high val error):
   - Symptoms: Large gap between train and val scores
   - Causes: Model too complex, insufficient data, bad CV strategy
   - Solutions: Add regularization, simplify model, improve CV, feature selection

3. **Poor Feature Engineering**:
   - Symptoms: Current model underperforms expected range
   - Causes: Missing important signals, poor encoding, no domain features
   - Solutions: Create interactions, transform skewed features, domain engineering

4. **Wrong Hyperparameters**:
   - Symptoms: Reasonable features but suboptimal performance
   - Causes: Learning rate, tree depth, regularization not tuned
   - Solutions: Grid/random search, use validation curve analysis

5. **Data Quality Issues**:
   - Symptoms: Unexpected patterns, "too good" scores, inconsistent val/test
   - Causes: Leakage, outliers, duplicates, incorrect splits
   - Solutions: Check for leakage, remove outliers, ensure proper CV

6. **Class Imbalance** (classification only):
   - Symptoms: High accuracy but poor minority class performance
   - Causes: Imbalanced target distribution
   - Solutions: SMOTE, class weights, stratified sampling, adjust threshold

7. **Inefficient Resource Usage**:
   - Symptoms: Low GPU/TPU utilization (<70%)
   - Causes: Small batch size, CPU bottlenecks, inefficient data loading
   - Solutions: Increase batch size, use GPU-enabled libraries, parallelize

**Analysis Questions**:
- What is the train-val gap? (check for over/underfitting)
- Have previous iterations tried similar changes? (avoid repetition)
- Is GPU/TPU utilization acceptable? (>80% GPU, >70% TPU MXU)
- Are there obvious feature engineering opportunities from dataset_profile?
- Does the diagnostic suggest specific next steps?

### Step 2: Implement Targeted Improvements

Make **1-2 focused changes** per iteration. Avoid changing too many things at once.
Prefer the highest realistic score ceiling over making the current iteration immediately submittable.
If forced to choose, preserve a stronger high-potential path rather than collapsing to a weak submit-ready fallback.

If current loop score is below the code reference notebook score, treat the
`Required Reference Notebook (Execution baseline)` in `context/code.md` as mandatory:
reuse its strongest leak-free baseline elements first, then iterate.

Use the top1 gap as a guide:
- **Far from top1** → major overhaul (model family/feature strategy)
- **Mid gap** → moderate update (features + key hyperparameters)
- **Near top1** → minor tuning (small hyperparameter tweaks)

Medal-aware override:
- If the plan includes `target_medal` or `target_rank_percentile` and that leaderboard percentile has not been reached, do **not** propose `minor_tuning`.
- Keep at least `moderate_update`, and prefer broader model-family search plus ensemble exploration until the target rank band is met.

High-accuracy tabular override:
- For tabular binary problems with large row count and meaningful categorical structure, keep multiple strong families active.
- Require CatBoost raw categorical, XGBoost with leak-safe target/stat encodings, and LightGBM or a second CatBoost/XGBoost variant.
- If 2 or more model pipelines are available, produce at least one OOF blend candidate (weighted/rank/logit blend).

Top1 campaign portfolio override:
- If `current_metrics_json.campaign.reference_reproduction_report.blocks_novelty` is true, reproduce or diagnose the required reference baseline before proposing novel model changes.
- If improvement_mode is `validation_redesign`, create group/time/leak-safe/proxy validation candidates first and justify the active validation profile using historical public outcomes.
- Persist OOF/test predictions and candidate metadata so `candidate_registry.json`, `portfolio_plan.json`, and `blend_report.json` can rank low-correlation candidates and allocate submissions by information value.
- For long-running multi-fold candidates, persist a valid intermediate candidate at the end of every fold. Save
  completed-fold OOF/test predictions, metadata, `candidate_<candidate>_fold<N>.json`, and a
  `submission_<candidate>_fold<N>.<suffix>` that matches the required submission format when tabular; never leave
  completed folds only in memory until all folds finish.

Candidate selection guard:
- Do not select the final submission by CV alone when multiple candidates exist.
- Log every candidate's primary CV score, holdout/validation score, and submission/test prediction distribution.
- If CV improves but holdout/validation or previous public outcomes regress, treat it as validation mismatch first and redesign validation before model-only tuning.
- Reject candidates whose holdout/validation score is materially worse than another available candidate or whose test prediction distribution collapses to implausibly sparse/constant outputs.
- Prefer a slightly lower CV candidate with stronger competition-faithful validation and stable prediction distribution over a higher-CV candidate that only fits the train-distribution split.

Implementation scope policy:
- Always apply changes to `artifacts/<slug>/kernel/kernel.py`.
- Keep `local_gpu` and `kaggle_gpu` algorithmically identical; only execution location differs.
- For non-tabular competitions (image/video/audio/text/document/medical-imaging/array/point-cloud/3D/geospatial/bio/sequence/graph/signal/annotation/model-artifact), implement/iterate `custom_main()` in `kernel.py`.
- If pretrained checkpoints are beneficial, add download + cache logic in `kernel.py` with internet-off fallback.

Anti-leakage policy for external data:
- External data may support pretraining, feature learning, or additional supervised training only when rules allow it.
- Never use external labels to directly assign competition test predictions through exact or near-duplicate matching on row IDs, filenames, hashes, timestamps, bounding boxes, or image/audio/text content.
- If a public/external dataset appears to contain labeled copies of the competition test rows, exclude that overlap from prediction logic, report it in diagnostics, and improve the generalizing model instead.
- Do not select pipelines named or behaving like `external_overlap_mapping`, `official_solution_mapping`, `label_transfer`, or `test_label_lookup`.

**Recommended Strategies** (pick based on diagnosis):

#### For Underfitting

**Add More Features**:
```python
# Polynomial features (be careful with high dimensions)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Interaction features (manual, interpretable)
df['feature_interaction'] = df['feature1'] * df['feature2']
df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-8)

# Domain-specific features (e.g., for house prices)
df['total_sqft'] = df['1stFlrSF'] + df['2ndFlrSF']
df['age'] = df['YrSold'] - df['YearBuilt']
```

**Use More Complex Model**:
```python
# Upgrade: Linear → Tree Ensemble → Gradient Boosting
# From Ridge/Logistic
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# To XGBoost (with GPU if available)
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    tree_method='gpu_hist' if '{compute_mode}' in ['local_gpu', 'kaggle_gpu'] else 'auto',
    random_state=42
)
```

**Reduce Regularization**:
```python
# For Ridge/Lasso: reduce alpha
model = Ridge(alpha=0.1)  # was 1.0

# For tree models: increase max_depth, reduce min_samples_leaf
model = RandomForestRegressor(max_depth=15, min_samples_leaf=1)
```

#### For Overfitting

**Add Regularization**:
```python
# For XGBoost
model = xgb.XGBRegressor(
    reg_alpha=1.0,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    gamma=0.1,          # Minimum loss reduction
    subsample=0.8,      # Row sampling
    colsample_bytree=0.8,  # Column sampling
    early_stopping_rounds=10
)
```

**Simplify Model**:
```python
# Reduce tree depth
model = RandomForestRegressor(max_depth=5, min_samples_leaf=10)

# Reduce number of estimators
model = xgb.XGBRegressor(n_estimators=50)
```

**Improve Cross-Validation**:
```python
# Use CV instead of holdout for more robust estimates
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
```

**Feature Selection**:
```python
# Remove highly correlated features
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Select top K features by importance
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=20)
X_selected = selector.fit_transform(X, y)
```

#### For Feature Engineering

**Better Missing Value Handling**:
```python
# Add missingness indicators
for col in df.columns:
    if df[col].isna().sum() > 0:
        df[f'{col}_is_missing'] = df[col].isna().astype(int)

# Model-based imputation
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=42)
df_imputed = imputer.fit_transform(df)
```

**Better Categorical Encoding**:
```python
# Target encoding for high-cardinality categoricals
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols=['high_cardinality_col'])
X_encoded = encoder.fit_transform(X, y)

# Frequency encoding
df['cat_freq'] = df['category'].map(df['category'].value_counts())
```

**Transform Skewed Features**:
```python
# Check skewness from dataset_profile.json, then transform
from scipy.stats import skew
skewed_features = [col for col in numeric_cols if abs(skew(df[col].dropna())) > 0.5]
for col in skewed_features:
    df[f'{col}_log'] = np.log1p(df[col])
```

#### For Hyperparameter Tuning

```python
# Grid search (if time budget allows)
from sklearn.model_selection import GridSearchCV
param_grid = {{
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3]
}}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

#### For Classification (Class Imbalance)

```python
# SMOTE for minority oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights
model = xgb.XGBClassifier(scale_pos_weight=ratio_negative/ratio_positive)

# Adjust decision threshold
y_pred_proba = model.predict_proba(X_val)[:, 1]
threshold = 0.3  # Tune based on precision-recall tradeoff
y_pred = (y_pred_proba >= threshold).astype(int)
```

#### For GPU/TPU Utilization

**GPU Optimization**:
```python
# Use GPU-enabled tree method
model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)

# Increase batch size for neural networks
model = tf.keras.Sequential([...])
model.fit(X_train, y_train, batch_size=1024, epochs=50)  # Larger batch
```

**TPU Optimization** (kaggle_tpu mode):
```python
# Use TensorFlow with TPU strategy
import tensorflow as tf
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile(...)
    model.fit(X_train, y_train, batch_size=128 * strategy.num_replicas_in_sync)
```

### Step 3: Validate Improvements

**Run Offline Evaluation**:
```bash
uv run kagglebot train {slug} --compute {compute_mode}
```

**Check Results**:
1. Did the score improve in the right direction?
   - minimize: new_score < old_score
   - maximize: new_score > old_score

2. Is the train-val gap reasonable?
   - Gap too large → overfitting
   - Both high → underfitting

3. Is GPU/TPU utilization acceptable?
   - GPU: >80%
   - TPU MXU: >70%

4. Did submission format stay valid?
   ```bash
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

**Run Tests**:
```bash
uv run pytest -q
```

### Step 4: Safety and Quality Checks

**CRITICAL SAFETY RULES**:
- ❌ NEVER automate rules acceptance or submission bypass
- ❌ NEVER commit secrets (kaggle.json, API keys, tokens)
- ❌ NEVER add interactive prompts (must be non-interactive)
- ❌ NEVER use test files for local validation split generation
- ❌ NEVER copy hidden/test labels from external labeled overlaps, exact file-hash matches, row-id mappings, or solution-like artifacts
- ❌ NEVER bypass safety guardrails
- ✅ DO keep changes incremental (1-2 focused improvements)
- ✅ DO document what was tried and why
- ✅ DO maximize GPU/TPU utilization
- ✅ DO use web search for documentation (never for secrets)

**Quality Checklist**:
- [ ] Changes address root cause from diagnostics
- [ ] Loop-decision score improves, or offline diagnostics provide actionable learning
- [ ] Train-val gap is reasonable (not overfitting)
- [ ] Submission artifact format still matches the required sample/format
- [ ] GPU/TPU utilization >80% (GPU) or >70% (TPU MXU)
- [ ] Tests pass: `uv run pytest -q`
- [ ] No secrets leaked into code/logs
- [ ] Change is documented in code comments

---

## Acceptance Criteria

Your improvement will be accepted if:

1. ✅ **Tests pass**: `uv run pytest -q` returns 0 exit code
2. ✅ **Decision score valid**: metrics.json has loop-decision and supporting offline metrics
3. ✅ **Submission valid**: submission artifact still matches format
4. ✅ **GPU/TPU utilized**: Utilization >80% (GPU) or >70% (TPU) logged
5. ✅ **No errors**: Training completes without exceptions
6. ✅ **Progress logged**: Diagnostics updated with change summary

If score doesn't improve, document what was tried and why it didn't work (learning for next iteration).

---

## Tips for Effective Improvements

**Strategy**:
1. **Make one change at a time**: Easier to debug and understand impact
2. **Optimize loop decision**: prioritize submission score when available; use offline as fallback
3. **Read diagnostics carefully**: They contain actionable hints
4. **Check for data leakage**: If score is "too good", investigate
5. **Don't over-optimize**: Diminishing returns after 3-5 iterations
6. **Document failures**: Failed experiments provide valuable learning

**Common Mistakes**:
- Changing too many things at once (can't isolate impact)
- Ignoring train-val gap (missing over/underfitting signals)
- Not monitoring resource usage (wasting GPU/TPU)
- Repeating failed experiments from previous iterations
- Breaking submission format (always validate against sample)

**Debugging**:
- If improvement doesn't help: Revert and try different approach
- If tests fail: Read error messages, check for syntax/import errors
- If GPU utilization low: Profile code, increase batch size
- If score regresses: Check for bugs in new code

**Time Management**:
- You have {time_remaining} minutes remaining (approximately)
- Focus on high-impact changes first
- If close to time limit: Ensure current iteration completes cleanly

Good luck improving the model! 🔧
