# Taxonomy

The taxonomy defines the controlled vocabulary of tags used to categorize Kaggle competitions in the Knowledge Base. It enables similarity search by standardizing how we describe competitions.

**Location**: `knowledge/taxonomy.yml`

## Purpose

The taxonomy serves three goals:

1. **Consistency**: Everyone uses the same tags (`binary` vs `binary_classification`)
2. **Similarity search**: Find competitions with overlapping characteristics
3. **Agent context**: Help the agent understand competition type and choose appropriate strategies

## Structure

The taxonomy is organized into **categories** of related tags:

```yaml
# Data Modality
data_modality:
  - tabular          # CSV/Parquet tables
  - text             # NLP tasks
  - image            # Computer vision
  - timeseries       # Sequential data
  - multi_modal      # Multiple data types

# Problem Type
problem_type:
  - regression       # Predict continuous values
  - binary           # Two-class classification
  - multiclass       # 3+ class classification

# Aliases (common variations)
aliases:
  binary_classification: binary
  nlp: text
```

### Categories

The full taxonomy includes:

- **`data_modality`**: Type of input data (tabular, text, image, etc.)
- **`problem_type`**: ML task category (regression, binary, multiclass, etc.)
- **`n_rows`**: Dataset size by row count (tiny, small, medium, large, huge)
- **`n_features`**: Dataset size by feature count (few, medium, many, wide)
- **`data_characteristics`**: Notable properties (missing values, high cardinality, class imbalance, etc.)
- **`domain`**: Subject matter (finance, healthcare, ecommerce, etc.)
- **`metric`**: Primary competition metric (rmse, accuracy, auc_roc, etc.)
- **`techniques`**: Successful approaches (tree_ensemble, feature_engineering, stacking, etc.)

See `knowledge/taxonomy.yml` for the complete list.

## Tag Lifecycle

### 1. Auto-Tagging (Bootstrap)

When you bootstrap a competition, tags are inferred automatically from the dataset profile:

```python
# Dataset profiling extracts statistics
profile = {
    "train_rows": 1000,          # → n_rows_small
    "task": "binary",            # → binary
    "modality": "tabular",       # → tabular
    "missingness": 0.25,         # → missingness_high
}
```

**Inference rules** in `taxonomy.yml` map statistics to tags:

```yaml
inference_rules:
  n_rows:
    n_rows_small:
      - row_count: ">= 1000 and < 10000"
```

### 2. Manual Tagging

Add domain-specific or technique tags manually:

```bash
# Edit meta.json
echo '{
  "slug": "titanic",
  "tags": ["tabular", "binary", "n_rows_small", "survival_analysis", "tree_ensemble"]
}' > artifacts/titanic/meta.json

# Re-sync to KB
uv run kagglebot bootstrap titanic
```

### 3. Tag Normalization

All tags pass through normalization:
- Resolve aliases: `binary_classification` → `binary`
- Filter invalid tags: Unknown tags are dropped with a warning
- Deduplicate: `["tabular", "tabular"]` → `["tabular"]`

## Extending the Taxonomy

### When to Add a New Tag

Add a tag when:
- **It affects solution strategy**: E.g., `imbalanced_classes` suggests SMOTE or class weights
- **It enables meaningful similarity**: E.g., `finance` helps find similar domain competitions
- **It captures a reusable lesson**: E.g., `tree_ensemble` worked well

**Don't add** overly specific tags:
- ❌ `titanic` (competition-specific)
- ❌ `rmse_below_0.1` (too specific)
- ✅ `survival_analysis` (generalizable domain)

### How to Add a Tag

1. **Edit `knowledge/taxonomy.yml`**:

```yaml
domain:
  - finance
  - healthcare
  - retail          # <-- Add new tag here
```

2. **Add aliases if needed**:

```yaml
aliases:
  ecommerce: retail  # <-- Map common variation
```

3. **Add inference rule** (optional, for auto-tagging):

```yaml
inference_rules:
  domain:
    retail:
      - column_names: ["product_id", "customer_id"]  # Heuristic
```

4. **Test**:

```bash
uv run kagglebot bootstrap my-competition
# Check artifacts/my-competition/meta.json for tags
```

5. **Document** (update this file if adding a new category):

Add a comment in `taxonomy.yml` explaining the new tag's purpose.

## Best Practices

### 1. Use Canonical Tags

Prefer the canonical form over aliases:
- ✅ `binary` (canonical)
- ❌ `binary_classification` (alias)

Aliases are for convenience when manually tagging, but the KB stores canonical forms.

### 2. Keep Tags Broad

Tags should generalize to multiple competitions:
- ✅ `healthcare` (applies to many competitions)
- ❌ `breast_cancer_wisconsin` (too specific)

### 3. Combine Tags

Use multiple tags to describe nuanced competitions:
- `["text", "multiclass", "imbalanced_classes"]` → Multi-label sentiment analysis with class imbalance
- `["tabular", "timeseries", "forecasting"]` → Time series forecasting on tabular data

### 4. Add Technique Tags Post-Run

After autopilot completes, add tags for what worked:
```yaml
techniques:
  - tree_ensemble       # XGBoost won
  - feature_engineering # Heavy feature engineering helped
  - stacking            # Ensemble of models
```

Future similar competitions will benefit from these learnings.

### 5. Update Aliases for Common Variations

If you find yourself typing a variation frequently, add an alias:

```yaml
aliases:
  cv: image              # "cv" is shorter than "computer_vision"
  ts: timeseries         # "ts" is common abbreviation
```

## Tag Inference Rules

Auto-tagging uses heuristics defined in `taxonomy.yml`:

### Example: Row Count

```yaml
inference_rules:
  n_rows:
    n_rows_tiny:
      - row_count: "< 1000"
    n_rows_small:
      - row_count: ">= 1000 and < 10000"
    n_rows_medium:
      - row_count: ">= 10000 and < 100000"
    n_rows_large:
      - row_count: ">= 100000 and < 1000000"
    n_rows_huge:
      - row_count: ">= 1000000"
```

### Example: Problem Type

```yaml
inference_rules:
  problem_type:
    binary:
      - target_unique_count: "== 2"
      - target_type: "categorical"
    regression:
      - target_type: "numeric"
      - target_unique_count: "> 20"
```

### Adding New Inference Rules

If you want a tag to be auto-inferred:

1. Identify a dataset characteristic that signals the tag
2. Add a rule to `inference_rules`:

```yaml
inference_rules:
  data_characteristics:
    temporal_leakage:
      - has_date_columns: true
      - test_date_range_after_train: true
```

3. Update the profiling logic in `src/kagglebot/knowledge/__init__.py` if needed

## Examples

### Minimal Taxonomy (Bootstrap)

```yaml
data_modality:
  - tabular

problem_type:
  - regression
  - binary

n_rows:
  - n_rows_small

aliases: {}
```

### Comprehensive Taxonomy (Production)

See `knowledge/taxonomy.yml` for the full production taxonomy with:
- 8 categories
- 100+ tags
- 30+ aliases
- Inference rules for auto-tagging

### Custom Domain Taxonomy

```yaml
# Extend with finance-specific tags
domain:
  - finance

finance_subtypes:
  - stock_prediction
  - fraud_detection
  - credit_scoring
  - algorithmic_trading

aliases:
  stocks: stock_prediction
  fraud: fraud_detection
```

## Migration Guide

If you're upgrading from the old JSON format:

### Old Format (JSON)

```json
{
  "tags": ["tabular", "text", "image"],
  "aliases": {"binary_classification": "binary"}
}
```

### New Format (YAML)

```yaml
data_modality:
  - tabular
  - text
  - image

aliases:
  binary_classification: binary
```

**Migration steps**:
1. Back up `knowledge/taxonomy.yml`
2. Replace with new YAML format
3. Re-run `kagglebot bootstrap` on all existing competitions
4. Verify tags in KB: `uv run kagglebot knowledge show <slug>`

## Troubleshooting

### "Unknown tag" Warning

**Problem**: Bootstrap warns about unknown tags in `meta.json`.

**Solution**: Add the tag to `taxonomy.yml` or remove it from `meta.json`.

### Tag Not Auto-Inferred

**Problem**: Expected tag not appearing after bootstrap.

**Possible causes**:
1. Inference rule threshold not met (e.g., missing values <20%, not 20%+)
2. Inference rule not defined for this tag
3. Dataset characteristic not detected during profiling

**Solution**: Manually add the tag to `meta.json` or adjust inference rules.

### Alias Not Working

**Problem**: Using `binary_classification` but KB stores `binary_classification`, not `binary`.

**Cause**: Normalization not applied.

**Solution**: Ensure you re-ran bootstrap after adding the alias.

## See Also

- [docs/knowledge.md](knowledge.md) - Knowledge Base system that uses tags
- [docs/autopilot.md](autopilot.md) - How autopilot uses similarity search
- `knowledge/taxonomy.yml` - Complete taxonomy specification
