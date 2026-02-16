# Knowledge Base

The Knowledge Base (KB) stores cross-competition learnings to help autopilot make better decisions. It records what worked (and what didn't) across different competitions, enabling the agent to learn from past experience.

**Key principle**: The KB stores only metadata and summaries, never raw data, predictions, or secrets.

## Purpose

The KB answers questions like:
- **"What similar competitions have we run before?"** → Find competitions with overlapping tags
- **"What improvements worked on similar tasks/datasets?"** → Surface successful strategies
- **"How did we perform on datasets with high missing values?"** → Learn from past challenges
- **"What errors happened before, and how were they fixed?"** → Reuse proven fixes

During autopilot bootstrap, the KB is queried to find similar competitions and inject successful improvement strategies into the initial-model prompt.
Problem/error insights are persisted only after a submission result (online score) is known, and each record is labeled as `good` or `low`.

## Storage

- **Location**: `knowledge/kb.sqlite` (SQLite database)
- **Taxonomy**: `knowledge/taxonomy.yml` (controlled tag vocabulary)
- **Persistent research artifacts**:
  - `knowledge/research/<problem_type>/<slug>/research_sources.jsonl`
  - `knowledge/research/<problem_type>/<slug>/research_summary.md`
- **Size**: Typically <10MB for hundreds of competitions (only metadata)

## Database Schema

The KB uses a simple relational schema:

### `competitions`

```sql
CREATE TABLE competitions (
    slug TEXT PRIMARY KEY,
    url TEXT,
    metric TEXT,
    task_type TEXT,
    created_at TIMESTAMP,
    last_seen_at TIMESTAMP
);
```

Stores basic competition metadata. Updated on first bootstrap and every subsequent run.

### `tags`

```sql
CREATE TABLE tags (
    tag TEXT PRIMARY KEY
);
```

All valid tags from `taxonomy.yml`. Normalized to canonical form (aliases resolved).

### `competition_tags`

```sql
CREATE TABLE competition_tags (
    slug TEXT,
    tag TEXT,
    PRIMARY KEY (slug, tag),
    FOREIGN KEY (slug) REFERENCES competitions(slug),
    FOREIGN KEY (tag) REFERENCES tags(tag)
);
```

Many-to-many relationship between competitions and tags. Enables similarity search by tag overlap.

### `runs`

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    slug TEXT,
    started_at TIMESTAMP,
    compute TEXT,
    goal_metric TEXT,
    goal_score REAL,
    direction TEXT,
    FOREIGN KEY (slug) REFERENCES competitions(slug)
);
```

Each autopilot run is recorded with its configuration (compute environment, target metric/score).

### `iterations`

```sql
CREATE TABLE iterations (
    run_id TEXT,
    iteration INTEGER,
    score_source TEXT,
    offline_value REAL,
    offline_std REAL,
    top1_public_score REAL,
    met_target BOOLEAN,
    created_at TIMESTAMP,
    PRIMARY KEY (run_id, iteration),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

Per-iteration metrics. Records offline performance, not predictions. Used to track convergence and best scores.

### `improvements`

```sql
CREATE TABLE improvements (
    run_id TEXT,
    iteration INTEGER,
    summary TEXT,
    delta_offline REAL,
    created_at TIMESTAMP,
    PRIMARY KEY (run_id, iteration),
    FOREIGN KEY (run_id, iteration) REFERENCES iterations(run_id, iteration)
);
```

Agent-generated summaries of what was changed and the score delta. Most valuable for similarity search.

### `competition_research`

Stores one canonical research artifact location per competition:
- `primary_problem_type` for folder classification
- `problem_types_json` with all inferred types
- relative paths to `research_sources.jsonl` and `research_summary.md`
- create/update timestamps

### `research_problem_types`

Lookup table mapping each competition slug to one or more problem types.
Used for cross-competition retrieval by overlapping problem type.

### `problem_type_insights`

Stores per-problem-type lessons with submission outcome labels:
- why the score was poor (`why_poor`)
- what changed (`how_improved`)
- final outcome bucket (`good` / `low`)
- known online submission score

### `error_fix_insights`

Stores error-resolution knowledge:
- error category + normalized error text
- fix summary
- whether the error was resolved
- final submission outcome bucket and online score

Example:
```json
{
  "summary": "Added polynomial features (degree 2), improved RMSE by 0.03",
  "delta_offline": -0.03
}
```

## Tagging

Tags are the primary mechanism for similarity search. They're generated automatically from dataset profiles during bootstrap.

### Tag Sources

1. **Dataset profiling**: Inferred from `train.csv` statistics
   - Modality: `tabular`, `text`, `image`, `timeseries`
   - Task: `regression`, `binary`, `multiclass`
   - Size: `n_rows_small`, `n_rows_medium`, `n_rows_large`
   - Characteristics: `missingness_high`, `high_cardinality_cats`

2. **Manual addition**: Edit `artifacts/<slug>/meta.json` to add domain-specific tags
   - Example: `finance`, `healthcare`, `nlp`

3. **Post-run tagging**: Add technique tags after autopilot completes
   - Example: `tree_ensemble`, `feature_engineering`

### Tag Normalization

All tags pass through the taxonomy to:
- Resolve aliases: `binary_classification` → `binary`
- Filter invalid tags: Unknown tags are dropped
- Canonical ordering: Deterministic sort

### Similarity Metric

Competitions are ranked by **tag overlap count**:

```sql
SELECT slug, COUNT(*) as overlap
FROM competition_tags
WHERE tag IN ('tabular', 'binary', 'n_rows_small')
GROUP BY slug
ORDER BY overlap DESC
LIMIT 3
```

Higher overlap = more similar competition = more relevant learnings.

## What the KB Does NOT Store

For privacy and safety:
- ❌ Raw competition data (train.csv, test.csv)
- ❌ Predictions or submission files
- ❌ Kaggle credentials or API keys
- ❌ Trained model weights
- ❌ Personal identifiable information

Only metadata, tags, and agent-generated summaries.

## CLI Usage

### Show Competition Details

```bash
uv run kagglebot knowledge show titanic
```

Output:
- Competition metadata (slug, metric, task)
- Tags (modality, task, size, characteristics)
- All runs with their configurations
- Iterations with offline scores
- Improvements and summaries

### Search by Tags

```bash
uv run kagglebot knowledge search --tag tabular --tag binary --limit 5
```

Output:
- Competitions ranked by tag overlap
- Improvement summaries from those competitions
- Best offline scores achieved

Use this to:
- Find similar competitions before starting a new one
- Learn what strategies worked on similar problems
- Avoid repeating failed experiments

### Add Manual Tags

Edit `artifacts/<slug>/meta.json`:

```json
{
  "slug": "titanic",
  "tags": ["tabular", "binary", "n_rows_small", "finance", "survival_analysis"]
}
```

Then re-run bootstrap to sync to KB:

```bash
uv run kagglebot bootstrap titanic
```

## KB Lifecycle

### 1. Bootstrap (First Run)

When you bootstrap a new competition:
1. Dataset is profiled → tags inferred
2. KB is queried for similar competitions (by tag overlap)
3. Top-K similar competitions' improvements are injected into the initial-model prompt
4. Competition metadata is inserted into KB

### 2. Autopilot Run

During autopilot:
1. Run config is recorded in `runs` table
2. After each iteration:
   - Metrics are recorded in `iterations` table
   - If improvement happened, summary is recorded in `improvements` table
3. No KB queries happen during iteration (all context loaded at bootstrap)

### 3. Post-Run

After autopilot completes:
- You can manually add technique tags (e.g., `tree_ensemble` if XGBoost worked)
- Future runs on similar competitions will benefit from your learnings

## Example Workflow

### Scenario: New Competition (House Prices)

```bash
# Bootstrap and autopilot
uv run kagglebot autopilot house-prices --compute local_gpu
```

**Behind the scenes**:
1. KB query finds similar competitions:
   - `titanic` (overlap: 3 tags - `tabular`, `regression`, `n_rows_small`)
   - `bike-sharing` (overlap: 2 tags - `tabular`, `regression`)
2. Improvements from those competitions are included in the initial-model prompt:
   - "titanic: Added polynomial features (degree 2), improved RMSE by 0.03"
   - "bike-sharing: Log-transformed target, improved RMSE by 0.05"
3. Agent uses these hints to create a stronger initial model for `house-prices`

### Scenario: Manual Tagging

After finishing a competition, add technique tags:

```bash
# Edit meta.json to add tags
echo '{
  "slug": "house-prices",
  "tags": ["tabular", "regression", "n_rows_small", "tree_ensemble", "feature_engineering"]
}' > artifacts/house-prices/meta.json

# Re-sync to KB
uv run kagglebot bootstrap house-prices
```

Now, future `regression` competitions with `n_rows_small` will learn that `tree_ensemble` and `feature_engineering` worked.

## Maintenance

### Reset KB (Clean Slate)

```bash
rm knowledge/kb.sqlite
```

The KB will be recreated on next bootstrap.

### Export KB to JSON

```bash
sqlite3 knowledge/kb.sqlite ".dump" > kb_backup.sql
```

### Inspect KB Directly

```bash
sqlite3 knowledge/kb.sqlite

.tables
.schema competitions
SELECT * FROM competitions LIMIT 5;
```

## Future Enhancements

Potential improvements to the KB:

1. **Ensemble suggestions**: If stacking worked on similar competitions, suggest it
2. **Hyperparameter recommendations**: Store best hyperparameters by task type
3. **Feature engineering patterns**: Abstract feature engineering into reusable templates
4. **Time-series specific**: Store lag/window settings for time series tasks
5. **Multi-modal competitions**: Better handling for image+text or tabular+text tasks

## Privacy & Safety

The KB is designed to be **shareable**:
- No competition data (only metadata)
- No credentials or secrets
- No model weights (only summaries)
- No PII

You could share your `kb.sqlite` with teammates to pool learnings across multiple Kaggle accounts.

## See Also

- [docs/taxonomy.md](taxonomy.md) - Tag vocabulary and extension guide
- [docs/autopilot.md](autopilot.md) - How autopilot uses the KB
