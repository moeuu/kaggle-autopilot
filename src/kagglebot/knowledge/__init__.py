from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kagglebot.paths import KnowledgePaths
from kagglebot.solver.io import find_competition_files, infer_target, infer_task


@dataclass(frozen=True)
class Taxonomy:
    tags: set[str]
    aliases: dict[str, str]

    def normalize(self, tags: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        for tag in tags:
            key = self.aliases.get(tag, tag)
            if key in self.tags and key not in normalized:
                normalized.append(key)
        return normalized

    def to_dict(self) -> dict[str, object]:
        return {"tags": sorted(self.tags), "aliases": dict(self.aliases)}


def ensure_taxonomy(paths: KnowledgePaths) -> dict[str, object]:
    paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
    if not paths.taxonomy_path.exists():
        taxonomy = Taxonomy(
            tags={
                "tabular",
                "text",
                "image",
                "timeseries",
                "regression",
                "binary",
                "multiclass",
                "n_rows_small",
                "n_rows_medium",
                "n_rows_large",
                "missingness_high",
                "high_cardinality_cats",
            },
            aliases={
                "binary_classification": "binary",
                "multiclass_classification": "multiclass",
            },
        )
        paths.taxonomy_path.write_text(json.dumps(taxonomy.to_dict(), indent=2), encoding="utf-8")
    return load_taxonomy(paths.taxonomy_path)


def load_taxonomy(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _parse_yaml_taxonomy(text)


def _parse_yaml_taxonomy(text: str) -> dict[str, object]:
    tags: set[str] = set()
    aliases: dict[str, str] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
            stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and stripped.endswith(":"):
            current = stripped[:-1]
            continue
        if current is None:
            continue
        if current == "aliases":
            if ":" in stripped:
                alias_key, alias_val = stripped.split(":", 1)
                alias_key = alias_key.strip()
                alias_val = alias_val.strip().strip('"').strip("'")
                if alias_key and alias_val:
                    aliases[alias_key] = alias_val
            continue
        if current == "inference_rules":
            continue
        if stripped.startswith("-"):
            item = stripped[1:].strip().strip('"').strip("'")
            if item:
                tags.add(item)
    return {"tags": sorted(tags), "aliases": aliases}


def _taxonomy_from_dict(payload: dict[str, object]) -> Taxonomy:
    tags = set(payload.get("tags", []) or [])
    aliases = payload.get("aliases", {}) or {}
    return Taxonomy(tags=tags, aliases=dict(aliases))


def build_dataset_profile(data_dir: Path) -> dict[str, object]:
    profile: dict[str, object] = {"data_dir": str(data_dir)}
    csvs = list(data_dir.rglob("*.csv"))
    if not csvs:
        profile["status"] = "missing_data"
        profile["tags"] = []
        return profile

    try:
        train_path, test_path, sample_path = find_competition_files(data_dir)
    except FileNotFoundError as exc:
        profile["status"] = "missing_required_files"
        profile["error"] = str(exc)
        profile["tags"] = []
        return profile

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    id_col, target_col, feature_cols = infer_target(train, test, sample)
    task = infer_task(train[target_col])
    n_rows = len(train)
    n_cols = len(train.columns)
    missingness = float(train.isna().mean().mean())
    missingness_by_column = {col: float(val) for col, val in train.isna().mean().items()}
    dtype_by_column = {col: str(dtype) for col, dtype in train.dtypes.items()}
    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
    high_cardinality = [c for c in cat_cols if train[c].nunique(dropna=True) > 50]

    modality = _infer_modality(data_dir, train)
    task_tag = _task_tag(task, train[target_col])
    size_tag = _size_tag(n_rows)

    tags = [modality, task_tag, size_tag]
    if missingness > 0.2:
        tags.append("missingness_high")
    if high_cardinality:
        tags.append("high_cardinality_cats")

    metric = "accuracy" if task != "regression" else "rmse"
    target_stats = _target_stats(train[target_col], task)

    profile.update(
        {
            "status": "ok",
            "train_file": train_path.name,
            "test_file": test_path.name,
            "sample_submission_file": sample_path.name,
            "train_rows": n_rows,
            "train_cols": n_cols,
            "test_rows": len(test),
            "test_cols": len(test.columns),
            "id_column": id_col,
            "target_column": target_col,
            "task": task,
            "metric": metric,
            "missingness": missingness,
            "missingness_by_column": missingness_by_column,
            "dtype_by_column": dtype_by_column,
            "categorical_columns": cat_cols,
            "numeric_columns": [c for c in feature_cols if c not in cat_cols],
            "high_cardinality_columns": high_cardinality,
            "modality": modality,
            "tags": tags,
            "target_stats": target_stats,
        }
    )
    return profile


def _target_stats(target: pd.Series, task: str) -> dict[str, object]:
    stats: dict[str, object] = {}
    if task == "regression":
        values = pd.to_numeric(target, errors="coerce").dropna()
        if not values.empty:
            stats["min"] = float(values.min())
            stats["max"] = float(values.max())
            stats["mean"] = float(values.mean())
            stats["std"] = float(values.std(ddof=0))
            skew = float(values.skew())
            stats["skew"] = None if pd.isna(skew) else skew
    else:
        counts = target.value_counts(dropna=False)
        stats["unique"] = int(counts.size)
        if counts.sum() > 0:
            stats["top_class_ratio"] = float(counts.iloc[0] / counts.sum())
    return stats


def _infer_modality(data_dir: Path, train: pd.DataFrame) -> str:
    image_exts = {".jpg", ".jpeg", ".png"}
    if any(path.suffix.lower() in image_exts for path in data_dir.rglob("*")):
        return "image"
    text_cols = [c for c in train.columns if train[c].dtype == "object"]
    if text_cols:
        avg_len = 0.0
        sample = train[text_cols].astype(str).head(200)
        if not sample.empty:
            avg_len = sample.apply(lambda col: col.map(len)).mean().mean()
        if avg_len >= 30:
            return "text"
    if any("date" in c.lower() or "time" in c.lower() for c in train.columns):
        return "timeseries"
    return "tabular"


def _task_tag(task: str, target: pd.Series) -> str:
    if task == "regression":
        return "regression"
    unique = target.nunique(dropna=True)
    return "binary" if unique <= 2 else "multiclass"


def _size_tag(n_rows: int) -> str:
    if n_rows < 10_000:
        return "n_rows_small"
    if n_rows < 100_000:
        return "n_rows_medium"
    return "n_rows_large"


def build_plan_and_baseline_prompt(
    *,
    slug: str,
    rules_url: str,
    profile: dict[str, object],
    taxonomy: dict[str, object],
    similar_improvements: list[dict[str, object]],
) -> str:
    tags = profile.get("tags", [])
    task = profile.get("task", "unknown")
    metric = profile.get("metric", "rmse")
    n_rows = profile.get("train_rows", 0)
    n_cols = profile.get("train_cols", 0)

    lines = [
        "# Kagglebot Codex: Plan + Baseline",
        "",
        "## Competition Overview",
        "",
        f"**Slug**: {slug}",
        "**Competition URL**: {{competition_url}}",
        f"**Rules URL**: {rules_url}",
        f"**Task**: {task}",
        f"**Metric (confirm via rules)**: {metric}",
        f"**Dataset**: {n_rows:,} rows × {n_cols} columns",
        f"**Tags**: {', '.join(tags) if tags else 'None'}",
        "",
        "## Compute Context",
        "",
        "- Compute mode: {{compute_mode}}",
        "- Accelerator: {{accelerator}}",
        "- Internet: {{internet}}",
        "- Run ID: {{run_id}}",
        "",
        "## Context Files (read these)",
        "",
        f"- artifacts/{slug}/context/dataset_profile.json",
        f"- artifacts/{slug}/context/sample_submission.csv",
        f"- artifacts/{slug}/context/sample_submission_head.csv",
        f"- artifacts/{slug}/context/top1_public.json",
        f"- artifacts/{slug}/context/rules_url.txt",
        f"- artifacts/{slug}/context/rules.md (if present)",
        f"- artifacts/{slug}/context/rules.html (if present)",
        f"- artifacts/{slug}/kernel_overrides.py (for kaggle_gpu/kaggle_tpu)",
        f"- artifacts/{slug}/context/knowledge_hints.txt",
        "",
        "## Knowledge Base: Similar Competitions",
        "",
    ]

    if similar_improvements:
        lines.append("We found past runs on similar competitions. Learn from what worked:")
        lines.append("")
        for item in similar_improvements:
            overlap = item.get("overlap", 0)
            summary = item.get("summary", "No summary")
            comp_slug = item.get("slug", "unknown")
            lines.append(f"- {comp_slug} ({overlap} tag overlap): {summary}")
    else:
        lines.append("No similar competitions found in knowledge base (this is the first run).")

    lines += [
        "",
        "---",
        "",
        "## Your Task",
        "",
        "### 1) Decide the plan (required)",
        "",
        f"Update artifacts/{slug}/plan.json with:",
        "",
        "```json",
        "{",
        '  "target_metric": "<derive_from_rules>",',
        '  "target_score": 0.0,',
        '  "target_direction": "<minimize|maximize>",',
        '  "score_source": "holdout",',
        '  "holdout_frac": 0.2,',
        '  "cv_folds": 5,',
        '  "seed": 42,',
        '  "internet": "on",',
        '  "max_iterations": 5,',
        '  "submit_policy": "on_target_only"',
        "}",
        "```",
        "",
        "Guidance:",
        "- Derive target_metric and direction from rules.md/rules.html and sample_submission.csv.",
        "- Use top1_public.json to set a realistic target_score; avoid generic metric heuristics.",
        "- Prefer CV for small datasets or high variance; otherwise holdout.",
        "- Do NOT change submit_policy; autopilot controls submission gating.",
        "",
        "### 2) Implement baseline pipeline",
        "",
        "Implement a robust baseline in kagglebot/solver/ that:",
        "- Loads train/test and sample_submission correctly.",
        "- Handles missing values and encodes categoricals.",
        "- Trains a Torch-based supervised baseline by default.",
        "- Falls back to a simpler linear model only when data/rules justify it.",
        "- Evaluates with the score_source from plan.json.",
        "- Writes submission.csv matching sample_submission.csv exactly.",
        "",
        "Compute-specific notes:",
        "- local_cpu/local_gpu: update kagglebot/solver/ as usual.",
        "- kaggle_gpu/kaggle_tpu: implement model + features in artifacts/{slug}/kernel_overrides.py.",
        "- Do NOT edit kernel_runner.py for competition-specific changes.",
        "",
        "### 3) Safety + verification",
        "",
        "- NEVER accept Kaggle rules via API.",
        "- NEVER include secrets in code/logs.",
        "- NO interactive prompts.",
        "",
        "Run tests before finishing:",
        "```bash",
        "uv run pytest -q",
        "```",
        "",
        "The autopilot will iterate up to 5 times and submit only when top1-tier or at iteration 5.",
    ]
    return "\n".join(lines) + "\n"


def build_improve_template() -> str:
    """Build the improvement prompt template for iterations 1-N.

    This template has placeholders that get filled with .format() at runtime:
    - {slug}, {iteration}, {plan_path}, {run_path}, {metrics_path}, {diagnostics_path}, {logs_dir}
    """
    return """\
# Kagglebot Codex: Improvement Iteration

## Context

**Competition**: `{slug}`
**Iteration**: {iteration}
**Goal**: Improve offline score toward top1-tier or best possible within 5 iterations
**Compute**: {compute} ({accelerator})

## Current Score Context

- **Metric**: {metric} ({direction})
- **Current score**: {current_score}
- **Target score**: {target_score}
- **Top1 public**: {top1_score} (source: {top1_source})

## Input Files

- **Plan**: `{plan_path}` - Target metric/score/direction, evaluation strategy
- **Run Config**: `{run_path}` - Current run settings and history
- **Current Metrics**: `{metrics_path}` - Latest offline evaluation results
- **Diagnostics**: `{diagnostics_path}` - Agent-readable analysis of current performance
- **Logs**: `{logs_dir}` - Training logs and error messages (if any)
- **Knowledge Hints**: `{knowledge_hints}`
- **Rules URL**: `{rules_url}`
- **Rules Markdown**: `{rules_md}` (preferred)
- **Rules HTML**: `{rules_html}` (fallback)
- **Dataset Profile**: `{dataset_profile}`
- **Sample Submission**: `{sample_submission}`
- **Kernel Overrides**: `{kernel_overrides}` (use this for kaggle_gpu/kaggle_tpu)

## Your Task

### Step 1: Analyze Current Performance

Read the diagnostics and metrics files to understand:

1. **What is the current score?**
   - Check `metrics.json` for `value`, `direction`, and `target`
   - Compare to `target_score` in `plan.json`

2. **Why is the score not meeting target?**
   - Underfitting? (high train + val error → model too simple)
   - Overfitting? (low train error, high val error → model too complex or bad CV)
   - Data leakage? (unrealistically good scores)
   - Poor feature engineering? (missing important signals)
   - Wrong hyperparameters? (learning rate, regularization, tree depth)
   - Class imbalance? (classification only)
   - Bad splits? (e.g., time series shuffled incorrectly)

3. **What has been tried before?**
   - Check `{run_path}` for previous iteration summaries
   - Don't repeat failed experiments

### Step 2: Implement Improvements

Make **targeted, incremental changes** to improve the offline score.

**Where to implement**:
- If `compute` starts with `kaggle_`: edit `{kernel_overrides}` only.
- Otherwise: edit `kagglebot/solver/`.

**Modeling policy**:
- Use rules.md + dataset_profile.json to decide the best supervised approach.
- Prefer Torch-based models by default.
- Use simpler linear models only when data/rules make them clearly better.
- Avoid generic metric heuristics; derive metric choices from the rules.

**Recommended strategies** (pick 1-2 per iteration):

**For Underfitting**:
- Add more features (interactions, polynomials, domain-specific)
- Use a more complex model (Linear → Tree Ensemble → Neural Network)
- Reduce regularization (lower L1/L2 penalties, increase max_depth)
- Increase training time (more epochs, trees)

**For Overfitting**:
- Add regularization (L1/L2, dropout, early stopping)
- Reduce model complexity (fewer trees, smaller max_depth, fewer layers)
- Improve cross-validation (more folds, stratified splits, grouped CV)
- Feature selection (drop noisy/redundant features)
- Data augmentation (if applicable)

**For Feature Engineering**:
- Handle missing values better (indicators, model-based imputation)
- Encode categoricals differently (target encoding, embeddings)
- Create domain-specific features (e.g., for time series: lags, rolling stats)
- Remove highly correlated features
- Log/sqrt transform skewed features

**For Hyperparameter Tuning**:
- Grid search or random search over key parameters
- Tune learning rate, max_depth, n_estimators, regularization
- Use appropriate loss function for the metric

**For Classification**:
- Handle class imbalance (SMOTE, class weights, stratified sampling)
- Adjust decision threshold (precision-recall tradeoff)
- Try different metrics during training (focal loss for imbalance)

**For Ensembling** (later iterations only):
- Blend predictions from multiple models
- Stack models with a meta-learner
- Average across CV folds

### Step 3: Validate Changes

Before finalizing:

1. **Run offline evaluation**:
   ```bash
   uv run kagglebot train {{slug}} --compute local_cpu
   ```

2. **Check metrics**:
   - Is the score improving in the right direction?
   - Is train-val gap reasonable? (not too large = overfitting)

3. **Run tests**:
   ```bash
   uv run pytest -q
   ```

### Step 4: Safety Checks

**CRITICAL SAFETY RULES**:
- ❌ NEVER automate rules acceptance or submission
- ❌ NEVER commit secrets (kaggle.json, API keys)
- ❌ NEVER add interactive prompts
- ❌ NEVER bypass safety guardrails (duplicate checks, rate limits)
- ✅ DO keep changes incremental and testable
- ✅ DO preserve backward compatibility with existing CLI
- ✅ DO document significant changes in code comments

**Quality Checklist**:
- [ ] Changes address root cause from diagnostics
- [ ] Offline score improves (or provides learning for next iteration)
- [ ] submission.csv format still matches sample_submission.csv
- [ ] Tests pass: `uv run pytest -q`
- [ ] No secrets leaked into code/logs

---

## Tips for Effective Improvements

1. **Make one change at a time**: Easier to debug and understand what worked
2. **Trust offline evaluation**: Don't chase public leaderboard scores
3. **Read diagnostics carefully**: The autopilot generates actionable hints
4. **Check for data leakage**: If score is "too good to be true", it probably is
5. **Don't over-optimize**: Diminishing returns after 3-5 iterations are normal
6. **Document what didn't work**: Failed experiments provide valuable learning

Good luck improving the model! 🔧
"""


def resolve_similar_improvements(
    *,
    knowledge_paths: KnowledgePaths,
    taxonomy: dict[str, object],
    tags: Iterable[str],
    limit: int = 3,
) -> list[dict[str, object]]:
    tax = _taxonomy_from_dict(taxonomy)
    normalized = tax.normalize(tags)
    if not normalized:
        return []
    _ensure_db(knowledge_paths)
    with _connect(knowledge_paths.kb_path) as conn:
        placeholders = ",".join("?" for _ in normalized)
        rows = conn.execute(
            f"""
            SELECT slug, COUNT(*) as overlap
            FROM competition_tags
            WHERE tag IN ({placeholders})
            GROUP BY slug
            ORDER BY overlap DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            slug = row["slug"]
            overlap = row["overlap"]
            improvement = conn.execute(
                """
                SELECT summary, delta_offline
                FROM improvements
                JOIN runs ON improvements.run_id = runs.run_id
                WHERE runs.slug = ?
                ORDER BY delta_offline DESC, created_at DESC
                LIMIT 1
                """,
                (slug,),
            ).fetchone()
            summary = improvement["summary"] if improvement else "No improvement summary recorded."
            results.append({"slug": slug, "overlap": overlap, "summary": summary})
        return results


def record_competition_profile(
    *,
    knowledge_paths: KnowledgePaths,
    taxonomy: dict[str, object],
    slug: str,
    competition_url: str | None,
    profile: dict[str, object],
) -> None:
    tax = _taxonomy_from_dict(taxonomy)
    tags = tax.normalize(profile.get("tags", []))
    _ensure_db(knowledge_paths)
    now = int(time.time())
    with _connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            """
            INSERT INTO competitions (slug, url, metric, task_type, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                url=excluded.url,
                metric=excluded.metric,
                task_type=excluded.task_type,
                last_seen_at=excluded.last_seen_at
            """,
            (
                slug,
                competition_url,
                profile.get("metric"),
                profile.get("task"),
                now,
                now,
            ),
        )
        for tag in tags:
            conn.execute("INSERT OR IGNORE INTO tags (tag) VALUES (?)", (tag,))
            conn.execute(
                "INSERT OR IGNORE INTO competition_tags (slug, tag) VALUES (?, ?)",
                (slug, tag),
            )


def record_run(
    *,
    knowledge_paths: KnowledgePaths,
    run_id: str,
    slug: str,
    compute: str,
    goal_metric: str,
    goal_score: float,
    direction: str,
) -> None:
    _ensure_db(knowledge_paths)
    now = int(time.time())
    with _connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, slug, started_at, compute, goal_metric, goal_score, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, slug, now, compute, goal_metric, goal_score, direction),
        )


def record_iteration(
    *,
    knowledge_paths: KnowledgePaths,
    run_id: str,
    iteration: int,
    score_source: str,
    offline_value: float,
    offline_std: float | None,
    top1_public_score: float | None,
    met_target: bool,
    git_commit: str | None,
) -> None:
    _ensure_db(knowledge_paths)
    now = int(time.time())
    with _connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            """
            INSERT INTO iterations (
                run_id, iter, score_source, offline_value, offline_std,
                top1_public_score, met_target, git_commit, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                iteration,
                score_source,
                offline_value,
                offline_std,
                top1_public_score,
                int(met_target),
                git_commit,
                now,
            ),
        )


def record_improvement(
    *,
    knowledge_paths: KnowledgePaths,
    run_id: str,
    iteration: int,
    summary: str,
    delta_offline: float | None,
) -> None:
    _ensure_db(knowledge_paths)
    now = int(time.time())
    with _connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            """
            INSERT INTO improvements (run_id, iter, summary, delta_offline, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, iteration, summary, delta_offline, now),
        )


def knowledge_show(knowledge_paths: KnowledgePaths, slug: str) -> dict[str, object]:
    _ensure_db(knowledge_paths)
    with _connect(knowledge_paths.kb_path) as conn:
        comp = conn.execute("SELECT * FROM competitions WHERE slug = ?", (slug,)).fetchone()
        if comp is None:
            return {"slug": slug, "found": False}
        tags = conn.execute(
            "SELECT tag FROM competition_tags WHERE slug = ? ORDER BY tag",
            (slug,),
        ).fetchall()
        runs = conn.execute(
            "SELECT run_id, started_at, compute, goal_metric, goal_score, direction FROM runs WHERE slug = ?",
            (slug,),
        ).fetchall()
        return {
            "slug": slug,
            "found": True,
            "competition": dict(comp),
            "tags": [row["tag"] for row in tags],
            "runs": [dict(row) for row in runs],
        }


def knowledge_search(
    knowledge_paths: KnowledgePaths,
    tags: Iterable[str],
    limit: int,
) -> list[dict[str, object]]:
    _ensure_db(knowledge_paths)
    tags_list = list(tags)
    if not tags_list:
        return []
    with _connect(knowledge_paths.kb_path) as conn:
        placeholders = ",".join("?" for _ in tags_list)
        rows = conn.execute(
            f"""
            SELECT slug, COUNT(*) as overlap
            FROM competition_tags
            WHERE tag IN ({placeholders})
            GROUP BY slug
            ORDER BY overlap DESC
            LIMIT ?
            """,
            [*tags_list, limit],
        ).fetchall()
        return [dict(row) for row in rows]


def _ensure_db(paths: KnowledgePaths) -> None:
    paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
    with _connect(paths.kb_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS competitions (
                slug TEXT PRIMARY KEY,
                url TEXT,
                metric TEXT,
                task_type TEXT,
                created_at INTEGER,
                last_seen_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS tags (
                tag TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS competition_tags (
                slug TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (slug, tag)
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL,
                started_at INTEGER,
                compute TEXT,
                goal_metric TEXT,
                goal_score REAL,
                direction TEXT
            );
            CREATE TABLE IF NOT EXISTS iterations (
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                score_source TEXT,
                offline_value REAL,
                offline_std REAL,
                top1_public_score REAL,
                met_target INTEGER,
                git_commit TEXT,
                created_at INTEGER,
                PRIMARY KEY (run_id, iter)
            );
            CREATE TABLE IF NOT EXISTS improvements (
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                summary TEXT,
                delta_offline REAL,
                created_at INTEGER,
                PRIMARY KEY (run_id, iter)
            );
            """
        )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
