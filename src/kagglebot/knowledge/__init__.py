from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kagglebot.knowledge.repositories import InsightRepository, TaxonomyRepository
from kagglebot.paths import KnowledgePaths
from kagglebot.solver.io import find_competition_files, infer_prediction_kind, infer_submission_layout, infer_task


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
    return TaxonomyRepository(paths).ensure()


def load_taxonomy(path: Path) -> dict[str, object]:
    return TaxonomyRepository.load(path)


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


def derive_problem_types(profile: dict[str, object]) -> list[str]:
    modality = str(profile.get("modality") or "").strip().lower()
    task = str(profile.get("task") or "").strip().lower()
    raw_tags = profile.get("tags", [])
    tags = [str(tag).strip().lower() for tag in raw_tags if isinstance(tag, str) and str(tag).strip()]

    problem_types: list[str] = []
    if modality and task:
        problem_types.append(f"{modality}:{task}")
    if modality:
        problem_types.append(modality)
    if task:
        problem_types.append(task)

    allowed_tags = {
        "tabular",
        "text",
        "image",
        "timeseries",
        "regression",
        "binary",
        "multiclass",
        "multitask",
        "missingness_high",
        "high_cardinality_cats",
    }
    for tag in tags:
        if tag in allowed_tags:
            problem_types.append(tag)

    if not problem_types:
        return ["unknown"]

    seen: set[str] = set()
    ordered: list[str] = []
    for item in problem_types:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def resolve_problem_type_insights(
    knowledge_paths: KnowledgePaths,
    problem_types: Iterable[str],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    _ensure_db(knowledge_paths)
    normalized = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not normalized:
        return []
    with _connect(knowledge_paths.kb_path) as conn:
        placeholders = ",".join("?" for _ in normalized)
        rows = conn.execute(
            f"""
            SELECT
                slug,
                run_id,
                iter,
                problem_type,
                cause_category,
                fix_category,
                why_poor,
                how_improved,
                delta_offline,
                outcome_bucket,
                submission_score,
                created_at
            FROM problem_type_insights
            WHERE problem_type IN ({placeholders})
            ORDER BY
                CASE outcome_bucket WHEN 'good' THEN 0 WHEN 'low' THEN 1 ELSE 2 END,
                created_at DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()
    return [dict(row) for row in rows]


def format_problem_type_insights(insights: list[dict[str, object]], *, limit: int = 5) -> str:
    if not insights:
        return "No prior problem-type insights available."
    lines = ["Problem-type knowledge (past failures and successful fixes):", ""]
    for item in insights[:limit]:
        problem_type = item.get("problem_type", "unknown")
        cause = item.get("cause_category", "unknown")
        fix = item.get("fix_category", "unknown")
        delta = item.get("delta_offline")
        outcome_bucket = str(item.get("outcome_bucket") or "unknown")
        submission_score = item.get("submission_score")
        slug = item.get("slug", "unknown")
        why = _shorten_text(str(item.get("why_poor") or ""), 220)
        how = _shorten_text(str(item.get("how_improved") or ""), 220)
        delta_text = "n/a" if delta is None else f"{float(delta):+.6f}"
        score_text = "n/a" if submission_score is None else f"{float(submission_score):.6f}"
        lines.append(
            f"- [{problem_type}] outcome={outcome_bucket} cause={cause} -> fix={fix} "
            f"(online={score_text}, delta={delta_text}, slug={slug})"
        )
        if why:
            lines.append(f"  why: {why}")
        if how:
            lines.append(f"  fix: {how}")
    return "\n".join(lines)


def record_problem_type_insight(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
    run_id: str,
    iteration: int,
    problem_types: Iterable[str],
    why_poor: str,
    how_improved: str,
    delta_offline: float | None,
    outcome_bucket: str | None = None,
    submission_score: float | None = None,
) -> None:
    _ensure_db(knowledge_paths)
    types = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not types:
        types = ["unknown"]

    cause_category = _classify_cause_category(why_poor)
    fix_category = _classify_fix_category(how_improved)
    if cause_category == "unknown":
        cause_category = _classify_cause_category(f"{why_poor}\n{how_improved}")
    if fix_category == "unknown":
        fix_category = _classify_fix_category(f"{how_improved}\n{why_poor}")
    normalized_outcome = _normalize_outcome_bucket(outcome_bucket)

    now = int(time.time())
    why_text = _shorten_text(why_poor, 2000)
    fix_text = _shorten_text(how_improved, 2000)
    with _connect(knowledge_paths.kb_path) as conn:
        for problem_type in types:
            conn.execute(
                """
                INSERT INTO problem_type_insights (
                    slug,
                    run_id,
                    iter,
                    problem_type,
                    cause_category,
                    fix_category,
                    why_poor,
                    how_improved,
                    delta_offline,
                    outcome_bucket,
                    submission_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, iter, problem_type) DO UPDATE SET
                    slug=excluded.slug,
                    cause_category=excluded.cause_category,
                    fix_category=excluded.fix_category,
                    why_poor=excluded.why_poor,
                    how_improved=excluded.how_improved,
                    delta_offline=excluded.delta_offline,
                    outcome_bucket=excluded.outcome_bucket,
                    submission_score=excluded.submission_score,
                    created_at=excluded.created_at
                """,
                (
                    slug,
                    run_id,
                    iteration,
                    problem_type,
                    cause_category,
                    fix_category,
                    why_text,
                    fix_text,
                    delta_offline,
                    normalized_outcome,
                    submission_score,
                    now,
                ),
            )


def resolve_error_fix_insights(
    knowledge_paths: KnowledgePaths,
    problem_types: Iterable[str],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    _ensure_db(knowledge_paths)
    normalized = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not normalized:
        return []
    with _connect(knowledge_paths.kb_path) as conn:
        placeholders = ",".join("?" for _ in normalized)
        rows = conn.execute(
            f"""
            SELECT
                slug,
                run_id,
                iter,
                problem_type,
                error_category,
                error_message,
                fix_summary,
                resolved,
                outcome_bucket,
                submission_score,
                created_at
            FROM error_fix_insights
            WHERE problem_type IN ({placeholders})
            ORDER BY
                CASE outcome_bucket WHEN 'good' THEN 0 WHEN 'low' THEN 1 ELSE 2 END,
                CASE resolved WHEN 1 THEN 0 ELSE 1 END,
                created_at DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()
    return [dict(row) for row in rows]


def format_error_fix_insights(insights: list[dict[str, object]], *, limit: int = 5) -> str:
    if not insights:
        return "No prior error-fix insights available."
    lines = ["Error-fix knowledge (errors and how they were fixed):", ""]
    for item in insights[:limit]:
        problem_type = item.get("problem_type", "unknown")
        category = item.get("error_category", "unknown")
        resolved = bool(item.get("resolved"))
        outcome_bucket = str(item.get("outcome_bucket") or "unknown")
        submission_score = item.get("submission_score")
        error_text = _shorten_text(str(item.get("error_message") or ""), 180)
        fix_text = _shorten_text(str(item.get("fix_summary") or ""), 220)
        score_text = "n/a" if submission_score is None else f"{float(submission_score):.6f}"
        lines.append(
            f"- [{problem_type}] outcome={outcome_bucket} resolved={resolved} error={category} (online={score_text})"
        )
        if error_text:
            lines.append(f"  issue: {error_text}")
        if fix_text:
            lines.append(f"  fix: {fix_text}")
    return "\n".join(lines)


def record_error_fix_insight(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
    run_id: str,
    iteration: int,
    problem_types: Iterable[str],
    error_message: str,
    fix_summary: str,
    resolved: bool,
    outcome_bucket: str | None = None,
    submission_score: float | None = None,
) -> None:
    _ensure_db(knowledge_paths)
    types = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not types:
        types = ["unknown"]

    normalized_outcome = _normalize_outcome_bucket(outcome_bucket)
    category = _classify_error_category(error_message)
    error_text = _shorten_text(error_message, 2000)
    fix_text = _shorten_text(fix_summary, 2000)
    fingerprint = hashlib.sha256(" ".join(error_text.split()).encode("utf-8")).hexdigest()[:16]
    now = int(time.time())
    with _connect(knowledge_paths.kb_path) as conn:
        for problem_type in types:
            conn.execute(
                """
                INSERT INTO error_fix_insights (
                    slug,
                    run_id,
                    iter,
                    problem_type,
                    error_fingerprint,
                    error_category,
                    error_message,
                    fix_summary,
                    resolved,
                    outcome_bucket,
                    submission_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, iter, problem_type, error_fingerprint) DO UPDATE SET
                    slug=excluded.slug,
                    error_category=excluded.error_category,
                    error_message=excluded.error_message,
                    fix_summary=excluded.fix_summary,
                    resolved=excluded.resolved,
                    outcome_bucket=excluded.outcome_bucket,
                    submission_score=excluded.submission_score,
                    created_at=excluded.created_at
                """,
                (
                    slug,
                    run_id,
                    iteration,
                    problem_type,
                    fingerprint,
                    category,
                    error_text,
                    fix_text,
                    int(resolved),
                    normalized_outcome,
                    submission_score,
                    now,
                ),
            )


def _shorten_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _classify_cause_category(text: str) -> str:
    normalized = text.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("data_leakage", ("leak", "target leak", "leakage")),
        ("overfitting", ("overfit", "train/val gap", "generalization gap")),
        ("underfitting", ("underfit", "model too simple", "high bias")),
        ("feature_engineering", ("feature", "encoding", "missing value", "imputation")),
        ("hyperparameter", ("hyperparameter", "learning rate", "max_depth", "n_estimators", "regularization")),
        ("validation_strategy", ("cross-validation", "cv", "fold", "holdout", "split strategy")),
        ("class_imbalance", ("imbalance", "minority class", "class weight", "threshold")),
        ("resource_constraints", ("gpu utilization", "resource", "timeout", "batch size")),
    )
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "unknown"


def _classify_fix_category(text: str) -> str:
    normalized = text.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("stronger_model", ("catboost", "xgboost", "lightgbm", "transformer", "neural network", "model upgrade")),
        ("feature_engineering", ("feature", "encoding", "imputation", "interaction", "transformation")),
        ("hyperparameter_tuning", ("hyperparameter", "learning rate", "max_depth", "n_estimators", "tuning")),
        ("regularization", ("regularization", "dropout", "early stopping", "l1", "l2")),
        ("validation_strategy", ("cross-validation", "cv", "fold", "holdout", "split")),
        ("ensembling", ("ensemble", "stacking", "blending", "average")),
        ("data_cleaning", ("outlier", "duplicate", "cleaning", "leakage fix")),
        ("training_budget", ("epoch", "iterations", "batch size", "training budget")),
    )
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "unknown"


def _classify_error_category(text: str) -> str:
    normalized = text.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("dependency_missing", ("modulenotfounderror", "no module named", "importerror")),
        ("schema_mismatch", ("missing columns", "column", "schema", "keyerror")),
        ("device_mismatch", ("same device", "cuda", "cpu", "device")),
        ("oom", ("out of memory", "cuda out of memory", "oom")),
        ("network", ("connectionerror", "dns", "network", "name resolution")),
        ("kaggle_cli", ("kaggle cli", "competitions submit", "kernels push")),
        ("timeout", ("timeout", "timed out", "deadline exceeded")),
        ("validation", ("row count mismatch", "submission", "validation error")),
    )
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "unknown"


def _normalize_outcome_bucket(value: str | None) -> str:
    if not value:
        return "unknown"
    lowered = str(value).strip().lower()
    if lowered in {"good", "low", "unknown"}:
        return lowered
    return "unknown"


def build_dataset_profile(data_dir: Path) -> dict[str, object]:
    profile: dict[str, object] = {"data_dir": str(data_dir)}
    file_count, extension_counts, file_samples = _summarize_files(data_dir)
    profile["file_count"] = file_count
    profile["file_extension_counts"] = dict(sorted(extension_counts.items()))
    profile["file_samples"] = file_samples

    tabular_files = _find_tabular_files(data_dir)
    if not tabular_files:
        profile["status"] = "non_tabular_data"
        profile["tags"] = []
        return profile

    try:
        train_path, test_path, sample_path = find_competition_files(data_dir)
    except FileNotFoundError as exc:
        profile["status"] = "missing_required_files"
        profile["error"] = str(exc)
        profile["tags"] = []
        return profile

    try:
        train = _read_table(train_path)
        test = _read_table(test_path)
        sample = _read_table(sample_path)
    except Exception as exc:  # noqa: BLE001
        profile["status"] = "unreadable_tabular"
        profile["error"] = str(exc)
        profile["tags"] = []
        return profile

    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
    if not target_cols:
        profile["status"] = "missing_target_columns"
        profile["tags"] = []
        return profile
    target_col = target_cols[0]
    task_by_target = {col: infer_task(train[col]) for col in target_cols}
    unique_tasks = sorted(set(task_by_target.values()))
    task = unique_tasks[0] if len(unique_tasks) == 1 else "mixed"
    prediction_kind_by_target = {
        col: infer_prediction_kind(sample[col]) if col in sample.columns else "continuous" for col in target_cols
    }
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

    if task == "regression":
        metric = "rmse"
    elif prediction_kind_by_target[target_col] == "probability":
        metric = "logloss"
    else:
        metric = "accuracy"
    target_stats = _target_stats(train[target_col], task)

    train_only = [c for c in train.columns if c not in test.columns and c not in target_cols]
    test_only = [c for c in test.columns if c not in train.columns]

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
            "target_columns": target_cols,
            "task": task,
            "task_by_target": task_by_target,
            "prediction_kind_by_target": prediction_kind_by_target,
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
            "train_only_columns": train_only,
            "test_only_columns": test_only,
        }
    )
    return profile


def _find_tabular_files(root: Path) -> list[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _summarize_files(
    data_dir: Path,
    *,
    max_files: int = 20000,
    max_samples: int = 30,
) -> tuple[int, dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    samples: list[str] = []
    total = 0
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        total += 1
        ext = path.suffix.lower() or "<none>"
        counts[ext] = counts.get(ext, 0) + 1
        if len(samples) < max_samples:
            try:
                samples.append(str(path.relative_to(data_dir)))
            except ValueError:
                samples.append(path.name)
        if total >= max_files:
            break
    return total, counts, samples


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
    if task == "mixed":
        return "multitask"
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


def build_plan_and_initial_prompt(
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
        "# Kagglebot Codex: Plan + Implement (Iteration 1)",
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
        f"- artifacts/{slug}/context/overview.md (if present)",
        f"- artifacts/{slug}/context/data.md (if present)",
        f"- artifacts/{slug}/context/submission_format.md (if present)",
        f"- artifacts/{slug}/kernel/kernel.py (authoritative for all compute modes)",
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
        '  "max_iterations": 3,',
        '  "submit_policy": "always"',
        "}",
        "```",
        "",
        "Guidance:",
        "- Derive target_metric and direction from rules.md/rules.html and sample_submission.csv.",
        "- Read overview.md/data.md for problem framing and data caveats.",
        "- If sample_submission is ambiguous, use submission_format.md for the required columns.",
        "- Use overview.md/data.md plus dataset_profile.json (file_extension_counts/file_samples)",
        "  to identify data format.",
        "- Use web search to choose the strongest initial approach; prefer official docs and competition discussions.",
        "- Use top1_public.json to set a realistic target_score; avoid generic metric heuristics.",
        "- Prefer CV for small datasets or high variance; otherwise holdout.",
        "- Do NOT change submit_policy; autopilot controls submission behavior.",
        "",
        "### 2) Implement the strongest initial model",
        "",
        "Implement the best initial approach based on overview/data/rules + web research:",
        "- Loads train/test and sample_submission correctly.",
        "- Handles missing values and encodes categoricals.",
        "- Uses a GPU-optimized supervised model if available.",
        "- Avoid simplistic starters; aim for competition-appropriate strength from the start.",
        "- Do NOT leave the default starter in place; replace it with a competition-specific model.",
        "- Evaluates with the score_source from plan.json.",
        "- Writes submission.csv matching sample_submission.csv exactly.",
        "- Cite the key sources you used (short notes).",
        "",
        "Implementation notes:",
        "- For local_gpu/kaggle_gpu/kaggle_tpu: use artifacts/{slug}/kernel/kernel.py",
        "  as the only training implementation.",
        "- local_gpu and kaggle_gpu must keep the same algorithm/pipeline; only execution location may differ.",
        "- Create artifacts/{slug}/kernel/kernel.py from scratch if missing.",
        "- If data is non-tabular or requires custom parsing, implement custom_main() in kernel.py.",
        "- If pretrained checkpoints are likely beneficial, implement download + cache logic in kernel.py",
        "  and provide a fallback path when internet is unavailable.",
        "- If GPU runs finish in under ~1 minute, increase model iterations/trees or use CV to better utilize GPU.",
        "- Do NOT implement competition-specific logic in src local trainers.",
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
        "The autopilot will iterate up to max_iterations (default 3),",
        "submit according to submission gate policy, and use readiness score as the primary loop decision signal.",
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
**Goal**: Improve loop-decision score (readiness primary; submission/rank as guardrails) toward top1-tier
or best possible within the max_iterations budget (default 3)
**Compute**: {compute} ({accelerator})
**Top1 gap**: {top1_gap}
**Delta vs previous best**: {delta_offline}
**Improvement mode**: {improvement_mode}
**Next iteration**: {next_iteration}

## Current Score Context

- **Metric**: {metric} ({direction})
- **Current score**: {current_score} (source: {current_score_source})
- **Target score**: {target_score}
- **Top1 public**: {top1_score} (source: {top1_source})

## Input Files

- **Plan**: `{plan_path}` - Target metric/score/direction, evaluation strategy
- **Run Config**: `{run_path}` - Current run settings and history
- **Current Metrics**: `{metrics_path}` - Latest loop-decision + offline-by-source results
- **Diagnostics**: `{diagnostics_path}` - Agent-readable analysis of current performance
- **Logs**: `{logs_dir}` - Training logs and error messages (if any)
- **Knowledge Hints**: `{knowledge_hints}`
- **Rules URL**: `{rules_url}`
- **Rules Markdown**: `{rules_md}` (preferred)
- **Rules HTML**: `{rules_html}` (fallback)
- **Overview Markdown**: `{overview_md}` (read this)
- **Data Markdown**: `{data_md}` (read this)
- **Submission Format**: `{submission_format}` (read this if present)
- **Dataset Profile**: `{dataset_profile}`
- **Sample Submission**: `{sample_submission}`
- **Kernel Main**: `{kernel_main}` (edit this for all compute modes)

## Your Task

### Step 1: Analyze Current Performance

Read the diagnostics and metrics files to understand:

1. **What is the current score?**
   - Check `metrics.json` for `loop_decision.source` and `loop_decision.value`
   - Also check offline-by-source metrics for model-quality signal
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

Make **targeted, incremental changes** to improve the current loop-decision score
(submission preferred; offline fallback).

Before changing the model, read overview.md/data.md and respect any constraints or data caveats.

**Where to implement**:
- Always edit `{kernel_main}` (create if missing).
- `local_gpu` and `kaggle_gpu` must use the same algorithm/pipeline.
- For non-tabular inputs, implement `custom_main()` in `{kernel_main}`.

**Modeling policy**:
- Use rules.md + dataset_profile.json + overview/data to decide the best supervised approach.
- Use dataset_profile.json (file_extension_counts/file_samples) to confirm the data format.
- Avoid weak starter implementations.
- Avoid generic metric heuristics; derive metric choices from the rules.
- Use web search each iteration to validate model/feature choices; prefer official docs or top Kaggle discussions.

**Mode guidance**:
- `major_overhaul`: switch model family or core feature strategy; remove dead code paths.
- `moderate_update`: add meaningful features + tune key hyperparameters.
- `minor_tuning`: small hyperparameter/feature tweaks, calibration, or ensembling.

**Pretrained model guidance**:
- If transfer learning can improve score, add checkpoint download + cache logic in `kernel.py`.
- Respect rules and internet settings, and include an internet-off fallback model path.

If the score did not improve (delta <= 0), treat it as `major_overhaul` even if the default mode is weaker.

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

1. **Run offline evaluation (supporting signal)**:
   - Run `kernel.py` via autopilot/train and use kernel metrics/logs for evaluation.
   - Keep one implementation path in `kernel.py` across local and kaggle execution.

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
- [ ] Loop-decision score improves, or offline diagnostics provide actionable learning
- [ ] submission.csv format still matches sample_submission.csv
- [ ] Tests pass: `uv run pytest -q`
- [ ] No secrets leaked into code/logs

---

## Tips for Effective Improvements

1. **Make one change at a time**: Easier to debug and understand what worked
2. **Optimize loop decision**: prioritize submission score when available; use offline as fallback
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
    InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).record_run(
        run_id=run_id,
        slug=slug,
        compute=compute,
        goal_metric=goal_metric,
        goal_score=goal_score,
        direction=direction,
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
    InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).record_iteration(
        run_id=run_id,
        iteration=iteration,
        score_source=score_source,
        offline_value=offline_value,
        offline_std=offline_std,
        top1_public_score=top1_public_score,
        met_target=met_target,
        git_commit=git_commit,
    )


def record_improvement(
    *,
    knowledge_paths: KnowledgePaths,
    run_id: str,
    iteration: int,
    summary: str,
    delta_offline: float | None,
) -> None:
    InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).record_improvement(
        run_id=run_id,
        iteration=iteration,
        summary=summary,
        delta_offline=delta_offline,
    )


def knowledge_show(knowledge_paths: KnowledgePaths, slug: str) -> dict[str, object]:
    return InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).show(slug)


def knowledge_search(
    knowledge_paths: KnowledgePaths,
    tags: Iterable[str],
    limit: int,
) -> list[dict[str, object]]:
    return InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).search(tags, limit)


def record_research_artifacts(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
    problem_types: Iterable[str],
    research_sources_jsonl: str,
    research_summary_md: str,
) -> dict[str, str]:
    _ensure_db(knowledge_paths)
    normalized_types = _normalize_problem_types(problem_types)
    primary_type = _primary_problem_type(normalized_types)
    safe_slug = _safe_label(slug)
    safe_primary = _safe_label(primary_type)
    base_dir = knowledge_paths.knowledge_dir / "research" / safe_primary / safe_slug
    base_dir.mkdir(parents=True, exist_ok=True)

    sources_path = base_dir / "research_sources.jsonl"
    summary_path = base_dir / "research_summary.md"
    sources_path.write_text(research_sources_jsonl.strip() + "\n", encoding="utf-8")
    summary_path.write_text(research_summary_md.strip() + "\n", encoding="utf-8")

    now = int(time.time())
    sources_rel = str(sources_path.relative_to(knowledge_paths.knowledge_dir))
    summary_rel = str(summary_path.relative_to(knowledge_paths.knowledge_dir))
    problem_types_json = json.dumps(normalized_types, ensure_ascii=True)
    with _connect(knowledge_paths.kb_path) as conn:
        existing = conn.execute(
            "SELECT slug FROM competition_research WHERE slug = ?",
            (slug,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO competition_research (
                    slug, primary_problem_type, problem_types_json,
                    sources_path, summary_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slug,
                    primary_type,
                    problem_types_json,
                    sources_rel,
                    summary_rel,
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE competition_research
                SET primary_problem_type = ?,
                    problem_types_json = ?,
                    sources_path = ?,
                    summary_path = ?,
                    updated_at = ?
                WHERE slug = ?
                """,
                (
                    primary_type,
                    problem_types_json,
                    sources_rel,
                    summary_rel,
                    now,
                    slug,
                ),
            )
        conn.execute("DELETE FROM research_problem_types WHERE slug = ?", (slug,))
        for problem_type in normalized_types:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_problem_types (slug, problem_type)
                VALUES (?, ?)
                """,
                (slug, problem_type),
            )
    return {
        "sources_path": str(sources_path),
        "summary_path": str(summary_path),
        "primary_problem_type": primary_type,
    }


def resolve_research_artifacts(
    *,
    knowledge_paths: KnowledgePaths,
    problem_types: Iterable[str],
    limit: int = 5,
) -> list[dict[str, object]]:
    _ensure_db(knowledge_paths)
    normalized = _normalize_problem_types(problem_types)
    if not normalized:
        return []
    placeholders = ",".join("?" for _ in normalized)
    with _connect(knowledge_paths.kb_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                cr.slug,
                cr.primary_problem_type,
                cr.problem_types_json,
                cr.sources_path,
                cr.summary_path,
                cr.updated_at,
                COUNT(*) AS overlap
            FROM competition_research AS cr
            JOIN research_problem_types AS rpt ON rpt.slug = cr.slug
            WHERE rpt.problem_type IN ({placeholders})
            GROUP BY cr.slug
            ORDER BY overlap DESC, cr.updated_at DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        sources_path = knowledge_paths.knowledge_dir / str(record.get("sources_path") or "")
        summary_path = knowledge_paths.knowledge_dir / str(record.get("summary_path") or "")
        problem_types_json = str(record.get("problem_types_json") or "[]")
        try:
            record["problem_types"] = json.loads(problem_types_json)
        except json.JSONDecodeError:
            record["problem_types"] = []
        record["sources_path"] = str(sources_path)
        record["summary_path"] = str(summary_path)
        results.append(record)
    return results


def resolve_research_paths_for_slug(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
) -> tuple[Path | None, Path | None]:
    _ensure_db(knowledge_paths)
    with _connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            "SELECT sources_path, summary_path FROM competition_research WHERE slug = ?",
            (slug,),
        ).fetchone()
    if row is None:
        return None, None
    sources_rel = str(row["sources_path"])
    summary_rel = str(row["summary_path"])
    sources_path = knowledge_paths.knowledge_dir / sources_rel if sources_rel else None
    summary_path = knowledge_paths.knowledge_dir / summary_rel if summary_rel else None
    return sources_path, summary_path


def format_research_artifacts(research_rows: list[dict[str, object]], *, limit: int = 5) -> str:
    if not research_rows:
        return "No cross-competition research artifacts available."
    lines = ["Cross-competition research artifacts:", ""]
    for row in research_rows[:limit]:
        slug = str(row.get("slug") or "unknown")
        primary = str(row.get("primary_problem_type") or "unknown")
        overlap = row.get("overlap")
        summary_path = str(row.get("summary_path") or "")
        sources_path = str(row.get("sources_path") or "")
        problem_types = row.get("problem_types")
        if not isinstance(problem_types, list):
            problem_types = []
        overlap_text = f", overlap={overlap}" if isinstance(overlap, int) else ""
        lines.append(f"- slug={slug}, class={primary}{overlap_text}, tags={problem_types[:6]}")
        if summary_path:
            lines.append(f"  summary: {summary_path}")
        if sources_path:
            lines.append(f"  sources: {sources_path}")
    return "\n".join(lines)


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
            CREATE TABLE IF NOT EXISTS problem_type_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL,
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                problem_type TEXT NOT NULL,
                cause_category TEXT NOT NULL,
                fix_category TEXT NOT NULL,
                why_poor TEXT,
                how_improved TEXT,
                delta_offline REAL,
                outcome_bucket TEXT,
                submission_score REAL,
                created_at INTEGER,
                UNIQUE (run_id, iter, problem_type)
            );
            CREATE INDEX IF NOT EXISTS idx_problem_type_insights_problem_type
                ON problem_type_insights(problem_type, created_at DESC);
            CREATE TABLE IF NOT EXISTS error_fix_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL,
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                problem_type TEXT NOT NULL,
                error_fingerprint TEXT NOT NULL,
                error_category TEXT NOT NULL,
                error_message TEXT,
                fix_summary TEXT,
                resolved INTEGER NOT NULL DEFAULT 0,
                outcome_bucket TEXT,
                submission_score REAL,
                created_at INTEGER,
                UNIQUE (run_id, iter, problem_type, error_fingerprint)
            );
            CREATE INDEX IF NOT EXISTS idx_error_fix_insights_problem_type
                ON error_fix_insights(problem_type, created_at DESC);
            CREATE TABLE IF NOT EXISTS competition_research (
                slug TEXT PRIMARY KEY,
                primary_problem_type TEXT NOT NULL,
                problem_types_json TEXT NOT NULL,
                sources_path TEXT NOT NULL,
                summary_path TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_problem_types (
                slug TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                PRIMARY KEY (slug, problem_type)
            );
            CREATE INDEX IF NOT EXISTS idx_research_problem_types_problem_type
                ON research_problem_types(problem_type, slug);
            """
        )
        _ensure_table_column(conn, "problem_type_insights", "outcome_bucket", "TEXT")
        _ensure_table_column(conn, "problem_type_insights", "submission_score", "REAL")
        _ensure_table_column(conn, "error_fix_insights", "resolved", "INTEGER NOT NULL DEFAULT 0")
        _ensure_table_column(conn, "error_fix_insights", "outcome_bucket", "TEXT")
        _ensure_table_column(conn, "error_fix_insights", "submission_score", "REAL")


def _normalize_problem_types(problem_types: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for item in problem_types:
        value = str(item).strip().lower()
        if not value:
            continue
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        normalized.append("unknown")
    return normalized


def _primary_problem_type(problem_types: list[str]) -> str:
    for item in problem_types:
        if item != "unknown":
            return item
    return "unknown"


def _safe_label(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip().lower())
    clean = clean.strip("-")
    return clean or "unknown"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column in existing:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def build_kernel_fix_template() -> str:
    """Build the kernel-failure fix prompt template."""
    return """\
# Kagglebot Codex: Kernel Failure Fix

## Context

**Competition**: `{slug}`
**Run ID**: {run_id}
**Iteration**: {iteration}
**Compute**: {compute} ({accelerator})

## Blocked Modules (do NOT import)

{blocked_modules}

## Error Summary

```
{error_message}
```

## Input Files

- **Kernel Main (authoritative)**: `{kernel_main}`
- **Kernel Script (generated copy; read-only)**: `{kernel_script}`
  Do **NOT** edit this file. It is regenerated on each run and any edits will be discarded.
- **Kernel Logs**: `{logs_dir}` (check for runtime traces)
- **Rules URL**: `{rules_url}`
- **Rules Markdown**: `{rules_md}` (preferred)
- **Overview Markdown**: `{overview_md}` (read this)
- **Data Markdown**: `{data_md}` (read this)
- **Submission Format**: `{submission_format}` (read this if present)
- **Dataset Profile**: `{dataset_profile}`
- **Sample Submission**: `{sample_submission}`

## Your Task

1) Identify the root cause of the kernel failure from logs and traceback.
2) Fix the issue with minimal, targeted changes.
   - Edit **only** the authoritative `kernel.py` (`Kernel Main` above) for competition-specific fixes.
   - Do **NOT** edit the generated `Kernel Script` copy (it will be overwritten).
   - If the failure is in the **runner/validators/CLI plumbing** (e.g., kernel packaging,
     path injection, validation, or Kaggle CLI status/output handling), you may edit **src/**
     to fix the root cause.
   - Use `custom_main()` when data is non-CSV or default loader is inadequate.
   - If the error mentions `decoder_input_ids` or `decoder_inputs_embeds`, you likely loaded a T5/ProtT5
     encoder-decoder with `AutoModel`. Use `T5EncoderModel` or `model.get_encoder()` and pass only
     `input_ids`/`attention_mask`.
3) Do NOT introduce secrets or network side-effects.
4) Keep the change small and explain what you changed in your response.

The autopilot will re-push the kernel after your fix.
"""
