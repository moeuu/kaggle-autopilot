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
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


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

    profile.update(
        {
            "status": "ok",
            "train_rows": n_rows,
            "train_cols": n_cols,
            "test_rows": len(test),
            "test_cols": len(test.columns),
            "id_column": id_col,
            "target_column": target_col,
            "task": task,
            "metric": metric,
            "missingness": missingness,
            "categorical_columns": cat_cols,
            "numeric_columns": [c for c in feature_cols if c not in cat_cols],
            "high_cardinality_columns": high_cardinality,
            "modality": modality,
            "tags": tags,
        }
    )
    return profile


def _infer_modality(data_dir: Path, train: pd.DataFrame) -> str:
    image_exts = {".jpg", ".jpeg", ".png"}
    if any(path.suffix.lower() in image_exts for path in data_dir.rglob("*")):
        return "image"
    text_cols = [c for c in train.columns if train[c].dtype == "object"]
    if text_cols:
        avg_len = 0.0
        sample = train[text_cols].astype(str).head(200)
        if not sample.empty:
            avg_len = sample.applymap(len).mean().mean()
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
    lines = [
        "# Kagglebot Codex Plan + Baseline Prompt",
        "",
        f"Competition slug: {slug}",
        f"Rules URL: {rules_url}",
        "",
        "Context:",
        f"- Dataset profile: artifacts/{slug}/context/dataset_profile.json",
        f"- Sample submission: artifacts/{slug}/context/sample_submission.csv",
        f"- Leaderboard snapshot: artifacts/{slug}/context/top1_public.json",
        f"- Rules reference: artifacts/{slug}/context/rules_url.txt (or rules.*)",
        f"- Tags: {tags}",
        "",
        "Knowledge base (similar competitions):",
    ]
    if similar_improvements:
        for item in similar_improvements:
            lines.append(f"- {item['slug']} (overlap={item['overlap']}): {item['summary']}")
    else:
        lines.append("- None found.")
    lines += [
        "",
        "Instructions:",
        "1) Read meta.json, dataset_profile.json, and sample_submission.csv.",
        "2) Update artifacts/<slug>/plan.json with target metric/score/direction and evaluation strategy.",
        "   - Fill: target_metric, target_score, target_direction, score_source.",
        "   - Set holdout_frac/cv_folds/seed/time_budget_min/kernel_name/internet when appropriate.",
        "   - Keep submit_policy as on_target_only unless explicitly justified.",
        "3) Implement a robust baseline under kagglebot/solver/.",
        "4) Keep CLI stable and non-interactive.",
        "5) Avoid secrets in code/logs.",
        "6) Ensure tests pass: uv run pytest -q",
    ]
    return "\n".join(lines) + "\n"


def build_improve_template() -> str:
    return (
        "# Kagglebot Codex Improvement Prompt\n\n"
        "Competition: {slug}\n"
        "Iteration: {iteration}\n\n"
        "Inputs:\n"
        "- Plan: {plan_path}\n"
        "- Run config: {run_path}\n"
        "- Metrics: {metrics_path}\n"
        "- Diagnostics: {diagnostics_path}\n"
        "- Logs: {logs_dir}\n\n"
        "Task:\n"
        "1) Identify likely causes of the current score.\n"
        "2) Implement improvements in kagglebot/solver/ to improve offline score (direction-aware).\n"
        "3) Keep CLI stable and non-interactive.\n"
        "4) Update/add tests so `uv run pytest -q` passes.\n"
        "5) Do NOT introduce secrets.\n"
    )


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
