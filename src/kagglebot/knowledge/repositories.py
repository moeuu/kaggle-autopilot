from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.json_utils import write_json_object
from kagglebot.paths import KnowledgePaths

_DEFAULT_TAXONOMY: dict[str, object] = {
    "tags": sorted(
        {
            "tabular",
            "text",
            "image",
            "timeseries",
            "regression",
            "binary",
            "multiclass",
            "multitask",
            "n_rows_small",
            "n_rows_medium",
            "n_rows_large",
            "missingness_high",
            "high_cardinality_cats",
        }
    ),
    "aliases": {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
    },
}


def _is_unique_constraint_violation(exc: sqlite3.IntegrityError) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    error_name = str(getattr(exc, "sqlite_errorname", ""))

    unique_code = getattr(sqlite3, "SQLITE_CONSTRAINT_UNIQUE", None)
    primary_key_code = getattr(sqlite3, "SQLITE_CONSTRAINT_PRIMARYKEY", None)
    if isinstance(error_code, int) and error_code in {unique_code, primary_key_code}:
        return True

    if error_name in {"SQLITE_CONSTRAINT_UNIQUE", "SQLITE_CONSTRAINT_PRIMARYKEY"}:
        return True

    message = str(exc).lower()
    return "unique constraint failed" in message


@dataclass(frozen=True)
class TaxonomyRepository:
    paths: KnowledgePaths

    def ensure(self) -> dict[str, object]:
        self.paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
        if not self.paths.taxonomy_path.exists():
            write_json_object(self.paths.taxonomy_path, _DEFAULT_TAXONOMY)
        return self.load(self.paths.taxonomy_path)

    @staticmethod
    def load(path: Path) -> dict[str, object]:
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return TaxonomyRepository._parse_yaml_taxonomy(text)

    @staticmethod
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


@dataclass(frozen=True)
class InsightRepository:
    paths: KnowledgePaths
    ensure_db: Callable[[KnowledgePaths], None]
    connect: Callable[[Path], sqlite3.Connection]

    def record_run(
        self,
        *,
        run_id: str,
        slug: str,
        compute: str,
        goal_metric: str,
        goal_score: float,
        direction: str,
    ) -> None:
        self.ensure_db(self.paths)
        now = int(time.time())
        with self.connect(self.paths.kb_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO runs (
                    run_id, slug, started_at, compute, goal_metric, goal_score, direction
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, slug, now, compute, goal_metric, goal_score, direction),
            )

    def record_iteration(
        self,
        *,
        run_id: str,
        iteration: int,
        score_source: str,
        offline_value: float,
        offline_std: float | None,
        top1_public_score: float | None,
        met_target: bool,
        git_commit: str | None,
    ) -> None:
        self.ensure_db(self.paths)
        now = int(time.time())
        met_target_int = int(met_target)
        values = (
            run_id,
            iteration,
            score_source,
            offline_value,
            offline_std,
            top1_public_score,
            met_target_int,
            git_commit,
            now,
        )
        with self.connect(self.paths.kb_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO iterations (
                        run_id, iter, score_source, offline_value, offline_std,
                        top1_public_score, met_target, git_commit, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, iter) DO UPDATE SET
                        score_source=excluded.score_source,
                        offline_value=excluded.offline_value,
                        offline_std=excluded.offline_std,
                        top1_public_score=excluded.top1_public_score,
                        met_target=excluded.met_target,
                        git_commit=excluded.git_commit,
                        created_at=excluded.created_at
                    """,
                    values,
                )
            except sqlite3.IntegrityError as exc:
                if not _is_unique_constraint_violation(exc):
                    raise
                # Defensive fallback for environments where retrying an existing
                # iteration record can still surface the uniqueness constraint.
                conn.execute(
                    """
                    UPDATE iterations
                    SET
                        score_source = ?,
                        offline_value = ?,
                        offline_std = ?,
                        top1_public_score = ?,
                        met_target = ?,
                        git_commit = ?,
                        created_at = ?
                    WHERE run_id = ? AND iter = ?
                    """,
                    (
                        score_source,
                        offline_value,
                        offline_std,
                        top1_public_score,
                        met_target_int,
                        git_commit,
                        now,
                        run_id,
                        iteration,
                    ),
                )

    def record_improvement(
        self,
        *,
        run_id: str,
        iteration: int,
        summary: str,
        delta_offline: float | None,
    ) -> None:
        self.ensure_db(self.paths)
        now = int(time.time())
        with self.connect(self.paths.kb_path) as conn:
            conn.execute(
                """
                INSERT INTO improvements (run_id, iter, summary, delta_offline, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, iter) DO UPDATE SET
                    summary=excluded.summary,
                    delta_offline=excluded.delta_offline,
                    created_at=excluded.created_at
                """,
                (run_id, iteration, summary, delta_offline, now),
            )

    def show(self, slug: str) -> dict[str, object]:
        self.ensure_db(self.paths)
        with self.connect(self.paths.kb_path) as conn:
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

    def search(self, tags: Iterable[str], limit: int) -> list[dict[str, object]]:
        self.ensure_db(self.paths)
        tags_list = list(tags)
        if not tags_list:
            return []
        with self.connect(self.paths.kb_path) as conn:
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
