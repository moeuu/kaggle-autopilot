from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from kagglebot.paths import KnowledgePaths


def ensure_skill_registry(paths: KnowledgePaths) -> None:
    paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
    with _connect(paths.kb_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                tags_json TEXT NOT NULL DEFAULT '[]',
                problem_types_json TEXT NOT NULL DEFAULT '[]',
                source TEXT,
                fitness_score REAL NOT NULL DEFAULT 0.0,
                usage_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS skill_versions (
                skill_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                body TEXT NOT NULL,
                source_event_id INTEGER,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (skill_id, version)
            );
            CREATE TABLE IF NOT EXISTS skill_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                slug TEXT,
                run_id TEXT,
                outcome TEXT NOT NULL,
                offline_delta REAL,
                top1_gap_delta REAL,
                submit_recovered INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skills_status_fitness
                ON skills(status, fitness_score DESC, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_skill_evaluations_skill
                ON skill_evaluations(skill_id, created_at DESC);
            """
        )


def upsert_skill(
    *,
    knowledge_paths: KnowledgePaths,
    skill_id: str,
    title: str,
    summary: str,
    body: str,
    tags: list[str] | tuple[str, ...] | None = None,
    problem_types: list[str] | tuple[str, ...] | None = None,
    status: str = "candidate",
    source: str | None = None,
    source_event_id: int | None = None,
) -> dict[str, object]:
    ensure_skill_registry(knowledge_paths)
    normalized_id = normalize_skill_id(skill_id or title)
    now = int(time.time())
    tags_json = json.dumps(_clean_list(tags), sort_keys=True)
    problem_types_json = json.dumps(_clean_list(problem_types), sort_keys=True)
    clean_status = _normalize_status(status)
    with _connect(knowledge_paths.kb_path) as conn:
        existing = conn.execute(
            "SELECT skill_id, created_at FROM skills WHERE skill_id = ?",
            (normalized_id,),
        ).fetchone()
        created_at = int(existing["created_at"]) if existing is not None else now
        conn.execute(
            """
            INSERT INTO skills (
                skill_id, title, summary, status, tags_json, problem_types_json, source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                title=excluded.title,
                summary=excluded.summary,
                status=excluded.status,
                tags_json=excluded.tags_json,
                problem_types_json=excluded.problem_types_json,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (
                normalized_id,
                title.strip() or normalized_id.replace("_", " ").title(),
                summary.strip(),
                clean_status,
                tags_json,
                problem_types_json,
                source,
                created_at,
                now,
            ),
        )
        version = _next_version(conn, normalized_id)
        conn.execute(
            """
            INSERT INTO skill_versions (skill_id, version, body, source_event_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (normalized_id, version, body.strip(), source_event_id, now),
        )
    path = write_skill_markdown(
        knowledge_paths=knowledge_paths,
        skill_id=normalized_id,
        title=title,
        summary=summary,
        body=body,
        tags=_clean_list(tags),
        problem_types=_clean_list(problem_types),
        status=clean_status,
        version=version,
    )
    return {"skill_id": normalized_id, "version": version, "path": str(path), "status": clean_status}


def record_skill_evaluation(
    *,
    knowledge_paths: KnowledgePaths,
    skill_id: str,
    outcome: str,
    slug: str | None = None,
    run_id: str | None = None,
    offline_delta: float | None = None,
    top1_gap_delta: float | None = None,
    submit_recovered: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    ensure_skill_registry(knowledge_paths)
    normalized_id = normalize_skill_id(skill_id)
    now = int(time.time())
    success = _outcome_is_success(outcome)
    with _connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            """
            INSERT INTO skill_evaluations (
                skill_id, slug, run_id, outcome, offline_delta, top1_gap_delta,
                submit_recovered, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_id,
                _clean(slug),
                _clean(run_id),
                _clean(outcome, default="unknown"),
                offline_delta,
                top1_gap_delta,
                None if submit_recovered is None else int(submit_recovered),
                json.dumps(metadata or {}, sort_keys=True),
                now,
            ),
        )
        conn.execute(
            """
            UPDATE skills
            SET
                usage_count = usage_count + 1,
                success_count = success_count + ?,
                fitness_score = CASE
                    WHEN usage_count + 1 <= 0 THEN fitness_score
                    ELSE CAST(success_count + ? AS REAL) / CAST(usage_count + 1 AS REAL)
                END,
                updated_at = ?
            WHERE skill_id = ?
            """,
            (int(success), int(success), now, normalized_id),
        )


def search_skills(
    *,
    knowledge_paths: KnowledgePaths,
    problem_types: list[str] | tuple[str, ...] | None = None,
    query: str = "",
    limit: int = 5,
) -> list[dict[str, object]]:
    ensure_skill_registry(knowledge_paths)
    wanted_types = set(_clean_list(problem_types))
    query_terms = set(_tokenize(query))
    with _connect(knowledge_paths.kb_path) as conn:
        rows = conn.execute(
            """
            SELECT
                s.skill_id, s.title, s.summary, s.status, s.tags_json, s.problem_types_json,
                s.source, s.fitness_score, s.usage_count, s.success_count, s.created_at, s.updated_at,
                v.version, v.body
            FROM skills s
            JOIN skill_versions v ON v.skill_id = s.skill_id
            WHERE v.version = (
                SELECT MAX(version) FROM skill_versions WHERE skill_id = s.skill_id
            )
            ORDER BY s.fitness_score DESC, s.updated_at DESC, s.skill_id
            """
        ).fetchall()
    ranked: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        item = _skill_row_to_dict(row)
        skill_types = set(item["problem_types"])
        tags = set(item["tags"])
        body_terms = set(_tokenize(f"{item['title']} {item['summary']} {item['body']}"))
        type_overlap = len(wanted_types & (skill_types | tags))
        query_overlap = len(query_terms & body_terms) if query_terms else 0
        status_bonus = {"active": 2.0, "candidate": 1.0, "deprecated": -2.0}.get(str(item["status"]), 0.0)
        score = (
            type_overlap * 5.0
            + query_overlap * 2.0
            + float(item["fitness_score"] or 0.0)
            + status_bonus
            + min(int(item["success_count"] or 0), 5) * 0.2
        )
        if wanted_types and type_overlap == 0 and query_overlap == 0:
            continue
        ranked.append((score, item))
    ranked.sort(key=lambda pair: (pair[0], pair[1]["updated_at"]), reverse=True)
    return [item for _, item in ranked[: max(1, limit)]]


def format_skills_for_prompt(skills: list[dict[str, object]], *, limit: int = 5) -> str:
    if not skills:
        return "Reusable Kaggle skills: none found."
    lines = ["Reusable Kaggle skills:"]
    for item in skills[:limit]:
        body = str(item.get("body") or "").strip()
        body = body[:1200] + "..." if len(body) > 1200 else body
        lines.extend(
            [
                f"- {item.get('skill_id')} ({item.get('status')}, fitness={item.get('fitness_score')}): "
                f"{item.get('summary')}",
                f"  Problem types: {', '.join(item.get('problem_types') or []) or 'unknown'}",
                f"  Procedure: {body}",
            ]
        )
    return "\n".join(lines)


def write_skill_markdown(
    *,
    knowledge_paths: KnowledgePaths,
    skill_id: str,
    title: str,
    summary: str,
    body: str,
    tags: list[str],
    problem_types: list[str],
    status: str,
    version: int,
) -> Path:
    skills_dir = knowledge_paths.knowledge_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    path = skills_dir / f"{skill_id}.md"
    path.write_text(
        "\n".join(
            [
                f"# {title.strip() or skill_id}",
                "",
                f"- skill_id: `{skill_id}`",
                f"- status: `{status}`",
                f"- version: `{version}`",
                f"- problem_types: {', '.join(problem_types) if problem_types else 'unknown'}",
                f"- tags: {', '.join(tags) if tags else 'none'}",
                "",
                "## Summary",
                summary.strip(),
                "",
                "## Procedure",
                body.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def normalize_skill_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "kaggle_skill"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _next_version(conn: sqlite3.Connection, skill_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) AS max_version FROM skill_versions WHERE skill_id = ?",
        (skill_id,),
    ).fetchone()
    max_version = row["max_version"] if row is not None else None
    return int(max_version or 0) + 1


def _skill_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "skill_id": row["skill_id"],
        "title": row["title"],
        "summary": row["summary"],
        "status": row["status"],
        "tags": _json_list(row["tags_json"]),
        "problem_types": _json_list(row["problem_types_json"]),
        "source": row["source"],
        "fitness_score": row["fitness_score"],
        "usage_count": row["usage_count"],
        "success_count": row["success_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "version": row["version"],
        "body": row["body"],
    }


def _json_list(value: object) -> list[str]:
    try:
        loaded = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded if str(item).strip()]


def _clean_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    cleaned: list[str] = []
    for value in values or []:
        text = str(value).strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _normalize_status(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"candidate", "active", "deprecated"} else "candidate"


def _outcome_is_success(value: str) -> bool:
    return str(value or "").strip().lower() in {"success", "improved", "recovered", "completed", "high"}


def _tokenize(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9_]+", value.lower()) if len(token) >= 3]


def _clean(value: object, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default
