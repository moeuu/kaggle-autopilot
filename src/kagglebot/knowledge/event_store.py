from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from kagglebot.paths import KnowledgePaths


def ensure_event_store(paths: KnowledgePaths) -> None:
    paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
    with _connect(paths.kb_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                slug TEXT,
                run_id TEXT,
                iteration INTEGER,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL,
                run_id TEXT NOT NULL,
                lesson_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_events_lookup
                ON agent_events(event_type, slug, run_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_run_lessons_lookup
                ON run_lessons(slug, run_id, lesson_type, created_at DESC);
            """
        )
        _ensure_fts(conn)


def record_agent_event(
    *,
    knowledge_paths: KnowledgePaths,
    event_type: str,
    title: str,
    body: str,
    slug: str | None = None,
    run_id: str | None = None,
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    ensure_event_store(knowledge_paths)
    now = int(time.time())
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    with _connect(knowledge_paths.kb_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO agent_events (
                event_type, slug, run_id, iteration, title, body, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _clean(event_type, default="event"),
                _clean(slug),
                _clean(run_id),
                iteration,
                _clean(title, default="Untitled event"),
                body.strip(),
                metadata_json,
                now,
            ),
        )
        event_id = int(cursor.lastrowid)
        _insert_event_fts(conn, event_id=event_id, title=title, body=body)
        return event_id


def record_run_lesson(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
    run_id: str,
    lesson_type: str,
    summary: str,
    evidence: str,
    tags: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    ensure_event_store(knowledge_paths)
    now = int(time.time())
    tags_json = json.dumps([str(tag) for tag in tags or [] if str(tag).strip()], sort_keys=True)
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    with _connect(knowledge_paths.kb_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO run_lessons (
                slug, run_id, lesson_type, summary, evidence, tags_json, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _clean(slug, default="unknown"),
                _clean(run_id, default="unknown"),
                _clean(lesson_type, default="lesson"),
                summary.strip(),
                evidence.strip(),
                tags_json,
                metadata_json,
                now,
            ),
        )
        lesson_id = int(cursor.lastrowid)
    record_agent_event(
        knowledge_paths=knowledge_paths,
        event_type=f"lesson:{_clean(lesson_type, default='lesson')}",
        slug=slug,
        run_id=run_id,
        title=summary.strip() or f"Lesson for {slug}",
        body=evidence.strip(),
        metadata={"lesson_id": lesson_id, "tags": json.loads(tags_json), **(metadata or {})},
    )
    return lesson_id


def search_agent_events(
    *,
    knowledge_paths: KnowledgePaths,
    query: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    ensure_event_store(knowledge_paths)
    clean_query = query.strip()
    if not clean_query:
        return []
    with _connect(knowledge_paths.kb_path) as conn:
        try:
            rows = conn.execute(
                """
                SELECT
                    e.id, e.event_type, e.slug, e.run_id, e.iteration, e.title, e.body,
                    e.metadata_json, e.created_at, bm25(agent_events_fts) AS rank
                FROM agent_events_fts
                JOIN agent_events e ON e.id = agent_events_fts.rowid
                WHERE agent_events_fts MATCH ?
                ORDER BY rank, e.created_at DESC
                LIMIT ?
                """,
                (clean_query, max(1, limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{clean_query}%"
            rows = conn.execute(
                """
                SELECT
                    id, event_type, slug, run_id, iteration, title, body,
                    metadata_json, created_at, 0.0 AS rank
                FROM agent_events
                WHERE title LIKE ? OR body LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (like, like, max(1, limit)),
            ).fetchall()
    return [_event_row_to_dict(row) for row in rows]


def _ensure_fts(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS agent_events_fts
            USING fts5(title, body)
            """
        )
    except sqlite3.OperationalError:
        return


def _insert_event_fts(conn: sqlite3.Connection, *, event_id: int, title: str, body: str) -> None:
    try:
        conn.execute(
            "INSERT INTO agent_events_fts(rowid, title, body) VALUES (?, ?, ?)",
            (event_id, title, body),
        )
    except sqlite3.OperationalError:
        return


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _event_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    metadata = {}
    try:
        loaded = json.loads(row["metadata_json"] or "{}")
        if isinstance(loaded, dict):
            metadata = loaded
    except json.JSONDecodeError:
        metadata = {}
    return {
        "id": row["id"],
        "event_type": row["event_type"],
        "slug": row["slug"],
        "run_id": row["run_id"],
        "iteration": row["iteration"],
        "title": row["title"],
        "body": row["body"],
        "metadata": metadata,
        "created_at": row["created_at"],
        "rank": row["rank"],
    }


def _clean(value: object, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default
