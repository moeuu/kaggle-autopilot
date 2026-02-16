"""Tests for knowledge base tagging and search."""

from __future__ import annotations

import sqlite3

import kagglebot.knowledge as knowledge_mod
from kagglebot.knowledge import (
    derive_problem_types,
    ensure_taxonomy,
    format_error_fix_insights,
    knowledge_search,
    load_taxonomy,
    record_competition_profile,
    record_error_fix_insight,
    record_improvement,
    record_iteration,
    record_problem_type_insight,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
)
from kagglebot.knowledge.repositories import InsightRepository
from kagglebot.paths import KnowledgePaths


def test_knowledge_search_orders_by_overlap(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    taxonomy = ensure_taxonomy(knowledge_paths)

    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="comp-a",
        competition_url=None,
        profile={"metric": "accuracy", "task": "classification", "tags": ["tabular", "binary"]},
    )
    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="comp-b",
        competition_url=None,
        profile={"metric": "rmse", "task": "regression", "tags": ["tabular"]},
    )

    results = knowledge_search(knowledge_paths, ["tabular", "binary"], limit=5)
    assert results[0]["slug"] == "comp-a"


def test_load_taxonomy_yaml(tmp_path) -> None:
    content = """
data_modality:
  - tabular
  - text
aliases:
  bin: binary
"""
    path = tmp_path / "taxonomy.yml"
    path.write_text(content, encoding="utf-8")
    data = load_taxonomy(path)
    assert "tabular" in data["tags"]
    assert data["aliases"]["bin"] == "binary"


def test_problem_type_insight_record_and_resolve(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    profile = {"modality": "tabular", "task": "regression", "tags": ["tabular", "regression"]}
    problem_types = derive_problem_types(profile)
    assert "tabular:regression" in problem_types

    record_problem_type_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        run_id="run-1",
        iteration=1,
        problem_types=problem_types,
        why_poor="Model was underfit with weak features and high validation error.",
        how_improved="Added CatBoost and richer feature engineering with longer training.",
        delta_offline=0.12,
        outcome_bucket="good",
        submission_score=0.8123,
    )

    insights = resolve_problem_type_insights(knowledge_paths, ["tabular:regression"], limit=5)
    assert insights
    first = insights[0]
    assert first["problem_type"] == "tabular:regression"
    assert first["cause_category"] != ""
    assert first["fix_category"] != ""
    assert first["outcome_bucket"] == "good"
    assert first["submission_score"] == 0.8123


def test_error_fix_insight_record_and_resolve(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    problem_types = ["tabular:binary", "tabular"]

    record_error_fix_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-b",
        run_id="run-2",
        iteration=2,
        problem_types=problem_types,
        error_message="ModuleNotFoundError: No module named 'featurewiz'",
        fix_summary="Removed featurewiz import and added sklearn fallback.",
        resolved=True,
        outcome_bucket="low",
        submission_score=0.731,
    )

    insights = resolve_error_fix_insights(knowledge_paths, ["tabular:binary"], limit=5)
    assert insights
    first = insights[0]
    assert first["error_category"] == "dependency_missing"
    assert bool(first["resolved"]) is True
    assert first["outcome_bucket"] == "low"
    assert first["submission_score"] == 0.731

    rendered = format_error_fix_insights(insights, limit=5)
    assert "dependency_missing" in rendered
    assert "featurewiz" in rendered


def test_record_iteration_upserts_on_duplicate_run_iteration(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_iteration(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    record_iteration(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_iteration_falls_back_to_update_on_unique_violation(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    state = {"insert_calls": 0, "raised": False}

    class _FlakyConnection:
        def __init__(self, path, state_map):
            self._conn = sqlite3.connect(path)
            self._state = state_map

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def execute(self, sql, params=()):
            if "INSERT INTO iterations" in sql:
                self._state["insert_calls"] += 1
                if self._state["insert_calls"] == 2 and not self._state["raised"]:
                    self._state["raised"] = True
                    raise sqlite3.IntegrityError("UNIQUE constraint failed: iterations.run_id, iterations.iter")
            return self._conn.execute(sql, params)

    repo = InsightRepository(
        knowledge_paths,
        ensure_db=knowledge_mod._ensure_db,
        connect=lambda path: _FlakyConnection(path, state),
    )

    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert state["raised"] is True
    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_iteration_falls_back_to_update_on_unique_violation_code_name(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    state = {"insert_calls": 0, "raised": False}

    class _FlakyConnection:
        def __init__(self, path, state_map):
            self._conn = sqlite3.connect(path)
            self._state = state_map

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def execute(self, sql, params=()):
            if "INSERT INTO iterations" in sql:
                self._state["insert_calls"] += 1
                if self._state["insert_calls"] == 2 and not self._state["raised"]:
                    self._state["raised"] = True
                    exc = sqlite3.IntegrityError("constraint failed")
                    exc.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_UNIQUE
                    exc.sqlite_errorname = "SQLITE_CONSTRAINT_UNIQUE"
                    raise exc
            return self._conn.execute(sql, params)

    repo = InsightRepository(
        knowledge_paths,
        ensure_db=knowledge_mod._ensure_db,
        connect=lambda path: _FlakyConnection(path, state),
    )

    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert state["raised"] is True
    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_improvement_upserts_on_duplicate_run_iteration(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=2,
        summary="first",
        delta_offline=0.01,
    )
    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=2,
        summary="second",
        delta_offline=0.08,
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT summary, delta_offline
            FROM improvements
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 2),
        ).fetchone()

    assert row == ("second", 0.08)
