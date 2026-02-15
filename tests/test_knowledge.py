"""Tests for knowledge base tagging and search."""

from __future__ import annotations

from kagglebot.knowledge import (
    derive_problem_types,
    ensure_taxonomy,
    format_error_fix_insights,
    knowledge_search,
    load_taxonomy,
    record_competition_profile,
    record_error_fix_insight,
    record_problem_type_insight,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
)
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
