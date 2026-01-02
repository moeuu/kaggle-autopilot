"""Tests for knowledge base tagging and search."""

from __future__ import annotations

from kagglebot.knowledge import ensure_taxonomy, knowledge_search, load_taxonomy, record_competition_profile
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
