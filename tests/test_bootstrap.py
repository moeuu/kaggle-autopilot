"""Tests for bootstrap helpers."""

from __future__ import annotations

import json

from kagglebot.bootstrap import _mirror_sample_submission_to_data, bootstrap_competition
from kagglebot.paths import CompetitionPaths, KnowledgePaths


def test_rules_file_written_to_markdown(tmp_path) -> None:
    rules_file = tmp_path / "rules.txt"
    rules_file.write_text("Rules text.\n", encoding="utf-8")
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    bootstrap_competition(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="file",
        rules_file=rules_file,
        download=False,
        dry_run=False,
    )

    assert paths.rules_url_path.exists()
    assert paths.rules_md_path.exists()
    assert "Rules text." in paths.rules_md_path.read_text(encoding="utf-8")


def test_rules_html_converts_to_markdown(tmp_path) -> None:
    rules_file = tmp_path / "rules.html"
    rules_file.write_text("<h1>Title</h1><p>Body text</p>", encoding="utf-8")
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    bootstrap_competition(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="file",
        rules_file=rules_file,
        download=False,
        dry_run=False,
    )
    assert paths.rules_html_path.exists()
    assert paths.rules_md_path.exists()
    text = paths.rules_md_path.read_text(encoding="utf-8")
    assert "Title" in text


def test_rules_overview_data_from_url(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Rules text"},
            {"name": "description", "content": "Overview text"},
            {"name": "evaluation", "content": "Evaluation text"},
            {"name": "data-description", "content": "Data text"},
            {"name": "Frequently Asked Questions", "content": "<h3>FAQ</h3><p>Answer</p>"},
        ]

    def fake_code_context(*, paths, slug: str, timeout: int = 10):  # noqa: ARG001
        return "# code\n\n- code item\n"

    def fake_discussion_context(*, paths, slug: str, timeout: int = 10):  # noqa: ARG001
        return "# discussion\n\n- discussion item\n"

    def fake_tab_snapshot(*, slug: str, tab: str, timeout: int = 10):  # noqa: ARG001
        return f"# {tab}\n\n- {tab} item\n"

    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_pages", fake_pages)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_and_download_competition_code", fake_code_context)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_and_download_competition_discussions", fake_discussion_context)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_tab_snapshot", fake_tab_snapshot)
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    bootstrap_competition(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="url",
        download=False,
        dry_run=False,
    )

    assert "Rules text" in paths.rules_md_path.read_text(encoding="utf-8")
    overview_text = paths.overview_md_path.read_text(encoding="utf-8")
    assert "Overview text" in overview_text
    assert "Evaluation text" in overview_text
    assert "FAQ" in overview_text
    assert "Data text" in paths.data_md_path.read_text(encoding="utf-8")
    assert "code item" in paths.code_md_path.read_text(encoding="utf-8")
    assert "models item" in paths.models_md_path.read_text(encoding="utf-8")
    assert "discussion item" in paths.discussion_md_path.read_text(encoding="utf-8")


def test_bootstrap_downloads_code_notebooks_and_discussions(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Rules text"},
            {"name": "description", "content": "Overview text"},
            {"name": "data-description", "content": "Data text"},
        ]

    def fake_fetch_text(*args, **kwargs):  # noqa: ANN002, ANN003
        url = args[0]
        if url.endswith("/code"):
            return """
            <html><body>
              <a href="/code/alice/strong-baseline">Strong Baseline</a>
              <div>Public Score: 0.95329</div>
            </body></html>
            """
        if url.endswith("/models"):
            return "<html><body><h1>Models</h1><p>model item</p></body></html>"
        if url.endswith("/discussions"):
            return """
            <html><body>
              <a href="/competitions/demo/discussion/101">How to avoid leakage</a>
              <a href="/competitions/demo/discussion/102">Fast CV tips</a>
            </body></html>
            """
        if "/discussion/101" in url:
            return (
                "<html><head><title>How to avoid leakage | Kaggle</title></head>"
                "<body><p>Use GroupKFold.</p></body></html>"
            )
        if "/discussion/102" in url:
            return (
                "<html><head><title>Fast CV tips | Kaggle</title></head>"
                "<body><p>Use repeated CV seeds.</p></body></html>"
            )
        return ""

    def fake_kernels_pull(kernel_id, output_dir, *, slug=None, dry_run=False, metadata=True):  # noqa: ANN001, ARG001
        notebook = output_dir / "strong-baseline.ipynb"
        notebook.write_text(
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "markdown", "source": ["Use CatBoost with robust CV."]},
                        {"cell_type": "code", "source": ["# train with 5 folds and seed ensemble"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return f"pulled {kernel_id}"

    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_pages", fake_pages)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_text_with_retry", fake_fetch_text)
    monkeypatch.setattr("kagglebot.bootstrap.kernels_pull", fake_kernels_pull)
    monkeypatch.setattr("kagglebot.bootstrap._list_competition_code_candidates_from_cli", lambda *, slug: [])
    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_topics_from_api", lambda *, slug, timeout: [])

    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    bootstrap_competition(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="url",
        download=False,
        dry_run=False,
    )

    code_index = json.loads(paths.code_notebooks_index_path.read_text(encoding="utf-8"))
    assert code_index["notebook_count"] == 1
    assert code_index["notebooks"][0]["kernel_id"] == "alice/strong-baseline"
    assert code_index["notebooks"][0]["score"] == 0.95329
    assert paths.code_notebooks_dir.exists()
    assert any(path.suffix == ".ipynb" for path in paths.code_notebooks_dir.rglob("*"))
    assert "notebook_score: 0.953290" in paths.code_md_path.read_text(encoding="utf-8")

    discussion_index = json.loads(paths.discussion_threads_index_path.read_text(encoding="utf-8"))
    assert discussion_index["thread_count"] == 2
    assert any("leakage" in str(item.get("title", "")).lower() for item in discussion_index["threads"])
    assert any(path.suffix == ".md" for path in paths.discussion_threads_dir.glob("*.md"))
    discussion_md = paths.discussion_md_path.read_text(encoding="utf-8")
    assert "How to avoid leakage" in discussion_md
    assert "Fast CV tips" in discussion_md


def test_bootstrap_code_uses_top_score_order_and_caps_at_10(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Rules text"},
            {"name": "description", "content": "Overview text"},
            {"name": "data-description", "content": "Data text"},
        ]

    def fake_fetch_text(*args, **kwargs):  # noqa: ANN002, ANN003
        url = args[0]
        if url.endswith("/code"):
            return "<html><body><h1>Predicting Heart Disease | Kaggle</h1></body></html>"
        if url.endswith("/models"):
            return "<html><body><h1>Models</h1></body></html>"
        if url.endswith("/discussions"):
            return "<html><body></body></html>"
        return ""

    pulled: list[str] = []

    def fake_kernels_pull(kernel_id, output_dir, *, slug=None, dry_run=False, metadata=True):  # noqa: ANN001, ARG001
        pulled.append(str(kernel_id))
        notebook = output_dir / "notebook.ipynb"
        notebook.write_text(json.dumps({"cells": []}), encoding="utf-8")
        return "ok"

    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_pages", fake_pages)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_text_with_retry", fake_fetch_text)
    monkeypatch.setattr("kagglebot.bootstrap.kernels_pull", fake_kernels_pull)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_topics_from_api", lambda *, slug, timeout: [])

    candidates = []
    for idx in range(12):
        nb_slug = f"nb-{idx:02d}"
        candidates.append(
            {
                "kernel_id": f"user/{nb_slug}",
                "title": f"Notebook {idx}",
                "url": f"https://www.kaggle.com/code/user/{nb_slug}",
                "score": 0.900 + (idx * 0.001),
                "source_order": idx,
            }
        )
    monkeypatch.setattr("kagglebot.bootstrap._list_competition_code_candidates_from_cli", lambda *, slug: candidates)

    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    bootstrap_competition(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="url",
        download=False,
        dry_run=False,
    )

    code_index = json.loads(paths.code_notebooks_index_path.read_text(encoding="utf-8"))
    assert code_index["notebook_count"] == 10
    assert code_index["candidate_source"] == "kaggle kernels list --competition --sort-by scoreDescending"
    assert len(pulled) == 10
    assert pulled[0] == "user/nb-11"
    assert pulled[-1] == "user/nb-02"
    assert code_index["notebooks"][0]["kernel_id"] == "user/nb-11"
    assert code_index["notebooks"][0]["score"] == 0.911
    assert code_index["notebooks"][-1]["kernel_id"] == "user/nb-02"


def test_mirror_sample_submission_to_data(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.sample_submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    _mirror_sample_submission_to_data(paths)

    mirrored = paths.data_dir / "sample_submission.csv"
    assert mirrored.exists()
    assert mirrored.read_text(encoding="utf-8") == "id,target\n1,0.1\n"
