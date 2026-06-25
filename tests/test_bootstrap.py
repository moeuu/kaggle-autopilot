"""Tests for bootstrap helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from kagglebot.bootstrap import (
    _cache_sample_submission,
    _mirror_sample_submission_to_data,
    _parse_last_run_epoch,
    _read_direction_from_json,
    _write_sample_head,
    bootstrap_competition,
)
from kagglebot.bootstrap_reference_inputs import stage_reference_notebook_inputs
from kagglebot.paths import CompetitionPaths, KnowledgePaths

pytestmark = pytest.mark.slow


def test_read_direction_from_json_ignores_missing_invalid_or_non_object_payload(tmp_path) -> None:
    assert _read_direction_from_json(tmp_path / "missing.json", ("direction",)) is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert _read_direction_from_json(invalid, ("direction",)) is None

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert _read_direction_from_json(array_payload, ("direction",)) is None


def test_parse_last_run_epoch_normalizes_iso_timestamps_to_utc() -> None:
    expected = datetime(2026, 2, 24, 1, tzinfo=UTC).timestamp()

    assert _parse_last_run_epoch("2026-02-24T01:00:00Z") == expected
    assert _parse_last_run_epoch("2026-02-24 10:00:00+09:00") == expected
    assert _parse_last_run_epoch("not a date") == 0.0


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


def test_bootstrap_stages_reference_notebook_inputs(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Rules text"},
            {"name": "description", "content": "Overview text"},
            {"name": "data-description", "content": "Data text"},
        ]

    def fake_fetch_text(*args, **kwargs):  # noqa: ANN002, ANN003
        url = args[0]
        if url.endswith("/code"):
            return "<html><body><a href='/code/alice/ref-kernel'>Ref</a></body></html>"
        if url.endswith("/models"):
            return "<html><body><h1>Models</h1></body></html>"
        if url.endswith("/discussions"):
            return "<html><body></body></html>"
        return ""

    dataset_calls: list[str] = []
    kernel_calls: list[str] = []
    competition_calls: list[tuple[str, str]] = []

    def fake_download_competition(slug, dest_dir, *, force, quiet, dry_run=False, progress_callback=None):  # noqa: ANN001, ARG001
        competition_calls.append((str(slug), str(dest_dir)))
        dest_dir.mkdir(parents=True, exist_ok=True)
        if slug == "demo":
            (dest_dir / "train.csv").write_text("id,feature,target\n1,10,0\n2,20,1\n", encoding="utf-8")
            (dest_dir / "test.csv").write_text("id,feature\n3,30\n", encoding="utf-8")
            (dest_dir / "sample_submission.csv").write_text("id,target\n3,0\n", encoding="utf-8")
        else:
            (dest_dir / "external.txt").write_text("external competition data\n", encoding="utf-8")
        return "ok"

    def fake_download_dataset(dataset_ref, dest_dir, *, slug=None, dry_run=False, force=True, quiet=True):  # noqa: ANN001, ARG001
        dataset_calls.append(str(dataset_ref))
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "dataset.csv").write_text("id,value\n1,0.5\n", encoding="utf-8")
        return "ok"

    def fake_kernels_pull(kernel_id, output_dir, *, slug=None, dry_run=False, metadata=True):  # noqa: ANN001, ARG001
        kernel_calls.append(str(kernel_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ref-kernel.ipynb").write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "source": [
                                "import kagglehub\n",
                                "kagglehub.dataset_download('bob/notebook-only-backup')\n",
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "dataset_sources": ["alice/original-churn"],
                    "competition_sources": ["external-comp"],
                    "kernel_sources": ["carol/shared-model"],
                }
            ),
            encoding="utf-8",
        )
        return f"pulled {kernel_id}"

    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_pages", fake_pages)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_text_with_retry", fake_fetch_text)
    monkeypatch.setattr("kagglebot.bootstrap.kernels_pull", fake_kernels_pull)
    monkeypatch.setattr("kagglebot.bootstrap.download_competition", fake_download_competition)
    monkeypatch.setattr("kagglebot.bootstrap.download_dataset", fake_download_dataset)
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
        download=True,
        dry_run=False,
    )

    manifest = json.loads(paths.reference_inputs_manifest_path.read_text(encoding="utf-8"))
    assert manifest["required_reference_kernel_id"] == "alice/ref-kernel"
    assert manifest["reference_notebooks"]
    entry = manifest["reference_notebooks"][0]
    refs = {(item["kind"], item["ref"]) for item in entry["input_sources"]}
    assert ("dataset", "alice/original-churn") in refs
    assert ("competition", "external-comp") in refs
    assert ("kernel", "carol/shared-model") in refs
    assert ("dataset", "bob/notebook-only-backup") in refs
    staged = {(item["kind"], item["ref"], item["status"]) for item in entry["staged_sources"]}
    assert ("dataset", "alice/original-churn", "staged_dataset") in staged
    assert ("competition", "external-comp", "staged_competition") in staged
    assert ("kernel", "carol/shared-model", "staged_kernel") in staged
    assert "alice/original-churn" in dataset_calls
    assert "bob/notebook-only-backup" in dataset_calls
    assert "carol/shared-model" in kernel_calls
    assert any(slug == "external-comp" for slug, _dest in competition_calls)


def test_bootstrap_code_uses_top_score_order_and_caps_at_50(tmp_path, monkeypatch) -> None:
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
    for idx in range(80):
        nb_slug = f"nb-{idx:02d}"
        candidates.append(
            {
                "kernel_id": f"user/{nb_slug}",
                "title": f"Notebook {idx}",
                "url": f"https://www.kaggle.com/code/user/{nb_slug}",
                "score": 0.900 + (idx * 0.001),
                "total_votes": idx,
                "last_run_time": f"2026-02-24 10:{idx:02d}:00",
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
    assert code_index["notebook_count"] == 50
    assert code_index["score_direction"] == "maximize"
    assert code_index["top_kernel_id"] == "user/nb-79"
    assert code_index["required_reference_kernel_id"] == "user/nb-79"
    assert code_index["candidate_source"] == "kaggle kernels list --competition --sort-by scoreDescending"
    assert len(pulled) == 50
    assert pulled[0] == "user/nb-79"
    assert pulled[-1] == "user/nb-30"
    assert code_index["notebooks"][0]["kernel_id"] == "user/nb-79"
    assert round(float(code_index["notebooks"][0]["score"]), 6) == 0.979
    assert code_index["notebooks"][-1]["kernel_id"] == "user/nb-30"
    code_text = paths.code_md_path.read_text(encoding="utf-8")
    assert "Top-ranked Notebook (Raw ranking)" in code_text
    assert "Required Reference Notebook (Execution baseline)" in code_text
    assert "instruction: Treat this notebook as a mandatory baseline reference" in code_text


def test_bootstrap_code_prefers_votes_when_scores_missing(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Rules text"},
            {"name": "description", "content": "Overview text"},
            {"name": "data-description", "content": "Data text"},
        ]

    def fake_fetch_text(*args, **kwargs):  # noqa: ANN002, ANN003
        url = args[0]
        if url.endswith("/code"):
            return "<html><body><h1>Code</h1></body></html>"
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

    candidates = [
        {
            "kernel_id": "user/score-missing-a",
            "title": "A",
            "url": "https://www.kaggle.com/code/user/score-missing-a",
            "score": None,
            "total_votes": 12,
            "last_run_time": "2026-02-24 10:00:00",
            "source_order": 0,
        },
        {
            "kernel_id": "user/score-missing-b",
            "title": "B",
            "url": "https://www.kaggle.com/code/user/score-missing-b",
            "score": None,
            "total_votes": 44,
            "last_run_time": "2026-02-24 09:00:00",
            "source_order": 1,
        },
        {
            "kernel_id": "user/score-missing-c",
            "title": "C",
            "url": "https://www.kaggle.com/code/user/score-missing-c",
            "score": None,
            "total_votes": 7,
            "last_run_time": "2026-02-24 11:00:00",
            "source_order": 2,
        },
    ]
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

    assert pulled[:3] == [
        "user/score-missing-b",
        "user/score-missing-a",
        "user/score-missing-c",
    ]


def test_bootstrap_code_respects_minimize_score_direction(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Metric: Brier score (lower is better)."},
            {"name": "description", "content": "Overview text"},
            {"name": "data-description", "content": "Data text"},
        ]

    def fake_fetch_text(*args, **kwargs):  # noqa: ANN002, ANN003
        url = args[0]
        if url.endswith("/code"):
            return "<html><body><h1>Code</h1></body></html>"
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

    candidates = [
        {
            "kernel_id": "user/high-score",
            "title": "High score",
            "url": "https://www.kaggle.com/code/user/high-score",
            "score": 0.2000,
            "total_votes": 10,
            "last_run_time": "2026-02-24 10:00:00",
            "source_order": 0,
        },
        {
            "kernel_id": "user/best-score",
            "title": "Best score",
            "url": "https://www.kaggle.com/code/user/best-score",
            "score": 0.0400,
            "total_votes": 8,
            "last_run_time": "2026-02-24 10:00:00",
            "source_order": 1,
        },
        {
            "kernel_id": "user/mid-score",
            "title": "Mid score",
            "url": "https://www.kaggle.com/code/user/mid-score",
            "score": 0.1000,
            "total_votes": 9,
            "last_run_time": "2026-02-24 10:00:00",
            "source_order": 2,
        },
    ]
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

    assert pulled[:3] == [
        "user/best-score",
        "user/mid-score",
        "user/high-score",
    ]


def test_bootstrap_code_uses_non_leak_required_reference_when_top_ranked_looks_leaky(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Metric: Brier score (lower is better)."},
            {"name": "description", "content": "Overview text"},
            {"name": "data-description", "content": "Data text"},
        ]

    def fake_fetch_text(*args, **kwargs):  # noqa: ANN002, ANN003
        url = args[0]
        if url.endswith("/code"):
            return "<html><body><h1>Code</h1></body></html>"
        if url.endswith("/models"):
            return "<html><body><h1>Models</h1></body></html>"
        if url.endswith("/discussions"):
            return "<html><body></body></html>"
        return ""

    def fake_kernels_pull(kernel_id, output_dir, *, slug=None, dry_run=False, metadata=True):  # noqa: ANN001, ARG001
        notebook = output_dir / "notebook.ipynb"
        notebook.write_text(json.dumps({"cells": []}), encoding="utf-8")
        return f"ok:{kernel_id}"

    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_pages", fake_pages)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_text_with_retry", fake_fetch_text)
    monkeypatch.setattr("kagglebot.bootstrap.kernels_pull", fake_kernels_pull)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_topics_from_api", lambda *, slug, timeout: [])

    candidates = [
        {
            "kernel_id": "user/lb-0-0-leak-submit",
            "title": "[LB 0.0] Leak Submit",
            "url": "https://www.kaggle.com/code/user/lb-0-0-leak-submit",
            "score": 0.0000,
            "total_votes": 40,
            "last_run_time": "2026-02-24 10:00:00",
            "source_order": 0,
        },
        {
            "kernel_id": "user/histgb-xgb-catboost",
            "title": "HistGB + XGB + CatBoost",
            "url": "https://www.kaggle.com/code/user/histgb-xgb-catboost",
            "score": 0.0734,
            "total_votes": 20,
            "last_run_time": "2026-02-24 11:00:00",
            "source_order": 1,
        },
    ]
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
    assert code_index["top_kernel_id"] == "user/lb-0-0-leak-submit"
    assert code_index["required_reference_kernel_id"] == "user/histgb-xgb-catboost"

    code_text = paths.code_md_path.read_text(encoding="utf-8")
    assert "selection_reason: Top-ranked notebook appears leak-like/placeholder" in code_text


def test_bootstrap_code_policy_selects_required_and_ensemble_references(tmp_path, monkeypatch) -> None:
    def fake_pages(*, slug: str, rules_url: str, timeout: int = 10):  # noqa: ARG001
        return [
            {"name": "rules", "content": "Rules text"},
            {"name": "description", "content": "Overview text"},
            {"name": "data-description", "content": "Data text"},
        ]

    def fake_fetch_text(*args, **kwargs):  # noqa: ANN002, ANN003
        url = args[0]
        if url.endswith("/code"):
            return "<html><body><h1>Code</h1></body></html>"
        if url.endswith("/models"):
            return "<html><body><h1>Models</h1></body></html>"
        if url.endswith("/discussions"):
            return "<html><body></body></html>"
        return ""

    def fake_kernels_pull(kernel_id, output_dir, *, slug=None, dry_run=False, metadata=True):  # noqa: ANN001, ARG001
        notebook = output_dir / "notebook.ipynb"
        notebook.write_text(json.dumps({"cells": []}), encoding="utf-8")
        return f"ok:{kernel_id}"

    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_pages", fake_pages)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_text_with_retry", fake_fetch_text)
    monkeypatch.setattr("kagglebot.bootstrap.kernels_pull", fake_kernels_pull)
    monkeypatch.setattr("kagglebot.bootstrap._fetch_competition_topics_from_api", lambda *, slug, timeout: [])

    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.competition_policy_path.write_text(
        json.dumps(
            {
                "required_capabilities": ["recoverable_original_dataset", "requires_oof_blend"],
                "notebook_selection": {
                    "keyword_boosts": {"original dataset": 0.05, "blend": 0.02},
                    "required_reference_keywords": ["original dataset"],
                    "ensemble_reference_keywords": ["blend", "oof"],
                },
                "prompt": {"prefer_ensemble_reference": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    candidates = [
        {
            "kernel_id": "user/highest-raw",
            "title": "Strong baseline",
            "url": "https://www.kaggle.com/code/user/highest-raw",
            "score": 0.9500,
            "total_votes": 100,
            "last_run_time": "2026-02-24 10:00:00",
            "source_order": 0,
        },
        {
            "kernel_id": "user/original-dataset-baseline",
            "title": "Original Dataset baseline",
            "url": "https://www.kaggle.com/code/user/original-dataset-baseline",
            "score": 0.9300,
            "total_votes": 20,
            "last_run_time": "2026-02-24 11:00:00",
            "source_order": 1,
        },
        {
            "kernel_id": "user/oof-blend-stack",
            "title": "OOF Blend Stack",
            "url": "https://www.kaggle.com/code/user/oof-blend-stack",
            "score": 0.9200,
            "total_votes": 30,
            "last_run_time": "2026-02-24 12:00:00",
            "source_order": 2,
        },
    ]
    monkeypatch.setattr("kagglebot.bootstrap._list_competition_code_candidates_from_cli", lambda *, slug: candidates)

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
    assert code_index["required_reference_kernel_id"] == "user/original-dataset-baseline"
    assert code_index["ensemble_reference_kernel_id"] == "user/oof-blend-stack"
    assert code_index["required_capabilities"] == ["recoverable_original_dataset", "requires_oof_blend"]
    code_text = paths.code_md_path.read_text(encoding="utf-8")
    assert "Ensemble Reference Notebook (Blend blueprint)" in code_text


def test_stage_reference_inputs_policy_records_required_datasets_and_ensemble_reference(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.code_notebooks_dir.mkdir(parents=True, exist_ok=True)
    first_dir = paths.code_notebooks_dir / "required"
    second_dir = paths.code_notebooks_dir / "ensemble"
    first_dir.mkdir(parents=True, exist_ok=True)
    second_dir.mkdir(parents=True, exist_ok=True)
    (first_dir / "kernel-metadata.json").write_text(
        json.dumps({"dataset_sources": ["alice/original-data"]}),
        encoding="utf-8",
    )
    (first_dir / "notebook.ipynb").write_text(json.dumps({"cells": []}), encoding="utf-8")
    (second_dir / "notebook.ipynb").write_text(json.dumps({"cells": []}), encoding="utf-8")
    paths.competition_policy_path.write_text(
        json.dumps(
            {
                "archetype_tags": ["recoverable_original_dataset"],
                "required_capabilities": ["recoverable_original_dataset"],
                "reference_inputs": {
                    "proactive": True,
                    "required_datasets": ["alice/original-data", "bob/missing-data"],
                    "extra_kernel_refs": ["carol/shared-model"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "ensemble_reference_kernel_id": "alice/ensemble-kernel",
                "notebooks": [
                    {
                        "kernel_id": "alice/ref-kernel",
                        "title": "Required",
                        "local_dir": str(first_dir),
                        "source_file": str(first_dir / "notebook.ipynb"),
                        "summary": "",
                    },
                    {
                        "kernel_id": "alice/ensemble-kernel",
                        "title": "Ensemble",
                        "local_dir": str(second_dir),
                        "source_file": str(second_dir / "notebook.ipynb"),
                        "summary": "",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stage_reference_notebook_inputs(
        paths=paths,
        slug="demo",
        download=False,
        quiet=True,
        dry_run=False,
    )

    manifest = json.loads(paths.reference_inputs_manifest_path.read_text(encoding="utf-8"))
    assert manifest["ensemble_reference_kernel_id"] == "alice/ensemble-kernel"
    assert manifest["required_datasets"] == ["alice/original-data", "bob/missing-data"]
    assert manifest["required_capabilities"] == ["recoverable_original_dataset"]
    assert manifest["missing_required_sources"] == ["bob/missing-data"]
    assert manifest["policy_tags"] == ["recoverable_original_dataset"]


def test_stage_reference_inputs_writes_default_manifest_for_invalid_index(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.code_notebooks_index_path.write_text("{", encoding="utf-8")

    stage_reference_notebook_inputs(
        paths=paths,
        slug="demo",
        download=False,
        quiet=True,
        dry_run=False,
    )

    manifest = json.loads(paths.reference_inputs_manifest_path.read_text(encoding="utf-8"))
    assert manifest["reference_notebooks"] == []
    assert manifest["missing_required_sources"] == []


def test_stage_reference_inputs_policy_proactively_downloads_required_datasets(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.code_notebooks_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir = paths.code_notebooks_dir / "required"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    (notebook_dir / "kernel-metadata.json").write_text(
        json.dumps({"dataset_sources": ["alice/original-data"]}),
        encoding="utf-8",
    )
    (notebook_dir / "notebook.ipynb").write_text(json.dumps({"cells": []}), encoding="utf-8")
    paths.competition_policy_path.write_text(
        json.dumps(
            {
                "reference_inputs": {
                    "proactive": True,
                    "required_datasets": ["alice/original-data"],
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "notebooks": [
                    {
                        "kernel_id": "alice/ref-kernel",
                        "title": "Required",
                        "local_dir": str(notebook_dir),
                        "source_file": str(notebook_dir / "notebook.ipynb"),
                        "summary": "",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset_calls: list[str] = []

    def fake_download_dataset(ref, output_dir, *, slug, dry_run, force, quiet):  # noqa: ANN001, ARG001
        dataset_calls.append(str(ref))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text("{}", encoding="utf-8")

    stage_reference_notebook_inputs(
        paths=paths,
        slug="demo",
        download=False,
        quiet=True,
        dry_run=False,
        download_dataset_fn=fake_download_dataset,
    )

    manifest = json.loads(paths.reference_inputs_manifest_path.read_text(encoding="utf-8"))
    staged = manifest["reference_notebooks"][0]["staged_sources"]
    assert dataset_calls == ["alice/original-data"]
    assert staged[0]["status"] == "staged_dataset"


def test_mirror_sample_submission_to_data(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.sample_submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    _mirror_sample_submission_to_data(paths)

    mirrored = paths.data_dir / "sample_submission.csv"
    assert mirrored.exists()
    assert mirrored.read_text(encoding="utf-8") == "id,target\n1,0.1\n"


def test_write_sample_head_reads_limited_rows(tmp_path, monkeypatch) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text("id,target\n1,0.1\n2,0.2\n3,0.3\n", encoding="utf-8")
    head_path = tmp_path / "sample_submission_head.csv"

    import pandas as pd

    called: dict[str, int | None] = {"nrows": None}
    real_read_csv = pd.read_csv

    def _spy_read_csv(*args, **kwargs):  # noqa: ANN002, ANN003
        called["nrows"] = kwargs.get("nrows")
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _spy_read_csv)

    _write_sample_head(sample_path, head_path, rows=2)

    assert called["nrows"] == 2
    assert head_path.read_text(encoding="utf-8").strip().splitlines() == [
        "id,target",
        "1,0.1",
        "2,0.2",
    ]


def test_cache_sample_submission_prefers_stage2_for_multistage_competition(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")

    stage1 = paths.data_dir / "SampleSubmissionStage1.csv"
    stage1.write_text("ID,Pred\n2022_1_2,0.5\n2022_1_3,0.5\n", encoding="utf-8")
    stage2 = paths.data_dir / "SampleSubmissionStage2.csv"
    stage2.write_text("ID,Pred\n2026_1_2,0.5\n2026_1_3,0.5\n2026_2_3,0.5\n", encoding="utf-8")

    _cache_sample_submission(paths)

    assert paths.sample_submission_path.read_text(encoding="utf-8") == stage2.read_text(encoding="utf-8")
    assert "2026_1_2" in paths.sample_submission_head_path.read_text(encoding="utf-8")


def test_mirror_sample_submission_to_data_overwrites_stale_destination(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.sample_submission_path.write_text("ID,Pred\n2026_1_2,0.5\n2026_1_3,0.5\n", encoding="utf-8")
    data_sample = paths.data_dir / "sample_submission.csv"
    data_sample.write_text("ID,Pred\n2022_1_2,0.5\n", encoding="utf-8")

    _mirror_sample_submission_to_data(paths)

    assert data_sample.read_text(encoding="utf-8") == paths.sample_submission_path.read_text(encoding="utf-8")
