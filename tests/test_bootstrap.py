"""Tests for bootstrap helpers."""

from __future__ import annotations

import urllib.error

from kagglebot.bootstrap import bootstrap_competition
from kagglebot.paths import CompetitionPaths, KnowledgePaths


def test_rules_fetch_failure_writes_error(tmp_path, monkeypatch) -> None:
    def fake_urlopen(*args, **kwargs):  # noqa: ARG001
        raise urllib.error.URLError("blocked")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    bootstrap_competition(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="fetch",
        download=False,
        dry_run=False,
    )

    assert paths.rules_url_path.exists()
    assert (paths.context_dir / "fetch_error.txt").exists()
