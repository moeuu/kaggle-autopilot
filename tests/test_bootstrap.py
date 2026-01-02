"""Tests for bootstrap helpers."""

from __future__ import annotations

from kagglebot.bootstrap import bootstrap_competition
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
    assert paths.kernel_overrides_path.exists()


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
