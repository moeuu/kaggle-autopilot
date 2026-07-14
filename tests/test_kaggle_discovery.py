from __future__ import annotations

from pathlib import Path

from kagglebot.exec_utils import CommandResult
from kagglebot.kaggle_discovery import refresh_kaggle_discovery
from kagglebot.paths import CompetitionPaths


def _result(args: list[str], stdout: str, *, returncode: int = 0) -> CommandResult:
    return CommandResult(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr="",
        duration_sec=0.1,
    )


def test_refresh_kaggle_discovery_collects_ranks_and_renders_all_surfaces(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="arc-prize-2026-arc-agi-3", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.dataset_profile_path.write_text(
        '{"modality":"text","task":"agent","tags":["reasoning"]}\n',
        encoding="utf-8",
    )

    def fake_run(args: list[str], **kwargs) -> CommandResult:  # noqa: ANN003
        assert kwargs["timeout"] == 45
        if args[1:3] == ["datasets", "list"]:
            return _result(
                args,
                "ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating\n"
                "owner/unrelated,Weather,10,2020-01-01,10000,500,1\n"
                "owner/arc-reasoning,ARC AGI Reasoning,20,2026-07-01,100,20,1\n",
            )
        if args[1:3] == ["models", "list"]:
            return _result(
                args,
                "id,ref,title,subtitle,author\n1,owner/arc-model,ARC reasoning model,Agent model,Owner\n",
            )
        if args[1:3] == ["kernels", "list"]:
            return _result(
                args,
                "ref,title,author,lastRunTime,totalVotes\nowner/arc-code,ARC AGI agent,Owner,2026-07-01,50\n",
            )
        if "benchmarks" in args:
            return _result(
                args,
                "id,title,authorName,commentCount,votes,postDate\n"
                "700,Agent reasoning benchmark,Owner,3,12,2026-06-01\n",
            )
        return _result(
            args,
            "id,title,authorName,commentCount,votes,postDate\n600,ARC AGI research resources,Owner,4,30,2026-06-01\n",
        )

    def fake_fetch(url: str) -> str:
        if url.endswith("game-arena"):
            return (
                "<html><head><title>Game Arena | Kaggle</title>"
                '<meta name="description" content="Dynamic game evaluation for model agents."></head></html>'
            )
        return (
            "<html><head><title>AI Benchmarks | Kaggle</title>"
            '<meta name="description" content="Evaluate AI models and agents."></head></html>'
        )

    payload = refresh_kaggle_discovery(
        paths=paths,
        force=True,
        run_command_fn=fake_run,
        fetch_url_fn=fake_fetch,
    )

    assert payload["query"] == "arc agi"
    assert payload["errors"] == []
    assert set(payload["surface_counts"]) == {
        "datasets",
        "models",
        "code",
        "discussions",
        "game_arena",
        "benchmarks",
    }
    records = payload["records"]
    assert isinstance(records, list)
    dataset_records = [record for record in records if record["surface"] == "datasets"]
    assert dataset_records[0]["ref"] == "owner/arc-reasoning"
    assert any(record["surface"] == "game_arena" for record in records)
    assert any(record["surface"] == "benchmarks" for record in records)
    assert any(record["surface"] == "benchmarks" and record["source"] == "kaggle_page_metadata" for record in records)
    assert paths.kaggle_discovery_path.exists()
    markdown = paths.kaggle_discovery_md_path.read_text(encoding="utf-8")
    assert "## Datasets" in markdown
    assert "## Game Arena" in markdown
    assert "## Benchmarks" in markdown
    assert "owner/arc-reasoning" in markdown


def test_refresh_kaggle_discovery_reuses_fresh_cache(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.dataset_profile_path.write_text("{}\n", encoding="utf-8")
    calls = {"count": 0}

    def fake_run(args: list[str], **kwargs) -> CommandResult:  # noqa: ANN003
        calls["count"] += 1
        return _result(args, "ref,title\nowner/demo,Demo\n")

    refresh_kaggle_discovery(
        paths=paths,
        force=True,
        run_command_fn=fake_run,
        fetch_url_fn=lambda url: "<title>Surface</title>",
    )
    first_count = calls["count"]
    cached = refresh_kaggle_discovery(
        paths=paths,
        run_command_fn=fake_run,
        fetch_url_fn=lambda url: "<title>Surface</title>",
    )

    assert first_count == 5
    assert calls["count"] == first_count
    assert cached["slug"] == "demo"
