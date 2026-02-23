"""Tests for kaggle_api helpers."""

from __future__ import annotations

from kagglebot import kaggle_api
from kagglebot.exceptions import KaggleCliError, RulesNotAcceptedError


def test_check_rules_accepted(monkeypatch) -> None:
    def fake_competitions_files(slug: str, *, dry_run: bool = False) -> str:  # noqa: ARG001
        if slug in {"foo", "no-entry"}:
            return "name,sizeBytes\nsample_submission.csv,123\n"
        if slug == "bar":
            raise RulesNotAcceptedError("Competition rules not accepted.")
        raise KaggleCliError(
            "Kaggle CLI failed with exit code 1.",
            ["kaggle", "competitions", "files", "-c", slug],
            exit_code=1,
            output="404 - Not Found",
        )

    def fake_list_competition_submissions(slug: str, *, dry_run: bool = False) -> list[dict[str, str]]:  # noqa: ARG001
        if slug == "foo":
            return []
        if slug == "no-entry":
            raise KaggleCliError(
                "Kaggle CLI failed with exit code 1.",
                ["kaggle", "competitions", "submissions", "-c", slug, "--csv"],
                exit_code=1,
                output="403 Forbidden: you do not have permission for this competition.",
            )
        raise AssertionError(f"unexpected slug in submissions probe: {slug}")

    monkeypatch.setattr(kaggle_api, "competitions_files", fake_competitions_files)
    monkeypatch.setattr(kaggle_api, "list_competition_submissions", fake_list_competition_submissions)
    assert kaggle_api.check_rules_accepted("foo") is True
    assert kaggle_api.check_rules_accepted("no-entry") is False
    assert kaggle_api.check_rules_accepted("bar") is False
    assert kaggle_api.check_rules_accepted("missing") is False


def test_leaderboard_top1_download_and_parse(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        captured["args"] = args
        csv_path = tmp_path / "leaderboard" / "leaderboard.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("Rank,Score\n1,0.123\n", encoding="utf-8")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_top1("demo", tmp_path)
    assert "-c" in captured["args"]
    assert "-p" in captured["args"]
    assert result["score"] == 0.123


def test_leaderboard_top1_prefers_score_columns(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        csv_path = tmp_path / "leaderboard" / "leaderboard.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("TeamId,PublicScore\n1301,0.81234\n", encoding="utf-8")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_top1("demo", tmp_path)
    assert result["score"] == 0.81234


def test_leaderboard_top1_extracts_zip(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        zip_path = tmp_path / "leaderboard" / "demo.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("demo-leaderboard.csv", "Rank,Score\n1,0.987\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_top1("demo", tmp_path)
    assert result["score"] == 0.987


def test_leaderboard_rank_for_score_maximize(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        csv_path = tmp_path / "leaderboard" / "leaderboard.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("Rank,Score\n1,0.99\n2,0.97\n3,0.95\n4,0.94\n", encoding="utf-8")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_rank_for_score("demo", tmp_path, score=0.95, direction="maximize")
    assert result["rank"] == 3
    assert result["total_teams"] == 4
    assert result["rank_percentile"] == 0.75


def test_leaderboard_rank_for_score_minimize(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        csv_path = tmp_path / "leaderboard" / "leaderboard.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("Rank,Score\n1,0.100\n2,0.120\n3,0.125\n4,0.140\n", encoding="utf-8")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_rank_for_score("demo", tmp_path, score=0.125, direction="minimize")
    assert result["rank"] == 3
    assert result["total_teams"] == 4
    assert result["rank_percentile"] == 0.75


def test_kernel_exists_matches_url_refs(monkeypatch) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        return "ref,title\nhttps://www.kaggle.com/code/MoeUuu/sample-kernel,Sample\n"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    assert kaggle_api.kernel_exists("moeuuu/sample-kernel") is True


def test_kernel_exists_matches_plain_refs(monkeypatch) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        return "ref,title\nmoeuuu/another-kernel,Another\n"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    assert kaggle_api.kernel_exists("moeuuu/another-kernel") is True


def test_parse_competition_files_csv_with_next_page_token() -> None:
    output = "Next Page Token = token-123\nname,size,creationDate\nfoo.csv,123,2026-01-01\nbar.tif,456,2026-01-02\n"
    files, token = kaggle_api._parse_competition_files_csv(output)
    assert token == "token-123"
    assert [(item.name, item.size_bytes) for item in files] == [("foo.csv", 123), ("bar.tif", 456)]


def test_list_competition_files_with_sizes_handles_paging(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        calls.append(args)
        if "--page-token" in args:
            return "name,size,creationDate\nb.csv,20,2026-01-02\n"
        return "Next Page Token = p2\nname,size,creationDate\na.csv,10,2026-01-01\n"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    rows = kaggle_api._list_competition_files_with_sizes("demo", dry_run=False)

    assert [(item.name, item.size_bytes) for item in rows] == [("a.csv", 10), ("b.csv", 20)]
    assert len(calls) == 2
    assert "--page-token" not in calls[0]
    assert calls[1][-2:] == ["--page-token", "p2"]


def test_list_competition_files_with_sizes_retries_retryable_errors(monkeypatch) -> None:
    monkeypatch.setattr(kaggle_api, "_download_attempts", lambda: None)
    monkeypatch.setattr(kaggle_api, "_download_retry_backoff_sec", lambda: 0.0)
    counter = {"n": 0}

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        counter["n"] += 1
        if counter["n"] < 3:
            raise KaggleCliError(
                "transient",
                args,
                exit_code=1,
                output="request timed out",
            )
        return "name,size,creationDate\na.csv,10,2026-01-01\n"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    rows = kaggle_api._list_competition_files_with_sizes("demo", dry_run=False)

    assert [(item.name, item.size_bytes) for item in rows] == [("a.csv", 10)]
    assert counter["n"] == 3


def test_download_competition_uses_single_shot_for_small_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [kaggle_api._CompetitionFile(name="small.csv", size_bytes=128)],  # noqa: ARG005
    )
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 1024)

    captured: list[list[str]] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        captured.append(args)
        return "done"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert output == "done"
    assert len(captured) == 1
    assert "-f" not in captured[0]
    assert captured[0][:3] == ["kaggle", "competitions", "download"]


def test_download_competition_splits_and_retries_large_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [  # noqa: ARG005
            kaggle_api._CompetitionFile(name="b.csv", size_bytes=70),
            kaggle_api._CompetitionFile(name="a.csv", size_bytes=70),
        ],
    )
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 100)
    monkeypatch.setattr(kaggle_api, "_download_attempts", lambda: 2)
    monkeypatch.setattr(kaggle_api, "_download_retry_backoff_sec", lambda: 0.0)

    seen: list[str] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        file_name = args[args.index("-f") + 1]
        seen.append(file_name)
        if file_name == "a.csv" and seen.count("a.csv") == 1:
            raise KaggleCliError(
                "transient",
                args,
                exit_code=1,
                output="request timed out",
            )
        return f"downloaded {file_name}"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert "downloaded a.csv" in output
    assert "downloaded b.csv" in output
    assert seen == ["a.csv", "a.csv", "b.csv"]


def test_download_competition_split_reports_progress(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [  # noqa: ARG005
            kaggle_api._CompetitionFile(name="b.csv", size_bytes=70),
            kaggle_api._CompetitionFile(name="a.csv", size_bytes=70),
        ],
    )
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 100)

    progress: list[tuple[int, int, str | None]] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        return f"downloaded {args[args.index('-f') + 1]}"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api.download_competition(
        "demo",
        tmp_path,
        force=True,
        quiet=True,
        progress_callback=lambda done, total, file_name: progress.append((done, total, file_name)),
    )

    assert "downloaded a.csv" in output
    assert "downloaded b.csv" in output
    assert progress == [
        (0, 2, None),
        (1, 2, "a.csv"),
        (2, 2, "b.csv"),
    ]


def test_download_competition_falls_back_to_split_on_retryable_single_shot_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [  # noqa: ARG005
            kaggle_api._CompetitionFile(name="a.csv", size_bytes=50),
            kaggle_api._CompetitionFile(name="b.csv", size_bytes=50),
        ],
    )
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 10_000)
    monkeypatch.setattr(kaggle_api, "_download_attempts", lambda: 1)

    seen: list[str] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        if "-f" not in args:
            raise KaggleCliError("transient", args, exit_code=137, output="killed")
        file_name = args[args.index("-f") + 1]
        seen.append(file_name)
        return f"downloaded {file_name}"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert "downloaded a.csv" in output
    assert "downloaded b.csv" in output
    assert seen == ["a.csv", "b.csv"]


def test_download_competition_split_skips_existing_basename_match(monkeypatch, tmp_path) -> None:
    existing = tmp_path / "1407735.tif"
    existing.write_bytes(b"x" * 7)
    files = [kaggle_api._CompetitionFile(name="deprecated_train_images/1407735.tif", size_bytes=7)]

    seen: list[list[str]] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        seen.append(args)
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api._download_competition_by_file(
        "demo",
        tmp_path,
        files,
        force=True,
        quiet=True,
        dry_run=False,
    )

    assert output == ""
    assert seen == []


def test_download_competition_split_unbounded_retry(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [kaggle_api._CompetitionFile(name="a.csv", size_bytes=70)],  # noqa: ARG005
    )
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 1)
    monkeypatch.setattr(kaggle_api, "_download_attempts", lambda: None)
    monkeypatch.setattr(kaggle_api, "_download_retry_backoff_sec", lambda: 0.0)

    counter = {"n": 0}

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        counter["n"] += 1
        if counter["n"] < 4:
            raise KaggleCliError("transient", args, exit_code=1, output="request timed out")
        return "done"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert output == "done"
    assert counter["n"] == 4


def test_download_competition_split_falls_back_to_single_shot_on_rate_limit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [  # noqa: ARG005
            kaggle_api._CompetitionFile(name="a.csv", size_bytes=70),
            kaggle_api._CompetitionFile(name="b.csv", size_bytes=70),
        ],
    )
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 1)
    monkeypatch.setattr(kaggle_api, "_download_attempts", lambda: 2)
    monkeypatch.setattr(kaggle_api, "_download_retry_backoff_sec", lambda: 0.0)

    seen: list[list[str]] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        seen.append(args)
        if "-f" in args:
            raise KaggleCliError("transient", args, exit_code=1, output="429 Too Many Requests")
        return "single-shot-ok"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert output == "single-shot-ok"
    assert len(seen) == 2
    assert seen[0][:3] == ["kaggle", "competitions", "download"]
    assert "-f" in seen[0]
    assert seen[1][:3] == ["kaggle", "competitions", "download"]
    assert "-f" not in seen[1]
