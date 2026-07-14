"""Tests for kaggle_api helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kagglebot import kaggle_api
from kagglebot.exceptions import KaggleCliError, KaggleCliResourceError, KernelCapacityError, RulesNotAcceptedError
from kagglebot.exec_utils import CommandResult

pytestmark = pytest.mark.slow


def test_submission_row_from_api_preserves_format_error_description() -> None:
    row = kaggle_api._submission_row_from_api(
        SimpleNamespace(
            _file_name="submission.csv",
            _status="SubmissionStatus.COMPLETE",
            _public_score="",
            _private_score="",
            _error_description="Your notebook generated a submission file with incorrect format.",
        )
    )

    assert row["fileName"] == "submission.csv"
    assert row["errorDescription"] == "Your notebook generated a submission file with incorrect format."
    assert row["publicScore"] == ""


def test_optional_datetime_normalizes_iso_values_to_utc() -> None:
    assert kaggle_api._optional_datetime("2026-02-24T01:00:00Z") == datetime(2026, 2, 24, 1, tzinfo=UTC)
    assert kaggle_api._optional_datetime("2026-02-24T10:00:00+09:00") == datetime(2026, 2, 24, 1, tzinfo=UTC)
    assert kaggle_api._optional_datetime("not a date") is None


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


def test_kaggle_api_credentials_skips_invalid_and_non_object_config_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    valid = tmp_path / "valid.json"
    valid.write_text('{"username": "cfg-user", "key": "cfg-key"}', encoding="utf-8")

    assert kaggle_api._kaggle_api_credentials(config_candidates=[invalid, array_payload, valid]) == (
        "cfg-user",
        "cfg-key",
    )


def test_entered_competition_optional_int_fields_reject_bool_and_fractional_values() -> None:
    class Competition:
        slug = "demo"
        title = "Demo"
        url = "https://www.kaggle.com/competitions/demo"
        category = "Playground"
        reward = ""
        evaluation_metric = "auc"
        deadline = None
        enabled_date = None
        new_entrant_deadline = None
        merger_deadline = None
        team_count = True
        max_daily_submissions = 3.5
        is_kernels_submissions_only = False
        submissions_disabled = False

    entered = kaggle_api._entered_competition_from_api(Competition())

    assert entered.team_count is None
    assert entered.max_daily_submissions is None


def test_entered_competition_optional_int_fields_accept_integral_float_values() -> None:
    class Competition:
        slug = "demo"
        title = "Demo"
        url = "https://www.kaggle.com/competitions/demo"
        category = "Playground"
        reward = ""
        evaluation_metric = "auc"
        deadline = None
        enabled_date = None
        new_entrant_deadline = None
        merger_deadline = None
        team_count = 12.0
        max_daily_submissions = "5"
        is_kernels_submissions_only = False
        submissions_disabled = False

    entered = kaggle_api._entered_competition_from_api(Competition())

    assert entered.team_count == 12
    assert entered.max_daily_submissions == 5


def test_entered_competition_normalizes_mixed_case_underscore_slug() -> None:
    class Competition:
        slug = "WiDSWorldWide_GlobalDathon26"
        title = "WiDS Worldwide Global Datathon 2026"
        url = "https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26"
        category = "Community"
        reward = "25,000 Usd"
        evaluation_metric = ""
        deadline = None
        enabled_date = None
        new_entrant_deadline = None
        merger_deadline = None
        team_count = 1754
        max_daily_submissions = 5
        is_kernels_submissions_only = False
        submissions_disabled = False

    entered = kaggle_api._entered_competition_from_api(Competition())

    assert entered.slug == "widsworldwide_globaldathon26"


def test_list_entered_competitions_dry_run_still_reads_entered_group(monkeypatch) -> None:
    from kaggle.api import kaggle_api_extended

    calls: list[dict[str, object]] = []

    class FakeApi:
        def authenticate(self) -> None:
            calls.append({"authenticate": True})

        def competitions_list(self, **kwargs):
            calls.append(kwargs)
            if kwargs["page"] > 1:
                return []
            return [
                type(
                    "Competition",
                    (),
                    {
                        "slug": "demo",
                        "title": "Demo",
                        "url": "https://www.kaggle.com/competitions/demo",
                        "category": "Playground",
                        "reward": "",
                        "evaluation_metric": "auc",
                        "deadline": None,
                        "enabled_date": None,
                        "new_entrant_deadline": None,
                        "merger_deadline": None,
                        "team_count": 10,
                        "max_daily_submissions": 5,
                        "is_kernels_submissions_only": False,
                        "submissions_disabled": False,
                    },
                )()
            ]

    monkeypatch.setattr(kaggle_api_extended, "KaggleApi", FakeApi)

    entered = kaggle_api.list_entered_competitions(page_limit=5, dry_run=True)

    assert [item.slug for item in entered] == ["demo"]
    assert calls == [
        {"authenticate": True},
        {"group": "entered", "page": 1, "sort_by": "latestDeadline"},
    ]


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


def test_kernels_push_detects_capacity_limit_even_on_zero_exit(monkeypatch) -> None:
    def fake_run_command(args, *, dry_run=False, **kwargs):  # noqa: ARG001
        return CommandResult(
            args=args,
            returncode=0,
            stdout="Kernel push error: Maximum batch GPU session count of 2 reached.",
            stderr="",
            duration_sec=0.1,
        )

    monkeypatch.setattr(kaggle_api, "run_command", fake_run_command)

    with pytest.raises(KernelCapacityError, match="GPU session limit"):
        kaggle_api.kernels_push(Path("kernel"), slug="demo", dry_run=False)


def test_kernels_push_detects_weekly_gpu_quota_even_on_zero_exit(monkeypatch) -> None:
    def fake_run_command(args, *, dry_run=False, **kwargs):  # noqa: ARG001
        return CommandResult(
            args=args,
            returncode=0,
            stdout="Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.",
            stderr="",
            duration_sec=0.1,
        )

    monkeypatch.setattr(kaggle_api, "run_command", fake_run_command)

    with pytest.raises(KernelCapacityError, match="GPU session limit"):
        kaggle_api.kernels_push(Path("kernel"), slug="demo", dry_run=False)


def test_kernels_push_detects_non_capacity_push_error_even_on_zero_exit(monkeypatch) -> None:
    def fake_run_command(args, *, dry_run=False, **kwargs):  # noqa: ARG001
        return CommandResult(
            args=args,
            returncode=0,
            stdout="Kernel push error: Notebook not found",
            stderr="",
            duration_sec=0.1,
        )

    monkeypatch.setattr(kaggle_api, "run_command", fake_run_command)

    with pytest.raises(KaggleCliError, match="Kaggle kernel push failed") as exc_info:
        kaggle_api.kernels_push(Path("kernel"), slug="demo", dry_run=False)

    assert "Notebook not found" in exc_info.value.output


def test_run_kaggle_applies_memory_limit_and_classifies_sigkill(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args, *, dry_run=False, memory_limit_mb=None, **kwargs):  # noqa: ARG001
        captured["memory_limit_mb"] = memory_limit_mb
        return CommandResult(args=args, returncode=-9, stdout="", stderr="", duration_sec=0.1)

    monkeypatch.setattr(kaggle_api, "run_command", fake_run_command)

    with pytest.raises(KaggleCliResourceError):
        kaggle_api._run_kaggle(["kaggle", "competitions", "download", "-c", "demo"], slug="demo", dry_run=False)

    assert captured["memory_limit_mb"] == 8192


def test_download_competition_does_not_retry_resource_guard(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        calls["n"] += 1
        raise KaggleCliResourceError("resource guard", args, exit_code=-9, output="")

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)

    with pytest.raises(KaggleCliResourceError):
        kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert calls["n"] == 1


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


def test_leaderboard_top1_finds_nested_csv_after_extract(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        zip_path = tmp_path / "leaderboard" / "demo.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("exports/demo/demo-leaderboard.csv", "Rank,Score\n1,0.876\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_top1("demo", tmp_path)
    assert result["score"] == 0.876


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


def test_competition_total_size_bytes_sums_listed_files(monkeypatch) -> None:
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [
            kaggle_api._CompetitionFile(name="train.csv", size_bytes=100),
            kaggle_api._CompetitionFile(name="test.csv", size_bytes=200),
        ],
    )

    assert kaggle_api.competition_total_size_bytes("demo") == 300


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


def test_download_min_interval_sec_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_MIN_INTERVAL_SEC", "1.5")
    assert kaggle_api._download_min_interval_sec() == 1.5


def test_download_rate_limit_attempts_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_RATE_LIMIT_RETRY_ATTEMPTS", "3")
    assert kaggle_api._download_rate_limit_attempts() == 3


def test_download_env_readers_fallback_for_invalid_and_non_finite(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_RATE_LIMIT_RETRY_ATTEMPTS", "3.5")
    assert kaggle_api._download_rate_limit_attempts() is None

    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_MIN_INTERVAL_SEC", "nan")
    assert kaggle_api._download_min_interval_sec() == 0.25


def test_kaggle_cli_memory_limit_env_uses_shared_number_parsing(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_KAGGLE_CLI_MEMORY_LIMIT_MB", "1024.0")
    assert kaggle_api._kaggle_cli_memory_limit_mb() == 1024

    monkeypatch.setenv("KAGGLEBOT_KAGGLE_CLI_MEMORY_LIMIT_MB", "0")
    assert kaggle_api._kaggle_cli_memory_limit_mb() is None

    monkeypatch.setenv("KAGGLEBOT_KAGGLE_CLI_MEMORY_LIMIT_MB", "nan")
    assert kaggle_api._kaggle_cli_memory_limit_mb() == 8192


def test_download_single_shot_first_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_SINGLE_SHOT_FIRST", "0")
    assert kaggle_api._download_single_shot_first_enabled() is False

    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_SINGLE_SHOT_FIRST", "maybe")
    assert kaggle_api._download_single_shot_first_enabled() is False


def test_download_streaming_and_preserve_path_flags_use_shared_env_parser(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_STREAMING", "off")
    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_PRESERVE_PATHS", "false")

    assert kaggle_api._download_streaming_enabled() is False
    assert kaggle_api._download_preserve_paths_enabled() is False

    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_STREAMING", "yes")
    monkeypatch.setenv("KAGGLEBOT_DOWNLOAD_PRESERVE_PATHS", "maybe")

    assert kaggle_api._download_streaming_enabled() is True
    assert kaggle_api._download_preserve_paths_enabled() is True


def test_rate_limit_retry_sleep_uses_longer_backoff(monkeypatch) -> None:
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_backoff_sec", lambda: 30.0)
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_max_backoff_sec", lambda: 120.0)
    error = KaggleCliError("transient", ["kaggle"], exit_code=1, output="429 Too Many Requests")

    assert kaggle_api._compute_retry_sleep_sec(attempt=3, base_backoff=2.0, error=error) == 120.0


def test_rate_limit_retry_sleep_respects_retry_after(monkeypatch) -> None:
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_backoff_sec", lambda: 2.0)
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_max_backoff_sec", lambda: 120.0)
    error = KaggleCliError(
        "transient",
        ["GET"],
        exit_code=429,
        output="HTTP 429 retry-after=45: Too Many Requests",
    )

    assert kaggle_api._compute_retry_sleep_sec(attempt=1, base_backoff=2.0, error=error) == 45.0


def test_unbounded_retry_backoff_calculation_remains_capped(monkeypatch) -> None:
    monkeypatch.setattr(kaggle_api, "_download_retry_max_backoff_sec", lambda: 120.0)

    assert kaggle_api._compute_retry_sleep_sec(attempt=100_000, base_backoff=2.0) == 120.0


def test_apply_download_pacing_sleeps_until_interval(monkeypatch) -> None:
    ticks = iter([10.2, 10.5])
    sleeps: list[float] = []

    monkeypatch.setattr(kaggle_api.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(kaggle_api.time, "sleep", lambda sec: sleeps.append(sec))

    started_at = kaggle_api._apply_download_pacing(
        min_interval_sec=0.5,
        last_request_started_at=10.0,
    )

    assert len(sleeps) == 1
    assert abs(sleeps[0] - 0.3) < 1e-6
    assert started_at == 10.5


def test_download_competition_uses_single_shot_for_small_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: True)
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: pytest.fail("single-shot download should avoid listing competition files"),  # noqa: ARG005
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


def test_download_competition_skips_when_all_files_already_present(monkeypatch, tmp_path) -> None:
    files = [
        kaggle_api._CompetitionFile(name="a.csv", size_bytes=3),
        kaggle_api._CompetitionFile(name="b.csv", size_bytes=4),
    ]
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_list_competition_files_with_sizes", lambda slug, dry_run: files)  # noqa: ARG005
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 10_000)

    (tmp_path / "a.csv").write_bytes(b"abc")
    (tmp_path / "b.csv").write_bytes(b"defg")

    seen: list[list[str]] = []

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        seen.append(args)
        return "should-not-run"

    progress: list[tuple[int, int, str | None]] = []
    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)

    output = kaggle_api.download_competition(
        "demo",
        tmp_path,
        force=True,
        quiet=True,
        progress_callback=lambda done, total, file_name: progress.append((done, total, file_name)),
    )

    assert output == ""
    assert seen == []
    assert progress == [(2, 2, None)]


def test_download_competition_splits_and_retries_large_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
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
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
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
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
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
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
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


def test_streaming_download_preserves_competition_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_chunk_bytes", lambda: 2)

    class FakeResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def iter_content(self, chunk_size):  # noqa: ANN001
            assert chunk_size == 2
            yield b"ab"
            yield b"c"

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def get(self, url, *, headers, stream, timeout):  # noqa: ANN001
            self.calls.append({"url": url, "headers": headers, "stream": stream, "timeout": timeout})
            return FakeResponse()

    session = FakeSession()
    output = kaggle_api._download_competition_file_streaming(
        slug="demo",
        dest_dir=tmp_path,
        file=kaggle_api._CompetitionFile(name="train_audio/123/a.ogg", size_bytes=3),
        force=True,
        quiet=True,
        session=session,
    )

    assert "train_audio/123/a.ogg" in output
    assert (tmp_path / "train_audio" / "123" / "a.ogg").read_bytes() == b"abc"
    assert not (tmp_path / "a.ogg").exists()
    assert session.calls[0]["stream"] is True
    assert "train_audio%2F123%2Fa.ogg" in str(session.calls[0]["url"])


def test_download_competition_streaming_single_shot_avoids_file_listing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: True)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: True)

    def fail_list(slug, dry_run):  # noqa: ANN001, ARG001
        raise AssertionError("streaming single-shot should not list files first")

    monkeypatch.setattr(kaggle_api, "_list_competition_files_with_sizes", fail_list)
    monkeypatch.setattr(
        kaggle_api,
        "_download_competition_all_streaming_with_retry",
        lambda *, slug, dest_dir, force, quiet: "streamed-all",  # noqa: ARG005
    )

    progress: list[tuple[int, int, str | None]] = []
    output = kaggle_api.download_competition(
        "demo",
        tmp_path,
        force=True,
        quiet=True,
        progress_callback=lambda done, total, file_name: progress.append((done, total, file_name)),
    )

    assert output == "streamed-all"
    assert progress == [(1, 1, "demo.zip")]


def test_download_competition_streams_large_data_without_kaggle_cli(monkeypatch, tmp_path) -> None:
    files = [kaggle_api._CompetitionFile(name="train_audio/123/a.ogg", size_bytes=3)]
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: True)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_list_competition_files_with_sizes", lambda slug, dry_run: files)  # noqa: ARG005
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 1)

    class FakeSession:
        def close(self) -> None:
            pass

    streamed: list[str] = []
    monkeypatch.setattr(kaggle_api, "_build_kaggle_download_session", lambda: FakeSession())

    def fake_stream_download(*, slug, dest_dir, file, force, quiet, session):  # noqa: ANN001, ARG001
        streamed.append(file.name)
        output_path = dest_dir / file.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"abc")
        return f"streamed {file.name}"

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ANN001, ARG001
        raise AssertionError("large streaming download must not call Kaggle CLI download")

    monkeypatch.setattr(kaggle_api, "_download_competition_file_streaming", fake_stream_download)
    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)

    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert output == "streamed train_audio/123/a.ogg"
    assert streamed == ["train_audio/123/a.ogg"]


def test_download_competition_streams_small_data_by_file_by_default(monkeypatch, tmp_path) -> None:
    files = [kaggle_api._CompetitionFile(name="train.csv", size_bytes=3)]
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: True)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_list_competition_files_with_sizes", lambda slug, dry_run: files)  # noqa: ARG005
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 10**12)

    class FakeSession:
        def close(self) -> None:
            pass

    monkeypatch.setattr(kaggle_api, "_build_kaggle_download_session", lambda: FakeSession())

    def fake_stream(**kwargs) -> str:  # noqa: ANN003
        (kwargs["dest_dir"] / kwargs["file"].name).write_bytes(b"abc")
        return "streamed"

    monkeypatch.setattr(kaggle_api, "_download_competition_file_streaming", fake_stream)
    monkeypatch.setattr(
        kaggle_api,
        "_run_kaggle",
        lambda *args, **kwargs: pytest.fail("streaming mode must not invoke the CLI bundle download"),
    )

    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert output == "streamed"
    assert (tmp_path / "train.csv").read_bytes() == b"abc"


def test_streaming_download_promotes_complete_partial_without_network(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_disk_reserve_bytes", lambda: 0)
    output_path = tmp_path / "nested" / "train.bin"
    output_path.parent.mkdir(parents=True)
    part_path = output_path.with_name("train.bin.part")
    part_path.write_bytes(b"abc")

    class NoNetworkSession:
        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("a complete partial file should be finalized without a request")

    output = kaggle_api._download_competition_file_streaming(
        slug="demo",
        dest_dir=tmp_path,
        file=kaggle_api._CompetitionFile(name="nested/train.bin", size_bytes=3),
        force=True,
        quiet=True,
        session=NoNetworkSession(),
    )

    assert output.startswith("resumed nested/train.bin")
    assert output_path.read_bytes() == b"abc"
    assert not part_path.exists()


def test_streaming_download_resumes_partial_file_with_valid_range(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_disk_reserve_bytes", lambda: 0)
    output_path = tmp_path / "nested" / "train.bin"
    output_path.parent.mkdir(parents=True)
    output_path.with_name("train.bin.part").write_bytes(b"ab")

    class FakeResponse:
        status_code = 206
        text = ""
        headers = {"Content-Range": "bytes 2-2/3", "Content-Length": "1"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def iter_content(self, chunk_size):  # noqa: ANN001, ARG002
            yield b"c"

    class FakeSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] | None = None

        def get(self, url, *, headers, stream, timeout):  # noqa: ANN001, ARG002
            assert stream is True
            self.headers = headers
            return FakeResponse()

    session = FakeSession()
    kaggle_api._download_competition_file_streaming(
        slug="demo",
        dest_dir=tmp_path,
        file=kaggle_api._CompetitionFile(name="nested/train.bin", size_bytes=3),
        force=True,
        quiet=True,
        session=session,
    )

    assert output_path.read_bytes() == b"abc"
    assert session.headers == {"Accept-Encoding": "identity", "Range": "bytes=2-"}


def test_streaming_session_supports_access_token_auth(monkeypatch) -> None:
    from kaggle.api import kaggle_api_extended

    class FakeApi:
        CONFIG_NAME_TOKEN = "token"
        CONFIG_NAME_USER = "username"
        CONFIG_NAME_KEY = "key"

        def __init__(self) -> None:
            self.config_values = {"token": "test-access-token"}

        def authenticate(self) -> None:
            pass

    monkeypatch.setattr(kaggle_api_extended, "KaggleApi", FakeApi)

    session = kaggle_api._build_kaggle_download_session()
    try:
        assert session.headers["Authorization"] == "Bearer test-access-token"
    finally:
        session.close()


def test_download_capacity_fails_before_partial_write(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_disk_reserve_bytes", lambda: 100)
    monkeypatch.setattr(kaggle_api.shutil, "disk_usage", lambda path: SimpleNamespace(free=105))

    with pytest.raises(KaggleCliResourceError, match="Insufficient disk space"):
        kaggle_api._ensure_download_capacity(
            tmp_path,
            expected_size=20,
            partial_size=10,
            file_name="huge.bin",
        )


def test_streaming_count_requires_preserved_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: True)
    monkeypatch.setattr(kaggle_api, "_download_preserve_paths_enabled", lambda: True)
    files = [kaggle_api._CompetitionFile(name="train_audio/123/a.ogg", size_bytes=3)]

    (tmp_path / "a.ogg").write_bytes(b"abc")
    assert kaggle_api._count_downloaded_competition_files(tmp_path, files) == 0

    nested = tmp_path / "train_audio" / "123" / "a.ogg"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"abc")
    assert kaggle_api._count_downloaded_competition_files(tmp_path, files) == 1


def test_count_downloaded_files_uses_size_to_disambiguate_duplicate_basenames(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    files = [
        kaggle_api._CompetitionFile(name="deprecated_train_images/1407735.tif", size_bytes=26641798),
        kaggle_api._CompetitionFile(name="train_images/1407735.tif", size_bytes=1059109),
    ]
    (tmp_path / "1407735.tif").write_bytes(b"x" * 26641798)

    completed = kaggle_api._count_downloaded_competition_files(tmp_path, files)

    assert completed == 1


def test_download_competition_split_unbounded_retry(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
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


def test_download_competition_split_retries_rate_limit_without_single_shot_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
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
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_attempts", lambda: None)
    monkeypatch.setattr(kaggle_api, "_download_retry_backoff_sec", lambda: 0.0)
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_backoff_sec", lambda: 0.0)

    seen: list[list[str]] = []
    attempts = {"a": 0}

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        seen.append(args)
        assert "-f" in args
        file_name = args[args.index("-f") + 1]
        if file_name == "a.csv":
            attempts["a"] += 1
            if attempts["a"] < 3:
                raise KaggleCliError("transient", args, exit_code=1, output="429 Too Many Requests")
        return f"ok-{file_name}"

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    output = kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert "ok-a.csv" in output
    assert "ok-b.csv" in output
    assert len(seen) == 4
    assert all("-f" in args for args in seen)


def test_download_competition_split_stops_after_rate_limit_retry_budget(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kaggle_api, "_download_streaming_enabled", lambda: False)
    monkeypatch.setattr(kaggle_api, "_download_single_shot_first_enabled", lambda: False)
    monkeypatch.setattr(
        kaggle_api,
        "_list_competition_files_with_sizes",
        lambda slug, dry_run: [kaggle_api._CompetitionFile(name="a.csv", size_bytes=70)],  # noqa: ARG005
    )
    monkeypatch.setattr(kaggle_api, "_split_download_threshold_bytes", lambda: 1)
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_attempts", lambda: 2)
    monkeypatch.setattr(kaggle_api, "_download_retry_backoff_sec", lambda: 0.0)
    monkeypatch.setattr(kaggle_api, "_download_rate_limit_backoff_sec", lambda: 0.0)

    counter = {"n": 0}

    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        counter["n"] += 1
        raise KaggleCliError("transient", args, exit_code=1, output="429 Too Many Requests")

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)

    with pytest.raises(KaggleCliError):
        kaggle_api.download_competition("demo", tmp_path, force=True, quiet=True)

    assert counter["n"] == 2
