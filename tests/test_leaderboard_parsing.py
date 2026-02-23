from __future__ import annotations

import pytest

from kagglebot import kaggle_api


def _write_leaderboard(tmp_path, content: str) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "leaderboard" / "leaderboard.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(content, encoding="utf-8")


def test_leaderboard_top1_empty_file_returns_none(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        _write_leaderboard(tmp_path, "")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    with pytest.warns(RuntimeWarning, match=r"Ignoring leaderboard CSV .*leaderboard\.csv.*empty"):
        result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")
    assert result["score"] is None


def test_leaderboard_top1_header_only_returns_none(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        _write_leaderboard(tmp_path, "Rank,Score\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    with pytest.warns(RuntimeWarning, match=r"Ignoring leaderboard CSV .*no score rows"):
        result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")
    assert result["score"] is None


def test_leaderboard_top1_nan_score_returns_none(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        _write_leaderboard(tmp_path, "Rank,Score\n1,NaN\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    with pytest.warns(RuntimeWarning, match=r"Ignoring leaderboard CSV .*No finite numeric score values"):
        result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")
    assert result["score"] is None


def test_leaderboard_top1_valid_score_parses_best(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        _write_leaderboard(tmp_path, "Rank,Score\n2,0.120\n1,0.110\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")
    assert result["score"] == pytest.approx(0.11)


def test_leaderboard_top1_wrong_score_column_returns_none(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        _write_leaderboard(tmp_path, "Rank,Value\n1,0.110\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    with pytest.warns(RuntimeWarning, match=r"Ignoring leaderboard CSV .*No finite numeric score values"):
        result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")
    assert result["score"] is None


def test_leaderboard_top1_zero_score_invalid_for_rmse(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        _write_leaderboard(tmp_path, "Rank,Score\n1,0.0\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    with pytest.warns(RuntimeWarning, match=r"expected a positive value"):
        result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")
    assert result["score"] is None
