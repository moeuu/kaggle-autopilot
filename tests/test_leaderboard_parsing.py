from __future__ import annotations

import os
import tarfile
import zipfile
from io import BytesIO

import py7zr
import pytest
import zstandard as zstd

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


def test_leaderboard_top1_handles_utf8_bom_rank_header(monkeypatch, tmp_path) -> None:
    def fake_run_kaggle(args, slug, dry_run):  # noqa: ARG001
        _write_leaderboard(tmp_path, "\ufeffRank,Score\n2,0.120\n1,0.110\n")
        return ""

    monkeypatch.setattr(kaggle_api, "_run_kaggle", fake_run_kaggle)
    result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")
    assert result["score"] == pytest.approx(0.11)


def test_leaderboard_top1_extracts_newer_zip_over_stale_csv(monkeypatch, tmp_path) -> None:
    leaderboard_dir = tmp_path / "leaderboard"
    leaderboard_dir.mkdir()
    stale_csv = leaderboard_dir / "demo-publicleaderboard-2026-05-01T00:00:00.csv"
    stale_csv.write_text("Rank,Score\n1,0.500\n", encoding="utf-8")
    zip_path = leaderboard_dir / "demo.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("demo-publicleaderboard-2026-05-22T00:00:00.csv", "Rank,Score\n1,0.110\n")
    os.utime(stale_csv, (1_000, 1_000))
    os.utime(zip_path, (2_000, 2_000))

    monkeypatch.setattr(kaggle_api, "_run_kaggle", lambda args, slug, dry_run: "")

    result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")

    assert result["score"] == pytest.approx(0.11)


def test_leaderboard_top1_extracts_newer_tar_xz_over_stale_csv(monkeypatch, tmp_path) -> None:
    leaderboard_dir = tmp_path / "leaderboard"
    leaderboard_dir.mkdir()
    stale_csv = leaderboard_dir / "demo-publicleaderboard-2026-05-01T00:00:00.csv"
    stale_csv.write_text("Rank,Score\n1,0.500\n", encoding="utf-8")
    tar_path = leaderboard_dir / "demo.tar.xz"
    payload = b"Rank,Score\n1,0.090\n"
    with tarfile.open(tar_path, "w:xz") as archive:
        info = tarfile.TarInfo("demo-publicleaderboard-2026-05-22T00:00:00.csv")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    os.utime(stale_csv, (1_000, 1_000))
    os.utime(tar_path, (2_000, 2_000))

    monkeypatch.setattr(kaggle_api, "_run_kaggle", lambda args, slug, dry_run: "")

    result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")

    assert result["score"] == pytest.approx(0.09)


def test_leaderboard_top1_extracts_newer_tar_zst_over_stale_csv(monkeypatch, tmp_path) -> None:
    leaderboard_dir = tmp_path / "leaderboard"
    leaderboard_dir.mkdir()
    stale_csv = leaderboard_dir / "demo-publicleaderboard-2026-05-01T00:00:00.csv"
    stale_csv.write_text("Rank,Score\n1,0.500\n", encoding="utf-8")
    tar_path = leaderboard_dir / "demo.tar.zst"
    raw_tar = BytesIO()
    payload = b"Rank,Score\n1,0.080\n"
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        info = tarfile.TarInfo("demo-publicleaderboard-2026-05-22T00:00:00.csv")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    tar_path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))
    os.utime(stale_csv, (1_000, 1_000))
    os.utime(tar_path, (2_000, 2_000))

    monkeypatch.setattr(kaggle_api, "_run_kaggle", lambda args, slug, dry_run: "")

    result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")

    assert result["score"] == pytest.approx(0.08)


def test_leaderboard_top1_extracts_newer_7z_over_stale_csv(monkeypatch, tmp_path) -> None:
    leaderboard_dir = tmp_path / "leaderboard"
    leaderboard_dir.mkdir()
    stale_csv = leaderboard_dir / "demo-publicleaderboard-2026-05-01T00:00:00.csv"
    stale_csv.write_text("Rank,Score\n1,0.500\n", encoding="utf-8")
    source_csv = tmp_path / "demo-publicleaderboard-2026-05-22T00:00:00.csv"
    source_csv.write_text("Rank,Score\n1,0.070\n", encoding="utf-8")
    archive_path = leaderboard_dir / "demo.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(source_csv, source_csv.name)
    os.utime(stale_csv, (1_000, 1_000))
    os.utime(archive_path, (2_000, 2_000))

    monkeypatch.setattr(kaggle_api, "_run_kaggle", lambda args, slug, dry_run: "")

    result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")

    assert result["score"] == pytest.approx(0.07)


def test_leaderboard_top1_dispatches_newer_rar_over_stale_csv(monkeypatch, tmp_path) -> None:
    leaderboard_dir = tmp_path / "leaderboard"
    leaderboard_dir.mkdir()
    stale_csv = leaderboard_dir / "demo-publicleaderboard-2026-05-01T00:00:00.csv"
    stale_csv.write_text("Rank,Score\n1,0.500\n", encoding="utf-8")
    archive_path = leaderboard_dir / "demo.rar"
    archive_path.write_bytes(b"fake-rar")
    os.utime(stale_csv, (1_000, 1_000))
    os.utime(archive_path, (2_000, 2_000))

    def fake_extract_rar(path, dest_dir):  # noqa: ANN001
        assert path == archive_path
        extracted = dest_dir / "demo-publicleaderboard-2026-05-22T00:00:00.csv"
        extracted.write_text("Rank,Score\n1,0.060\n", encoding="utf-8")
        return [extracted]

    monkeypatch.setattr(kaggle_api, "_run_kaggle", lambda args, slug, dry_run: "")
    monkeypatch.setattr(kaggle_api, "safe_extract_rar", fake_extract_rar)

    result = kaggle_api.leaderboard_top1("demo", tmp_path, metric_hint="rmse_log1p")

    assert result["score"] == pytest.approx(0.06)


def test_leaderboard_rank_for_score_dry_run_reads_cached_csv(monkeypatch, tmp_path) -> None:
    _write_leaderboard(tmp_path, "Rank,Score\n1,0.100\n2,0.200\n3,0.300\n")
    monkeypatch.setattr(
        kaggle_api,
        "_run_kaggle",
        lambda args, slug, dry_run: pytest.fail("dry-run should not call Kaggle CLI"),  # noqa: ARG005
    )

    result = kaggle_api.leaderboard_rank_for_score(
        "demo",
        tmp_path,
        score=0.250,
        direction="minimize",
        dry_run=True,
    )

    assert result["rank"] == 3
    assert result["total_teams"] == 3
    assert result["rank_percentile"] == pytest.approx(1.0)


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
