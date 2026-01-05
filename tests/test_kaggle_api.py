"""Tests for kaggle_api helpers."""

from __future__ import annotations

from types import SimpleNamespace

from kagglebot import kaggle_api


def test_check_rules_accepted(monkeypatch) -> None:
    def fake_run(args, **kwargs):  # noqa: ARG001
        stdout = (
            "ref,title,userHasEntered\n"
            "https://www.kaggle.com/competitions/foo,Example,True\n"
            "https://www.kaggle.com/competitions/bar,Other,False\n"
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="", output=stdout)

    monkeypatch.setattr(kaggle_api, "run_command", fake_run)
    assert kaggle_api.check_rules_accepted("foo") is True
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
