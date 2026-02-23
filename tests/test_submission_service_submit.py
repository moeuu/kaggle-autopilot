from __future__ import annotations

import json
from pathlib import Path

from kagglebot.submission.guard import SubmitResult
from kagglebot.submission_service import SubmissionConfig, SubmissionService


def _build_service(*, tmp_path: Path, dry_run: bool) -> tuple[SubmissionService, Path]:
    ledger_path = tmp_path / "ledger.jsonl"
    config = SubmissionConfig(
        slug="demo",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "sample_submission.csv",
        submission_ledger_path=ledger_path,
        dry_run=dry_run,
        force_submit=False,
    )
    return SubmissionService(config), ledger_path


def test_submit_prepared_dry_run_does_not_record_submission(tmp_path: Path) -> None:
    service, ledger_path = _build_service(tmp_path=tmp_path, dry_run=True)
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.123\n", encoding="utf-8")

    service.submit_prepared(prepared_path=submission_path, message="dry run", run_id=None)

    assert not ledger_path.exists()


def test_submit_prepared_records_submission_on_success(monkeypatch, tmp_path: Path) -> None:
    service, ledger_path = _build_service(tmp_path=tmp_path, dry_run=False)
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.123\n", encoding="utf-8")

    def fake_submit(*, slug: str, submission_file: Path, message: str, dry_run: bool = False) -> SubmitResult:
        assert slug == "demo"
        assert submission_file == submission_path
        assert message == "real run"
        assert dry_run is False
        return SubmitResult(
            returncode=0,
            stdout="ok",
            stderr="",
            command=["kaggle", "competitions", "submit"],
            duration_sec=0.1,
        )

    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)

    service.submit_prepared(prepared_path=submission_path, message="real run", run_id="run-1")

    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["slug"] == "demo"
    assert rows[0]["message"] == "real run"
    assert rows[0]["run_id"] == "run-1"


def test_submit_prepared_truncates_overlong_message(monkeypatch, tmp_path: Path) -> None:
    service, ledger_path = _build_service(tmp_path=tmp_path, dry_run=False)
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.123\n", encoding="utf-8")

    long_message = "m" * 150

    def fake_submit(*, slug: str, submission_file: Path, message: str, dry_run: bool = False) -> SubmitResult:
        assert slug == "demo"
        assert submission_file == submission_path
        assert dry_run is False
        assert len(message) <= 100
        assert message.endswith("...")
        return SubmitResult(
            returncode=0,
            stdout="ok",
            stderr="",
            command=["kaggle", "competitions", "submit"],
            duration_sec=0.1,
        )

    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)

    service.submit_prepared(prepared_path=submission_path, message=long_message, run_id="run-2")

    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert len(rows[0]["message"]) <= 100
    assert rows[0]["message"].endswith("...")
