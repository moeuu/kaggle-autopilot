from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kagglebot.autofix_context import prepare_autofix_context
from kagglebot.exceptions import KaggleCliError, SubmitAbortedError
from kagglebot.submit_autofix import SubmitFileAutofixPreparation
from kagglebot.submit_failure_context import SubmitAutofixRunContext


class DummyPaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "data"
        self.sample_submission_path = self.data_dir / "sample_submission.csv"
        self.submission_ledger_path = root / "submission_ledger.jsonl"

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def iter_dir(self, run_id: str, iteration: int) -> Path:
        return self.run_dir(run_id) / f"iter-{iteration}"


def test_prepare_autofix_context_writes_kaggle_cli_transcript(tmp_path: Path) -> None:
    config = SimpleNamespace(slug="demo", paths=DummyPaths(tmp_path))
    error = KaggleCliError(
        "submit failed",
        command=["kaggle", "competitions", "submit", "-c", "demo"],
        output="rules not accepted",
    )

    prepared = prepare_autofix_context(
        config=config,
        run_id="run-1",
        attempt=1,
        error=error,
        max_search_iteration=3,
        sha256_or_none=lambda path: None,
    )

    assert prepared.submit_autofix is False
    assert prepared.submit_context == ""
    assert prepared.submit_file_fix_required is False
    assert prepared.error_path.exists()
    text = prepared.error_path.read_text(encoding="utf-8")
    assert "Kaggle CLI command:" in text
    assert "kaggle competitions submit -c demo" in text
    assert "Kaggle CLI output:" in text
    assert "rules not accepted" in text


def test_prepare_autofix_context_appends_submit_repair_summary(monkeypatch, tmp_path: Path) -> None:
    config = SimpleNamespace(slug="demo", paths=DummyPaths(tmp_path))
    fixed_submission = config.paths.iter_dir("run-1", 1) / "output" / "submission-fixed.csv"
    fixed_submission.parent.mkdir(parents=True, exist_ok=True)
    fixed_submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    latest_attempt = {
        "reason": "local_submission_validation_failed",
        "error_kind": "validation",
        "stderr_tail": "bad format",
    }

    monkeypatch.setattr(
        "kagglebot.submit_failure_context.load_submit_autofix_run_context",
        lambda **kwargs: SubmitAutofixRunContext(
            failure_context={"repair_target": "submission_artifact"},
            run_state={"last_submission_path": "submission.csv"},
            latest_submit_attempt=latest_attempt,
            formatted_context="existing submit context",
        ),
    )
    monkeypatch.setattr("kagglebot.submit_autofix.submit_file_fix_required_for_attempt", lambda attempt: True)
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_autofix_submission_artifact",
        lambda **kwargs: tmp_path / "bad-submission.csv",
    )
    monkeypatch.setattr(
        "kagglebot.submit_autofix.prepare_submit_file_autofix_for_run",
        lambda **kwargs: SubmitFileAutofixPreparation(
            path=fixed_submission,
            summary=f"fixed_submission_path: {fixed_submission}",
            file_fix_required=True,
        ),
    )

    prepared = prepare_autofix_context(
        config=config,
        run_id="run-1",
        attempt=2,
        error=SubmitAbortedError("Local submission validation failed"),
        max_search_iteration=3,
        sha256_or_none=lambda path: "abc123",
    )

    assert prepared.submit_autofix is True
    assert prepared.submit_file_fix_required is True
    assert prepared.submit_file_fix_baseline_sha256 == "abc123"
    assert "existing submit context" in prepared.submit_context
    assert "fixed_submission_path" in prepared.submit_context
    text = prepared.error_path.read_text(encoding="utf-8")
    assert "Deterministic Submit File Autofix:" in text
    assert "Submit Failure Context:" in text
