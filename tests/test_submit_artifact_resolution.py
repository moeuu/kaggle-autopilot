from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.hashing import sha256_file_or_none
from kagglebot.kernel_outputs import find_submission_file
from kagglebot.submission_artifact_resolution import (
    SubmissionArtifactResolutionError,
    resolve_valid_submission_artifact,
)
from kagglebot.submit_autofix import prepare_submit_file_autofix_for_run
from kagglebot.submit_failure_context import save_submit_autofix_repaired_path_for_run


def _write_valid_submission(path: Path, *, value: str = "fit") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"id,health_condition\n1,{value}\n", encoding="utf-8")
    return path


def _validate_submission(path: Path) -> Path:
    if path.suffix != ".csv" or not path.read_text(encoding="utf-8").startswith("id,health_condition\n"):
        raise SubmissionValidationError(f"invalid submission: {path}")
    return path


def _write_submission_report(path: Path, submission: Path, *, sha256: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "submission": {
                    "path": str(submission),
                    "sha256": sha256 or sha256_file_or_none(submission),
                }
            }
        ),
        encoding="utf-8",
    )


def _file_fix_attempt(rejected: Path) -> dict[str, object]:
    return {
        "reason": "local_submission_validation_failed",
        "error_kind": "validation",
        "sub_path": str(rejected),
        "stderr_tail": "columns mismatch; row count mismatch; id column missing",
    }


def test_kernel_reported_csv_wins_over_predictions_manifest(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter-3"
    submission = _write_valid_submission(iteration_dir / "output" / "submission.csv")
    (iteration_dir / "predictions_manifest.json").write_text(
        json.dumps([{"kind": "test", "path": "preds.npy"}] * 95),
        encoding="utf-8",
    )
    _write_submission_report(iteration_dir / "output" / "metrics.json", submission)
    _write_submission_report(iteration_dir / "output" / "run_manifest.json", submission)

    assert find_submission_file(iteration_dir) == submission.resolve()


def test_autofix_copies_reported_csv_and_records_repaired_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    iteration_dir = run_dir / "iter-3"
    rejected = iteration_dir / "predictions_manifest.json"
    rejected.parent.mkdir(parents=True)
    rejected.write_text(json.dumps([{"kind": "test"}] * 95), encoding="utf-8")
    submission = _write_valid_submission(iteration_dir / "output" / "0.csv")
    _write_submission_report(iteration_dir / "output" / "metrics.json", submission)
    state_updates: dict[str, object] = {}

    def save_repaired(path: Path) -> None:
        save_submit_autofix_repaired_path_for_run(
            run_dir=run_dir,
            repaired_path=path,
            save_run_state_for_run=lambda _run_dir, updates: state_updates.update(updates),
        )

    result = prepare_submit_file_autofix_for_run(
        latest_submit_attempt=_file_fix_attempt(rejected),
        run_state={"last_submission_path": str(rejected)},
        failure_context={
            "repair_target": "submission_artifact",
            "submission_artifact_path": str(rejected),
        },
        fallback_iteration_dirs=lambda: [iteration_dir],
        resolve_iteration_submission_artifact=lambda _iteration_dir: rejected,
        validate_and_prepare=_validate_submission,
        save_repaired_path=save_repaired,
    )

    expected = iteration_dir / "submission_autofix.csv"
    assert result.path == expected
    assert expected.read_bytes() == submission.read_bytes()
    assert _validate_submission(expected) == expected
    assert state_updates["submit_autofix_submission_path"] == str(expected)
    assert state_updates["last_submission_path"] == str(expected)
    assert state_updates["last_submission_sha256"] == sha256_file_or_none(expected)


def test_metadata_json_is_never_admitted_as_file_submission(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter-1"
    iteration_dir.mkdir()
    (iteration_dir / "predictions_manifest.json").write_text(
        json.dumps([{"prediction": 0.5}]),
        encoding="utf-8",
    )

    assert find_submission_file(iteration_dir) is None


def test_two_valid_uncorroborated_csvs_are_ambiguous(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter-1"
    _write_valid_submission(iteration_dir / "output" / "candidate_a.csv", value="fit")
    _write_valid_submission(iteration_dir / "output" / "candidate_b.csv", value="unhealthy")

    with pytest.raises(SubmissionArtifactResolutionError, match="uncorroborated.*ambiguous"):
        resolve_valid_submission_artifact(
            iteration_dir=iteration_dir,
            validate_and_prepare=_validate_submission,
        )


def test_report_path_and_sha_select_one_of_multiple_valid_csvs(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter-1"
    selected = _write_valid_submission(iteration_dir / "output" / "0.csv", value="fit")
    _write_valid_submission(iteration_dir / "output" / "candidate.csv", value="unhealthy")
    _write_submission_report(iteration_dir / "output" / "metrics.json", selected)

    resolved = resolve_valid_submission_artifact(
        iteration_dir=iteration_dir,
        validate_and_prepare=_validate_submission,
    )

    assert resolved.source_path == selected.resolve()
    assert resolved.provenance.startswith("metrics.json:")


def test_invalid_explicit_operator_path_is_not_silently_replaced(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iter-1"
    invalid = iteration_dir / "operator.json"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(json.dumps([{"prediction": 0.5}]), encoding="utf-8")
    _write_valid_submission(iteration_dir / "output" / "submission.csv")

    with pytest.raises(SubmissionValidationError, match="operator.json"):
        resolve_valid_submission_artifact(
            iteration_dir=iteration_dir,
            validate_and_prepare=_validate_submission,
            explicit_submission_path=invalid,
        )
