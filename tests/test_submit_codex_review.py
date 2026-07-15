from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kagglebot.exceptions import (
    DuplicateSubmissionError,
    SubmissionRateLimitError,
    SubmissionValidationError,
)
from kagglebot.submit_codex_review import (
    assert_code_submission_review_approved,
    recheck_code_submission_execution_guard,
    record_code_submission_execution,
    review_code_submission_before_execute,
)


def _candidate(tmp_path: Path) -> dict[str, Path]:
    package_dir = tmp_path / "package"
    output_dir = tmp_path / "output"
    logs_dir = tmp_path / "logs"
    package_dir.mkdir()
    output_dir.mkdir()
    logs_dir.mkdir()
    (package_dir / "kernel.py").write_text("MODEL = 'real-model'\n", encoding="utf-8")
    (package_dir / "kernel-metadata.json").write_text(
        json.dumps({"id": "user/kernel", "competition_sources": ["demo"]}),
        encoding="utf-8",
    )
    submission_path = output_dir / "submission.json"
    submission_path.write_text('{"task": [{"attempt_1": [[1]], "attempt_2": [[2]]}]}', encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "chosen_pipeline": "model",
                "score": 0.5,
                "score_source": "cv",
                "active_model_source": "dataset/model/1",
                "pipelines": [{"name": "model", "score": 0.5, "correct": 1, "total": 2}],
                "test_prediction_distribution": {"source_top10": [["model_decode", 1]], "n_outputs": 1},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "runtime.log").write_text("Model loaded\nGenerated model_decode output\n", encoding="utf-8")
    return {
        "package_dir": package_dir,
        "output_dir": output_dir,
        "logs_dir": logs_dir,
        "submission_path": submission_path,
        "metrics_path": metrics_path,
    }


def _codex_decision(*, decision: str = "approve", checks: bool = True):
    def fake(prompt_path: Path, output_dir: Path, **_kwargs):
        evidence = json.loads((prompt_path.parent / "evidence.json").read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        last_message_path = output_dir / "codex_last_message.txt"
        last_message_path.write_text(
            json.dumps(
                {
                    "decision": decision,
                    "checks": {
                        "notebook": checks,
                        "model": checks,
                        "output_contract": checks,
                        "runtime_logs": checks,
                    },
                    "reasons": [] if decision == "approve" else ["runtime evidence failed"],
                    "evidence_digest": evidence["evidence_digest"],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(last_message_path=last_message_path, returncode=0)

    return fake


def _review(tmp_path: Path, candidate: dict[str, Path]):
    return review_code_submission_before_execute(
        slug="demo",
        run_id="run-1",
        iteration=1,
        kernel_id="user/kernel",
        kernel_version="2",
        package_dir=candidate["package_dir"],
        output_dir=candidate["output_dir"],
        runtime_logs_dir=candidate["logs_dir"],
        submission_path=candidate["submission_path"],
        metrics_path=candidate["metrics_path"],
        expected_output_file=candidate["submission_path"].name,
        message="human-readable submission",
        review_dir=tmp_path / "review",
        run_codex_func=_codex_decision(),
    )


def test_review_guard_and_ledger_bind_exact_kernel_version(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    approval = _review(tmp_path, candidate)
    ledger_path = tmp_path / "ledger.jsonl"
    iteration_state_path = tmp_path / "run-1" / "iter-1" / "iteration_state.json"
    iteration_state_path.parent.mkdir(parents=True)
    iteration_state_path.write_text(
        json.dumps({"run_id": "run-1", "iteration": 1, "submitted": False}),
        encoding="utf-8",
    )
    permit = recheck_code_submission_execution_guard(
        approval=approval,
        slug="demo",
        kernel_id="user/kernel",
        kernel_version="2",
        expected_output_file="submission.json",
        submission_path=candidate["submission_path"],
        message="human-readable submission",
        submission_ledger_path=ledger_path,
        submission_limit_per_day=None,
        fetch_submission_rows=lambda _slug: [],
        force_submit=False,
    )
    record_code_submission_execution(
        permit=permit,
        slug="demo",
        message="human-readable submission",
        submission_path=candidate["submission_path"],
        submission_ledger_path=ledger_path,
        run_id="run-1",
        iteration=1,
        submission_ref="kernel:user/kernel",
        iteration_state_path=iteration_state_path,
    )
    pending_state = json.loads(iteration_state_path.read_text(encoding="utf-8"))
    assert pending_state["submitted"] is True
    assert pending_state["submit_phase_state"] == "submitted_pending"
    assert pending_state["submit_phase_finished"] is False
    assert pending_state["submission_identity"] == "kernel:user/kernel:version:2:output:submission.json"

    with pytest.raises(DuplicateSubmissionError):
        recheck_code_submission_execution_guard(
            approval=approval,
            slug="demo",
            kernel_id="user/kernel",
            kernel_version="2",
            expected_output_file="submission.json",
            submission_path=candidate["submission_path"],
            message="human-readable submission",
            submission_ledger_path=ledger_path,
            submission_limit_per_day=None,
            fetch_submission_rows=lambda _slug: [],
            force_submit=False,
        )


def test_review_evidence_surfaces_reference_model_provenance(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    metrics = json.loads(candidate["metrics_path"].read_text(encoding="utf-8"))
    metrics.update(
        {
            "selected_pipeline": "verified_reference_inference",
            "metric_name": "adjusted_edge_jaccard + 0.1 * division_jaccard",
            "artifact_mode": "reference_public_artifact",
            "authoritative": False,
            "evidence_note": "Public leaderboard evidence; current test inference completed.",
            "reference_public_artifact": {"status": "verified", "blockers": []},
            "reference_public_score_normalized": 0.897,
            "artifact_hashes": ["abc123"],
        }
    )
    candidate["metrics_path"].write_text(json.dumps(metrics), encoding="utf-8")

    approval = _review(tmp_path, candidate)
    evidence = json.loads(approval.evidence_path.read_text(encoding="utf-8"))
    prompt = (approval.review_path.parent / "prompt.md").read_text(encoding="utf-8")

    assert evidence["metrics_summary"]["selected_pipeline"] == "verified_reference_inference"
    assert evidence["metrics_summary"]["reference_public_artifact"]["status"] == "verified"
    assert evidence["metrics_summary"]["authoritative"] is False
    assert "runtime_logs_dir" not in evidence["paths"]
    assert evidence["paths"]["runtime_log_paths"] == [str((candidate["output_dir"] / "runtime.log").resolve())]
    assert "Do not search parent or sibling" in prompt
    assert "directories for other logs" in prompt
    assert "non-authoritative public-reference score" in prompt


def test_review_rejects_restored_zero_total_and_all_fallback(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate["metrics_path"].write_text(
        json.dumps(
            {
                "pipelines": [
                    {
                        "name": "model",
                        "cv_score": 86.5,
                        "correct": 0,
                        "total": 0,
                        "notes": ["CV skipped; restored from local qualification run"],
                    }
                ],
                "test_prediction_distribution": {
                    "source_top10": [["fallback_shape_color", 259], ["fallback_diversity", 259]],
                    "n_outputs": 259,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="restored.*total is zero"):
        _review(tmp_path, candidate)


def test_review_allows_restored_selection_score_after_full_model_backed_inference(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate["metrics_path"].write_text(
        json.dumps(
            {
                "reference_path_used": True,
                "reference_test_output_count": 259,
                "reference_test_output_ratio": 1.0,
                "pipelines": [
                    {
                        "name": "model",
                        "cv_score": 86.5,
                        "correct": 0,
                        "total": 0,
                        "notes": ["CV skipped; restored from local qualification run"],
                    }
                ],
                "test_prediction_distribution": {
                    "source_top10": [["nvarc_qwen3_ttt_turbo_dfs", 259]],
                    "n_outputs": 259,
                },
            }
        ),
        encoding="utf-8",
    )

    approval = _review(tmp_path, candidate)

    assert_code_submission_review_approved(approval)


def test_review_rejects_repeated_runtime_exception(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    (candidate["output_dir"] / "runtime.log").write_text(
        "Traceback\nNameError: name 'flash_attn_func' is not defined\n"
        "Traceback\nNameError: name 'flash_attn_func' is not defined\n",
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="repeated exceptions"):
        _review(tmp_path, candidate)


def test_review_ignores_stale_iteration_error_when_current_output_has_log(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    (candidate["logs_dir"] / "old_failure.log").write_text(
        "NameError: old failure\nNameError: old failure\n",
        encoding="utf-8",
    )

    approval = _review(tmp_path, candidate)

    assert_code_submission_review_approved(approval)


def test_review_rejects_persisted_dependency_output(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    dependency = candidate["output_dir"] / ".kagglebot_reference_packages" / "unsloth" / "model.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("pass\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="persisted dependency/cache"):
        _review(tmp_path, candidate)


def test_review_rejects_row_constant_runtime_predictions(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate["submission_path"].unlink()
    submission_path = candidate["output_dir"] / "submission.csv"
    submission_path.write_text(
        "row_id,bird_a,bird_b\nsegment-1,0.2,0.8\nsegment-2,0.2,0.8\nsegment-3,0.2,0.8\n",
        encoding="utf-8",
    )
    candidate["submission_path"] = submission_path

    with pytest.raises(SubmissionValidationError, match="identical predictions"):
        _review(tmp_path, candidate)


def test_review_rejects_identical_multioutput_prediction_heads(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate["submission_path"].unlink()
    submission_path = candidate["output_dir"] / "submission.csv"
    submission_path.write_text(
        "row_id,class_a,class_b,class_c\nsegment-1,0.1,0.1,0.1\nsegment-2,0.4,0.4,0.4\nsegment-3,0.8,0.8,0.8\n",
        encoding="utf-8",
    )
    candidate["submission_path"] = submission_path

    with pytest.raises(SubmissionValidationError, match="exact copies"):
        _review(tmp_path, candidate)


def test_review_rejects_metrics_declared_fallback_submission(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    metrics = json.loads(candidate["metrics_path"].read_text(encoding="utf-8"))
    metrics["fallback_only"] = True
    candidate["metrics_path"].write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="fallback/dummy output"):
        _review(tmp_path, candidate)


def test_review_revalidation_rejects_artifact_tampering(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    approval = _review(tmp_path, candidate)
    candidate["submission_path"].write_text("{}", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="changed or disappeared"):
        assert_code_submission_review_approved(approval)


def test_review_fails_closed_when_codex_is_unavailable(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)

    with pytest.raises(SubmissionValidationError, match="did not complete"):
        review_code_submission_before_execute(
            slug="demo",
            run_id="run-1",
            iteration=1,
            kernel_id="user/kernel",
            kernel_version="2",
            package_dir=candidate["package_dir"],
            output_dir=candidate["output_dir"],
            runtime_logs_dir=candidate["logs_dir"],
            submission_path=candidate["submission_path"],
            metrics_path=candidate["metrics_path"],
            expected_output_file="submission.json",
            message="human-readable submission",
            review_dir=tmp_path / "review",
            run_codex_func=lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("codex")),
        )


def test_execution_guard_rechecks_known_daily_quota(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    approval = _review(tmp_path, candidate)

    with pytest.raises(SubmissionRateLimitError, match="daily submission limit reached"):
        recheck_code_submission_execution_guard(
            approval=approval,
            slug="demo",
            kernel_id="user/kernel",
            kernel_version="2",
            expected_output_file="submission.json",
            submission_path=candidate["submission_path"],
            message="human-readable submission",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            submission_limit_per_day=1,
            fetch_submission_rows=lambda _slug: [{"date": "2026-07-15 00:00:00"}],
            force_submit=False,
            now=datetime(2026, 7, 15, 1, tzinfo=UTC),
        )
