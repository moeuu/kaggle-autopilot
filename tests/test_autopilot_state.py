from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kagglebot.autopilot_state import (
    _apply_final_run_status,
    _apply_run_status,
    _build_run_payload,
    _build_run_summary_payload,
    _load_run_state,
    _load_submitted_iteration_tracking_score,
    _save_run_state,
    _write_iteration_state_marker,
    _write_run_payload,
)
from kagglebot.solver.evaluate import EvaluationResult


def test_build_run_payload_records_config_and_resolved_state() -> None:
    config = SimpleNamespace(
        slug="demo",
        agent="codex",
        compute="local_gpu",
        accelerator="gpu",
        method_scout="auto",
        method_scout_max_sources=12,
        candidate_budget_min=20,
        max_candidates_per_iteration=3,
        kaggle_username="user",
        submit=True,
        message="submit message",
    )
    resolved = {
        "deliverable_mode": "leaderboard",
        "submit_mode": "file",
        "target_metric": "rmse",
        "target_score": 0.5,
        "target_direction": "minimize",
        "max_iterations": 5,
        "evaluation_contract": {"metric_name": "rmse"},
    }

    payload = _build_run_payload(run_id="run-1", config=config, resolved=resolved, status="running")

    assert payload["run_id"] == "run-1"
    assert payload["slug"] == "demo"
    assert payload["status"] == "running"
    assert payload["started_at"]
    assert payload["config"]["compute"] == "local_gpu"
    assert payload["config"]["target_metric"] == "rmse"
    assert payload["config"]["target_score"] == 0.5
    assert payload["config"]["submit"] is True
    assert payload["config"]["evaluation_contract"] == {"metric_name": "rmse"}


def test_write_run_payload_writes_run_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_run_payload(run_dir, {"run_id": "run-1", "status": "running"})

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload == {"run_id": "run-1", "status": "running"}


def test_apply_run_status_sets_status_and_optional_stop_reason() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}

    returned = _apply_run_status(payload, status="stopped", stop_reason="max_total_min reached")

    assert returned is payload
    assert payload["status"] == "stopped"
    assert payload["stop_reason"] == "max_total_min reached"


def test_apply_run_status_omits_empty_stop_reason() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}

    _apply_run_status(payload, status="completed", stop_reason="")

    assert payload == {"run_id": "run-1", "status": "completed"}


def test_apply_final_run_status_marks_submitted_when_submission_result_exists() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}

    _apply_final_run_status(
        payload,
        submitted=True,
        has_submission_result=True,
        writeup_mode=False,
        writeup_bundle_meta=None,
    )

    assert payload["status"] == "submitted"


def test_apply_final_run_status_marks_manual_finalization_for_writeup_bundle() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}
    bundle = {"path": "writeup.md"}

    _apply_final_run_status(
        payload,
        submitted=False,
        has_submission_result=False,
        writeup_mode=True,
        writeup_bundle_meta=bundle,
    )

    assert payload["status"] == "manual_finalization_required"
    assert payload["writeup_bundle"] == bundle


def test_apply_final_run_status_preserves_terminal_failure_states() -> None:
    interrupted: dict[str, object] = {"status": "interrupted"}
    submit_failed: dict[str, object] = {"status": "submit_failed"}

    for payload in (interrupted, submit_failed):
        _apply_final_run_status(
            payload,
            submitted=False,
            has_submission_result=False,
            writeup_mode=False,
            writeup_bundle_meta=None,
        )

    assert interrupted["status"] == "interrupted"
    assert submit_failed["status"] == "submit_failed"


def test_apply_final_run_status_defaults_to_completed() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}

    _apply_final_run_status(
        payload,
        submitted=False,
        has_submission_result=False,
        writeup_mode=False,
        writeup_bundle_meta=None,
    )

    assert payload["status"] == "completed"


def test_build_run_summary_payload_stringifies_paths(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    faithful = tmp_path / "faithful.csv"
    high_potential = tmp_path / "high.csv"

    payload = _build_run_summary_payload(
        best_score=0.9,
        best_submission=trusted,
        best_submittable_score=0.8,
        best_submittable_submission=faithful,
        best_high_potential_score=0.95,
        best_high_potential_submission=high_potential,
        best_high_potential_iteration=3,
        best_high_potential_meta={"trusted": False},
        fallback_submit_blocked_reason="higher_potential_unsubmitted_candidate_exists",
    )

    assert payload == {
        "best_trusted_score": 0.9,
        "best_trusted_submission": str(trusted),
        "best_competition_faithful_score": 0.8,
        "best_competition_faithful_submission": str(faithful),
        "best_high_potential_score": 0.95,
        "best_high_potential_submission": str(high_potential),
        "best_high_potential_iteration": 3,
        "best_high_potential_meta": {"trusted": False},
        "fallback_submit_blocked_reason": "higher_potential_unsubmitted_candidate_exists",
    }


def test_load_run_state_defaults_for_missing_invalid_or_non_object_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    missing = _load_run_state(run_dir)
    assert missing == {"submit_attempted": False, "submit_ok": False}

    state_path = run_dir / "run_state.json"
    state_path.write_text("{", encoding="utf-8")
    invalid = _load_run_state(run_dir)
    assert invalid == {"submit_attempted": False, "submit_ok": False}

    state_path.write_text("[]", encoding="utf-8")
    non_object = _load_run_state(run_dir)
    assert non_object == {"submit_attempted": False, "submit_ok": False}


def test_save_run_state_merges_existing_state_and_writes_json_object(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_state.json").write_text(
        json.dumps({"submit_attempted": True, "last_fingerprint": "abc"}),
        encoding="utf-8",
    )

    _save_run_state(run_dir, {"last_reason": "validation_failed"})

    payload = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert payload["submit_attempted"] is True
    assert payload["submit_ok"] is False
    assert payload["last_reason"] == "validation_failed"
    assert payload["last_submit_fingerprint"] == "abc"
    assert payload["last_fingerprint"] == "abc"
    assert isinstance(payload["updated_at"], str)


def test_write_iteration_state_marker_writes_json_object(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    iter_dir.mkdir()
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0\n", encoding="utf-8")
    metrics_path = iter_dir / "metrics.json"
    evaluation_report_path = iter_dir / "evaluation_report.json"

    _write_iteration_state_marker(
        iter_dir=iter_dir,
        run_id="run-1",
        iteration=1,
        submission_path=submission_path,
        metrics_path=metrics_path,
        evaluation_report_path=evaluation_report_path,
        submit_phase_required=True,
        submit_allowed_by_gate=False,
        submit_phase_state="not_submitted",
        submitted=False,
        readiness_score=0.42,
    )

    payload = json.loads((iter_dir / "iteration_state.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["iteration"] == 1
    assert payload["submission_exists"] is True
    assert payload["submit_phase_finished"] is True
    assert payload["readiness_score"] == 0.42


def test_load_submitted_iteration_tracking_score_ignores_non_finite_submission_score(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"submission_score": "inf"}), encoding="utf-8")

    def load_kernel_metrics(path: Path, direction: str, target_metric: str) -> EvaluationResult:
        assert path == metrics_path
        assert direction == "minimize"
        assert target_metric == "rmse"
        return EvaluationResult(
            score_source="cv",
            metric="rmse",
            direction="minimize",
            value=0.37,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )

    score = _load_submitted_iteration_tracking_score(
        metrics_path=metrics_path,
        metric_direction="minimize",
        target_metric="rmse",
        load_kernel_metrics=load_kernel_metrics,
    )

    assert score == 0.37
