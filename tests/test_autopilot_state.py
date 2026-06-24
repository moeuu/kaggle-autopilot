from __future__ import annotations

import json
from pathlib import Path

from kagglebot.autopilot_state import _load_run_state, _save_run_state, _write_iteration_state_marker


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
