from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

from kagglebot.autopilot_state import (
    ResumeRunResolutionError,
    _load_submitted_iteration_tracking_score,
    apply_final_run_status,
    apply_run_status,
    build_run_payload,
    build_run_summary_payload,
    copy_kernel_support_artifacts_to_iteration_dir,
    copy_submission_artifact_to_iteration_dir,
    find_latest_run_id,
    list_run_ids,
    load_run_state,
    resolve_iteration_artifact,
    resolve_resume_run_id,
    resume_best_submittable_iteration_state,
    resume_best_submitted_offline_score,
    save_run_state,
    write_iteration_state_marker,
    write_run_payload,
)
from kagglebot.paths import CompetitionPaths
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

    payload = build_run_payload(run_id="run-1", config=config, resolved=resolved, status="running")

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

    write_run_payload(run_dir, {"run_id": "run-1", "status": "running"})

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload == {"run_id": "run-1", "status": "running"}


def test_resolve_iteration_artifact_finds_same_stem_tabular_suffix(tmp_path: Path) -> None:
    iter_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1"
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True)
    oof_path = output_dir / "oof_predictions.jsonl"
    oof_path.write_text('{"y":0,"oof_pred":0.1}\n', encoding="utf-8")

    assert resolve_iteration_artifact(iter_dir, "oof_predictions.csv") == oof_path


def test_resolve_iteration_artifact_finds_compressed_same_stem_tabular_suffix(tmp_path: Path) -> None:
    iter_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1"
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True)
    oof_path = output_dir / "oof_predictions.csv.gz"
    with gzip.open(oof_path, "wt", encoding="utf-8") as handle:
        handle.write("y,oof_pred\n0,0.1\n")

    assert resolve_iteration_artifact(iter_dir, "oof_predictions.csv") == oof_path


def test_resolve_iteration_artifact_finds_excel_same_stem_tabular_suffix(tmp_path: Path) -> None:
    iter_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1"
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True)
    oof_path = output_dir / "oof_predictions.xlsx"
    oof_path.write_bytes(b"excel-bytes")

    assert resolve_iteration_artifact(iter_dir, "oof_predictions.csv") == oof_path


def test_copy_kernel_support_artifacts_preserves_non_csv_tabular_suffix(tmp_path: Path) -> None:
    kernel_output_dir = tmp_path / "kernel-output"
    nested_dir = kernel_output_dir / "nested"
    iter_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1"
    nested_dir.mkdir(parents=True)
    oof_path = nested_dir / "oof_predictions.jsonl.gz"
    with gzip.open(oof_path, "wt", encoding="utf-8") as handle:
        handle.write('{"row_id":1,"oof_pred":0.1}\n')
    feature_path = kernel_output_dir / "feature_suspects.xlsx"
    feature_path.write_bytes(b"excel-bytes")
    split_path = kernel_output_dir / "split_diagnostics.json"
    split_path.write_text('{"folds": 3}\n', encoding="utf-8")

    copy_kernel_support_artifacts_to_iteration_dir(kernel_output_dir=kernel_output_dir, iter_dir=iter_dir)

    assert (iter_dir / "output" / "oof_predictions.jsonl.gz").is_file()
    assert (iter_dir / "output" / "feature_suspects.xlsx").read_bytes() == b"excel-bytes"
    assert (iter_dir / "output" / "split_diagnostics.json").read_text(encoding="utf-8") == '{"folds": 3}\n'


def test_copy_submission_artifact_to_iteration_dir_preserves_manifest_reference(tmp_path: Path) -> None:
    output_dir = tmp_path / "kernel-output"
    iter_dir = tmp_path / "iter-1"
    output_dir.mkdir()
    submission = output_dir / "answers.nii.gz"
    submission.write_bytes(b"volume")
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    copied = copy_submission_artifact_to_iteration_dir(source=manifest, iter_dir=iter_dir)

    assert copied == iter_dir / "submission_manifest.json"
    assert (iter_dir / "answers.nii.gz").read_bytes() == b"volume"
    copied_manifest = json.loads(copied.read_text(encoding="utf-8"))
    assert copied_manifest["submission_path"] == "answers.nii.gz"


def test_copy_submission_artifact_to_iteration_dir_preserves_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "kernel-output"
    iter_dir = tmp_path / "iter-1"
    output_dir.mkdir()
    source = output_dir / "submission.shp"
    source.write_bytes(b"shape")
    (output_dir / "submission.dbf").write_bytes(b"attributes")
    (output_dir / "submission.shx").write_bytes(b"index")
    (output_dir / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")

    copied = copy_submission_artifact_to_iteration_dir(source=source, iter_dir=iter_dir)

    assert copied == iter_dir / "submission.shp"
    assert (iter_dir / "submission.dbf").read_bytes() == b"attributes"
    assert (iter_dir / "submission.shx").read_bytes() == b"index"
    assert (iter_dir / "submission.prj").read_text(encoding="utf-8") == "EPSG:4326\n"


def test_apply_run_status_sets_status_and_optional_stop_reason() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}

    returned = apply_run_status(payload, status="stopped", stop_reason="max_total_min reached")

    assert returned is payload
    assert payload["status"] == "stopped"
    assert payload["stop_reason"] == "max_total_min reached"


def test_apply_run_status_omits_empty_stop_reason() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}

    apply_run_status(payload, status="completed", stop_reason="")

    assert payload == {"run_id": "run-1", "status": "completed"}


def test_apply_final_run_status_marks_submitted_when_submission_result_exists() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}

    apply_final_run_status(
        payload,
        submitted=True,
        has_submission_result=True,
        writeup_mode=False,
        writeup_bundle_meta=None,
    )

    assert payload["status"] == "submitted"


def test_apply_final_run_status_marks_validated_writeup_ready() -> None:
    payload: dict[str, object] = {"run_id": "run-1", "status": "running"}
    bundle = {"path": "writeup.md"}

    apply_final_run_status(
        payload,
        submitted=False,
        has_submission_result=False,
        writeup_mode=True,
        writeup_bundle_meta=bundle,
    )

    assert payload["status"] == "writeup_ready"
    assert payload["writeup_bundle"] == bundle


def test_apply_final_run_status_preserves_terminal_failure_states() -> None:
    interrupted: dict[str, object] = {"status": "interrupted"}
    submit_failed: dict[str, object] = {"status": "submit_failed"}

    for payload in (interrupted, submit_failed):
        apply_final_run_status(
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

    apply_final_run_status(
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

    payload = build_run_summary_payload(
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

    missing = load_run_state(run_dir)
    assert missing == {"submit_attempted": False, "submit_ok": False}

    state_path = run_dir / "run_state.json"
    state_path.write_text("{", encoding="utf-8")
    invalid = load_run_state(run_dir)
    assert invalid == {"submit_attempted": False, "submit_ok": False}

    state_path.write_text("[]", encoding="utf-8")
    non_object = load_run_state(run_dir)
    assert non_object == {"submit_attempted": False, "submit_ok": False}


def test_save_run_state_merges_existing_state_and_writes_json_object(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_state.json").write_text(
        json.dumps({"submit_attempted": True, "last_fingerprint": "abc"}),
        encoding="utf-8",
    )

    save_run_state(run_dir, {"last_reason": "validation_failed"})

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

    write_iteration_state_marker(
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


def test_resolve_resume_run_id_accepts_exact_and_unique_prefix(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_id = "20260216T000000Z-abcd1234"
    paths.run_dir(run_id).mkdir(parents=True)

    assert resolve_resume_run_id(paths=paths, resume_run_id=run_id, resume_latest=False) == run_id
    assert resolve_resume_run_id(paths=paths, resume_run_id="20260216T000000Z-abcd", resume_latest=False) == run_id


def test_resolve_resume_run_id_selects_latest_by_mtime(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    older = paths.run_dir("run-old")
    newer = paths.run_dir("run-new")
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older_file = older / "not-a-run-dir.txt"
    older_file.write_text("ignored", encoding="utf-8")
    (paths.runs_dir / "not-a-dir").write_text("ignored", encoding="utf-8")
    older_mtime = 1
    newer_mtime = 2
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    assert sorted(list_run_ids(paths)) == ["run-new", "run-old"]
    assert find_latest_run_id(paths) == "run-new"
    assert resolve_resume_run_id(paths=paths, resume_run_id=None, resume_latest=True) == "run-new"


def test_resolve_resume_run_id_reports_ambiguous_prefix(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.run_dir("20260216T000000Z-abcd1111").mkdir(parents=True)
    paths.run_dir("20260216T000000Z-abcd2222").mkdir(parents=True)

    try:
        resolve_resume_run_id(paths=paths, resume_run_id="20260216T000000Z-abcd", resume_latest=False)
    except ResumeRunResolutionError as exc:
        assert exc.param_hint == "--resume-run-id"
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("expected ambiguous prefix to raise")


def test_resolve_resume_run_id_rejects_conflicting_flags(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")

    try:
        resolve_resume_run_id(paths=paths, resume_run_id="run-1", resume_latest=True)
    except ResumeRunResolutionError as exc:
        assert exc.param_hint == "--resume-run-id"
        assert "either --resume-run-id or --resume-latest" in str(exc)
    else:
        raise AssertionError("expected conflicting flags to raise")


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


def test_resume_best_submitted_offline_score_uses_submitted_iteration_marker(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    iter1 = run_dir / "iter-1"
    iter2 = run_dir / "iter-2"
    iter3 = run_dir / "iter-3"
    for iter_dir in (iter1, iter2, iter3):
        iter_dir.mkdir(parents=True)
        (iter_dir / "metrics.json").write_text(json.dumps({"submission_score": 0.5}), encoding="utf-8")
    (iter1 / "iteration_state.json").write_text(json.dumps({"submitted": True}), encoding="utf-8")
    (iter2 / "iteration_state.json").write_text(json.dumps({"submitted": False}), encoding="utf-8")
    (iter3 / "iteration_state.json").write_text(json.dumps({"submitted": True}), encoding="utf-8")
    (iter3 / "metrics.json").write_text(json.dumps({"submission_score": 0.4}), encoding="utf-8")

    score = resume_best_submitted_offline_score(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=2,
        load_kernel_metrics=lambda path, direction, target_metric: None,
    )

    assert score == 0.5


def test_resume_best_submittable_iteration_state_uses_submission_artifact_and_gate(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    iter1 = run_dir / "iter-1"
    iter2 = run_dir / "iter-2"
    for iter_dir in (iter1, iter2):
        iter_dir.mkdir(parents=True)
        (iter_dir / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
        (iter_dir / "metrics.json").write_text("{}", encoding="utf-8")

    def load_kernel_metrics(path: Path, direction: str, target_metric: str) -> EvaluationResult:
        assert direction == "maximize"
        assert target_metric == "auc"
        value = 0.7 if path.parent.name == "iter-1" else 0.9
        return EvaluationResult(
            score_source="cv",
            metric="auc",
            direction="maximize",
            value=value,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )

    score, submission = resume_best_submittable_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="maximize",
        target_metric="auc",
        max_iterations=2,
        load_kernel_metrics=load_kernel_metrics,
        iteration_metrics_allow_submit=lambda path, evaluation: path.parent.name == "iter-1",
    )

    assert score == 0.7
    assert submission == iter1 / "submission.csv"
