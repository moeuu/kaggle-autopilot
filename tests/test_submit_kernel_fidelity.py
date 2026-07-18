from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError, SubmissionCliError, SubmissionValidationError
from kagglebot.kernel_runtime.submit_runtime_fidelity import record_runtime_fidelity
from kagglebot.submit_kernel_fidelity import (
    build_submit_runtime_env,
    load_expected_submit_metrics_snapshot,
    stage_submit_fidelity_expected_contract,
    validate_reference_submission_readiness,
    validate_submit_kernel_runtime_fidelity,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_metrics_snapshot_uses_matching_detailed_local_metrics_and_ignores_stale_remote(tmp_path: Path) -> None:
    canonical = _write_json(
        tmp_path / "metrics.json",
        {"metric": "accuracy", "direction": "maximize", "offline_value": 86.5},
    )
    local = _write_json(
        tmp_path / "local" / "metrics.json",
        {"offline_value": 86.5, "chosen_pipeline": "qwen", "reference_path_used": True},
    )
    stale_remote = _write_json(
        tmp_path / "remote" / "metrics.json",
        {"offline_value": 0.46, "chosen_pipeline": "simple_baseline"},
    )

    payload = load_expected_submit_metrics_snapshot([canonical, local, stale_remote])

    assert payload is not None
    assert payload["chosen_pipeline"] == "qwen"
    assert payload["reference_path_used"] is True


def test_build_submit_runtime_env_carries_selected_candidate_contract() -> None:
    assert build_submit_runtime_env(
        {
            "chosen_pipeline": "qwen",
            "offline_value": 86.5,
            "reference_path_used": True,
            "active_model_source": "owner/qwen-2b/Transformers/default/1",
        }
    ) == {
        "KAGGLEBOT_SELECTED_PIPELINE": "qwen",
        "KAGGLEBOT_SELECTED_OFFLINE_SCORE": "86.5",
        "KAGGLEBOT_REQUIRE_REFERENCE_PATH": "1",
        "KAGGLEBOT_SELECTED_MODEL_SOURCE": "owner/qwen-2b/Transformers/default/1",
    }


def test_build_submit_runtime_env_accepts_selected_profile_alias() -> None:
    runtime_env = build_submit_runtime_env(
        {
            "selected_profile": "adaptive_profile",
            "offline_value": 119.0,
        }
    )

    assert runtime_env["KAGGLEBOT_SELECTED_PIPELINE"] == "adaptive_profile"


def test_runtime_fidelity_rejects_remote_pipeline_and_dependency_degradation(tmp_path: Path) -> None:
    actual = _write_json(
        tmp_path / "metrics.json",
        {
            "metric": "accuracy",
            "direction": "maximize",
            "offline_value": 0.46,
            "chosen_pipeline": "simple_baseline",
            "reference_path_used": False,
            "missing_kernel_sources": ["owner/unsloth-patch"],
        },
    )

    with pytest.raises(SubmissionCliError, match="does not match the selected candidate") as exc:
        validate_submit_kernel_runtime_fidelity(
            artifact_mode="inference",
            expected_metrics={
                "metric": "accuracy",
                "direction": "maximize",
                "offline_value": 86.5,
                "chosen_pipeline": "qwen",
                "reference_path_used": True,
                "missing_kernel_sources": [],
            },
            actual_metrics_path=actual,
        )

    assert "pipeline changed" in exc.value.stderr
    assert "reference execution path was not used" in exc.value.stderr
    assert "missing_kernel_sources added" in exc.value.stderr
    assert "score regressed from 86.5 to 0.46" in exc.value.stderr


def test_reference_reproduction_gate_protects_submission_slot(tmp_path: Path) -> None:
    report = _write_json(
        tmp_path / "reference_reproduction_report.json",
        {
            "status": "blocked",
            "blocks_novelty": True,
            "gate_reason": "reference_reproduction_below_campaign_baseline",
            "campaign_baseline_score": 0.86,
            "candidate_score": 0.000002714,
        },
    )

    with pytest.raises(KernelFailedError, match="required public reference has not been reproduced"):
        validate_reference_submission_readiness(
            reproduction_report_path=report,
            expected_metrics={"score": 0.000002714},
        )


def test_reference_reproduction_gate_allows_reproduced_candidate(tmp_path: Path) -> None:
    report = _write_json(
        tmp_path / "reference_reproduction_report.json",
        {
            "status": "blocked",
            "blocks_novelty": True,
            "campaign_baseline_score": 0.86,
        },
    )

    validate_reference_submission_readiness(
        reproduction_report_path=report,
        expected_metrics={"score": 0.87},
    )


def test_runtime_fidelity_accepts_matching_remote_inference(tmp_path: Path) -> None:
    actual = _write_json(
        tmp_path / "metrics.json",
        {
            "metric": "accuracy",
            "direction": "maximize",
            "offline_value": 85.0,
            "chosen_pipeline": "qwen",
            "reference_path_used": True,
            "missing_kernel_sources": [],
        },
    )

    validate_submit_kernel_runtime_fidelity(
        artifact_mode="gateway",
        expected_metrics={
            "metric": "accuracy",
            "direction": "maximize",
            "offline_value": 86.5,
            "chosen_pipeline": "qwen",
            "reference_path_used": True,
            "missing_kernel_sources": [],
        },
        actual_metrics_path=actual,
    )


def test_runtime_fidelity_accepts_generated_kernel_metric_name_alias(tmp_path: Path) -> None:
    actual = _write_json(
        tmp_path / "metrics.json",
        {
            "metric_name": "adjusted_edge_jaccard + 0.1 * division_jaccard",
            "direction": "maximize",
            "score": 0.897,
            "selected_pipeline": "lb897_reference_reproduction",
        },
    )

    validate_submit_kernel_runtime_fidelity(
        artifact_mode="inference",
        expected_metrics={
            "metric": "adjusted_edge_jaccard + 0.1 * division_jaccard",
            "direction": "maximize",
            "score": 0.897,
            "selected_pipeline": "lb897_reference_reproduction",
        },
        actual_metrics_path=actual,
    )


@pytest.mark.parametrize("actual_metrics_path", [None, Path("missing.json")])
def test_runtime_fidelity_rejects_missing_remote_metrics(
    tmp_path: Path,
    actual_metrics_path: Path | None,
) -> None:
    resolved_path = tmp_path / actual_metrics_path if actual_metrics_path is not None else None

    with pytest.raises(SubmissionCliError, match="does not match the selected candidate") as exc:
        validate_submit_kernel_runtime_fidelity(
            artifact_mode="gateway",
            expected_metrics={"metric": "accuracy", "offline_value": 0.9},
            actual_metrics_path=resolved_path,
        )

    assert "runtime fidelity" in exc.value.stderr


def test_runtime_fidelity_rejects_unreported_selection_contract(tmp_path: Path) -> None:
    actual = _write_json(tmp_path / "metrics.json", {"direction": "maximize"})

    with pytest.raises(SubmissionCliError) as exc:
        validate_submit_kernel_runtime_fidelity(
            artifact_mode="inference",
            expected_metrics={
                "metric": "accuracy",
                "direction": "maximize",
                "offline_value": 0.9,
                "chosen_pipeline": "qwen",
                "active_model_source": "owner/qwen/Transformers/default/1",
            },
            actual_metrics_path=actual,
        )

    assert "selected pipeline 'qwen' was not reported" in exc.value.stderr
    assert "selected metric 'accuracy' was not reported" in exc.value.stderr
    assert "selected active model" in exc.value.stderr
    assert "selected score 0.9 was not reported" in exc.value.stderr


def _strong_fidelity_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_mode: str = "inference",
) -> dict[str, object]:
    package_dir = tmp_path / "package"
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    package_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "demo").mkdir(parents=True)
    (package_dir / "kernel.py").write_text("print('inference')\n", encoding="utf-8")
    (package_dir / "kernel-metadata.json").write_text(
        json.dumps({"model_sources": ["owner/model/1"], "competition_sources": ["demo"]}),
        encoding="utf-8",
    )
    local_candidate = tmp_path / "local-candidate.csv"
    local_candidate.write_text("id,target\n1,0.1\n2,0.9\n", encoding="utf-8")
    metrics = {
        "chosen_pipeline": "model",
        "metric": "accuracy",
        "direction": "maximize",
        "score_source": "cv",
        "score": 0.8,
        "active_model_source": "owner/model/1",
        "submission_output_file": "submission.csv",
        "test_prediction_distribution": {"source_top10": [["model_inference", 2]]},
    }
    expected_path = stage_submit_fidelity_expected_contract(
        package_dir=package_dir,
        slug="demo",
        run_id="run-1",
        iteration=1,
        kernel_id="user/demo",
        artifact_mode=artifact_mode,
        expected_output_file="submission.csv",
        expected_metrics=metrics,
        selected_candidate_path=local_candidate,
        requested_accelerator="gpu",
        executed_accelerator="gpu",
        machine_shape="NvidiaTeslaT4",
        capacity_fallback_used=False,
    )
    submission_path = output_dir / "submission.csv"
    submission_path.write_text("id,target\n2,0.8\n1,0.2\n", encoding="utf-8")
    metrics_path = _write_json(output_dir / "metrics.json", metrics)
    (input_dir / "demo" / "test.csv").write_text("id\n2\n1\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_REQUESTED_ACCELERATOR", "gpu")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_EXECUTED_ACCELERATOR", "gpu")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_MACHINE_SHAPE", "NvidiaTeslaT4")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_CAPACITY_FALLBACK_USED", "0")
    record_runtime_fidelity(package_root=package_dir, output_root=output_dir, input_root=input_dir)
    return {
        "package_dir": package_dir,
        "output_dir": output_dir,
        "expected_path": expected_path,
        "runtime_path": output_dir / "submit_fidelity_runtime.json",
        "submission_path": submission_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def test_normalized_runtime_fidelity_report_passes_and_binds_exact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _strong_fidelity_candidate(tmp_path, monkeypatch)
    report_path = tmp_path / "logs" / "submission_fidelity_report-v3.json"

    report = validate_submit_kernel_runtime_fidelity(
        artifact_mode="inference",
        expected_metrics=candidate["metrics"],
        actual_metrics_path=candidate["metrics_path"],
        expected_contract_path=candidate["expected_path"],
        runtime_fidelity_path=candidate["runtime_path"],
        submission_path=candidate["submission_path"],
        package_dir=candidate["package_dir"],
        report_path=report_path,
        kernel_id="user/demo",
        kernel_version="3",
        run_id="run-1",
        iteration=1,
    )

    assert report is not None
    assert report["verdict"] == "pass"
    assert report["reason_codes"] == []
    assert report["selected_output"]["sha256"]
    assert (
        report["comparison"]["expected"]["package_fingerprint"] == report["comparison"]["actual"]["package_fingerprint"]
    )
    assert str(candidate["runtime_path"].resolve()) in report["supporting_artifact_paths"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["attempt_fingerprint"]


def test_gateway_runtime_fidelity_accepts_visible_constant_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _strong_fidelity_candidate(tmp_path, monkeypatch, artifact_mode="gateway")
    candidate["submission_path"].write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    record_runtime_fidelity(
        package_root=candidate["package_dir"],
        output_root=candidate["output_dir"],
        input_root=tmp_path / "input",
    )

    report = validate_submit_kernel_runtime_fidelity(
        artifact_mode="gateway",
        expected_metrics=candidate["metrics"],
        actual_metrics_path=candidate["metrics_path"],
        expected_contract_path=candidate["expected_path"],
        runtime_fidelity_path=candidate["runtime_path"],
        submission_path=candidate["submission_path"],
        package_dir=candidate["package_dir"],
        kernel_id="user/demo",
        kernel_version="3",
        run_id="run-1",
        iteration=1,
    )

    assert report is not None
    assert report["verdict"] == "pass"


def test_normalized_runtime_fidelity_failure_is_non_retryable_and_records_stable_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _strong_fidelity_candidate(tmp_path, monkeypatch)
    runtime = json.loads(candidate["runtime_path"].read_text(encoding="utf-8"))
    runtime["errors"] = {
        "unhandled_exception": None,
        "traceback_count": 0,
        "transcripts": [
            {
                "relative_path": "kernel_error.txt",
                "size": 20,
                "sha256": "abc",
                "traceback_count": 2,
                "nonempty": True,
            }
        ],
    }
    candidate["runtime_path"].write_text(json.dumps(runtime), encoding="utf-8")
    report_path = tmp_path / "report.json"

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submit_kernel_runtime_fidelity(
            artifact_mode="inference",
            expected_metrics=candidate["metrics"],
            actual_metrics_path=candidate["metrics_path"],
            expected_contract_path=candidate["expected_path"],
            runtime_fidelity_path=candidate["runtime_path"],
            submission_path=candidate["submission_path"],
            package_dir=candidate["package_dir"],
            report_path=report_path,
            kernel_id="user/demo",
            kernel_version="3",
            run_id="run-1",
            iteration=1,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "fail"
    assert "runtime_error_transcript_present" in report["reason_codes"]
    assert "runtime_error_evidence_contradictory" in report["reason_codes"]
    assert exc.value.reason_codes == report["reason_codes"]
    assert exc.value.report_path == report_path
    assert "package_fingerprint=" in str(exc.value)
    assert "selected_output_sha256=" in str(exc.value)


def test_normalized_fidelity_detects_platform_error_transcript_created_after_recorder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _strong_fidelity_candidate(tmp_path, monkeypatch)
    (candidate["output_dir"] / "kernel_error.txt").write_text(
        "Traceback (most recent call last):\nRuntimeError: late platform transcript\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(SubmissionValidationError):
        validate_submit_kernel_runtime_fidelity(
            artifact_mode="inference",
            expected_metrics=candidate["metrics"],
            actual_metrics_path=candidate["metrics_path"],
            expected_contract_path=candidate["expected_path"],
            runtime_fidelity_path=candidate["runtime_path"],
            submission_path=candidate["submission_path"],
            package_dir=candidate["package_dir"],
            report_path=report_path,
            kernel_id="user/demo",
            kernel_version="3",
            run_id="run-1",
            iteration=1,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "runtime_error_transcript_present" in report["reason_codes"]
    assert "runtime_error_evidence_incomplete" in report["reason_codes"]
    assert str((candidate["output_dir"] / "kernel_error.txt").resolve()) in report["supporting_artifact_paths"]


def test_unchanged_failed_fidelity_attempt_requires_changed_contract_package_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _strong_fidelity_candidate(tmp_path, monkeypatch)
    report_path = tmp_path / "report.json"
    runtime = json.loads(candidate["runtime_path"].read_text(encoding="utf-8"))
    runtime["expected_contract"]["sha256"] = "mutated"
    candidate["runtime_path"].write_text(json.dumps(runtime), encoding="utf-8")
    kwargs = {
        "artifact_mode": "inference",
        "expected_metrics": candidate["metrics"],
        "actual_metrics_path": candidate["metrics_path"],
        "expected_contract_path": candidate["expected_path"],
        "runtime_fidelity_path": candidate["runtime_path"],
        "submission_path": candidate["submission_path"],
        "package_dir": candidate["package_dir"],
        "report_path": report_path,
        "kernel_id": "user/demo",
        "kernel_version": "3",
        "run_id": "run-1",
        "iteration": 1,
    }
    with pytest.raises(SubmissionValidationError):
        validate_submit_kernel_runtime_fidelity(**kwargs)

    runtime["expected_contract"]["sha256"] = json.loads(report_path.read_text(encoding="utf-8"))[
        "expected_contract_sha256"
    ]
    candidate["runtime_path"].write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(SubmissionValidationError):
        validate_submit_kernel_runtime_fidelity(**kwargs)

    assert "unchanged_failed_fidelity_attempt" in json.loads(report_path.read_text(encoding="utf-8"))["reason_codes"]
