from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError, SubmissionCliError
from kagglebot.submit_kernel_fidelity import (
    build_submit_runtime_env,
    load_expected_submit_metrics_snapshot,
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
