from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kagglebot.autopilot import (
    _add_kernel_score_provenance,
    _evaluation_spec_conflict_warnings,
    _load_contract_aware_kernel_metrics,
    _resolve_authoritative_evaluation_contract,
    _resolve_local_score,
    _resolve_unscored_diagnostic_kernel_result,
)
from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_runner import KernelRunResult
from kagglebot.training_route import TrainingRouteDecision

EXPECTED_METRIC = (
    "mean paired lift (with-skills minus without-skills) with safety fail-fast; "
    "final ranking also weights compliance/safety, generalization, and writeup quality"
)


@pytest.fixture
def training_route_decision() -> TrainingRouteDecision:
    return TrainingRouteDecision(
        skip_local_training=True,
        direct_notebook=False,
        reason="validated_non_training_path_preferred_over_very_heavy_local_training",
        mode="optimization",
        validation_mode="offline",
    )


@pytest.fixture
def authoritative_contract() -> dict[str, object]:
    return {
        "expected_metric": EXPECTED_METRIC,
        "expected_direction": "maximize",
        "expected_split_strategy": "kfold",
        "accepted_score_sources": ["cv", "holdout"],
    }


def _write_kernel_result(
    tmp_path: Path,
    *,
    metric_overrides: dict[str, object] | None = None,
    route_overrides: dict[str, object] | None = None,
    write_archive: bool = True,
    write_failure: bool = False,
) -> tuple[KernelRunResult, dict[str, object]]:
    output_dir = tmp_path / "kernel-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "execution_mode": "non_training_submission",
        "training_performed": False,
        "execution_status": "success",
        "non_training_validation_passed": True,
        "non_training_validation_mode": "offline",
        "offline_artifact_validation_passed": True,
        "official_paired_validation_passed": False,
        "submission_ready": False,
        "primary_metric": "mean paired verifier lift with safety fail-fast",
        "primary_score": None,
        "score_source": "holdout",
        "selected_pipeline": "trajectory_optimized_sparse_portfolio",
        "selected_candidate_hash": "candidate-sha256",
        "routing_eligibility": {"candidate": {"routing_objective": 0.91}},
        "remediation": ["Mount the official paired evaluator."],
    }
    metrics.update(metric_overrides or {})
    route_result: dict[str, object] = {
        "mode": "skill_artifact",
        "validation_status": "success",
        "artifact_paths": [str(output_dir / "diagnostic_skill_portfolio.zip")],
        "manifest": {"archive": "diagnostic_skill_portfolio.zip", "submission_ready": False},
    }
    route_result.update(route_overrides or {})
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (output_dir / "route_result.json").write_text(json.dumps(route_result), encoding="utf-8")
    if write_archive:
        (output_dir / "diagnostic_skill_portfolio.zip").write_bytes(b"diagnostic")
    if write_failure:
        (output_dir / "failure.json").write_text(
            json.dumps({"execution_status": "failed", "reason": "kernel execution failed"}),
            encoding="utf-8",
        )
    return (
        KernelRunResult(
            kernel_id="local/demo",
            output_dir=output_dir,
            submission_path=output_dir / "route_result.json",
            metrics_path=metrics_path,
        ),
        metrics,
    )


def _write_trained_artifact_result(
    tmp_path: Path,
    *,
    metric_overrides: dict[str, object] | None = None,
    write_archive: bool = True,
    write_failure: bool = False,
) -> tuple[KernelRunResult, dict[str, object]]:
    archive_payload = b"validated skill artifact"
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    kernel_result, metrics = _write_kernel_result(tmp_path, write_archive=False, write_failure=write_failure)
    metrics.pop("non_training_validation_passed")
    metrics.pop("non_training_validation_mode")
    metrics.update(
        {
            "execution_mode": "train_and_validate",
            "training_performed": True,
            "execution_status": "paired_validation_unavailable",
            "result_kind": "unscored_diagnostic_artifact",
            "score_status": "official_paired_evaluation_unavailable",
            "offline_validation_passed": False,
            "primary_metric_available": False,
            "primary_metric_unavailable_reason": "official paired holdout evidence is unavailable",
            "metric_value": None,
            "primary_metric_value": None,
            "cv_metric_value": 0.7377811499399995,
            "cv_metric_name": "routing_screening_objective",
            "cv_metric_is_primary_competition_metric": False,
            "final_ready": False,
            "validation_status": "paired_validation_unavailable",
            "readiness_blockers": ["official_final_holdout_unavailable"],
            "archive_hash": archive_sha256,
            "asset_hashes": {"archive": archive_sha256},
            "routing_eligibility": {"candidate": {"routing_objective": 0.91}},
        }
    )
    metrics.update(metric_overrides or {})
    output_dir = kernel_result.output_dir
    (output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (output_dir / "route_result.json").write_text(
        json.dumps(
            {
                "mode": "skill_artifact",
                "validation_status": "paired_validation_unavailable",
                "artifact_paths": [str(output_dir / "submission.zip")],
                "manifest": {
                    "archive": "submission.zip",
                    "archive_sha256": archive_sha256,
                    "submission_ready": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "submission_contract.json").write_text(
        json.dumps(
            {
                "archive_name": "submission.zip",
                "archive_sha256": archive_sha256,
                "submission_ready": False,
                "canonical_submission_ready": False,
                "offline_artifact_validation_passed": True,
                "official_paired_validation_passed": False,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "final_artifact_manifest.json").write_text(
        json.dumps(
            {
                "submission_ready": False,
                "deliverables": [
                    {
                        "path": "submission.zip",
                        "sha256": archive_sha256,
                        "size": len(archive_payload),
                        "validation_status": "validated",
                        "intended_for_kaggle_submission": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "submission_validation.json").write_text(
        json.dumps(
            {
                "archive_name": "submission.zip",
                "archive_sha256": archive_sha256,
                "archive_static_valid": True,
                "passed": False,
            }
        ),
        encoding="utf-8",
    )
    if write_archive:
        (output_dir / "submission.zip").write_bytes(archive_payload)
    return (
        KernelRunResult(
            kernel_id=kernel_result.kernel_id,
            output_dir=output_dir,
            submission_path=output_dir / "submission.zip",
            metrics_path=output_dir / "metrics.json",
        ),
        metrics,
    )


def test_finite_primary_score_keeps_the_existing_scored_path(
    tmp_path: Path,
    training_route_decision: TrainingRouteDecision,
    authoritative_contract: dict[str, object],
) -> None:
    kernel_result, metrics = _write_kernel_result(
        tmp_path,
        metric_overrides={
            "primary_score": 0.42,
            "official_paired_validation_passed": True,
            "metric_direction": "maximize",
            "split_strategy": "kfold",
        },
    )

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_route_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
        )
        is None
    )
    evaluation = _load_contract_aware_kernel_metrics(
        metrics_path=kernel_result.metrics_path,
        metrics=metrics,
        direction="maximize",
        target_metric=EXPECTED_METRIC,
        authoritative_contract=authoritative_contract,
    )
    assert evaluation is not None
    assert evaluation.value == pytest.approx(0.42)


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), float("-inf"), "0.42", True, False],
    ids=["nan", "positive-infinity", "negative-infinity", "string", "true", "false"],
)
def test_primary_score_accepts_only_finite_non_boolean_numbers(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
    invalid_score: object,
) -> None:
    kernel_result, metrics = _write_kernel_result(
        tmp_path,
        metric_overrides={
            "primary_score": invalid_score,
            "official_paired_validation_passed": True,
            "metric_direction": "maximize",
            "split_strategy": "kfold",
        },
    )

    assert (
        _load_contract_aware_kernel_metrics(
            metrics_path=kernel_result.metrics_path,
            metrics=metrics,
            direction="maximize",
            target_metric=EXPECTED_METRIC,
            authoritative_contract=authoritative_contract,
        )
        is None
    )


def test_cv_score_is_resolved_without_promoting_nullable_official_scores(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
) -> None:
    metrics = {
        "score_source": "cv",
        "primary_score": None,
        "primary_metric_value": None,
        "cv_score": 0.7377811499399995,
        "cv_metric_name": "routing_screening_objective",
        "cv_metric_is_primary_competition_metric": False,
        "submission_ready": False,
        "official_paired_validation_passed": False,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    score, key = _resolve_local_score(metrics, "cv")
    assert score == pytest.approx(0.7377811499399995)
    assert key == "cv_score"
    assert _resolve_local_score(metrics, "holdout") == (None, None)

    evaluation = _load_contract_aware_kernel_metrics(
        metrics_path=metrics_path,
        metrics=metrics,
        direction="maximize",
        target_metric=EXPECTED_METRIC,
        authoritative_contract=authoritative_contract,
    )

    assert evaluation is not None
    assert evaluation.value == pytest.approx(0.7377811499399995)
    assert evaluation.metric == "routing_screening_objective"
    assert evaluation.score_source == "cv"
    assert metrics["score_key_used"] == "cv_score"
    assert metrics["primary_score"] is None
    assert metrics["primary_metric_value"] is None
    assert metrics["submission_ready"] is False
    assert metrics["official_paired_validation_passed"] is False


@pytest.mark.parametrize("validation_status", ["validation_unavailable", "paired_validation_unavailable"])
def test_declared_unscored_artifact_never_promotes_cv_proxy(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
    validation_status: str,
) -> None:
    metrics = {
        "result_kind": "unscored_diagnostic_artifact",
        "score_status": "official_paired_evaluation_unavailable",
        "score_source": "cv",
        "primary_metric_available": False,
        "metric_value": None,
        "primary_score": None,
        "primary_metric_value": None,
        "cv_metric_value": 0.7377811499399995,
        "cv_metric_name": "routing_screening_objective",
        "cv_metric_is_primary_competition_metric": False,
        "submission_ready": False,
        "final_ready": False,
        "validation_status": validation_status,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    assert (
        _load_contract_aware_kernel_metrics(
            metrics_path=metrics_path,
            metrics=metrics,
            direction="maximize",
            target_metric=EXPECTED_METRIC,
            authoritative_contract=authoritative_contract,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submission_ready", True),
        ("primary_metric_available", True),
        ("final_ready", True),
        ("validation_status", "success"),
        ("cv_metric_value", float("nan")),
        ("cv_metric_value", float("inf")),
    ],
)
def test_inconsistent_declared_unscored_artifact_is_rejected(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
    field: str,
    value: object,
) -> None:
    metrics = {
        "result_kind": "unscored_diagnostic_artifact",
        "score_status": "official_paired_evaluation_unavailable",
        "primary_metric_available": False,
        "metric_value": None,
        "primary_score": None,
        "primary_metric_value": None,
        "cv_metric_value": 0.7377811499399995,
        "cv_metric_is_primary_competition_metric": False,
        "submission_ready": False,
        "final_ready": False,
        "validation_status": "validation_unavailable",
        field: value,
    }
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(KernelFailedError, match="unscored_diagnostic_contract_rejected"):
        _load_contract_aware_kernel_metrics(
            metrics_path=metrics_path,
            metrics=metrics,
            direction="maximize",
            target_metric=EXPECTED_METRIC,
            authoritative_contract=authoritative_contract,
        )


@pytest.mark.parametrize("missing_field", ["result_kind", "score_status"])
def test_incomplete_declared_unscored_artifact_is_rejected(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
    missing_field: str,
) -> None:
    metrics = {
        "result_kind": "unscored_diagnostic_artifact",
        "score_status": "official_paired_evaluation_unavailable",
        "primary_metric_available": False,
        "metric_value": None,
        "primary_score": None,
        "primary_metric_value": None,
        "cv_metric_value": 0.7377811499399995,
        "cv_metric_is_primary_competition_metric": False,
        "submission_ready": False,
        "final_ready": False,
        "validation_status": "validation_unavailable",
    }
    metrics.pop(missing_field)
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(KernelFailedError, match="unscored_diagnostic_contract_rejected"):
        _load_contract_aware_kernel_metrics(
            metrics_path=metrics_path,
            metrics=metrics,
            direction="maximize",
            target_metric=EXPECTED_METRIC,
            authoritative_contract=authoritative_contract,
        )


@pytest.mark.parametrize("invalid_cv_score", [None, "0.7", True, float("nan"), float("inf")])
def test_local_score_resolver_skips_invalid_values(invalid_cv_score: object) -> None:
    score, key = _resolve_local_score(
        {"cv_score": invalid_cv_score, "cv_metric_value": 0.5},
        "cv",
    )

    assert score == pytest.approx(0.5)
    assert key == "cv_metric_value"


def test_kernel_score_provenance_keeps_official_readiness_false() -> None:
    payload: dict[str, object] = {"offline_value": 0.7377811499399995}
    _add_kernel_score_provenance(
        payload,
        {
            "score_key_used": "cv_score",
            "cv_score": 0.7377811499399995,
            "cv_metric_name": "routing_screening_objective",
            "cv_metric_is_primary_competition_metric": False,
            "primary_score": None,
            "primary_metric_value": None,
            "primary_metric_available": False,
            "official_paired_validation_passed": False,
            "submission_ready": False,
        },
    )

    assert payload["score_key_used"] == "cv_score"
    assert payload["cv_metric_is_primary_competition_metric"] is False
    assert payload["primary_score"] is None
    assert payload["primary_metric_value"] is None
    assert payload["official_paired_validation_passed"] is False
    assert payload["submission_ready"] is False


def test_exact_skill_lift_diagnostic_is_validated_but_unscored(
    tmp_path: Path,
    training_route_decision: TrainingRouteDecision,
    authoritative_contract: dict[str, object],
) -> None:
    kernel_result, metrics = _write_kernel_result(tmp_path)

    result = _resolve_unscored_diagnostic_kernel_result(
        kernel_result=kernel_result,
        kernel_metrics=metrics,
        training_route_decision=training_route_decision,
        authoritative_evaluation_contract=authoritative_contract,
        deliverable_mode="writeup",
        submit_mode="file",
    )

    assert result is not None
    assert result.diagnostic_archive_path.name == "diagnostic_skill_portfolio.zip"
    assert metrics["primary_score"] is None
    assert metrics["routing_eligibility"] == {"candidate": {"routing_objective": 0.91}}


def test_trained_skill_lift_submission_archive_is_validated_but_unscored(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
) -> None:
    kernel_result, metrics = _write_trained_artifact_result(tmp_path)
    training_decision = TrainingRouteDecision(
        skip_local_training=False,
        direct_notebook=False,
        reason="local_training_not_explicitly_optional",
    )

    result = _resolve_unscored_diagnostic_kernel_result(
        kernel_result=kernel_result,
        kernel_metrics=metrics,
        training_route_decision=training_decision,
        authoritative_evaluation_contract=authoritative_contract,
        deliverable_mode="writeup",
        submit_mode="file",
    )

    assert result is not None
    assert result.diagnostic_archive_path.name == "submission.zip"
    assert metrics["primary_score"] is None
    assert metrics["routing_eligibility"] == {"candidate": {"routing_objective": 0.91}}


@pytest.mark.parametrize("proxy_key", ["primary_metric_value", "offline_value", "value"])
def test_trained_unscored_artifact_does_not_promote_proxy_score_fields(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
    proxy_key: str,
) -> None:
    kernel_result, metrics = _write_trained_artifact_result(
        tmp_path,
        metric_overrides={proxy_key: 0.91},
    )
    training_decision = TrainingRouteDecision(
        skip_local_training=False,
        direct_notebook=False,
        reason="local_training_not_explicitly_optional",
    )

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
        )
        is None
    )


def test_trained_null_score_is_rejected_for_prediction_route(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
) -> None:
    kernel_result, metrics = _write_trained_artifact_result(tmp_path)
    training_decision = TrainingRouteDecision(
        skip_local_training=False,
        direct_notebook=False,
        reason="local_training_not_explicitly_optional",
    )

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="leaderboard",
            submit_mode="file",
        )
        is None
    )


def test_trained_null_score_is_rejected_for_code_competition(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
) -> None:
    kernel_result, metrics = _write_trained_artifact_result(tmp_path)
    training_decision = TrainingRouteDecision(
        skip_local_training=False,
        direct_notebook=False,
        reason="local_training_not_explicitly_optional",
    )

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
            code_competition=True,
        )
        is None
    )


@pytest.mark.parametrize("failure_mode", ["missing-archive", "failure-json"])
def test_trained_unscored_artifact_requires_archive_and_successful_kernel(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
    failure_mode: str,
) -> None:
    kernel_result, metrics = _write_trained_artifact_result(
        tmp_path,
        write_archive=failure_mode != "missing-archive",
        write_failure=failure_mode == "failure-json",
    )
    training_decision = TrainingRouteDecision(
        skip_local_training=False,
        direct_notebook=False,
        reason="local_training_not_explicitly_optional",
    )

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
        )
        is None
    )


@pytest.mark.parametrize(
    "evidence_failure",
    [
        "missing-manifest",
        "missing-validation",
        "hash-mismatch",
        "metrics-hash-mismatch",
        "route-manifest-hash-missing",
        "contract-hash-missing",
        "static-validation-failed",
    ],
)
def test_trained_unscored_artifact_requires_matching_manifest_and_hash_evidence(
    tmp_path: Path,
    authoritative_contract: dict[str, object],
    evidence_failure: str,
) -> None:
    kernel_result, metrics = _write_trained_artifact_result(tmp_path)
    if evidence_failure == "missing-manifest":
        (kernel_result.output_dir / "final_artifact_manifest.json").unlink()
    elif evidence_failure == "missing-validation":
        (kernel_result.output_dir / "submission_validation.json").unlink()
    elif evidence_failure == "hash-mismatch":
        (kernel_result.output_dir / "submission.zip").write_bytes(b"tampered archive")
    elif evidence_failure == "metrics-hash-mismatch":
        metrics["archive_hash"] = "0" * 64
        (kernel_result.output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    elif evidence_failure == "route-manifest-hash-missing":
        route_path = kernel_result.output_dir / "route_result.json"
        route_result = json.loads(route_path.read_text(encoding="utf-8"))
        route_result["manifest"].pop("archive_sha256")
        route_path.write_text(json.dumps(route_result), encoding="utf-8")
    elif evidence_failure == "contract-hash-missing":
        contract_path = kernel_result.output_dir / "submission_contract.json"
        submission_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        submission_contract.pop("archive_sha256")
        contract_path.write_text(json.dumps(submission_contract), encoding="utf-8")
    else:
        validation_path = kernel_result.output_dir / "submission_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["archive_static_valid"] = False
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
    training_decision = TrainingRouteDecision(
        skip_local_training=False,
        direct_notebook=False,
        reason="local_training_not_explicitly_optional",
    )

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
        )
        is None
    )


@pytest.mark.parametrize(
    "metric_overrides",
    [
        {"offline_artifact_validation_passed": False},
        {"execution_status": "failed"},
        {"submission_ready": True},
        {"official_paired_validation_passed": True},
        {"primary_score": 0.0},
        {"primary_score": float("nan")},
        {"primary_score": float("inf")},
        {"primary_metric": "routing objective"},
        {"primary_metric": None},
        {"score_source": "routing_screen"},
        {"selected_pipeline": None},
    ],
    ids=[
        "offline-validation-failed",
        "execution-failed",
        "submission-ready",
        "official-paired-passed",
        "zero-proxy",
        "nan-proxy",
        "infinite-proxy",
        "routing-metric",
        "missing-primary-metric",
        "routing-score-source",
        "missing-selected-pipeline",
    ],
)
def test_inconsistent_or_proxy_null_score_states_are_rejected(
    tmp_path: Path,
    training_route_decision: TrainingRouteDecision,
    authoritative_contract: dict[str, object],
    metric_overrides: dict[str, object],
) -> None:
    kernel_result, metrics = _write_kernel_result(tmp_path, metric_overrides=metric_overrides)

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_route_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
        )
        is None
    )
    try:
        evaluation = _load_contract_aware_kernel_metrics(
            metrics_path=kernel_result.metrics_path,
            metrics=metrics,
            direction="maximize",
            target_metric=EXPECTED_METRIC,
            authoritative_contract=authoritative_contract,
        )
    except KernelFailedError as exc:
        assert "authoritative evaluation contract" in str(exc)
        evaluation = None
    assert evaluation is None


@pytest.mark.parametrize("failure_mode", ["missing-archive", "failure-json"])
def test_unscored_diagnostic_requires_archive_and_no_kernel_failure(
    tmp_path: Path,
    training_route_decision: TrainingRouteDecision,
    authoritative_contract: dict[str, object],
    failure_mode: str,
) -> None:
    kernel_result, metrics = _write_kernel_result(
        tmp_path,
        write_archive=failure_mode != "missing-archive",
        write_failure=failure_mode == "failure-json",
    )

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_route_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
        )
        is None
    )


def test_unscored_diagnostic_requires_every_routed_artifact_to_exist(
    tmp_path: Path,
    training_route_decision: TrainingRouteDecision,
    authoritative_contract: dict[str, object],
) -> None:
    kernel_result, metrics = _write_kernel_result(tmp_path)
    route_result = json.loads((kernel_result.output_dir / "route_result.json").read_text(encoding="utf-8"))
    route_result["artifact_paths"].append(str(kernel_result.output_dir / "missing-diagnostic.json"))
    (kernel_result.output_dir / "route_result.json").write_text(json.dumps(route_result), encoding="utf-8")

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_route_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode="writeup",
            submit_mode="file",
        )
        is None
    )


@pytest.mark.parametrize(
    ("deliverable_mode", "submit_mode"),
    [("leaderboard", "file"), ("writeup", "notebook")],
)
def test_unscored_diagnostic_is_limited_to_writeup_file_routes(
    tmp_path: Path,
    training_route_decision: TrainingRouteDecision,
    authoritative_contract: dict[str, object],
    deliverable_mode: str,
    submit_mode: str,
) -> None:
    kernel_result, metrics = _write_kernel_result(tmp_path)

    assert (
        _resolve_unscored_diagnostic_kernel_result(
            kernel_result=kernel_result,
            kernel_metrics=metrics,
            training_route_decision=training_route_decision,
            authoritative_evaluation_contract=authoritative_contract,
            deliverable_mode=deliverable_mode,
            submit_mode=submit_mode,
        )
        is None
    )


def test_authoritative_run_contract_wins_over_stale_rmse_context() -> None:
    run_payload = {
        "config": {
            "evaluation_contract": {
                "expected_metric": EXPECTED_METRIC,
                "expected_direction": "maximize",
                "accepted_score_sources": ["holdout"],
            }
        }
    }
    frozen_plan = {
        "target_metric": EXPECTED_METRIC,
        "target_direction": "maximize",
        "score_source": "holdout",
    }

    contract, source = _resolve_authoritative_evaluation_contract(
        run_payload=run_payload,
        frozen_plan=frozen_plan,
        fallback_metric="rmse",
        fallback_direction="minimize",
        fallback_score_source="cv",
    )
    warnings = _evaluation_spec_conflict_warnings(
        authoritative_contract=contract,
        evaluation_spec={"metric_name": "rmse", "direction": "minimize", "n_splits": 5},
    )

    assert source == "run.json.config.evaluation_contract"
    assert contract["expected_metric"] == EXPECTED_METRIC
    assert contract["expected_direction"] == "maximize"
    assert contract["accepted_score_sources"] == ["holdout"]
    assert len(warnings) == 2
    assert all("was ignored" in warning for warning in warnings)
