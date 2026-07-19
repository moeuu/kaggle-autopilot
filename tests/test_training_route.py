from __future__ import annotations

from kagglebot.training_route import (
    decide_training_route,
    is_unscored_non_training_diagnostic,
    plan_requests_non_training,
    resolve_non_training_validation_blockers,
    validate_non_training_metrics,
    validate_non_training_source,
)


def _ready_plan() -> dict[str, object]:
    return {
        "runtime_budget": {
            "local_training_required": False,
            "estimated_local_training_min": 1_500,
            "non_training_submission": {
                "mode": "pretrained_inference",
                "implementation_ready": True,
                "validation_mode": "reference_reproduction",
                "source": "kernel.py loads the attached frozen checkpoint and reproduces reference outputs",
            },
        }
    }


def test_ready_non_training_path_skips_local_training() -> None:
    decision = decide_training_route(_ready_plan())

    assert decision.skip_local_training is True
    assert decision.direct_notebook is False
    assert decision.mode == "pretrained_inference"


def test_ready_code_competition_uses_direct_notebook_route() -> None:
    decision = decide_training_route(
        _ready_plan(),
        compute="local_gpu",
        deliverable_mode="leaderboard",
        submit_mode="notebook",
        code_competition=True,
    )

    assert decision.skip_local_training is True
    assert decision.direct_notebook is True


def test_heavy_training_alone_does_not_skip_training() -> None:
    decision = decide_training_route(
        {
            "runtime_budget": {
                "local_training_required": False,
                "estimated_local_training_min": 10_000,
            }
        }
    )

    assert decision.skip_local_training is False
    assert decision.reason == "non_training_mode_missing_or_unsupported"


def test_sample_submission_is_never_accepted_as_non_training_evidence() -> None:
    plan = _ready_plan()
    proposal = plan["runtime_budget"]["non_training_submission"]  # type: ignore[index]
    proposal["source"] = "copy sample_submission.csv as a placeholder"  # type: ignore[index]

    decision = decide_training_route(plan)

    assert decision.skip_local_training is False
    assert decision.reason == "unsafe_sample_or_placeholder_evidence"


def test_lightweight_optional_training_keeps_normal_training_route() -> None:
    plan = _ready_plan()
    plan["runtime_budget"]["estimated_local_training_min"] = 90  # type: ignore[index]

    decision = decide_training_route(plan)

    assert decision.skip_local_training is False
    assert decision.reason == "local_training_not_proven_very_heavy"


def test_just_under_24_hours_keeps_normal_training_route() -> None:
    plan = _ready_plan()
    plan["runtime_budget"]["estimated_local_training_min"] = 1_439  # type: ignore[index]

    decision = decide_training_route(plan)

    assert decision.skip_local_training is False
    assert decision.reason == "local_training_not_proven_very_heavy"


def test_exactly_24_hours_allows_a_ready_non_training_route() -> None:
    plan = _ready_plan()
    plan["runtime_budget"]["estimated_local_training_min"] = 1_440  # type: ignore[index]

    decision = decide_training_route(plan)

    assert decision.skip_local_training is True


def test_explicitly_required_local_training_is_never_skipped() -> None:
    plan = _ready_plan()
    plan["runtime_budget"]["local_training_required"] = True  # type: ignore[index]
    plan["runtime_budget"]["estimated_local_training_min"] = 100_000  # type: ignore[index]

    decision = decide_training_route(plan)

    assert decision.skip_local_training is False
    assert decision.reason == "local_training_not_explicitly_optional"


def test_cost_class_without_24_hour_estimate_does_not_skip_training() -> None:
    plan = _ready_plan()
    plan["runtime_budget"].pop("estimated_local_training_min")  # type: ignore[union-attr]
    plan["runtime_budget"]["training_cost_class"] = "extreme"  # type: ignore[index]

    decision = decide_training_route(plan)

    assert decision.skip_local_training is False
    assert decision.reason == "local_training_not_proven_very_heavy"


def test_non_training_metrics_must_prove_the_approved_route_ran() -> None:
    decision = decide_training_route(_ready_plan())

    assert (
        validate_non_training_metrics(
            {
                "execution_mode": "non_training_submission",
                "training_performed": False,
                "non_training_validation_passed": True,
                "non_training_validation_mode": "reference_reproduction",
            },
            decision,
        )
        == ()
    )
    assert validate_non_training_metrics({"offline_value": 0.8}, decision)


def test_validation_unavailable_diagnostic_is_nonfatal_but_not_submittable() -> None:
    decision = decide_training_route(_ready_plan())
    metrics = {
        "execution_mode": "non_training_submission",
        "training_performed": False,
        "non_training_validation_passed": False,
        "non_training_validation_mode": "reference_reproduction",
        "execution_status": "validation_unavailable",
        "primary_score": None,
        "remediation": ["Mount the pinned public evaluator."],
    }

    assert resolve_non_training_validation_blockers(
        metrics=metrics,
        route_result={"manifest": {"submission_ready": False}},
        submission_contract={"submission_ready": False, "canonical_submission_emitted": False},
        decision=decision,
        canonical_submission_emitted=False,
    ) == ("Mount the pinned public evaluator.",)


def test_false_non_training_validation_cannot_hide_a_submission_artifact() -> None:
    decision = decide_training_route(_ready_plan())
    metrics = {
        "execution_mode": "non_training_submission",
        "training_performed": False,
        "non_training_validation_passed": False,
        "non_training_validation_mode": "reference_reproduction",
        "execution_status": "validation_unavailable",
        "primary_score": None,
        "remediation": ["Mount the pinned public evaluator."],
    }

    assert (
        resolve_non_training_validation_blockers(
            metrics=metrics,
            route_result={"manifest": {"submission_ready": True}},
            submission_contract={"submission_ready": True, "canonical_submission_emitted": True},
            decision=decision,
            canonical_submission_emitted=True,
        )
        is None
    )


def test_successful_static_diagnostic_can_remain_unscored_and_non_submittable() -> None:
    decision = decide_training_route(_ready_plan())
    metrics = {
        "execution_mode": "non_training_submission",
        "training_performed": False,
        "execution_status": "success",
        "non_training_validation_passed": True,
        "non_training_validation_mode": "reference_reproduction",
        "offline_artifact_validation_passed": True,
        "official_paired_validation_passed": False,
        "submission_ready": False,
        "primary_score": None,
    }

    assert is_unscored_non_training_diagnostic(
        metrics=metrics,
        route_result={
            "mode": "skill_artifact",
            "validation_status": "success",
            "artifact_paths": ["/tmp/diagnostic_skill_portfolio.zip"],
            "manifest": {"archive": "diagnostic_skill_portfolio.zip", "submission_ready": False},
        },
        submission_contract={
            "archive_name": "diagnostic_skill_portfolio.zip",
            "submission_ready": False,
            "canonical_submission_emitted": False,
            "official_paired_validation_passed": False,
        },
        decision=decision,
        canonical_submission_emitted=False,
    )


def test_ordinary_non_training_route_cannot_be_accepted_without_a_score() -> None:
    decision = decide_training_route(_ready_plan())
    metrics = {
        "execution_mode": "non_training_submission",
        "training_performed": False,
        "execution_status": "success",
        "non_training_validation_passed": True,
        "non_training_validation_mode": "reference_reproduction",
        "offline_artifact_validation_passed": True,
        "official_paired_validation_passed": False,
        "submission_ready": False,
        "primary_score": None,
    }

    assert not is_unscored_non_training_diagnostic(
        metrics=metrics,
        route_result={
            "mode": "tabular",
            "validation_status": "success",
            "manifest": {"archive": "diagnostic_skill_portfolio.zip", "submission_ready": False},
        },
        submission_contract={},
        decision=decision,
        canonical_submission_emitted=False,
    )


def test_unscored_diagnostic_rejects_score_and_readiness_contradictions() -> None:
    decision = decide_training_route(_ready_plan())
    base_metrics = {
        "execution_mode": "non_training_submission",
        "training_performed": False,
        "execution_status": "success",
        "non_training_validation_passed": True,
        "non_training_validation_mode": "reference_reproduction",
        "offline_artifact_validation_passed": True,
        "official_paired_validation_passed": False,
        "submission_ready": False,
        "primary_score": None,
    }
    contradictions = (
        {"official_paired_validation_passed": True},
        {"submission_ready": True},
        {"primary_score": 0.4},
        {"primary_score": float("nan")},
        {"primary_score": float("inf")},
    )

    for contradiction in contradictions:
        assert not is_unscored_non_training_diagnostic(
            metrics={**base_metrics, **contradiction},
            route_result={
                "mode": "skill_artifact",
                "validation_status": "success",
                "manifest": {"archive": "diagnostic_skill_portfolio.zip", "submission_ready": False},
            },
            submission_contract={
                "archive_name": "diagnostic_skill_portfolio.zip",
                "submission_ready": False,
                "canonical_submission_emitted": False,
                "official_paired_validation_passed": False,
            },
            decision=decision,
            canonical_submission_emitted=False,
        )


def test_unscored_diagnostic_cannot_hide_a_canonical_submission() -> None:
    decision = decide_training_route(_ready_plan())
    metrics = {
        "execution_mode": "non_training_submission",
        "training_performed": False,
        "execution_status": "success",
        "non_training_validation_passed": True,
        "non_training_validation_mode": "reference_reproduction",
        "offline_artifact_validation_passed": True,
        "official_paired_validation_passed": False,
        "submission_ready": False,
        "primary_score": None,
    }

    assert not is_unscored_non_training_diagnostic(
        metrics=metrics,
        route_result={
            "mode": "skill_artifact",
            "validation_status": "success",
            "manifest": {"archive": "submission.zip", "submission_ready": True},
        },
        submission_contract={
            "archive_name": "submission.zip",
            "submission_ready": True,
            "canonical_submission_emitted": True,
            "official_paired_validation_passed": False,
        },
        decision=decision,
        canonical_submission_emitted=True,
    )


def test_runner_does_not_trust_a_bare_approved_marker() -> None:
    assert not plan_requests_non_training({"execution_route": {"mode": "non_training_submission", "approved": True}})

    plan = _ready_plan()
    plan["execution_route"] = {"mode": "non_training_submission", "approved": True}
    assert plan_requests_non_training(plan)


def test_non_training_source_must_implement_runtime_and_metrics_contract() -> None:
    assert validate_non_training_source("print('ordinary training kernel')")
    assert (
        validate_non_training_source(
            "KAGGLEBOT_EXECUTION_MODE non_training_submission training_performed "
            "non_training_validation_passed non_training_validation_mode"
        )
        == ()
    )
