from __future__ import annotations

from kagglebot.training_route import (
    decide_training_route,
    plan_requests_non_training,
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
