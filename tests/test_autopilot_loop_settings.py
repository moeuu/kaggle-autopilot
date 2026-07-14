from __future__ import annotations

from types import SimpleNamespace

from kagglebot.autopilot_loop_settings import (
    AutopilotLoopSettingsDefaults,
    resolve_autopilot_loop_settings,
)
from kagglebot.campaign import TOP1_TARGET_RANK_PERCENTILE


def _resolved_plan(**overrides: object) -> dict[str, object]:
    resolved: dict[str, object] = {
        "target_direction": "minimize",
        "deliverable_mode": "leaderboard",
        "submit_mode": "notebook",
        "submit_policy": "improved",
        "max_iterations": 3,
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "split_strategy": "Stratified_KFold",
        "seed": 42,
        "eval_seeds": None,
        "eval_repeats": None,
        "score_source": "cv",
        "max_total_min": 12.5,
        "time_budget_min": 30,
        "kernel_name": "demo-kernel",
        "internet": "off",
        "submission_gate": "always",
        "submission_limit_per_day": 2,
        "evaluation_contract": {"metric": "log_loss"},
        "readiness_target_score": None,
        "readiness_method": "ci_bound",
        "readiness_k": 1.5,
        "ci_method": "bootstrap",
        "ci_alpha": 0.1,
        "target_medal": None,
        "target_rank_percentile": None,
        "drift_check": True,
        "drift_weight": 2.0,
        "stop_min_delta": 0.01,
        "stop_no_improve_patience": 3,
        "stop_same_config_patience": 2,
        "rank_force_major_max_percentile": None,
        "rank_force_major_min_teams": None,
    }
    resolved.update(overrides)
    return resolved


def _defaults() -> AutopilotLoopSettingsDefaults:
    return AutopilotLoopSettingsDefaults(
        strict_competition_metric=True,
        require_submit_improvement=True,
        force_major_on_no_improve=False,
        force_major_rank_max_percentile=0.01,
        force_major_rank_min_teams=100,
    )


def test_resolve_autopilot_loop_settings_normalizes_and_updates_resolved(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_STRICT_COMPETITION_METRIC", raising=False)
    monkeypatch.setenv("KAGGLEBOT_REQUIRE_SUBMIT_IMPROVEMENT", "0")
    monkeypatch.setenv("KAGGLEBOT_FORCE_MAJOR_ON_NO_IMPROVE", "1")
    config = SimpleNamespace(
        campaign_mode="top1",
        portfolio_execution="disabled",
        validation_lab="off",
        research_scout="off",
        top1_submit_policy="value_only",
        top1_exhaustive=True,
        submit=True,
    )
    resolved = _resolved_plan()

    settings = resolve_autopilot_loop_settings(
        config=config,
        resolved=resolved,
        target_metric="log_loss",
        target_score=1.0,
        defaults=_defaults(),
    )

    assert settings.metric_direction == "minimize"
    assert settings.campaign_mode == "top1"
    assert settings.submit_mode == "notebook"
    assert settings.submit_enabled is True
    assert settings.require_submit_improvement is False
    assert settings.force_major_on_no_improve is True
    assert settings.max_iterations == 3
    assert settings.split_strategy == "stratified_kfold"
    assert settings.eval_seeds == [42]
    assert settings.time_budget_min == 30
    assert settings.enable_internet is False
    assert settings.target_rank_percentile == TOP1_TARGET_RANK_PERCENTILE
    assert settings.rank_force_major_max_percentile == TOP1_TARGET_RANK_PERCENTILE
    assert resolved["target_direction"] == "minimize"
    assert resolved["campaign_mode"] == "top1"
    assert resolved["submit_mode"] == "notebook"
    assert resolved["target_rank_percentile"] == TOP1_TARGET_RANK_PERCENTILE


def test_resolve_autopilot_loop_settings_disables_submit_for_writeup() -> None:
    config = SimpleNamespace(
        campaign_mode="auto",
        portfolio_execution=None,
        validation_lab=None,
        research_scout=None,
        top1_submit_policy=None,
        top1_exhaustive=False,
        submit=True,
    )
    resolved = _resolved_plan(deliverable_mode="writeup", max_iterations=0, submission_limit_per_day=0)

    settings = resolve_autopilot_loop_settings(
        config=config,
        resolved=resolved,
        target_metric="accuracy",
        target_score=0.5,
        defaults=_defaults(),
    )

    assert settings.writeup_mode is True
    assert settings.submit_enabled is False
    assert settings.writeup_submit_enabled is True
    assert settings.max_iterations == 1
    assert settings.submission_limit_per_day is None
    assert resolved["deliverable_mode"] == "writeup"
