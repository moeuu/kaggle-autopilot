from __future__ import annotations

import json
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.plan_resolution import resolve_plan_for_autopilot
from kagglebot.types import PlanConfig


def test_resolve_plan_for_autopilot_returns_resolved_payload(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")

    resolved = resolve_plan_for_autopilot(
        plan=PlanConfig(max_iterations=2, score_source="cv"),
        paths=paths,
        compute="local_cpu",
        target_metric=None,
        target_score=None,
        target_direction=None,
        score_source=None,
        holdout_frac=None,
        cv_folds=None,
        seed=42,
        time_budget_min=None,
        kernel_name=None,
        internet="off",
        max_iterations=None,
        max_total_min=None,
        patience=2,
        min_improvement=0.0,
        submit_policy="never",
        default_strict_competition_metric=True,
        default_target_medal="gold",
        default_limited_submission_gate="readiness_or_final",
        default_max_iterations=5,
        heavy_local_gpu_max_cv_folds=3,
        long_local_gpu_iteration_budget_min=12 * 60,
        long_local_gpu_max_iterations=3,
        default_force_major_rank_max_percentile=0.35,
        default_force_major_rank_min_teams=200,
        on_message=lambda _message: None,
    )

    assert resolved["max_iterations"] == 2
    assert resolved["score_source"] == "cv"
    assert resolved["submit_policy"] == "always"
    assert resolved["evaluation_contract"]["accepted_score_sources"] == ["cv", "holdout"]
    assert resolved["evaluation_contract"]["require_metric_match"] is True


def test_resolve_plan_for_autopilot_uses_dataset_profile_full_dataset_contract(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.dataset_profile_path.parent.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"data_root_layout": "flat_full"}),
        encoding="utf-8",
    )

    resolved = resolve_plan_for_autopilot(
        plan=PlanConfig(max_iterations=2, score_source="cv"),
        paths=paths,
        compute="local_cpu",
        target_metric="rmse",
        target_score=None,
        target_direction="minimize",
        score_source=None,
        holdout_frac=None,
        cv_folds=None,
        seed=42,
        time_budget_min=None,
        kernel_name=None,
        internet="off",
        max_iterations=None,
        max_total_min=None,
        patience=2,
        min_improvement=0.0,
        submit_policy="never",
        default_strict_competition_metric=True,
        default_target_medal="gold",
        default_limited_submission_gate="readiness_or_final",
        default_max_iterations=5,
        heavy_local_gpu_max_cv_folds=3,
        long_local_gpu_iteration_budget_min=12 * 60,
        long_local_gpu_max_iterations=3,
        default_force_major_rank_max_percentile=0.35,
        default_force_major_rank_min_teams=200,
        on_message=lambda _message: None,
    )

    assert resolved["evaluation_contract"]["require_full_dataset"] is True


def test_resolve_plan_for_autopilot_uses_competition_policy_evaluation_overrides(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.competition_policy_path.parent.mkdir(parents=True, exist_ok=True)
    paths.competition_policy_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "fallback_overrides": {
                        "metric": "auc",
                        "direction": "maximize",
                        "split_strategy_hint": "GroupKFold(user_id)",
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    resolved = resolve_plan_for_autopilot(
        plan=PlanConfig(max_iterations=2, score_source="cv"),
        paths=paths,
        compute="local_cpu",
        target_metric="accuracy",
        target_score=None,
        target_direction="maximize",
        score_source=None,
        holdout_frac=None,
        cv_folds=None,
        seed=42,
        time_budget_min=None,
        kernel_name=None,
        internet="off",
        max_iterations=None,
        max_total_min=None,
        patience=2,
        min_improvement=0.0,
        submit_policy="never",
        default_strict_competition_metric=True,
        default_target_medal="gold",
        default_limited_submission_gate="readiness_or_final",
        default_max_iterations=5,
        heavy_local_gpu_max_cv_folds=3,
        long_local_gpu_iteration_budget_min=12 * 60,
        long_local_gpu_max_iterations=3,
        default_force_major_rank_max_percentile=0.35,
        default_force_major_rank_min_teams=200,
        on_message=lambda _message: None,
    )

    assert resolved["target_metric"] == "auc"
    assert resolved["target_direction"] == "maximize"
    assert resolved["split_strategy"] == "group_kfold"
    assert resolved["evaluation_contract"]["expected_metric"] == "auc"
    assert resolved["evaluation_contract"]["expected_split_strategy"] == "group_kfold"
