from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.paths import CompetitionPaths
from kagglebot.plan_policy import (
    normalize_plan_payload,
    plan_config_from_resolved,
    repair_plan_payload_for_profile,
    validate_plan_payload,
    write_resolved_plan_config,
)


def test_normalize_plan_payload_injects_pipeline_names() -> None:
    payload = {
        "pipelines": [
            {
                "features": ["a"],
                "models": ["LightGBM Classifier"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "name": "already_set",
                "features": [],
                "models": [],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "features": [],
                "models": [],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
        ]
    }

    normalized = normalize_plan_payload(payload)
    pipelines = normalized["pipelines"]
    assert isinstance(pipelines, list)
    assert pipelines[0]["name"] == "lightgbm_classifier"
    assert pipelines[1]["name"] == "already_set"
    assert pipelines[2]["name"] == "pipeline_3"


def test_normalize_plan_payload_injects_suite_names() -> None:
    payload = {
        "suites": [
            {
                "train_mode": "competition_only",
                "feature_recipe": "full",
                "lightweight": False,
                "promotion_stage": "full_eval",
            },
            {
                "name": "already_set",
                "train_mode": "competition_plus_original",
                "feature_recipe": "full",
                "lightweight": False,
                "promotion_stage": "ablation_fast",
            },
            {
                "feature_recipe": "orig_signal_only",
                "lightweight": True,
                "promotion_stage": "ablation_fast",
            },
        ]
    }

    normalized = normalize_plan_payload(payload)
    suites = normalized["suites"]
    assert isinstance(suites, list)
    assert suites[0]["name"] == "competition_only"
    assert suites[1]["name"] == "already_set"
    assert suites[2]["name"] == "orig_signal_only"


def test_validate_plan_payload_tolerates_missing_pipeline_names() -> None:
    payload = {
        "target_metric": "roc_auc",
        "target_direction": "maximize",
        "target_score": 0.9,
        "score_source": "cv",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 3,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "features": ["a"],
                "models": ["XGBoost"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "features": ["b"],
                "models": ["LightGBM"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
        ],
        "toggles": {"FAST_DEV": False},
        "evaluation_protocol": {
            "cv_type": "StratifiedKFold",
            "n_folds": 5,
            "seeds": [42, 2024, 3407],
            "primary_metric": "roc_auc",
        },
        "stop_policy": {"max_iterations": 3, "error_fingerprint_abort": True},
    }

    issues = validate_plan_payload(payload)
    assert issues == []


@pytest.mark.parametrize("score_source", ["auto", "test"])
def test_validate_plan_payload_rejects_non_generalizable_score_source(score_source: str) -> None:
    payload = {
        "target_metric": "roc_auc",
        "target_direction": "maximize",
        "target_score": 0.9,
        "score_source": score_source,
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 3,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "name": "pipe_a",
                "features": ["a"],
                "models": ["XGBoost"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "name": "pipe_b",
                "features": ["b"],
                "models": ["LightGBM"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
        ],
        "toggles": {"FAST_DEV": False},
        "evaluation_protocol": {
            "cv_type": "StratifiedKFold",
            "n_folds": 5,
            "seeds": [42, 2024, 3407],
            "primary_metric": "roc_auc",
        },
        "stop_policy": {"max_iterations": 3, "error_fingerprint_abort": True},
    }

    issues = validate_plan_payload(payload)
    assert any("score_source must be one of: holdout, cv." in issue for issue in issues)


def test_normalize_plan_payload_scalarizes_pipeline_hyperparameters() -> None:
    payload = {
        "pipelines": [
            {
                "name": "pipe_a",
                "features": ["a"],
                "models": ["XGBoost"],
                "key_hyperparameters": {
                    "dropout": [0.05, 0.1],
                    "optimizer": {"lr": [0.001, 0.0005], "weight_decay": [0.01]},
                    "empty": [],
                },
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            }
        ]
    }

    normalized = normalize_plan_payload(payload)
    pipelines = normalized["pipelines"]
    assert isinstance(pipelines, list)
    key_hyperparameters = pipelines[0]["key_hyperparameters"]
    assert key_hyperparameters == {
        "dropout": 0.05,
        "optimizer": {"lr": 0.001, "weight_decay": 0.01},
    }


def test_repair_plan_payload_for_profile_adds_high_accuracy_suites() -> None:
    payload = {
        "suite_aware_ablations": ["competition only", "competition plus original", "orig signal only"],
        "toggles": {"suite_ablations": ["ignored duplicate alias"]},
    }
    profile = {
        "task": "classification",
        "modality": "tabular",
        "tags": ["binary"],
        "train_rows": 10000,
        "categorical_columns": ["a", "b", "c"],
    }

    repaired = repair_plan_payload_for_profile(payload, profile)

    suites = repaired["suites"]
    assert isinstance(suites, list)
    assert [suite["name"] for suite in suites[:3]] == [
        "competition_only",
        "competition_plus_original",
        "orig_signal_only",
    ]
    assert "suite_aware_ablations" not in repaired
    assert repaired["toggles"] == {}


def test_plan_config_from_resolved_preserves_autopilot_defaults() -> None:
    plan = plan_config_from_resolved(
        {
            "deliverable_mode": "leaderboard",
            "submit_mode": "file",
            "target_medal": "gold",
            "target_rank_percentile": "0.5",
            "target_metric": "log_loss",
            "target_direction": "minimize",
            "score_source": "cv",
            "internet": "off",
            "submit_policy": "always",
            "submission_gate": "none",
            "rank_force_major_max_percentile": "bad",
            "rank_force_major_min_teams": "bad",
        },
        default_max_iterations=7,
        default_force_major_rank_max_percentile=2.5,
        default_force_major_rank_min_teams=100,
    )

    assert plan.max_iterations == 7
    assert plan.patience == 2
    assert plan.min_improvement == 0.0
    assert plan.readiness_k == 1.0
    assert plan.ci_alpha == 0.05
    assert plan.drift_weight == 1.0
    assert plan.stop_no_improve_patience == 0
    assert plan.rank_force_major_max_percentile == 2.5
    assert plan.rank_force_major_min_teams == 100


def test_write_resolved_plan_config_persists_policy_defaults(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")

    write_resolved_plan_config(
        paths,
        {
            "target_metric": "log_loss",
            "target_direction": "minimize",
            "target_score": 0.5,
            "score_source": "cv",
            "internet": "off",
            "rank_force_major_max_percentile": "bad",
            "rank_force_major_min_teams": "bad",
        },
        default_max_iterations=7,
        default_force_major_rank_max_percentile=2.5,
        default_force_major_rank_min_teams=100,
    )

    payload = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    assert payload["max_iterations"] == 5
    assert payload["rank_force_major_max_percentile"] == 2.5
    assert payload["rank_force_major_min_teams"] == 100
    assert payload["toggles"]["ENABLE_TRAINING"] is True


def test_validate_plan_payload_rejects_non_object_key_hyperparameters() -> None:
    payload = {
        "target_metric": "roc_auc",
        "target_direction": "maximize",
        "target_score": 0.9,
        "score_source": "cv",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 3,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "name": "pipe_a",
                "features": ["a"],
                "models": ["XGBoost"],
                "key_hyperparameters": ["bad"],
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "name": "pipe_b",
                "features": ["b"],
                "models": ["LightGBM"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
        ],
        "toggles": {"FAST_DEV": False},
        "evaluation_protocol": {
            "cv_type": "StratifiedKFold",
            "n_folds": 5,
            "seeds": [42, 2024, 3407],
            "primary_metric": "roc_auc",
        },
        "stop_policy": {"max_iterations": 3, "error_fingerprint_abort": True},
    }

    issues = validate_plan_payload(payload)
    assert any("key_hyperparameters must be an object" in issue for issue in issues)
