from __future__ import annotations

import pytest

import kagglebot.orchestrator.agent_pipeline as agent_pipeline


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

    normalized = agent_pipeline._normalize_plan_payload(payload)  # noqa: SLF001
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

    normalized = agent_pipeline._normalize_plan_payload(payload)  # noqa: SLF001
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

    issues = agent_pipeline._validate_plan_payload(payload)  # noqa: SLF001
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

    issues = agent_pipeline._validate_plan_payload(payload)  # noqa: SLF001
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

    normalized = agent_pipeline._normalize_plan_payload(payload)  # noqa: SLF001
    pipelines = normalized["pipelines"]
    assert isinstance(pipelines, list)
    key_hyperparameters = pipelines[0]["key_hyperparameters"]
    assert key_hyperparameters == {
        "dropout": 0.05,
        "optimizer": {"lr": 0.001, "weight_decay": 0.01},
    }


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

    issues = agent_pipeline._validate_plan_payload(payload)  # noqa: SLF001
    assert any("key_hyperparameters must be an object" in issue for issue in issues)
