from __future__ import annotations

import json

from kagglebot.context_artifacts import (
    count_csv_data_rows_capped,
    load_dataset_profile,
    load_evaluation_spec,
)


def test_count_csv_data_rows_capped_returns_none_for_missing_file(tmp_path) -> None:
    assert count_csv_data_rows_capped(tmp_path / "missing.csv") is None


def test_count_csv_data_rows_capped_counts_data_rows_and_caps(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    assert count_csv_data_rows_capped(csv_path, cap=10) == 3
    assert count_csv_data_rows_capped(csv_path, cap=2) == 3


def test_load_dataset_profile_returns_empty_for_missing_or_invalid(tmp_path) -> None:
    assert load_dataset_profile(slug="demo", dataset_profile_path=tmp_path / "missing.json") == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")

    assert load_dataset_profile(slug="demo", dataset_profile_path=invalid) == {}


def test_load_dataset_profile_applies_competition_override(tmp_path) -> None:
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps({"task": "classification"}), encoding="utf-8")

    profile = load_dataset_profile(
        slug="deep-past-initiative-machine-translation",
        dataset_profile_path=profile_path,
    )

    assert profile["task"] == "translation"
    assert profile["split_strategy_hint"] == "group_kfold"


def test_load_evaluation_spec_returns_empty_for_missing_or_invalid(tmp_path) -> None:
    assert load_evaluation_spec(slug="demo", evaluation_spec_path=tmp_path / "missing.json") == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")

    assert load_evaluation_spec(slug="demo", evaluation_spec_path=invalid) == {}


def test_load_evaluation_spec_validates_and_normalizes(tmp_path) -> None:
    spec_path = tmp_path / "evaluation_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "metric_name": "accuracy",
                "direction": "maximize",
                "split_strategy": "stratified_kfold",
                "n_splits": 5,
                "seeds": [11, 22],
                "repeats": 2,
                "ci_method": "normal",
                "ci_alpha": 0.05,
                "readiness_rule": {
                    "method": "mean_std",
                    "k": 1.5,
                    "target_score": 0.8,
                    "submission_gate": "readiness_only",
                },
                "drift_check": {"enabled": True, "drift_weight": 0.25},
                "stop_policy": {
                    "min_delta": 0.01,
                    "no_improve_patience": 3,
                    "same_config_patience": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_evaluation_spec(slug="demo", evaluation_spec_path=spec_path)

    assert spec["metric_name"] == "accuracy"
    assert spec["direction"] == "maximize"
    assert spec["split_strategy"] == "stratified_kfold"
