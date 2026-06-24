from __future__ import annotations

import json
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.plan_policy import (
    apply_competition_eval_override,
    competition_eval_override,
    extract_plan_split_strategy_hints,
    infer_split_strategy_from_hint_text,
    normalize_split_strategy_name,
    resolve_split_strategy_from_artifacts,
)


def test_normalize_split_strategy_name_handles_aliases() -> None:
    assert normalize_split_strategy_name("groupkfold") == "group_kfold"
    assert normalize_split_strategy_name("time series") is None
    assert normalize_split_strategy_name("timeseriessplit") == "timeseries_split"
    assert normalize_split_strategy_name("unknown") is None


def test_infer_split_strategy_from_hint_text_handles_natural_language() -> None:
    assert infer_split_strategy_from_hint_text("Use chronological validation for forecasting") == "timeseries_split"
    assert infer_split_strategy_from_hint_text("Use grouped CV by patient") == "group_kfold"
    assert infer_split_strategy_from_hint_text("Stratified CV on the target") == "stratified_kfold"


def test_extract_plan_split_strategy_hints_reads_nested_fields() -> None:
    payload = {
        "evaluation_protocol": {"cv_type": "group_kfold"},
        "toggles": {"SPLIT_STRATEGY": "timeseries_split"},
        "split_strategy": "kfold",
    }

    assert extract_plan_split_strategy_hints(payload) == ["group_kfold", "timeseries_split", "kfold"]


def test_resolve_split_strategy_from_artifacts_prefers_highest_priority_hint(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.base_dir.mkdir(parents=True)
    paths.plan_path.write_text(
        json.dumps({"evaluation_protocol": {"cv_type": "group_kfold"}, "toggles": {"CV_TYPE": "timeseries_split"}}),
        encoding="utf-8",
    )

    split_strategy, note = resolve_split_strategy_from_artifacts(paths=paths, split_strategy="kfold")

    assert split_strategy == "timeseries_split"
    assert note is not None
    assert "plan evaluation hints" in note


def test_resolve_split_strategy_from_artifacts_uses_profile_classification_default(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"task": "classification", "modality": "tabular"}),
        encoding="utf-8",
    )

    split_strategy, note = resolve_split_strategy_from_artifacts(paths=paths, split_strategy="kfold")

    assert split_strategy == "stratified_kfold"
    assert note is not None
    assert "classification task" in note


def test_competition_eval_override_applies_deep_past_contract() -> None:
    override = competition_eval_override("deep-past-initiative-machine-translation")
    payload = apply_competition_eval_override(
        slug="deep-past-initiative-machine-translation",
        payload={"task": "classification"},
        include_spec_keys=True,
    )

    assert override["split_strategy"] == "group_kfold"
    assert payload["task"] == "translation"
    assert payload["metric_name"] == override["metric_name"]
    assert payload["split_strategy"] == "group_kfold"
