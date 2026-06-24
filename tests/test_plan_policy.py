from __future__ import annotations

import json
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.plan_policy import (
    apply_competition_eval_override,
    competition_eval_override,
    expanded_eval_seeds,
    extract_plan_split_strategy_hints,
    infer_split_strategy_from_hint_text,
    normalize_eval_repeats,
    normalize_eval_seeds,
    normalize_rank_force_min_teams,
    normalize_rank_force_percentile,
    normalize_split_strategy_name,
    resolve_deliverable_mode,
    resolve_split_strategy_from_artifacts,
    resolve_submit_mode,
    upgrade_improvement_mode,
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


def test_normalize_eval_seeds_deduplicates_and_uses_defaults() -> None:
    defaults = (42, 2024, 777)

    assert normalize_eval_seeds([1, "x", 1, 2], default_seeds=defaults) == [1, 2]
    assert normalize_eval_seeds(None, fallback=[3, 3], default_seeds=defaults) == [3]
    assert normalize_eval_seeds(None, default_seeds=defaults) == [42, 2024, 777]


def test_resolve_deliverable_mode_prefers_spec_then_inferred_then_plan() -> None:
    assert (
        resolve_deliverable_mode(plan_value="leaderboard", spec_value="writeup", inferred_value="leaderboard")
        == "writeup"
    )
    assert (
        resolve_deliverable_mode(plan_value="writeup", spec_value=None, inferred_value="leaderboard") == "leaderboard"
    )
    assert resolve_deliverable_mode(plan_value="writeup", spec_value=None, inferred_value=None) == "writeup"
    assert resolve_deliverable_mode(plan_value=None, spec_value=None, inferred_value=None) == "leaderboard"


def test_resolve_submit_mode_prefers_spec_then_inferred_then_plan() -> None:
    assert resolve_submit_mode(plan_value="file", spec_value="notebook", inferred_value="file") == "notebook"
    assert resolve_submit_mode(plan_value="file", spec_value=None, inferred_value="notebook") == "notebook"
    assert resolve_submit_mode(plan_value="notebook", spec_value=None, inferred_value=None) == "notebook"
    assert resolve_submit_mode(plan_value=None, spec_value=None, inferred_value=None) == "file"


def test_normalize_eval_repeats_clamps_range() -> None:
    assert normalize_eval_repeats(0, default_repeats=2) == 1
    assert normalize_eval_repeats(99, default_repeats=2) == 10
    assert normalize_eval_repeats(None, fallback=3, default_repeats=2) == 3
    assert normalize_eval_repeats(None, default_repeats=2) == 2


def test_normalize_rank_force_values() -> None:
    assert normalize_rank_force_percentile(0.2, fallback=0.5) == 0.2
    assert normalize_rank_force_percentile(2.0, fallback=0.5) == 0.5
    assert normalize_rank_force_min_teams(10.9, fallback=100) == 10
    assert normalize_rank_force_min_teams(-1, fallback=100) == 1


def test_upgrade_improvement_mode_respects_priority() -> None:
    assert upgrade_improvement_mode("minor_tuning", "major_overhaul") == "major_overhaul"
    assert upgrade_improvement_mode("validation_redesign", "major_overhaul") == "validation_redesign"
    assert upgrade_improvement_mode("moderate_update", None) == "moderate_update"


def test_expanded_eval_seeds_offsets_repeats() -> None:
    assert expanded_eval_seeds(
        base_seeds=[1, 2],
        repeats=2,
        default_seeds=(42,),
        default_repeats=1,
        repeat_seed_offset=100,
    ) == [1, 2, 101, 102]
