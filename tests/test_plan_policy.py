from __future__ import annotations

import json
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.plan_policy import (
    DEFAULT_EVAL_REPEATS,
    DEFAULT_EVAL_SEEDS,
    EVAL_REPEAT_SEED_OFFSET,
    FULL_DATASET_REQUIRED_COMPETITIONS,
    apply_competition_eval_override,
    build_evaluation_contract,
    competition_eval_override,
    expanded_default_eval_seeds,
    expanded_eval_seeds,
    extract_evaluation_spec_values,
    extract_plan_split_strategy_hints,
    infer_split_strategy_from_hint_text,
    normalize_default_eval_repeats,
    normalize_default_eval_seeds,
    normalize_eval_repeats,
    normalize_eval_seeds,
    normalize_rank_force_min_teams,
    normalize_rank_force_percentile,
    normalize_split_strategy_name,
    resolve_deliverable_mode,
    resolve_split_strategy_from_artifacts,
    resolve_submit_mode,
    resolve_target_metric_direction,
    upgrade_improvement_mode,
)


def test_normalize_split_strategy_name_handles_aliases() -> None:
    assert normalize_split_strategy_name("groupkfold") == "group_kfold"
    assert normalize_split_strategy_name("time series") is None
    assert normalize_split_strategy_name("timeseriessplit") == "timeseries_split"
    assert normalize_split_strategy_name("unknown") is None


def test_extract_evaluation_spec_values_reads_nested_policy_sections() -> None:
    values = extract_evaluation_spec_values(
        {
            "metric_name": "log_loss",
            "direction": "minimize",
            "split_strategy": "stratified_kfold",
            "n_splits": 5,
            "seeds": [11, "bad", 22],
            "repeats": 3,
            "ci_method": "bootstrap",
            "ci_alpha": 0.1,
            "readiness_rule": {
                "method": "mean_std",
                "k": 1.5,
                "target_score": 0.4,
                "submission_gate": "readiness_only",
            },
            "drift_check": {"enabled": True, "drift_weight": 0.25},
            "stop_policy": {"min_delta": 0.01, "no_improve_patience": 4, "same_config_patience": 2},
        }
    )

    assert values.metric_name == "log_loss"
    assert values.direction == "minimize"
    assert values.split_strategy == "stratified_kfold"
    assert values.n_splits == 5
    assert values.seed == 11
    assert values.eval_seeds == (11, 22)
    assert values.repeats == 3
    assert values.ci_method == "bootstrap"
    assert values.ci_alpha == 0.1
    assert values.readiness_method == "mean_std"
    assert values.readiness_k == 1.5
    assert values.readiness_target_score == 0.4
    assert values.submission_gate == "readiness_only"
    assert values.drift_enabled is True
    assert values.drift_weight == 0.25
    assert values.stop_min_delta == 0.01
    assert values.stop_no_improve_patience == 4
    assert values.stop_same_config_patience == 2


def test_extract_evaluation_spec_values_ignores_wrong_shapes() -> None:
    values = extract_evaluation_spec_values(
        {
            "metric_name": 123,
            "direction": ["minimize"],
            "n_splits": "5",
            "seeds": "11",
            "readiness_rule": "bad",
            "drift_check": {"enabled": "yes"},
            "stop_policy": {"no_improve_patience": "4"},
        }
    )

    assert values.metric_name is None
    assert values.direction is None
    assert values.n_splits is None
    assert values.seed is None
    assert values.eval_seeds == ()
    assert values.readiness_method is None
    assert values.drift_enabled is None
    assert values.stop_no_improve_patience is None


def test_resolve_target_metric_direction_applies_competition_override() -> None:
    decision = resolve_target_metric_direction(
        target_metric="accuracy",
        target_direction="maximize",
        spec_metric="accuracy",
        spec_direction="maximize",
        explicit_target_metric=False,
        explicit_target_direction=False,
        strict_competition_metric=True,
        competition_override={
            "metric_name": "Geometric Mean of the BLEU and the chrF++ scores",
            "direction": "maximize",
            "split_strategy": "group_kfold",
        },
    )

    assert decision.target_metric == "Geometric Mean of the BLEU and the chrF++ scores"
    assert decision.target_direction == "maximize"
    assert decision.override_split_strategy == "group_kfold"
    assert decision.messages == (
        "[yellow]note[/yellow]: competition override is active; "
        "forcing target_metric 'accuracy' -> 'Geometric Mean of the BLEU and the chrF++ scores'.",
    )


def test_resolve_target_metric_direction_strict_mode_uses_spec_when_not_explicit() -> None:
    decision = resolve_target_metric_direction(
        target_metric="accuracy",
        target_direction="maximize",
        spec_metric="rmse",
        spec_direction="minimize",
        explicit_target_metric=False,
        explicit_target_direction=False,
        strict_competition_metric=True,
        competition_override={},
    )

    assert decision.target_metric == "rmse"
    assert decision.target_direction == "minimize"
    assert decision.messages == (
        "[yellow]note[/yellow]: strict competition metric mode is enabled; "
        "overriding target_metric 'accuracy' -> 'rmse'.",
        "[yellow]note[/yellow]: strict competition metric mode is enabled; "
        "overriding target_direction 'maximize' -> 'minimize'.",
    )


def test_resolve_target_metric_direction_strict_mode_keeps_explicit_values() -> None:
    decision = resolve_target_metric_direction(
        target_metric="balanced_accuracy",
        target_direction="maximize",
        spec_metric="accuracy",
        spec_direction="minimize",
        explicit_target_metric=True,
        explicit_target_direction=True,
        strict_competition_metric=True,
        competition_override={},
    )

    assert decision.target_metric == "balanced_accuracy"
    assert decision.target_direction == "maximize"
    assert decision.messages == (
        "[yellow]note[/yellow]: strict competition metric mode is enabled, "
        "but keeping explicit target_metric 'balanced_accuracy' over evaluation_spec metric 'accuracy'.",
        "[yellow]note[/yellow]: strict competition metric mode is enabled, "
        "but keeping explicit target_direction 'maximize' over evaluation_spec direction 'minimize'.",
    )


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


def test_build_evaluation_contract_applies_competition_override() -> None:
    contract = build_evaluation_contract(
        slug="deep-past-initiative-machine-translation",
        eval_spec={},
        target_metric="accuracy",
        target_direction="maximize",
        split_strategy="kfold",
    )

    assert contract["expected_metric"] == "geometric mean of the bleu and the chrf++ scores"
    assert contract["expected_direction"] == "maximize"
    assert contract["expected_split_strategy"] == "group_kfold"


def test_build_evaluation_contract_uses_faithfulness_overrides_and_full_dataset_defaults() -> None:
    assert "urban-flood-modelling" in FULL_DATASET_REQUIRED_COMPETITIONS

    contract = build_evaluation_contract(
        slug="urban-flood-modelling",
        eval_spec={
            "faithfulness": {
                "accepted_score_sources": ["cv"],
                "require_metric_match": False,
                "require_full_dataset": False,
            }
        },
        target_metric="rmse",
        target_direction="minimize",
        split_strategy="group_kfold",
    )
    default_contract = build_evaluation_contract(
        slug="urban-flood-modelling",
        eval_spec={},
        target_metric="rmse",
        target_direction="minimize",
        split_strategy="group_kfold",
    )

    assert contract["accepted_score_sources"] == ["cv"]
    assert contract["require_metric_match"] is False
    assert contract["require_full_dataset"] is False
    assert default_contract["require_full_dataset"] is True


def test_normalize_eval_seeds_deduplicates_and_uses_defaults() -> None:
    defaults = (42, 2024, 777)

    assert normalize_eval_seeds([1, "x", 1, 2], default_seeds=defaults) == [1, 2]
    assert normalize_eval_seeds(None, fallback=[3, 3], default_seeds=defaults) == [3]
    assert normalize_eval_seeds(None, default_seeds=defaults) == [42, 2024, 777]


def test_default_eval_seed_helpers_use_shared_policy_defaults() -> None:
    assert DEFAULT_EVAL_SEEDS == (42, 2024, 777)
    assert DEFAULT_EVAL_REPEATS == 2
    assert EVAL_REPEAT_SEED_OFFSET == 1009
    assert normalize_default_eval_seeds(None) == [42, 2024, 777]
    assert normalize_default_eval_repeats(None) == 2
    assert expanded_default_eval_seeds(base_seeds=[1], repeats=2) == [1, 1010]


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
