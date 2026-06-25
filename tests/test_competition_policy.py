from __future__ import annotations

import json
from pathlib import Path

from kagglebot.competition_policy import load_competition_policy
from kagglebot.paths import CompetitionPaths


def test_load_competition_policy_returns_inactive_default_for_missing_invalid_or_non_object_payload(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")

    missing = load_competition_policy(paths)
    assert missing.slug == "demo"
    assert missing.active is False

    paths.competition_policy_path.parent.mkdir(parents=True, exist_ok=True)
    paths.competition_policy_path.write_text("{", encoding="utf-8")
    invalid = load_competition_policy(paths)
    assert invalid.slug == "demo"
    assert invalid.active is False

    paths.competition_policy_path.write_text("[]", encoding="utf-8")
    non_object = load_competition_policy(paths)
    assert non_object.slug == "demo"
    assert non_object.active is False


def test_load_competition_policy_normalizes_nested_policy_values(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.competition_policy_path.parent.mkdir(parents=True, exist_ok=True)
    paths.competition_policy_path.write_text(
        json.dumps(
            {
                "slug": "from-policy",
                "archetype_tags": ["text", "", "ensemble"],
                "required_capabilities": ["requires_oof_blend"],
                "execution_hints": {"folds": 3},
                "notebook_selection": {
                    "keyword_boosts": {
                        "xgb": "2.5",
                        "bad": "nan-value",
                        "bool": True,
                        "nan": float("nan"),
                        "inf": float("inf"),
                    },
                    "required_reference_keywords": ["gold"],
                },
                "reference_inputs": {
                    "proactive": "yes",
                    "required_datasets": ["owner/dataset"],
                    "block_on_missing_required": 1,
                },
                "prompt": {
                    "ablation_groups": ["text_runtime"],
                    "min_model_families_before_stop": "2",
                    "require_oof_blend_before_stop": True,
                    "prefer_ensemble_reference": "on",
                    "extra_notes": ["keep fold artifacts"],
                },
                "repair": {"same_family_plateau_signal": True},
                "evaluation": {
                    "fallback_overrides": {"metric": "auc"},
                    "search_stop_rank_percentile": "0.01",
                },
            }
        ),
        encoding="utf-8",
    )

    policy = load_competition_policy(paths)

    assert policy.slug == "from-policy"
    assert policy.active is True
    assert policy.archetype_tags == ("text", "ensemble")
    assert policy.has_capability("requires_oof_blend") is True
    assert policy.execution_hint("folds") == 3
    assert policy.notebook_selection.keyword_boosts == {"xgb": 2.5}
    assert policy.notebook_selection.required_reference_keywords == ("gold",)
    assert policy.reference_inputs.proactive is True
    assert policy.reference_inputs.required_datasets == ("owner/dataset",)
    assert policy.reference_inputs.block_on_missing_required is True
    assert policy.prompt.ablation_groups == ("text_runtime",)
    assert policy.prompt.min_model_families_before_stop == 2
    assert policy.prompt.require_oof_blend_before_stop is True
    assert policy.prompt.prefer_ensemble_reference is True
    assert policy.prompt.extra_notes == ("keep fold artifacts",)
    assert policy.repair.same_family_plateau_signal is True
    assert policy.evaluation.fallback_overrides == {"metric": "auc"}
    assert policy.evaluation.search_stop_rank_percentile == 0.01


def test_load_competition_policy_rejects_unsafe_numeric_policy_values(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.competition_policy_path.parent.mkdir(parents=True, exist_ok=True)
    paths.competition_policy_path.write_text(
        json.dumps(
            {
                "prompt": {"min_model_families_before_stop": "2.5"},
                "evaluation": {"search_stop_rank_percentile": True},
                "notebook_selection": {"keyword_boosts": {"boost": False}},
            }
        ),
        encoding="utf-8",
    )

    policy = load_competition_policy(paths)

    assert policy.notebook_selection.keyword_boosts == {}
    assert policy.prompt.min_model_families_before_stop is None
    assert policy.evaluation.search_stop_rank_percentile is None
