from __future__ import annotations

from kagglebot.iteration_signals import (
    apply_iteration_repair_signal_policy,
    detect_online_mismatch_signal,
    extract_missing_ensemble_signal,
    extract_orig_proba_signal,
    extract_original_data_unused_signal,
    extract_pseudo_label_failure_signal,
    extract_same_family_plateau_signal,
    requires_tabular_multi_family_policy,
)


def _upgrade_mode(current: str, requested: str) -> str:
    order = ["minor_tuning", "moderate_update", "major_overhaul", "validation_redesign"]
    return requested if order.index(requested) > order.index(current) else current


def test_iteration_signals_detect_repair_targets() -> None:
    orig_signal = extract_orig_proba_signal(
        {
            "original_data_found": False,
            "orig_proba_feature_status": "constant_fallback",
            "orig_proba_constant_cols": ["ORIG_proba_a", "ORIG_proba_b"],
        }
    )
    assert orig_signal is not None
    assert len(orig_signal["constant_cols"]) == 2
    assert "context/reference_inputs_manifest.json" in str(orig_signal["note"])

    pseudo_signal = extract_pseudo_label_failure_signal(
        kernel_metrics_payload={"pseudo_label": {"accepted_folds": "0", "total_folds": "5.0"}},
        diagnostics_text="Pseudo-label result: 0/5 accepted folds.",
    )
    assert pseudo_signal is not None
    assert pseudo_signal["accepted"] == 0
    assert pseudo_signal["total"] == 5

    missing_ensemble = extract_missing_ensemble_signal(
        {
            "model_families": ["xgboost", "catboost"],
            "blend_method": "single",
            "component_models": ["xgb_only"],
        }
    )
    assert missing_ensemble is not None
    assert "weighted or rank OOF blend" in str(missing_ensemble["note"])

    original_data_unused = extract_original_data_unused_signal(
        kernel_metrics_payload={"original_data_found": False, "external_data_used": False},
        reference_inputs_manifest_payload={
            "required_datasets": ["alice/original-data"],
            "reference_notebooks": [{"staged_sources": [{"kind": "dataset", "ref": "alice/original-data"}]}],
        },
    )
    assert original_data_unused is not None
    assert "staged but the kernel did not use them" in str(original_data_unused["note"])

    same_family = extract_same_family_plateau_signal(
        {
            "model_families": ["xgboost"],
            "pipelines": [{"name": "xgb_a"}, {"name": "xgb_b"}],
            "selected_pipeline": "xgb_b",
        }
    )
    assert same_family is not None
    assert "same-family plateau" in str(same_family["note"])


def test_apply_iteration_repair_signal_policy_builds_notes_errors_and_major_overhaul() -> None:
    decision = apply_iteration_repair_signal_policy(
        iteration=2,
        orig_proba_signal={"note": "orig fallback"},
        original_data_unused_signal=None,
        pseudo_label_signal={"accepted": 0, "total": 5, "note": "pseudo failed"},
        missing_ensemble_signal={"note": "missing blend"},
        same_family_plateau_signal={"note": "same family"},
        subgroup_collapse_signal={"note": "subgroup collapsed"},
        online_mismatch_signal=None,
        online_history_regression_signal=None,
        minimum_improvement_mode="minor_tuning",
        minimum_improvement_reason=None,
        force_major_overhaul=False,
        forced_major_overhaul_reason=None,
        prefer_validation_redesign=False,
        upgrade_improvement_mode=_upgrade_mode,
    )

    assert decision.minimum_improvement_mode == "moderate_update"
    assert decision.force_major_overhaul is True
    assert decision.forced_major_overhaul_reason == "missing blend same family"
    assert decision.forced_validation_redesign_reason is None
    assert decision.extra_policy_notes == [
        "orig fallback",
        "pseudo failed",
        "missing blend",
        "same family",
        "subgroup collapsed",
    ]
    assert [item["error_message"] for item in decision.loop_signal_errors] == [
        "ORIG_proba external signal fell back to constants because original data was unavailable.",
        "Pseudo-labeling yielded 0/5 accepted folds or candidates.",
    ]
    assert [item["why_poor"] for item in decision.loop_signal_problems] == [
        "missing blend",
        "same family",
        "subgroup collapsed",
    ]
    assert decision.repair_signals is not None
    assert decision.next_iteration_policy["forced_improvement_mode"] == "major_overhaul"
    assert decision.next_iteration_policy["forced_improvement_reason"] == "missing blend same family"


def test_apply_iteration_repair_signal_policy_prefers_validation_redesign_for_online_mismatch() -> None:
    decision = apply_iteration_repair_signal_policy(
        iteration=3,
        orig_proba_signal=None,
        original_data_unused_signal=None,
        pseudo_label_signal=None,
        missing_ensemble_signal=None,
        same_family_plateau_signal=None,
        subgroup_collapse_signal=None,
        online_mismatch_signal={"note": "online mismatch"},
        online_history_regression_signal={"note": "online history regression"},
        minimum_improvement_mode="minor_tuning",
        minimum_improvement_reason="medal policy",
        force_major_overhaul=False,
        forced_major_overhaul_reason=None,
        prefer_validation_redesign=True,
        upgrade_improvement_mode=_upgrade_mode,
    )

    assert decision.minimum_improvement_mode == "validation_redesign"
    assert decision.force_major_overhaul is False
    assert decision.forced_major_overhaul_reason is None
    assert decision.forced_validation_redesign_reason == "online history regression"
    assert decision.next_iteration_policy == {
        "minimum_improvement_mode": "validation_redesign",
        "minimum_improvement_reason": "medal policy",
        "forced_improvement_mode": "validation_redesign",
        "forced_improvement_reason": "online history regression",
        "extra_policy_notes": ["online mismatch", "online history regression"],
    }


def test_detect_online_mismatch_signal_requires_offline_only_improvement() -> None:
    mismatch = detect_online_mismatch_signal(
        previous_best_offline=0.91,
        current_offline=0.92,
        previous_best_online=0.905,
        current_online=0.901,
        direction="maximize",
    )
    assert mismatch is not None
    assert "public leaderboard regressed" in str(mismatch["note"]).lower()

    assert (
        detect_online_mismatch_signal(
            previous_best_offline=0.91,
            current_offline=0.90,
            previous_best_online=0.905,
            current_online=0.901,
            direction="maximize",
        )
        is None
    )


def test_requires_tabular_multi_family_policy_for_large_binary_mixed_categoricals() -> None:
    assert requires_tabular_multi_family_policy(
        {
            "modality": "tabular",
            "task": "binary",
            "train_rows": "5,500",
            "categorical_columns": ["a", "b", "c"],
        }
    )
    assert not requires_tabular_multi_family_policy(
        {
            "modality": "tabular",
            "task": "binary",
            "train_rows": 200,
            "categorical_columns": ["a", "b", "c"],
        }
    )
