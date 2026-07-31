from __future__ import annotations

import json

import numpy as np
import pytest

from kagglebot.eval import EvaluationReport
from kagglebot.iteration_metrics import (
    append_run_evaluation_report,
    build_eval_data_cache_fallback,
    build_final_metrics_payload,
    build_iteration_record_kwargs,
    build_metrics_payload,
    build_noise_guard_payload,
    build_rank_guard_payload,
    build_regression_guard_payload,
    build_split_index_fingerprints,
    enrich_evaluation_report_with_kernel_provenance,
    evaluation_to_payload,
    extract_fold_scores_for_report,
    iteration_metrics_allow_submit,
    resolve_iteration_submit_phase_completion,
    resume_best_kernel_metric_score,
    resume_best_readiness_score,
    resume_noise_guard_state,
    write_iteration_evaluation_report,
    write_iteration_metrics_payload,
)
from kagglebot.score_sources import FROZEN_GROUPED_OOF_CONTRACT
from kagglebot.solver.evaluate import EvaluationResult


def test_evaluation_to_payload_preserves_optional_scores() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=0.42,
        std=0.03,
        train_score=0.2,
        val_score=0.43,
        fold_scores=[0.4, 0.44],
    )

    assert evaluation_to_payload(evaluation) == {
        "score_source": "cv",
        "metric": "rmse",
        "direction": "minimize",
        "value": 0.42,
        "std": 0.03,
        "fold_scores": [0.4, 0.44],
        "train_score": 0.2,
        "val_score": 0.43,
    }


def test_build_metrics_payload_includes_readiness_and_offline_sources() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.81,
        std=0.02,
        train_score=None,
        val_score=None,
        fold_scores=[0.8, 0.82],
    )
    holdout = EvaluationResult(
        score_source="holdout",
        metric="auc",
        direction="maximize",
        value=0.79,
        std=None,
        train_score=None,
        val_score=0.79,
        fold_scores=None,
    )
    report = EvaluationReport(
        metric_name="auc",
        direction="maximize",
        split_strategy="stratified_kfold",
        n_splits=5,
        seeds=[42, 2024],
        repeats=2,
        per_fold_scores=[0.8, 0.82],
        mean=0.81,
        std=0.02,
        ci_low=0.78,
        ci_high=0.84,
        drift_auc=0.55,
        readiness_score=0.77,
    )

    payload = build_metrics_payload(
        run_id="run-1",
        iteration=3,
        evaluation=evaluation,
        target_score=0.9,
        met_target=False,
        top1_info={"score": 0.95, "timestamp": "2026-01-01T00:00:00Z"},
        compute="local_gpu",
        accelerator="gpu",
        holdout_frac=0.2,
        cv_folds=5,
        seed=123,
        evaluation_by_source={"holdout": holdout},
        evaluation_report=report,
        readiness_target=0.85,
        evaluation_contract={"faithful": True},
        competition_faithfulness={"faithful": True},
        accuracy_potential={"status": "frontier"},
        timestamp=999,
    )

    assert payload["timestamp"] == 999
    assert payload["folds"] == 5
    assert payload["holdout_frac"] is None
    assert payload["offline_by_source"] == {
        "holdout": {
            "score_source": "holdout",
            "metric": "auc",
            "direction": "maximize",
            "value": 0.79,
            "std": None,
            "val_score": 0.79,
        }
    }
    assert payload["readiness"] == {
        "score": 0.77,
        "mean": 0.81,
        "std": 0.02,
        "ci_low": 0.78,
        "ci_high": 0.84,
        "target": 0.85,
        "split_strategy": "stratified_kfold",
        "n_splits": 5,
        "seeds": [42, 2024],
        "repeats": 2,
        "drift_auc": 0.55,
    }
    assert payload["evaluation_contract"] == {"faithful": True}
    assert payload["competition_faithfulness"] == {"faithful": True}
    assert payload["accuracy_potential"] == {"status": "frontier"}


def test_iteration_metrics_write_helpers_write_json_objects(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "evaluation_report.json"

    write_iteration_metrics_payload(metrics_path, {"run_id": "run-1", "iter": 2})
    write_iteration_evaluation_report(report_path, {"readiness_score": 0.91})

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"run_id": "run-1", "iter": 2}
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"readiness_score": 0.91}


def test_resume_best_kernel_metric_score_preserves_native_rubric_scale(tmp_path) -> None:
    for iteration, rubric_score in ((1, 84.0), (2, 87.5), (3, 86.0)):
        output_dir = tmp_path / f"iter-{iteration}" / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "score": 0.64 + iteration / 100.0,
                    "rubric_readiness_score_0_100": rubric_score,
                }
            ),
            encoding="utf-8",
        )

    assert resume_best_kernel_metric_score(
        run_dir=tmp_path,
        metric_key="rubric_readiness_score_0_100",
        direction="maximize",
        max_iterations=3,
    ) == pytest.approx(87.5)


def test_enrich_evaluation_report_uses_exact_kernel_metric_contract() -> None:
    display_metric = "grouped_macro_f1_moment_type (CV technical proxy; official judging is a 100-point manual rubric)"
    enriched = enrich_evaluation_report_with_kernel_provenance(
        {
            "metric_name": display_metric,
            "split_strategy": "kfold",
            "score_source": "cv",
        },
        kernel_metrics={
            "authoritative_display_metric": display_metric,
            "canonical_technical_metric": "grouped_macro_f1_moment_type",
            "score_source": "grouped_oof_cv",
            "technical_proxies": {
                "grouped_macro_f1_moment_type": {
                    "cv_type": "LeaveOneGroupOut_session_id",
                    "score_source": "grouped_oof_cv",
                }
            },
        },
    )

    assert enriched["metric_name"] == display_metric
    assert enriched["authoritative_display_metric"] == display_metric
    assert enriched["canonical_technical_metric"] == "grouped_macro_f1_moment_type"
    assert enriched["split_strategy"] == "LeaveOneGroupOut_session_id"
    assert enriched["score_source"] == "grouped_oof_cv"


def test_enrich_evaluation_report_ingests_real_grouped_fold_records() -> None:
    contract = FROZEN_GROUPED_OOF_CONTRACT
    decision = {key: value for key, value in contract.items() if key not in {"biometric_sha256", "mapping_sha256"}}
    decision.update(
        {
            "value": 0.702,
            "data_hashes": {
                "biometric": contract["biometric_sha256"],
                "mapping": contract["mapping_sha256"],
            },
            "per_session_fold_scores": {
                f"S00{index}": score for index, score in enumerate([0.61, 0.68, 0.72, 0.75, 0.77], start=1)
            },
            "seed_fold_records": [
                {"seed": seed, "fold": fold, "score": 0.6 + fold / 100}
                for seed in [42, 2024, 777]
                for fold in range(1, 6)
            ],
            "split_index_fingerprints": [
                {"seed": seed, "fold": fold, "split_index_fingerprint": f"{seed}-{fold}"}
                for seed in [42, 2024, 777]
                for fold in range(1, 6)
            ],
        }
    )
    enriched = enrich_evaluation_report_with_kernel_provenance(
        {"per_fold_scores": [0.702], "ci_low": 0.702, "ci_high": 0.702},
        kernel_metrics={
            "authoritative_display_metric": "display",
            "canonical_technical_metric": "grouped_macro_f1_moment_type",
            "model_selection_decision": decision,
            "loop_decision": {
                "metric": "rubric_readiness_score_0_100",
                "source": "offline_artifact_rubric",
                "value": 82.0,
            },
        },
    )

    assert enriched["model_selection_decision"]["trusted"] is True
    assert len(enriched["per_fold_scores"]) == 5
    assert len(enriched["seed_fold_records"]) == 15
    assert len(enriched["split_index_fingerprints"]) == 15
    assert enriched["ci_low"] < enriched["ci_high"]
    assert enriched["loop_decision"]["value"] == pytest.approx(82.0)


def test_build_iteration_record_kwargs_uses_evaluation_and_top1_score() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.81,
        std=0.02,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    assert build_iteration_record_kwargs(
        knowledge_paths="knowledge",
        run_id="run-1",
        iteration=2,
        evaluation=evaluation,
        top1_info={"score": 0.95},
        met_target=False,
    ) == {
        "knowledge_paths": "knowledge",
        "run_id": "run-1",
        "iteration": 2,
        "score_source": "cv",
        "offline_value": 0.81,
        "offline_std": 0.02,
        "top1_public_score": 0.95,
        "met_target": False,
        "git_commit": None,
    }


def test_resolve_iteration_submit_phase_completion_waits_for_required_submit() -> None:
    decision = resolve_iteration_submit_phase_completion(
        submit_enabled=True,
        allow_submit=True,
        submit_phase_required=True,
        submission_result=None,
        submit_failed_deferred=False,
    )

    assert decision.submit_allowed_by_gate is True
    assert decision.submit_phase_finished is False


def test_resolve_iteration_submit_phase_completion_finishes_when_gate_blocks_submit() -> None:
    decision = resolve_iteration_submit_phase_completion(
        submit_enabled=True,
        allow_submit=False,
        submit_phase_required=True,
        submission_result=None,
        submit_failed_deferred=False,
    )

    assert decision.submit_allowed_by_gate is False
    assert decision.submit_phase_finished is True


def test_resolve_iteration_submit_phase_completion_finishes_after_submit_result() -> None:
    decision = resolve_iteration_submit_phase_completion(
        submit_enabled=True,
        allow_submit=True,
        submit_phase_required=True,
        submission_result={"status": "submitted"},
        submit_failed_deferred=False,
    )

    assert decision.submit_allowed_by_gate is True
    assert decision.submit_phase_finished is True


def test_build_guard_payload_helpers_preserve_expected_keys() -> None:
    assert build_noise_guard_payload(
        delta_srs_vs_prev=0.01,
        noise_threshold=0.02,
        noise_limited_streak=2,
        force_major_overhaul_next=True,
    ) == {
        "delta_srs_vs_prev": 0.01,
        "threshold": 0.02,
        "streak": 2,
        "force_major_overhaul_next": True,
    }

    assert build_rank_guard_payload(
        target_medal="gold",
        target_rank_percentile=0.01,
        target_rank_met=False,
        minimum_improvement_mode="moderate_update",
        rank=12,
        total_teams=100,
        rank_percentile=0.12,
        rank_source="submission_row",
        estimated_rank=10,
        estimated_total_teams=100,
        estimated_rank_percentile=0.10,
        rank_estimate_source="leaderboard_score_estimate",
        max_percentile=0.01,
        min_teams=200,
        force_major_overhaul_next=True,
    ) == {
        "target_medal": "gold",
        "target_rank_percentile": 0.01,
        "target_rank_met": False,
        "minimum_improvement_mode": "moderate_update",
        "rank": 12,
        "total_teams": 100,
        "rank_percentile": 0.12,
        "rank_source": "submission_row",
        "estimated_rank": 10,
        "estimated_total_teams": 100,
        "estimated_rank_percentile": 0.10,
        "rank_estimate_source": "leaderboard_score_estimate",
        "max_percentile": 0.01,
        "min_teams": 200,
        "force_major_overhaul_next": True,
    }

    assert build_regression_guard_payload(
        best_score_before_iteration=0.8,
        score_drop_vs_best=0.05,
        severe_regression_detected=True,
        conservative_feature_collapse=False,
        conservative_regression_detected=True,
        first_iteration_below_code_reference=False,
        code_reference_score=0.9,
        code_reference_comparison_score=0.88,
        code_reference_delta_vs_current=-0.02,
        code_reference_forced_reproduction=True,
    ) == {
        "best_score_before_iteration": 0.8,
        "score_drop_vs_best": 0.05,
        "severe_regression_detected": True,
        "conservative_feature_collapse": False,
        "conservative_regression_detected": True,
        "first_iteration_below_code_reference": False,
        "code_reference_score": 0.9,
        "code_reference_comparison_score": 0.88,
        "code_reference_delta_vs_current": -0.02,
        "code_reference_forced_reproduction": True,
    }


def test_build_final_metrics_payload_adds_optional_sections_without_mutating_base() -> None:
    base = {"run_id": "run-1", "iter": 2}
    payload = build_final_metrics_payload(
        base_payload=base,
        loop_decision_source="readiness",
        loop_decision_value=0.81,
        noise_guard={"streak": 1},
        rank_guard={"rank": 12},
        top1_tier_offline_decision=False,
        top1_tier_by_readiness=True,
        top1_tier_by_submission=False,
        forced_submit_reason=None,
        online_score=0.73,
        campaign_payload={"state_path": "campaign.json"},
        best_score_guard={"blocked": False},
        quality_guard={"allow_submit": True},
        regression_guard={"severe_regression_detected": False},
    )

    assert base == {"run_id": "run-1", "iter": 2}
    assert payload["loop_decision"] == {
        "metric": "unknown",
        "source": "readiness",
        "value": 0.81,
        "direction": "unknown",
        "scale": "native",
    }
    assert payload["noise_guard"] == {"streak": 1}
    assert payload["rank_guard"] == {"rank": 12}
    assert payload["top1_tier"] == {
        "offline_decision": False,
        "offline_readiness": True,
        "submission_score": False,
    }
    assert payload["forced_submit_reason"] == ""
    assert payload["submission_score"] == 0.73
    assert payload["campaign"] == {"state_path": "campaign.json"}
    assert payload["best_score_guard"] == {"blocked": False}
    assert payload["quality_guard"] == {"allow_submit": True}
    assert payload["regression_guard"] == {"severe_regression_detected": False}


def test_build_final_metrics_payload_omits_absent_optional_sections() -> None:
    payload = build_final_metrics_payload(
        base_payload={},
        loop_decision_source="offline",
        loop_decision_value=0.5,
        noise_guard={},
        rank_guard={},
        top1_tier_offline_decision=False,
        top1_tier_by_readiness=False,
        top1_tier_by_submission=False,
        forced_submit_reason="manual",
        online_score=None,
        campaign_payload=None,
        best_score_guard=None,
        quality_guard={},
        regression_guard={},
    )

    assert payload["forced_submit_reason"] == "manual"
    assert "submission_score" not in payload
    assert "campaign" not in payload
    assert "best_score_guard" not in payload


def test_iteration_metrics_allow_submit_prefers_quality_guard(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"quality_guard": {"allow_submit": false}}\n', encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    assert iteration_metrics_allow_submit(path, evaluation) is False


def test_iteration_metrics_allow_submit_uses_faithfulness_when_quality_missing(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"competition_faithfulness": {"faithful": true}}\n', encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="sample",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    assert iteration_metrics_allow_submit(path, evaluation) is True


def test_iteration_metrics_allow_submit_blocks_external_label_transfer(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"external_test_label_transfer": true}\n', encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    assert iteration_metrics_allow_submit(path, evaluation) is False


def test_build_eval_data_cache_fallback_normalizes_split() -> None:
    fallback = build_eval_data_cache_fallback(split_strategy="timeseries_split", cv_folds=5)

    assert fallback == {
        "split_strategy": "timeseries_split",
        "n_splits": 5,
        "split_index_fingerprints": [],
        "drift_train_x": None,
        "drift_test_x": None,
    }


def test_extract_fold_scores_for_report_prefers_cv_fold_scores() -> None:
    evaluation = EvaluationResult(
        score_source="holdout",
        metric="auc",
        direction="maximize",
        value=0.7,
        std=None,
        train_score=None,
        val_score=0.7,
        fold_scores=None,
    )
    cv = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.79, 0.81],
    )

    assert extract_fold_scores_for_report(evaluation=evaluation, evaluation_by_source={"cv": cv}) == [0.79, 0.81]


def test_build_split_index_fingerprints_are_stable() -> None:
    class DemoSplitter:
        name = "kfold"

        @property
        def splitter(self):
            return self

        def split(self, x, y):  # noqa: ANN001
            del x, y
            yield np.array([0, 1]), np.array([2])
            yield np.array([2]), np.array([0, 1])

    fingerprints = build_split_index_fingerprints(split=DemoSplitter(), y=np.array([0, 1, 0]), seed=123)

    assert fingerprints == [
        {
            "seed": 123,
            "fold": 0,
            "train_size": 2,
            "valid_size": 1,
            "train_hash": "9d34149fbd1fe777",
            "valid_hash": "d86e8112f3c4c444",
        },
        {
            "seed": 123,
            "fold": 1,
            "train_size": 1,
            "valid_size": 2,
            "train_hash": "d86e8112f3c4c444",
            "valid_hash": "9d34149fbd1fe777",
        },
    ]


def test_append_run_evaluation_report_recovers_invalid_existing_report(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text("{", encoding="utf-8")

    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=1,
        payload={"iteration": 1, "readiness_score": 0.42},
    )

    assert report_path.read_text(encoding="utf-8").strip().startswith("{")
    best = resume_best_readiness_score(run_dir=run_dir, direction="maximize", max_iterations=1)
    assert best == 0.42


def test_resume_best_readiness_score_honors_direction_and_max_iteration(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_run_evaluation_report(run_dir=run_dir, iteration=1, payload={"iteration": 1, "readiness_score": 0.50})
    append_run_evaluation_report(run_dir=run_dir, iteration=2, payload={"iteration": 2, "readiness_score": 0.40})
    append_run_evaluation_report(run_dir=run_dir, iteration=3, payload={"iteration": 3, "readiness_score": 0.30})

    assert resume_best_readiness_score(run_dir=run_dir, direction="minimize", max_iterations=2) == 0.40
    assert resume_best_readiness_score(run_dir=run_dir, direction="maximize", max_iterations=2) == 0.50


def test_resume_best_readiness_score_ignores_non_finite_scores(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_run_evaluation_report(run_dir=run_dir, iteration=1, payload={"iteration": 1, "readiness_score": "nan"})
    append_run_evaluation_report(run_dir=run_dir, iteration=2, payload={"iteration": 2, "readiness_score": "0.45"})

    assert resume_best_readiness_score(run_dir=run_dir, direction="maximize", max_iterations=2) == 0.45


def test_resume_best_readiness_score_preserves_tie_as_best(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_run_evaluation_report(run_dir=run_dir, iteration=1, payload={"iteration": 1, "readiness_score": 0.45})
    append_run_evaluation_report(run_dir=run_dir, iteration=2, payload={"iteration": 2, "readiness_score": 0.45})

    assert resume_best_readiness_score(run_dir=run_dir, direction="maximize", max_iterations=2) == 0.45


def test_resume_noise_guard_state_counts_small_changes(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=1,
        payload={"iteration": 1, "readiness_score": 0.500, "std": 0.02},
    )
    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=2,
        payload={"iteration": 2, "readiness_score": 0.505, "std": 0.02},
    )
    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=3,
        payload={"iteration": 3, "readiness_score": 0.508, "std": 0.02},
    )

    assert resume_noise_guard_state(run_dir=run_dir, max_iterations=3) == (0.508, 2)
