from __future__ import annotations

import json
from pathlib import Path

from kagglebot.campaign import CampaignCandidate, candidate_registry_path, upsert_candidate
from kagglebot.top1_campaign import (
    BLEND_REPORT_FILENAME,
    PORTFOLIO_PLAN_FILENAME,
    build_blend_report,
    build_candidate_portfolio_plan,
    build_reference_reproduction_report,
    private_robustness_score,
    reference_reproduction_report_path,
)


def test_reference_reproduction_gate_blocks_novelty_until_reference_passes(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    candidate = CampaignCandidate(
        candidate_id="run-1-i001-strong_single",
        category="strong_single",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.79,
    )
    report = build_reference_reproduction_report(
        context_dir=context_dir,
        campaign_state={"campaign_id": "demo-run-1", "historical_best_score": 0.82, "champion_score": 0.82},
        method_registry={
            "methods": [
                {
                    "method_id": "ref-top-notebook",
                    "status": "active",
                    "candidate_category": "reference_reproduction",
                    "source_type": "competition_specific",
                    "source_ids": ["source-1:https://www.kaggle.com/code/demo/ref"],
                }
            ]
        },
        direction="maximize",
        current_candidate=candidate,
    )

    assert report["status"] == "pending"
    assert report["blocks_novelty"] is True
    assert report["gate_reason"] == "reference_reproduction_required_before_novelty"
    assert json.loads(reference_reproduction_report_path(context_dir).read_text(encoding="utf-8"))["status"] == (
        "pending"
    )


def test_reference_reproduction_below_historical_best_is_blocked(tmp_path: Path) -> None:
    candidate = CampaignCandidate(
        candidate_id="run-1-i001-reference_reproduction",
        category="reference_reproduction",
        run_id="run-1",
        iteration=1,
        direction="minimize",
        offline_score=10.2,
    )
    report = build_reference_reproduction_report(
        context_dir=tmp_path / "context",
        campaign_state={"historical_best_score": 9.6, "champion_score": 9.6},
        method_registry={"methods": []},
        direction="minimize",
        current_candidate=candidate,
    )

    assert report["status"] == "blocked"
    assert report["gate_reason"] == "reference_reproduction_below_campaign_baseline"
    assert report["baseline_delta"] < 0


def test_portfolio_plan_requires_all_candidate_categories_and_blend_report(tmp_path: Path) -> None:
    registry_path = candidate_registry_path(tmp_path / "context")
    first = CampaignCandidate(
        candidate_id="run-1-i001-strong_single",
        category="strong_single",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.81,
        fold_scores=[0.80, 0.82, 0.81],
        prediction_correlation={"run-1-i001-feature_variant": 0.72},
    )
    second = CampaignCandidate(
        candidate_id="run-1-i001-feature_variant",
        category="feature_variant",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.805,
    )
    upsert_candidate(registry_path, first)
    upsert_candidate(registry_path, second)

    iter_dir = tmp_path / "runs" / "run-1" / "iter-1"
    plan = build_candidate_portfolio_plan(
        iter_dir=iter_dir,
        registry_path=registry_path,
        method_registry={
            "methods": [
                {
                    "method_id": "validation-redesign-from-public-regression",
                    "status": "active",
                    "candidate_category": "validation_variant",
                }
            ]
        },
        validation_registry={"priority": True, "active_profile": "group_or_proxy_cv"},
        campaign_state={"campaign_id": "demo-run-1", "offline_online_correlation": 0.1},
        run_id="run-1",
        iteration=1,
        direction="maximize",
    )
    blend = build_blend_report(
        iter_dir=iter_dir,
        registry_path=registry_path,
        campaign_state={"offline_online_correlation": 0.1},
        validation_registry={"priority": True},
        direction="maximize",
    )

    assert plan["required_categories"] == [
        "reference_reproduction",
        "strong_single",
        "feature_variant",
        "validation_variant",
        "blend",
    ]
    assert "reference_reproduction" in plan["missing_categories"]
    assert plan["active_validation_profile"] == "group_or_proxy_cv"
    assert plan["blend_pairs"] == [{"left": first.candidate_id, "right": second.candidate_id}]
    assert (iter_dir / PORTFOLIO_PLAN_FILENAME).exists()
    assert blend["status"] == "deferred_for_validation_redesign"
    assert (iter_dir / BLEND_REPORT_FILENAME).exists()


def test_private_robustness_penalizes_fold_variance_and_public_mismatch() -> None:
    stable = CampaignCandidate(
        candidate_id="stable",
        category="strong_single",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.82,
        public_score=0.82,
        fold_scores=[0.819, 0.821, 0.820],
    )
    noisy = CampaignCandidate(
        candidate_id="noisy",
        category="strong_single",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.82,
        public_score=0.70,
        fold_scores=[0.70, 0.90, 0.82],
        prediction_correlation={"other": 0.99},
    )

    assert private_robustness_score(stable, campaign_state={}) > private_robustness_score(noisy, campaign_state={})
