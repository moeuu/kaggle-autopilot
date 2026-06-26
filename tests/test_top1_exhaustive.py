from __future__ import annotations

import json
from pathlib import Path

from kagglebot.campaign import CampaignCandidate, candidate_registry_path, upsert_candidate
from kagglebot.top1_exhaustive import (
    build_portfolio_optimizer_report,
    build_private_robustness_report,
    build_top1_exhaustion_report,
    build_win_contract,
    format_top1_public_score_message,
    normalize_top1_submit_policy,
    portfolio_optimizer_report_path,
    private_robustness_report_path,
    top1_exhaustion_report_path,
    win_contract_path,
    write_top1_public_snapshot,
)


def test_format_top1_public_score_message() -> None:
    assert format_top1_public_score_message({"score": 0.123, "source": "leaderboard"}) == (
        "[cyan]top1 public score[/cyan]: 0.123 (source: leaderboard)"
    )
    assert format_top1_public_score_message({"score": None}) == "[yellow]top1 public score[/yellow]: unavailable"
    assert format_top1_public_score_message(None) == "[yellow]top1 public score[/yellow]: unavailable"


def test_write_top1_public_snapshot_writes_json_object(tmp_path: Path) -> None:
    path = tmp_path / "top1_public.json"

    write_top1_public_snapshot(path, {"score": 0.123, "source": "leaderboard"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"score": 0.123, "source": "leaderboard"}

    write_top1_public_snapshot(path, None)

    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_win_contract_records_baselines_sources_and_guardrails(tmp_path: Path) -> None:
    contract = build_win_contract(
        context_dir=tmp_path / "context",
        slug="demo",
        direction="maximize",
        campaign_state={"campaign_id": "demo-run", "champion_score": 0.81, "top1_gap": -0.02},
        top1_info={"score": 0.83},
        submission_history={"best_score": 0.80},
        method_registry={"active_method_ids": ["gbdt", "blend"]},
        source_registry={"active_source_ids": ["source-1"]},
        validation_registry={"active_profile": "group_cv", "priority": True},
        submission_limit_per_day=5,
    )

    assert contract["win_score_contract"]["top1_score"] == 0.83
    assert contract["method_contract"]["reference_required"] is True
    assert contract["submission_contract"]["no_limit_bypass"] is True
    assert win_contract_path(tmp_path / "context").exists()


def test_private_robustness_report_penalizes_public_overfit_and_regression(tmp_path: Path) -> None:
    registry_path = candidate_registry_path(tmp_path / "context")
    upsert_candidate(
        registry_path,
        CampaignCandidate(
            candidate_id="stable",
            category="strong_single",
            run_id="run-1",
            iteration=1,
            direction="maximize",
            offline_score=0.84,
            public_score=0.835,
            validation_profile_id="group_cv",
        ),
    )
    upsert_candidate(
        registry_path,
        CampaignCandidate(
            candidate_id="overfit",
            category="strong_single",
            run_id="run-1",
            iteration=2,
            direction="maximize",
            offline_score=0.70,
            public_score=0.60,
            validation_profile_id="default_cv",
            prediction_correlation={"stable": 0.99},
        ),
    )

    report = build_private_robustness_report(
        context_dir=tmp_path / "context",
        registry_path=registry_path,
        campaign_state={"historical_best_score": 0.82, "champion_score": 0.82, "active_validation_profile": "group_cv"},
        validation_lab_report={"active_profile": "group_cv"},
        direction="maximize",
    )

    by_id = {item["candidate_id"]: item for item in report["top_candidates"]}
    assert by_id["stable"]["private_robustness_score"] > by_id["overfit"]["private_robustness_score"]
    assert "public_overfit_risk" in by_id["overfit"]["risk_flags"]
    assert "below_campaign_baseline" in by_id["overfit"]["risk_flags"]
    assert private_robustness_report_path(tmp_path / "context").exists()


def test_portfolio_optimizer_blocks_regression_and_selects_value_candidate(tmp_path: Path) -> None:
    registry_path = candidate_registry_path(tmp_path / "context")
    upsert_candidate(
        registry_path,
        CampaignCandidate(
            candidate_id="regression",
            category="strong_single",
            run_id="run-1",
            iteration=1,
            direction="maximize",
            offline_score=0.70,
        ),
    )
    upsert_candidate(
        registry_path,
        CampaignCandidate(
            candidate_id="candidate-win",
            category="blend",
            run_id="run-1",
            iteration=2,
            direction="maximize",
            offline_score=0.86,
        ),
    )

    private_report = {
        "top_candidates": [
            {"candidate_id": "candidate-win", "private_robustness_score": 0.9},
            {"candidate_id": "regression", "private_robustness_score": 0.2},
        ]
    }
    report = build_portfolio_optimizer_report(
        iter_dir=tmp_path / "iter",
        registry_path=registry_path,
        campaign_state={"direction": "maximize", "historical_best_score": 0.82, "champion_score": 0.82},
        validation_registry={"offline_online_correlation": 0.7},
        private_robustness_report=private_report,
        remaining_daily_slots=2,
        submit_policy="value_only",
        direction="maximize",
    )

    assert report["selected_candidate_id"] == "candidate-win"
    ranked = {item["candidate_id"]: item for item in report["ranked_candidates"]}
    assert ranked["regression"]["allocation"]["allow_submit"] is False
    assert ranked["regression"]["allocation"]["reason"] == "below_campaign_baseline"
    assert portfolio_optimizer_report_path(tmp_path / "iter").exists()


def test_exhaustion_report_records_missing_work_and_markdown(tmp_path: Path) -> None:
    registry_path = candidate_registry_path(tmp_path / "context")
    upsert_candidate(
        registry_path,
        CampaignCandidate(
            candidate_id="single",
            category="strong_single",
            run_id="run-1",
            iteration=1,
            direction="minimize",
            offline_score=1.0,
        ),
    )

    report = build_top1_exhaustion_report(
        context_dir=tmp_path / "context",
        run_id="run-1",
        iteration=1,
        campaign_state={"campaign_id": "demo-run", "top1_gap": 0.2},
        win_contract={"win_score_contract": {"top1_score": 0.8}},
        method_registry={"active_method_ids": ["gbdt"]},
        source_registry={"active_source_ids": []},
        validation_lab_report={"status": "disabled", "active_profile": "default_cv"},
        private_robustness_report={"risk_summary": {}},
        portfolio_optimizer_report={"decision": "no_positive_value_submission", "selected_candidate_id": None},
        experiment_graph={"blocked_nodes": ["reference:reference_reproduction"]},
    )

    assert report["exhaustion_status"] == "not_exhausted_missing_candidate_categories"
    assert "reference_reproduction" in report["missing_categories"]
    assert top1_exhaustion_report_path(tmp_path / "context").exists()
    assert top1_exhaustion_report_path(tmp_path / "context").with_suffix(".md").exists()


def test_normalize_top1_submit_policy() -> None:
    assert normalize_top1_submit_policy(None) == "value_only"
    assert normalize_top1_submit_policy("calibration") == "calibration"
    assert normalize_top1_submit_policy("final_lock") == "final_lock"


def test_exhaustive_artifacts_are_json_readable(tmp_path: Path) -> None:
    build_win_contract(
        context_dir=tmp_path / "context",
        slug="demo",
        direction="minimize",
        campaign_state={},
        top1_info={},
        submission_history={},
        method_registry={},
        source_registry={},
        validation_registry={},
    )
    loaded = json.loads(win_contract_path(tmp_path / "context").read_text(encoding="utf-8"))
    assert loaded["slug"] == "demo"
