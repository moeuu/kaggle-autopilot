from __future__ import annotations

import json
from pathlib import Path

from kagglebot.campaign import CampaignCandidate, SubmissionAllocation
from kagglebot.experiment_graph import (
    ALLOCATOR_DECISION_FILENAME,
    CAMPAIGN_OUTCOMES_FILENAME,
    EXPERIMENT_GRAPH_FILENAME,
    append_campaign_outcome,
    build_experiment_graph,
    experiment_graph_path,
    normalize_portfolio_execution,
    write_allocator_decision,
)


def test_experiment_graph_blocks_novelty_when_reference_gate_blocks(tmp_path: Path) -> None:
    iter_dir = tmp_path / "runs" / "run-1" / "iter-1"
    graph = build_experiment_graph(
        context_dir=tmp_path / "context",
        iter_dir=iter_dir,
        run_id="run-1",
        iteration=1,
        portfolio_execution="serial",
        portfolio_plan={
            "active_validation_profile": "default_cv",
            "candidates": [
                {
                    "candidate_id": "run-1-i001-reference_reproduction",
                    "category": "reference_reproduction",
                    "status": "planned",
                    "method_id": "ref",
                },
                {
                    "candidate_id": "run-1-i001-strong_single",
                    "category": "strong_single",
                    "status": "planned",
                    "method_id": "gbdt",
                },
            ],
        },
        reference_report={"status": "pending", "blocks_novelty": True, "gate_reason": "reference_required"},
        blend_report={"status": "insufficient_diverse_candidates"},
        validation_registry={"priority": False, "active_profile": "default_cv"},
        method_registry={
            "methods": [
                {
                    "method_id": "gbdt",
                    "implementation_adapter": {"adapter": "tabular_model_candidate", "contract": "emit OOF"},
                }
            ]
        },
        campaign_state={"campaign_id": "demo-run-1"},
    )

    assert graph["ready_nodes"] == ["reference:reference_reproduction"]
    assert graph["blocked_nodes"] == ["model_candidate:strong_single"]
    assert "reference reproduction" in str(graph["next_action"]).lower()
    first_node = graph["nodes"][0]
    assert first_node["evidence"]["decision"] == "pending_execution"
    assert "execution_priority" in first_node["runtime"]
    assert experiment_graph_path(tmp_path / "context").exists()
    assert (iter_dir / EXPERIMENT_GRAPH_FILENAME).exists()


def test_experiment_graph_prioritizes_validation_before_model_nodes(tmp_path: Path) -> None:
    graph = build_experiment_graph(
        context_dir=tmp_path / "context",
        iter_dir=tmp_path / "iter",
        run_id="run-1",
        iteration=2,
        portfolio_execution="serial",
        portfolio_plan={
            "active_validation_profile": "group_or_proxy_cv",
            "candidates": [
                {"candidate_id": "val", "category": "validation_variant", "status": "planned"},
                {"candidate_id": "single", "category": "strong_single", "status": "planned"},
                {"candidate_id": "blend", "category": "blend", "status": "planned"},
            ],
        },
        reference_report={"status": "passed", "blocks_novelty": False},
        blend_report={"status": "ready"},
        validation_registry={"priority": True, "active_profile": "group_or_proxy_cv"},
        method_registry={},
        campaign_state={"campaign_id": "demo-run-1"},
    )

    assert graph["ready_nodes"] == ["split_probe:validation_variant"]
    assert "model_candidate:strong_single" in graph["blocked_nodes"]
    assert "blend:blend" in graph["blocked_nodes"]


def test_experiment_graph_unblocks_models_after_validation_candidate_available(tmp_path: Path) -> None:
    graph = build_experiment_graph(
        context_dir=tmp_path / "context",
        iter_dir=tmp_path / "iter",
        run_id="run-1",
        iteration=3,
        portfolio_execution="serial",
        portfolio_plan={
            "active_validation_profile": "group_or_proxy_cv",
            "candidates": [
                {"candidate_id": "val", "category": "validation_variant", "status": "available"},
                {"candidate_id": "single", "category": "strong_single", "status": "planned"},
            ],
        },
        reference_report={"status": "passed", "blocks_novelty": False},
        blend_report={"status": "insufficient_diverse_candidates"},
        validation_registry={"priority": True, "active_profile": "group_or_proxy_cv"},
        method_registry={},
        campaign_state={"campaign_id": "demo-run-1"},
    )

    assert "model_candidate:strong_single" in graph["ready_nodes"]


def test_allocator_decision_artifact_records_allocation(tmp_path: Path) -> None:
    candidate = CampaignCandidate(
        candidate_id="run-1-i001-strong_single",
        category="strong_single",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.83,
    )
    allocation = SubmissionAllocation(
        allow_submit=True,
        reason="selected",
        allocation_score=0.4,
        information_value=0.5,
    )

    payload = write_allocator_decision(
        iter_dir=tmp_path / "iter",
        candidate=candidate,
        allocation=allocation,
        campaign_state={"campaign_id": "demo-run-1"},
        experiment_graph={"version": 1},
    )

    assert payload["candidate_id"] == candidate.candidate_id
    loaded = json.loads((tmp_path / "iter" / ALLOCATOR_DECISION_FILENAME).read_text(encoding="utf-8"))
    assert loaded["allocation"]["information_value"] == 0.5


def test_campaign_outcome_jsonl_records_method_and_validation_profile(tmp_path: Path) -> None:
    candidate = CampaignCandidate(
        candidate_id="run-1-i001-validation_variant",
        category="validation_variant",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.83,
        public_score=0.82,
        method_id="validation-redesign",
        validation_profile_id="group_or_proxy_cv",
        submitted=True,
    )
    append_campaign_outcome(
        context_dir=tmp_path / "context",
        run_id="run-1",
        iteration=1,
        phase="post_submit",
        candidate=candidate,
        allocation=None,
        campaign_state={"campaign_id": "demo-run-1", "top1_gap": -0.01},
        experiment_graph={"mode": "serial"},
    )

    line = (tmp_path / "context" / CAMPAIGN_OUTCOMES_FILENAME).read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["method_id"] == "validation-redesign"
    assert payload["validation_profile_id"] == "group_or_proxy_cv"
    assert payload["phase"] == "post_submit"


def test_normalize_portfolio_execution_modes() -> None:
    assert normalize_portfolio_execution(None) == "serial"
    assert normalize_portfolio_execution("off") == "off"
    assert normalize_portfolio_execution("parallel") == "parallel"
    assert normalize_portfolio_execution("budgeted") == "budgeted"


def test_budgeted_graph_prioritizes_validation_node_when_validation_is_priority(tmp_path: Path) -> None:
    graph = build_experiment_graph(
        context_dir=tmp_path / "context",
        iter_dir=tmp_path / "iter",
        run_id="run-1",
        iteration=1,
        portfolio_execution="budgeted",
        portfolio_plan={
            "active_validation_profile": "group_or_proxy_cv",
            "candidates": [
                {"candidate_id": "val", "category": "validation_variant", "status": "planned"},
                {"candidate_id": "single", "category": "strong_single", "status": "planned"},
            ],
        },
        reference_report={"status": "passed", "blocks_novelty": False},
        blend_report={"status": "ready"},
        validation_registry={"priority": False, "active_profile": "group_or_proxy_cv"},
        method_registry={},
        campaign_state={"campaign_id": "demo-run-1"},
    )

    priorities = {node["node_id"]: node["runtime"]["execution_priority"] for node in graph["nodes"]}
    assert graph["mode"] == "budgeted"
    assert priorities["model_candidate:strong_single"] > priorities["split_probe:validation_variant"]
