from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from kagglebot.campaign import CampaignCandidate, SubmissionAllocation

EXPERIMENT_GRAPH_FILENAME = "experiment_graph.json"
ALLOCATOR_DECISION_FILENAME = "allocator_decision.json"
CAMPAIGN_OUTCOMES_FILENAME = "campaign_outcomes.jsonl"

PortfolioExecutionMode = Literal["off", "serial", "parallel", "budgeted"]
ExperimentNodeType = Literal[
    "reference",
    "split_probe",
    "feature_probe",
    "model_candidate",
    "blend",
    "calibration_submit",
    "final_candidate",
]


@dataclass(frozen=True)
class ExperimentNode:
    node_id: str
    node_type: ExperimentNodeType
    candidate_id: str | None
    category: str
    status: str
    dependencies: list[str] = field(default_factory=list)
    method_id: str | None = None
    validation_profile_id: str | None = None
    expected_outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    runtime: dict[str, object] = field(default_factory=dict)
    risks: dict[str, object] = field(default_factory=dict)
    evidence: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def normalize_portfolio_execution(value: str | None) -> PortfolioExecutionMode:
    normalized = str(value or "serial").strip().lower()
    if normalized in {"off", "none", "disabled"}:
        return "off"
    if normalized in {"serial", "on", "auto"}:
        return "serial"
    if normalized == "parallel":
        return "parallel"
    if normalized in {"budgeted", "budget", "value"}:
        return "budgeted"
    raise ValueError("portfolio_execution must be one of: off, serial, parallel, budgeted")


def experiment_graph_path(context_dir: Path) -> Path:
    return context_dir / EXPERIMENT_GRAPH_FILENAME


def build_experiment_graph(
    *,
    context_dir: Path,
    iter_dir: Path,
    run_id: str,
    iteration: int,
    portfolio_execution: str | None,
    portfolio_plan: dict[str, object] | None,
    reference_report: dict[str, object] | None,
    blend_report: dict[str, object] | None,
    validation_registry: dict[str, object] | None,
    method_registry: dict[str, object] | None,
    campaign_state: dict[str, object],
) -> dict[str, object]:
    mode = normalize_portfolio_execution(portfolio_execution)
    validation_priority = bool((validation_registry or {}).get("priority"))
    reference_blocks = bool((reference_report or {}).get("blocks_novelty"))
    active_validation_profile = str(
        (portfolio_plan or {}).get("active_validation_profile")
        or (validation_registry or {}).get("active_profile")
        or "default_cv"
    )
    method_adapters = _method_adapters(method_registry or {})
    candidate_items = (portfolio_plan or {}).get("candidates")
    if not isinstance(candidate_items, list):
        candidate_items = []
    validation_satisfied = any(
        isinstance(item, dict)
        and str(item.get("category") or "") == "validation_variant"
        and str(item.get("status") or "") == "available"
        for item in candidate_items
    )

    nodes: list[ExperimentNode] = []
    reference_node_id = "reference:reference_reproduction"
    validation_node_id = "split_probe:validation_variant"
    for item in candidate_items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "strong_single")
        node_type = _node_type_for_category(category)
        node_id = f"{node_type}:{category}"
        dependencies = _dependencies_for_category(
            category=category,
            reference_node_id=reference_node_id,
            validation_node_id=validation_node_id,
            reference_blocks=reference_blocks,
            validation_priority=validation_priority,
        )
        status = _node_status(
            mode=mode,
            category=category,
            portfolio_status=str(item.get("status") or "planned"),
            reference_blocks=reference_blocks,
            validation_priority=validation_priority,
            validation_satisfied=validation_satisfied,
        )
        method_id = _optional_str(item.get("method_id"))
        adapter = method_adapters.get(method_id or "")
        execution_priority = _node_execution_priority(
            category=category,
            item=item,
            validation_priority=validation_priority,
            reference_blocks=reference_blocks,
        )
        nodes.append(
            ExperimentNode(
                node_id=node_id,
                node_type=node_type,
                candidate_id=_optional_str(item.get("candidate_id")),
                category=category,
                status=status,
                dependencies=dependencies,
                method_id=method_id,
                validation_profile_id=_optional_str(item.get("validation_profile_id")) or active_validation_profile,
                expected_outputs={
                    "oof": str(item.get("expected_oof_path") or ""),
                    "test_prediction": str(item.get("expected_prediction_path") or ""),
                },
                metrics={
                    "offline_score": item.get("offline_score"),
                    "rank_score": item.get("rank_score"),
                    "private_robustness_score": item.get("private_robustness_score"),
                },
                runtime={
                    "execution_priority": execution_priority,
                    "budget_hint_minutes": item.get("budget_hint_minutes"),
                },
                risks={
                    "reference_blocked": reference_blocks and category != "reference_reproduction",
                    "validation_priority": validation_priority,
                    "blend_status": (blend_report or {}).get("status") if category == "blend" else None,
                },
                evidence={
                    "decision": _node_decision_label(status=status),
                    "reason": _node_decision_reason(
                        status=status,
                        category=category,
                        reference_blocks=reference_blocks,
                        validation_priority=validation_priority,
                    ),
                },
                metadata={
                    "model_family": item.get("model_family"),
                    "adapter": adapter.get("adapter") if isinstance(adapter, dict) else None,
                    "adapter_contract": adapter.get("contract") if isinstance(adapter, dict) else None,
                    "dependency_check": adapter.get("dependency_check") if isinstance(adapter, dict) else None,
                },
            )
        )

    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "iteration": iteration,
        "mode": mode,
        "campaign_id": campaign_state.get("campaign_id"),
        "active_validation_profile": active_validation_profile,
        "reference_gate": {
            "status": (reference_report or {}).get("status"),
            "blocks_novelty": reference_blocks,
            "reason": (reference_report or {}).get("gate_reason"),
        },
        "validation_priority": validation_priority,
        "blend_status": (blend_report or {}).get("status"),
        "nodes": [node.to_payload() for node in nodes],
        "ready_nodes": [node.node_id for node in nodes if node.status == "ready"],
        "blocked_nodes": [node.node_id for node in nodes if node.status == "blocked"],
        "next_action": _graph_next_action(mode=mode, nodes=nodes, reference_blocks=reference_blocks),
    }
    context_dir.mkdir(parents=True, exist_ok=True)
    iter_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=True)
    experiment_graph_path(context_dir).write_text(serialized, encoding="utf-8")
    (iter_dir / EXPERIMENT_GRAPH_FILENAME).write_text(serialized, encoding="utf-8")
    return payload


def write_allocator_decision(
    *,
    iter_dir: Path,
    candidate: CampaignCandidate | None,
    allocation: SubmissionAllocation | None,
    campaign_state: dict[str, object],
    experiment_graph: dict[str, object] | None,
) -> dict[str, object]:
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "campaign_id": campaign_state.get("campaign_id"),
        "candidate_id": candidate.candidate_id if candidate is not None else None,
        "candidate": candidate.to_payload() if candidate is not None else None,
        "allocation": allocation.to_payload() if allocation is not None else None,
        "experiment_graph_path": str(iter_dir / EXPERIMENT_GRAPH_FILENAME) if experiment_graph is not None else None,
    }
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / ALLOCATOR_DECISION_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return payload


def append_campaign_outcome(
    *,
    context_dir: Path,
    run_id: str,
    iteration: int,
    phase: str,
    candidate: CampaignCandidate | None,
    allocation: SubmissionAllocation | None,
    campaign_state: dict[str, object],
    experiment_graph: dict[str, object] | None,
) -> dict[str, object]:
    payload = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "iteration": iteration,
        "phase": phase,
        "campaign_id": campaign_state.get("campaign_id"),
        "candidate_id": candidate.candidate_id if candidate is not None else None,
        "candidate_category": candidate.category if candidate is not None else None,
        "method_id": candidate.method_id if candidate is not None else None,
        "validation_profile_id": candidate.validation_profile_id if candidate is not None else None,
        "offline_score": candidate.offline_score if candidate is not None else None,
        "public_score": candidate.public_score if candidate is not None else None,
        "submitted": candidate.submitted if candidate is not None else None,
        "allocation": allocation.to_payload() if allocation is not None else None,
        "experiment_mode": (experiment_graph or {}).get("mode") if isinstance(experiment_graph, dict) else None,
        "active_validation_profile": campaign_state.get("active_validation_profile"),
        "offline_online_correlation": campaign_state.get("offline_online_correlation"),
        "top1_gap": campaign_state.get("top1_gap"),
    }
    context_dir.mkdir(parents=True, exist_ok=True)
    with (context_dir / CAMPAIGN_OUTCOMES_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return payload


def _node_type_for_category(category: str) -> ExperimentNodeType:
    if category == "reference_reproduction":
        return "reference"
    if category == "validation_variant":
        return "split_probe"
    if category == "feature_variant":
        return "feature_probe"
    if category == "blend":
        return "blend"
    if category == "calibration":
        return "calibration_submit"
    return "model_candidate"


def _dependencies_for_category(
    *,
    category: str,
    reference_node_id: str,
    validation_node_id: str,
    reference_blocks: bool,
    validation_priority: bool,
) -> list[str]:
    dependencies: list[str] = []
    if category != "reference_reproduction":
        dependencies.append(reference_node_id)
    if validation_priority and category not in {"reference_reproduction", "validation_variant", "calibration"}:
        dependencies.append(validation_node_id)
    if category == "blend":
        dependencies.extend(["model_candidate:strong_single", "feature_probe:feature_variant"])
    if not reference_blocks and category == "validation_variant":
        return []
    return list(dict.fromkeys(dependencies))


def _node_status(
    *,
    mode: PortfolioExecutionMode,
    category: str,
    portfolio_status: str,
    reference_blocks: bool,
    validation_priority: bool,
    validation_satisfied: bool,
) -> str:
    if mode == "off":
        return "disabled"
    if reference_blocks and category != "reference_reproduction":
        return "blocked"
    if (
        validation_priority
        and not validation_satisfied
        and category not in {"reference_reproduction", "validation_variant", "calibration"}
    ):
        return "blocked"
    if portfolio_status == "available":
        return "completed"
    return "ready"


def _graph_next_action(*, mode: PortfolioExecutionMode, nodes: list[ExperimentNode], reference_blocks: bool) -> str:
    if mode == "off":
        return "Portfolio execution is disabled; continue legacy single-candidate loop."
    if reference_blocks:
        return "Run or diagnose the reference reproduction node before novelty."
    ready = [node.node_id for node in nodes if node.status == "ready"]
    if ready:
        if mode == "budgeted":
            return "Run the highest expected-value ready node under the candidate budget: " + ", ".join(ready)
        return "Run ready experiment nodes in serial order: " + ", ".join(ready)
    blocked = [node.node_id for node in nodes if node.status == "blocked"]
    if blocked:
        return "Resolve blockers before running: " + ", ".join(blocked)
    return "No pending experiment nodes; promote final candidate or refresh method scout."


def _node_execution_priority(
    *,
    category: str,
    item: dict[str, object],
    validation_priority: bool,
    reference_blocks: bool,
) -> float:
    base_by_category = {
        "reference_reproduction": 1.0,
        "validation_variant": 0.92 if validation_priority else 0.62,
        "strong_single": 0.68,
        "feature_variant": 0.58,
        "blend": 0.72,
        "calibration": 0.55,
    }
    base = base_by_category.get(category, 0.5)
    if reference_blocks and category != "reference_reproduction":
        base -= 0.45
    rank_score = _to_float(item.get("rank_score"))
    robustness = _to_float(item.get("private_robustness_score"))
    if rank_score is not None:
        base += min(0.2, max(-0.1, rank_score * 0.1))
    if robustness is not None:
        base += min(0.15, max(-0.1, (robustness - 0.5) * 0.2))
    return round(max(0.0, min(1.0, base)), 6)


def _node_decision_label(*, status: str) -> str:
    if status == "completed":
        return "adopted"
    if status == "blocked":
        return "rejected"
    if status == "ready":
        return "pending_execution"
    return status


def _node_decision_reason(
    *,
    status: str,
    category: str,
    reference_blocks: bool,
    validation_priority: bool,
) -> str:
    if status == "blocked" and reference_blocks and category != "reference_reproduction":
        return "reference_reproduction_gate_blocks_novelty"
    if status == "blocked" and validation_priority and category not in {"reference_reproduction", "validation_variant"}:
        return "validation_redesign_required_before_model_search"
    if status == "completed":
        return "candidate_already_available"
    if status == "ready":
        return "candidate_ready_for_execution"
    return "candidate_not_selected"


def _method_adapters(method_registry: dict[str, object]) -> dict[str, dict[str, object]]:
    methods = method_registry.get("methods")
    if not isinstance(methods, list):
        return {}
    adapters: dict[str, dict[str, object]] = {}
    for item in methods:
        if not isinstance(item, dict):
            continue
        method_id = _optional_str(item.get("method_id"))
        adapter = item.get("implementation_adapter")
        if method_id and isinstance(adapter, dict):
            enriched = dict(adapter)
            dependency_check = item.get("dependency_check")
            if isinstance(dependency_check, dict):
                enriched["dependency_check"] = dependency_check
            adapters[method_id] = enriched
    return adapters


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed
