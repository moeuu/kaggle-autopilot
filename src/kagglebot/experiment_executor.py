from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.experiment_graph import EXPERIMENT_GRAPH_FILENAME, normalize_portfolio_execution
from kagglebot.json_utils import write_json_object
from kagglebot.runners.base import CandidateRunResult, CandidateRunSpec, RunContext, Runner
from kagglebot.scalar_utils import optional_str as _optional_str

GRAPH_EXECUTION_REPORT_FILENAME = "graph_execution_report.json"


@dataclass(frozen=True)
class GraphExecutionResult:
    mode: str
    status: str
    selected_nodes: list[str]
    completed_nodes: list[str]
    failed_nodes: list[str]
    skipped_nodes: list[str]
    report_path: Path
    graph_path: Path

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["report_path"] = str(self.report_path)
        payload["graph_path"] = str(self.graph_path)
        return payload


def execute_experiment_graph(
    *,
    graph: dict[str, object],
    context: RunContext,
    runner: Runner,
    iter_dir: Path,
) -> GraphExecutionResult:
    mode = normalize_portfolio_execution(str(graph.get("mode") or "serial"))
    graph_path = iter_dir / EXPERIMENT_GRAPH_FILENAME
    report_path = iter_dir / GRAPH_EXECUTION_REPORT_FILENAME
    if mode == "off":
        result = GraphExecutionResult(
            mode=mode,
            status="disabled",
            selected_nodes=[],
            completed_nodes=[],
            failed_nodes=[],
            skipped_nodes=[],
            report_path=report_path,
            graph_path=graph_path,
        )
        _write_report(report_path=report_path, graph=graph, result=result, run_results=[])
        return result

    nodes = _typed_nodes(graph)
    ready_nodes = _select_ready_nodes(
        nodes=nodes,
        mode=mode,
        max_candidates=context.max_candidates_per_iteration,
    )
    specs = []
    for node in ready_nodes:
        spec = _candidate_spec_from_node(node)
        if spec is not None:
            specs.append(spec)
    run_results = runner.run_candidate_batch(context, specs) if specs else []
    by_node = {result.node_id: result for result in run_results}
    updated_nodes = []
    completed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for node in nodes:
        updated = dict(node)
        result = by_node.get(str(node.get("node_id") or ""))
        if result is None:
            if str(node.get("status") or "") == "ready":
                skipped.append(str(node.get("node_id") or ""))
            updated_nodes.append(updated)
            continue
        if result.status in {"completed", "planned", "queued"}:
            updated["status"] = result.status
            completed.append(result.node_id)
        else:
            updated["status"] = "failed"
            updated["error"] = result.error or "candidate_execution_failed"
            failed.append(result.node_id)
        updated["artifacts"] = {
            "metrics_path": str(result.metrics_path) if result.metrics_path else None,
            "oof_path": str(result.oof_path) if result.oof_path else None,
            "prediction_path": str(result.prediction_path) if result.prediction_path else None,
        }
        evidence = dict(updated.get("evidence") if isinstance(updated.get("evidence"), dict) else {})
        evidence["decision"] = "adopted" if result.status in {"completed", "planned", "queued"} else "rejected"
        evidence["reason"] = result.error or f"candidate_execution_{result.status}"
        updated["evidence"] = evidence
        updated_nodes.append(updated)

    updated_graph = dict(graph)
    updated_graph["updated_at"] = datetime.now(UTC).isoformat()
    updated_graph["nodes"] = updated_nodes
    updated_graph["ready_nodes"] = [
        str(node.get("node_id")) for node in updated_nodes if str(node.get("status") or "") == "ready"
    ]
    updated_graph["blocked_nodes"] = [
        str(node.get("node_id")) for node in updated_nodes if str(node.get("status") or "") == "blocked"
    ]
    updated_graph["completed_nodes"] = [
        str(node.get("node_id"))
        for node in updated_nodes
        if str(node.get("status") or "") in {"completed", "planned", "queued"}
    ]
    iter_dir.mkdir(parents=True, exist_ok=True)
    context.paths.context_dir.mkdir(parents=True, exist_ok=True)
    write_json_object(graph_path, updated_graph)
    write_json_object(context.paths.experiment_graph_path, updated_graph)

    status = "completed" if completed and not failed else "failed" if failed else "skipped"
    exec_result = GraphExecutionResult(
        mode=mode,
        status=status,
        selected_nodes=[str(node.get("node_id") or "") for node in ready_nodes],
        completed_nodes=completed,
        failed_nodes=failed,
        skipped_nodes=skipped,
        report_path=report_path,
        graph_path=graph_path,
    )
    _write_report(report_path=report_path, graph=updated_graph, result=exec_result, run_results=run_results)
    return exec_result


def _write_report(
    *,
    report_path: Path,
    graph: dict[str, object],
    result: GraphExecutionResult,
    run_results: list[CandidateRunResult],
) -> None:
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "graph": {
            "run_id": graph.get("run_id"),
            "iteration": graph.get("iteration"),
            "mode": graph.get("mode"),
            "campaign_id": graph.get("campaign_id"),
        },
        "result": result.to_payload(),
        "candidate_results": [
            {
                "candidate_id": item.candidate_id,
                "node_id": item.node_id,
                "status": item.status,
                "metrics_path": str(item.metrics_path) if item.metrics_path else None,
                "oof_path": str(item.oof_path) if item.oof_path else None,
                "prediction_path": str(item.prediction_path) if item.prediction_path else None,
                "error": item.error,
            }
            for item in run_results
        ],
    }
    write_json_object(report_path, payload)


def _typed_nodes(graph: dict[str, object]) -> list[dict[str, object]]:
    raw = graph.get("nodes")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _select_ready_nodes(
    *,
    nodes: list[dict[str, object]],
    mode: str,
    max_candidates: int | None = None,
) -> list[dict[str, object]]:
    ready = [node for node in nodes if str(node.get("status") or "") == "ready"]
    limit = max_candidates if max_candidates is not None and max_candidates > 0 else None
    if mode == "serial":
        return ready[:1]
    if mode == "budgeted":
        return sorted(ready, key=_node_priority, reverse=True)[: max(1, limit or 1)]
    return ready[:limit] if limit is not None else ready


def _candidate_spec_from_node(node: dict[str, object]) -> CandidateRunSpec | None:
    candidate_id = str(node.get("candidate_id") or "").strip()
    node_id = str(node.get("node_id") or "").strip()
    if not candidate_id or not node_id:
        return None
    expected_outputs = node.get("expected_outputs")
    runtime = node.get("runtime") if isinstance(node.get("runtime"), dict) else {}
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    return CandidateRunSpec(
        candidate_id=candidate_id,
        node_id=node_id,
        node_type=str(node.get("node_type") or ""),
        category=str(node.get("category") or ""),
        method_id=_optional_str(node.get("method_id")),
        validation_profile_id=_optional_str(node.get("validation_profile_id")),
        expected_outputs=expected_outputs if isinstance(expected_outputs, dict) else {},
        adapter=_optional_str(metadata.get("adapter")),
        runtime_budget=runtime if isinstance(runtime, dict) else {},
        data_contract=metadata.get("data_contract") if isinstance(metadata.get("data_contract"), dict) else {},
        metric_contract=metadata.get("metric_contract") if isinstance(metadata.get("metric_contract"), dict) else {},
        dependency_check=metadata.get("dependency_check") if isinstance(metadata.get("dependency_check"), dict) else {},
    )


def _node_priority(node: dict[str, object]) -> float:
    runtime = node.get("runtime") if isinstance(node.get("runtime"), dict) else {}
    value = runtime.get("execution_priority")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if parsed != parsed:
        return 0.0
    return parsed
