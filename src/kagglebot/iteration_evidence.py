from __future__ import annotations

import difflib
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot import score_sources
from kagglebot import submission_history as _submission_history
from kagglebot.json_utils import load_json_object, load_jsonl_records, write_json_object
from kagglebot.scalar_utils import tolerant_finite_float

EVIDENCE_FILENAME = "iteration_evidence.json"
EVIDENCE_MARKDOWN_FILENAME = "iteration_evidence.md"
KERNEL_SNAPSHOT_FILENAME = "kernel_before_improvement.py"
PLAN_SNAPSHOT_FILENAME = "plan_before_improvement.json"

_TEXT_EXCERPT_CHARS = 6_000
_STRATEGY_EXCERPT_CHARS = 12_000
_DIFF_EXCERPT_CHARS = 16_000
_LOG_TAIL_BYTES = 96_000
_MAX_ERROR_SIGNATURES = 12
_MAX_CANDIDATES = 16
_TRUSTED_SCORE_SOURCES = frozenset(
    {
        "cv",
        "holdout",
        "offline",
        "oof",
        "cross_validation",
        "grouped_oof_cv",
        "offline_artifact_rubric",
    }
)
_ERROR_MARKERS = re.compile(
    r"(?:traceback|error|exception|failed|failure|out of memory|\boom\b|timed?\s*out|timeout)",
    flags=re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\b(?:sk|ghp|glpat)-[A-Za-z0-9_-]{12,}\b"),
)


class IterationEvidencePaths(Protocol):
    plan_path: Path
    kernel_source_dir: Path

    def run_dir(self, run_id: str) -> Path: ...

    def iter_dir(self, run_id: str, iteration: int) -> Path: ...


class IterationEvidenceIntegrityError(RuntimeError):
    """Persisted Oracle evidence changed while an improvement workflow was interrupted."""


@dataclass(frozen=True)
class IterationEvidenceBundle:
    path: Path
    markdown_path: Path
    sha256: str
    payload: dict[str, object]
    prompt_summary: str


def verify_iteration_evidence_bundle(bundle: IterationEvidenceBundle) -> None:
    if not bundle.path.exists() or _sha256_file(bundle.path) != bundle.sha256:
        raise IterationEvidenceIntegrityError(
            f"Frozen iteration evidence was modified during improvement implementation: {bundle.path}"
        )
    iterations = bundle.payload.get("iterations")
    if not isinstance(iterations, list):
        raise IterationEvidenceIntegrityError(f"Frozen iteration evidence has no iteration records: {bundle.path}")
    for record in iterations:
        if not isinstance(record, dict):
            continue
        for key in ("kernel_snapshot", "plan_snapshot"):
            snapshot = record.get(key)
            if not isinstance(snapshot, dict) or snapshot.get("exists") is not True:
                continue
            path = Path(str(snapshot.get("path") or ""))
            expected = str(snapshot.get("sha256") or "")
            if not path.exists() or not expected or _sha256_file(path) != expected:
                raise IterationEvidenceIntegrityError(
                    f"Frozen {key} was modified during improvement implementation: {path}"
                )


def prepare_iteration_evidence(
    *,
    paths: IterationEvidencePaths,
    slug: str,
    run_id: str,
    iteration: int,
    evaluation: object,
    target_score: float,
    current_score: float | None,
    current_score_source: str,
    delta_offline: float | None,
    pending_problem_insights: list[dict[str, object]],
    previous_submission_history: dict[str, object] | None,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> IterationEvidenceBundle:
    """Freeze the evidence used to choose the next iteration.

    A recovering Oracle workflow must consume the exact bundle it started with. This is
    especially important when Codex may already have partially edited the mutable kernel.
    """

    iter_dir = paths.iter_dir(run_id, iteration)
    iter_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = iter_dir / EVIDENCE_FILENAME
    markdown_path = iter_dir / EVIDENCE_MARKDOWN_FILENAME
    if expected_path is not None or expected_sha256 is not None:
        _verify_recovery_bundle(
            evidence_path=evidence_path,
            expected_path=expected_path,
            expected_sha256=expected_sha256,
        )
        payload = load_json_object(evidence_path)
        if payload is None:
            raise IterationEvidenceIntegrityError(f"Cannot parse persisted iteration evidence: {evidence_path}")
        prompt_summary = render_prompt_summary(payload, evidence_path=evidence_path)
        if not markdown_path.exists():
            markdown_path.write_text(render_evidence_markdown(payload), encoding="utf-8")
        return IterationEvidenceBundle(
            path=evidence_path,
            markdown_path=markdown_path,
            sha256=_sha256_file(evidence_path),
            payload=payload,
            prompt_summary=prompt_summary,
        )

    _freeze_iteration_inputs(paths=paths, iter_dir=iter_dir)
    payload = build_iteration_evidence_payload(
        paths=paths,
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        evaluation=evaluation,
        target_score=target_score,
        current_score=current_score,
        current_score_source=current_score_source,
        delta_offline=delta_offline,
        pending_problem_insights=pending_problem_insights,
        previous_submission_history=previous_submission_history,
    )
    write_json_object(evidence_path, payload, ensure_ascii=False, sort_keys=True)
    markdown_path.write_text(render_evidence_markdown(payload), encoding="utf-8")
    return IterationEvidenceBundle(
        path=evidence_path,
        markdown_path=markdown_path,
        sha256=_sha256_file(evidence_path),
        payload=payload,
        prompt_summary=render_prompt_summary(payload, evidence_path=evidence_path),
    )


def build_iteration_evidence_payload(
    *,
    paths: IterationEvidencePaths,
    slug: str,
    run_id: str,
    iteration: int,
    evaluation: object,
    target_score: float,
    current_score: float | None,
    current_score_source: str,
    delta_offline: float | None,
    pending_problem_insights: list[dict[str, object]],
    previous_submission_history: dict[str, object] | None,
) -> dict[str, object]:
    run_dir = paths.run_dir(run_id)
    attempts = load_jsonl_records(run_dir / "submit_attempts.jsonl", limit=100)
    records = [
        _build_iteration_record(
            iter_dir=paths.iter_dir(run_id, number),
            iteration=number,
            attempts=attempts,
        )
        for number in range(1, iteration + 1)
    ]
    transitions = [_build_transition(records[index - 1], records[index]) for index in range(1, len(records))]
    current_record = records[-1] if records else {}
    invocation = _build_invocation_evaluation(
        evaluation=evaluation,
        current_score=current_score,
        current_score_source=current_score_source,
        target_score=target_score,
        delta_offline=delta_offline,
    )
    evidence_gaps = _evidence_gaps(current_record=current_record, transitions=transitions)
    do_not_repeat = _do_not_repeat(transitions)
    return {
        "schema_version": 1,
        "competition": slug,
        "run_id": run_id,
        "transition": {
            "completed_iteration": iteration,
            "next_iteration": iteration + 1,
        },
        "decision_objective": {
            "metric": str(getattr(evaluation, "metric", "unknown")),
            "direction": str(getattr(evaluation, "direction", "unknown")),
            "target_score": target_score,
            "rule": (
                "Prefer accuracy. Attribute gains only across comparable, trusted evaluations; "
                "otherwise repair measurement or execution fidelity before claiming improvement."
            ),
        },
        "invocation_evaluation": invocation,
        "iterations": records,
        "transitions_observed": transitions,
        "recent_run_submission_attempts": _summarize_attempts(attempts[-12:]),
        "pending_problem_insights": _bounded_json_value(pending_problem_insights, max_items=12),
        "public_submission_history": _submission_history_summary(previous_submission_history),
        "public_score_feedback": _submission_history.build_public_score_feedback(previous_submission_history),
        "decision_requirements": {
            "evidence_gaps": evidence_gaps,
            "do_not_repeat": do_not_repeat,
            "required_oracle_contract": [
                "Cite the exact evidence fields used for the diagnosis.",
                "State one falsifiable root-cause hypothesis and why it outranks alternatives.",
                "Specify a material implementation delta from the current and previously failed approaches.",
                "Define an attribution-safe validation plan using a comparable metric/source.",
                "State expected observations, stop/rollback criteria, and a fallback.",
                "Do not call incomparable or untrusted scores an iteration improvement.",
            ],
        },
    }


def render_prompt_summary(payload: dict[str, object], *, evidence_path: Path) -> str:
    iterations = payload.get("iterations")
    transitions = payload.get("transitions_observed")
    requirements = payload.get("decision_requirements")
    lines = [
        "## Iteration Evidence Contract",
        f"- Authoritative structured evidence: {evidence_path}",
        f"- Human-readable evidence: {evidence_path.with_name(EVIDENCE_MARKDOWN_FILENAME)}",
        "- Read the full evidence file before choosing a change. Do not rely on the scalar delta alone.",
    ]
    if isinstance(iterations, list):
        for record in iterations:
            if not isinstance(record, dict):
                continue
            score = record.get("score")
            if not isinstance(score, dict):
                continue
            lines.append(
                "- iter-{iteration}: value={value}, source={source}, trusted={trusted}, metric={metric}, "
                "public={public}".format(
                    iteration=record.get("iteration"),
                    value=score.get("value"),
                    source=score.get("source"),
                    trusted=score.get("trusted"),
                    metric=score.get("metric"),
                    public=_format_optional_score(score.get("public_submission_score")),
                )
            )
    public_feedback = payload.get("public_score_feedback")
    if isinstance(public_feedback, dict):
        lines.append(
            "- Public leaderboard feedback: latest={latest}, best={best}, prior_best={prior}, "
            "improvement_delta={delta}, result={result}".format(
                latest=_format_optional_score(public_feedback.get("latest_public_score")),
                best=_format_optional_score(public_feedback.get("best_public_score")),
                prior=_format_optional_score(public_feedback.get("prior_best_public_score")),
                delta=_format_optional_score(
                    public_feedback.get("improvement_delta_vs_prior_best"),
                    signed=True,
                ),
                result=public_feedback.get("result"),
            )
        )
    if isinstance(transitions, list):
        for transition in transitions:
            if not isinstance(transition, dict):
                continue
            comparison = transition.get("score_comparison")
            comparison = comparison if isinstance(comparison, dict) else {}
            lines.append(
                f"- iter-{transition.get('from_iteration')}→iter-{transition.get('to_iteration')}: "
                f"comparable={comparison.get('comparable')}, "
                f"improvement_delta={comparison.get('improvement_delta')}, "
                f"outcome={transition.get('hypothesis_outcome')}, "
                f"material_change={transition.get('material_change_detected')}"
            )
            reasons = comparison.get("reasons")
            if isinstance(reasons, list) and reasons:
                lines.append("  comparison warnings: " + "; ".join(str(item) for item in reasons[:4]))
    if isinstance(requirements, dict):
        gaps = requirements.get("evidence_gaps")
        avoid = requirements.get("do_not_repeat")
        if isinstance(gaps, list) and gaps:
            lines.append("- Evidence gaps: " + "; ".join(str(item) for item in gaps[:6]))
        if isinstance(avoid, list) and avoid:
            lines.append("- Do not repeat: " + "; ".join(str(item) for item in avoid[:6]))
    lines.extend(
        [
            "",
            "Your response must contain an Improvement Contract with: evidence diagnosis, falsifiable hypothesis, "
            "material delta, validation/attribution plan, expected observation, stop/rollback criteria, and fallback.",
        ]
    )
    return "\n".join(lines)


def render_evidence_markdown(payload: dict[str, object]) -> str:
    lines = ["# Iteration Evidence", ""]
    transition = payload.get("transition")
    if isinstance(transition, dict):
        lines.append(
            f"Decision: iter-{transition.get('completed_iteration')} → iter-{transition.get('next_iteration')}"
        )
        lines.append("")
    lines.extend(
        [
            "## Score provenance",
            "",
            "| Iteration | Value | Source | Trusted | Metric |",
            "|---:|---:|---|---|---|",
        ]
    )
    iterations = payload.get("iterations")
    if isinstance(iterations, list):
        for record in iterations:
            if not isinstance(record, dict):
                continue
            score = record.get("score")
            score = score if isinstance(score, dict) else {}
            lines.append(
                f"| {record.get('iteration')} | {score.get('value')} | {score.get('source')} | "
                f"{score.get('trusted')} | {score.get('metric')} |"
            )
    lines.extend(["", "## Observed transitions", ""])
    transitions = payload.get("transitions_observed")
    if isinstance(transitions, list) and transitions:
        for item in transitions:
            if not isinstance(item, dict):
                continue
            comparison = item.get("score_comparison")
            comparison = comparison if isinstance(comparison, dict) else {}
            lines.extend(
                [
                    f"### iter-{item.get('from_iteration')} → iter-{item.get('to_iteration')}",
                    "",
                    f"- Comparable: {comparison.get('comparable')}",
                    f"- Improvement delta: {comparison.get('improvement_delta')}",
                    f"- Hypothesis outcome: {item.get('hypothesis_outcome')}",
                    f"- Material change detected: {item.get('material_change_detected')}",
                    f"- Reasons: {comparison.get('reasons')}",
                    "",
                ]
            )
    else:
        lines.append("No prior transition is available yet.")
        lines.append("")
    requirements = payload.get("decision_requirements")
    lines.extend(["## Oracle decision requirements", ""])
    if isinstance(requirements, dict):
        for key in ("evidence_gaps", "do_not_repeat", "required_oracle_contract"):
            lines.append(f"### {key.replace('_', ' ').title()}")
            values = requirements.get(key)
            if isinstance(values, list) and values:
                lines.extend(f"- {value}" for value in values)
            else:
                lines.append("- None recorded")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _freeze_iteration_inputs(*, paths: IterationEvidencePaths, iter_dir: Path) -> None:
    kernel_path = paths.kernel_source_dir / "kernel.py"
    kernel_snapshot_path = iter_dir / KERNEL_SNAPSHOT_FILENAME
    if kernel_path.exists() and kernel_path.is_file() and not kernel_snapshot_path.exists():
        shutil.copyfile(kernel_path, kernel_snapshot_path)
    plan_payload = load_json_object(paths.plan_path)
    plan_snapshot_path = iter_dir / PLAN_SNAPSHOT_FILENAME
    if plan_payload is not None and not plan_snapshot_path.exists():
        write_json_object(plan_snapshot_path, plan_payload, ensure_ascii=False, sort_keys=True)


def _verify_recovery_bundle(
    *,
    evidence_path: Path,
    expected_path: Path | None,
    expected_sha256: str | None,
) -> None:
    if expected_path is None or expected_sha256 is None:
        raise IterationEvidenceIntegrityError("Recovery evidence requires both path and SHA-256.")
    if expected_path.resolve() != evidence_path.resolve():
        raise IterationEvidenceIntegrityError(
            f"Recovery evidence path is outside the expected iteration: {expected_path} != {evidence_path}"
        )
    if not evidence_path.exists():
        raise IterationEvidenceIntegrityError(f"Persisted iteration evidence is missing: {evidence_path}")
    actual = _sha256_file(evidence_path)
    if actual != expected_sha256:
        raise IterationEvidenceIntegrityError(
            f"Persisted iteration evidence digest changed: expected {expected_sha256}, got {actual}."
        )


def _build_iteration_record(
    *,
    iter_dir: Path,
    iteration: int,
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    metrics = load_json_object(iter_dir / "metrics.json") or {}
    evaluation = load_json_object(iter_dir / "evaluation_report.json") or {}
    state = load_json_object(iter_dir / "iteration_state.json") or {}
    portfolio = load_json_object(iter_dir / "portfolio_plan.json") or {}
    candidate_contracts = _load_candidate_contract_index(iter_dir)
    execution_attempt_resolution = _load_execution_attempt_resolution(iter_dir)
    graph = load_json_object(iter_dir / "graph_execution_report.json") or {}
    diagnostics_path = iter_dir / "diagnostics.md"
    strategy_path = iter_dir / "agent" / f"improve_strategy-{iteration:02d}" / "strategy_last_message.txt"
    implementation_path = iter_dir / "agent" / "codex_last_message.txt"
    kernel_snapshot = iter_dir / KERNEL_SNAPSHOT_FILENAME
    plan_snapshot = iter_dir / PLAN_SNAPSHOT_FILENAME
    evidence_files = [
        iter_dir / name
        for name in (
            "metrics.json",
            "evaluation_report.json",
            "iteration_state.json",
            "diagnostics.md",
            "experiment_graph.json",
            "allocator_decision.json",
            "graph_execution_report.json",
            "validation_lab_report.json",
            "private_robustness_report.json",
            "portfolio_optimizer_report.json",
            "top1_exhaustion_report.json",
            "blend_report.json",
            "portfolio_plan.json",
        )
    ]
    contract_index_path = candidate_contracts.get("index_path")
    if isinstance(contract_index_path, str) and contract_index_path:
        evidence_files.append(Path(contract_index_path))
    for contract in candidate_contracts.get("contracts", []):
        if isinstance(contract, dict) and isinstance(contract.get("path"), str):
            evidence_files.append(Path(contract["path"]))
    resolution_path = execution_attempt_resolution.get("path")
    if isinstance(resolution_path, str) and resolution_path:
        evidence_files.append(Path(resolution_path))
    error_signatures = _apply_execution_attempt_resolution(
        _collect_error_signatures(iter_dir / "logs"),
        execution_attempt_resolution,
    )
    return {
        "iteration": iteration,
        "score": _score_record(metrics=metrics, evaluation=evaluation),
        "execution": {
            "iteration_complete": state.get("iteration_complete"),
            "trained": state.get("trained"),
            "submission_exists": state.get("submission_exists"),
            "submitted": state.get("submitted"),
            "submit_phase_state": state.get("submit_phase_state"),
            "submission_ref": state.get("submission_ref"),
            "submission_sha256": state.get("submission_sha256"),
            "completed_at": state.get("completed_at"),
        },
        "diagnostics_excerpt": _read_text_excerpt(diagnostics_path, _TEXT_EXCERPT_CHARS),
        "error_signatures": error_signatures,
        "execution_attempt_resolution": _bounded_json_value(
            execution_attempt_resolution,
            max_items=40,
        ),
        "portfolio": _portfolio_summary(portfolio, candidate_contracts=candidate_contracts),
        "graph_execution": _graph_summary(graph),
        "submission_attempts": _attempts_for_iteration(attempts, iteration=iteration),
        "kernel_snapshot": _snapshot_record(kernel_snapshot),
        "plan_snapshot": _snapshot_record(plan_snapshot),
        "next_iteration_strategy": {
            "path": str(strategy_path),
            "excerpt": _read_text_excerpt(strategy_path, _STRATEGY_EXCERPT_CHARS),
        },
        "implementation_report": {
            "path": str(implementation_path),
            "excerpt": _read_text_excerpt(implementation_path, _STRATEGY_EXCERPT_CHARS),
        },
        "evidence_files": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in evidence_files
            if path.exists() and path.is_file()
        ],
    }


def _score_record(*, metrics: dict[str, object], evaluation: dict[str, object]) -> dict[str, object]:
    loop_decision = metrics.get("loop_decision")
    loop_decision = loop_decision if isinstance(loop_decision, dict) else {}
    value = tolerant_finite_float(loop_decision.get("value"))
    if value is None:
        value = tolerant_finite_float(metrics.get("offline_value"))
    if value is None:
        value = tolerant_finite_float(evaluation.get("metric_value"))
    source = str(
        loop_decision.get("source") or metrics.get("score_source") or evaluation.get("score_source") or "unknown"
    )
    authoritative_metric = str(
        metrics.get("authoritative_display_metric")
        or evaluation.get("authoritative_display_metric")
        or evaluation.get("metric_name")
        or metrics.get("metric")
        or "unknown"
    )
    canonical_metric = str(
        metrics.get("canonical_technical_metric")
        or evaluation.get("canonical_technical_metric")
        or metrics.get("metric")
        or "unknown"
    )
    model_selection = metrics.get("model_selection_decision")
    model_selection = dict(model_selection) if isinstance(model_selection, dict) else {}
    if not model_selection:
        report_decision = evaluation.get("model_selection_decision")
        model_selection = dict(report_decision) if isinstance(report_decision, dict) else {}
    technical_value = tolerant_finite_float(model_selection.get("value"))
    if technical_value is None:
        technical_value = tolerant_finite_float(metrics.get("offline_value"))
    if _normalize_token(source) == "offline_artifact_rubric":
        metric = "rubric_readiness_score_0_100"
    else:
        metric = canonical_metric
    direction = str(
        loop_decision.get("direction") or metrics.get("direction") or evaluation.get("direction") or "unknown"
    )
    faithfulness = metrics.get("competition_faithfulness")
    faithfulness = faithfulness if isinstance(faithfulness, dict) else {}
    potential = metrics.get("accuracy_potential")
    potential = potential if isinstance(potential, dict) else {}
    trusted_marker = potential.get("trusted")
    faithful_marker = faithfulness.get("faithful")
    grouped_contract_valid, grouped_contract_errors = score_sources.validate_grouped_oof_contract(
        model_selection if model_selection else None
    )
    if isinstance(trusted_marker, bool):
        trusted = trusted_marker
    elif isinstance(faithful_marker, bool):
        trusted = faithful_marker and _normalize_token(source) in _TRUSTED_SCORE_SOURCES
    else:
        trusted = _normalize_token(source) in _TRUSTED_SCORE_SOURCES
    if _normalize_token(source) == "grouped_oof_cv":
        trusted = grouped_contract_valid
    reasons = _string_list(potential.get("quality_reasons")) + _string_list(faithfulness.get("reasons"))
    warnings = _string_list(faithfulness.get("warnings"))
    return {
        "value": value,
        "std": tolerant_finite_float(metrics.get("offline_std")),
        "metric": metric,
        "direction": direction,
        "source": source,
        "trusted": trusted,
        "faithful": faithful_marker,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "fold_scores": _bounded_json_value(evaluation.get("per_fold_scores"), max_items=20),
        "split_strategy": evaluation.get("split_strategy"),
        "n_splits": evaluation.get("n_splits"),
        "readiness_score": tolerant_finite_float(evaluation.get("readiness_score")),
        "public_submission_score": tolerant_finite_float(metrics.get("submission_score")),
        "technical_proxy": {
            "metric": str(model_selection.get("metric") or canonical_metric),
            "authoritative_display_metric": authoritative_metric,
            "value": technical_value,
            "source": str(
                model_selection.get("source")
                or metrics.get("score_source")
                or evaluation.get("score_source")
                or "unknown"
            ),
            "role": "technical_proxy_only",
            "trusted": grouped_contract_valid,
            "trust_errors": grouped_contract_errors,
            "outer_split": model_selection.get("outer_split"),
            "folds": model_selection.get("folds"),
            "seeds": _bounded_json_value(model_selection.get("seeds"), max_items=10),
            "evaluated_rows": model_selection.get("evaluated_rows"),
            "data_hashes": _bounded_json_value(model_selection.get("data_hashes"), max_items=10),
            "evaluation_mask_sha256": model_selection.get("evaluation_mask_sha256"),
            "global_class_list": _bounded_json_value(model_selection.get("global_class_list"), max_items=20),
            "technical_champion_variant": model_selection.get("technical_champion_variant"),
            "deployment_variant": model_selection.get("deployment_variant"),
        },
    }


def _build_transition(previous: dict[str, object], current: dict[str, object]) -> dict[str, object]:
    previous_score = previous.get("score")
    current_score = current.get("score")
    previous_score = previous_score if isinstance(previous_score, dict) else {}
    current_score = current_score if isinstance(current_score, dict) else {}
    comparison = _compare_scores(previous_score, current_score)
    previous_kernel = _snapshot_path(previous.get("kernel_snapshot"))
    current_kernel = _snapshot_path(current.get("kernel_snapshot"))
    previous_plan = _snapshot_path(previous.get("plan_snapshot"))
    current_plan = _snapshot_path(current.get("plan_snapshot"))
    kernel_diff = _text_diff(previous_kernel, current_kernel)
    plan_changes = _json_changes(previous_plan, current_plan)
    material_change = bool(kernel_diff.get("changed") or plan_changes)
    if not comparison["comparable"]:
        outcome = "not_assessable"
    else:
        delta = tolerant_finite_float(comparison.get("improvement_delta"))
        outcome = "supported" if delta is not None and delta > 0 else "not_supported"
    return {
        "from_iteration": previous.get("iteration"),
        "to_iteration": current.get("iteration"),
        "score_comparison": comparison,
        "hypothesis_outcome": outcome,
        "material_change_detected": material_change,
        "kernel_change": kernel_diff,
        "plan_changed_fields": plan_changes,
        "strategy_that_requested_change": previous.get("next_iteration_strategy"),
        "implementation_report": previous.get("implementation_report"),
        "observed_errors": current.get("error_signatures"),
        "submission_attempts": current.get("submission_attempts"),
    }


def _compare_scores(previous: dict[str, object], current: dict[str, object]) -> dict[str, object]:
    reasons: list[str] = []
    previous_value = tolerant_finite_float(previous.get("value"))
    current_value = tolerant_finite_float(current.get("value"))
    previous_metric = _normalize_token(previous.get("metric"))
    current_metric = _normalize_token(current.get("metric"))
    previous_direction = _normalize_token(previous.get("direction"))
    current_direction = _normalize_token(current.get("direction"))
    previous_source = _normalize_token(previous.get("source"))
    current_source = _normalize_token(current.get("source"))
    if previous_value is None or current_value is None:
        reasons.append("missing_score_value")
    if not previous_metric or previous_metric != current_metric:
        reasons.append("metric_mismatch")
    if previous_direction not in {"maximize", "minimize"} or previous_direction != current_direction:
        reasons.append("direction_mismatch")
    if previous_source != current_source:
        reasons.append(f"score_source_mismatch:{previous_source or 'unknown'}!={current_source or 'unknown'}")
    if previous.get("trusted") is not True:
        reasons.append("previous_score_untrusted")
    if current.get("trusted") is not True:
        reasons.append("current_score_untrusted")
    comparable = not reasons
    raw_delta = None if previous_value is None or current_value is None else current_value - previous_value
    improvement_delta: float | None = None
    if comparable and raw_delta is not None:
        improvement_delta = raw_delta if current_direction == "maximize" else -raw_delta
    return {
        "comparable": comparable,
        "reasons": reasons,
        "previous_value": previous_value,
        "current_value": current_value,
        "raw_delta": raw_delta if comparable else None,
        "improvement_delta": improvement_delta,
    }


def _build_invocation_evaluation(
    *,
    evaluation: object,
    current_score: float | None,
    current_score_source: str,
    target_score: float,
    delta_offline: float | None,
) -> dict[str, object]:
    return {
        "metric": str(getattr(evaluation, "metric", "unknown")),
        "direction": str(getattr(evaluation, "direction", "unknown")),
        "value": tolerant_finite_float(getattr(evaluation, "value", None)),
        "score_source": str(getattr(evaluation, "score_source", current_score_source)),
        "std": tolerant_finite_float(getattr(evaluation, "std", None)),
        "fold_scores": _bounded_json_value(getattr(evaluation, "fold_scores", None), max_items=20),
        "loop_current_score": tolerant_finite_float(current_score),
        "loop_current_score_source": current_score_source,
        "legacy_delta_offline": tolerant_finite_float(delta_offline),
        "target_score": target_score,
        "warning": (
            "legacy_delta_offline is context only; use transitions_observed.score_comparison after provenance checks."
        ),
    }


def _load_candidate_contract_index(iter_dir: Path) -> dict[str, object]:
    index_path = next(
        (
            path
            for path in (
                iter_dir / "output" / "candidate_contracts" / "index.json",
                iter_dir / "candidate_contracts" / "index.json",
            )
            if path.is_file()
        ),
        None,
    )
    if index_path is None:
        return {"ingested": False, "contracts": [], "errors": []}
    index = load_json_object(index_path)
    if index is None:
        return {
            "ingested": False,
            "index_path": str(index_path),
            "contracts": [],
            "errors": ["index:invalid_json_object"],
        }
    entries = index.get("contracts")
    if not isinstance(entries, list):
        return {
            "ingested": False,
            "index_path": str(index_path),
            "contracts": [],
            "errors": ["index:contracts_not_list"],
        }
    output_root = index_path.parent.parent
    contracts: list[dict[str, object]] = []
    errors: list[str] = []
    for entry in entries[:_MAX_CANDIDATES]:
        if not isinstance(entry, dict):
            errors.append("entry:not_object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append("entry:path_missing")
            continue
        contract_path = output_root / relative
        contract = load_json_object(contract_path)
        if contract is None:
            errors.append(f"{relative}:invalid_or_missing")
            continue
        category = str(contract.get("category") or entry.get("category") or "")
        status = str(contract.get("status") or entry.get("status") or "")
        if not category or not status:
            errors.append(f"{relative}:category_or_status_missing")
            continue
        score = tolerant_finite_float(contract.get("score"))
        contract_valid = False
        contract_errors: list[str] = []
        if status == "completed":
            if score is None:
                errors.append(f"{relative}:completed_score_nonfinite")
                continue
            if (
                contract.get("technical_metric") != "grouped_macro_f1_moment_type"
                or contract.get("direction") != "maximize"
                or contract.get("score_source") != "grouped_oof_cv"
            ):
                errors.append(f"{relative}:technical_provenance_mismatch")
                continue
            contract_valid, contract_errors = score_sources.validate_grouped_oof_contract(contract)
        contracts.append(
            {
                "candidate_id": contract.get("candidate_id"),
                "category": category,
                "status": status,
                "offline_score": score,
                "score_source": contract.get("score_source"),
                "model_family": contract.get("feature_recipe"),
                "technical_valid": contract.get("technical_valid"),
                "deployment_stable": contract.get("deployment_stable"),
                "rejection_reason": contract.get("rejection_reason"),
                "deployment_rejection_reason": contract.get("deployment_rejection_reason"),
                "trusted": contract_valid,
                "trust_errors": contract_errors,
                "path": str(contract_path),
            }
        )
    return {
        "ingested": bool(contracts) and not errors,
        "index_path": str(index_path),
        "contracts": contracts,
        "errors": errors,
    }


def _load_execution_attempt_resolution(iter_dir: Path) -> dict[str, object]:
    path = next(
        (
            candidate
            for candidate in (
                iter_dir / "output" / "execution_attempt_resolution.json",
                iter_dir / "execution_attempt_resolution.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        return {"ingested": False, "superseded_attempts": [], "active_fatal_attempts": []}
    payload = load_json_object(path)
    if payload is None:
        return {
            "ingested": False,
            "path": str(path),
            "superseded_attempts": [],
            "active_fatal_attempts": [],
            "error": "invalid_json_object",
        }
    return {
        "ingested": True,
        "path": str(path),
        "successful_plan_sha256": payload.get("successful_plan_sha256"),
        "completed_phases": _bounded_json_value(payload.get("completed_phases"), max_items=40),
        "superseded_attempts": _bounded_json_value(payload.get("superseded_attempts"), max_items=40),
        "active_fatal_attempts": _bounded_json_value(payload.get("active_fatal_attempts"), max_items=40),
    }


def _apply_execution_attempt_resolution(
    signatures: list[dict[str, object]],
    resolution: dict[str, object],
) -> list[dict[str, object]]:
    superseded = resolution.get("superseded_attempts")
    superseded = superseded if isinstance(superseded, list) else []
    superseded_paths: set[str] = set()
    fingerprints: list[str] = []
    for attempt in superseded:
        if not isinstance(attempt, dict) or attempt.get("status") != "superseded_by_successful_attempt":
            continue
        fingerprint = str(attempt.get("error_fingerprint") or "")
        if fingerprint:
            fingerprints.append(fingerprint)
        paths = attempt.get("original_log_relative_paths")
        if isinstance(paths, list):
            superseded_paths.update(str(path).replace("\\", "/") for path in paths)
    successful_sha = resolution.get("successful_plan_sha256")
    resolved: list[dict[str, object]] = []
    for signature in signatures:
        item = dict(signature)
        paths = item.get("paths")
        normalized_paths = [str(path).replace("\\", "/") for path in paths] if isinstance(paths, list) else []
        is_superseded = any(
            observed.endswith(expected) for observed in normalized_paths for expected in superseded_paths
        )
        item["active"] = not is_superseded
        item["status"] = "superseded_by_successful_attempt" if is_superseded else "active"
        if is_superseded:
            item["superseded_error_fingerprints"] = fingerprints
            item["superseded_by_plan_sha256"] = successful_sha
        resolved.append(item)
    return resolved


def _portfolio_summary(
    payload: dict[str, object],
    *,
    candidate_contracts: dict[str, object] | None = None,
) -> dict[str, object]:
    ingested = candidate_contracts or {}
    contract_items = ingested.get("contracts")
    contract_by_category = (
        {str(item.get("category")): item for item in contract_items if isinstance(item, dict) and item.get("category")}
        if isinstance(contract_items, list)
        else {}
    )
    candidates = payload.get("candidates")
    summaries: list[dict[str, object]] = []
    if isinstance(candidates, list):
        for candidate in candidates[:_MAX_CANDIDATES]:
            if not isinstance(candidate, dict):
                continue
            summary = {
                key: candidate.get(key)
                for key in (
                    "candidate_id",
                    "category",
                    "status",
                    "method_id",
                    "model_family",
                    "validation_profile_id",
                    "offline_score",
                    "private_robustness_score",
                    "rank_score",
                    "failure_reason",
                )
                if key in candidate
            }
            contract = contract_by_category.get(str(candidate.get("category") or ""))
            if contract is not None:
                summary.update(
                    {
                        "status": contract.get("status"),
                        "offline_score": contract.get("offline_score"),
                        "score_source": contract.get("score_source"),
                        "model_family": contract.get("model_family"),
                        "candidate_contract_path": contract.get("path"),
                    }
                )
            summaries.append(summary)
    existing_categories = {
        str(item.get("category")) for item in summaries if isinstance(item, dict) and item.get("category")
    }
    for category, contract in contract_by_category.items():
        if category in existing_categories:
            continue
        summaries.append(dict(contract))
    original_missing = payload.get("missing_categories")
    missing = (
        [item for item in original_missing if str(item) not in contract_by_category]
        if isinstance(original_missing, list)
        else original_missing
    )
    return {
        "active_validation_profile": payload.get("active_validation_profile"),
        "required_categories": _bounded_json_value(payload.get("required_categories"), max_items=20),
        "missing_categories": _bounded_json_value(missing, max_items=20),
        "next_action": (
            "Candidate contracts ingested; use completed comparable scores "
            "and preserve reporting-only validation status."
            if contract_by_category
            else payload.get("next_action")
        ),
        "candidates": summaries,
        "candidate_contract_ingestion": _bounded_json_value(ingested, max_items=40),
    }


def _graph_summary(payload: dict[str, object]) -> dict[str, object]:
    result = payload.get("result")
    result = result if isinstance(result, dict) else {}
    return {
        key: _bounded_json_value(result.get(key), max_items=20)
        for key in ("mode", "status", "selected_nodes", "completed_nodes", "failed_nodes", "skipped_nodes")
        if key in result
    }


def _attempts_for_iteration(attempts: list[dict[str, object]], *, iteration: int) -> list[dict[str, object]]:
    marker = f"/iter-{iteration}/"
    selected = [record for record in attempts if marker in str(record.get("sub_path") or "")]
    return _summarize_attempts(selected[-10:])


def _summarize_attempts(attempts: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "ts": record.get("ts"),
            "ok": record.get("ok"),
            "exit_code": record.get("exit_code"),
            "error_kind": record.get("error_kind"),
            "action_taken": record.get("action_taken"),
            "reason": record.get("reason"),
            "submission_sha256": record.get("sub_sha256"),
            "code_fingerprint": record.get("code_fingerprint"),
            "submission_fidelity": _bounded_json_value(record.get("submission_fidelity"), max_items=32),
            "error_excerpt": _redact(str(record.get("stderr_tail") or ""))[:1_200],
        }
        for record in attempts
    ]


def _collect_error_signatures(logs_dir: Path) -> list[dict[str, object]]:
    if not logs_dir.exists():
        return []
    signatures: dict[str, dict[str, object]] = {}
    for path in sorted(logs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".log", ".txt", ".json", ".jsonl"}:
            continue
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - _LOG_TAIL_BYTES))
                text = handle.read(_LOG_TAIL_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            clean = _redact(line.strip())
            if not clean or not _ERROR_MARKERS.search(clean):
                continue
            clean = clean[:500]
            key = re.sub(r"\b\d+(?:\.\d+)?\b", "#", clean.lower())
            item = signatures.setdefault(
                key,
                {"sample": clean, "count": 0, "paths": []},
            )
            item["count"] = int(item["count"]) + 1
            paths = item["paths"]
            if isinstance(paths, list) and str(path) not in paths and len(paths) < 4:
                paths.append(str(path))
    return sorted(
        signatures.values(),
        key=lambda item: (-int(item["count"]), str(item["sample"])),
    )[:_MAX_ERROR_SIGNATURES]


def _snapshot_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256_file(path) if path.exists() and path.is_file() else None,
    }


def _snapshot_path(value: object) -> Path | None:
    if not isinstance(value, dict) or value.get("exists") is not True:
        return None
    raw = str(value.get("path") or "").strip()
    return Path(raw) if raw else None


def _text_diff(previous: Path | None, current: Path | None) -> dict[str, object]:
    if previous is None or current is None or not previous.exists() or not current.exists():
        return {"changed": None, "reason": "snapshot_unavailable", "diff_excerpt": ""}
    previous_text = previous.read_text(encoding="utf-8", errors="replace")
    current_text = current.read_text(encoding="utf-8", errors="replace")
    if previous_text == current_text:
        return {"changed": False, "reason": "identical", "diff_excerpt": ""}
    diff = "".join(
        difflib.unified_diff(
            previous_text.splitlines(keepends=True),
            current_text.splitlines(keepends=True),
            fromfile=str(previous),
            tofile=str(current),
            n=2,
        )
    )
    return {
        "changed": True,
        "reason": "sha256_changed",
        "diff_excerpt": diff[:_DIFF_EXCERPT_CHARS],
        "diff_truncated": len(diff) > _DIFF_EXCERPT_CHARS,
    }


def _json_changes(previous: Path | None, current: Path | None) -> list[str]:
    previous_payload = load_json_object(previous) if previous is not None else None
    current_payload = load_json_object(current) if current is not None else None
    if previous_payload is None or current_payload is None:
        return []
    changes: list[str] = []
    _walk_json_changes(previous_payload, current_payload, prefix="", output=changes)
    return changes[:80]


def _walk_json_changes(previous: object, current: object, *, prefix: str, output: list[str]) -> None:
    if len(output) >= 80:
        return
    if isinstance(previous, dict) and isinstance(current, dict):
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous:
                output.append(f"{path}: added")
            elif key not in current:
                output.append(f"{path}: removed")
            else:
                _walk_json_changes(previous[key], current[key], prefix=path, output=output)
        return
    if previous != current:
        output.append(f"{prefix}: {str(previous)[:160]} -> {str(current)[:160]}")


def _evidence_gaps(*, current_record: dict[str, object], transitions: list[dict[str, object]]) -> list[str]:
    gaps: list[str] = []
    score = current_record.get("score")
    score = score if isinstance(score, dict) else {}
    if score.get("trusted") is not True:
        gaps.append(
            f"Current score is untrusted ({score.get('source')}): establish a competition-faithful "
            "comparable evaluation."
        )
    if not transitions:
        gaps.append("No prior frozen kernel/plan transition exists; treat this iteration as the attribution baseline.")
    elif transitions[-1].get("material_change_detected") is not True:
        gaps.append("The latest transition has no verified material kernel or plan change.")
    errors = current_record.get("error_signatures")
    active_errors = (
        [item for item in errors if isinstance(item, dict) and item.get("active", True)]
        if isinstance(errors, list)
        else []
    )
    if active_errors:
        gaps.append(
            f"Resolve {len(active_errors)} current runtime error signature(s) before attributing a model-quality gap."
        )
    portfolio = current_record.get("portfolio")
    portfolio = portfolio if isinstance(portfolio, dict) else {}
    missing = portfolio.get("missing_categories")
    if isinstance(missing, list) and missing:
        gaps.append("Planned candidate categories remain unevaluated: " + ", ".join(str(item) for item in missing[:8]))
    attempts = current_record.get("submission_attempts")
    if isinstance(attempts, list) and any(item.get("ok") is not True for item in attempts if isinstance(item, dict)):
        gaps.append("Submission attempts failed; separate output/runtime contract repair from model-quality changes.")
    if isinstance(attempts, list):
        fidelity_failures = []
        for item in attempts:
            if not isinstance(item, dict):
                continue
            fidelity = item.get("submission_fidelity")
            if not isinstance(fidelity, dict) or fidelity.get("verdict") != "fail":
                continue
            fidelity_failures.append(fidelity)
        if fidelity_failures:
            latest = fidelity_failures[-1]
            codes = ", ".join(str(code) for code in list(latest.get("reason_codes") or [])[:12])
            gaps.append(
                "Submission fidelity repair required: "
                f"reason_codes={codes or 'legacy_unknown'}; report={latest.get('report_path')}"
            )
    return gaps


def _do_not_repeat(transitions: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for transition in transitions:
        outcome = str(transition.get("hypothesis_outcome") or "")
        from_iteration = transition.get("from_iteration")
        to_iteration = transition.get("to_iteration")
        if outcome == "not_supported":
            warnings.append(
                f"Do not repeat the iter-{from_iteration}→iter-{to_iteration} strategy unchanged; "
                "its comparable trusted score did not improve."
            )
        elif outcome == "not_assessable":
            warnings.append(
                f"Do not claim the iter-{from_iteration}→iter-{to_iteration} strategy worked until its result is "
                "measured with a comparable trusted source."
            )
        if transition.get("material_change_detected") is False:
            warnings.append(
                f"Do not spend iter-{to_iteration + 1 if isinstance(to_iteration, int) else '?'} on another "
                "no-op/same-artifact transition."
            )
    return warnings[-8:]


def _submission_history_summary(history: dict[str, object] | None) -> dict[str, object] | None:
    if not history:
        return None
    return {
        "source": history.get("source"),
        "direction": history.get("direction"),
        "best_score": tolerant_finite_float(history.get("best_score")),
        "latest_score": tolerant_finite_float(history.get("latest_score")),
        "best": _bounded_json_value(history.get("best"), max_items=20),
        "latest": _bounded_json_value(history.get("latest"), max_items=20),
        "recent": _bounded_json_value(history.get("recent"), max_items=10),
        "recent_unscored": _bounded_json_value(history.get("recent_unscored"), max_items=10),
    }


def _format_optional_score(value: object, *, signed: bool = False) -> str:
    score = tolerant_finite_float(value)
    if score is None:
        return "unavailable"
    return f"{score:+.6f}" if signed else f"{score:.6f}"


def _bounded_json_value(value: object, *, max_items: int) -> object:
    if isinstance(value, dict):
        return {
            str(key): _bounded_json_value(item, max_items=max_items) for key, item in list(value.items())[:max_items]
        }
    if isinstance(value, list):
        return [_bounded_json_value(item, max_items=max_items) for item in value[:max_items]]
    if isinstance(value, tuple):
        return [_bounded_json_value(item, max_items=max_items) for item in value[:max_items]]
    if isinstance(value, str):
        return _redact(value)[:_TEXT_EXCERPT_CHARS]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact(str(value))[:_TEXT_EXCERPT_CHARS]


def _read_text_excerpt(path: Path, max_chars: int) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return _redact(text)
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return _redact(text[:head_chars] + "\n...[excerpt truncated]...\n" + text[-tail_chars:])


def _redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_token(value: object) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip().lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
