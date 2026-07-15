from __future__ import annotations

import difflib
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
_TRUSTED_SCORE_SOURCES = frozenset({"cv", "holdout", "offline", "oof", "cross_validation"})
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
                "- iter-{iteration}: value={value}, source={source}, trusted={trusted}, metric={metric}".format(
                    iteration=record.get("iteration"),
                    value=score.get("value"),
                    source=score.get("source"),
                    trusted=score.get("trusted"),
                    metric=score.get("metric"),
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
        "error_signatures": _collect_error_signatures(iter_dir / "logs"),
        "portfolio": _portfolio_summary(portfolio),
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
    metric = str(metrics.get("metric") or evaluation.get("metric_name") or "unknown")
    direction = str(metrics.get("direction") or evaluation.get("direction") or "unknown")
    faithfulness = metrics.get("competition_faithfulness")
    faithfulness = faithfulness if isinstance(faithfulness, dict) else {}
    potential = metrics.get("accuracy_potential")
    potential = potential if isinstance(potential, dict) else {}
    trusted_marker = potential.get("trusted")
    faithful_marker = faithfulness.get("faithful")
    if isinstance(trusted_marker, bool):
        trusted = trusted_marker
    elif isinstance(faithful_marker, bool):
        trusted = faithful_marker and _normalize_token(source) in _TRUSTED_SCORE_SOURCES
    else:
        trusted = _normalize_token(source) in _TRUSTED_SCORE_SOURCES
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


def _portfolio_summary(payload: dict[str, object]) -> dict[str, object]:
    candidates = payload.get("candidates")
    summaries: list[dict[str, object]] = []
    if isinstance(candidates, list):
        for candidate in candidates[:_MAX_CANDIDATES]:
            if not isinstance(candidate, dict):
                continue
            summaries.append(
                {
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
            )
    return {
        "active_validation_profile": payload.get("active_validation_profile"),
        "required_categories": _bounded_json_value(payload.get("required_categories"), max_items=20),
        "missing_categories": _bounded_json_value(payload.get("missing_categories"), max_items=20),
        "next_action": payload.get("next_action"),
        "candidates": summaries,
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
    if isinstance(errors, list) and errors:
        gaps.append(f"Resolve {len(errors)} current runtime error signature(s) before attributing a model-quality gap.")
    portfolio = current_record.get("portfolio")
    portfolio = portfolio if isinstance(portfolio, dict) else {}
    missing = portfolio.get("missing_categories")
    if isinstance(missing, list) and missing:
        gaps.append("Planned candidate categories remain unevaluated: " + ", ".join(str(item) for item in missing[:8]))
    attempts = current_record.get("submission_attempts")
    if isinstance(attempts, list) and any(item.get("ok") is not True for item in attempts if isinstance(item, dict)):
        gaps.append("Submission attempts failed; separate output/runtime contract repair from model-quality changes.")
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
