from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot import autopilot_state, json_utils, kernel_metrics, metric_matching

if TYPE_CHECKING:
    from kagglebot.solver.evaluate import EvaluationResult


def recheck_kernel_metrics_from_artifacts(
    *,
    submission_path: Path,
    iter_dir: Path,
    metrics_artifact_path: Path | None,
    target_metric: str | None,
    metric_direction: str,
) -> tuple[EvaluationResult, dict[str, object] | None, Path]:
    """Recheck metric parsing from existing artifacts without retraining."""
    rechecked_submission_path = submission_path
    if not rechecked_submission_path.exists():
        resolved_submission = autopilot_state.resolve_iteration_submission_artifact(iter_dir)
        if resolved_submission is None:
            raise RuntimeError(
                "Metric recheck failed: submission artifact is missing for same-iteration metric-only recheck."
            )
        rechecked_submission_path = autopilot_state.copy_submission_artifact_to_iteration_dir(
            source=resolved_submission,
            iter_dir=iter_dir,
        )

    output_metrics_path = iter_dir / "output" / "metrics.json"
    metrics_candidates: list[Path] = []
    seen_metric_candidates: set[str] = set()

    def add_metrics_candidate(path: Path | None) -> None:
        if path is None:
            return
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen_metric_candidates:
            return
        seen_metric_candidates.add(key)
        metrics_candidates.append(path)

    add_metrics_candidate(metrics_artifact_path)
    add_metrics_candidate(iter_dir / "metrics.json")
    add_metrics_candidate(output_metrics_path)
    add_metrics_candidate(autopilot_state.resolve_iteration_artifact(iter_dir, "metrics.json"))

    loaded_candidates: list[tuple[Path, dict[str, object], EvaluationResult | None]] = []
    for metrics_candidate in metrics_candidates:
        if not metrics_candidate.exists():
            continue
        candidate_payload = json_utils.load_json_object(metrics_candidate)
        if candidate_payload is None:
            continue
        if str(candidate_payload.get("kind") or "").strip().lower() == "submit_only":
            continue
        candidate_evaluation = kernel_metrics.load_kernel_metrics(
            metrics_candidate,
            metric_direction,
            target_metric,
        )
        loaded_candidates.append((metrics_candidate, candidate_payload, candidate_evaluation))

    valid_candidates = [candidate for candidate in loaded_candidates if candidate[2] is not None]
    selected_candidate: tuple[Path, dict[str, object], EvaluationResult | None] | None = None
    if valid_candidates:
        selected_candidate = valid_candidates[0]
        output_candidate = next(
            (candidate for candidate in valid_candidates if candidate[0] == output_metrics_path),
            None,
        )
        if output_candidate is not None:
            try:
                output_mtime = output_metrics_path.stat().st_mtime
                other_mtimes = [
                    candidate[0].stat().st_mtime
                    for candidate in valid_candidates
                    if candidate[0] != output_metrics_path
                ]
            except OSError:
                other_mtimes = []
                output_mtime = 0.0
            if not other_mtimes or output_mtime >= max(other_mtimes):
                selected_candidate = output_candidate
    elif loaded_candidates:
        selected_candidate = loaded_candidates[0]

    if selected_candidate is None:
        raise RuntimeError(
            "Metric recheck failed: metrics.json artifact is missing for same-iteration metric-only recheck."
        )

    resolved_metrics_path, payload, evaluation = selected_candidate
    metric_mismatch = bool(
        target_metric
        and evaluation
        and evaluation.metric
        and not metric_matching.metrics_equivalent(evaluation.metric, target_metric)
    )
    needs_recompute = evaluation is None or metric_mismatch
    if needs_recompute:
        recomputed = kernel_metrics.recompute_metric_from_oof_artifact(
            iter_dir=iter_dir,
            payload=payload,
            target_metric=target_metric,
            metric_direction=metric_direction,
            resolve_iteration_artifact=autopilot_state.resolve_iteration_artifact,
        )
        if recomputed is not None:
            evaluation, payload = recomputed
            kernel_metrics.persist_metric_recheck_payload(
                iter_dir=iter_dir,
                resolved_metrics_path=resolved_metrics_path,
                payload=payload,
            )
    if evaluation is None:
        raise RuntimeError("Metric recheck failed: kernel metrics missing expected score.")
    return evaluation, payload, rechecked_submission_path
