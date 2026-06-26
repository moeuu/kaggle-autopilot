from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich import print

from kagglebot import submit_attempts as _submit_attempts
from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.json_utils import load_json_object, load_json_object_or_empty, write_json_object
from kagglebot.kernel_outputs import find_newest_existing_path, find_submission_file
from kagglebot.scalar_utils import tolerant_finite_float
from kagglebot.score_utils import should_update_best_score

if TYPE_CHECKING:
    from collections.abc import Callable

    from kagglebot.paths import CompetitionPaths
    from kagglebot.solver.evaluate import EvaluationResult


_ITERATION_STATE_FILENAME = "iteration_state.json"


class ResumeRunResolutionError(ValueError):
    def __init__(self, message: str, *, param_hint: str) -> None:
        super().__init__(message)
        self.param_hint = param_hint


def resolve_resume_run_id(
    *,
    paths: CompetitionPaths,
    resume_run_id: str | None,
    resume_latest: bool,
) -> str | None:
    if resume_run_id and resume_latest:
        raise ResumeRunResolutionError(
            "Use either --resume-run-id or --resume-latest, not both.",
            param_hint="--resume-run-id",
        )
    if resume_run_id:
        candidate = resume_run_id.strip()
        if not candidate:
            raise ResumeRunResolutionError("--resume-run-id cannot be empty.", param_hint="--resume-run-id")
        if paths.run_dir(candidate).exists():
            return candidate
        run_ids = sorted(list_run_ids(paths))
        prefix_matches = [run_id for run_id in run_ids if run_id.startswith(candidate)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        if len(prefix_matches) > 1:
            options = ", ".join(prefix_matches[:5])
            raise ResumeRunResolutionError(
                f"Run ID prefix is ambiguous: {candidate} ({options})",
                param_hint="--resume-run-id",
            )
        if run_ids:
            hints = ", ".join(run_ids[-3:])
            raise ResumeRunResolutionError(
                f"Run ID not found: {candidate}. Recent run IDs: {hints}",
                param_hint="--resume-run-id",
            )
        raise ResumeRunResolutionError(f"Run ID not found: {candidate}", param_hint="--resume-run-id")
    if not resume_latest:
        return None
    latest = find_latest_run_id(paths)
    if latest is None:
        raise ResumeRunResolutionError(f"No prior runs found under {paths.runs_dir}", param_hint="--resume-latest")
    return latest


def find_latest_run_id(paths: CompetitionPaths) -> str | None:
    runs_dir = paths.runs_dir
    if not runs_dir.exists():
        return None
    latest_name: str | None = None
    latest_mtime: float | None = None
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            mtime = run_dir.stat().st_mtime
        except OSError:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_name = run_dir.name
            latest_mtime = mtime
    return latest_name


def list_run_ids(paths: CompetitionPaths) -> list[str]:
    runs_dir = paths.runs_dir
    if not runs_dir.exists():
        return []
    run_ids: list[str] = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            run_ids.append(run_dir.name)
    return run_ids


def build_run_payload(
    *,
    run_id: str,
    config: object,
    resolved: dict[str, object],
    status: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "slug": getattr(config, "slug"),
        "started_at": datetime.now(UTC).isoformat(),
        "status": status,
        "config": {
            "agent": getattr(config, "agent"),
            "compute": getattr(config, "compute"),
            "accelerator": getattr(config, "accelerator"),
            "deliverable_mode": resolved.get("deliverable_mode"),
            "submit_mode": resolved.get("submit_mode"),
            "code_competition": resolved.get("code_competition"),
            "notebook_submit_artifact_mode": resolved.get("notebook_submit_artifact_mode"),
            "target_medal": resolved.get("target_medal"),
            "target_rank_percentile": resolved.get("target_rank_percentile"),
            "campaign_mode": resolved.get("campaign_mode"),
            "method_scout": getattr(config, "method_scout"),
            "research_scout": resolved.get("research_scout"),
            "method_scout_max_sources": getattr(config, "method_scout_max_sources"),
            "validation_lab": resolved.get("validation_lab"),
            "portfolio_execution": resolved.get("portfolio_execution"),
            "candidate_budget_min": getattr(config, "candidate_budget_min"),
            "max_candidates_per_iteration": getattr(config, "max_candidates_per_iteration"),
            "top1_exhaustive": resolved.get("top1_exhaustive"),
            "top1_submit_policy": resolved.get("top1_submit_policy"),
            "kaggle_username": getattr(config, "kaggle_username"),
            "kernel_name": resolved.get("kernel_name"),
            "internet": resolved.get("internet"),
            "score_source": resolved.get("score_source"),
            "holdout_frac": resolved.get("holdout_frac"),
            "cv_folds": resolved.get("cv_folds"),
            "split_strategy": resolved.get("split_strategy"),
            "target_metric": resolved.get("target_metric"),
            "target_score": resolved.get("target_score"),
            "target_direction": resolved.get("target_direction"),
            "max_iterations": resolved.get("max_iterations"),
            "max_total_min": resolved.get("max_total_min"),
            "patience": resolved.get("patience"),
            "min_improvement": resolved.get("min_improvement"),
            "time_budget_min": resolved.get("time_budget_min"),
            "seed": resolved.get("seed"),
            "eval_seeds": resolved.get("eval_seeds"),
            "eval_repeats": resolved.get("eval_repeats"),
            "submit_policy": resolved.get("submit_policy"),
            "submission_gate": resolved.get("submission_gate"),
            "submission_limit_per_day": resolved.get("submission_limit_per_day"),
            "readiness_target_score": resolved.get("readiness_target_score"),
            "readiness_method": resolved.get("readiness_method"),
            "readiness_k": resolved.get("readiness_k"),
            "ci_method": resolved.get("ci_method"),
            "ci_alpha": resolved.get("ci_alpha"),
            "drift_check": resolved.get("drift_check"),
            "drift_weight": resolved.get("drift_weight"),
            "stop_min_delta": resolved.get("stop_min_delta"),
            "stop_no_improve_patience": resolved.get("stop_no_improve_patience"),
            "stop_same_config_patience": resolved.get("stop_same_config_patience"),
            "rank_force_major_max_percentile": resolved.get("rank_force_major_max_percentile"),
            "rank_force_major_min_teams": resolved.get("rank_force_major_min_teams"),
            "evaluation_contract": resolved.get("evaluation_contract"),
            "submit": getattr(config, "submit"),
            "message": getattr(config, "message"),
        },
    }


def build_run_summary_payload(
    *,
    best_score: float | None,
    best_submission: Path | None,
    best_submittable_score: float | None,
    best_submittable_submission: Path | None,
    best_high_potential_score: float | None,
    best_high_potential_submission: Path | None,
    best_high_potential_iteration: int | None,
    best_high_potential_meta: dict[str, object] | None,
    fallback_submit_blocked_reason: str | None,
) -> dict[str, object]:
    return {
        "best_trusted_score": best_score,
        "best_trusted_submission": str(best_submission) if best_submission is not None else None,
        "best_competition_faithful_score": best_submittable_score,
        "best_competition_faithful_submission": (
            str(best_submittable_submission) if best_submittable_submission is not None else None
        ),
        "best_high_potential_score": best_high_potential_score,
        "best_high_potential_submission": (
            str(best_high_potential_submission) if best_high_potential_submission is not None else None
        ),
        "best_high_potential_iteration": best_high_potential_iteration,
        "best_high_potential_meta": best_high_potential_meta,
        "fallback_submit_blocked_reason": fallback_submit_blocked_reason,
    }


def apply_run_status(
    payload: dict[str, object],
    *,
    status: str,
    stop_reason: str | None = None,
) -> dict[str, object]:
    payload["status"] = status
    if stop_reason:
        payload["stop_reason"] = stop_reason
    return payload


def apply_final_run_status(
    payload: dict[str, object],
    *,
    submitted: bool,
    has_submission_result: bool,
    writeup_mode: bool,
    writeup_bundle_meta: dict[str, object] | None,
) -> dict[str, object]:
    if submitted and has_submission_result:
        return apply_run_status(payload, status="submitted")
    if writeup_mode and writeup_bundle_meta:
        payload["writeup_bundle"] = writeup_bundle_meta
        return apply_run_status(payload, status="manual_finalization_required")
    if payload.get("status") not in {"interrupted", "submit_failed"}:
        return apply_run_status(payload, status="completed")
    return payload


def write_run_payload(run_dir: Path, payload: dict[str, object]) -> None:
    write_json_object(run_dir / "run.json", payload)


def write_iteration_state_marker(
    *,
    iter_dir: Path,
    run_id: str,
    iteration: int,
    submission_path: Path,
    metrics_path: Path,
    evaluation_report_path: Path,
    submit_phase_required: bool,
    submit_phase_finished: bool | None = None,
    submit_allowed_by_gate: bool,
    submit_phase_state: str,
    forced_submit_reason: str | None = None,
    submitted: bool,
    readiness_score: float,
) -> None:
    if submit_phase_finished is None:
        submit_phase_finished = (not submit_phase_required) or (not submit_allowed_by_gate) or submitted

    payload = {
        "run_id": run_id,
        "iteration": iteration,
        "iteration_complete": True,
        "trained": True,
        "submission_exists": submission_path.exists(),
        "submission_path": str(submission_path),
        "metrics_exists": metrics_path.exists(),
        "metrics_path": str(metrics_path),
        "evaluation_report_exists": evaluation_report_path.exists(),
        "evaluation_report_path": str(evaluation_report_path),
        "submit_phase_required": submit_phase_required,
        "submit_phase_finished": submit_phase_finished,
        "submit_allowed_by_gate": submit_allowed_by_gate,
        "submit_phase_state": submit_phase_state,
        "forced_submit_reason": forced_submit_reason or "",
        "submitted": submitted,
        "readiness_score": readiness_score,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    path = iter_dir / _ITERATION_STATE_FILENAME
    write_json_object(path, payload)


def _load_iteration_state_marker(path: Path) -> dict[str, object]:
    return load_json_object_or_empty(path)


def _is_iteration_marker_complete(payload: dict[str, object], *, require_submit_phase: bool) -> bool:
    if not payload:
        return False
    if not bool(payload.get("iteration_complete")):
        return False
    if require_submit_phase and not bool(payload.get("submit_phase_finished")):
        return False
    if require_submit_phase and bool(payload.get("submit_allowed_by_gate")) and not bool(payload.get("submitted")):
        phase_state = str(payload.get("submit_phase_state") or "").strip().lower()
        if phase_state not in _submit_attempts.TERMINAL_UNSUBMITTED_PHASE_STATES:
            return False
    return True


def _load_submitted_iteration_tracking_score(
    *,
    metrics_path: Path,
    metric_direction: str,
    target_metric: str,
    load_kernel_metrics: Callable[[Path, str, str], EvaluationResult | None],
) -> float | None:
    payload = load_json_object(metrics_path)
    if isinstance(payload, dict):
        submission_score = tolerant_finite_float(payload.get("submission_score"))
        if isinstance(submission_score, float) and math.isfinite(submission_score):
            return float(submission_score)
        loop_decision = payload.get("loop_decision")
        if isinstance(loop_decision, dict):
            source = str(loop_decision.get("source") or "").strip().lower()
            if source.startswith("submission"):
                loop_value = tolerant_finite_float(loop_decision.get("value"))
                if isinstance(loop_value, float) and math.isfinite(loop_value):
                    return float(loop_value)
    evaluation = load_kernel_metrics(metrics_path, metric_direction, target_metric)
    if evaluation is None:
        return None
    return float(evaluation.value)


def resume_best_submitted_offline_score(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
    load_kernel_metrics: Callable[[Path, str, str], EvaluationResult | None],
) -> float | None:
    run_dir = paths.run_dir(run_id)
    if not run_dir.exists():
        return None
    best_score: float | None = None
    for iter_dir in sorted(run_dir.glob("iter-*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if iteration > max_iterations:
            continue
        marker_path = iter_dir / _ITERATION_STATE_FILENAME
        marker_payload = _load_iteration_state_marker(marker_path)
        if not bool(marker_payload.get("submitted")):
            continue
        metrics_path = iter_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        submitted_score = _load_submitted_iteration_tracking_score(
            metrics_path=metrics_path,
            metric_direction=metric_direction,
            target_metric=target_metric,
            load_kernel_metrics=load_kernel_metrics,
        )
        if submitted_score is None:
            continue
        if should_update_best_score(best_score, submitted_score, metric_direction, 0.0):
            best_score = submitted_score
    return best_score


def resume_best_submittable_iteration_state(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
    load_kernel_metrics: Callable[[Path, str, str], EvaluationResult | None],
    iteration_metrics_allow_submit: Callable[[Path, EvaluationResult], bool],
) -> tuple[float | None, Path | None]:
    run_dir = paths.run_dir(run_id)
    if not run_dir.exists():
        return None, None
    best_score: float | None = None
    best_submission: Path | None = None
    for iter_dir in sorted(run_dir.glob("iter-*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if iteration > max_iterations:
            continue
        submission_path = resolve_iteration_submission_artifact(iter_dir)
        metrics_path = iter_dir / "metrics.json"
        if submission_path is None or not metrics_path.exists():
            continue
        evaluation = load_kernel_metrics(metrics_path, metric_direction, target_metric)
        if evaluation is None or (not iteration_metrics_allow_submit(metrics_path, evaluation)):
            continue
        if should_update_best_score(best_score, evaluation.value, metric_direction, 0.0):
            best_score = evaluation.value
            best_submission = submission_path
    return best_score, best_submission


def resume_iteration_state(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
    require_submit_phase: bool = False,
    load_kernel_metrics: Callable[[Path, str, str], EvaluationResult | None],
    infer_iteration_from_submission_path: Callable[[Path], int | None],
) -> tuple[int, float | None, Path | None]:
    run_dir = paths.run_dir(run_id)
    if not run_dir.exists():
        return 1, None, None
    best_score: float | None = None
    best_submission: Path | None = None
    completed_iters: list[int] = []
    legacy_submit_phase_iters = (
        _submit_attempts.load_submit_phase_completed_iterations(
            run_dir,
            infer_iteration_from_submission_path=infer_iteration_from_submission_path,
        )
        if require_submit_phase
        else set()
    )
    for iter_dir in sorted(run_dir.glob("iter-*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if iteration > max_iterations:
            continue
        submission_path = resolve_iteration_submission_artifact(iter_dir)
        metrics_path = iter_dir / "metrics.json"
        if submission_path is None and not metrics_path.exists():
            continue
        if submission_path is not None and not metrics_path.exists():
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has submission artifact but no metrics.json; treating as incomplete."
            )
            continue
        if metrics_path.exists() and submission_path is None:
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has metrics.json but no submission artifact; treating as incomplete."
            )
            continue

        marker_path = iter_dir / _ITERATION_STATE_FILENAME
        marker_payload = _load_iteration_state_marker(marker_path)
        marker_complete = _is_iteration_marker_complete(marker_payload, require_submit_phase=require_submit_phase)
        legacy_complete = False
        if not marker_complete:
            if require_submit_phase:
                legacy_complete = iteration in legacy_submit_phase_iters
            else:
                legacy_complete = True
            if not legacy_complete:
                phase_note = "submit phase completion" if require_submit_phase else "completion marker"
                print(
                    "[yellow]resume[/yellow]: "
                    f"iter-{iteration} missing {phase_note} ({marker_path.name}); treating as incomplete."
                )
                continue
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has no {marker_path.name}; inferred completion from legacy artifacts."
            )

        try:
            evaluation = load_kernel_metrics(metrics_path, metric_direction, target_metric)
        except Exception:  # noqa: BLE001
            evaluation = None
        if evaluation is None:
            print(
                "[yellow]resume[/yellow]: "
                f"{metrics_path} is missing a valid offline metric; treating iter-{iteration} as incomplete."
            )
            continue

        completed_iters.append(iteration)
        if submission_path is None:
            continue
        if best_submission is None:
            best_submission = submission_path
        if best_score is None or should_update_best_score(best_score, evaluation.value, metric_direction, 0.0):
            best_score = evaluation.value
            best_submission = submission_path
    if not completed_iters:
        return 1, best_score, best_submission
    next_iter = max(completed_iters) + 1
    return next_iter, best_score, best_submission


def resolve_iteration_artifact(iter_dir: Path, filename: str) -> Path | None:
    primary = find_newest_existing_path(
        [
            iter_dir / filename,
            iter_dir / "output" / filename,
        ]
    )
    if primary is not None:
        return primary

    run_dir = iter_dir.parent
    runs_dir = run_dir.parent
    competition_dir = runs_dir.parent
    kernel_run_dir = competition_dir / "kernels" / run_dir.name
    fallback_candidates: list[Path] = [
        kernel_run_dir / "outputs" / filename,
        competition_dir / "kernel" / "outputs" / filename,
    ]
    try:
        iteration = int(iter_dir.name.split("-", 1)[1])
    except (IndexError, ValueError):
        iteration = None
    if iteration is not None:
        fallback_candidates.extend(
            [
                kernel_run_dir / f"local-iter-{iteration}" / "outputs" / filename,
                kernel_run_dir / f"submit-iter-{iteration}" / "outputs" / filename,
            ]
        )
    for root in (kernel_run_dir, competition_dir / "kernel" / "outputs"):
        if not root.exists():
            continue
        try:
            fallback_candidates.extend(path for path in root.rglob(filename) if path.is_file())
        except OSError:
            continue
    return find_newest_existing_path(fallback_candidates)


def resolve_iteration_submission_artifact(iter_dir: Path) -> Path | None:
    return find_submission_file(iter_dir)


def _is_submit_only_metrics_payload(metrics_path: Path) -> bool:
    payload = load_json_object(metrics_path)
    if not isinstance(payload, dict):
        return False
    return str(payload.get("kind") or "").strip().lower() == "submit_only"


def _submit_retry_metrics_candidates(iter_dir: Path, marker_payload: dict[str, object]) -> list[Path]:
    candidates: list[Path] = []
    marker_metrics_path = marker_payload.get("metrics_path")
    if isinstance(marker_metrics_path, str) and marker_metrics_path.strip():
        candidates.append(Path(marker_metrics_path))
    candidates.append(iter_dir / "metrics.json")
    resolved_path = resolve_iteration_artifact(iter_dir, "metrics.json")
    if resolved_path is not None:
        candidates.append(resolved_path)

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def copy_submission_artifact_to_iteration_dir(*, source: Path, iter_dir: Path) -> Path:
    destination = iter_dir / source.name
    return copy_artifact_if_needed(source=source, destination=destination)


def copy_kernel_support_artifacts_to_iteration_dir(*, kernel_output_dir: Path, iter_dir: Path) -> None:
    if not kernel_output_dir.exists():
        return
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("oof_predictions.csv", "split_diagnostics.json", "feature_suspects.csv"):
        source = kernel_output_dir / filename
        if not source.exists() or not source.is_file():
            continue
        destination = output_dir / filename
        copy_artifact_if_needed(source=source, destination=destination)


def _latest_iteration_with_training_artifacts(*, run_dir: Path, max_iterations: int) -> int | None:
    latest: int | None = None
    for iter_dir in sorted(run_dir.glob("iter-*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if iteration > max_iterations:
            continue
        submission_path = resolve_iteration_submission_artifact(iter_dir)
        metrics_path = resolve_iteration_artifact(iter_dir, "metrics.json")
        if submission_path is None or metrics_path is None:
            continue
        if _is_submit_only_metrics_payload(metrics_path):
            continue
        if latest is None or iteration > latest:
            latest = iteration
    return latest


def load_submit_retry_artifacts(
    *,
    run_dir: Path,
    iter_dir: Path,
    iteration: int,
    max_iterations: int,
    metric_direction: str,
    target_metric: str,
    require_submit_phase: bool,
    load_kernel_metrics: Callable[[Path, str, str], EvaluationResult | None],
) -> tuple[Path, Path, EvaluationResult] | None:
    if not require_submit_phase:
        return None

    marker_payload = _load_iteration_state_marker(iter_dir / _ITERATION_STATE_FILENAME)
    latest_attempt = _submit_attempts.latest_submit_attempt_for_iteration(run_dir, iteration)
    latest_attempt_complete = latest_attempt is not None and _submit_attempts.is_submit_attempt_complete_for_resume(
        latest_attempt
    )
    marker_pending = (
        bool(marker_payload.get("trained"))
        and bool(marker_payload.get("submit_allowed_by_gate"))
        and (not bool(marker_payload.get("submit_phase_finished")))
        and not latest_attempt_complete
    )

    legacy_pending = False
    if not marker_pending:
        latest_iter = _latest_iteration_with_training_artifacts(run_dir=run_dir, max_iterations=max_iterations)
        marker_has_submit_phase_fields = any(
            key in marker_payload for key in ("submit_phase_finished", "submit_allowed_by_gate", "submitted")
        )
        if (
            latest_iter == iteration
            and not latest_attempt_complete
            and (not marker_payload or not marker_has_submit_phase_fields)
        ):
            legacy_pending = True
    if not (marker_pending or legacy_pending):
        return None

    submission_path = resolve_iteration_submission_artifact(iter_dir)
    if submission_path is None:
        return None
    for metrics_path in _submit_retry_metrics_candidates(iter_dir, marker_payload):
        if not metrics_path.exists():
            continue
        if _is_submit_only_metrics_payload(metrics_path):
            continue
        evaluation = load_kernel_metrics(metrics_path, metric_direction, target_metric)
        if evaluation is not None:
            return submission_path, metrics_path, evaluation
    return None


def load_run_state(run_dir: Path) -> dict[str, object]:
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        attempted = _submit_attempts.has_submit_attempt_records(run_dir)
        return {"submit_attempted": attempted, "submit_ok": False}
    payload = load_json_object_or_empty(state_path)
    if not payload.get("submit_attempted"):
        payload["submit_attempted"] = _submit_attempts.has_submit_attempt_records(run_dir)
    if "submit_ok" not in payload:
        payload["submit_ok"] = False
    if "last_submit_fingerprint" not in payload and payload.get("last_fingerprint"):
        payload["last_submit_fingerprint"] = payload.get("last_fingerprint")
    if "last_fingerprint" not in payload and payload.get("last_submit_fingerprint"):
        payload["last_fingerprint"] = payload.get("last_submit_fingerprint")
    if bool(payload.get("submit_attempted")) and not bool(payload.get("submit_ok")):
        if _submit_attempts.has_successful_submit_attempt(run_dir):
            payload["submit_ok"] = True
    return payload


def save_run_state(run_dir: Path, updates: dict[str, object]) -> None:
    state = load_run_state(run_dir)
    state.update(updates)
    state["submit_attempted"] = bool(state.get("submit_attempted")) or _submit_attempts.has_submit_attempt_records(
        run_dir
    )
    state["submit_ok"] = bool(state.get("submit_ok")) or _submit_attempts.has_successful_submit_attempt(run_dir)
    state["updated_at"] = datetime.now(UTC).isoformat()
    state_path = run_dir / "run_state.json"
    write_json_object(state_path, state)
