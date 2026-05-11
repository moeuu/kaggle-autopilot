from __future__ import annotations

import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich import print

from kagglebot.autopilot_helpers import _to_float, _to_int, _update_best_score
from kagglebot.submission_artifacts import find_submission_manifest, resolve_manifest_references

if TYPE_CHECKING:
    from collections.abc import Callable

    from kagglebot.paths import CompetitionPaths
    from kagglebot.solver.evaluate import EvaluationResult


_ITERATION_STATE_FILENAME = "iteration_state.json"
_LEGACY_SUBMIT_PHASE_COMPLETE_ACTIONS = frozenset({"submit"})
_TERMINAL_UNSUBMITTED_PHASE_STATES = frozenset({"duplicate_submission_sha_seen"})


def _write_iteration_state_marker(
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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_iteration_state_marker(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _is_iteration_marker_complete(payload: dict[str, object], *, require_submit_phase: bool) -> bool:
    if not payload:
        return False
    if not bool(payload.get("iteration_complete")):
        return False
    if require_submit_phase and not bool(payload.get("submit_phase_finished")):
        return False
    if require_submit_phase and bool(payload.get("submit_allowed_by_gate")) and not bool(payload.get("submitted")):
        phase_state = str(payload.get("submit_phase_state") or "").strip().lower()
        if phase_state not in _TERMINAL_UNSUBMITTED_PHASE_STATES:
            return False
    return True


def _load_submit_phase_completed_iterations(
    run_dir: Path,
    *,
    infer_iteration_from_submission_path: Callable[[Path], int | None],
) -> set[int]:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return set()
    completed: set[int] = set()
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return completed
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if not _is_legacy_submit_attempt_complete(payload):
            continue
        iteration = _to_int(payload.get("iteration"))
        if iteration is None:
            sub_path = str(payload.get("sub_path") or "").strip()
            if sub_path:
                iteration = infer_iteration_from_submission_path(Path(sub_path))
        if iteration is not None and iteration > 0:
            completed.add(iteration)
    return completed


def _is_legacy_submit_attempt_complete(payload: dict[str, object]) -> bool:
    action = str(payload.get("action_taken") or "").strip().lower()
    if action in _LEGACY_SUBMIT_PHASE_COMPLETE_ACTIONS:
        return True
    if action != "skip":
        return False
    reason = str(payload.get("reason") or "").strip().lower()
    return reason in _TERMINAL_UNSUBMITTED_PHASE_STATES


def _infer_iteration_from_submit_attempt(payload: dict[str, object]) -> int | None:
    iteration = _to_int(payload.get("iteration"))
    if iteration is not None and iteration > 0:
        return iteration
    sub_path = str(payload.get("sub_path") or "").strip()
    if not sub_path:
        return None
    try:
        parts = Path(sub_path).parts
    except (TypeError, ValueError):
        return None
    for part in parts:
        if not part.startswith("iter-"):
            continue
        parsed = _to_int(part.split("-", 1)[1])
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _latest_submit_attempt_for_iteration(run_dir: Path, iteration: int) -> dict[str, object] | None:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return None
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if _infer_iteration_from_submit_attempt(payload) == iteration:
            return payload
    return None


def _load_submitted_iteration_tracking_score(
    *,
    metrics_path: Path,
    metric_direction: str,
    target_metric: str,
    load_kernel_metrics: Callable[[Path, str, str], EvaluationResult | None],
) -> float | None:
    payload = _load_json_object(metrics_path)
    if isinstance(payload, dict):
        submission_score = _to_float(payload.get("submission_score"))
        if isinstance(submission_score, float) and math.isfinite(submission_score):
            return float(submission_score)
        loop_decision = payload.get("loop_decision")
        if isinstance(loop_decision, dict):
            source = str(loop_decision.get("source") or "").strip().lower()
            if source.startswith("submission"):
                loop_value = _to_float(loop_decision.get("value"))
                if isinstance(loop_value, float) and math.isfinite(loop_value):
                    return float(loop_value)
    evaluation = load_kernel_metrics(metrics_path, metric_direction, target_metric)
    if evaluation is None:
        return None
    return float(evaluation.value)


def _resume_best_submitted_offline_score(
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
        if _update_best_score(best_score, submitted_score, metric_direction, 0.0):
            best_score = submitted_score
    return best_score


def _resume_best_submittable_iteration_state(
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
        submission_path = _resolve_iteration_submission_artifact(iter_dir)
        metrics_path = iter_dir / "metrics.json"
        if submission_path is None or not metrics_path.exists():
            continue
        evaluation = load_kernel_metrics(metrics_path, metric_direction, target_metric)
        if evaluation is None or (not iteration_metrics_allow_submit(metrics_path, evaluation)):
            continue
        if _update_best_score(best_score, evaluation.value, metric_direction, 0.0):
            best_score = evaluation.value
            best_submission = submission_path
    return best_score, best_submission


def _resume_iteration_state(
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
        _load_submit_phase_completed_iterations(
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
        submission_path = _resolve_iteration_submission_artifact(iter_dir)
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
        if best_score is None or _update_best_score(best_score, evaluation.value, metric_direction, 0.0):
            best_score = evaluation.value
            best_submission = submission_path
    if not completed_iters:
        return 1, best_score, best_submission
    next_iter = max(completed_iters) + 1
    return next_iter, best_score, best_submission


def _newest_existing_path(candidates: list[Path]) -> Path | None:
    existing: list[tuple[float, int, Path]] = []
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            stat = candidate.stat()
            existing.append((float(stat.st_mtime), int(stat.st_size), candidate))
        except OSError:
            continue
    if not existing:
        return None
    existing.sort(reverse=True)
    return existing[0][2]


def _resolve_iteration_artifact(iter_dir: Path, filename: str) -> Path | None:
    primary = _newest_existing_path(
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
    return _newest_existing_path(fallback_candidates)


def _resolve_iteration_submission_artifact(iter_dir: Path) -> Path | None:
    manifest_path = find_submission_manifest(iter_dir)
    if manifest_path is not None:
        _, submission_path, staging_dir, members = resolve_manifest_references(manifest_path)
        if submission_path is not None and submission_path.exists() and submission_path.is_file():
            return submission_path
        if staging_dir is not None or members:
            return manifest_path
    candidates: list[Path] = [iter_dir / "submission.csv", iter_dir / "output" / "submission.csv"]
    for root in (iter_dir, iter_dir / "output"):
        if not root.exists():
            continue
        try:
            for path in root.rglob("submission.*"):
                if path.is_file():
                    candidates.append(path)
        except OSError:
            continue
    return _newest_existing_path(candidates)


def _is_submit_only_metrics_payload(metrics_path: Path) -> bool:
    payload = _load_json_object(metrics_path)
    if not isinstance(payload, dict):
        return False
    return str(payload.get("kind") or "").strip().lower() == "submit_only"


def _submit_retry_metrics_candidates(iter_dir: Path, marker_payload: dict[str, object]) -> list[Path]:
    candidates: list[Path] = []
    marker_metrics_path = marker_payload.get("metrics_path")
    if isinstance(marker_metrics_path, str) and marker_metrics_path.strip():
        candidates.append(Path(marker_metrics_path))
    candidates.append(iter_dir / "metrics.json")
    resolved_path = _resolve_iteration_artifact(iter_dir, "metrics.json")
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


def _copy_submission_artifact_to_iteration_dir(*, source: Path, iter_dir: Path) -> Path:
    destination = iter_dir / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.resolve() == destination.resolve():
            return destination
    except OSError:
        pass
    shutil.copy2(source, destination)
    return destination


def _copy_kernel_support_artifacts_to_iteration_dir(*, kernel_output_dir: Path, iter_dir: Path) -> None:
    if not kernel_output_dir.exists():
        return
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("oof_predictions.csv", "split_diagnostics.json", "feature_suspects.csv"):
        source = kernel_output_dir / filename
        if not source.exists() or not source.is_file():
            continue
        destination = output_dir / filename
        try:
            if source.resolve() == destination.resolve():
                continue
        except OSError:
            pass
        shutil.copy2(source, destination)


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
        submission_path = _resolve_iteration_submission_artifact(iter_dir)
        metrics_path = _resolve_iteration_artifact(iter_dir, "metrics.json")
        if submission_path is None or metrics_path is None:
            continue
        if _is_submit_only_metrics_payload(metrics_path):
            continue
        if latest is None or iteration > latest:
            latest = iteration
    return latest


def _load_submit_retry_artifacts(
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
    latest_attempt = _latest_submit_attempt_for_iteration(run_dir, iteration)
    latest_attempt_complete = latest_attempt is not None and _is_legacy_submit_attempt_complete(latest_attempt)
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

    submission_path = _resolve_iteration_submission_artifact(iter_dir)
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


def _load_run_state(run_dir: Path) -> dict[str, object]:
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        attempted = _has_submit_attempt_records(run_dir)
        return {"submit_attempted": attempted, "submit_ok": False}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("submit_attempted"):
        payload["submit_attempted"] = _has_submit_attempt_records(run_dir)
    if "last_submit_fingerprint" not in payload and payload.get("last_fingerprint"):
        payload["last_submit_fingerprint"] = payload.get("last_fingerprint")
    if "last_fingerprint" not in payload and payload.get("last_submit_fingerprint"):
        payload["last_fingerprint"] = payload.get("last_submit_fingerprint")
    if bool(payload.get("submit_attempted")) and not bool(payload.get("submit_ok")):
        if _has_successful_submit_attempt(run_dir):
            payload["submit_ok"] = True
    return payload


def _save_run_state(run_dir: Path, updates: dict[str, object]) -> None:
    state = _load_run_state(run_dir)
    state.update(updates)
    state["submit_attempted"] = bool(state.get("submit_attempted")) or _has_submit_attempt_records(run_dir)
    state["submit_ok"] = bool(state.get("submit_ok")) or _has_successful_submit_attempt(run_dir)
    state["updated_at"] = datetime.now(UTC).isoformat()
    state_path = run_dir / "run_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _has_submit_attempt_records(run_dir: Path) -> bool:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return False
    try:
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return True
    except OSError:
        return False
    return False


def _has_successful_submit_attempt(run_dir: Path) -> bool:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return False
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and bool(payload.get("ok")):
            return True
    return False


def _count_successful_submit_attempts(run_dir: Path) -> int:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return 0
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    count = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if not bool(payload.get("ok")):
            continue
        action_taken = str(payload.get("action_taken") or "").strip().lower()
        if action_taken and action_taken != "submit":
            continue
        count += 1
    return count


def _load_submit_fingerprints(run_dir: Path) -> list[str]:
    fingerprints: list[str] = []
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return fingerprints
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return fingerprints
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fingerprint = str(row.get("fingerprint") or "").strip()
        if not fingerprint:
            continue
        fingerprints.append(fingerprint)
    return fingerprints


def _load_latest_submit_attempt(run_dir: Path) -> dict[str, object]:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return {}
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
