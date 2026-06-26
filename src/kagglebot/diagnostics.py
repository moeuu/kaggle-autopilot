from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot import submission_policy

if TYPE_CHECKING:
    from kagglebot.solver.evaluate import EvaluationResult


def diagnostics_path_for_iteration(iter_dir: Path) -> Path:
    return iter_dir / "diagnostics.md"


def write_iteration_diagnostics(*, iter_dir: Path, diagnostics: str) -> Path:
    path = diagnostics_path_for_iteration(iter_dir)
    path.write_text(diagnostics, encoding="utf-8")
    return path


def load_iteration_diagnostics_text(iter_dir: Path) -> str:
    path = diagnostics_path_for_iteration(iter_dir)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def pipeline_config_hash(*, model_summary: dict[str, object], metric: str, accelerator: str) -> str:
    stable_payload: dict[str, object] = {
        "metric": metric,
        "accelerator": accelerator,
    }
    for key, value in model_summary.items():
        if key in {"evaluation_by_source", "timing", "elapsed", "duration"}:
            continue
        stable_payload[key] = value
    encoded = json.dumps(stable_payload, sort_keys=True, default=diagnostics_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def diagnostics_json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted([str(item) for item in obj])
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    for attr in ("item", "tolist"):
        func = getattr(obj, attr, None)
        if callable(func):
            try:
                return func()
            except Exception:  # noqa: BLE001
                break
    return {
        "__type__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
        "__repr__": repr(obj),
    }


def build_diagnostics(
    *,
    evaluation: EvaluationResult,
    model_summary: dict[str, object],
    best_score: float | None,
    target_score: float,
    dataset_profile: dict[str, object],
    top1_score: float | None,
    top1_tier: bool,
    diff_summary: str,
    evaluation_by_source: dict[str, EvaluationResult] | None = None,
    loop_decision_score: float | None = None,
    loop_decision_source: str = "offline",
    quality_guard: dict[str, object] | None = None,
    accuracy_potential: dict[str, object] | None = None,
) -> str:
    direction = evaluation.direction
    decision_score = evaluation.value if loop_decision_score is None else loop_decision_score
    delta_to_target = target_score - decision_score if direction == "minimize" else decision_score - target_score
    best_line = best_score if best_score is not None else decision_score
    trend = (
        "improving"
        if best_score is None or submission_policy.meets_target(decision_score, best_line, direction)
        else "stalled"
    )
    top1_delta = None
    if top1_score is not None:
        top1_delta = top1_score - decision_score if direction == "minimize" else decision_score - top1_score
    gap = None
    if evaluation.train_score is not None and evaluation.val_score is not None:
        gap = evaluation.train_score - evaluation.val_score

    dataset_lines = []
    if dataset_profile:
        dataset_lines = [
            f"- Train rows/cols: {dataset_profile.get('train_rows')} / {dataset_profile.get('train_cols')}",
            f"- Test rows/cols: {dataset_profile.get('test_rows')} / {dataset_profile.get('test_cols')}",
            f"- Missingness: {dataset_profile.get('missingness')}",
            f"- Categorical cols: {len(dataset_profile.get('categorical_columns', []))}",
            f"- High-cardinality cols: {len(dataset_profile.get('high_cardinality_columns', []))}",
        ]
    else:
        dataset_lines = ["- Dataset profile unavailable."]

    lines = [
        "# Diagnostics",
        "",
        f"Loop decision: source={loop_decision_source} score={decision_score:.6f}",
        f"Score vs target: {decision_score:.6f} vs {target_score:.6f} (delta {delta_to_target:.6f})",
        f"Best so far: {best_line:.6f} ({trend})",
        f"Evaluation: {evaluation.score_source}",
    ]
    if evaluation_by_source:
        lines.append(
            "Offline by source: "
            + ", ".join(f"{source}={result.value:.6f}" for source, result in evaluation_by_source.items())
        )
    else:
        lines.append(f"Offline ({evaluation.score_source}): {evaluation.value:.6f}")
    if top1_score is None:
        lines.append("Top1 public score: unavailable")
    else:
        lines.append(f"Top1 public score: {top1_score:.6f} (delta {top1_delta:.6f}, top1-tier={top1_tier})")
    if gap is not None:
        lines.append(f"Train/val gap: {gap:.6f}")
    if evaluation.std is not None:
        lines.append(f"CV std: {evaluation.std:.6f}")
    if quality_guard:
        reasons = quality_guard.get("reasons")
        warning_values = quality_guard.get("warnings")
        faithfulness = quality_guard.get("competition_faithfulness")
        reason_text = (
            ", ".join(str(item) for item in reasons if isinstance(item, str))
            if isinstance(reasons, list) and reasons
            else "none"
        )
        warning_text = (
            ", ".join(str(item) for item in warning_values if isinstance(item, str))
            if isinstance(warning_values, list) and warning_values
            else "none"
        )
        lines.append(
            "Kernel quality guard: "
            f"allow_submit={bool(quality_guard.get('allow_submit', True))}, "
            f"reasons={reason_text}, warnings={warning_text}"
        )
        if isinstance(faithfulness, dict):
            lines.append(
                "Competition faithfulness: "
                f"faithful={bool(faithfulness.get('faithful', False))}, "
                f"metric={faithfulness.get('actual_metric') or 'unknown'}"
                f"/{faithfulness.get('expected_metric') or 'unknown'}, "
                f"split={faithfulness.get('actual_split_strategy') or 'unknown'}"
                f"/{faithfulness.get('expected_split_strategy') or 'unknown'}, "
                f"dataset_mode={faithfulness.get('dataset_mode') or 'unknown'}"
            )
    if accuracy_potential:
        lines.append(
            "Accuracy frontier: "
            f"status={accuracy_potential.get('status')}, "
            f"eligible={bool(accuracy_potential.get('eligible', False))}, "
            f"capacity_tier={accuracy_potential.get('capacity_tier')}, "
            f"data_tier={accuracy_potential.get('data_tier')}, "
            f"reason={accuracy_potential.get('primary_reason')}"
        )
    lines += [
        "",
        "Dataset summary:",
        *dataset_lines,
        "",
        "Pipeline summary:",
        json.dumps(model_summary, indent=2, default=diagnostics_json_default),
        "",
        "Suspected causes:",
        "- Underfit if train/val both low; overfit if gap large.",
        "- Check categorical encoding, leakage, and missing value handling.",
        "",
        "Next improvements (ranked):",
        "1) Try a stronger model or tuning.",
        "2) Add features or target transformations.",
        "3) Adjust validation strategy.",
        "",
        "Diff summary:",
        diff_summary or "No code changes.",
    ]
    return "\n".join(lines) + "\n"
