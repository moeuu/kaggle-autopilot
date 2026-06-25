from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from kagglebot.scalar_utils import tolerant_int
from kagglebot.score_utils import should_update_best_score


@dataclass(frozen=True)
class IterationRepairSignalPolicyDecision:
    extra_policy_notes: list[str]
    minimum_improvement_mode: str | None
    minimum_improvement_reason: str | None
    force_major_overhaul: bool
    forced_major_overhaul_reason: str | None
    forced_validation_redesign_reason: str | None
    loop_signal_errors: list[dict[str, object]]
    loop_signal_problems: list[dict[str, object]]
    repair_signals: dict[str, object] | None
    next_iteration_policy: dict[str, object]


@dataclass(frozen=True)
class IterationRepairSignals:
    orig_proba_signal: dict[str, object] | None
    original_data_unused_signal: dict[str, object] | None
    pseudo_label_signal: dict[str, object] | None
    missing_ensemble_signal: dict[str, object] | None
    same_family_plateau_signal: dict[str, object] | None
    subgroup_collapse_signal: dict[str, object] | None
    online_mismatch_signal: dict[str, object] | None
    online_history_regression_signal: dict[str, object] | None


def extract_orig_proba_signal(kernel_metrics_payload: dict[str, object] | None) -> dict[str, object] | None:
    payload = kernel_metrics_payload or {}
    if not isinstance(payload, dict):
        return None
    original_data_found = payload.get("original_data_found")
    feature_status = str(payload.get("orig_proba_feature_status") or "").strip().lower()
    constant_cols_raw = payload.get("orig_proba_constant_cols")
    constant_cols = (
        [str(item) for item in constant_cols_raw if isinstance(item, str)]
        if isinstance(constant_cols_raw, list)
        else []
    )
    if original_data_found is not False and feature_status != "constant_fallback" and not constant_cols:
        return None
    informative_cols_raw = payload.get("orig_proba_informative_cols")
    informative_cols = (
        [str(item) for item in informative_cols_raw if isinstance(item, str)]
        if isinstance(informative_cols_raw, list)
        else []
    )
    return {
        "original_data_found": bool(original_data_found) if isinstance(original_data_found, bool) else None,
        "feature_status": feature_status or None,
        "constant_cols": constant_cols,
        "informative_cols": informative_cols,
        "note": (
            "External/original-data signal collapsed: "
            f"original_data_found={original_data_found}, "
            f"orig_proba_feature_status={feature_status or 'unknown'}, "
            f"constant_cols={len(constant_cols)}. "
            "Next iteration must recover the required reference/original dataset inputs from "
            "context/reference_inputs_manifest.json and stage them under context/reference_inputs "
            "before keeping ORIG_proba-style features."
        ),
    }


def extract_pseudo_label_failure_signal(
    *,
    kernel_metrics_payload: dict[str, object] | None,
    diagnostics_text: str,
) -> dict[str, object] | None:
    payload = kernel_metrics_payload or {}
    haystacks: list[str] = []
    if isinstance(payload, dict) and payload:
        try:
            haystacks.append(json.dumps(payload, ensure_ascii=False))
        except TypeError:
            pass
    if diagnostics_text.strip():
        haystacks.append(diagnostics_text)

    accepted: int | None = None
    total: int | None = None
    pseudo_detected = False
    for node in _iter_nested_mappings(payload):
        keys = [str(key).strip().lower() for key in node.keys()]
        if any("pseudo" in key for key in keys):
            pseudo_detected = True
            for accepted_key in (
                "accepted",
                "accepted_count",
                "accepted_folds",
                "accepted_candidates",
                "pseudo_label_accepted_count",
            ):
                value = tolerant_int(node.get(accepted_key))
                if value is not None:
                    accepted = value
                    break
            for total_key in (
                "total",
                "total_count",
                "total_folds",
                "candidate_count",
                "attempted",
                "attempted_count",
                "pseudo_label_total_count",
            ):
                value = tolerant_int(node.get(total_key))
                if value is not None and value > 0:
                    total = value
                    break
    joined_text = "\n".join(haystacks)
    if "pseudo" in joined_text.lower():
        pseudo_detected = True
    match = re.search(r"pseudo[- ]?label[^\n]{0,120}?(\d+)\s*/\s*(\d+)\s+accepted", joined_text, flags=re.IGNORECASE)
    if match:
        accepted = int(match.group(1))
        total = int(match.group(2))
    generic_match = re.search(r"\b(\d+)\s*/\s*(\d+)\s+accepted\b", joined_text, flags=re.IGNORECASE)
    if generic_match and pseudo_detected and total is None:
        accepted = int(generic_match.group(1))
        total = int(generic_match.group(2))
    if not pseudo_detected or accepted is None or total is None or total <= 0 or accepted > 0:
        return None
    return {
        "accepted": accepted,
        "total": total,
        "note": (
            f"Pseudo-labeling yielded {accepted}/{total} accepted folds or candidates. "
            "Disable pseudo-labeling in the next iteration and increase model-family diversity plus OOF blending "
            "instead of spending another pass on the same pseudo-label path."
        ),
    }


def extract_missing_ensemble_signal(kernel_metrics_payload: dict[str, object] | None) -> dict[str, object] | None:
    payload = kernel_metrics_payload or {}
    if not isinstance(payload, dict):
        return None
    model_families_raw = payload.get("model_families")
    model_families = (
        sorted({str(item).strip().lower() for item in model_families_raw if str(item).strip()})
        if isinstance(model_families_raw, list)
        else []
    )
    if len(model_families) < 2:
        return None
    blend_method = str(payload.get("blend_method") or payload.get("final_kind") or "").strip().lower()
    component_models_raw = payload.get("component_models")
    component_models = (
        [str(item).strip() for item in component_models_raw if str(item).strip()]
        if isinstance(component_models_raw, list)
        else []
    )
    if blend_method in {"blend", "rank_blend", "weighted_blend"} and len(component_models) >= 2:
        return None
    return {
        "model_families": model_families,
        "component_models": component_models,
        "note": (
            "Multiple model families were trained but no OOF blend was selected or emitted. "
            f"model_families={model_families}. "
            "Next iteration must keep heterogeneous pipelines and produce a weighted or rank OOF blend candidate."
        ),
    }


def extract_original_data_unused_signal(
    *,
    kernel_metrics_payload: dict[str, object] | None,
    reference_inputs_manifest_payload: dict[str, object] | None,
) -> dict[str, object] | None:
    manifest = reference_inputs_manifest_payload or {}
    payload = kernel_metrics_payload or {}
    if not isinstance(manifest, dict) or not isinstance(payload, dict):
        return None
    required_datasets_raw = manifest.get("required_datasets")
    required_datasets = (
        [str(item).strip() for item in required_datasets_raw if str(item).strip()]
        if isinstance(required_datasets_raw, list)
        else []
    )
    reference_notebooks = manifest.get("reference_notebooks")
    staged_dataset_count = 0
    if isinstance(reference_notebooks, list):
        for notebook in reference_notebooks:
            if not isinstance(notebook, dict):
                continue
            staged_sources = notebook.get("staged_sources")
            if not isinstance(staged_sources, list):
                continue
            for item in staged_sources:
                if not isinstance(item, dict):
                    continue
                if str(item.get("kind") or "").strip().lower() == "dataset":
                    staged_dataset_count += 1
    if not required_datasets and staged_dataset_count == 0:
        return None
    original_data_found = payload.get("original_data_found")
    external_data_used = payload.get("external_data_used")
    if original_data_found is True or external_data_used is True:
        return None
    return {
        "required_datasets": required_datasets,
        "staged_dataset_count": staged_dataset_count,
        "note": (
            "Reference/original datasets were staged but the kernel did not use them. "
            f"required_datasets={required_datasets}, staged_dataset_count={staged_dataset_count}, "
            f"original_data_found={original_data_found}, external_data_used={external_data_used}. "
            "Next iteration must wire the staged original data into feature generation instead of "
            "falling back to competition-only features."
        ),
    }


def extract_same_family_plateau_signal(kernel_metrics_payload: dict[str, object] | None) -> dict[str, object] | None:
    payload = kernel_metrics_payload or {}
    if not isinstance(payload, dict):
        return None
    model_families_raw = payload.get("model_families")
    model_families = (
        sorted({str(item).strip().lower() for item in model_families_raw if str(item).strip()})
        if isinstance(model_families_raw, list)
        else []
    )
    if len(model_families) != 1:
        return None
    pipelines_raw = payload.get("pipelines")
    pipeline_count = len(pipelines_raw) if isinstance(pipelines_raw, list) else 0
    selected_pipeline = str(payload.get("selected_pipeline") or payload.get("final_pipeline") or "").strip()
    return {
        "model_family": model_families[0],
        "pipeline_count": pipeline_count,
        "selected_pipeline": selected_pipeline,
        "note": (
            "Search remains stuck in a same-family plateau. "
            f"model_family={model_families[0]}, pipeline_count={pipeline_count}, "
            f"selected_pipeline={selected_pipeline or 'unknown'}. "
            "Next iteration must add an orthogonal family instead of spending another pass on same-family tuning."
        ),
    }


def detect_online_mismatch_signal(
    *,
    previous_best_offline: float | None,
    current_offline: float,
    previous_best_online: float | None,
    current_online: float | None,
    direction: str,
) -> dict[str, object] | None:
    if previous_best_offline is None or previous_best_online is None or current_online is None:
        return None
    offline_improved = should_update_best_score(previous_best_offline, current_offline, direction, 0.0)
    online_improved = should_update_best_score(previous_best_online, current_online, direction, 0.0)
    if not offline_improved or online_improved:
        return None
    return {
        "previous_best_offline": previous_best_offline,
        "current_offline": current_offline,
        "previous_best_online": previous_best_online,
        "current_online": current_online,
        "note": (
            "Offline score improved but public leaderboard regressed. "
            f"offline: {previous_best_offline:.6f} -> {current_offline:.6f}; "
            f"online: {previous_best_online:.6f} -> {current_online:.6f}. "
            "Treat this as a major online mismatch: ban same-family-only tuning next iteration and require "
            "model-family diversification plus OOF blend exploration."
        ),
    }


def collect_iteration_repair_signals(
    *,
    kernel_metrics_payload: dict[str, object] | None,
    diagnostics_text: str,
    reference_inputs_manifest_payload: dict[str, object] | None,
    enable_missing_ensemble_signal: bool,
    enable_original_data_unused_signal: bool,
    enable_same_family_plateau_signal: bool,
    direction: str,
    previous_best_offline: float | None,
    current_offline: float,
    previous_best_online: float | None,
    current_online: float | None,
    previous_submission_history: dict[str, object],
    detect_subgroup_collapse_signal: Callable[..., dict[str, object] | None],
    detect_online_history_regression_signal: Callable[..., dict[str, object] | None],
) -> IterationRepairSignals:
    return IterationRepairSignals(
        orig_proba_signal=extract_orig_proba_signal(kernel_metrics_payload),
        original_data_unused_signal=(
            extract_original_data_unused_signal(
                kernel_metrics_payload=kernel_metrics_payload,
                reference_inputs_manifest_payload=reference_inputs_manifest_payload,
            )
            if enable_original_data_unused_signal
            else None
        ),
        pseudo_label_signal=extract_pseudo_label_failure_signal(
            kernel_metrics_payload=kernel_metrics_payload,
            diagnostics_text=diagnostics_text,
        ),
        missing_ensemble_signal=(
            extract_missing_ensemble_signal(kernel_metrics_payload) if enable_missing_ensemble_signal else None
        ),
        same_family_plateau_signal=(
            extract_same_family_plateau_signal(kernel_metrics_payload) if enable_same_family_plateau_signal else None
        ),
        subgroup_collapse_signal=detect_subgroup_collapse_signal(
            kernel_metrics_payload=kernel_metrics_payload,
            direction=direction,
        ),
        online_mismatch_signal=detect_online_mismatch_signal(
            previous_best_offline=previous_best_offline,
            current_offline=current_offline,
            previous_best_online=previous_best_online,
            current_online=current_online,
            direction=direction,
        ),
        online_history_regression_signal=detect_online_history_regression_signal(
            previous_best_online=previous_best_online,
            current_online=current_online,
            direction=direction,
            history=previous_submission_history,
        ),
    )


def apply_iteration_repair_signal_policy(
    *,
    iteration: int,
    orig_proba_signal: dict[str, object] | None,
    original_data_unused_signal: dict[str, object] | None,
    pseudo_label_signal: dict[str, object] | None,
    missing_ensemble_signal: dict[str, object] | None,
    same_family_plateau_signal: dict[str, object] | None,
    subgroup_collapse_signal: dict[str, object] | None,
    online_mismatch_signal: dict[str, object] | None,
    online_history_regression_signal: dict[str, object] | None,
    minimum_improvement_mode: str | None,
    minimum_improvement_reason: str | None,
    force_major_overhaul: bool,
    forced_major_overhaul_reason: str | None,
    prefer_validation_redesign: bool,
    upgrade_improvement_mode: Callable[[str, str], str],
) -> IterationRepairSignalPolicyDecision:
    extra_policy_notes: list[str] = []
    minimum_improvement_mode_next = minimum_improvement_mode
    minimum_improvement_reason_next = minimum_improvement_reason
    forced_validation_redesign_reason: str | None = None
    loop_signal_errors: list[dict[str, object]] = []
    loop_signal_problems: list[dict[str, object]] = []

    def _append_note_reason(note: object) -> None:
        nonlocal minimum_improvement_reason_next
        minimum_improvement_reason_next = (
            f"{minimum_improvement_reason_next} {note}".strip() if minimum_improvement_reason_next else str(note)
        )

    def _upgrade_to(mode: str) -> None:
        nonlocal minimum_improvement_mode_next
        minimum_improvement_mode_next = upgrade_improvement_mode(
            minimum_improvement_mode_next or "minor_tuning",
            mode,
        )

    def _append_major_reason(note: object) -> None:
        nonlocal force_major_overhaul, forced_major_overhaul_reason
        force_major_overhaul = True
        forced_major_overhaul_reason = (
            f"{forced_major_overhaul_reason} {note}".strip() if forced_major_overhaul_reason else str(note)
        )

    if orig_proba_signal is not None:
        note = str(orig_proba_signal["note"])
        extra_policy_notes.append(note)
        _upgrade_to("moderate_update")
        _append_note_reason(note)
        loop_signal_errors.append(
            {
                "iteration": iteration,
                "error_message": (
                    "ORIG_proba external signal fell back to constants because original data was unavailable."
                ),
                "fix_summary": note,
                "resolved": False,
                "outcome_bucket": "unknown",
            }
        )
    if original_data_unused_signal is not None:
        note = str(original_data_unused_signal["note"])
        extra_policy_notes.append(note)
        _upgrade_to("moderate_update")
        _append_note_reason(note)
        loop_signal_errors.append(
            {
                "iteration": iteration,
                "error_message": "Original/reference datasets were staged but never consumed by the kernel.",
                "fix_summary": note,
                "resolved": False,
                "outcome_bucket": "unknown",
            }
        )
    if pseudo_label_signal is not None:
        note = str(pseudo_label_signal["note"])
        extra_policy_notes.append(note)
        _upgrade_to("moderate_update")
        _append_note_reason(note)
        loop_signal_errors.append(
            {
                "iteration": iteration,
                "error_message": (
                    f"Pseudo-labeling yielded {int(pseudo_label_signal['accepted'])}/"
                    f"{int(pseudo_label_signal['total'])} accepted folds or candidates."
                ),
                "fix_summary": note,
                "resolved": False,
                "outcome_bucket": "unknown",
            }
        )
    if missing_ensemble_signal is not None:
        note = str(missing_ensemble_signal["note"])
        extra_policy_notes.append(note)
        _append_major_reason(note)
        loop_signal_problems.append(
            {
                "iteration": iteration,
                "why_poor": note,
                "how_improved": "Keep heterogeneous families and emit at least one weighted/rank OOF blend.",
                "delta_offline": None,
                "outcome_bucket": "low",
            }
        )
    if same_family_plateau_signal is not None:
        note = str(same_family_plateau_signal["note"])
        extra_policy_notes.append(note)
        _append_major_reason(note)
        loop_signal_problems.append(
            {
                "iteration": iteration,
                "why_poor": note,
                "how_improved": "Add an orthogonal family instead of repeating same-family tuning.",
                "delta_offline": None,
                "outcome_bucket": "low",
            }
        )
    if subgroup_collapse_signal is not None:
        note = str(subgroup_collapse_signal["note"])
        extra_policy_notes.append(note)
        _upgrade_to("moderate_update")
        _append_note_reason(note)
        loop_signal_problems.append(
            {
                "iteration": iteration,
                "why_poor": note,
                "how_improved": (
                    "Make pipeline and fallback selection subgroup-aware at (model_id,node_type) granularity, "
                    "and target the collapsed slice before broad model-family tuning."
                ),
                "delta_offline": None,
                "outcome_bucket": "low",
            }
        )
    if online_mismatch_signal is not None:
        note = str(online_mismatch_signal["note"])
        extra_policy_notes.append(note)
        if prefer_validation_redesign:
            _upgrade_to("validation_redesign")
            forced_validation_redesign_reason = note
        else:
            _append_major_reason(note)
        loop_signal_problems.append(
            {
                "iteration": iteration,
                "why_poor": note,
                "how_improved": (
                    "Ban same-family-only tuning after an online mismatch and require model-family "
                    "diversification plus OOF blending."
                ),
                "delta_offline": None,
                "outcome_bucket": "low",
            }
        )
    if online_history_regression_signal is not None:
        note = str(online_history_regression_signal["note"])
        extra_policy_notes.append(note)
        if prefer_validation_redesign:
            _upgrade_to("validation_redesign")
            forced_validation_redesign_reason = note
        else:
            _append_major_reason(note)
        loop_signal_problems.append(
            {
                "iteration": iteration,
                "why_poor": forced_major_overhaul_reason or note,
                "how_improved": (
                    "Use the best historical public-score submission as the baseline and require a different "
                    "model/feature/blend path before submitting another regressed artifact."
                ),
                "delta_offline": None,
                "outcome_bucket": "low",
            }
        )

    repair_signals = (
        {
            "orig_proba_constant_fallback": orig_proba_signal,
            "original_data_unused": original_data_unused_signal,
            "pseudo_label_failure": pseudo_label_signal,
            "missing_ensemble": missing_ensemble_signal,
            "same_family_plateau": same_family_plateau_signal,
            "subgroup_collapse": subgroup_collapse_signal,
            "online_mismatch": online_mismatch_signal,
            "online_history_regression": online_history_regression_signal,
        }
        if extra_policy_notes
        else None
    )
    next_iteration_policy = {
        "minimum_improvement_mode": minimum_improvement_mode_next,
        "minimum_improvement_reason": minimum_improvement_reason_next,
        "forced_improvement_mode": (
            "validation_redesign"
            if forced_validation_redesign_reason and not force_major_overhaul
            else "major_overhaul"
            if force_major_overhaul
            else None
        ),
        "forced_improvement_reason": forced_major_overhaul_reason or forced_validation_redesign_reason,
        "extra_policy_notes": extra_policy_notes,
    }
    return IterationRepairSignalPolicyDecision(
        extra_policy_notes=extra_policy_notes,
        minimum_improvement_mode=minimum_improvement_mode_next,
        minimum_improvement_reason=minimum_improvement_reason_next,
        force_major_overhaul=force_major_overhaul,
        forced_major_overhaul_reason=forced_major_overhaul_reason,
        forced_validation_redesign_reason=forced_validation_redesign_reason,
        loop_signal_errors=loop_signal_errors,
        loop_signal_problems=loop_signal_problems,
        repair_signals=repair_signals,
        next_iteration_policy=next_iteration_policy,
    )


def record_iteration_repair_signal_knowledge(
    *,
    knowledge_paths: object,
    slug: str,
    run_id: str,
    iteration: int,
    problem_types: list[str],
    loop_signal_errors: list[dict[str, object]],
    loop_signal_problems: list[dict[str, object]],
    submission_score: float | None,
    record_error_fix_insight: Callable[..., object],
    record_problem_type_insight: Callable[..., object],
) -> tuple[int, int]:
    recorded_errors = 0
    recorded_problems = 0
    for issue in loop_signal_errors:
        try:
            record_error_fix_insight(
                knowledge_paths=knowledge_paths,
                slug=slug,
                run_id=run_id,
                iteration=int(issue.get("iteration") or iteration),
                problem_types=problem_types,
                error_message=str(issue.get("error_message") or ""),
                fix_summary=str(issue.get("fix_summary") or ""),
                resolved=bool(issue.get("resolved")),
                outcome_bucket=str(issue.get("outcome_bucket") or "unknown"),
                submission_score=submission_score,
            )
        except Exception:  # noqa: BLE001
            continue
        recorded_errors += 1
    for issue in loop_signal_problems:
        try:
            record_problem_type_insight(
                knowledge_paths=knowledge_paths,
                slug=slug,
                run_id=run_id,
                iteration=int(issue.get("iteration") or iteration),
                problem_types=problem_types,
                why_poor=str(issue.get("why_poor") or ""),
                how_improved=str(issue.get("how_improved") or ""),
                delta_offline=None,
                outcome_bucket=str(issue.get("outcome_bucket") or "unknown"),
                submission_score=submission_score,
            )
        except Exception:  # noqa: BLE001
            continue
        recorded_problems += 1
    return recorded_errors, recorded_problems


def requires_tabular_multi_family_policy(dataset_profile: dict[str, object] | None) -> bool:
    profile = dataset_profile or {}
    modality = str(profile.get("modality") or "").strip().lower()
    task = str(profile.get("task") or "").strip().lower()
    tags_raw = profile.get("tags")
    tags = (
        [str(item).strip().lower() for item in tags_raw if isinstance(item, str)] if isinstance(tags_raw, list) else []
    )
    train_rows = tolerant_int(profile.get("train_rows")) or 0
    categorical_count = (
        len(profile.get("categorical_columns", [])) if isinstance(profile.get("categorical_columns"), list) else 0
    )
    high_cardinality_count = (
        len(profile.get("high_cardinality_columns", []))
        if isinstance(profile.get("high_cardinality_columns"), list)
        else 0
    )
    binary_like = task == "binary" or "binary" in tags or (task == "classification" and "multiclass" not in tags)
    mixed_categorical = categorical_count >= 3 or high_cardinality_count >= 1
    return modality == "tabular" and binary_like and train_rows >= 5000 and mixed_categorical


def _iter_nested_mappings(payload: object):
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _iter_nested_mappings(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_nested_mappings(item)
