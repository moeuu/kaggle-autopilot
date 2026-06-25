from __future__ import annotations

import json
import re

from kagglebot.json_utils import load_json_object
from kagglebot.paths import CompetitionPaths
from kagglebot.scalar_utils import parse_finite_float, parse_int


def _to_float(value: object) -> float | None:
    return parse_finite_float(value, allow_commas=True)


def _to_int(value: object) -> int | None:
    return parse_int(value, allow_commas=True, allow_float=True, require_integral_float=False)


def _update_best_score(best: float | None, current: float, direction: str, min_improvement: float) -> bool:
    if best is None:
        return True
    eps = 1e-9
    if direction == "minimize":
        improvement = best - current
        return improvement >= (min_improvement - eps)
    improvement = current - best
    return improvement >= (min_improvement - eps)


def _resume_best_online_submission_score(
    *,
    paths: CompetitionPaths,
    run_id: str,
    direction: str,
    max_iterations: int,
) -> float | None:
    best: float | None = None
    for iteration in range(1, max_iterations + 1):
        metrics_path = paths.iter_dir(run_id, iteration) / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = load_json_object(metrics_path)
        if payload is None:
            continue
        score = _to_float(payload.get("submission_score"))
        if score is None:
            continue
        if _update_best_score(best, score, direction, 0.0):
            best = score
    return best


def _should_force_major_overhaul_by_rank(
    *,
    rank: int | None,
    total_teams: int | None,
    max_percentile: float,
    min_teams: int,
) -> bool:
    if rank is None or total_teams is None:
        return False
    if total_teams < max(1, min_teams):
        return False
    if rank <= 0:
        return False
    return (rank / total_teams) > max_percentile


def _meets_rank_percentile_target(
    *,
    rank_percentile: float | None,
    estimated_rank_percentile: float | None,
    target_rank_percentile: float | None,
) -> bool:
    if target_rank_percentile is None:
        return False
    for candidate in (rank_percentile, estimated_rank_percentile):
        if candidate is not None and candidate <= target_rank_percentile:
            return True
    return False


def _build_medal_target_reason(
    *,
    target_medal: str | None,
    target_rank_percentile: float | None,
    rank_percentile: float | None,
    estimated_rank_percentile: float | None,
) -> str | None:
    if target_rank_percentile is None:
        return None
    target_text = f"top {target_rank_percentile * 100:.2f}%"
    if target_medal:
        target_text = f"{target_medal} target ({target_text})"
    observed = rank_percentile if rank_percentile is not None else estimated_rank_percentile
    if observed is None:
        return (
            "Medal-aware search policy is active. "
            f"Keep exploration broader until the competition reaches the {target_text} band."
        )
    return (
        "Medal-aware search policy is active. "
        f"Current leaderboard percentile is {(observed * 100):.2f}%, so keep exploring until {target_text} is met."
    )


def _iter_nested_mappings(payload: object):
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _iter_nested_mappings(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_nested_mappings(item)


def _extract_orig_proba_signal(kernel_metrics_payload: dict[str, object] | None) -> dict[str, object] | None:
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


def _extract_pseudo_label_failure_signal(
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
                value = _to_int(node.get(accepted_key))
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
                value = _to_int(node.get(total_key))
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


def _extract_missing_ensemble_signal(kernel_metrics_payload: dict[str, object] | None) -> dict[str, object] | None:
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


def _extract_original_data_unused_signal(
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


def _extract_same_family_plateau_signal(kernel_metrics_payload: dict[str, object] | None) -> dict[str, object] | None:
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


def _detect_online_mismatch_signal(
    *,
    previous_best_offline: float | None,
    current_offline: float,
    previous_best_online: float | None,
    current_online: float | None,
    direction: str,
) -> dict[str, object] | None:
    if previous_best_offline is None or previous_best_online is None or current_online is None:
        return None
    offline_improved = _update_best_score(previous_best_offline, current_offline, direction, 0.0)
    online_improved = _update_best_score(previous_best_online, current_online, direction, 0.0)
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


def _requires_tabular_multi_family_policy(dataset_profile: dict[str, object] | None) -> bool:
    profile = dataset_profile or {}
    modality = str(profile.get("modality") or "").strip().lower()
    task = str(profile.get("task") or "").strip().lower()
    tags_raw = profile.get("tags")
    tags = (
        [str(item).strip().lower() for item in tags_raw if isinstance(item, str)] if isinstance(tags_raw, list) else []
    )
    train_rows = _to_int(profile.get("train_rows")) or 0
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
