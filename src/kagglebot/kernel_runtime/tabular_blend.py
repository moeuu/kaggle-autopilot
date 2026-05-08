from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - local kernel import path
    from .tabular_ensemble import PipelineResult, clip_predictions, safe_auc
except ImportError:  # pragma: no cover - direct sys.path import from artifact kernels
    from tabular_ensemble import PipelineResult, clip_predictions, safe_auc


def seed_prediction_map(result: PipelineResult, key: str) -> dict[int, np.ndarray]:
    raw = result.metadata.get(key, {})
    if not isinstance(raw, dict):
        return {}
    parsed: dict[int, np.ndarray] = {}
    for seed, preds in raw.items():
        try:
            parsed[int(seed)] = np.asarray(preds, dtype=np.float64)
        except Exception:
            continue
    return parsed


def seed_score_rows(result: PipelineResult) -> list[dict[str, Any]]:
    raw = result.metadata.get("seed_scores", [])
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            rows.append({"seed": int(item["seed"]), "auc": float(item["auc"])})
        except Exception:
            continue
    return rows


def should_select_blend_candidate(
    blend_result: PipelineResult,
    best_single: PipelineResult,
    *,
    min_margin: float,
) -> bool:
    if float(blend_result.cv_score) <= float(best_single.cv_score) + min_margin:
        return False
    blend_seed_scores = {row["seed"]: row["auc"] for row in seed_score_rows(blend_result)}
    single_seed_scores = {row["seed"]: row["auc"] for row in seed_score_rows(best_single)}
    common_seeds = sorted(set(blend_seed_scores) & set(single_seed_scores))
    if not common_seeds:
        return True
    not_worse_count = sum(
        1 for seed in common_seeds if float(blend_seed_scores[seed]) >= float(single_seed_scores[seed]) - 1e-12
    )
    required_not_worse = 2 if len(common_seeds) >= 3 else max(1, (len(common_seeds) + 1) // 2)
    return not_worse_count >= required_not_worse


def rank_average(*arrays: np.ndarray) -> np.ndarray:
    stacked = np.vstack([np.asarray(arr, dtype=np.float64) for arr in arrays])
    ranked = np.zeros_like(stacked, dtype=np.float64)
    for idx, arr in enumerate(stacked):
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(arr), dtype=np.float64)
        ranked[idx] = ranks / max(1, len(arr) - 1)
    return ranked.mean(axis=0)


def weighted_logit_average(first: np.ndarray, second: np.ndarray, first_weight: float) -> np.ndarray:
    second_weight = 1.0 - first_weight
    first_logit = np.log(clip_predictions(first) / (1.0 - clip_predictions(first)))
    second_logit = np.log(clip_predictions(second) / (1.0 - clip_predictions(second)))
    blended = first_weight * first_logit + second_weight * second_logit
    return 1.0 / (1.0 + np.exp(-blended))


def _normalize_weights(component_weights: dict[str, float]) -> dict[str, float]:
    positive_weights = {name: max(float(weight), 0.0) for name, weight in component_weights.items()}
    total = float(sum(positive_weights.values()))
    if total <= 0.0:
        raise ValueError("Blend weights must sum to a positive value.")
    return {name: weight / total for name, weight in positive_weights.items()}


def _weighted_rank_average(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    stacked = np.vstack([np.asarray(arr, dtype=np.float64) for arr in arrays])
    ranked = np.zeros_like(stacked, dtype=np.float64)
    for idx, arr in enumerate(stacked):
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(arr), dtype=np.float64)
        ranked[idx] = ranks / max(1, len(arr) - 1)
    return np.average(ranked, axis=0, weights=np.asarray(weights, dtype=np.float64))


def _combine_predictions(arrays: list[np.ndarray], weights: list[float], method: str) -> np.ndarray:
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights = normalized_weights / normalized_weights.sum()
    method_key = str(method).strip().lower()
    if method_key == "weighted":
        return clip_predictions(
            sum(
                weight * np.asarray(arr, dtype=np.float64)
                for weight, arr in zip(normalized_weights, arrays, strict=True)
            )
        )
    if method_key == "logit":
        logits = [np.log(clip_predictions(arr) / (1.0 - clip_predictions(arr))) for arr in arrays]
        blended = sum(weight * logit for weight, logit in zip(normalized_weights, logits, strict=True))
        return clip_predictions(1.0 / (1.0 + np.exp(-blended)))
    if method_key == "rank":
        return clip_predictions(_weighted_rank_average(arrays, list(normalized_weights)))
    raise ValueError(f"Unsupported blend method: {method}")


def _blend_name(name_prefix: str, component_names: list[str], weights: dict[str, float]) -> str:
    weight_suffix = "_".join(f"{name}{int(round(weight * 100)):02d}" for name, weight in weights.items())
    return f"{name_prefix}_{'__'.join(component_names)}__{weight_suffix}"


def _component_correlation(first: PipelineResult, second: PipelineResult) -> float:
    first_preds = np.asarray(first.oof_preds, dtype=np.float64)
    second_preds = np.asarray(second.oof_preds, dtype=np.float64)
    if first_preds.shape != second_preds.shape or first_preds.size <= 1:
        return 0.0
    corr = np.corrcoef(first_preds, second_preds)[0, 1]
    return float(abs(corr)) if np.isfinite(corr) else 0.0


def make_component_blend_result(
    *,
    bundle: Any,
    artifacts: Any,
    results_by_name: dict[str, PipelineResult],
    component_weights: dict[str, float],
    method: str,
    outer_folds: int,
    kind: str = "hill_climb_blend",
    name_prefix: str | None = None,
) -> PipelineResult:
    normalized_weights = _normalize_weights(component_weights)
    component_names = list(normalized_weights)
    component_results = [results_by_name[name] for name in component_names]
    weight_values = [normalized_weights[name] for name in component_names]
    blend_name = _blend_name(
        name_prefix or f"{kind}_{method}",
        component_names,
        normalized_weights,
    )
    oof_preds = _combine_predictions([result.oof_preds for result in component_results], weight_values, method)
    test_preds = _combine_predictions([result.test_preds for result in component_results], weight_values, method)
    fold_scores: list[dict[str, Any]] = []
    test_predictions_by_fold: dict[str, np.ndarray] = {}
    oof_predictions_by_fold: dict[str, np.ndarray] = {}
    valid_indices_by_fold: dict[str, np.ndarray] = {}
    seed_oof_predictions: dict[int, np.ndarray] = {}
    seed_test_predictions: dict[int, np.ndarray] = {}
    seed_scores: list[dict[str, Any]] = []
    seed_oof_maps = [seed_prediction_map(result, "seed_oof_preds") for result in component_results]
    seed_test_maps = [seed_prediction_map(result, "seed_test_preds") for result in component_results]
    common_seeds = sorted(set.intersection(*(set(mapping) for mapping in seed_oof_maps))) if seed_oof_maps else []
    for seed in common_seeds:
        seed_oof_predictions[seed] = _combine_predictions(
            [mapping[seed] for mapping in seed_oof_maps],
            weight_values,
            method,
        )
        if all(seed in mapping for mapping in seed_test_maps):
            seed_test_predictions[seed] = _combine_predictions(
                [mapping[seed] for mapping in seed_test_maps],
                weight_values,
                method,
            )
        seed_scores.append({"seed": seed, "auc": safe_auc(bundle.target_values, seed_oof_predictions[seed])})
    for fold_number in range(1, outer_folds + 1):
        fold_key = f"fold_{fold_number}"
        anchor = component_results[0]
        valid_idx = anchor.valid_indices_by_fold[fold_key]
        fold_valid_preds = _combine_predictions(
            [result.oof_predictions_by_fold[fold_key] for result in component_results],
            weight_values,
            method,
        )
        fold_test_preds = _combine_predictions(
            [result.test_predictions_by_fold[fold_key] for result in component_results],
            weight_values,
            method,
        )
        oof_predictions_by_fold[fold_key] = fold_valid_preds
        test_predictions_by_fold[fold_key] = fold_test_preds
        valid_indices_by_fold[fold_key] = valid_idx
        fold_scores.append(
            {
                "suite": artifacts.suite_name,
                "pipeline": blend_name,
                "fold": fold_number,
                "roc_auc": safe_auc(bundle.target_values[valid_idx], fold_valid_preds),
                "pseudo_statuses": kind,
                "pseudo_improved": False,
                "pseudo_candidates": 0,
            }
        )
    cv_score = safe_auc(bundle.target_values, oof_preds)
    max_pair_corr = max(
        (
            _component_correlation(first, second)
            for idx, first in enumerate(component_results[:-1])
            for second in component_results[idx + 1 :]
        ),
        default=0.0,
    )
    return PipelineResult(
        name=blend_name,
        oof_preds=oof_preds,
        test_preds=test_preds,
        cv_score=cv_score,
        fold_scores=fold_scores,
        feature_manifest={
            "suite_name": artifacts.suite_name,
            "kind": kind,
            "components": component_names,
            "weights": normalized_weights,
            "final_feature_count": int(component_results[0].feature_manifest.get("final_feature_count", 0)),
            "train_mode": getattr(artifacts, "train_mode", None),
            "feature_recipe": getattr(artifacts, "feature_recipe", None),
            "original_row_weight": getattr(artifacts, "original_row_weight", None),
        },
        metadata={
            "suite_name": artifacts.suite_name,
            "kind": kind,
            "method": method,
            "weights": normalized_weights,
            "model_family": "blend",
            "model_backend": "blend",
            "model_seeds": sorted(
                {int(seed) for result in component_results for seed in result.metadata.get("model_seeds", [])}
            ),
            "seed_scores": seed_scores,
            "seed_oof_preds": seed_oof_predictions,
            "seed_test_preds": seed_test_predictions,
            "blend_components": component_names,
            "component_count": len(component_names),
            "max_component_abs_corr": max_pair_corr,
            "pipeline_name": blend_name,
            "prediction_range": [float(test_preds.min()), float(test_preds.max())],
            "overall_oof_auc": cv_score,
            "train_mode": getattr(artifacts, "train_mode", None),
            "feature_recipe": getattr(artifacts, "feature_recipe", None),
            "original_row_weight": getattr(artifacts, "original_row_weight", None),
        },
        test_predictions_by_fold=test_predictions_by_fold,
        oof_predictions_by_fold=oof_predictions_by_fold,
        valid_indices_by_fold=valid_indices_by_fold,
    )


def make_weighted_blend_result(
    *,
    bundle: Any,
    artifacts: Any,
    results_by_name: dict[str, PipelineResult],
    first_name: str,
    second_name: str,
    first_weight: float,
    outer_folds: int,
) -> PipelineResult:
    second_weight = 1.0 - first_weight
    first = results_by_name[first_name]
    second = results_by_name[second_name]
    blend_name = f"blend_{first_name}_{second_name}_w{int(round(first_weight * 100)):02d}"
    oof_preds = clip_predictions(first_weight * first.oof_preds + second_weight * second.oof_preds)
    test_preds = clip_predictions(first_weight * first.test_preds + second_weight * second.test_preds)
    fold_scores: list[dict[str, Any]] = []
    test_predictions_by_fold: dict[str, np.ndarray] = {}
    oof_predictions_by_fold: dict[str, np.ndarray] = {}
    valid_indices_by_fold: dict[str, np.ndarray] = {}
    seed_oof_predictions: dict[int, np.ndarray] = {}
    seed_test_predictions: dict[int, np.ndarray] = {}
    seed_scores: list[dict[str, Any]] = []
    first_seed_oof = seed_prediction_map(first, "seed_oof_preds")
    second_seed_oof = seed_prediction_map(second, "seed_oof_preds")
    first_seed_test = seed_prediction_map(first, "seed_test_preds")
    second_seed_test = seed_prediction_map(second, "seed_test_preds")
    for seed in sorted(set(first_seed_oof) & set(second_seed_oof)):
        seed_oof_predictions[seed] = clip_predictions(
            first_weight * first_seed_oof[seed] + second_weight * second_seed_oof[seed]
        )
        if seed in first_seed_test and seed in second_seed_test:
            seed_test_predictions[seed] = clip_predictions(
                first_weight * first_seed_test[seed] + second_weight * second_seed_test[seed]
            )
        seed_scores.append({"seed": seed, "auc": safe_auc(bundle.target_values, seed_oof_predictions[seed])})
    for fold_number in range(1, outer_folds + 1):
        fold_key = f"fold_{fold_number}"
        valid_idx = first.valid_indices_by_fold[fold_key]
        fold_valid_preds = clip_predictions(
            first_weight * first.oof_predictions_by_fold[fold_key]
            + second_weight * second.oof_predictions_by_fold[fold_key]
        )
        fold_test_preds = clip_predictions(
            first_weight * first.test_predictions_by_fold[fold_key]
            + second_weight * second.test_predictions_by_fold[fold_key]
        )
        oof_predictions_by_fold[fold_key] = fold_valid_preds
        test_predictions_by_fold[fold_key] = fold_test_preds
        valid_indices_by_fold[fold_key] = valid_idx
        fold_scores.append(
            {
                "suite": artifacts.suite_name,
                "pipeline": blend_name,
                "fold": fold_number,
                "roc_auc": safe_auc(bundle.target_values[valid_idx], fold_valid_preds),
                "pseudo_statuses": "blend",
                "pseudo_improved": False,
                "pseudo_candidates": 0,
            }
        )
    cv_score = safe_auc(bundle.target_values, oof_preds)
    weights = {first_name: first_weight, second_name: second_weight}
    feature_manifest = {
        "suite_name": artifacts.suite_name,
        "kind": "blend",
        "components": [first_name, second_name],
        "weights": weights,
        "final_feature_count": int(first.feature_manifest.get("final_feature_count", 0)),
        "orig_available": getattr(artifacts, "orig_df", None) is not None,
        "orig_source_path": getattr(artifacts, "orig_source_path", None),
        "orig_feature_status": getattr(artifacts, "orig_feature_status", {}),
    }
    metadata = {
        "suite_name": artifacts.suite_name,
        "kind": "blend",
        "method": "weighted",
        "weights": weights,
        "model_family": "blend",
        "model_seeds": sorted(set(first.metadata["model_seeds"]) | set(second.metadata["model_seeds"])),
        "seed_scores": seed_scores,
        "seed_oof_preds": seed_oof_predictions,
        "seed_test_preds": seed_test_predictions,
        "prediction_range": [float(test_preds.min()), float(test_preds.max())],
        "overall_oof_auc": cv_score,
        "blend_components": [first_name, second_name],
        "train_mode": getattr(artifacts, "train_mode", None),
        "feature_recipe": getattr(artifacts, "feature_recipe", None),
    }
    return PipelineResult(
        name=blend_name,
        oof_preds=oof_preds,
        test_preds=test_preds,
        cv_score=cv_score,
        fold_scores=fold_scores,
        feature_manifest=feature_manifest,
        metadata=metadata,
        test_predictions_by_fold=test_predictions_by_fold,
        oof_predictions_by_fold=oof_predictions_by_fold,
        valid_indices_by_fold=valid_indices_by_fold,
    )


def make_logit_blend_result(
    *,
    bundle: Any,
    artifacts: Any,
    results_by_name: dict[str, PipelineResult],
    first_name: str,
    second_name: str,
    first_weight: float,
    outer_folds: int,
) -> PipelineResult:
    second_weight = 1.0 - first_weight
    first = results_by_name[first_name]
    second = results_by_name[second_name]
    blend_name = f"logit_blend_{first_name}_{second_name}_w{int(round(first_weight * 100)):02d}"
    oof_preds = clip_predictions(weighted_logit_average(first.oof_preds, second.oof_preds, first_weight))
    test_preds = clip_predictions(weighted_logit_average(first.test_preds, second.test_preds, first_weight))
    fold_scores: list[dict[str, Any]] = []
    test_predictions_by_fold: dict[str, np.ndarray] = {}
    oof_predictions_by_fold: dict[str, np.ndarray] = {}
    valid_indices_by_fold: dict[str, np.ndarray] = {}
    seed_oof_predictions: dict[int, np.ndarray] = {}
    seed_test_predictions: dict[int, np.ndarray] = {}
    seed_scores: list[dict[str, Any]] = []
    first_seed_oof = seed_prediction_map(first, "seed_oof_preds")
    second_seed_oof = seed_prediction_map(second, "seed_oof_preds")
    first_seed_test = seed_prediction_map(first, "seed_test_preds")
    second_seed_test = seed_prediction_map(second, "seed_test_preds")
    for seed in sorted(set(first_seed_oof) & set(second_seed_oof)):
        seed_oof_predictions[seed] = clip_predictions(
            weighted_logit_average(first_seed_oof[seed], second_seed_oof[seed], first_weight)
        )
        if seed in first_seed_test and seed in second_seed_test:
            seed_test_predictions[seed] = clip_predictions(
                weighted_logit_average(first_seed_test[seed], second_seed_test[seed], first_weight)
            )
        seed_scores.append({"seed": seed, "auc": safe_auc(bundle.target_values, seed_oof_predictions[seed])})
    for fold_number in range(1, outer_folds + 1):
        fold_key = f"fold_{fold_number}"
        valid_idx = first.valid_indices_by_fold[fold_key]
        fold_valid_preds = clip_predictions(
            weighted_logit_average(
                first.oof_predictions_by_fold[fold_key],
                second.oof_predictions_by_fold[fold_key],
                first_weight,
            )
        )
        fold_test_preds = clip_predictions(
            weighted_logit_average(
                first.test_predictions_by_fold[fold_key],
                second.test_predictions_by_fold[fold_key],
                first_weight,
            )
        )
        oof_predictions_by_fold[fold_key] = fold_valid_preds
        test_predictions_by_fold[fold_key] = fold_test_preds
        valid_indices_by_fold[fold_key] = valid_idx
        fold_scores.append(
            {
                "suite": artifacts.suite_name,
                "pipeline": blend_name,
                "fold": fold_number,
                "roc_auc": safe_auc(bundle.target_values[valid_idx], fold_valid_preds),
                "pseudo_statuses": "logit_blend",
                "pseudo_improved": False,
                "pseudo_candidates": 0,
            }
        )
    cv_score = safe_auc(bundle.target_values, oof_preds)
    weights = {first_name: first_weight, second_name: second_weight}
    return PipelineResult(
        name=blend_name,
        oof_preds=oof_preds,
        test_preds=test_preds,
        cv_score=cv_score,
        fold_scores=fold_scores,
        feature_manifest={
            "suite_name": artifacts.suite_name,
            "kind": "logit_blend",
            "components": [first_name, second_name],
            "weights": weights,
            "final_feature_count": int(first.feature_manifest.get("final_feature_count", 0)),
            "train_mode": getattr(artifacts, "train_mode", None),
            "feature_recipe": getattr(artifacts, "feature_recipe", None),
        },
        metadata={
            "suite_name": artifacts.suite_name,
            "kind": "logit_blend",
            "method": "logit",
            "weights": weights,
            "model_family": "blend",
            "model_seeds": sorted(set(first.metadata["model_seeds"]) | set(second.metadata["model_seeds"])),
            "seed_scores": seed_scores,
            "seed_oof_preds": seed_oof_predictions,
            "seed_test_preds": seed_test_predictions,
            "blend_components": [first_name, second_name],
            "prediction_range": [float(test_preds.min()), float(test_preds.max())],
            "overall_oof_auc": cv_score,
            "train_mode": getattr(artifacts, "train_mode", None),
            "feature_recipe": getattr(artifacts, "feature_recipe", None),
        },
        test_predictions_by_fold=test_predictions_by_fold,
        oof_predictions_by_fold=oof_predictions_by_fold,
        valid_indices_by_fold=valid_indices_by_fold,
    )


def select_top_blend_components(results: list[PipelineResult]) -> list[PipelineResult]:
    by_family: dict[str, PipelineResult] = {}
    for result in sorted(results, key=lambda item: item.cv_score, reverse=True):
        family = str(result.metadata.get("model_family") or "unknown")
        by_family.setdefault(family, result)
    if len(by_family) >= 2:
        return sorted(by_family.values(), key=lambda item: item.cv_score, reverse=True)[:2]
    return sorted(results, key=lambda item: item.cv_score, reverse=True)[:2]


def select_blend_candidate_pool(
    results: list[PipelineResult],
    *,
    max_candidates: int = 6,
    max_per_family: int = 2,
    max_per_suite: int = 2,
) -> list[PipelineResult]:
    singles = [result for result in results if str(result.metadata.get("kind", "single")) == "single"]
    ordered = sorted(singles, key=lambda item: item.cv_score, reverse=True)
    selected: list[PipelineResult] = []
    family_counts: dict[str, int] = {}
    suite_counts: dict[str, int] = {}
    seen_names: set[str] = set()

    def _try_add(result: PipelineResult) -> None:
        if result.name in seen_names or len(selected) >= max_candidates:
            return
        family = str(result.metadata.get("model_family") or "unknown")
        suite = str(result.metadata.get("suite_name") or "unknown")
        if family_counts.get(family, 0) >= max_per_family:
            return
        if suite_counts.get(suite, 0) >= max_per_suite:
            return
        selected.append(result)
        seen_names.add(result.name)
        family_counts[family] = family_counts.get(family, 0) + 1
        suite_counts[suite] = suite_counts.get(suite, 0) + 1

    best_by_family: dict[str, PipelineResult] = {}
    for result in ordered:
        family = str(result.metadata.get("model_family") or "unknown")
        best_by_family.setdefault(family, result)
    for result in best_by_family.values():
        _try_add(result)

    best_by_suite: dict[str, PipelineResult] = {}
    for result in ordered:
        suite = str(result.metadata.get("suite_name") or "unknown")
        best_by_suite.setdefault(suite, result)
    for result in best_by_suite.values():
        _try_add(result)

    for result in ordered:
        _try_add(result)
    return selected


def build_hill_climb_candidates(
    *,
    bundle: Any,
    artifacts: Any,
    results_by_name: dict[str, PipelineResult],
    candidate_results: list[PipelineResult],
    outer_folds: int,
    max_components: int = 4,
    add_weight_grid: tuple[float, ...] = (0.15, 0.25, 0.35),
    corr_threshold: float = 0.995,
) -> list[PipelineResult]:
    if len(candidate_results) < 3:
        return []
    ordered_candidates = sorted(candidate_results, key=lambda item: item.cv_score, reverse=True)
    current_weights = {ordered_candidates[0].name: 1.0}
    used_names = {ordered_candidates[0].name}
    generated: list[PipelineResult] = []

    for _ in range(2, max_components + 1):
        best_candidate: PipelineResult | None = None
        best_weights: dict[str, float] | None = None
        best_score = float("-inf")
        for candidate in ordered_candidates:
            if candidate.name in used_names:
                continue
            max_corr = max(
                (_component_correlation(candidate, results_by_name[name]) for name in current_weights),
                default=0.0,
            )
            weight_grid = (0.1, 0.15, 0.2) if max_corr > corr_threshold else add_weight_grid
            for method in ("weighted", "logit"):
                for add_weight in weight_grid:
                    if max_corr > corr_threshold and add_weight > 0.2:
                        continue
                    candidate_weights = {name: weight * (1.0 - add_weight) for name, weight in current_weights.items()}
                    candidate_weights[candidate.name] = candidate_weights.get(candidate.name, 0.0) + add_weight
                    blended = make_component_blend_result(
                        bundle=bundle,
                        artifacts=artifacts,
                        results_by_name=results_by_name,
                        component_weights=candidate_weights,
                        method=method,
                        outer_folds=outer_folds,
                        kind="hill_climb_blend",
                        name_prefix=f"hill_climb_{method}",
                    )
                    improvement = float(blended.cv_score) - max(
                        results_by_name[name].cv_score for name in candidate_weights
                    )
                    if max_corr > corr_threshold and improvement <= 1e-5:
                        continue
                    if blended.cv_score > best_score:
                        best_candidate = blended
                        best_weights = candidate_weights
                        best_score = float(blended.cv_score)
        if best_candidate is None or best_weights is None:
            break
        generated.append(best_candidate)
        current_weights = _normalize_weights(best_weights)
        used_names = set(current_weights)

    if not any(len(result.metadata.get("blend_components", [])) >= 3 for result in generated):
        remaining = [result for result in ordered_candidates if result.name not in used_names]
        if remaining and len(current_weights) >= 2:
            forced_candidate = remaining[0]
            forced_weights = {name: weight * 0.85 for name, weight in current_weights.items()}
            forced_weights[forced_candidate.name] = 0.15
            generated.append(
                make_component_blend_result(
                    bundle=bundle,
                    artifacts=artifacts,
                    results_by_name=results_by_name,
                    component_weights=forced_weights,
                    method="logit",
                    outer_folds=outer_folds,
                    kind="hill_climb_blend",
                    name_prefix="hill_climb_logit",
                )
            )
    return generated


def make_rank_blend_result(
    *,
    bundle: Any,
    artifacts: Any,
    first: PipelineResult,
    second: PipelineResult,
    outer_folds: int,
) -> PipelineResult:
    blend_name = f"rank_blend_{first.name}_{second.name}"
    oof_preds = clip_predictions(rank_average(first.oof_preds, second.oof_preds))
    test_preds = clip_predictions(rank_average(first.test_preds, second.test_preds))
    fold_scores: list[dict[str, Any]] = []
    test_predictions_by_fold: dict[str, np.ndarray] = {}
    oof_predictions_by_fold: dict[str, np.ndarray] = {}
    valid_indices_by_fold: dict[str, np.ndarray] = {}
    seed_oof_predictions: dict[int, np.ndarray] = {}
    seed_test_predictions: dict[int, np.ndarray] = {}
    seed_scores: list[dict[str, Any]] = []
    first_seed_oof = seed_prediction_map(first, "seed_oof_preds")
    second_seed_oof = seed_prediction_map(second, "seed_oof_preds")
    first_seed_test = seed_prediction_map(first, "seed_test_preds")
    second_seed_test = seed_prediction_map(second, "seed_test_preds")
    for seed in sorted(set(first_seed_oof) & set(second_seed_oof)):
        seed_oof_predictions[seed] = clip_predictions(rank_average(first_seed_oof[seed], second_seed_oof[seed]))
        if seed in first_seed_test and seed in second_seed_test:
            seed_test_predictions[seed] = clip_predictions(rank_average(first_seed_test[seed], second_seed_test[seed]))
        seed_scores.append({"seed": seed, "auc": safe_auc(bundle.target_values, seed_oof_predictions[seed])})
    for fold_number in range(1, outer_folds + 1):
        fold_key = f"fold_{fold_number}"
        valid_idx = first.valid_indices_by_fold[fold_key]
        fold_valid_preds = clip_predictions(
            rank_average(first.oof_predictions_by_fold[fold_key], second.oof_predictions_by_fold[fold_key])
        )
        fold_test_preds = clip_predictions(
            rank_average(first.test_predictions_by_fold[fold_key], second.test_predictions_by_fold[fold_key])
        )
        oof_predictions_by_fold[fold_key] = fold_valid_preds
        test_predictions_by_fold[fold_key] = fold_test_preds
        valid_indices_by_fold[fold_key] = valid_idx
        fold_scores.append(
            {
                "suite": artifacts.suite_name,
                "pipeline": blend_name,
                "fold": fold_number,
                "roc_auc": safe_auc(bundle.target_values[valid_idx], fold_valid_preds),
                "pseudo_statuses": "rank_blend",
                "pseudo_improved": False,
                "pseudo_candidates": 0,
            }
        )
    cv_score = safe_auc(bundle.target_values, oof_preds)
    weights = {first.name: "rank", second.name: "rank"}
    return PipelineResult(
        name=blend_name,
        oof_preds=oof_preds,
        test_preds=test_preds,
        cv_score=cv_score,
        fold_scores=fold_scores,
        feature_manifest={
            "suite_name": artifacts.suite_name,
            "kind": "rank_blend",
            "components": [first.name, second.name],
            "final_feature_count": int(first.feature_manifest.get("final_feature_count", 0)),
            "train_mode": getattr(artifacts, "train_mode", None),
            "feature_recipe": getattr(artifacts, "feature_recipe", None),
        },
        metadata={
            "suite_name": artifacts.suite_name,
            "kind": "rank_blend",
            "method": "rank",
            "weights": weights,
            "model_family": "blend",
            "model_seeds": sorted(set(first.metadata["model_seeds"]) | set(second.metadata["model_seeds"])),
            "seed_scores": seed_scores,
            "seed_oof_preds": seed_oof_predictions,
            "seed_test_preds": seed_test_predictions,
            "blend_components": [first.name, second.name],
            "prediction_range": [float(test_preds.min()), float(test_preds.max())],
            "overall_oof_auc": cv_score,
            "train_mode": getattr(artifacts, "train_mode", None),
            "feature_recipe": getattr(artifacts, "feature_recipe", None),
        },
        test_predictions_by_fold=test_predictions_by_fold,
        oof_predictions_by_fold=oof_predictions_by_fold,
        valid_indices_by_fold=valid_indices_by_fold,
    )
