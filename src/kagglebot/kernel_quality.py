from __future__ import annotations

import math
import re
from collections.abc import Iterator

from kagglebot import plan_policy, score_progress, score_sources
from kagglebot.autopilot_helpers import _to_float, _to_int
from kagglebot.metric_matching import metrics_equivalent

QUALITY_GUARD_BASELINE_REL_MARGIN = 0.01
QUALITY_GUARD_BASELINE_ABS_MARGIN = 1e-6
QUALITY_GUARD_MISMATCH_REL_MARGIN_MINIMIZE = 2.0
QUALITY_GUARD_MISMATCH_REL_MARGIN_MAXIMIZE = 0.30
QUALITY_GUARD_MISMATCH_ABS_MARGIN = 0.05
QUALITY_GUARD_STEP_BUCKET_RATIO = 2.5
QUALITY_GUARD_SUBGROUP_RATIO = 2.5
QUALITY_GUARD_SUBGROUP_ABS_MARGIN = 0.05
QUALITY_GUARD_CODE_REF_REL_MARGIN = 0.0
QUALITY_GUARD_CODE_REF_ABS_MARGIN = 0.02
QUALITY_GUARD_CANDIDATE_HOLDOUT_REL_MARGIN = 0.20
QUALITY_GUARD_CANDIDATE_HOLDOUT_ABS_MARGIN = 0.01
QUALITY_GUARD_PREDICTION_COUNT_RATIO = 0.60
QUALITY_GUARD_PREDICTION_COUNT_ABS_MARGIN = 1.0
MODEL_NODE_METRIC_KEY = re.compile(r"^model_(?P<model_id>\d+)_node_type_(?P<node_type>\d+)$")
HARD_POLICY_BLOCK_REASONS = frozenset({"external_test_label_transfer_detected"})
COMPETITION_FAITHFULNESS_FALSE_SCORE_SOURCE_TOKENS = (
    "sample",
    "smoke",
    "surrogate",
    "oracle",
    "proxy",
)
CAPACITY_TIER_PRIORITY = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "extreme": 3,
}
LOW_CAPACITY_MARKERS = ("naive", "mean", "persistence", "baseline", "ridge", "linear")
MEDIUM_CAPACITY_MARKERS = ("lightgbm", "xgboost", "catboost", "randomforest", "tabnet", "tabm", "mlp")
HIGH_CAPACITY_MARKERS = (
    "graph",
    "gnn",
    "transformer",
    "seq2seq",
    "hybrid",
    "multihorizon",
    "autoregressive",
    "ensemble",
    "blend",
    "stack",
)
EXTREME_CAPACITY_MARKERS = ("diffusion", "llm", "convnext", "foundation", "pretrained")


def is_significantly_worse(
    *,
    current: float,
    reference: float,
    direction: str,
    rel_margin: float,
    abs_margin: float,
) -> bool:
    margin = max(abs(reference) * max(rel_margin, 0.0), max(abs_margin, 0.0))
    if direction == "minimize":
        return (current - reference) > margin
    return (reference - current) > margin


def build_score_source_quality_signal(score_source: str | None) -> dict[str, object]:
    normalized = score_sources.normalize_score_source_name(score_source)
    trusted = score_sources.is_trusted_offline_score_source(normalized)
    reasons: list[str] = []
    warnings: list[str] = []
    if not trusted:
        reasons.append("untrusted_score_source")
        warnings.append(f"score_source={normalized}")
    return {
        "normalized_score_source": normalized,
        "trusted": trusted,
        "reasons": reasons,
        "warnings": warnings,
    }


def merge_quality_signal_messages(
    *,
    reasons: list[str],
    warnings: list[str],
    signal: dict[str, object],
    dedupe: bool = False,
) -> dict[str, list[str]]:
    merged_reasons = list(reasons)
    merged_warnings = list(warnings)
    raw_reasons = signal.get("reasons")
    raw_warnings = signal.get("warnings")
    if isinstance(raw_reasons, list):
        for reason in raw_reasons:
            if isinstance(reason, str) and (not dedupe or reason not in merged_reasons):
                merged_reasons.append(reason)
    if isinstance(raw_warnings, list):
        for warning in raw_warnings:
            if isinstance(warning, str) and (not dedupe or warning not in merged_warnings):
                merged_warnings.append(warning)
    return {
        "reasons": merged_reasons,
        "warnings": merged_warnings,
    }


def build_oracle_override_signal(payload: dict[str, object] | None) -> dict[str, object]:
    oracle_payload = payload.get("oracle") if isinstance(payload, dict) else None
    mode: str | None = None
    applied = False
    if isinstance(oracle_payload, dict):
        raw_mode = str(oracle_payload.get("mode_setting") or "").strip().lower()
        mode = raw_mode or None
        applied = bool(oracle_payload.get("applied"))
    detected = bool(applied or (mode and mode != "off"))
    reasons: list[str] = []
    warnings: list[str] = []
    if detected:
        reasons.append("oracle_override_detected")
        warnings.append(f"oracle_mode={mode or 'unknown'}")
    return {
        "detected": detected,
        "mode": mode,
        "applied": applied,
        "reasons": reasons,
        "warnings": warnings,
    }


def extract_cv_breakdown_by_model_node(payload: dict[str, object] | None) -> dict[tuple[int, int], float]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("cv_breakdown_by_model_node")
    if not isinstance(raw, dict):
        return {}
    out: dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        match = MODEL_NODE_METRIC_KEY.match(key.strip())
        if match is None:
            continue
        parsed = _to_float(value)
        if parsed is None or (not math.isfinite(parsed)):
            continue
        out[(int(match.group("model_id")), int(match.group("node_type")))] = float(parsed)
    return out


def detect_subgroup_collapse_signal(
    *, kernel_metrics_payload: dict[str, object] | None, direction: str
) -> dict[str, object] | None:
    if direction != "minimize":
        return None
    scores = extract_cv_breakdown_by_model_node(kernel_metrics_payload)
    if len(scores) < 2:
        return None

    worst_key, worst_score = max(scores.items(), key=lambda item: item[1])
    best_key, best_score = min(scores.items(), key=lambda item: item[1])
    if best_score <= 0.0:
        ratio = float("inf") if worst_score > 0.0 else 1.0
    else:
        ratio = worst_score / best_score

    same_model_peers = {key: score for key, score in scores.items() if key[0] == worst_key[0] and key != worst_key}
    peer_key: tuple[int, int] | None = None
    peer_score: float | None = None
    peer_ratio: float | None = None
    if same_model_peers:
        peer_key, peer_score = min(same_model_peers.items(), key=lambda item: item[1])
        if peer_score <= 0.0:
            peer_ratio = float("inf") if worst_score > 0.0 else 1.0
        else:
            peer_ratio = worst_score / peer_score

    collapsed_vs_global = worst_score > max(
        best_score * QUALITY_GUARD_SUBGROUP_RATIO,
        best_score + QUALITY_GUARD_SUBGROUP_ABS_MARGIN,
    )
    collapsed_vs_peer = False
    if peer_score is not None:
        collapsed_vs_peer = worst_score > max(
            peer_score * QUALITY_GUARD_SUBGROUP_RATIO,
            peer_score + QUALITY_GUARD_SUBGROUP_ABS_MARGIN,
        )
    if not collapsed_vs_global and not collapsed_vs_peer:
        return None

    worst_bucket_label: str | None = None
    worst_bucket_score: float | None = None
    step_buckets = kernel_metrics_payload.get("cv_step_buckets") if isinstance(kernel_metrics_payload, dict) else None
    if isinstance(step_buckets, dict):
        parsed_buckets: list[tuple[str, float]] = []
        for key, value in step_buckets.items():
            if not isinstance(key, str):
                continue
            parsed = _to_float(value)
            if parsed is None or (not math.isfinite(parsed)):
                continue
            parsed_buckets.append((key, float(parsed)))
        if parsed_buckets:
            worst_bucket_label, worst_bucket_score = max(parsed_buckets, key=lambda item: item[1])

    note_parts = [
        "Subgroup collapse detected:",
        f"model={worst_key[0]} node_type={worst_key[1]} score={worst_score:.6f}",
    ]
    if peer_key is not None and peer_score is not None and peer_ratio is not None:
        note_parts.append(
            "same-model peer "
            f"model={peer_key[0]} node_type={peer_key[1]} score={peer_score:.6f} "
            f"ratio={peer_ratio:.2f}x"
        )
    note_parts.append(f"best subgroup score={best_score:.6f} ratio={ratio:.2f}x")
    if worst_bucket_label is not None and worst_bucket_score is not None:
        note_parts.append(f"worst step bucket={worst_bucket_label} score={worst_bucket_score:.6f}")
    note_parts.append("Next iteration must use subgroup-aware selection/fallbacks at (model_id,node_type) granularity.")

    return {
        "note": "; ".join(note_parts),
        "model_id": int(worst_key[0]),
        "node_type": int(worst_key[1]),
        "worst_key": f"model_{worst_key[0]}_node_type_{worst_key[1]}",
        "worst_score": float(worst_score),
        "best_key": f"model_{best_key[0]}_node_type_{best_key[1]}",
        "best_score": float(best_score),
        "ratio_vs_best": float(ratio),
        "peer_key": (f"model_{peer_key[0]}_node_type_{peer_key[1]}" if peer_key is not None else None),
        "peer_score": float(peer_score) if peer_score is not None else None,
        "ratio_vs_peer": float(peer_ratio) if peer_ratio is not None else None,
        "worst_step_bucket": worst_bucket_label,
        "worst_step_bucket_score": float(worst_bucket_score) if worst_bucket_score is not None else None,
    }


def detect_step_bucket_collapse_signal(
    payload: dict[str, object] | None,
) -> dict[str, object]:
    step_bucket_payload = payload.get("cv_step_buckets") if isinstance(payload, dict) else None
    step_bucket_scores: list[float] = []
    if isinstance(step_bucket_payload, dict):
        for value in step_bucket_payload.values():
            parsed = _to_float(value)
            if parsed is not None:
                step_bucket_scores.append(float(parsed))
    step_bucket_collapse = False
    median_bucket: float | None = None
    worst_bucket: float | None = None
    if len(step_bucket_scores) >= 4:
        sorted_scores = sorted(step_bucket_scores)
        midpoint = len(sorted_scores) // 2
        if len(sorted_scores) % 2:
            median_bucket = float(sorted_scores[midpoint])
        else:
            median_bucket = float((sorted_scores[midpoint - 1] + sorted_scores[midpoint]) / 2)
        worst_bucket = float(max(step_bucket_scores))
        collapse_threshold = max(median_bucket * QUALITY_GUARD_STEP_BUCKET_RATIO, median_bucket + 0.5)
        step_bucket_collapse = worst_bucket > collapse_threshold
    return {
        "count": len(step_bucket_scores),
        "collapse_detected": step_bucket_collapse,
        "median_score": median_bucket,
        "worst_score": worst_bucket,
    }


def build_validation_metric_alignment(
    *,
    current_value: float,
    validation_scores: list[float],
    direction: str,
) -> dict[str, object]:
    best_validation: float | None = None
    severe_mismatch = False
    if validation_scores:
        best_validation = min(validation_scores) if direction == "minimize" else max(validation_scores)
        mismatch_rel = (
            QUALITY_GUARD_MISMATCH_REL_MARGIN_MINIMIZE
            if direction == "minimize"
            else QUALITY_GUARD_MISMATCH_REL_MARGIN_MAXIMIZE
        )
        severe_mismatch = is_significantly_worse(
            current=float(current_value),
            reference=float(best_validation),
            direction=direction,
            rel_margin=mismatch_rel,
            abs_margin=QUALITY_GUARD_MISMATCH_ABS_MARGIN,
        )
    return {
        "best_validation_score": best_validation,
        "validation_score_count": len(validation_scores),
        "severe_mismatch": severe_mismatch,
    }


def build_validation_stability_quality_signal(
    *,
    current_value: float,
    validation_scores: list[float],
    payload: dict[str, object] | None,
    direction: str,
    is_final_iteration: bool,
    force_submit: bool,
) -> dict[str, object]:
    metric_alignment = build_validation_metric_alignment(
        current_value=current_value,
        validation_scores=validation_scores,
        direction=direction,
    )
    step_bucket_signal = detect_step_bucket_collapse_signal(payload)
    severe_validation_mismatch = bool(metric_alignment.get("severe_mismatch"))
    reasons: list[str] = []
    warnings: list[str] = []
    if severe_validation_mismatch:
        reasons.append("validation_metric_mismatch_vs_final_metric")
    if bool(step_bucket_signal.get("collapse_detected")):
        warnings.append("cv_step_bucket_collapse_detected")
        if severe_validation_mismatch:
            reasons.append("severe_step_bucket_instability")
    return {
        "metric_alignment": metric_alignment,
        "step_bucket": step_bucket_signal,
        "severe_validation_mismatch": severe_validation_mismatch,
        "reasons": reasons,
        "warnings": warnings,
        "block_submit": bool(reasons) and not is_final_iteration and not force_submit,
    }


def build_baseline_quality_signal(
    *,
    current_value: float,
    baseline_candidates: list[tuple[str, float]],
    direction: str,
) -> dict[str, object]:
    best_source: str | None = None
    best_score: float | None = None
    if baseline_candidates:
        if direction == "minimize":
            best_source, best_score = min(baseline_candidates, key=lambda item: item[1])
        else:
            best_source, best_score = max(baseline_candidates, key=lambda item: item[1])

    selected_worse = False
    if best_score is not None:
        selected_worse = is_significantly_worse(
            current=float(current_value),
            reference=float(best_score),
            direction=direction,
            rel_margin=QUALITY_GUARD_BASELINE_REL_MARGIN,
            abs_margin=QUALITY_GUARD_BASELINE_ABS_MARGIN,
        )
    return {
        "best_source": best_source,
        "best_score": best_score,
        "candidate_count": len(baseline_candidates),
        "selected_worse_than_baseline": selected_worse,
    }


def build_baseline_regression_quality_signal(
    *,
    current_value: float,
    baseline_candidates: list[tuple[str, float]],
    direction: str,
    is_final_iteration: bool,
    force_submit: bool,
) -> dict[str, object]:
    baseline = build_baseline_quality_signal(
        current_value=current_value,
        baseline_candidates=baseline_candidates,
        direction=direction,
    )
    reasons: list[str] = []
    if bool(baseline.get("selected_worse_than_baseline")):
        reasons.append("selected_worse_than_detected_baseline")
    return {
        "baseline": baseline,
        "reasons": reasons,
        "warnings": [],
        "block_submit": bool(reasons) and not is_final_iteration and not force_submit,
    }


def build_code_reference_quality_signal(
    *,
    current_value: float,
    metric: str,
    code_reference_score: float | None,
    code_reference_source: str | None,
    direction: str,
) -> dict[str, object]:
    comparison_score = score_progress.normalize_code_reference_score_for_comparison(
        current=float(current_value),
        reference=code_reference_score,
        metric=metric,
    )
    delta: float | None = None
    below_reference = False
    warning: str | None = None
    if code_reference_score is not None:
        reference_for_comparison = (
            float(comparison_score) if comparison_score is not None else float(code_reference_score)
        )
        delta = score_progress.score_delta_vs_reference(float(current_value), reference_for_comparison, direction)
        below_reference = is_significantly_worse(
            current=float(current_value),
            reference=reference_for_comparison,
            direction=direction,
            rel_margin=QUALITY_GUARD_CODE_REF_REL_MARGIN,
            abs_margin=QUALITY_GUARD_CODE_REF_ABS_MARGIN,
        )
        if below_reference:
            warning = (
                "code_reference_score="
                f"{float(code_reference_score):.6f},current={float(current_value):.6f},"
                f"comparison_score={reference_for_comparison:.6f},"
                f"delta={delta:+.6f},source={code_reference_source or 'unknown'}"
            )
    return {
        "score": code_reference_score,
        "comparison_score": comparison_score,
        "source": code_reference_source,
        "delta_vs_current": delta,
        "below_reference": below_reference,
        "abs_margin": QUALITY_GUARD_CODE_REF_ABS_MARGIN,
        "rel_margin": QUALITY_GUARD_CODE_REF_REL_MARGIN,
        "warning": warning,
    }


def build_code_reference_regression_quality_signal(
    *,
    current_value: float,
    metric: str,
    code_reference_score: float | None,
    code_reference_source: str | None,
    direction: str,
    force_submit: bool,
) -> dict[str, object]:
    code_reference = build_code_reference_quality_signal(
        current_value=current_value,
        metric=metric,
        code_reference_score=code_reference_score,
        code_reference_source=code_reference_source,
        direction=direction,
    )
    reasons: list[str] = []
    warnings: list[str] = []
    if bool(code_reference.get("below_reference")):
        reasons.append("below_code_reference_baseline")
        warning = code_reference.get("warning")
        if isinstance(warning, str) and warning:
            warnings.append(warning)
    return {
        "code_reference": code_reference,
        "reasons": reasons,
        "warnings": warnings,
        "block_submit": bool(reasons) and not force_submit,
    }


def iter_payload_mappings(payload: object) -> Iterator[dict[object, object]]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from iter_payload_mappings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from iter_payload_mappings(item)


def as_guard_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def first_nested_value(payload: dict[str, object], keys: tuple[str, ...]) -> object | None:
    normalized_keys = {key.lower() for key in keys}
    for node in iter_payload_mappings(payload):
        for key, value in node.items():
            if str(key).strip().lower() in normalized_keys:
                return value
    return None


def max_nested_float(payload: dict[str, object], keys: tuple[str, ...]) -> float | None:
    normalized_keys = {key.lower() for key in keys}
    values: list[float] = []
    for node in iter_payload_mappings(payload):
        for key, value in node.items():
            if str(key).strip().lower() not in normalized_keys:
                continue
            parsed = _to_float(value)
            if parsed is not None and math.isfinite(parsed):
                values.append(float(parsed))
    return max(values) if values else None


def max_nested_int(payload: dict[str, object], keys: tuple[str, ...]) -> int | None:
    normalized_keys = {key.lower() for key in keys}
    values: list[int] = []
    for node in iter_payload_mappings(payload):
        for key, value in node.items():
            if str(key).strip().lower() not in normalized_keys:
                continue
            parsed = _to_int(value)
            if parsed is not None:
                values.append(int(parsed))
    return max(values) if values else None


def min_nested_int(payload: dict[str, object], keys: tuple[str, ...]) -> int | None:
    normalized_keys = {key.lower() for key in keys}
    values: list[int] = []
    for node in iter_payload_mappings(payload):
        for key, value in node.items():
            if str(key).strip().lower() not in normalized_keys:
                continue
            parsed = _to_int(value)
            if parsed is not None:
                values.append(int(parsed))
    return min(values) if values else None


def any_nested_bool(payload: dict[str, object], keys: tuple[str, ...]) -> bool:
    normalized_keys = {key.lower() for key in keys}
    for node in iter_payload_mappings(payload):
        for key, value in node.items():
            if str(key).strip().lower() not in normalized_keys:
                continue
            parsed = as_guard_bool(value)
            if parsed is True:
                return True
    return False


def nested_text(payload: dict[str, object], *, limit: int = 20000) -> str:
    parts: list[str] = []
    total = 0
    for node in iter_payload_mappings(payload):
        for key, value in node.items():
            if isinstance(value, (str, int, float, bool)):
                fragment = f"{key}={value}".lower()
                parts.append(fragment)
                total += len(fragment)
                if total >= limit:
                    return "\n".join(parts)
    return "\n".join(parts)


def detect_external_test_label_transfer_signal(payload: dict[str, object] | None) -> dict[str, object] | None:
    """Detect submissions built by copying hidden test labels from labeled external overlaps.

    External data can be legitimate for pretraining or representation learning. This guard targets a narrower
    failure mode: all or most competition test rows are matched to a labeled external dataset by exact/near-exact
    identifiers, image hashes, or bounding boxes, and those external labels become the submission predictions.
    """
    if not isinstance(payload, dict) or not payload:
        return None

    explicit = any_nested_bool(
        payload,
        (
            "external_test_label_transfer",
            "test_label_transfer",
            "test_label_leakage",
            "external_label_transfer_detected",
        ),
    )
    external_trusted = any_nested_bool(
        payload,
        (
            "external_overlap_trusted",
            "external_data_allowed",
            "external_overlap_used",
            "external_labels_used",
        ),
    )
    exact_coverage = any_nested_bool(
        payload,
        (
            "exact_coverage_pass",
            "external_require_full_exact_coverage",
            "sha1_equal",
        ),
    )
    text = nested_text(payload)
    method_or_manifest_mentions_transfer = (
        "external_test_label_transfer" in text
        or "test_label_transfer" in text
        or "official_multiview_overlap_mapping" in text
        or ("external" in text and "overlap" in text and "mapping" in text)
        or ("external_row_id" in text and "predicted_class" in text)
        or ("match_type_counts" in text and "exact_sha1" in text and "test_selected" in text)
    )

    test_selected_rows = max_nested_int(payload, ("test_selected_row_count", "test_override_rows"))
    submission_rows = max_nested_int(payload, ("submission_rows", "submission_row_count", "test_row_count"))
    uncovered_test_rows = min_nested_int(payload, ("uncovered_test_row_count",))
    exact_test_image_count = max_nested_int(
        payload,
        ("test_exact_sha1_matched_image_count", "test_selected_exact_sha1_image_count"),
    )
    max_image_distance = max_nested_float(
        payload,
        ("max_selected_image_distance", "test_max_selected_image_distance"),
    )
    max_bbox_distance = max_nested_float(
        payload,
        ("max_selected_bbox_distance", "test_max_selected_bbox_distance"),
    )
    final_method = first_nested_value(payload, ("final_selected_method", "name"))
    external_root = first_nested_value(payload, ("external_root_path", "external_root"))

    full_or_near_full_test_coverage = False
    if test_selected_rows is not None and test_selected_rows > 0:
        if uncovered_test_rows == 0:
            full_or_near_full_test_coverage = True
        elif submission_rows is not None and submission_rows > 0 and test_selected_rows >= int(0.95 * submission_rows):
            full_or_near_full_test_coverage = True

    exact_or_near_exact = bool(exact_coverage)
    if max_image_distance is not None and max_bbox_distance is not None:
        exact_or_near_exact = exact_or_near_exact or (max_image_distance <= 1e-12 and max_bbox_distance <= 1e-12)
    if exact_test_image_count is not None and exact_test_image_count > 0:
        exact_or_near_exact = True

    if not explicit and not (
        external_trusted
        and full_or_near_full_test_coverage
        and exact_or_near_exact
        and method_or_manifest_mentions_transfer
    ):
        return None

    return {
        "detected": True,
        "reason": "external labeled data appears to directly determine competition test predictions",
        "test_selected_row_count": test_selected_rows,
        "submission_rows": submission_rows,
        "uncovered_test_row_count": uncovered_test_rows,
        "test_exact_sha1_image_count": exact_test_image_count,
        "max_selected_image_distance": max_image_distance,
        "max_selected_bbox_distance": max_bbox_distance,
        "final_selected_method": str(final_method) if final_method is not None else None,
        "external_root": str(external_root) if external_root is not None else None,
    }


def build_external_label_transfer_quality_signal(payload: dict[str, object] | None) -> dict[str, object]:
    transfer = detect_external_test_label_transfer_signal(payload)
    detected = transfer is not None
    reasons: list[str] = []
    warnings: list[str] = []
    if detected:
        reasons.append("external_test_label_transfer_detected")
        warnings.append(
            "external_test_label_transfer="
            f"rows={transfer.get('test_selected_row_count')},"
            f"uncovered={transfer.get('uncovered_test_row_count')},"
            f"method={transfer.get('final_selected_method') or 'unknown'}"
        )
    return {
        "detected": detected,
        "hard_block": detected,
        "transfer": transfer,
        "reasons": reasons,
        "warnings": warnings,
    }


def pipeline_name_from_payload(pipeline: dict[str, object]) -> str | None:
    for key in ("name", "pipeline", "pipeline_name", "method", "model_name"):
        value = pipeline.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_selected_pipeline_name(payload: dict[str, object]) -> str | None:
    selected = payload.get("selected")
    if isinstance(selected, dict):
        name = pipeline_name_from_payload(selected)
        if name:
            return name
    for key in (
        "chosen_pipeline",
        "selected_pipeline",
        "selected_method",
        "final_pipeline",
        "final_method",
        "best_pipeline",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_pipeline_candidates(payload: dict[str, object]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for key in ("pipelines", "candidates", "leaderboard", "models"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                candidates.append(item)
    return candidates


def pipeline_float(pipeline: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        parsed = _to_float(pipeline.get(key))
        if parsed is not None and math.isfinite(parsed):
            return float(parsed)
    return None


def find_selected_pipeline(
    *,
    payload: dict[str, object],
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    selected_name = extract_selected_pipeline_name(payload)
    if selected_name:
        selected_norm = selected_name.strip().lower()
        for candidate in candidates:
            name = pipeline_name_from_payload(candidate)
            if name and name.strip().lower() == selected_norm:
                return candidate
    selected = payload.get("selected")
    if isinstance(selected, dict):
        return selected
    return None


def detect_candidate_selection_mismatch(
    *,
    payload: dict[str, object] | None,
    direction: str,
) -> dict[str, object] | None:
    if not isinstance(payload, dict) or not payload:
        return None
    candidates = extract_pipeline_candidates(payload)
    if len(candidates) < 2:
        return None
    selected = find_selected_pipeline(payload=payload, candidates=candidates)
    if selected is None:
        return None

    secondary_keys = (
        "holdout_score",
        "holdout_value",
        "holdout_f1",
        "val_score",
        "validation_score",
        "val_f1",
        "secondary_score",
    )
    selected_secondary = pipeline_float(selected, secondary_keys)
    if selected_secondary is None:
        return None

    scored_candidates = [
        (candidate, score)
        for candidate in candidates
        if (score := pipeline_float(candidate, secondary_keys)) is not None
    ]
    if len(scored_candidates) < 2:
        return None
    best_candidate, best_secondary = (
        min(scored_candidates, key=lambda item: item[1])
        if direction == "minimize"
        else max(scored_candidates, key=lambda item: item[1])
    )
    if best_candidate is selected:
        return None
    if not is_significantly_worse(
        current=float(selected_secondary),
        reference=float(best_secondary),
        direction=direction,
        rel_margin=QUALITY_GUARD_CANDIDATE_HOLDOUT_REL_MARGIN,
        abs_margin=QUALITY_GUARD_CANDIDATE_HOLDOUT_ABS_MARGIN,
    ):
        return None

    selected_primary = pipeline_float(
        selected,
        (
            "cv_score",
            "offline_value",
            "value",
            "score",
            "oof_score",
            "oof_f1",
            "mean_cv",
        ),
    )
    best_primary = pipeline_float(
        best_candidate,
        (
            "cv_score",
            "offline_value",
            "value",
            "score",
            "oof_score",
            "oof_f1",
            "mean_cv",
        ),
    )
    return {
        "selected": pipeline_name_from_payload(selected) or extract_selected_pipeline_name(payload) or "unknown",
        "selected_secondary_score": float(selected_secondary),
        "selected_primary_score": selected_primary,
        "best_secondary_candidate": pipeline_name_from_payload(best_candidate) or "unknown",
        "best_secondary_score": float(best_secondary),
        "best_secondary_primary_score": best_primary,
        "direction": direction,
        "candidate_count": len(scored_candidates),
    }


def build_candidate_selection_quality_signal(
    *,
    payload: dict[str, object] | None,
    direction: str,
) -> dict[str, object]:
    mismatch = detect_candidate_selection_mismatch(payload=payload, direction=direction)
    detected = mismatch is not None
    reasons: list[str] = []
    warnings: list[str] = []
    if detected:
        reasons.append("selected_pipeline_validation_mismatch")
        warnings.append(
            "candidate_selection_mismatch="
            f"selected={mismatch.get('selected')},"
            f"selected_secondary={mismatch.get('selected_secondary_score')},"
            f"best_secondary_candidate={mismatch.get('best_secondary_candidate')},"
            f"best_secondary={mismatch.get('best_secondary_score')}"
        )
    return {
        "detected": detected,
        "mismatch": mismatch,
        "reasons": reasons,
        "warnings": warnings,
    }


def prediction_count_mean(pipeline: dict[str, object]) -> float | None:
    summary = pipeline.get("prediction_count_summary")
    if not isinstance(summary, dict):
        return None
    for split in ("test", "submission", "val", "holdout"):
        split_summary = summary.get(split)
        if isinstance(split_summary, dict):
            parsed = _to_float(split_summary.get("mean"))
            if parsed is not None and math.isfinite(parsed):
                return float(parsed)
    parsed = _to_float(summary.get("mean"))
    if parsed is not None and math.isfinite(parsed):
        return float(parsed)
    return None


def detect_prediction_distribution_collapse(payload: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(payload, dict) or not payload:
        return None
    candidates = extract_pipeline_candidates(payload)
    if len(candidates) < 2:
        return None
    selected = find_selected_pipeline(payload=payload, candidates=candidates)
    if selected is None:
        return None
    selected_mean = prediction_count_mean(selected)
    if selected_mean is None:
        return None
    candidate_means = [
        (pipeline_name_from_payload(candidate) or "unknown", mean)
        for candidate in candidates
        if (mean := prediction_count_mean(candidate)) is not None
    ]
    if len(candidate_means) < 2:
        return None
    max_name, max_mean = max(candidate_means, key=lambda item: item[1])
    if max_mean < 3.0:
        return None
    if (
        selected_mean <= max_mean * QUALITY_GUARD_PREDICTION_COUNT_RATIO
        and (max_mean - selected_mean) >= QUALITY_GUARD_PREDICTION_COUNT_ABS_MARGIN
    ):
        return {
            "selected": pipeline_name_from_payload(selected) or extract_selected_pipeline_name(payload) or "unknown",
            "selected_test_prediction_mean": float(selected_mean),
            "largest_mean_candidate": max_name,
            "largest_test_prediction_mean": float(max_mean),
            "candidate_count": len(candidate_means),
        }
    return None


def build_prediction_distribution_quality_signal(
    *,
    payload: dict[str, object] | None,
    candidate_selection_mismatch: object,
) -> dict[str, object]:
    collapse = detect_prediction_distribution_collapse(payload)
    detected = collapse is not None
    reasons: list[str] = []
    warnings: list[str] = []
    if detected:
        warnings.append(
            "prediction_distribution_collapse="
            f"selected={collapse.get('selected')},"
            f"selected_mean={collapse.get('selected_test_prediction_mean')},"
            f"largest_mean_candidate={collapse.get('largest_mean_candidate')},"
            f"largest_mean={collapse.get('largest_test_prediction_mean')}"
        )
        if candidate_selection_mismatch is not None:
            reasons.append("prediction_distribution_collapse_vs_candidates")
    return {
        "detected": detected,
        "collapse": collapse,
        "reasons": reasons,
        "warnings": warnings,
    }


def extract_competition_faithfulness(
    *,
    evaluation_metric: object,
    evaluation_score_source: object,
    kernel_metrics_payload: dict[str, object] | None,
    evaluation_report_split_strategy: object | None,
    evaluation_contract: dict[str, object] | None,
) -> dict[str, object]:
    payload = kernel_metrics_payload or {}
    contract = evaluation_contract or {}
    accepted_score_sources = score_sources.normalize_score_source_list(contract.get("accepted_score_sources"))
    if not accepted_score_sources:
        accepted_score_sources = list(score_sources.DEFAULT_ACCEPTED_SCORE_SOURCES)

    expected_metric = str(contract.get("expected_metric") or "").strip() or None
    actual_metric = None
    metric_name_raw = payload.get("metric_name")
    if isinstance(metric_name_raw, str) and metric_name_raw.strip():
        actual_metric = metric_name_raw.strip()
    else:
        metric_raw = payload.get("metric")
        if isinstance(metric_raw, str) and metric_raw.strip():
            actual_metric = metric_raw.strip()
        else:
            actual_metric = str(evaluation_metric or "").strip() or None
    expected_split = plan_policy.normalize_split_strategy_name(contract.get("expected_split_strategy"))
    actual_split = plan_policy.normalize_split_strategy_name(payload.get("split_strategy"))
    readiness_payload = payload.get("readiness")
    if actual_split is None and isinstance(readiness_payload, dict):
        actual_split = plan_policy.normalize_split_strategy_name(readiness_payload.get("split_strategy"))
    if actual_split is None:
        actual_split = plan_policy.normalize_split_strategy_name(evaluation_report_split_strategy)

    score_source = score_sources.normalize_score_source_name(payload.get("score_source") or evaluation_score_source)
    score_source_mismatch = score_source not in accepted_score_sources
    derived_noncompetitive = any(token in score_source for token in COMPETITION_FAITHFULNESS_FALSE_SCORE_SOURCE_TOKENS)

    dataset_mode_raw = payload.get("dataset_mode")
    if dataset_mode_raw is None:
        dataset_mode_raw = payload.get("data_mode")
    if dataset_mode_raw is None:
        data_resolution = payload.get("data_resolution")
        if isinstance(data_resolution, dict):
            dataset_mode_raw = data_resolution.get("mode")
    dataset_mode = str(dataset_mode_raw).strip().lower() if isinstance(dataset_mode_raw, str) else None

    full_dataset_resolved_raw = payload.get("full_dataset_resolved")
    if full_dataset_resolved_raw is None:
        data_resolution = payload.get("data_resolution")
        if isinstance(data_resolution, dict):
            full_dataset_resolved_raw = data_resolution.get("full_dataset_resolved")
    full_dataset_resolved = bool(full_dataset_resolved_raw) if isinstance(full_dataset_resolved_raw, bool) else None

    competition_faithful_raw = payload.get("competition_faithful")
    if isinstance(competition_faithful_raw, bool):
        competition_faithful = competition_faithful_raw
    elif isinstance(payload.get("noncompetitive"), bool):
        competition_faithful = not bool(payload.get("noncompetitive"))
    elif derived_noncompetitive:
        competition_faithful = False
    else:
        competition_faithful = None

    if full_dataset_resolved is None and dataset_mode is not None:
        full_dataset_resolved = dataset_mode in {"full", "competitive", "complete"}
    elif full_dataset_resolved is None and derived_noncompetitive:
        full_dataset_resolved = False

    metric_match = True
    if expected_metric:
        metric_match = metrics_equivalent(expected_metric, actual_metric)

    split_match = True
    if expected_split:
        split_match = actual_split == expected_split

    data_faithful = True
    if bool(contract.get("require_full_dataset")) and full_dataset_resolved is False:
        data_faithful = False
    if bool(contract.get("require_competition_faithful")) and competition_faithful is False:
        data_faithful = False

    reasons: list[str] = []
    warnings: list[str] = []
    if bool(contract.get("require_metric_match")) and expected_metric and not metric_match:
        reasons.append("competition_metric_mismatch")
        warnings.append(f"expected_metric={expected_metric},actual_metric={actual_metric or 'unknown'}")
    if bool(contract.get("require_split_match")) and expected_split and not split_match:
        reasons.append("competition_split_mismatch")
        warnings.append(f"expected_split={expected_split},actual_split={actual_split or 'unknown'}")
    if bool(contract.get("require_trusted_score_source")) and score_source_mismatch:
        reasons.append("competition_score_source_mismatch")
        warnings.append(
            f"accepted_score_sources={','.join(accepted_score_sources)},actual_score_source={score_source or 'unknown'}"
        )
    if bool(contract.get("require_competition_faithful")) and competition_faithful is False:
        reasons.append("competition_evaluation_unfaithful")
        warnings.append("competition_faithful=false")
    if bool(contract.get("require_full_dataset")) and full_dataset_resolved is False:
        reasons.append("missing_competitive_data")
        warnings.append(f"dataset_mode={dataset_mode or 'unknown'}")

    faithful = not reasons
    return {
        "faithful": faithful,
        "expected_metric": expected_metric,
        "actual_metric": actual_metric,
        "expected_split_strategy": expected_split,
        "actual_split_strategy": actual_split,
        "score_source": score_source,
        "accepted_score_sources": accepted_score_sources,
        "competition_faithful": competition_faithful,
        "dataset_mode": dataset_mode,
        "full_dataset_resolved": full_dataset_resolved,
        "metric_match": metric_match,
        "split_match": split_match,
        "data_faithful": data_faithful,
        "reasons": reasons,
        "warnings": warnings,
    }


def build_competition_faithfulness_quality_signal(
    *,
    evaluation_metric: object,
    evaluation_score_source: object,
    kernel_metrics_payload: dict[str, object] | None,
    evaluation_report_split_strategy: object | None,
    evaluation_contract: dict[str, object] | None,
    force_submit: bool,
) -> dict[str, object]:
    faithfulness = extract_competition_faithfulness(
        evaluation_metric=evaluation_metric,
        evaluation_score_source=evaluation_score_source,
        kernel_metrics_payload=kernel_metrics_payload,
        evaluation_report_split_strategy=evaluation_report_split_strategy,
        evaluation_contract=evaluation_contract,
    )
    reasons_raw = faithfulness.get("reasons")
    warnings_raw = faithfulness.get("warnings")
    reasons = [reason for reason in reasons_raw if isinstance(reason, str)] if isinstance(reasons_raw, list) else []
    warnings = (
        [warning for warning in warnings_raw if isinstance(warning, str)] if isinstance(warnings_raw, list) else []
    )
    return {
        "faithfulness": faithfulness,
        "reasons": reasons,
        "warnings": warnings,
        "block_submit": bool(reasons) and not force_submit,
    }


def infer_capacity_tier(
    *,
    kernel_metrics_payload: dict[str, object] | None,
    model_summary: dict[str, object] | None,
) -> str:
    payload = kernel_metrics_payload or {}
    summary = model_summary or {}
    text_candidates: list[str] = []
    for key in ("selected_pipeline", "chosen_pipeline", "model_name", "pipeline_name"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            text_candidates.append(raw.strip().lower())
    pipelines = payload.get("pipelines")
    if isinstance(pipelines, list):
        for item in pipelines[:5]:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    text_candidates.append(name.strip().lower())
    for key in ("model_name", "selected_pipeline", "pipeline_name"):
        raw = summary.get(key)
        if isinstance(raw, str) and raw.strip():
            text_candidates.append(raw.strip().lower())
    models = summary.get("models")
    if isinstance(models, list):
        for item in models[:5]:
            if isinstance(item, str) and item.strip():
                text_candidates.append(item.strip().lower())

    tier = "medium"
    for text in text_candidates:
        if any(marker in text for marker in EXTREME_CAPACITY_MARKERS):
            return "extreme"
        if any(marker in text for marker in HIGH_CAPACITY_MARKERS):
            tier = "high"
            continue
        if tier != "high" and any(marker in text for marker in LOW_CAPACITY_MARKERS):
            tier = "low"
            continue
        if tier not in {"high", "extreme"} and any(marker in text for marker in MEDIUM_CAPACITY_MARKERS):
            tier = "medium"
    return tier


def infer_data_tier(
    *,
    competition_faithfulness: dict[str, object] | None,
    evaluation_contract: dict[str, object] | None,
) -> str:
    faithfulness = competition_faithfulness or {}
    contract = evaluation_contract or {}
    if bool(contract.get("require_full_dataset")) and faithfulness.get("full_dataset_resolved") is False:
        return "minimum_submit_data"
    if bool(faithfulness.get("faithful")):
        return "high_accuracy_data"
    if bool(faithfulness.get("metric_match")) and bool(faithfulness.get("split_match")):
        return "trusted_eval_data"
    return "minimum_submit_data"


def build_accuracy_potential(
    *,
    score_source: object,
    kernel_metrics_payload: dict[str, object] | None,
    model_summary: dict[str, object] | None,
    quality_guard: dict[str, object] | None,
    evaluation_contract: dict[str, object] | None,
) -> dict[str, object]:
    payload = kernel_metrics_payload or {}
    guard = quality_guard or {}
    faithfulness = (
        guard.get("competition_faithfulness") if isinstance(guard.get("competition_faithfulness"), dict) else {}
    )
    reasons_raw = guard.get("reasons")
    reasons = [str(item) for item in reasons_raw if isinstance(item, str)] if isinstance(reasons_raw, list) else []
    policy_blocked = any(reason in HARD_POLICY_BLOCK_REASONS for reason in reasons)
    capacity_tier = infer_capacity_tier(kernel_metrics_payload=payload, model_summary=model_summary)
    data_tier = infer_data_tier(
        competition_faithfulness=faithfulness if isinstance(faithfulness, dict) else None,
        evaluation_contract=evaluation_contract,
    )
    trusted = score_sources.is_trusted_offline_score_source(score_source)
    faithful = bool(faithfulness.get("faithful", False))
    capacity_priority = CAPACITY_TIER_PRIORITY.get(capacity_tier, 0)
    data_priority = {"minimum_submit_data": 0, "trusted_eval_data": 1, "high_accuracy_data": 2}.get(data_tier, 0)
    blocked_by_submit_only = (
        all(
            reason in {"competition_split_mismatch", "competition_evaluation_unfaithful", "missing_competitive_data"}
            for reason in reasons
        )
        if reasons
        else False
    )
    eligible = (
        False
        if policy_blocked
        else faithful or trusted or (capacity_priority >= 2 and (blocked_by_submit_only or data_priority >= 1))
    )
    status = (
        "blocked"
        if policy_blocked
        else "frontier"
        if eligible and capacity_priority >= 2
        else ("trusted" if faithful or trusted else "blocked")
    )
    primary_reason = "trusted_competition_faithful"
    if status == "frontier" and not faithful:
        primary_reason = "high_capacity_candidate_requires_better_data_or_eval"
    elif reasons:
        primary_reason = reasons[0]
    return {
        "status": status,
        "eligible": eligible,
        "trusted": trusted,
        "faithful": faithful,
        "capacity_tier": capacity_tier,
        "capacity_priority": capacity_priority,
        "data_tier": data_tier,
        "data_priority": data_priority,
        "frontier_priority": capacity_priority * 10 + data_priority * 3 + int(faithful) + int(trusted),
        "primary_reason": primary_reason,
        "quality_reasons": reasons,
    }
