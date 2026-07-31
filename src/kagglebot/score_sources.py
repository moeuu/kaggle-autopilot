from __future__ import annotations

from collections.abc import Mapping

TRUSTED_SCORE_SOURCES = frozenset({"cv", "holdout", "consensus", "grouped_oof_cv"})
DEFAULT_ACCEPTED_SCORE_SOURCES = ("cv", "holdout")
GENERALIZABLE_SCORE_SOURCES = frozenset({"cv", "holdout"})

FROZEN_GROUPED_OOF_CONTRACT = {
    "metric": "grouped_macro_f1_moment_type",
    "direction": "maximize",
    "source": "grouped_oof_cv",
    "outer_split": "LeaveOneGroupOut_session_id",
    "folds": 5,
    "seeds": [42, 2024, 777],
    "evaluated_rows": 72,
    "biometric_sha256": "51591d1d7cffdf717edd8df557cc83d410ee08f7690f8ab4ed77b122500e87a2",
    "mapping_sha256": "fcc7c53f1eaa1e232a0d08f238b9aa7d7655c950fbb6ff2081304611829c5909",
    "evaluation_mask_sha256": "91a7f90ad72c176c18b53798d3e4195a26ccb32bdd4e0f9c843839a578605b70",
    "global_class_list": [
        "active_recovery",
        "breakthrough_wall",
        "early_push",
        "final_rep",
        "finishing_strong",
        "peak_effort",
        "post_workout",
        "pre_workout",
        "recovery_window",
        "redline",
        "rest_set",
        "steady_state",
        "warmup",
        "working_set",
    ],
}


def normalize_score_source_name(value: object) -> str:
    """Normalize score_source labels for trust checks."""
    text = str(value or "").strip().lower()
    if not text:
        return "holdout"
    normalized = text.replace("-", "_").replace(" ", "_")
    alias_map = {
        "cross_validation": "cv",
        "crossval": "cv",
        "grouped_oof_cv": "cv",
        "validation": "holdout",
        "lbproxy": "lb_proxy",
    }
    return alias_map.get(normalized, normalized)


def validate_grouped_oof_contract(payload: object) -> tuple[bool, list[str]]:
    """Validate the only grouped OOF contract trusted for this competition."""
    if not isinstance(payload, Mapping):
        return False, ["grouped_oof_contract_missing"]
    expected = FROZEN_GROUPED_OOF_CONTRACT
    errors: list[str] = []
    metric = payload.get("metric") or payload.get("technical_metric")
    source = payload.get("source") or payload.get("score_source")
    outer_split = payload.get("outer_split")
    if metric != expected["metric"]:
        errors.append("metric_mismatch")
    if payload.get("direction") != expected["direction"]:
        errors.append("direction_mismatch")
    if source != expected["source"]:
        errors.append("source_mismatch")
    if outer_split != expected["outer_split"]:
        errors.append("outer_split_mismatch")
    if payload.get("folds") != expected["folds"]:
        errors.append("fold_count_mismatch")
    if payload.get("seeds") != expected["seeds"]:
        errors.append("seed_list_mismatch")
    if payload.get("evaluated_rows") != expected["evaluated_rows"]:
        errors.append("evaluated_rows_mismatch")
    if payload.get("evaluation_mask_sha256") != expected["evaluation_mask_sha256"]:
        errors.append("evaluation_mask_sha256_mismatch")
    if payload.get("global_class_list") != expected["global_class_list"]:
        errors.append("global_class_list_mismatch")
    data_hashes = payload.get("data_hashes")
    data_hashes = data_hashes if isinstance(data_hashes, Mapping) else {}
    if data_hashes.get("biometric") != expected["biometric_sha256"]:
        errors.append("biometric_sha256_mismatch")
    if data_hashes.get("mapping") != expected["mapping_sha256"]:
        errors.append("mapping_sha256_mismatch")
    return not errors, errors


def is_trusted_offline_score_source(
    score_source: object,
    *,
    provenance: object | None = None,
) -> bool:
    """Return whether score source is trusted for offline model-selection decisions."""
    raw = str(score_source or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw == "grouped_oof_cv":
        valid, _ = validate_grouped_oof_contract(provenance)
        return valid
    normalized = normalize_score_source_name(score_source)
    return normalized in TRUSTED_SCORE_SOURCES


def normalize_generalizable_score_source(value: object) -> str:
    """Normalize a user-selectable offline score source.

    Trust checks can include broader labels such as consensus, but score-source
    selection is intentionally restricted to direct offline validation modes.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Unknown score source: {value}. Allowed values: holdout, cv.")
    normalized = normalize_score_source_name(text)
    if normalized in {"auto", "test"}:
        raise ValueError("score-source auto/test is removed; use holdout or cv.")
    if normalized in GENERALIZABLE_SCORE_SOURCES:
        return normalized
    raise ValueError(f"Unknown score source: {value}. Allowed values: holdout, cv.")


def normalize_score_source_list(value: object) -> list[str]:
    normalized: list[str] = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        candidate = normalize_score_source_name(item)
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized
