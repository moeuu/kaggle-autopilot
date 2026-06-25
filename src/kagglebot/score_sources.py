from __future__ import annotations

TRUSTED_SCORE_SOURCES = frozenset({"cv", "holdout", "consensus"})
DEFAULT_ACCEPTED_SCORE_SOURCES = ("cv", "holdout")
GENERALIZABLE_SCORE_SOURCES = frozenset({"cv", "holdout"})


def normalize_score_source_name(value: object) -> str:
    """Normalize score_source labels for trust checks."""
    text = str(value or "").strip().lower()
    if not text:
        return "holdout"
    normalized = text.replace("-", "_").replace(" ", "_")
    alias_map = {
        "cross_validation": "cv",
        "crossval": "cv",
        "validation": "holdout",
        "lbproxy": "lb_proxy",
    }
    return alias_map.get(normalized, normalized)


def is_trusted_offline_score_source(score_source: object) -> bool:
    """Return whether score source is trusted for offline model-selection decisions."""
    return normalize_score_source_name(score_source) in TRUSTED_SCORE_SOURCES


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
