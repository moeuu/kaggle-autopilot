from __future__ import annotations

DEFAULT_TARGET_MEDAL = "winner"
MEDAL_TARGET_PERCENTILES = {
    "winner": 0.001,
    "bronze": 0.10,
    "silver": 0.05,
    "gold": 0.01,
}
SUPPORTED_MEDAL_TARGETS = frozenset(MEDAL_TARGET_PERCENTILES)
TARGET_MEDAL_SCHEMA = "winner|bronze|silver|gold|null"
TARGET_MEDAL_ERROR = "evaluation_spec.target_medal must be one of: winner, bronze, silver, gold"
TARGET_RANK_PERCENTILE_NUMBER_ERROR = "evaluation_spec.target_rank_percentile must be a number in (0, 1]"
TARGET_RANK_PERCENTILE_RANGE_ERROR = "evaluation_spec.target_rank_percentile must be in (0, 1]"


def normalize_target_medal(value: object, *, default: str | None = None) -> str | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SUPPORTED_MEDAL_TARGETS:
            return normalized
    return default


def normalize_target_rank_percentile(
    value: object,
    *,
    medal: str | None = None,
    fallback: float | None = None,
) -> float | None:
    parsed = _parse_rank_percentile(value)
    if parsed is not None and 0.0 < parsed <= 1.0:
        return parsed
    if medal is not None:
        medal_target = MEDAL_TARGET_PERCENTILES.get(medal)
        if medal_target is not None:
            return medal_target
    if fallback is not None and not isinstance(fallback, bool) and 0.0 < float(fallback) <= 1.0:
        return float(fallback)
    return None


def validate_target_rank_percentile(
    value: object,
    *,
    medal: str | None,
) -> tuple[float | None, str | None]:
    parsed = _parse_rank_percentile(value)
    if parsed is None:
        if value is not None and str(value).strip():
            return None, TARGET_RANK_PERCENTILE_NUMBER_ERROR
        if medal is not None:
            return MEDAL_TARGET_PERCENTILES.get(medal), None
        return None, None
    if 0.0 < parsed <= 1.0:
        return parsed, None
    return None, TARGET_RANK_PERCENTILE_RANGE_ERROR


def _parse_rank_percentile(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None
