from __future__ import annotations

from datetime import UTC, datetime


def parse_iso_datetime_utc(value: object) -> datetime | None:
    """Parse an ISO-like datetime and return a UTC-aware datetime."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
