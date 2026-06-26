from __future__ import annotations

from collections.abc import Iterable
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


def parse_datetime_utc(
    value: object,
    *,
    formats: Iterable[str] = (),
    accept_utc_suffix: bool = True,
) -> datetime | None:
    """Parse an ISO-like or explicitly formatted datetime as UTC."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    assume_utc = False
    if accept_utc_suffix and text.upper().endswith(" UTC"):
        text = text[:-4].strip()
        assume_utc = True

    parsed = parse_iso_datetime_utc(text)
    if parsed is not None and not assume_utc:
        return parsed
    if parsed is None:
        for fmt in formats:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None

    if parsed.tzinfo is None or assume_utc:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
