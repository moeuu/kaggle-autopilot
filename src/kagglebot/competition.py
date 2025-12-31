from __future__ import annotations

import re
from urllib.parse import urlparse

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_competition_slug(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Competition value is empty.")

    if "://" not in raw and ("kaggle.com/" in raw or raw.startswith("www.kaggle.com/")):
        raw = f"https://{raw}"

    if "://" not in raw:
        slug = raw.strip("/")
        if "/" in slug:
            raise ValueError(f"Expected a slug, got '{value}'.")
        _validate_slug(slug)
        return slug

    parsed = urlparse(raw)
    if not parsed.netloc:
        raise ValueError(f"Invalid competition URL: '{value}'.")
    if "kaggle.com" not in parsed.netloc:
        raise ValueError("Competition URL must be from kaggle.com.")

    segments = [seg for seg in parsed.path.split("/") if seg]
    if len(segments) >= 2 and segments[0] in {"c", "competitions"}:
        slug = segments[1]
        _validate_slug(slug)
        return slug

    raise ValueError(f"Unable to parse competition slug from '{value}'.")


def rules_url_for_slug(slug: str) -> str:
    _validate_slug(slug)
    return f"https://www.kaggle.com/competitions/{slug}/rules"


def _validate_slug(slug: str) -> None:
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"Invalid competition slug '{slug}'.")
