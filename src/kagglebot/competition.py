from __future__ import annotations

from urllib.parse import urlparse

from kagglebot.validators import validate_slug


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
        return _normalize_slug(slug)

    parsed = urlparse(raw)
    if not parsed.netloc:
        raise ValueError(f"Invalid competition URL: '{value}'.")
    if "kaggle.com" not in parsed.netloc:
        raise ValueError("Competition URL must be from kaggle.com.")

    segments = [seg for seg in parsed.path.split("/") if seg]
    if len(segments) >= 2 and segments[0] in {"c", "competitions"}:
        slug = segments[1]
        return _normalize_slug(slug)

    raise ValueError(f"Unable to parse competition slug from '{value}'.")


def rules_url_for_slug(slug: str) -> str:
    slug = _normalize_slug(slug)
    return f"https://www.kaggle.com/competitions/{slug}/rules"


def _normalize_slug(slug: str) -> str:
    return validate_slug(slug.lower())
