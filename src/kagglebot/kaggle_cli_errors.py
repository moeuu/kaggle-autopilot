from __future__ import annotations

from kagglebot.exceptions import KaggleCliError


def is_missing_kaggle_credentials_error(exc: KaggleCliError) -> bool:
    text = "\n".join(
        part
        for part in (
            getattr(exc, "message", ""),
            getattr(exc, "output", ""),
            getattr(exc, "stdout", ""),
            getattr(exc, "stderr", ""),
            str(exc),
        )
        if part
    )
    lowered = text.lower()
    if "kaggle.json" in lowered and "could not find" in lowered:
        return True
    if "kaggle.json" in lowered and "environment method" in lowered:
        return True
    if "api.authenticate" in lowered and "kaggle.json" in lowered:
        return True
    if "kaggle api credentials" in lowered and "not found" in lowered:
        return True
    return False
