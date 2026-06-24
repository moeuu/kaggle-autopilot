from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file_or_none(path: str | Path | None) -> str | None:
    """Return SHA256 for an existing file path, otherwise None."""
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    try:
        return sha256_file(str(resolved))
    except OSError:
        return None


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()
