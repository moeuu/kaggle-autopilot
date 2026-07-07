from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Return a stable SHA256 digest for a file or directory tree."""
    resolved = Path(path)
    if resolved.is_symlink():
        raise OSError(f"unsupported symlink path for hashing: {resolved}")
    if resolved.is_file():
        return sha256_file(str(resolved))
    if not resolved.is_dir():
        raise FileNotFoundError(str(resolved))
    h = hashlib.sha256()
    h.update(b"kagglebot-dir-v1\0")
    members = sorted(resolved.rglob("*"), key=lambda member: member.relative_to(resolved).as_posix())
    for member in members:
        if member.is_symlink():
            raise OSError(f"unsupported symlink path for hashing: {member}")
        if not member.is_file() and not member.is_dir():
            continue
        relative = member.relative_to(resolved).as_posix().encode("utf-8", errors="surrogateescape")
        h.update(b"dir\0" if member.is_dir() else b"file\0")
        h.update(relative)
        h.update(b"\0")
        if member.is_dir():
            continue
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0end\0")
    return h.hexdigest()


def sha256_file_or_none(path: str | Path | None) -> str | None:
    """Return SHA256 for an existing file or directory path, otherwise None."""
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    try:
        return sha256_path(resolved)
    except OSError:
        return None


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()
