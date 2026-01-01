from __future__ import annotations

import re
import zipfile
from pathlib import Path

_SLUG_RE = re.compile(r"^[a-z0-9-]+$")

SECRET_PATTERNS = [
    r"kaggle\.json",
    r"kaggle_key",
    r"kaggle_username",
    r"api_key",
    r"password",
    r"token",
    r"secret",
]


def validate_slug(slug: str) -> str:
    cleaned = slug.strip()
    if not cleaned:
        raise ValueError("Competition slug is empty.")
    if not _SLUG_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid competition slug '{slug}'.")
    return cleaned


def ensure_safe_extract_path(dest_dir: Path, member: zipfile.ZipInfo) -> Path:
    dest_dir = dest_dir.resolve()
    candidate = (dest_dir / member.filename).resolve()
    if not str(candidate).startswith(str(dest_dir)):
        raise ValueError(f"Unsafe path detected in zip: {member.filename}")
    return candidate


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = ensure_safe_extract_path(dest_dir, member)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


def scan_text_for_secrets(text: str) -> list[str]:
    matches = []
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)
    return matches


def validate_kernel_package(package_dir: Path) -> None:
    main_path = package_dir / "main.py"
    meta_path = package_dir / "kernel-metadata.json"
    content = []
    if main_path.exists():
        content.append(main_path.read_text(encoding="utf-8", errors="ignore"))
    if meta_path.exists():
        content.append(meta_path.read_text(encoding="utf-8", errors="ignore"))
    matches = []
    for text in content:
        matches.extend(scan_text_for_secrets(text))
    if matches:
        unique = sorted(set(matches))
        raise ValueError(f"Secret pattern detected in kernel package: {unique}")
