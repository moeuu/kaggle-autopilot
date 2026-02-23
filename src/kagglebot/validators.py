from __future__ import annotations

import py_compile
import re
import shutil
import zipfile
from pathlib import Path

_SLUG_RE = re.compile(r"^[a-z0-9-]+$")

SECRET_PATTERNS = [
    r"kaggle\.json",
    r"\bkaggle[_-]?key\b\s*[:=]\s*['\"]?[^\s'\"]{3,}",
    r"\bkaggle[_-]?username\b\s*[:=]\s*['\"]?[^\s'\"]{3,}",
    r"\bapi[_-]?key\b\s*[:=]\s*['\"]?[^\s'\"]{8,}",
    r"\bpassword\b\s*[:=]\s*['\"]?[^\s'\"]{4,}",
    r"\bsecret\b\s*[:=]\s*['\"]?[^\s'\"]{4,}",
    r"\b(?:access|refresh|auth|bearer)?_?token\b\s*[:=]\s*['\"][^'\"]{8,}",
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
                # Stream extraction to avoid loading large archive members fully into memory.
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted.append(target)
    return extracted


def scan_text_for_secrets(text: str) -> list[str]:
    matches = []
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)
    return matches


def validate_kernel_package(package_dir: Path) -> None:
    meta_path = package_dir / "kernel-metadata.json"
    content = []
    code_file = None
    if meta_path.exists():
        meta_text = meta_path.read_text(encoding="utf-8", errors="ignore")
        content.append(meta_text)
        try:
            import json

            payload = json.loads(meta_text)
            code_file = payload.get("code_file")
        except json.JSONDecodeError:
            code_file = None

    for candidate in ("main.py", "kernel.py", code_file):
        if not candidate:
            continue
        path = package_dir / candidate
        if path.exists():
            content.append(path.read_text(encoding="utf-8", errors="ignore"))
    matches = []
    for text in content:
        matches.extend(scan_text_for_secrets(text))
    if matches:
        unique = sorted(set(matches))
        raise ValueError(f"Secret pattern detected in kernel package: {unique}")


def validate_kernel_sources(kernel_dir: Path, *, require_kaggle_input: bool = True) -> list[str]:
    issues: list[str] = []
    if not kernel_dir.exists():
        return [f"Kernel directory not found: {kernel_dir}"]

    py_files = sorted(kernel_dir.rglob("*.py"))
    if not py_files:
        issues.append("No Python files found in kernel directory.")
        return issues

    kernel_py = kernel_dir / "kernel.py"
    if not kernel_py.exists():
        issues.append("kernel.py not found in kernel directory.")

    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            issues.append(f"Syntax error in {path.name}: {exc.msg}")

    content = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in py_files if path.is_file())
    if require_kaggle_input and "/kaggle/input" not in content:
        issues.append("Kernel sources do not reference /kaggle/input for data loading.")
    if "submission.csv" not in content:
        issues.append("Kernel sources do not reference submission.csv output.")
    if "metrics.json" not in content:
        issues.append("Kernel sources do not reference metrics.json output.")

    lowered = content.lower()
    if "prot_t5" in lowered or "t5" in lowered:
        if "automodel.from_pretrained" in lowered and "t5encodermodel" not in lowered and ".get_encoder" not in lowered:
            issues.append(
                "Detected T5/ProtT5 with AutoModel; use T5EncoderModel or "
                "model.get_encoder() to avoid decoder_input_ids errors."
            )
    return issues


def ensure_kernel_sources_valid(kernel_dir: Path, *, require_kaggle_input: bool = True) -> None:
    issues = validate_kernel_sources(kernel_dir, require_kaggle_input=require_kaggle_input)
    if issues:
        detail = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"Kernel source validation failed:\n{detail}")
