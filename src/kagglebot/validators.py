from __future__ import annotations

import py_compile
import re
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

from kagglebot.json_utils import parse_json_object_text

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

_KERNEL_SECRET_SCAN_SUFFIXES = {".json", ".py", ".ipynb", ".md", ".txt", ".yaml", ".yml"}

FORBIDDEN_EVALUATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bKAGGLEBOT_ORACLE_MODE\b", re.IGNORECASE),
        "Kernel sources reference KAGGLEBOT_ORACLE_MODE, which is forbidden for offline metric integrity.",
    ),
    (
        re.compile(r"\bbuild_oracle_game_map\b", re.IGNORECASE),
        "Kernel sources include oracle label-map construction, which is forbidden for offline metric integrity.",
    ),
    (
        re.compile(r"\bapply_oracle_override\b", re.IGNORECASE),
        "Kernel sources include oracle prediction overrides, which is forbidden for offline metric integrity.",
    ),
    (
        re.compile(r"\bscore_source\b[^\n]{0,80}\boracle\b", re.IGNORECASE),
        "Kernel sources emit score_source=oracle, which is forbidden for model selection.",
    ),
    (
        re.compile(r"\bscore_source\b[^\n]{0,80}\blb_proxy\b", re.IGNORECASE),
        "Kernel sources emit score_source=lb_proxy, which is forbidden for model selection.",
    ),
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
    try:
        candidate.relative_to(dest_dir)
    except ValueError:
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
        payload = parse_json_object_text(meta_text)
        if payload is not None:
            raw_code_file = payload.get("code_file")
            code_file = raw_code_file if isinstance(raw_code_file, str) else None

    for candidate in ("main.py", "kernel.py", code_file):
        if not candidate:
            continue
        path = package_dir / candidate
        if path.exists():
            content.append(path.read_text(encoding="utf-8", errors="ignore"))
    for path in package_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _KERNEL_SECRET_SCAN_SUFFIXES:
            continue
        try:
            content.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    matches = []
    for text in content:
        matches.extend(scan_text_for_secrets(text))
    if matches:
        unique = sorted(set(matches))
        raise ValueError(f"Secret pattern detected in kernel package: {unique}")


def validate_kernel_sources(kernel_dir: Path, *, require_kaggle_input: bool = True) -> list[str]:
    """Validate kernel source quality and reject non-generalizable evaluation shortcuts."""
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
    for pattern, message in FORBIDDEN_EVALUATION_PATTERNS:
        if pattern.search(content):
            issues.append(message)
    return issues


def ensure_kernel_sources_valid(kernel_dir: Path, *, require_kaggle_input: bool = True) -> None:
    issues = validate_kernel_sources(kernel_dir, require_kaggle_input=require_kaggle_input)
    if issues:
        detail = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"Kernel source validation failed:\n{detail}")


def kernel_source_preflight_error(
    kernel_dir: Path,
    *,
    require_kaggle_input: bool = True,
    format_error: Callable[[Exception], str] | None = None,
) -> str | None:
    """Return kernel source preflight error text, or None when sources are launch-ready."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return (
            "RuntimeError: Local autopilot requires kernel.py, but "
            f"{kernel_path} was not found. "
            "Run planning/implement to generate kernel.py first."
        )
    try:
        ensure_kernel_sources_valid(kernel_dir, require_kaggle_input=require_kaggle_input)
    except Exception as exc:  # noqa: BLE001
        if format_error is not None:
            return format_error(exc)
        return f"{exc.__class__.__name__}: {exc}".strip()
    return None
