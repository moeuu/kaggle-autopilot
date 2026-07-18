from __future__ import annotations

import py_compile
import re
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import py7zr
import rarfile

from kagglebot.asset_modality import (
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_NAME_TOKENS,
    MODEL_ARTIFACT_SUFFIXES,
    artifact_stem,
    artifact_suffix,
)
from kagglebot.compression_suffixes import open_zstd_tar
from kagglebot.json_utils import parse_json_object_text
from kagglebot.sample_name_aliases import SAMPLE_OUTPUT_NAME_TOKENS
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_SUFFIXES,
    NON_TABULAR_SUBMISSION_SUFFIXES,
    TAR_ARCHIVE_SUBMISSION_SUFFIXES,
    ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES,
)
from kagglebot.submission_sample_discovery import TABULAR_SUBMISSION_SUFFIXES, tabular_suffix

_SLUG_RE = re.compile(r"^[a-z0-9_-]+$")

SECRET_PATTERNS = [
    r"kaggle\.json",
    r"\bkaggle[_-]?key\b\s*[:=]\s*['\"]?[^\s'\"]{3,}",
    r"\bkaggle[_-]?username\b\s*[:=]\s*['\"]?[^\s'\"]{3,}",
    r"\bapi[_-]?key\b\s*[:=]\s*(?!(?:os\.)?(?:getenv|environ)\b)(?![A-Za-z_][A-Za-z0-9_]*\s*\()"
    r"['\"]?[^\s'\"]{8,}",
    r"\bpassword\b\s*[:=]\s*['\"]?[^\s'\"]{4,}",
    r"\bsecret\b\s*[:=]\s*['\"]?[^\s'\"]{4,}",
    r"\b(?:access|refresh|auth|bearer)?_?token\b\s*[:=]\s*['\"][^'\"]{8,}",
]

_KERNEL_SECRET_SCAN_SUFFIXES = {".json", ".py", ".ipynb", ".md", ".txt", ".yaml", ".yml"}
_SUPPORTED_TABULAR_SUBMISSION_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)
_SUPPORTED_NON_TABULAR_SINGLE_FILE_SUFFIXES = set(NON_TABULAR_SUBMISSION_SUFFIXES)
_SUPPORTED_ARCHIVE_SUBMISSION_SUFFIXES = set(ARCHIVE_SUBMISSION_SUFFIXES)
_SUPPORTED_TAR_ARCHIVE_SUFFIXES = set(TAR_ARCHIVE_SUBMISSION_SUFFIXES)
_SUPPORTED_ZSTD_TAR_ARCHIVE_SUFFIXES = set(ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES)
_SUPPORTED_COMPOUND_SUBMISSION_OUTPUT_NAMES = {
    f"submission{suffix}" for suffix in _SUPPORTED_ARCHIVE_SUBMISSION_SUFFIXES
}
_SUBMISSION_OUTPUT_FILENAME_RE = re.compile(
    r"(?<![A-Za-z0-9_])submission[A-Za-z0-9_.-]*(?:\.[A-Za-z0-9]+)+",
    re.IGNORECASE,
)
_GENERIC_OUTPUT_FILENAME_RE = re.compile(
    r"([A-Za-z0-9_./-]*[A-Za-z0-9][A-Za-z0-9_.-]*(?:\.[A-Za-z0-9]+)+)",
    re.IGNORECASE,
)
_SUBMISSION_OUTPUT_CONTRACT_MARKERS = (
    "kagglebot_submission_filename",
    "submission_output_name",
    "submission_manifest.json",
)
_GENERIC_SUBMISSION_NAME_TOKENS = SAMPLE_OUTPUT_NAME_TOKENS | {
    "forecast",
    "forecasts",
    "mask",
    "masks",
    "pred",
    "preds",
    "result",
    "results",
    "segmentation",
    "segmentations",
    "sub",
    "submit",
}
_GENERIC_SUBMISSION_EXCLUDE_TOKENS = {
    "cv",
    "diagnostic",
    "diagnostics",
    "feature",
    "features",
    "fold",
    "format",
    "metric",
    "metrics",
    "oof",
    "sample",
    "schema",
    "split",
    "template",
    "train",
    "valid",
    "validation",
}
_GENERIC_OUTPUT_WRITE_MARKERS = (
    "/kaggle/working",
    "np.save",
    ".save(",
    ".to_csv",
    ".to_excel",
    ".to_feather",
    ".to_json",
    ".to_parquet",
    ".to_pickle",
    ".write_bytes",
    ".write_text",
    "open(",
    "output_dir",
    "working_dir",
)

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


def _references_supported_submission_output(content: str) -> bool:
    lowered = content.lower()
    if any(marker in lowered for marker in _SUBMISSION_OUTPUT_CONTRACT_MARKERS):
        return True
    if any(
        _is_supported_submission_output_name(match.group(0))
        for match in _SUBMISSION_OUTPUT_FILENAME_RE.finditer(content)
    ):
        return True
    return any(_line_references_supported_generic_output(line) for line in content.splitlines())


def _is_supported_submission_output_name(raw_name: str) -> bool:
    name = Path(raw_name).name.lower()
    if name in _SUPPORTED_COMPOUND_SUBMISSION_OUTPUT_NAMES:
        return True
    if _archive_submission_suffix(Path(name)) in _SUPPORTED_ARCHIVE_SUBMISSION_SUFFIXES:
        return True
    suffix = tabular_suffix(Path(name))
    if suffix in _SUPPORTED_TABULAR_SUBMISSION_SUFFIXES:
        return True
    return artifact_suffix(Path(name)) in _SUPPORTED_NON_TABULAR_SINGLE_FILE_SUFFIXES


def _line_references_supported_generic_output(line: str) -> bool:
    lowered = line.lower()
    if "/kaggle/input" in lowered:
        return False
    if not any(marker in lowered for marker in _GENERIC_OUTPUT_WRITE_MARKERS):
        return False
    return any(
        _is_supported_submission_output_name(match.group(1)) and _generic_submission_name_score(match.group(1)) > 0
        for match in _GENERIC_OUTPUT_FILENAME_RE.finditer(line)
    )


def _generic_submission_name_score(raw_name: str) -> int:
    path = Path(raw_name)
    stem = _submission_output_stem(path).lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    if tokens & _GENERIC_SUBMISSION_EXCLUDE_TOKENS:
        return 0
    if tokens & _GENERIC_SUBMISSION_NAME_TOKENS:
        return 3
    artifact_candidate = artifact_suffix(path)
    if artifact_candidate in (MODEL_ARTIFACT_SUFFIXES | MODEL_ARTIFACT_COMPOUND_SUFFIXES) and (
        tokens & MODEL_ARTIFACT_NAME_TOKENS
    ):
        return 3
    compact = re.sub(r"[^a-z0-9]+", "", stem)
    if compact in _GENERIC_SUBMISSION_NAME_TOKENS:
        return 2
    if artifact_candidate in (MODEL_ARTIFACT_SUFFIXES | MODEL_ARTIFACT_COMPOUND_SUFFIXES) and (
        compact in MODEL_ARTIFACT_NAME_TOKENS
    ):
        return 2
    return 0


def _submission_output_stem(path: Path) -> str:
    return artifact_stem(path)


def _archive_submission_suffix(path: Path) -> str:
    suffix = artifact_suffix(path)
    return suffix if suffix in _SUPPORTED_ARCHIVE_SUBMISSION_SUFFIXES else ""


def validate_slug(slug: str) -> str:
    cleaned = slug.strip()
    if not cleaned:
        raise ValueError("Competition slug is empty.")
    if not _SLUG_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid competition slug '{slug}'.")
    return cleaned


def ensure_safe_extract_path(dest_dir: Path, member: zipfile.ZipInfo | tarfile.TarInfo | str) -> Path:
    dest_dir = dest_dir.resolve()
    member_name = member if isinstance(member, str) else getattr(member, "filename", getattr(member, "name", ""))
    candidate = (dest_dir / member_name).resolve()
    try:
        candidate.relative_to(dest_dir)
    except ValueError:
        raise ValueError(f"Unsafe path detected in archive: {member_name}")
    return candidate


def _remember_archive_target(seen_targets: set[Path], target: Path, member_name: str) -> None:
    if target in seen_targets:
        raise ValueError(f"Duplicate archive member target: {member_name}")
    seen_targets.add(target)


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    if member.create_system != 3:
        return False
    return stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK


def safe_extract_zip(zip_path: Path, dest_dir: Path, *, overwrite: bool = True) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        seen_targets: set[Path] = set()
        for member in archive.infolist():
            target = ensure_safe_extract_path(dest_dir, member)
            _remember_archive_target(seen_targets, target, member.filename)
            if member.flag_bits & 0x1:
                raise ValueError(f"Unsupported encrypted zip member: {member.filename}")
            if _zip_member_is_symlink(member):
                raise ValueError(f"Unsupported zip symlink member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                # Stream extraction to avoid loading large archive members fully into memory.
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted.append(target)
    return extracted


def safe_extract_tar(tar_path: Path, dest_dir: Path, *, overwrite: bool = True) -> list[Path]:
    with tarfile.open(tar_path, "r:*") as archive:
        return _safe_extract_tar_members(archive, dest_dir, overwrite=overwrite)


def safe_extract_tar_zst(tar_path: Path, dest_dir: Path, *, overwrite: bool = True) -> list[Path]:
    with open_zstd_tar(tar_path) as archive:
        return _safe_extract_tar_members(archive, dest_dir, overwrite=overwrite)


def _safe_extract_tar_members(archive: tarfile.TarFile, dest_dir: Path, *, overwrite: bool) -> list[Path]:
    extracted: list[Path] = []
    seen_targets: set[Path] = set()
    for member in archive:
        target = ensure_safe_extract_path(dest_dir, member)
        _remember_archive_target(seen_targets, target, member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"Unsupported tar member type: {member.name}")
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            continue
        with source, target.open("wb") as dst:
            shutil.copyfileobj(source, dst, length=1024 * 1024)
        extracted.append(target)
    return extracted


def safe_extract_7z(archive_path: Path, dest_dir: Path, *, overwrite: bool = True) -> list[Path]:
    extracted: list[Path] = []
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        if archive.needs_password():
            raise ValueError(f"Unsupported password-protected 7z archive: {archive_path}")
        targets: list[str] = []
        seen_targets: set[Path] = set()
        for member in archive.list():
            name = getattr(member, "filename", "")
            target = ensure_safe_extract_path(dest_dir, name)
            _remember_archive_target(seen_targets, target, name)
            if getattr(member, "is_directory", False):
                continue
            if getattr(member, "is_symlink", False) or not getattr(member, "is_file", False):
                raise ValueError(f"Unsupported 7z member type: {name}")
            if target.exists() and not overwrite:
                continue
            targets.append(name)
            extracted.append(target)
        if targets:
            archive.extract(path=dest_dir, targets=targets)
    return extracted


def safe_extract_rar(rar_path: Path, dest_dir: Path, *, overwrite: bool = True) -> list[Path]:
    extracted: list[Path] = []
    with rarfile.RarFile(rar_path) as archive:
        seen_targets: set[Path] = set()
        for member in archive.infolist():
            member_name = getattr(member, "filename", "")
            target = ensure_safe_extract_path(dest_dir, member_name)
            _remember_archive_target(seen_targets, target, member_name)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.needs_password():
                raise ValueError(f"Unsupported password-protected rar member: {member_name}")
            if member.is_symlink() or not member.is_file():
                raise ValueError(f"Unsupported rar member type: {member_name}")
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted.append(target)
    return extracted


def _is_supported_tar_archive(path: Path) -> bool:
    return _archive_submission_suffix(path) in _SUPPORTED_TAR_ARCHIVE_SUFFIXES


def _is_zstd_tar_archive(path: Path) -> bool:
    return _archive_submission_suffix(path) in _SUPPORTED_ZSTD_TAR_ARCHIVE_SUFFIXES


def _is_supported_data_archive(path: Path) -> bool:
    return _archive_submission_suffix(path) in _SUPPORTED_ARCHIVE_SUBMISSION_SUFFIXES


def extract_data_archives(data_dir: Path, *, overwrite: bool = False, max_depth: int = 2) -> list[Path]:
    if not data_dir.exists():
        return []
    extracted: list[Path] = []
    processed: set[Path] = set()
    for _depth in range(max(max_depth, 0) + 1):
        archives = [
            path
            for path in sorted(data_dir.rglob("*"))
            if path.is_file() and _is_supported_data_archive(path) and path.resolve() not in processed
        ]
        if not archives:
            break
        for archive_path in archives:
            processed.add(archive_path.resolve())
            suffix = _archive_submission_suffix(archive_path)
            if suffix == ".zip":
                extracted.extend(safe_extract_zip(archive_path, data_dir, overwrite=overwrite))
            elif suffix == ".7z":
                extracted.extend(safe_extract_7z(archive_path, data_dir, overwrite=overwrite))
            elif suffix == ".rar":
                extracted.extend(safe_extract_rar(archive_path, data_dir, overwrite=overwrite))
            elif suffix in _SUPPORTED_ZSTD_TAR_ARCHIVE_SUFFIXES:
                extracted.extend(safe_extract_tar_zst(archive_path, data_dir, overwrite=overwrite))
            else:
                extracted.extend(safe_extract_tar(archive_path, data_dir, overwrite=overwrite))
    return extracted


def extract_zip_archives(data_dir: Path, *, overwrite: bool = False) -> list[Path]:
    return extract_data_archives(data_dir, overwrite=overwrite)


extract_archives = extract_data_archives


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


def validate_kernel_sources(
    kernel_dir: Path,
    *,
    require_kaggle_input: bool = True,
    deliverable_mode: str = "leaderboard",
    required_output_names: tuple[str, ...] = (),
) -> list[str]:
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
    normalized_deliverable_mode = str(deliverable_mode or "leaderboard").strip().lower()
    if normalized_deliverable_mode == "writeup":
        missing_outputs = [
            name for name in required_output_names if not _references_required_output_name(content, name)
        ]
        if missing_outputs:
            issues.append(
                "Kernel sources do not reference required writeup notebook output artifact(s): "
                + ", ".join(missing_outputs)
                + "."
            )
    elif not _references_supported_submission_output(content):
        issues.append("Kernel sources do not reference a supported submission output artifact.")
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


def _references_required_output_name(content: str, raw_name: str) -> bool:
    name = Path(str(raw_name or "")).name.strip()
    if not name:
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])", re.IGNORECASE)
    return any(pattern.search(line) and "/kaggle/input" not in line.lower() for line in content.splitlines())


def ensure_kernel_sources_valid(
    kernel_dir: Path,
    *,
    require_kaggle_input: bool = True,
    deliverable_mode: str = "leaderboard",
    required_output_names: tuple[str, ...] = (),
) -> None:
    issues = validate_kernel_sources(
        kernel_dir,
        require_kaggle_input=require_kaggle_input,
        deliverable_mode=deliverable_mode,
        required_output_names=required_output_names,
    )
    if issues:
        detail = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"Kernel source validation failed:\n{detail}")


def kernel_source_preflight_error(
    kernel_dir: Path,
    *,
    require_kaggle_input: bool = True,
    deliverable_mode: str = "leaderboard",
    required_output_names: tuple[str, ...] = (),
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
        ensure_kernel_sources_valid(
            kernel_dir,
            require_kaggle_input=require_kaggle_input,
            deliverable_mode=deliverable_mode,
            required_output_names=required_output_names,
        )
    except Exception as exc:  # noqa: BLE001
        if format_error is not None:
            return format_error(exc)
        return f"{exc.__class__.__name__}: {exc}".strip()
    return None
