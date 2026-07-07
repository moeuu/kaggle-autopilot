from __future__ import annotations

from pathlib import Path
from re import IGNORECASE, finditer

from kagglebot.asset_modality import artifact_stem
from kagglebot.sample_name_aliases import SAMPLE_COMPACT_NAME_ALIASES, SAMPLE_OUTPUT_NAME_TOKENS
from kagglebot.submission_extension_hints import ARCHIVE_SUBMISSION_SUFFIXES, NON_TABULAR_SUBMISSION_SUFFIXES
from kagglebot.submission_sample_discovery import SQLITE_TABULAR_SUFFIXES, TABULAR_SUBMISSION_SUFFIXES

EXPECTED_OUTPUT_FILENAME_PATTERNS = (
    r"`([^`]+)`",
    r"['\"]([^'\"]+)['\"]",
    (
        r"\b(?:named|called)\s+(?:as\s+)?"
        r"([A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9][A-Za-z0-9_.-]*)"
    ),
    (
        r"\b(?:filename|file name|output file|submission file)\s*(?:is|:|-)?\s*"
        r"(?:a\s+single\s+)?([A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9][A-Za-z0-9_.-]*)"
    ),
    (
        r"\b(?:save|write|upload|submit)\s+(?:your\s+)?(?:predictions?|submission|output|file)?\s*"
        r"(?:as|to|in|:)\s*([A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9][A-Za-z0-9_.-]*)"
    ),
    (
        r"\b(?:file|output|upload|submit)\s+"
        r"(?:a\s+single\s+)?([A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9][A-Za-z0-9_.-]*)"
    ),
)
EXPECTED_OUTPUT_EXCLUDED_STEMS = {
    "example",
    "sample",
    "template",
    "train",
    "training",
    "test",
    "testing",
    "valid",
    "validation",
    "oof",
    "fold",
}
CONFIGURED_TEMPLATE_STEMS = {
    "example",
    "example_submission",
    "sample",
    "sample_submission",
    "submission_format",
    "submission_sample",
    "submission_template",
    "template",
    "template_submission",
}


def all_submission_output_suffixes() -> frozenset[str]:
    return frozenset(
        set(TABULAR_SUBMISSION_SUFFIXES)
        | set(SQLITE_TABULAR_SUFFIXES)
        | set(NON_TABULAR_SUBMISSION_SUFFIXES)
        | set(ARCHIVE_SUBMISSION_SUFFIXES)
    )


def all_submission_output_suffixes_ordered() -> tuple[str, ...]:
    return tuple(sorted(all_submission_output_suffixes(), key=len, reverse=True))


def tabular_submission_output_suffixes() -> frozenset[str]:
    return frozenset(TABULAR_SUBMISSION_SUFFIXES)


def tabular_submission_output_suffixes_ordered() -> tuple[str, ...]:
    return tuple(sorted(tabular_submission_output_suffixes(), key=len, reverse=True))


def non_tabular_submission_output_suffixes() -> frozenset[str]:
    return all_submission_output_suffixes() - tabular_submission_output_suffixes()


def non_tabular_submission_output_suffixes_ordered() -> tuple[str, ...]:
    return tuple(sorted(non_tabular_submission_output_suffixes(), key=len, reverse=True))


def configured_submission_filename_is_template(name: str) -> bool:
    stem = artifact_stem(Path(str(name or "").strip())).lower()
    if stem in CONFIGURED_TEMPLATE_STEMS:
        return True
    compact = stem.replace("-", "").replace("_", "")
    if compact in SAMPLE_COMPACT_NAME_ALIASES:
        return True
    tokens = {token for token in stem.replace("-", "_").split("_") if token}
    template_tokens = {"example", "sample", "template"}
    return bool(tokens & template_tokens and tokens & SAMPLE_OUTPUT_NAME_TOKENS)


def first_allowed_expected_output_suffix(
    expected_suffixes: list[str] | tuple[str, ...] | None,
    *,
    allowed_suffixes: set[str] | frozenset[str] | tuple[str, ...],
) -> str | None:
    allowed = set(normalize_expected_output_suffixes(tuple(allowed_suffixes)))
    for suffix in normalize_expected_output_suffixes(expected_suffixes):
        if suffix in allowed:
            return suffix
    return None


def expected_output_filename_from_text(
    text: str,
    *,
    expected_suffixes: list[str] | tuple[str, ...] | None,
    allowed_suffixes: set[str] | frozenset[str] | tuple[str, ...],
) -> str | None:
    expected = normalize_expected_output_suffixes(expected_suffixes)
    allowed = normalize_expected_output_suffixes(tuple(allowed_suffixes))
    candidates: list[str] = []
    seen: set[str] = set()
    for pattern in EXPECTED_OUTPUT_FILENAME_PATTERNS:
        for match in finditer(pattern, text, flags=IGNORECASE):
            raw = next((group for group in match.groups() if group), "")
            cleaned = clean_expected_output_filename(
                raw,
                expected_suffixes=expected,
                allowed_suffixes=allowed,
            )
            if cleaned is None or cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            candidates.append(cleaned)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda value: expected_output_filename_score(
            value,
            expected_suffixes=expected,
            allowed_suffixes=allowed,
        ),
        reverse=True,
    )[0]


def output_filename_from_format_text(
    text: str,
    *,
    expected_suffixes: list[str] | tuple[str, ...] | None,
    allowed_suffixes: set[str] | frozenset[str] | tuple[str, ...],
    default_stem: str = "submission",
) -> str | None:
    """Return an explicit output filename from format text, or a stem plus the first allowed suffix."""

    explicit_filename = expected_output_filename_from_text(
        text,
        expected_suffixes=expected_suffixes,
        allowed_suffixes=allowed_suffixes,
    )
    if explicit_filename is not None:
        return explicit_filename
    suffix = first_allowed_expected_output_suffix(
        expected_suffixes,
        allowed_suffixes=allowed_suffixes,
    )
    if suffix is None:
        return None
    stem = str(default_stem or "submission").strip() or "submission"
    return f"{stem}{suffix}"


def normalize_expected_output_suffixes(expected_suffixes: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_suffix in expected_suffixes or []:
        suffix = str(raw_suffix or "").strip().lower()
        if not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix not in normalized:
            normalized.append(suffix)
    return tuple(normalized)


def clean_expected_output_filename(
    raw: str,
    *,
    expected_suffixes: tuple[str, ...],
    allowed_suffixes: tuple[str, ...],
) -> str | None:
    value = str(raw or "").strip().strip("`'\" \t\r\n.,;:)]}")
    if not value or "/" in value or "\\" in value:
        return None
    path = Path(value)
    if path.name != value or value in {".", ".."} or ".." in path.parts:
        return None
    lower = value.lower()
    matched_suffix = output_suffix(lower, allowed_suffixes=allowed_suffixes)
    if matched_suffix is None:
        return None
    if expected_suffixes and matched_suffix not in expected_suffixes:
        return None
    stem = lower[: -len(matched_suffix)] if matched_suffix else path.stem.lower()
    stem_tokens = {token for token in stem.replace("-", "_").split("_") if token}
    if stem_tokens & EXPECTED_OUTPUT_EXCLUDED_STEMS:
        return None
    return value


def output_suffix(filename: str, *, allowed_suffixes: tuple[str, ...]) -> str | None:
    lower = filename.lower()
    for suffix in sorted(allowed_suffixes, key=len, reverse=True):
        if lower.endswith(suffix):
            return suffix
    return None


def expected_output_filename_score(
    value: str,
    *,
    expected_suffixes: tuple[str, ...],
    allowed_suffixes: tuple[str, ...],
) -> tuple[int, int, int, str]:
    lower = value.lower()
    suffix = output_suffix(lower, allowed_suffixes=allowed_suffixes)
    stem = lower[: -len(suffix)] if suffix else Path(value).stem.lower()
    tokens = {token for token in stem.replace("-", "_").split("_") if token}
    score = 0
    if suffix in expected_suffixes:
        score += 4
    if tokens & (SAMPLE_OUTPUT_NAME_TOKENS | {"pred", "preds", "result", "results"}):
        score += 3
    return score, len(str(suffix or "")), len(value), value
