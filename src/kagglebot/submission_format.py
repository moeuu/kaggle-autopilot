from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_UNKNOWN,
)


@dataclass(frozen=True)
class SubmissionFormatHint:
    columns: list[str] | None
    delimiter: str | None
    expected_suffixes: list[str] | None
    artifact_class: str | None = None
    artifact_container: str | None = None


_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", re.S)
_DISALLOWED_SUBMISSION_HEADING_TERMS = (
    "submission code requirements",
    "code requirements",
    "foundational rules",
    "general competition rules",
)
_BULLET_COLUMN_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s*`(?P<name>[^`]+)`\s*(?::|-|–|—)\s*",
)
_BOLD_BULLET_COLUMN_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s*\\*\\*(?P<name>[^*]+)\\*\\*\s*(?::|-|–|—)\s*",
)
_KNOWN_SUBMISSION_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl", ".zip")
_SUBMISSION_EXTENSION_RE = re.compile(r"(?<![A-Za-z0-9_])\.(csv|tsv|txt|parquet|jsonl|json|zip)\b", re.I)
_SUBMISSION_CODE_FENCE_LANG_RE = re.compile(r"^\s*```(?P<lang>[A-Za-z0-9_+-]+)\s*$")
_SUBMISSION_REQUIREMENT_RE = re.compile(r"\b(must|required|required to|should|need to|has to|only)\b", re.I)
_SUBMISSION_ACTION_RE = re.compile(r"\b(submit|submission|upload)\b", re.I)
_SUBMISSION_FILE_RE = re.compile(r"\b(file|format|archive)\b", re.I)
_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bzip(?:ped)?\b", re.I), ".zip"),
    (re.compile(r"\bjsonl\b|\bndjson\b", re.I), ".jsonl"),
    (re.compile(r"\bparquet\b", re.I), ".parquet"),
    (re.compile(r"\btsv\b|\btab[-\s]*separated\b", re.I), ".tsv"),
    (re.compile(r"\bcsv\b", re.I), ".csv"),
    (re.compile(r"\bjson\b", re.I), ".json"),
    (re.compile(r"\btxt\b|\btext\s+file\b", re.I), ".txt"),
)
_CODE_FENCE_LANG_TO_SUFFIX = {
    "csv": ".csv",
    "tsv": ".tsv",
    "jsonl": ".jsonl",
    "json": ".json",
    "parquet": ".parquet",
    "txt": ".txt",
}
_BUNDLE_MARKERS = (
    "model weights",
    "weights",
    "inference script",
    "checkpoint",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
)
_MULTI_FILE_ZIP_MARKERS = (
    "zip file containing",
    "one file per",
    "per-image",
    "per image",
    "per-sample",
    "per sample",
    "masks",
    ".tif",
    ".tiff",
    ".png",
)
_NON_ARTIFACT_JSONTXT_CONTEXT_MARKERS = (
    "example json",
    "topology json",
    "metadata json",
    "json metadata",
    "description json",
    "json description",
    "text description",
    "txt description",
    "sample json",
    "sample txt",
    "input json",
    "input txt",
)


def extract_submission_section(markdown: str) -> str | None:
    text = markdown.strip()
    if not text:
        return None
    section = _extract_section(text, keywords=("submission format", "submission file", "submission schema"))
    if section:
        return section
    return _extract_section(
        text,
        keywords=("submission",),
        disallowed_keywords=_DISALLOWED_SUBMISSION_HEADING_TERMS,
    )


def _extract_section(
    markdown: str,
    *,
    keywords: tuple[str, ...],
    disallowed_keywords: tuple[str, ...] = (),
) -> str | None:
    lines = markdown.splitlines()
    matches: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        match = _HEADING_RE.match(line.strip())
        if not match:
            continue
        title = match.group(2).strip().lower()
        if any(term in title for term in disallowed_keywords):
            continue
        if any(keyword in title for keyword in keywords):
            # Prefer higher-level (smaller) headings when multiple sections match.
            matches.append((len(match.group(1)), idx))
    if not matches:
        return None
    start_level, start_idx = min(matches)
    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        match = _HEADING_RE.match(lines[idx].strip())
        if match and len(match.group(1)) <= start_level:
            end_idx = idx
            break
    section = "\n".join(lines[start_idx:end_idx]).strip()
    return section or None


def parse_submission_format(markdown: str) -> SubmissionFormatHint:
    columns, delimiter = _parse_columns_from_markdown(markdown)
    expected_suffixes = _parse_expected_suffixes_from_markdown(markdown)
    artifact_class, artifact_container = _infer_artifact_shape(
        markdown=markdown,
        columns=columns,
        expected_suffixes=expected_suffixes,
    )
    if columns and not columns_look_plausible(columns):
        columns = None
    return SubmissionFormatHint(
        columns=columns,
        delimiter=delimiter,
        expected_suffixes=expected_suffixes,
        artifact_class=artifact_class,
        artifact_container=artifact_container,
    )


def load_submission_format_hint(path: Path) -> SubmissionFormatHint | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    hint = parse_submission_format(text)
    columns = hint.columns
    delimiter = hint.delimiter
    expected_suffixes = hint.expected_suffixes
    artifact_class = hint.artifact_class
    artifact_container = hint.artifact_container
    if columns and not columns_look_plausible(columns):
        columns = None
    if columns or delimiter or expected_suffixes or artifact_class:
        return SubmissionFormatHint(
            columns=columns,
            delimiter=delimiter,
            expected_suffixes=expected_suffixes,
            artifact_class=artifact_class,
            artifact_container=artifact_container,
        )
    return None


def _parse_columns_from_markdown(markdown: str) -> tuple[list[str] | None, str | None]:
    for block in _CODE_BLOCK_RE.findall(markdown):
        cols, delim = _parse_columns_from_block(block)
        if cols:
            return cols, delim
    lines = markdown.splitlines()
    table_header = _parse_markdown_table_header(lines)
    if table_header:
        return table_header, None
    bullet_columns = _parse_bullet_column_definitions(lines)
    if bullet_columns:
        return bullet_columns, None
    for line in lines:
        cols, delim = _parse_columns_from_line(line)
        if cols:
            return cols, delim
    return None, None


def _parse_columns_from_block(block: str) -> tuple[list[str] | None, str | None]:
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        cols, delim = _parse_columns_from_line(line)
        if cols:
            return cols, delim
    return None, None


def _parse_markdown_table_header(lines: list[str]) -> list[str] | None:
    for idx, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        next_line = lines[idx + 1].strip()
        if not next_line or "|" not in next_line:
            continue
        if "-" not in next_line:
            continue
        header = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(header) >= 2 and all(header):
            return header
    return None


def _parse_bullet_column_definitions(lines: list[str]) -> list[str] | None:
    """Parse column lists expressed as bullet definitions.

    Common in competition "Submission Format" blocks:
    - `Id`: ...
    - `Category`: ...
    """
    columns: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue
        match = _BULLET_COLUMN_RE.match(line)
        if match is None:
            match = _BOLD_BULLET_COLUMN_RE.match(line)
        if match is None:
            continue
        name = match.group("name").strip()
        if not name or name in columns:
            continue
        columns.append(name)
    if len(columns) >= 2 and columns_look_plausible(columns):
        return columns
    return None


def _parse_columns_from_line(line: str) -> tuple[list[str] | None, str | None]:
    if "\t" in line:
        cols = [cell.strip() for cell in line.split("\t")]
        return _clean_columns(cols), "\t"
    if "," in line:
        cols = [cell.strip() for cell in line.split(",")]
        return _clean_columns(cols), ","
    if "|" in line and line.count("|") >= 2:
        cols = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return _clean_columns(cols), None
    return None, None


def _clean_columns(cols: list[str]) -> list[str] | None:
    cleaned = [c for c in cols if c]
    if len(cleaned) < 2:
        return None
    return cleaned


def _parse_expected_suffixes_from_markdown(markdown: str) -> list[str] | None:
    """Infer expected submission file suffixes from submission format text."""
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    lines = markdown.splitlines()
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        line_lower = line.lower()
        base_score = 1.0
        if _SUBMISSION_ACTION_RE.search(line_lower):
            base_score += 2.0
        if _SUBMISSION_FILE_RE.search(line_lower):
            base_score += 1.0
        if _SUBMISSION_REQUIREMENT_RE.search(line_lower):
            base_score += 2.0
        explicit_tokens = _extract_suffixes_from_line(line)
        implicit_tokens = _extract_suffix_tokens_from_line(line)
        if explicit_tokens:
            base_score += 0.5
        language_suffix = _extract_suffix_from_code_fence_language(line)
        if language_suffix:
            implicit_tokens.append(language_suffix)

        merged_tokens: list[str] = []
        for token in [*explicit_tokens, *implicit_tokens]:
            if token not in merged_tokens:
                merged_tokens.append(token)

        for token in merged_tokens:
            if token in {".json", ".txt"} and _line_looks_like_non_artifact_suffix_context(line_lower):
                continue
            if token not in order:
                order[token] = idx
            score = base_score
            if token == ".zip" and "contain" in line_lower:
                score += 1.5
            scores[token] = scores.get(token, 0.0) + score

    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: (-item[1], order.get(item[0], 0), item[0]))
    suffixes = [suffix for suffix, _ in ranked if suffix in _KNOWN_SUBMISSION_SUFFIXES]
    return suffixes or None


def _infer_artifact_shape(
    *,
    markdown: str,
    columns: list[str] | None,
    expected_suffixes: list[str] | None,
) -> tuple[str | None, str | None]:
    lowered = markdown.lower()
    suffixes = expected_suffixes or []
    if ".zip" in suffixes:
        if any(marker in lowered for marker in _BUNDLE_MARKERS):
            return ARTIFACT_CLASS_BUNDLE, "zip"
        if any(marker in lowered for marker in _MULTI_FILE_ZIP_MARKERS):
            return ARTIFACT_CLASS_MULTI_FILE_ZIP, "zip"
        return ARTIFACT_CLASS_MULTI_FILE_ZIP, "zip"
    if columns:
        return ARTIFACT_CLASS_TABULAR, "file"
    non_tabular_suffixes = [
        suffix for suffix in suffixes if suffix not in {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
    ]
    if non_tabular_suffixes:
        return ARTIFACT_CLASS_SINGLE_FILE, "file"
    if any(marker in lowered for marker in _BUNDLE_MARKERS):
        return ARTIFACT_CLASS_BUNDLE, "zip"
    if suffixes:
        filtered = [suffix for suffix in suffixes if not _suffix_is_noise(suffix=suffix, text=lowered)]
        if filtered and all(suffix in {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"} for suffix in filtered):
            if any(suffix in {".csv", ".tsv", ".parquet"} for suffix in filtered):
                return ARTIFACT_CLASS_TABULAR, "file"
            if any(suffix in {".json", ".jsonl", ".txt"} for suffix in filtered):
                return ARTIFACT_CLASS_SINGLE_FILE, "file"
    return ARTIFACT_CLASS_UNKNOWN, None


def _suffix_is_noise(*, suffix: str, text: str) -> bool:
    if suffix not in {".json", ".txt"}:
        return False
    return any(marker in text for marker in _NON_ARTIFACT_JSONTXT_CONTEXT_MARKERS)


def _line_looks_like_non_artifact_suffix_context(line_lower: str) -> bool:
    return any(marker in line_lower for marker in _NON_ARTIFACT_JSONTXT_CONTEXT_MARKERS)


def _extract_suffixes_from_line(line: str) -> list[str]:
    """Collect explicit extension mentions like '.csv' or '.zip' from a line."""
    suffixes: list[str] = []
    for match in _SUBMISSION_EXTENSION_RE.finditer(line):
        suffix = f".{match.group(1).lower()}"
        if suffix == ".jsonl":
            suffixes.append(suffix)
            continue
        normalized = _normalize_suffix(suffix)
        if normalized is not None:
            suffixes.append(normalized)
    return suffixes


def _extract_suffix_tokens_from_line(line: str) -> list[str]:
    """Collect implicit format tokens like 'zip file' or 'tab-separated' from a line."""
    suffixes: list[str] = []
    for pattern, suffix in _SUBMISSION_TOKEN_PATTERNS:
        if pattern.search(line):
            suffixes.append(suffix)
    return suffixes


def _extract_suffix_from_code_fence_language(line: str) -> str | None:
    """Map fenced-code language markers to likely submission suffixes."""
    match = _SUBMISSION_CODE_FENCE_LANG_RE.match(line)
    if not match:
        return None
    lang = match.group("lang").strip().lower()
    return _CODE_FENCE_LANG_TO_SUFFIX.get(lang)


def _normalize_suffix(suffix: str) -> str | None:
    """Normalize suffix aliases to the canonical extension set."""
    value = suffix.strip().lower()
    if value == ".ndjson":
        return ".jsonl"
    if value in _KNOWN_SUBMISSION_SUFFIXES:
        return value
    return None


def _columns_look_plausible(columns: list[str]) -> bool:
    if len(columns) < 2:
        return False
    for col in columns:
        value = str(col).strip()
        if not value:
            return False
        if len(value) > 80:
            return False
        if value.count(" ") > 4:
            return False
        if not any(ch.isalnum() for ch in value):
            return False
        for ch in value:
            if ch.isalnum() or ch in " _-./():":  # allow common header punctuation
                continue
            return False
    return True


def columns_look_plausible(columns: list[str]) -> bool:
    return _columns_look_plausible(columns)
