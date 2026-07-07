from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.asset_modality import (
    DOCUMENT_SUFFIXES,
    archive_container,
)
from kagglebot.compression_suffixes import strip_compression_suffix
from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_UNKNOWN,
)
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_EXTENSION_PATTERN,
    ARCHIVE_SUBMISSION_SUFFIXES,
    ARCHIVE_SUBMISSION_SUFFIXES_ORDERED,
    CODE_FENCE_LANG_TO_SUFFIX,
    COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES,
    COMPRESSION_TOKEN_PATTERNS,
    JSON_TEXT_NOISE_CONTEXT_MARKERS,
    MODEL_BUNDLE_MARKERS,
    MULTI_FILE_SUBMISSION_MARKERS,
    NON_TABULAR_SUBMISSION_SUFFIXES,
    SUBMISSION_TOKEN_PATTERNS,
    drop_shadowed_submission_suffixes,
    submission_extension_pattern,
)
from kagglebot.submission_sample_discovery import (
    SQLITE_TABULAR_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES_ORDERED,
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
    r"^\s*(?:[-*]|\d+[.)])\s*\*\*(?P<name>[^*]+)\*\*\s*(?::|-|–|—)\s*",
)
_PROSE_COLUMN_TOKEN = r"`?[A-Za-z_][A-Za-z0-9_./:-]*`?"
_PROSE_COLUMN_LIST_RE = re.compile(
    rf"\bcolumns?\s+(?P<columns>{_PROSE_COLUMN_TOKEN}"
    rf"(?:\s*,\s*(?:and\s+)?{_PROSE_COLUMN_TOKEN}|\s+and\s+{_PROSE_COLUMN_TOKEN}|"
    rf"\s*&\s*{_PROSE_COLUMN_TOKEN})+)",
    re.I,
)
_HEADER_COLUMN_PREFIX_RE = re.compile(
    r"\b(?:csv\s+)?(?:header\s+row|headers|header|schema|fields?)\b[^A-Za-z0-9_./:-]*(?P<columns>.+)$",
    re.I,
)
_HEADER_COLUMN_IS_RE = re.compile(
    r"\b(?:csv\s+)?(?:header\s+row|headers|header|schema|fields?)\s+(?:is|are|should\s+be|must\s+be)\s+"
    r"(?P<columns>.+)$",
    re.I,
)
_PROSE_COLUMN_TOKEN_RE = re.compile(_PROSE_COLUMN_TOKEN)
_TABULAR_SUBMISSION_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)
_ARCHIVE_SUBMISSION_SUFFIXES = set(ARCHIVE_SUBMISSION_SUFFIXES)
_NON_TABULAR_SINGLE_FILE_SUFFIXES = set(NON_TABULAR_SUBMISSION_SUFFIXES) | SQLITE_TABULAR_SUFFIXES
_PICKLE_SUBMISSION_SUFFIXES = {
    ".pkl",
    ".pickle",
    ".pkl.gz",
    ".pickle.gz",
    ".pkl.bz2",
    ".pickle.bz2",
    ".pkl.xz",
    ".pickle.xz",
    ".pkl.zst",
    ".pickle.zst",
}
_HDF_MODEL_SUBMISSION_SUFFIXES = {".hdf", ".hdf5"}
_AMBIGUOUS_MODEL_FILE_SUFFIXES = _PICKLE_SUBMISSION_SUFFIXES | _HDF_MODEL_SUBMISSION_SUFFIXES
_KNOWN_SUBMISSION_SUFFIXES = (
    *TABULAR_SUBMISSION_SUFFIXES_ORDERED,
    *ARCHIVE_SUBMISSION_SUFFIXES_ORDERED,
    *tuple(sorted(_NON_TABULAR_SINGLE_FILE_SUFFIXES, key=len, reverse=True)),
)
_KNOWN_SUBMISSION_EXTENSION_PATTERN = submission_extension_pattern(_KNOWN_SUBMISSION_SUFFIXES)
_SUBMISSION_ARCHIVE_EXTENSION_RE = re.compile(
    rf"(?<![A-Za-z0-9_])\.({ARCHIVE_SUBMISSION_EXTENSION_PATTERN})\b",
    re.I,
)
_SUBMISSION_EXTENSION_RE = re.compile(
    rf"(?P<suffix>\.(?:{_KNOWN_SUBMISSION_EXTENSION_PATTERN}))\b",
    re.I,
)
_SUBMISSION_CODE_FENCE_LANG_RE = re.compile(r"^\s*```(?P<lang>[A-Za-z0-9_+-]+)\s*$")
_SUBMISSION_REQUIREMENT_RE = re.compile(r"\b(must|required|required to|should|need to|has to|only)\b", re.I)
_SUBMISSION_ACTION_RE = re.compile(r"\b(submit|submission|upload)\b", re.I)
_SUBMISSION_FILE_RE = re.compile(r"\b(file|format|archive)\b", re.I)


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
    submission_markdown = extract_submission_section(markdown) or markdown
    columns, delimiter = _parse_columns_from_markdown(submission_markdown)
    expected_suffixes = _parse_expected_suffixes_from_markdown(submission_markdown)
    artifact_class, artifact_container = _infer_artifact_shape(
        markdown=submission_markdown,
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
    prose_columns = _parse_prose_column_list(lines)
    if prose_columns:
        return prose_columns, None
    header_columns = _parse_header_column_list(lines)
    if header_columns:
        return header_columns
    for line in lines:
        cols, delim = _parse_columns_from_line(line)
        if cols:
            return cols, delim
    return None, None


def _parse_columns_from_block(block: str) -> tuple[list[str] | None, str | None]:
    json_columns = _parse_json_object_columns_from_line(block)
    if json_columns:
        return json_columns, None
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


def _parse_prose_column_list(lines: list[str]) -> list[str] | None:
    for raw_line in lines:
        line = raw_line.strip()
        if not line or "column" not in line.lower():
            continue
        for match in _PROSE_COLUMN_LIST_RE.finditer(line):
            columns = _clean_prose_columns(match.group("columns"))
            if columns and columns_look_plausible(columns):
                return columns
    return None


def _parse_header_column_list(lines: list[str]) -> tuple[list[str], str | None] | None:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        for pattern in (_HEADER_COLUMN_IS_RE, _HEADER_COLUMN_PREFIX_RE):
            match = pattern.search(line)
            if match is None:
                continue
            cols, delim = _parse_columns_from_line(match.group("columns").strip().strip("`"))
            cols = _clean_header_columns(cols)
            if cols and columns_look_plausible(cols):
                return cols, delim
    return None


def _clean_header_columns(columns: list[str] | None) -> list[str] | None:
    if columns is None:
        return None
    cleaned = [str(col).strip().strip("`").strip().strip(".,;:").strip() for col in columns]
    return _clean_columns(cleaned)


def _clean_prose_columns(text: str) -> list[str] | None:
    columns: list[str] = []
    for match in _PROSE_COLUMN_TOKEN_RE.finditer(text):
        token = match.group(0).strip().strip("`").strip().strip(".,;:")
        if not token:
            continue
        if token.lower() in {"and", "or"}:
            continue
        if token not in columns:
            columns.append(token)
    if len(columns) < 2:
        return None
    return columns


def _parse_columns_from_line(line: str) -> tuple[list[str] | None, str | None]:
    json_columns = _parse_json_object_columns_from_line(line)
    if json_columns:
        return json_columns, None
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


def _parse_json_object_columns_from_line(line: str) -> list[str] | None:
    candidates = [line.strip()]
    object_match = re.search(r"(\{.*\}|\[.*\])", line, flags=re.S)
    if object_match is not None:
        candidates.append(object_match.group(1))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            payload = payload[0]
        if not isinstance(payload, dict):
            continue
        columns = [str(key).strip() for key in payload.keys() if str(key).strip()]
        if len(columns) >= 2 and columns_look_plausible(columns):
            return columns
    return None


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
        if explicit_tokens:
            base_score += 0.5
        language_suffix = _extract_suffix_from_code_fence_language(line)
        implicit_line_tokens = _extract_suffix_tokens_from_line(line)
        if language_suffix and not explicit_tokens:
            implicit_tokens = [language_suffix]
        else:
            implicit_tokens = (
                [token for token in implicit_line_tokens if token in _ARCHIVE_SUBMISSION_SUFFIXES]
                if explicit_tokens
                else implicit_line_tokens
            )
        if language_suffix and not explicit_tokens:
            base_score += 0.5

        merged_tokens: list[str] = []
        for token in [*explicit_tokens, *implicit_tokens]:
            if token not in merged_tokens:
                merged_tokens.append(token)
        if any(token in _ARCHIVE_SUBMISSION_SUFFIXES for token in merged_tokens) and "contain" in line_lower:
            merged_tokens = [token for token in merged_tokens if token in _ARCHIVE_SUBMISSION_SUFFIXES]

        for token in merged_tokens:
            if token in {".json", ".txt"} and _line_looks_like_non_artifact_suffix_context(line_lower):
                continue
            if (
                token != language_suffix
                and token in _NON_TABULAR_SINGLE_FILE_SUFFIXES
                and not _line_looks_like_submission_artifact_context(line_lower)
            ):
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
    suffixes = drop_shadowed_submission_suffixes(
        [suffix for suffix, _ in ranked if suffix in _KNOWN_SUBMISSION_SUFFIXES]
    )
    return suffixes or None


def _infer_artifact_shape(
    *,
    markdown: str,
    columns: list[str] | None,
    expected_suffixes: list[str] | None,
) -> tuple[str | None, str | None]:
    lowered = markdown.lower()
    suffixes = expected_suffixes or []
    archive_suffixes = [suffix for suffix in suffixes if suffix in _ARCHIVE_SUBMISSION_SUFFIXES]
    if archive_suffixes:
        container = archive_container(archive_suffixes, default="zip") or "zip"
        if any(marker in lowered for marker in MODEL_BUNDLE_MARKERS):
            return ARTIFACT_CLASS_BUNDLE, container
        if any(marker in lowered for marker in MULTI_FILE_SUBMISSION_MARKERS):
            return ARTIFACT_CLASS_MULTI_FILE_ZIP, container
        if ".zip" in archive_suffixes:
            return ARTIFACT_CLASS_MULTI_FILE_ZIP, "zip"
        return ARTIFACT_CLASS_SINGLE_FILE, container
    html_suffixes = [suffix for suffix in suffixes if _base_submission_suffix(suffix) in {".html", ".htm"}]
    if html_suffixes and not columns:
        return ARTIFACT_CLASS_SINGLE_FILE, "file"
    non_tabular_suffixes = [suffix for suffix in suffixes if suffix not in _TABULAR_SUBMISSION_SUFFIXES]
    if non_tabular_suffixes:
        return ARTIFACT_CLASS_SINGLE_FILE, "file"
    if columns:
        return ARTIFACT_CLASS_TABULAR, "file"
    if _looks_like_model_file_submission(suffixes=suffixes, text=lowered):
        return ARTIFACT_CLASS_SINGLE_FILE, "file"
    if any(marker in lowered for marker in MODEL_BUNDLE_MARKERS):
        return ARTIFACT_CLASS_BUNDLE, "zip"
    if suffixes:
        filtered = [suffix for suffix in suffixes if not _suffix_is_noise(suffix=suffix, text=lowered)]
        if filtered and all(suffix in _TABULAR_SUBMISSION_SUFFIXES for suffix in filtered):
            if any(_is_tabular_file_submission_suffix(suffix) for suffix in filtered):
                return ARTIFACT_CLASS_TABULAR, "file"
            base_suffixes = {_base_submission_suffix(suffix) for suffix in filtered}
            if any(suffix in {".json", ".txt"} for suffix in base_suffixes):
                return ARTIFACT_CLASS_SINGLE_FILE, "file"
    return ARTIFACT_CLASS_UNKNOWN, None


def _looks_like_model_file_submission(*, suffixes: list[str], text: str) -> bool:
    if not any(suffix in _AMBIGUOUS_MODEL_FILE_SUFFIXES for suffix in suffixes):
        return False
    return bool(re.search(r"\b(?:model|models|weights?|checkpoint|estimator|booster|pipeline|predictor)\b", text))


def _suffix_is_noise(*, suffix: str, text: str) -> bool:
    if _base_submission_suffix(suffix) not in {".json", ".txt"}:
        return False
    return any(marker in text for marker in JSON_TEXT_NOISE_CONTEXT_MARKERS)


def _is_tabular_file_submission_suffix(suffix: str) -> bool:
    if suffix not in _TABULAR_SUBMISSION_SUFFIXES:
        return False
    return _base_submission_suffix(suffix) not in {".json", ".txt"}


def _line_looks_like_non_artifact_suffix_context(line_lower: str) -> bool:
    return any(marker in line_lower for marker in JSON_TEXT_NOISE_CONTEXT_MARKERS)


def _line_looks_like_submission_artifact_context(line_lower: str) -> bool:
    if _SUBMISSION_ACTION_RE.search(line_lower):
        return True
    if _SUBMISSION_REQUIREMENT_RE.search(line_lower):
        return True
    if _SUBMISSION_FILE_RE.search(line_lower):
        return True
    return "scoring" in line_lower


def _extract_suffixes_from_line(line: str) -> list[str]:
    """Collect explicit extension mentions like '.csv' or '.zip' from a line."""
    suffixes: list[str] = []
    for match in _SUBMISSION_ARCHIVE_EXTENSION_RE.finditer(line):
        normalized = _normalize_suffix(f".{match.group(1).lower()}")
        if normalized is not None:
            suffixes.append(normalized)
    for match in _SUBMISSION_EXTENSION_RE.finditer(line):
        suffix = match.group("suffix").lower()
        normalized = _normalize_suffix(suffix)
        if normalized is not None:
            if normalized in SQLITE_TABULAR_SUFFIXES and _suffix_reference_looks_like_input_sample(
                line=line,
                start=match.start("suffix"),
            ):
                continue
            suffixes.append(normalized)
    return suffixes


def _suffix_reference_looks_like_input_sample(*, line: str, start: int) -> bool:
    prefix = line[:start].rstrip("`'\" )]")
    return bool(re.search(r"(?:^|[^A-Za-z0-9])sample[_-]?submission$", prefix, re.I))


def _extract_suffix_tokens_from_line(line: str) -> list[str]:
    """Collect implicit format tokens like 'zip file' or 'tab-separated' from a line."""
    suffixes: list[str] = []
    for pattern, suffix in SUBMISSION_TOKEN_PATTERNS:
        if pattern.search(line):
            suffixes.append(suffix)
    if ".jsonl" in suffixes and ".json" in suffixes:
        suffixes = [suffix for suffix in suffixes if suffix != ".json"]
    if ".txt" in suffixes and any(suffix in DOCUMENT_SUFFIXES for suffix in suffixes):
        suffixes = [suffix for suffix in suffixes if suffix != ".txt"]
    compression_suffix = _extract_compression_suffix_from_line(line)
    if compression_suffix is not None:
        compressed = [
            f"{suffix}{compression_suffix}" for suffix in suffixes if suffix in COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES
        ]
        if compressed:
            return compressed
    return suffixes


def _extract_compression_suffix_from_line(line: str) -> str | None:
    for pattern, suffix in COMPRESSION_TOKEN_PATTERNS:
        if pattern.search(line):
            return suffix
    return None


def _extract_suffix_from_code_fence_language(line: str) -> str | None:
    """Map fenced-code language markers to likely submission suffixes."""
    match = _SUBMISSION_CODE_FENCE_LANG_RE.match(line)
    if not match:
        return None
    lang = match.group("lang").strip().lower()
    return CODE_FENCE_LANG_TO_SUFFIX.get(lang)


def _normalize_suffix(suffix: str) -> str | None:
    """Normalize suffix aliases to the canonical extension set."""
    value = suffix.strip().lower()
    if value in _KNOWN_SUBMISSION_SUFFIXES:
        return value
    return None


def _base_submission_suffix(suffix: str) -> str:
    return strip_compression_suffix(suffix)


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
