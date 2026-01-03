from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubmissionFormatHint:
    columns: list[str] | None
    delimiter: str | None


_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", re.S)


def extract_submission_section(markdown: str) -> str | None:
    text = markdown.strip()
    if not text:
        return None
    section = _extract_section(text, keywords=("submission format",))
    if section:
        return section
    return _extract_section(text, keywords=("submission",))


def _extract_section(markdown: str, *, keywords: tuple[str, ...]) -> str | None:
    lines = markdown.splitlines()
    start_idx = None
    start_level = None
    for idx, line in enumerate(lines):
        match = _HEADING_RE.match(line.strip())
        if not match:
            continue
        title = match.group(2).strip().lower()
        if any(keyword in title for keyword in keywords):
            start_idx = idx
            start_level = len(match.group(1))
            break
    if start_idx is None or start_level is None:
        return None
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
    return SubmissionFormatHint(columns=columns, delimiter=delimiter)


def load_submission_format_hint(path: Path) -> SubmissionFormatHint | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    hint = parse_submission_format(text)
    if hint.columns or hint.delimiter:
        return hint
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
