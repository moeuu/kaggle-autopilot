from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.json_utils import load_json_object
from kagglebot.paths import CompetitionPaths
from kagglebot.scalar_utils import parse_finite_float

CODE_REFERENCE_IMPL_MARKER_PREFIX = "# KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED:"
CODE_SCORE_RE = re.compile(r"(?<!\d)(0\.\d{3,6})(?!\d)")


@dataclass(frozen=True)
class CodeReferenceNotebook:
    kernel_id: str
    title: str
    source_file: str | None = None
    local_dir: str | None = None
    summary: str = ""


def extract_score_from_text(text: str) -> float | None:
    for match in CODE_SCORE_RE.finditer(text):
        value = parse_finite_float(match.group(1))
        if value is None:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return None


def extract_code_reference_score_from_index(path: Path) -> tuple[float | None, str]:
    payload = load_json_object(path)
    if payload is None:
        if not path.exists():
            return None, "missing_code_index"
        return None, "invalid_code_index"
    notebooks = payload.get("notebooks")
    if not isinstance(notebooks, list) or not notebooks:
        return None, "empty_code_index"
    selected = _select_reference_notebook_row(payload=payload, id_key="required_reference_kernel_id")
    if selected is None:
        return None, "empty_code_index"

    kernel_id = str(selected.get("kernel_id") or "top_entry").strip() or "top_entry"
    score = parse_finite_float(selected.get("score"))
    if score is not None:
        return score, f"code_index:{kernel_id}"

    title_score = extract_score_from_text(str(selected.get("title") or ""))
    if title_score is not None:
        return title_score, f"code_title:{kernel_id}"
    return None, "code_index_without_numeric_score"


def extract_code_reference_score_from_markdown(path: Path) -> tuple[float | None, str]:
    if not path.exists():
        return None, "missing_code_md"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, "missing_code_md"
    if not text.strip():
        return None, "empty_code_md"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "notebook_score:" not in line.lower():
            continue
        score = extract_score_from_text(line)
        if score is not None:
            return score, "code_md:notebook_score"

    lowered = text.lower()
    required_start = lowered.find("required reference notebook")
    if required_start >= 0:
        required_section = text[required_start : required_start + 2200]
        score = extract_score_from_text(required_section)
        if score is not None:
            return score, "code_md:required_reference_section"

    top_snapshot = text[:3000]
    score = extract_score_from_text(top_snapshot)
    if score is not None:
        return score, "code_md:top_snapshot"
    return None, "code_md_without_numeric_score"


def extract_code_reference_score(paths: CompetitionPaths) -> tuple[float | None, str]:
    score, source = extract_code_reference_score_from_index(paths.code_notebooks_index_path)
    if score is not None:
        return score, source
    score, source = extract_code_reference_score_from_markdown(paths.code_md_path)
    if score is not None:
        return score, source
    if source and source != "code_md_without_numeric_score":
        return None, source
    return None, "unavailable"


def load_required_reference_notebook(paths: CompetitionPaths) -> CodeReferenceNotebook | None:
    return load_code_reference_notebook(paths, id_key="required_reference_kernel_id")


def load_ensemble_reference_notebook(paths: CompetitionPaths) -> CodeReferenceNotebook | None:
    return load_code_reference_notebook(paths, id_key="ensemble_reference_kernel_id")


def load_code_reference_notebook(paths: CompetitionPaths, *, id_key: str) -> CodeReferenceNotebook | None:
    payload = load_json_object(paths.code_notebooks_index_path)
    if payload is None:
        return None
    selected = _select_reference_notebook_row(payload=payload, id_key=id_key)
    if selected is None:
        return None
    kernel_id = str(selected.get("kernel_id") or "").strip()
    if not kernel_id:
        return None
    title = str(selected.get("title") or kernel_id).strip() or kernel_id
    source_file = _optional_string(selected.get("source_file"))
    local_dir = _optional_string(selected.get("local_dir"))
    summary = _optional_string(selected.get("summary")) or ""
    return CodeReferenceNotebook(
        kernel_id=kernel_id,
        title=title,
        source_file=source_file,
        local_dir=local_dir,
        summary=summary,
    )


def reference_requires_tabicl(reference: CodeReferenceNotebook) -> bool:
    text = " ".join([reference.kernel_id, reference.title, reference.summary]).lower()
    return "tabicl" in text


def code_reference_marker(reference: CodeReferenceNotebook) -> str:
    return f"{CODE_REFERENCE_IMPL_MARKER_PREFIX} {reference.kernel_id}"


def validate_code_reference_implementation(*, kernel_path: Path, reference: CodeReferenceNotebook) -> list[str]:
    if not kernel_path.exists():
        return ["kernel_source_missing"]
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    marker = code_reference_marker(reference).lower()
    issues: list[str] = []
    if marker not in lowered:
        issues.append("missing_code_reference_marker")
    if reference_requires_tabicl(reference) and "tabicl" not in lowered:
        issues.append("missing_tabicl_implementation_path")
    return issues


def _select_reference_notebook_row(*, payload: dict[str, object], id_key: str) -> dict[str, object] | None:
    notebooks = payload.get("notebooks")
    if not isinstance(notebooks, list) or not notebooks:
        return None
    required_id_raw = payload.get(id_key)
    required_id = str(required_id_raw).strip() if isinstance(required_id_raw, str) else ""
    if required_id:
        for row in notebooks:
            if not isinstance(row, dict):
                continue
            if str(row.get("kernel_id") or "").strip() == required_id:
                return row
    for row in notebooks:
        if isinstance(row, dict):
            return row
    return None


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
