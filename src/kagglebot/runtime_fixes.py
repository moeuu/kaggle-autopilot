from __future__ import annotations

import ast
import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.json_utils import load_json_array, load_json_object_or_empty, write_json_array, write_json_object
from kagglebot.submission.guard import normalize_error_text

COLUMN_MAP_FILENAME = "column_map.json"
COLUMN_FILL_FILENAME = "column_fill.json"
OBJECT_COERCE_FILENAME = "object_coerce.json"
DEVICE_COERCE_FILENAME = "device_coerce.json"
BLOCKED_MODULES_FILENAME = "blocked_modules.json"
MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
MISSING_COLUMNS_RE = re.compile(r"missing columns?:\s*\[([^\]]+)\]", re.IGNORECASE)
MISSING_COLUMNS_KEYERROR_RE = re.compile(
    r"KeyError:\s*[\"']?\[([^\]]+)\]\s*not in index[\"']?",
    re.IGNORECASE,
)
MISSING_COLUMNS_FILE_RE = re.compile(
    r"([A-Za-z0-9_.-]+\.(?:csv|tsv|txt|parquet|json|jsonl))\s+missing columns",
    re.IGNORECASE,
)
OBJECT_DTYPE_RE = re.compile(r"numpy\.object_", re.IGNORECASE)
DEVICE_MISMATCH_RE = re.compile(
    r"Expected all tensors to be on the same device|found at least two devices",
    re.IGNORECASE,
)
COLUMN_ERROR_PATTERNS = (
    "could not resolve column",
    "unable to locate session",
    "missing columns",
    "not in index",
    "are in the [columns]",
)


@dataclass(frozen=True)
class LightweightRuntimeFixResult:
    artifact_name: str
    reason: str
    note: str


def error_strategy_skip_reason(*, stage: str, error_text: str) -> str | None:
    """Return a deterministic reason to skip GPT strategy analysis, if any."""
    normalized_stage = str(stage or "").strip().lower()
    lowered = normalize_error_text(error_text or "", max_chars=8000).lower()
    if not lowered:
        return None

    cross_stage_patterns = (
        (
            "competition metric mismatch",
            "strict competition metric mismatch escalation is deterministic",
        ),
    )
    for needle, reason in cross_stage_patterns:
        if needle in lowered:
            return reason

    common_patterns = (
        (
            "kernel source validation failed",
            "deterministic kernel source validation failure",
        ),
        (
            "do not reference metrics.json output",
            "missing metrics.json output contract is deterministic",
        ),
        (
            "do not reference submission.csv output",
            "missing submission.csv output contract is deterministic",
        ),
        (
            "unexpected keyword argument 'evaluation_strategy'",
            "known transformers eval_strategy API mismatch",
        ),
        (
            "modulenotfounderror: no module named",
            "deterministic missing module error",
        ),
        (
            "keyerror:",
            "deterministic dataframe key/column error",
        ),
        (
            "not in index",
            "deterministic dataframe column mismatch",
        ),
        (
            "missing columns",
            "deterministic missing-column error",
        ),
        (
            "data directory not found:",
            "deterministic local data path resolution failure",
        ),
        (
            "unable to resolve competition data root",
            "deterministic competition data path resolution failure",
        ),
    )
    if normalized_stage != "submit_autofix":
        for needle, reason in common_patterns:
            if needle in lowered:
                return reason

    if normalized_stage == "submit_autofix":
        submit_patterns = (
            (
                "cannot use internet access in this competition",
                "competition internet policy violation is deterministic",
            ),
            (
                "disable internet in the notebook editor",
                "competition internet policy violation is deterministic",
            ),
            (
                "submission file must be named submission.csv",
                "submission filename contract violation is deterministic",
            ),
        )
        for needle, reason in submit_patterns:
            if needle in lowered:
                return reason
    return None


def is_non_autofixable_runtime_error(error: Exception) -> bool:
    text = str(error).strip().lower()
    if not text:
        return False
    return "requires kernel.py" in text or "kernel-first training" in text


def write_lightweight_autofix_note(*, note_path: Path, artifact_name: str, reason: str) -> str:
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note = (
        f"autofix_note: {artifact_name} created for {reason}.\nautofix will retry without modifying kernel sources.\n"
    )
    note_path.write_text(note, encoding="utf-8")
    return note


def maybe_write_column_fill(config: object, error_text: str) -> bool:
    raw_error = error_text or ""
    file_name: str | None = None
    match = MISSING_COLUMNS_RE.search(raw_error)
    if match:
        missing_columns = parse_missing_columns(match.group(1))
        file_match = MISSING_COLUMNS_FILE_RE.search(raw_error)
        file_name = file_match.group(1) if file_match else None
    else:
        keyerror_match = MISSING_COLUMNS_KEYERROR_RE.search(raw_error)
        if not keyerror_match:
            return False
        missing_columns = parse_missing_columns(keyerror_match.group(1))
    if not missing_columns:
        return False

    deduped_missing: list[str] = []
    seen_missing: set[str] = set()
    for column in missing_columns:
        normalized = str(column).strip()
        if not normalized or normalized in seen_missing:
            continue
        seen_missing.add(normalized)
        deduped_missing.append(normalized)
    if not deduped_missing:
        return False

    context_dir = _context_dir(config)
    fill_path = context_dir / COLUMN_FILL_FILENAME
    payload = load_json_object_or_empty(fill_path)

    changed = not fill_path.exists()
    files_payload = payload.get("files")
    files: dict[str, list[str]]
    if isinstance(files_payload, dict):
        files = {}
        for key, value in files_payload.items():
            if not isinstance(key, str) or not isinstance(value, list):
                continue
            cleaned: list[str] = []
            seen_cols: set[str] = set()
            for col in value:
                col_name = str(col).strip()
                if not col_name or col_name in seen_cols:
                    continue
                seen_cols.add(col_name)
                cleaned.append(col_name)
            files[key] = cleaned
    else:
        files = {}
        if files_payload is not None:
            changed = True

    if file_name:
        existing = files.get(file_name, [])
        merged = list(existing)
        for col in deduped_missing:
            if col not in merged:
                merged.append(col)
        if merged != existing:
            files[file_name] = merged
            changed = True
    else:
        missing_payload = payload.get("missing_columns")
        if isinstance(missing_payload, list):
            existing_missing: list[str] = []
            seen_cols: set[str] = set()
            for col in missing_payload:
                col_name = str(col).strip()
                if not col_name or col_name in seen_cols:
                    continue
                seen_cols.add(col_name)
                existing_missing.append(col_name)
        else:
            existing_missing = []
            if missing_payload is not None:
                changed = True
        merged_missing = list(existing_missing)
        for col in deduped_missing:
            if col not in merged_missing:
                merged_missing.append(col)
        if merged_missing != existing_missing:
            payload["missing_columns"] = merged_missing
            changed = True

    if not changed:
        return False

    payload["files"] = files
    payload.setdefault("source", "autofix")
    payload.setdefault("created_at", datetime.now(UTC).isoformat())
    payload["updated_at"] = datetime.now(UTC).isoformat()
    if "missing_columns" not in payload:
        payload["missing_columns"] = []
    write_json_object(fill_path, payload)
    return True


def maybe_write_object_coerce(config: object, error_text: str) -> bool:
    if not OBJECT_DTYPE_RE.search(error_text or ""):
        return False
    coerce_path = _context_dir(config) / OBJECT_COERCE_FILENAME
    if coerce_path.exists():
        return False
    payload = {
        "source": "autofix",
        "created_at": datetime.now(UTC).isoformat(),
        "enabled": True,
        "reason": "numpy.object_ conversion error",
    }
    write_json_object(coerce_path, payload)
    return True


def maybe_write_device_coerce(config: object, error_text: str) -> bool:
    if not DEVICE_MISMATCH_RE.search(error_text or ""):
        return False
    coerce_path = _context_dir(config) / DEVICE_COERCE_FILENAME
    if coerce_path.exists():
        return False
    payload = {
        "source": "autofix",
        "created_at": datetime.now(UTC).isoformat(),
        "enabled": True,
        "prefer": "cuda",
        "reason": "torch device mismatch error",
    }
    write_json_object(coerce_path, payload)
    return True


def parse_missing_columns(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    try:
        parsed = ast.literal_eval(f"[{text}]")
    except Exception:
        parsed = [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    if isinstance(parsed, list | tuple):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def maybe_write_column_map(config: object, error_text: str) -> bool:
    lowered = error_text.lower()
    if not any(pattern in lowered for pattern in COLUMN_ERROR_PATTERNS):
        return False
    paths = _paths(config)
    map_path = paths.context_dir / COLUMN_MAP_FILENAME
    if map_path.exists():
        return False
    columns_by_file = scan_tabular_headers(paths.data_dir)
    if not columns_by_file:
        return False
    candidate_groups = extract_candidate_groups(error_text)
    if not candidate_groups:
        return False
    mapping = infer_column_mapping(columns_by_file, candidate_groups)
    if not mapping:
        return False
    payload = {
        "mapping": mapping,
        "source": "autofix",
        "created_at": datetime.now(UTC).isoformat(),
        "candidates": candidate_groups,
        "files": columns_by_file,
    }
    write_json_object(map_path, payload)
    return True


LIGHTWEIGHT_RUNTIME_FIX_ACTIONS: tuple[tuple[str, str, Callable[[object, str], bool]], ...] = (
    (
        COLUMN_FILL_FILENAME,
        "missing column error",
        maybe_write_column_fill,
    ),
    (
        OBJECT_COERCE_FILENAME,
        "numpy.object_ conversion error",
        maybe_write_object_coerce,
    ),
    (
        DEVICE_COERCE_FILENAME,
        "torch device mismatch error",
        maybe_write_device_coerce,
    ),
    (
        COLUMN_MAP_FILENAME,
        "column alias mismatch",
        maybe_write_column_map,
    ),
)


def apply_lightweight_runtime_fix(
    *,
    config: object,
    error_text: str,
    note_path: Path,
    actions: tuple[tuple[str, str, Callable[[object, str], bool]], ...] = LIGHTWEIGHT_RUNTIME_FIX_ACTIONS,
) -> LightweightRuntimeFixResult | None:
    for artifact_name, reason, action in actions:
        try:
            changed = bool(action(config, error_text))
        except Exception:
            changed = False
        if not changed:
            continue
        note = write_lightweight_autofix_note(
            note_path=note_path,
            artifact_name=artifact_name,
            reason=reason,
        )
        return LightweightRuntimeFixResult(artifact_name=artifact_name, reason=reason, note=note)
    return None


def scan_tabular_headers(data_dir: Path) -> dict[str, list[str]]:
    columns: dict[str, list[str]] = {}
    if not data_dir.exists():
        return columns
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".tsv"}:
            continue
        header = read_header(path)
        if not header:
            continue
        try:
            rel = str(path.relative_to(data_dir))
        except ValueError:
            rel = str(path)
        columns[rel] = header
    return columns


def read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline()
    except OSError:
        return []
    if not line:
        return []
    sep = "\t" if "\t" in line and path.suffix.lower() == ".tsv" else ","
    try:
        row = next(csv.reader([line], delimiter=sep))
    except Exception:
        return []
    return [col.strip().strip('"').strip("'") for col in row if col.strip()]


def extract_candidate_groups(error_text: str) -> list[list[str]]:
    groups: list[list[str]] = []
    list_match = re.findall(r"candidates:\s*\[([^\]]+)\]", error_text, flags=re.IGNORECASE)
    for match in list_match:
        try:
            items = [item.strip().strip("'\"") for item in match.split(",") if item.strip()]
        except Exception:
            items = []
        if items:
            groups.append(items)
    slash_match = re.findall(r"([A-Za-z0-9_]+)\s*/\s*([A-Za-z0-9_]+)", error_text)
    for left, right in slash_match:
        groups.append([left, right])
    lowered = error_text.lower()
    if "session" in lowered or "visit" in lowered:
        groups.append(["session_id", "visit_id"])
    if "product" in lowered or "item" in lowered:
        groups.append(["product_id", "item_id", "sku"])
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        norm = tuple(group)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(group)
    return deduped


def infer_column_mapping(columns_by_file: dict[str, list[str]], groups: list[list[str]]) -> dict[str, str]:
    all_columns = []
    for cols in columns_by_file.values():
        all_columns.extend(cols)
    mapping: dict[str, str] = {}
    normalized = {normalize_column(col): col for col in all_columns}
    for group in groups:
        normalized_group = normalize_group_tokens(group)
        if not normalized_group:
            continue
        canonical = normalized_group[0]
        match = match_column(normalized_group, normalized, all_columns)
        if match and match not in mapping:
            mapping[match] = canonical
    return mapping


def match_column(group: list[str], normalized: dict[str, str], all_columns: list[str]) -> str | None:
    for cand in group:
        norm = normalize_column(cand)
        if norm in normalized:
            return normalized[norm]
    keywords = keywords_from_group(group)
    if not keywords:
        return None
    best: tuple[int, str] | None = None
    for col in all_columns:
        score = keyword_score(col, keywords)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, col)
    return best[1] if best else None


def normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_group_tokens(group: list[object]) -> list[str]:
    normalized: list[str] = []
    for cand in group:
        if cand is None:
            continue
        text = cand if isinstance(cand, str) else str(cand)
        stripped = text.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def keywords_from_group(group: list[str]) -> set[str]:
    keywords: set[str] = set()
    for cand in group:
        for token in re.split(r"[^a-zA-Z0-9]+", cand):
            if token:
                keywords.add(token.lower())
    return keywords


def keyword_score(column: str, keywords: set[str]) -> int:
    lowered = column.lower()
    return sum(1 for key in keywords if key in lowered)


def extract_missing_module(error_text: str) -> str | None:
    match = MISSING_MODULE_RE.search(error_text or "")
    if not match:
        return None
    return match.group(1)


def load_blocked_modules(context_dir: Path) -> list[str]:
    payload = load_json_array(context_dir / BLOCKED_MODULES_FILENAME)
    return [str(item) for item in payload if item] if payload is not None else []


def save_blocked_modules(context_dir: Path, modules: list[str]) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    path = context_dir / BLOCKED_MODULES_FILENAME
    if modules:
        write_json_array(path, [str(module) for module in modules])
        return
    if path.exists():
        path.unlink()


def record_blocked_module(context_dir: Path, module: str) -> list[str]:
    context_dir.mkdir(parents=True, exist_ok=True)
    existing = load_blocked_modules(context_dir)
    if module not in existing:
        existing.append(module)
        save_blocked_modules(context_dir, existing)
    return existing


def _paths(config: object) -> object:
    return getattr(config, "paths")


def _context_dir(config: object) -> Path:
    return _paths(config).context_dir
