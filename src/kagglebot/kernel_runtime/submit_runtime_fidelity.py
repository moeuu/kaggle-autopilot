"""Pure-standard-library runtime evidence for code-submission notebooks.

This module is copied into freshly packaged gateway/inference kernels.  Keep it
cheap and import-order neutral: in particular, do not import pandas, numpy,
torch, transformers, or any other optional/model dependency here.
"""

from __future__ import annotations

import atexit
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

EXPECTED_FILE_NAME = "submit_fidelity_expected.json"
RUNTIME_FILE_NAME = "submit_fidelity_runtime.json"

_PACKAGE_FINGERPRINT_VERSION = b"kagglebot-submit-package-v1\0"
_SOURCE_SUFFIXES = frozenset({".ipynb", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"})
_SOURCE_NAMES = frozenset({"kernel-metadata.json"})
_IGNORED_DIR_NAMES = frozenset({".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"})
_MAX_SOURCE_FILES = 2_000
_MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
_MAX_INPUT_FILES = 500
_MAX_INPUT_METADATA_BYTES = 128 * 1024
_MAX_OUTPUT_FILES = 500
_MAX_ERROR_BYTES = 2 * 1024 * 1024
_MAX_TABULAR_COLUMNS = 256
_MAX_UNIQUE_VALUES_PER_COLUMN = 1_024
_MAX_IDENTIFIER_HASHES = 2_000_000
_FALLBACK_FLAG_KEYS = (
    "dummy_submission",
    "fallback_only",
    "fallback_submission",
    "placeholder_submission",
    "prediction_fallback_used",
    "submission_fallback_used",
)
_MODEL_SOURCE_KEYS = (
    "active_model_source",
    "loaded_model_source",
    "model_source",
    "model_sources",
)
_REFERENCE_SOURCE_KEYS = (
    "active_reference_source",
    "loaded_reference_source",
    "reference_source",
    "reference_sources",
)
_ARTIFACT_HASH_KEYS = (
    "active_model_sha256",
    "artifact_hashes",
    "loaded_artifact_hashes",
    "model_artifact_sha256",
    "reference_artifact_sha256",
)
_ERROR_NAME_RE = re.compile(
    r"(?:^kernel_error(?:[-_][^.]+)?\.txt$|^stderr\.txt$|^error(?:[-_][^.]+)?\.(?:log|txt)$|"
    r"^.*(?:exception|traceback).*\.(?:log|txt)$)",
    flags=re.IGNORECASE,
)
_OUTPUT_NAME_RE = re.compile(r"(?:^submission(?:[-_.].*)?$|submission)", flags=re.IGNORECASE)
_ID_NAME_RE = re.compile(r"(?:^id$|^.*[_-]id$|^id[_-].*$|identifier|row[_-]?id)", flags=re.IGNORECASE)
_SECRET_TEXT_RE = re.compile(r"(?i)\b(api[_ -]?key|authorization|bearer|password|secret|token)\b(\s*[:=]\s*)([^\s,;]+)")
_BEARER_TEXT_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+={0,2}")
_TRACEBACK_FILE_RE = re.compile(r'(File\s+")[^"]+[/\\]([^/\\"]+)(")')
_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9:])/(?:[^\s,;'\"()]+)")
_WINDOWS_PATH_RE = re.compile(r"(?i)\b[A-Z]:\\(?:[^\s,;'\"()]+)")

_installed = False
_unhandled_exception: dict[str, object] | None = None
_install_state: dict[str, Path] = {}


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_contract_digest(path: Path) -> str:
    """Hash the exact expected contract bytes staged in the package."""
    return sha256_file(path)


def package_source_fingerprint(package_root: Path) -> str:
    """Hash bounded executable/package metadata without reading weights or data.

    The expected contract is excluded to avoid a self-referential digest.  It is
    separately hashed verbatim and the outer kernel source fingerprint includes
    the completed contract file.
    """
    root = package_root.resolve()
    digest = hashlib.sha256()
    digest.update(_PACKAGE_FINGERPRINT_VERSION)
    count = 0
    for path in _bounded_files(root, max_files=_MAX_SOURCE_FILES):
        relative = path.relative_to(root).as_posix()
        if relative == EXPECTED_FILE_NAME or path.name == RUNTIME_FILE_NAME:
            continue
        if path.suffix.lower() not in _SOURCE_SUFFIXES and path.name not in _SOURCE_NAMES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_SOURCE_FILE_BYTES and path.suffix.lower() not in {".ipynb", ".py"}:
            digest.update(b"bounded-metadata\0")
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(f"\0{size}\0".encode("ascii"))
            continue
        count += 1
        digest.update(b"file\0")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0end\0")
    digest.update(f"count\0{count}".encode("ascii"))
    return digest.hexdigest()


def install(
    *,
    package_root: str | Path | None = None,
    output_root: str | Path | None = None,
    input_root: str | Path | None = None,
) -> None:
    """Install a chained exception hook and an atexit recorder exactly once."""
    global _installed
    if _installed:
        return
    _installed = True
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    output = (
        Path(output_root)
        if output_root is not None
        else Path(os.environ.get("KAGGLEBOT_WORKING_DIR", "/kaggle/working"))
    )
    inputs = (
        Path(input_root) if input_root is not None else Path(os.environ.get("KAGGLEBOT_INPUT_ROOT", "/kaggle/input"))
    )
    _install_state.update(package_root=root, output_root=output, input_root=inputs)

    previous_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def chained_hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        global _unhandled_exception
        _unhandled_exception = {
            "type": getattr(exc_type, "__name__", str(exc_type)),
            "message": _sanitize_error_text(str(exc))[:4_000],
            "traceback": _sanitize_error_text("".join(traceback.format_exception(exc_type, exc, tb)))[-30_000:],
        }
        previous_hook(exc_type, exc, tb)

    sys.excepthook = chained_hook

    def chained_thread_hook(args: threading.ExceptHookArgs) -> None:
        global _unhandled_exception
        _unhandled_exception = {
            "type": getattr(args.exc_type, "__name__", str(args.exc_type)),
            "message": _sanitize_error_text(str(args.exc_value))[:4_000],
            "thread": str(getattr(args.thread, "name", ""))[:500],
            "traceback": _sanitize_error_text(
                "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
            )[-30_000:],
        }
        previous_thread_hook(args)

    threading.excepthook = chained_thread_hook
    atexit.register(_record_installed_runtime)


def record_runtime_fidelity(
    *,
    package_root: str | Path,
    output_root: str | Path,
    input_root: str | Path,
    unhandled_exception: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build and persist the completed runtime attestation."""
    package = Path(package_root)
    output = Path(output_root)
    inputs = Path(input_root)
    expected_path = package / EXPECTED_FILE_NAME
    expected = _load_json_object(expected_path)
    expected_digest = _safe_sha256(expected_path)
    expected_output = _nested_text(expected, "output", "filename")
    metrics_path = _find_named_file(output, "metrics.json")
    metrics = _load_json_object(metrics_path) if metrics_path is not None else None
    candidates = _output_candidates(output, expected_output=expected_output)
    selected = [item for item in candidates if item.get("filename") == expected_output]
    selected_record = selected[0] if len(selected) == 1 else None
    selected_path = output / str(selected_record["relative_path"]) if selected_record is not None else None
    metrics_sha = _safe_sha256(metrics_path)
    error_evidence = _error_evidence(output, unhandled_exception=unhandled_exception)
    report: dict[str, object] = {
        "schema_version": 1,
        "expected_contract": {
            "relative_path": EXPECTED_FILE_NAME,
            "sha256": expected_digest,
            "loaded": expected is not None,
        },
        "package": {
            "source_sha256": _safe_package_fingerprint(package),
            "source_file_count_limit": _MAX_SOURCE_FILES,
        },
        "kernel": {
            "contract_kernel_id": _nested_text(expected, "kernel", "id"),
            "environment_kernel_id": _allowlisted_env("KAGGLE_KERNEL_ID"),
            "run_id": _text(expected.get("run_id")) if expected else "",
            "iteration": expected.get("iteration") if expected else None,
            "machine": {
                "system": platform.system(),
                "release": platform.release(),
                "architecture": platform.machine(),
                "cpu_count": os.cpu_count(),
            },
        },
        "accelerator": {
            "requested": _allowlisted_env("KAGGLEBOT_FIDELITY_REQUESTED_ACCELERATOR")
            or _nested_text(expected, "accelerator", "requested"),
            "executed": _allowlisted_env("KAGGLEBOT_FIDELITY_EXECUTED_ACCELERATOR")
            or _nested_text(expected, "accelerator", "executed"),
            "machine_shape": _allowlisted_env("KAGGLEBOT_FIDELITY_MACHINE_SHAPE")
            or _nested_text(expected, "accelerator", "machine_shape"),
            "capacity_fallback_used": _env_truthy("KAGGLEBOT_FIDELITY_CAPACITY_FALLBACK_USED"),
            "gpu_visible_devices": _bounded_env_value("CUDA_VISIBLE_DEVICES"),
        },
        "input_inventory": _input_inventory(inputs),
        "runtime_metrics": _runtime_metrics_evidence(metrics, metrics_path=metrics_path, metrics_sha=metrics_sha),
        "outputs": {
            "candidate_count": len(candidates),
            "candidates": candidates,
            "selected_candidate_count": len(selected),
            "selected": selected_record,
            "metrics_relative_path": _relative_or_none(metrics_path, output),
            "metrics_sha256": metrics_sha,
        },
        "tabular_prediction": _tabular_stats(selected_path) if selected_path is not None else None,
        "errors": error_evidence,
    }
    output.mkdir(parents=True, exist_ok=True)
    runtime_path = output / RUNTIME_FILE_NAME
    _atomic_write_json(runtime_path, report)
    return report


def _record_installed_runtime() -> None:
    try:
        record_runtime_fidelity(
            package_root=_install_state["package_root"],
            output_root=_install_state["output_root"],
            input_root=_install_state["input_root"],
            unhandled_exception=_unhandled_exception,
        )
    except Exception as exc:  # noqa: BLE001 - audit failure must not alter kernel exit behavior
        try:
            output = _install_state.get("output_root")
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                _atomic_write_json(
                    output / RUNTIME_FILE_NAME,
                    {
                        "schema_version": 1,
                        "recorder_failure": {
                            "type": type(exc).__name__,
                            "message": _sanitize_error_text(str(exc))[:4_000],
                        },
                    },
                )
        except Exception:
            pass


def _input_inventory(root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    test_like: list[str] = []
    roots: set[str] = set()
    metadata_bytes = 0
    truncated = False
    if root.is_dir():
        for path in _bounded_files(root, max_files=_MAX_INPUT_FILES + 1):
            if len(files) >= _MAX_INPUT_FILES:
                truncated = True
                break
            try:
                relative = path.relative_to(root).as_posix()
                size = path.stat().st_size
            except (OSError, ValueError):
                continue
            entry = {"relative_path": relative, "size": size}
            encoded_size = len(canonical_json_bytes(entry))
            if metadata_bytes + encoded_size > _MAX_INPUT_METADATA_BYTES:
                truncated = True
                break
            metadata_bytes += encoded_size
            files.append(entry)
            if relative:
                roots.add(relative.split("/", 1)[0])
            lowered = relative.lower()
            if "test" in lowered or "sample_submission" in lowered or "answer_template" in lowered:
                test_like.append(relative)
    attached_roots = sorted(root_name for root_name in roots if _looks_like_asset_root(root_name))
    tabular_contract_paths = sorted(
        (item for item in test_like if Path(item).suffix.lower() in {".csv", ".tsv"}),
        key=_input_contract_priority,
    )[:3]
    tabular_contracts = [
        contract
        for relative in tabular_contract_paths
        if (contract := _tabular_identifier_contract(root / relative, relative_path=relative)) is not None
    ]
    return {
        "root_present": root.is_dir(),
        "file_count": len(files),
        "files": files,
        "truncated": truncated,
        "metadata_bytes": metadata_bytes,
        "test_like_inputs": test_like[:100],
        "attached_roots": sorted(roots)[:200],
        "model_or_reference_roots": attached_roots[:100],
        "tabular_identifier_contracts": tabular_contracts,
    }


def _runtime_metrics_evidence(
    metrics: dict[str, object] | None,
    *,
    metrics_path: Path | None,
    metrics_sha: str | None,
) -> dict[str, object]:
    payload = metrics or {}
    fallback_flags = {key: _truthy(payload.get(key)) for key in _FALLBACK_FLAG_KEYS if key in payload}
    return {
        "present": metrics is not None,
        "relative_path": metrics_path.name if metrics_path is not None else None,
        "sha256": metrics_sha,
        "pipeline": _first_text(payload, ("chosen_pipeline", "selected_pipeline", "pipeline")),
        "emitted_pipeline": _first_text(
            payload,
            ("emitted_pipeline", "inference_pipeline", "runtime_pipeline", "submission_pipeline"),
        ),
        "metric": _first_text(payload, ("metric", "metric_name")),
        "direction": _text(payload.get("direction")),
        "score_source": _text(payload.get("score_source")),
        "model_sources": _collect_values(payload, _MODEL_SOURCE_KEYS),
        "reference_sources": _collect_values(payload, _REFERENCE_SOURCE_KEYS),
        "reference_path_used": payload.get("reference_path_used") is True,
        "artifact_hashes": _collect_values(payload, _ARTIFACT_HASH_KEYS),
        "fallback_flags": fallback_flags,
        "prediction_source_distribution": _bounded_json_value(
            payload.get("test_prediction_distribution") or payload.get("prediction_source_distribution")
        ),
        "reported_output_filename": _first_text(
            payload,
            ("expected_output_file", "submission_filename", "submission_output_file"),
        ),
        "reported_output_sha256": _first_text(
            payload,
            ("submission_sha256", "submission_output_sha256"),
        ).lower(),
        "reported_row_count": _first_int(
            payload,
            ("emitted_row_count", "submission_row_count", "submission_rows"),
        ),
    }


def _output_candidates(root: Path, *, expected_output: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if not root.is_dir():
        return candidates
    for path in _bounded_files(root, max_files=_MAX_OUTPUT_FILES):
        relative = _relative_or_none(path, root)
        if relative is None or path.name in {EXPECTED_FILE_NAME, RUNTIME_FILE_NAME, "metrics.json"}:
            continue
        if _ERROR_NAME_RE.search(path.name):
            continue
        if path.name != expected_output and not _OUTPUT_NAME_RE.search(path.name):
            continue
        try:
            candidates.append(
                {
                    "relative_path": relative,
                    "filename": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except OSError:
            continue
    return candidates


def _error_evidence(root: Path, *, unhandled_exception: dict[str, object] | None) -> dict[str, object]:
    transcripts: list[dict[str, object]] = []
    traceback_count = 0
    if root.is_dir():
        for path in _bounded_files(root, max_files=_MAX_OUTPUT_FILES):
            if not _ERROR_NAME_RE.search(path.name):
                continue
            try:
                relative = path.relative_to(root).as_posix()
                size = path.stat().st_size
                raw = path.read_bytes()[:_MAX_ERROR_BYTES]
            except (OSError, ValueError):
                continue
            text = raw.decode("utf-8", errors="replace")
            current_tracebacks = text.lower().count("traceback")
            traceback_count += current_tracebacks
            transcripts.append(
                {
                    "relative_path": relative,
                    "size": size,
                    "sha256": _safe_sha256(path),
                    "traceback_count": current_tracebacks,
                    "nonempty": bool(text.strip()),
                    "truncated": size > len(raw),
                }
            )
    if unhandled_exception is not None:
        traceback_count += str(unhandled_exception.get("traceback") or "").lower().count("traceback") or 1
    return {
        "unhandled_exception": unhandled_exception,
        "transcripts": transcripts,
        "traceback_count": traceback_count,
    }


def _tabular_stats(path: Path) -> dict[str, object] | None:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return None
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            columns = next(reader)
            columns = [_bounded_column_name(column) for column in columns[:_MAX_TABULAR_COLUMNS]]
            id_indices = [index for index, name in enumerate(columns) if _ID_NAME_RE.search(name.strip())]
            states = [_new_column_state() for _ in columns]
            identifier_state = _new_identifier_state()
            row_count = 0
            for raw_row in reader:
                row_count += 1
                row = raw_row[: len(columns)] + [""] * max(0, len(columns) - len(raw_row))
                for index, value in enumerate(row):
                    _update_column_state(states[index], value)
                _update_identifier_state(identifier_state, row, id_indices)
    except (OSError, csv.Error, StopIteration) as exc:
        return {"applicable": True, "parse_error": f"{type(exc).__name__}: {exc}"}

    identifier = _identifier_stats(states, columns, id_indices, identifier_state)
    column_stats = [_finish_column_state(state, name=columns[index]) for index, state in enumerate(states)]
    prediction_indices = [index for index in range(len(columns)) if index not in id_indices]
    return {
        "applicable": True,
        "schema": {"columns": columns, "delimiter": delimiter, "column_count": len(columns)},
        "row_count": row_count,
        "null_count": sum(int(item["null_count"]) for item in column_stats),
        "nonfinite_count": sum(int(item["nonfinite_count"]) for item in column_stats),
        "identifier": identifier,
        "prediction_columns": [columns[index] for index in prediction_indices],
        "prediction_null_count": sum(int(column_stats[index]["null_count"]) for index in prediction_indices),
        "prediction_nonfinite_count": sum(int(column_stats[index]["nonfinite_count"]) for index in prediction_indices),
        "numeric_dispersion": [column_stats[index] for index in prediction_indices],
    }


def _new_column_state() -> dict[str, object]:
    return {
        "null_count": 0,
        "nonfinite_count": 0,
        "numeric_count": 0,
        "mean": 0.0,
        "m2": 0.0,
        "minimum": None,
        "maximum": None,
        "unique": set(),
        "unique_truncated": False,
        "order_digest": hashlib.sha256(),
    }


def _update_column_state(state: dict[str, object], raw_value: str) -> None:
    value = str(raw_value)
    order_digest = state["order_digest"]
    assert hasattr(order_digest, "update")
    order_digest.update(value.encode("utf-8", errors="surrogateescape"))
    order_digest.update(b"\0")
    stripped = value.strip()
    if not stripped or stripped.lower() in {"nan", "na", "null", "none"}:
        state["null_count"] = int(state["null_count"]) + 1
    unique = state["unique"]
    assert isinstance(unique, set)
    if len(unique) < _MAX_UNIQUE_VALUES_PER_COLUMN:
        unique.add(value)
    else:
        state["unique_truncated"] = True
    try:
        numeric = float(stripped)
    except ValueError:
        return
    if not math.isfinite(numeric):
        state["nonfinite_count"] = int(state["nonfinite_count"]) + 1
        return
    count = int(state["numeric_count"]) + 1
    mean = float(state["mean"])
    delta = numeric - mean
    mean += delta / count
    state["m2"] = float(state["m2"]) + delta * (numeric - mean)
    state["mean"] = mean
    state["numeric_count"] = count
    minimum = state["minimum"]
    maximum = state["maximum"]
    state["minimum"] = numeric if minimum is None else min(float(minimum), numeric)
    state["maximum"] = numeric if maximum is None else max(float(maximum), numeric)


def _finish_column_state(state: dict[str, object], *, name: str) -> dict[str, object]:
    count = int(state["numeric_count"])
    order_digest = state["order_digest"]
    assert hasattr(order_digest, "hexdigest")
    unique = state["unique"]
    assert isinstance(unique, set)
    return {
        "column": name,
        "null_count": int(state["null_count"]),
        "nonfinite_count": int(state["nonfinite_count"]),
        "numeric_count": count,
        "minimum": state["minimum"],
        "maximum": state["maximum"],
        "mean": float(state["mean"]) if count else None,
        "stddev": math.sqrt(float(state["m2"]) / count) if count else None,
        "unique_count": len(unique),
        "unique_truncated": bool(state["unique_truncated"]),
        "order_sha256": order_digest.hexdigest(),
    }


def _identifier_stats(
    states: list[dict[str, object]],
    columns: list[str],
    indices: list[int],
    identifier_state: dict[str, object],
) -> dict[str, object] | None:
    if not indices:
        return None
    rows = [_finish_column_state(states[index], name=columns[index]) for index in indices]
    return {
        "columns": [columns[index] for index in indices],
        "unique": not bool(identifier_state["duplicate"]) and not bool(identifier_state["truncated"]),
        "unique_check_truncated": bool(identifier_state["truncated"]),
        "null_count": sum(int(item["null_count"]) for item in rows),
        "order_digests": {str(item["column"]): item["order_sha256"] for item in rows},
        "composite_order_sha256": _identifier_order_hexdigest(identifier_state),
    }


def _new_identifier_state() -> dict[str, object]:
    return {
        "hashes": set(),
        "duplicate": False,
        "truncated": False,
        "order_digest": hashlib.sha256(),
    }


def _update_identifier_state(state: dict[str, object], row: list[str], indices: list[int]) -> None:
    if not indices:
        return
    raw = b"\x1f".join(row[index].encode("utf-8", errors="surrogateescape") for index in indices)
    order_digest = state["order_digest"]
    assert hasattr(order_digest, "update")
    order_digest.update(raw)
    order_digest.update(b"\0")
    hashes = state["hashes"]
    assert isinstance(hashes, set)
    value_hash = hashlib.sha256(raw).digest()[:16]
    if value_hash in hashes:
        state["duplicate"] = True
    elif len(hashes) < _MAX_IDENTIFIER_HASHES:
        hashes.add(value_hash)
    else:
        state["truncated"] = True


def _identifier_order_hexdigest(state: dict[str, object]) -> str:
    order_digest = state["order_digest"]
    assert hasattr(order_digest, "hexdigest")
    return order_digest.hexdigest()


def _tabular_identifier_contract(path: Path, *, relative_path: str) -> dict[str, object] | None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            columns = [_bounded_column_name(column) for column in next(reader)[:_MAX_TABULAR_COLUMNS]]
            id_indices = [index for index, name in enumerate(columns) if _ID_NAME_RE.search(name.strip())]
            if not id_indices and columns:
                id_indices = [0]
            states = [_new_column_state() for _ in id_indices]
            identifier_state = _new_identifier_state()
            row_count = 0
            for raw_row in reader:
                row_count += 1
                row = raw_row[: len(columns)] + [""] * max(0, len(columns) - len(raw_row))
                for state_index, column_index in enumerate(id_indices):
                    _update_column_state(states[state_index], row[column_index])
                _update_identifier_state(identifier_state, row, id_indices)
    except (OSError, csv.Error, StopIteration):
        return None
    identifier_columns = [columns[index] for index in id_indices]
    finished = [_finish_column_state(state, name=identifier_columns[index]) for index, state in enumerate(states)]
    return {
        "relative_path": relative_path,
        "schema": {"columns": columns, "delimiter": delimiter, "column_count": len(columns)},
        "row_count": row_count,
        "identifier": {
            "columns": identifier_columns,
            "unique": not bool(identifier_state["duplicate"]) and not bool(identifier_state["truncated"]),
            "unique_check_truncated": bool(identifier_state["truncated"]),
            "null_count": sum(int(item["null_count"]) for item in finished),
            "order_digests": {str(item["column"]): item["order_sha256"] for item in finished},
            "composite_order_sha256": _identifier_order_hexdigest(identifier_state),
        },
    }


def _input_contract_priority(relative_path: str) -> tuple[int, str]:
    name = Path(relative_path).name.lower()
    if "sample_submission" in name or "answer_template" in name:
        return (0, relative_path)
    return (1, relative_path)


def _bounded_column_name(value: object) -> str:
    return str(value)[:500]


def _bounded_json_value(value: object, *, max_bytes: int = 50_000) -> object:
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError):
        return None
    if len(encoded) <= max_bytes:
        return value
    return {
        "truncated": True,
        "byte_count": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _bounded_files(root: Path, *, max_files: int) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(name for name in dir_names if name not in _IGNORED_DIR_NAMES)
        for file_name in sorted(file_names):
            path = Path(current_root) / file_name
            if path.is_symlink() or not path.is_file():
                continue
            files.append(path)
            if len(files) >= max_files:
                return files
    return files


def _collect_values(payload: dict[str, object], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            values.extend(f"{name}:{item}"[:2_000] for name, item in sorted(value.items()) if _text(item))
        elif isinstance(value, (list, tuple, set)):
            values.extend(_text(item)[:2_000] for item in value if _text(item))
        elif _text(value):
            values.append(_text(value)[:2_000])
    return list(dict.fromkeys(values))[:200]


def _find_named_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    for path in _bounded_files(root, max_files=_MAX_OUTPUT_FILES):
        if path.name == name:
            return path
    return None


def _load_json_object(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None


def _safe_package_fingerprint(root: Path) -> str | None:
    try:
        return package_source_fingerprint(root)
    except OSError:
        return None


def _relative_or_none(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _allowlisted_env(name: str) -> str:
    return str(os.environ.get(name, "")).strip()[:500]


def _bounded_env_value(name: str) -> str | None:
    value = _allowlisted_env(name)
    return value or None


def _nested_text(payload: dict[str, object] | None, parent: str, key: str) -> str:
    value = payload.get(parent) if payload else None
    return _text(value.get(key)) if isinstance(value, dict) else ""


def _first_text(payload: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _text(payload.get(key))
        if value:
            return value[:2_000]
    return ""


def _first_int(payload: dict[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        try:
            value = int(payload.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _env_truthy(name: str) -> bool:
    return _truthy(os.environ.get(name))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "on"}


def _looks_like_asset_root(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ("checkpoint", "model", "reference", "weight"))


def _text(value: object) -> str:
    return str(value or "").strip()


def _sanitize_error_text(value: str) -> str:
    redacted = _SECRET_TEXT_RE.sub(r"\1\2<redacted>", value)
    redacted = _BEARER_TEXT_RE.sub("Bearer <redacted>", redacted)
    redacted = _TRACEBACK_FILE_RE.sub(r"\1<runtime>:\2\3", redacted)
    redacted = _WINDOWS_PATH_RE.sub("<runtime-path>", redacted)
    return _ABSOLUTE_PATH_RE.sub("<runtime-path>", redacted)
