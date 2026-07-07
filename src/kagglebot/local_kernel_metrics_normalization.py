from __future__ import annotations

from pathlib import Path

from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.submission_sample_discovery import (
    is_tabular_data_path,
    sample_name_score,
    tabular_stem,
    tabular_suffix,
)

TRUSTED_KERNEL_SCORE_SOURCES = frozenset({"cv", "holdout", "consensus"})
URBAN_FLOOD_SAMPLEISH_SCORE_SOURCES = frozenset(
    {
        "sample_diagnostic",
        "sample_mode_smoke_cv",
        "sample",
        "fallback",
    }
)
URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES = frozenset(
    {
        "1d_nodes_dynamic_all.csv",
        "2d_nodes_dynamic_all.csv",
        "test_1d_nodes_dynamic_all.csv",
        "test_2d_nodes_dynamic_all.csv",
        "timesteps.csv",
        "test_timesteps.csv",
        "sample_submission.csv",
    }
)
FULL_DATASET_LAYOUT_REQUIRED_FILES = {
    "flat_full": URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES,
}


def normalize_kernel_score_source(value: object) -> str:
    return str(value or "").strip().lower()


def detect_full_dataset_layout(data_dir: Path) -> str | None:
    if not data_dir.exists() or not data_dir.is_dir():
        return None
    names = {child.name for child in data_dir.iterdir() if child.is_file()}
    for layout, required_files in FULL_DATASET_LAYOUT_REQUIRED_FILES.items():
        if _required_layout_files_present(data_dir=data_dir, names=names, required_files=required_files):
            return layout
    return None


def _required_layout_files_present(*, data_dir: Path, names: set[str], required_files: frozenset[str]) -> bool:
    for required_file in required_files:
        if required_file in names:
            continue
        if required_file == "sample_submission.csv":
            if not _has_top_level_sample_submission(data_dir):
                return False
            continue
        if not _has_top_level_tabular_stem(data_dir, required_file):
            return False
    return True


def _has_top_level_tabular_stem(data_dir: Path, required_file: str) -> bool:
    required_path = Path(required_file)
    required_suffix = tabular_suffix(required_path)
    if not required_suffix:
        return False
    required_stem = tabular_stem(required_path).lower()
    try:
        children = list(data_dir.iterdir())
    except OSError:
        return False
    for child in children:
        if not child.is_file():
            continue
        if not is_tabular_data_path(child):
            continue
        if tabular_stem(child).lower() == required_stem:
            return True
    return False


def _has_top_level_sample_submission(data_dir: Path) -> bool:
    for child in data_dir.iterdir():
        if not child.is_file():
            continue
        if not is_tabular_data_path(child):
            continue
        if sample_name_score(child) >= 2:
            return True
    return False


def looks_like_urban_flood_flat_full_root(data_dir: Path) -> bool:
    return detect_full_dataset_layout(data_dir) == "flat_full"


def normalize_local_kernel_metrics(
    *,
    slug: str,
    data_dir: Path,
    metrics_path: Path | None,
    score_source: str,
) -> Path | None:
    del slug
    if metrics_path is None or not metrics_path.exists():
        return metrics_path
    data_root_layout = detect_full_dataset_layout(data_dir)
    if data_root_layout is None:
        return metrics_path

    payload = load_json_object(metrics_path)
    if payload is None:
        return metrics_path

    normalized_payload_source = normalize_kernel_score_source(payload.get("score_source"))
    requested_source = normalize_kernel_score_source(score_source)
    if requested_source not in TRUSTED_KERNEL_SCORE_SOURCES:
        requested_source = "cv"

    if normalized_payload_source not in URBAN_FLOOD_SAMPLEISH_SCORE_SOURCES and bool(
        payload.get("full_dataset_resolved")
    ):
        return metrics_path

    payload["score_source"] = requested_source
    payload["dataset_kind"] = "full"
    payload["dataset_mode"] = "full"
    payload["full_dataset_resolved"] = True
    payload["data_root_layout"] = data_root_layout
    payload["metrics_normalized_by"] = "kernel_runner.local_full_data_guard"
    try:
        write_json_object(metrics_path, payload)
    except OSError:
        return metrics_path
    return metrics_path
