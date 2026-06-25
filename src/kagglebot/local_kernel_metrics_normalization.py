from __future__ import annotations

from pathlib import Path

from kagglebot.json_utils import load_json_object, write_json_object

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


def normalize_kernel_score_source(value: object) -> str:
    return str(value or "").strip().lower()


def looks_like_urban_flood_flat_full_root(data_dir: Path) -> bool:
    if not data_dir.exists() or not data_dir.is_dir():
        return False
    names = {child.name for child in data_dir.iterdir() if child.is_file()}
    return URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES.issubset(names)


def normalize_local_kernel_metrics(
    *,
    slug: str,
    data_dir: Path,
    metrics_path: Path | None,
    score_source: str,
) -> Path | None:
    if slug.strip().lower() != "urban-flood-modelling":
        return metrics_path
    if metrics_path is None or not metrics_path.exists():
        return metrics_path
    if not looks_like_urban_flood_flat_full_root(data_dir):
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
    payload["data_root_layout"] = "flat_full"
    payload["metrics_normalized_by"] = "kernel_runner.local_full_data_guard"
    try:
        write_json_object(metrics_path, payload)
    except OSError:
        return metrics_path
    return metrics_path
