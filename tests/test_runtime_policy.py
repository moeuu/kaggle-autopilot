from __future__ import annotations

from kagglebot.runtime_policy import (
    DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN,
    is_heavy_deep_learning_modality,
    is_local_gpu_compute,
    local_gpu_time_budget_limit_min,
)


def test_is_local_gpu_compute_normalizes_case_and_whitespace() -> None:
    assert is_local_gpu_compute(" Local_GPU ")
    assert not is_local_gpu_compute("kaggle_gpu")


def test_is_heavy_deep_learning_modality_includes_asset_and_rna_modalities() -> None:
    for modality in (
        "image",
        "video",
        "audio",
        "text",
        "document",
        "medical_imaging",
        "array",
        "point_cloud",
        "3d",
        "point_cloud_3d",
        "geospatial",
        "bio",
        "sequence",
        "structure",
        "rna",
        "rna_structure",
        "graph",
        "signal",
        "annotation",
        "model_artifact",
        "artifact",
    ):
        assert is_heavy_deep_learning_modality(modality)
    assert is_heavy_deep_learning_modality("medical-imaging")
    assert is_heavy_deep_learning_modality("point cloud")
    assert is_heavy_deep_learning_modality("point-cloud-3D")
    assert is_heavy_deep_learning_modality("model-artifact")
    assert not is_heavy_deep_learning_modality("tabular")


def test_local_gpu_time_budget_limit_min_parses_env_with_floor() -> None:
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "15") == 60
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "120") == 120
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "0") is None
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "bad") is None
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: None) == DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN
