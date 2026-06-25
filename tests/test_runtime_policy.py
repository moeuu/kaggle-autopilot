from __future__ import annotations

from kagglebot.runtime_policy import (
    is_heavy_deep_learning_modality,
    is_local_gpu_compute,
    local_gpu_time_budget_limit_min,
)


def test_is_local_gpu_compute_normalizes_case_and_whitespace() -> None:
    assert is_local_gpu_compute(" Local_GPU ")
    assert not is_local_gpu_compute("kaggle_gpu")


def test_is_heavy_deep_learning_modality_includes_rna_structure() -> None:
    for modality in ("image", "video", "audio", "text", "rna_structure"):
        assert is_heavy_deep_learning_modality(modality)
    assert not is_heavy_deep_learning_modality("tabular")


def test_local_gpu_time_budget_limit_min_parses_env_with_floor() -> None:
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "15") == 60
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "120") == 120
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "0") is None
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: "bad") is None
    assert local_gpu_time_budget_limit_min(getenv=lambda _key: None) is None
