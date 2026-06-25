from __future__ import annotations

from kagglebot.kernel_progress import (
    extract_catboost_fallback_reason_from_line,
    extract_pipeline_done_from_line,
    extract_pipeline_start_from_line,
    extract_pipeline_suite_from_line,
    extract_train_model_start_from_line,
    extract_training_stage_from_line,
    resolve_fold_current,
    resolve_seed_current,
)


def test_extract_training_stage_from_inline_and_path_lines() -> None:
    inline = "[kernel] yolo_ensemble_wbf_geometry: seed=2024 fold=0 imgsz=768 epochs=250"
    assert extract_training_stage_from_line(inline) == ("yolo_ensemble_wbf_geometry", 2024, 0)

    path_line = "/tmp/runs/yolo_ensemble_wbf_geometry_seed2024_fold1/weights/best.pt saved"
    assert extract_training_stage_from_line(path_line) == ("yolo_ensemble_wbf_geometry", 2024, 1)

    assert extract_training_stage_from_line("unrelated line") is None


def test_extract_pipeline_progress_from_line() -> None:
    assert extract_pipeline_start_from_line("[kernel] Running pipeline: tri_blend_stack") == "tri_blend_stack"
    assert extract_pipeline_start_from_line("Training pipeline: tri_blend_stack") == "tri_blend_stack"
    assert extract_pipeline_start_from_line("pipeline: missing verb") is None
    assert (
        extract_pipeline_done_from_line("[kernel] Pipeline tri_blend_stack: CV=0.125 method=weighted_mean_log")
        == "tri_blend_stack"
    )
    assert extract_pipeline_suite_from_line("[kernel] Suite: tabular_full") == "tabular_full"


def test_extract_model_and_fallback_progress_from_line() -> None:
    assert extract_train_model_start_from_line("[kernel] train start: model=lgbm") == "lgbm"
    assert extract_train_model_start_from_line("train done: model=lgbm") is None
    assert (
        extract_catboost_fallback_reason_from_line("[kernel] CatBoost GPU failed; retrying on CPU: CUDA unavailable")
        == "CUDA unavailable"
    )


def test_progress_position_helpers() -> None:
    assert resolve_seed_current(seed=2024, expected_seeds=[42, 2024, 777]) == 2
    assert resolve_seed_current(seed=999, expected_seeds=[42, 2024, 777]) is None
    assert resolve_fold_current(fold_raw=0, expected_folds=5, zero_based=True) == 1
    assert resolve_fold_current(fold_raw=2, expected_folds=5, zero_based=True) == 3
    assert resolve_fold_current(fold_raw=2, expected_folds=5, zero_based=False) == 2
    assert resolve_fold_current(fold_raw=8, expected_folds=5, zero_based=False) is None
