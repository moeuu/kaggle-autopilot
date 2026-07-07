from __future__ import annotations

import pytest

from kagglebot.submit_failure_policy import (
    SUBMIT_FAILURE_REPAIR_TARGET_MANUAL,
    SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM,
    SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT,
    SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE,
    classify_submit_failure_repair,
    normalize_loaded_submit_failure_context,
    should_retry_ambiguous_notebook_submit_error,
    should_use_notebook_submit_fallback,
    submit_error_requires_file_fix,
)


def test_classify_submit_failure_repair_treats_submission_limit_as_manual() -> None:
    decision = classify_submit_failure_repair(
        reason="bad_request",
        error_kind="permanent",
        detail="You have reached the maximum number of submissions for this competition.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_MANUAL
    assert decision.repairable is False
    assert "submission limit" in decision.manual_next_step.lower()


def test_classify_submit_failure_repair_treats_daily_allowance_as_manual() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_limit",
        error_kind="permanent",
        detail="Submission not allowed: Your team has used its daily Submission allowance (10) today.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_MANUAL
    assert decision.repairable is False
    assert "submission limit" in decision.manual_next_step.lower()


def test_classify_submit_failure_repair_detects_scoring_file_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_complete_no_score",
        error_kind="validation",
        detail="Kaggle scoring error inferred: invalid submission file with row count mismatch.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True
    assert decision.manual_next_step == ""


def test_classify_submit_failure_repair_detects_compressed_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.jsonl.gz has invalid JSON lines.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_zstd_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.csv.zst has invalid columns.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_archive_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.tar.gz is missing required files.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_external_archive_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.rar is missing required files.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_excel_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.xlsx has invalid workbook contents.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


@pytest.mark.parametrize("suffix", [".tar.xz", ".tar.zst"])
def test_classify_submit_failure_repair_detects_compressed_tar_submission_artifact_issue(suffix: str) -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail=f"Kaggle scoring error: submission{suffix} is missing required files.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_plain_tar_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.tar is missing required files.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_array_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.npy has an invalid output shape.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_generic_array_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: predictions.npy has an invalid output shape.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_generic_model_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: results.onnx failed model artifact validation.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_compound_model_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.safetensors.index.json failed model artifact validation.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_medical_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.svs is not a valid whole-slide image.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_compound_wsi_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.ome.tif is not a valid tiled image.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_microscopy_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.czi could not be parsed as a microscopy image.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


@pytest.mark.parametrize(
    "filename",
    [
        "submission.avif",
        "submission.heic",
        "submission.heif",
        "results.aiff",
        "submission.opus",
        "submission.m4v",
        "predictions.wmv",
    ],
)
def test_classify_submit_failure_repair_detects_modern_media_submission_artifact_issue(filename: str) -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail=f"Kaggle scoring error: {filename} could not be parsed.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_document_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.pdf could not be parsed.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_geospatial_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.geojson has invalid geometry.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_bio_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.pdb has invalid atom coordinates.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_graph_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.graphml has invalid graph structure.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_point_cloud_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.e57 has invalid point cloud coordinates.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_scientific_array_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.nc has invalid NetCDF dimensions.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_array_store_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.h5ad has invalid AnnData contents.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_model_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.tflite failed model artifact validation.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_detects_compressed_pickle_submission_artifact_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle scoring error: submission.pkl.zst could not be decoded.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


@pytest.mark.parametrize(
    "filename",
    [
        "submission.nrrd.zst",
        "submission.dicom.bz2",
        "submission.fastq.bz2",
        "results.ply.gz",
        "predictions.graphml.gz",
        "answers.geojson.zst",
    ],
)
def test_classify_submit_failure_repair_detects_registry_backed_compound_artifact_issue(filename: str) -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail=f"Kaggle scoring error: {filename} could not be parsed.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True


def test_classify_submit_failure_repair_does_not_treat_sample_only_detail_as_file_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Observed sample_submission.csv.gz while inspecting input data.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM
    assert decision.repairable is True


def test_classify_submit_failure_repair_does_not_treat_input_generic_prediction_as_file_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Observed train_predictions.npy while inspecting input data.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM
    assert decision.repairable is True


def test_classify_submit_failure_repair_routes_notebook_mode_errors() -> None:
    decision = classify_submit_failure_repair(
        reason="notebook_only_submission_required",
        error_kind="permanent",
        detail="Only accepts submissions from notebooks.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE
    assert decision.repairable is True


def test_classify_submit_failure_repair_routes_notebook_submit_argument_errors() -> None:
    decision = classify_submit_failure_repair(
        reason="notebook_submit_argument_missing",
        error_kind="permanent",
        detail="Code competition submissions require both the output file name and the version label.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE
    assert decision.repairable is True


def test_classify_submit_failure_repair_routes_polling_without_file_hint_to_platform() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle submission status: error",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM
    assert decision.repairable is True


def test_submit_error_requires_file_fix_for_local_validation() -> None:
    assert submit_error_requires_file_fix(
        reason="local_submission_validation_failed",
        error_kind="validation",
        detail="",
    )


def test_notebook_submit_fallback_requires_clear_hint() -> None:
    assert should_use_notebook_submit_fallback(
        reason="notebook_submit_argument_missing",
        stdout="",
        stderr="",
    )
    assert should_use_notebook_submit_fallback(
        reason="bad_request",
        stdout="",
        stderr="Code competition submissions require both the output file name and the version label.",
    )
    assert not should_use_notebook_submit_fallback(reason="bad_request", stdout="", stderr="generic 400")


def test_ambiguous_notebook_retry_requires_clear_hint() -> None:
    assert should_retry_ambiguous_notebook_submit_error(
        reason="ambiguous_notebook_bad_request",
        stdout="",
        stderr="kernel must be specified as <owner>/<notebook>",
    )
    assert not should_retry_ambiguous_notebook_submit_error(
        reason="ambiguous_notebook_bad_request",
        stdout="",
        stderr="generic bad request",
    )


def test_normalize_loaded_submit_failure_context_backfills_manual_blocker() -> None:
    payload = {
        "reason": "ambiguous_notebook_bad_request",
        "repair_target": "submit_mode_or_kernel",
        "repairable": True,
        "stderr_tail": "400 Client Error: Bad Request",
    }

    normalized = normalize_loaded_submit_failure_context(payload)

    assert normalized["repair_target"] == SUBMIT_FAILURE_REPAIR_TARGET_MANUAL
    assert normalized["repairable"] is False
    assert "submit-notebook 400" in str(normalized["manual_next_step"])
