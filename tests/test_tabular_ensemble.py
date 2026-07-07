from __future__ import annotations

import gzip
import io
import json

import numpy as np
import pandas as pd
import pytest
import zstandard as zstd

import kagglebot.kernel_runtime.tabular_ensemble as tabular_ensemble
from kagglebot.kernel_runtime.tabular_ensemble import (
    PipelineResult,
    build_prediction_correlation_summary,
    maybe_apply_pseudo_labels,
    resolve_component_models,
    safe_auc,
    train_catboost_model,
    train_xgb_model,
    write_fold_intermediate_artifacts,
    write_submission_manifest,
    write_table,
)
from kagglebot.solver.io import read_table as read_solver_table
from kagglebot.submission_output_naming import non_tabular_submission_output_suffixes
from kagglebot.submission_sample_discovery import SQLITE_TABULAR_SUFFIXES, TABULAR_SUBMISSION_SUFFIXES


def _pipeline_result(
    name: str,
    preds: list[float],
    *,
    kind: str = "single",
    blend_components: list[str] | None = None,
) -> PipelineResult:
    arr = np.asarray(preds, dtype=np.float64)
    metadata = {"kind": kind}
    if blend_components is not None:
        metadata["blend_components"] = blend_components
    return PipelineResult(
        name=name,
        oof_preds=arr,
        test_preds=arr,
        cv_score=0.9,
        fold_scores=[],
        feature_manifest={},
        metadata=metadata,
        test_predictions_by_fold={},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )


def test_resolve_component_models_prefers_blend_components() -> None:
    result = _pipeline_result("blend_top", [0.1, 0.9], kind="weighted_blend", blend_components=["xgb_a", "cb_b"])

    assert resolve_component_models(result) == ["xgb_a", "cb_b"]


def test_tabular_ensemble_output_suffixes_follow_shared_submission_suffixes(monkeypatch) -> None:
    assert tabular_ensemble._TABULAR_OUTPUT_SUFFIXES == TABULAR_SUBMISSION_SUFFIXES
    assert not (tabular_ensemble._TABULAR_OUTPUT_SUFFIXES & SQLITE_TABULAR_SUFFIXES)
    assert tabular_ensemble._REQUESTED_NON_TABULAR_OUTPUT_SUFFIXES == non_tabular_submission_output_suffixes()

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.ndjson.zst")
    assert tabular_ensemble.resolve_submission_filename() == "submission.ndjson.zst"
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.sqlite3")
    assert tabular_ensemble.resolve_submission_filename() == "submission.tabular.csv"
    assert tabular_ensemble.requested_non_tabular_submission_filename() == "submission.sqlite3"


def test_resolve_submission_filename_derives_default_from_sample_env(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_SUBMISSION_FILENAME", raising=False)
    monkeypatch.setenv("KAGGLEBOT_SAMPLE_SUBMISSION_PATH", "/kaggle/input/demo/sample_submission.jsonl.zst")
    assert tabular_ensemble.resolve_submission_filename() == "submission.jsonl.zst"

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.sqlite3")
    assert tabular_ensemble.resolve_submission_filename() == "submission.tabular.jsonl.zst"
    assert tabular_ensemble.requested_non_tabular_submission_filename() == "submission.sqlite3"


@pytest.mark.parametrize("configured_name", ["sample_submission.csv", "sample-submission.csv.gz", "metrics.json"])
def test_resolve_submission_filename_rejects_configured_template_or_reserved_names(
    monkeypatch: pytest.MonkeyPatch,
    configured_name: str,
) -> None:
    monkeypatch.setenv("KAGGLEBOT_SAMPLE_SUBMISSION_PATH", "/kaggle/input/demo/sample_submission.tsv.gz")
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", configured_name)

    assert tabular_ensemble.resolve_submission_filename() == "submission.tsv.gz"
    assert tabular_ensemble.requested_non_tabular_submission_filename() is None


@pytest.mark.parametrize(
    ("requested_name", "fallback_name"),
    [
        ("answers.nii.gz", "answers.tabular.csv"),
        ("predictions.sqlite", "predictions.tabular.csv"),
        ("submission.tar.gz", "submission.tabular.csv"),
    ],
)
def test_requested_non_tabular_submission_filename_is_manifest_only(
    monkeypatch, requested_name: str, fallback_name: str
) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", requested_name)

    assert tabular_ensemble.resolve_submission_filename() == fallback_name
    assert tabular_ensemble.requested_non_tabular_submission_filename() == requested_name


def test_requested_non_tabular_submission_filename_uses_sample_tabular_suffix(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "answers.nii.gz")
    monkeypatch.setenv("KAGGLEBOT_SAMPLE_SUBMISSION_PATH", "/kaggle/input/demo/sample_submission.tsv.zst")

    assert tabular_ensemble.resolve_submission_filename() == "answers.tabular.tsv.zst"
    assert tabular_ensemble.requested_non_tabular_submission_filename() == "answers.nii.gz"


def test_build_prediction_correlation_summary_uses_single_models_only() -> None:
    result_a = _pipeline_result("xgb_a", [0.1, 0.2, 0.8, 0.9])
    result_b = _pipeline_result("cb_b", [0.12, 0.22, 0.82, 0.88])
    blend = _pipeline_result(
        "blend_ab", [0.11, 0.21, 0.81, 0.89], kind="rank_blend", blend_components=["xgb_a", "cb_b"]
    )

    summary = build_prediction_correlation_summary([result_a, result_b, blend])

    assert summary["pair_count"] == 1
    assert summary["mean_abs_corr"] is not None
    assert summary["max_abs_corr"] is not None
    assert summary["min_abs_corr"] is not None


def test_safe_auc_returns_neutral_score_for_single_class_fold() -> None:
    assert safe_auc(np.array([1, 1], dtype=np.int8), np.array([0.2, 0.8], dtype=np.float64)) == 0.5


def test_build_prediction_range_defaults_to_probability_clip() -> None:
    prediction_range = tabular_ensemble.build_prediction_range(np.array([-5.0, 2.0], dtype=np.float64))

    assert prediction_range == [1e-6, 1 - 1e-6]


def test_build_prediction_range_preserves_explicit_regression_values() -> None:
    prediction_range = tabular_ensemble.build_prediction_range(
        np.array([-5.0, 20.0], dtype=np.float64),
        prediction_kind="regression",
    )

    assert prediction_range == [-5.0, 20.0]


def test_build_prediction_range_clips_structured_regression_values() -> None:
    count_range = tabular_ensemble.build_prediction_range(
        np.array([-5.0, 20.0], dtype=np.float64),
        prediction_kind="count_regression",
    )
    bounded_range = tabular_ensemble.build_prediction_range(
        np.array([-0.5, 1.5], dtype=np.float64),
        prediction_kind="regression",
        target_labels=pd.Series([0.0, 0.1, 0.4, 0.9, 1.0], name="conversion_rate"),
        target_col="conversion_rate",
    )

    assert count_range == [0.0, 20.0]
    assert bounded_range == [0.0, 1.0]


def test_validate_submission_preserves_explicit_regression_predictions() -> None:
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})

    submission = tabular_ensemble.validate_submission(
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
        preds=np.array([12.5, 30.0], dtype=np.float64),
        prediction_kind="regression",
    )

    assert submission["target"].tolist() == [12.5, 30.0]


def test_validate_submission_default_keeps_probability_clipping() -> None:
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})

    submission = tabular_ensemble.validate_submission(
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
        preds=np.array([-5.0, 2.0], dtype=np.float64),
    )

    assert submission["target"].between(1e-6, 1 - 1e-6).all()


def test_validate_submission_clips_explicit_count_regression_predictions() -> None:
    sample = pd.DataFrame({"id": [10, 20], "count": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})

    submission = tabular_ensemble.validate_submission(
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="count",
        preds=np.array([-3.0, 14.0], dtype=np.float64),
        prediction_kind="count_regression",
    )

    assert submission["count"].tolist() == [0.0, 14.0]


def test_validate_submission_clips_bounded_regression_from_target_labels() -> None:
    sample = pd.DataFrame({"id": [10, 20], "conversion_rate": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    labels = pd.Series([0.0, 0.1, 0.5, 0.9, 1.0], name="conversion_rate")

    submission = tabular_ensemble.validate_submission(
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="conversion_rate",
        preds=np.array([-0.5, 1.4], dtype=np.float64),
        prediction_kind="regression",
        target_labels=labels,
    )

    assert submission["conversion_rate"].tolist() == [0.0, 1.0]


def test_validate_submission_clips_positive_skew_regression_from_target_labels() -> None:
    sample = pd.DataFrame({"id": [10, 20], "SalePrice": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    labels = pd.Series([8000, 5000, 170, 160, 150, 140, 130, 120, 110, 100], name="SalePrice")

    submission = tabular_ensemble.validate_submission(
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="SalePrice",
        preds=np.array([-100.0, 4200.0], dtype=np.float64),
        prediction_kind="regression",
        target_labels=labels,
    )

    assert submission["SalePrice"].tolist() == [0.0, 4200.0]


def test_write_fold_intermediate_artifacts_writes_valid_submission_and_manifest(tmp_path) -> None:
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="model/a",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.25, 0.75], dtype=np.float64)},
        oof_predictions_by_fold={"fold_1": np.array([0.2], dtype=np.float64)},
        valid_indices_by_fold={"fold_1": np.array([0])},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
    )
    write_submission_manifest(
        output_dirs=[tmp_path],
        final_result=result,
        summary={"prediction_range": [0.25, 0.75]},
        fold_artifacts=records,
    )

    assert records[0]["status"] == "available"
    submission = pd.read_csv(tmp_path / "submission_model_a_fold1.csv")
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [10, 20]
    assert np.allclose(submission["target"], [0.25, 0.75])
    assert (tmp_path / "test_preds_model_a_fold1.npy").exists()
    assert (tmp_path / "oof_preds_model_a_fold1.npy").exists()
    assert (tmp_path / "candidate_model_a_fold1.json").exists()
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fold_artifacts"][0]["submission_path"] == "submission_model_a_fold1.csv"


def test_write_fold_intermediate_artifacts_respects_submission_filename_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.tsv")
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="model/a",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.25, 0.75], dtype=np.float64)},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
    )
    final_submission = tmp_path / "submission.tsv"
    pd.DataFrame({"id": [10, 20], "target": [0.25, 0.75]}).to_csv(final_submission, sep="\t", index=False)
    write_submission_manifest(
        output_dirs=[tmp_path],
        final_result=result,
        summary={"prediction_range": [0.25, 0.75]},
        fold_artifacts=records,
    )

    assert records[0]["submission_path"] == "submission_model_a_fold1.tsv"
    submission = pd.read_csv(tmp_path / "submission_model_a_fold1.tsv", sep="\t")
    assert submission["target"].tolist() == [0.25, 0.75]
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "submission.tsv"
    assert manifest["fold_artifacts"][0]["submission_path"] == "submission_model_a_fold1.tsv"


@pytest.mark.parametrize(
    ("requested_name", "fallback_name"),
    [
        ("answers.nii.gz", "answers.tabular.csv"),
        ("submission.zip", "submission.tabular.csv"),
    ],
)
def test_write_submission_manifest_records_requested_non_tabular_output_path(
    tmp_path, monkeypatch, requested_name: str, fallback_name: str
) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", requested_name)
    result = _pipeline_result("model/a", [0.25, 0.75])
    pd.DataFrame({"id": [10, 20], "target": [0.25, 0.75]}).to_csv(
        tmp_path / fallback_name,
        index=False,
    )

    write_submission_manifest(
        output_dirs=[tmp_path],
        final_result=result,
        summary={"prediction_range": [0.25, 0.75]},
        fold_artifacts=[],
    )

    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "tabular"
    assert manifest["submission_path"] == fallback_name
    assert manifest["requested_output_path"] == requested_name
    assert "tabular fallback" in manifest["note"]


def test_write_fold_intermediate_artifacts_respects_compressed_submission_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "custom_submission.csv.gz")
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="model/a",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.25, 0.75], dtype=np.float64)},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
    )
    final_submission = tmp_path / "custom_submission.csv.gz"
    pd.DataFrame({"id": [10, 20], "target": [0.25, 0.75]}).to_csv(final_submission, index=False)
    write_submission_manifest(
        output_dirs=[tmp_path],
        final_result=result,
        summary={"prediction_range": [0.25, 0.75]},
        fold_artifacts=records,
    )

    assert records[0]["submission_path"] == "submission_model_a_fold1.csv.gz"
    submission = pd.read_csv(tmp_path / "submission_model_a_fold1.csv.gz")
    assert submission["target"].tolist() == [0.25, 0.75]
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "custom_submission.csv.gz"
    assert manifest["fold_artifacts"][0]["submission_path"] == "submission_model_a_fold1.csv.gz"


def test_write_fold_intermediate_artifacts_respects_excel_submission_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "custom_submission.xlsx")
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="model/a",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.25, 0.75], dtype=np.float64)},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
    )
    final_submission = tmp_path / "custom_submission.xlsx"
    pd.DataFrame({"id": [10, 20], "target": [0.25, 0.75]}).to_excel(final_submission, index=False)
    write_submission_manifest(
        output_dirs=[tmp_path],
        final_result=result,
        summary={"prediction_range": [0.25, 0.75]},
        fold_artifacts=records,
    )

    assert records[0]["submission_path"] == "submission_model_a_fold1.xlsx"
    submission = pd.read_excel(tmp_path / "submission_model_a_fold1.xlsx")
    assert submission["target"].tolist() == [0.25, 0.75]
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "custom_submission.xlsx"
    assert manifest["fold_artifacts"][0]["submission_path"] == "submission_model_a_fold1.xlsx"


def test_write_fold_intermediate_artifacts_respects_html_submission_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "custom_submission.html")
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="model/a",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.25, 0.75], dtype=np.float64)},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
    )
    final_submission = tmp_path / "custom_submission.html"
    pd.DataFrame({"id": [10, 20], "target": [0.25, 0.75]}).to_html(final_submission, index=False)
    write_submission_manifest(
        output_dirs=[tmp_path],
        final_result=result,
        summary={"prediction_range": [0.25, 0.75]},
        fold_artifacts=records,
    )

    assert records[0]["submission_path"] == "submission_model_a_fold1.html"
    submission = pd.read_html(tmp_path / "submission_model_a_fold1.html")[0]
    assert submission["target"].tolist() == [0.25, 0.75]
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "custom_submission.html"
    assert manifest["fold_artifacts"][0]["submission_path"] == "submission_model_a_fold1.html"


@pytest.mark.parametrize(("suffix", "reader"), [(".orc", pd.read_orc), (".hdf5", pd.read_hdf)])
def test_write_fold_intermediate_artifacts_respects_binary_submission_suffix(
    tmp_path, monkeypatch, suffix, reader
) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", f"custom_submission{suffix}")
    sample = pd.DataFrame({"id": [10, 20], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="model/a",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.25, 0.75], dtype=np.float64)},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
    )

    assert records[0]["submission_path"] == f"submission_model_a_fold1{suffix}"
    submission = reader(tmp_path / f"submission_model_a_fold1{suffix}")
    assert submission["target"].tolist() == [0.25, 0.75]


def test_write_table_writes_compressed_jsonl(tmp_path) -> None:
    path = tmp_path / "submission.jsonl.gz"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    loaded = pd.read_json(path, lines=True)
    assert loaded.to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


def test_write_table_stabilizes_problematic_columns_without_mutating_input(tmp_path) -> None:
    path = tmp_path / "submission.tsv"
    frame = pd.DataFrame([[1, "-", 0.2, 0.8]], columns=["id", "", "target", "target"])

    write_table(frame, path)

    loaded = pd.read_csv(path, sep="\t")
    assert list(loaded.columns) == ["id", "column_2", "target", "target_1"]
    assert list(frame.columns) == ["id", "", "target", "target"]


def test_write_table_writes_zstd_csv(tmp_path) -> None:
    path = tmp_path / "submission.csv.zst"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    payload = zstd.ZstdDecompressor().decompress(path.read_bytes()).decode("utf-8")
    assert payload == "id,target\n1,0.2\n2,0.8\n"


def test_write_table_writes_compressed_tsv(tmp_path) -> None:
    path = tmp_path / "submission.tsv.gz"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert handle.read() == "id\ttarget\n1\t0.2\n2\t0.8\n"


def test_write_table_writes_txt_with_tab_default(tmp_path) -> None:
    path = tmp_path / "submission.txt"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    assert path.read_text(encoding="utf-8") == "id\ttarget\n1\t0.2\n2\t0.8\n"


def test_write_table_writes_compressed_psv(tmp_path) -> None:
    path = tmp_path / "submission.psv.gz"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert handle.read() == "id|target\n1|0.2\n2|0.8\n"


def test_write_table_writes_zstd_jsonl(tmp_path) -> None:
    path = tmp_path / "submission.jsonl.zst"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    payload = zstd.ZstdDecompressor().decompress(path.read_bytes()).decode("utf-8")
    assert payload == '{"id":1,"target":0.2}\n{"id":2,"target":0.8}\n'


@pytest.mark.parametrize("suffix", [".jsonlines.xz", ".ndjson.zst"])
def test_write_table_writes_json_lines_aliases(tmp_path, suffix: str) -> None:
    path = tmp_path / f"submission{suffix}"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    loaded = read_solver_table(path)
    assert loaded.to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


@pytest.mark.parametrize("suffix", [".html", ".html.zst"])
def test_write_table_writes_html(tmp_path, suffix: str) -> None:
    path = tmp_path / f"submission{suffix}"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    if suffix.endswith(".zst"):
        html = zstd.ZstdDecompressor().decompress(path.read_bytes()).decode("utf-8")
        loaded = pd.read_html(io.StringIO(html))[0]
    else:
        loaded = pd.read_html(path)[0]
    assert loaded.to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


@pytest.mark.parametrize("suffix", [".yaml", ".yaml.xz", ".yml.zst"])
def test_write_table_writes_yaml(tmp_path, suffix: str) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / f"submission{suffix}"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    loaded = read_solver_table(path)
    assert loaded.to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm", ".ods"])
def test_write_table_writes_excel(tmp_path, suffix: str) -> None:
    path = tmp_path / f"submission{suffix}"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    loaded = pd.read_excel(path)
    assert loaded.to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


def test_write_table_writes_orc_and_hdf5(tmp_path) -> None:
    frame = pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]})
    orc_path = tmp_path / "submission.orc"
    hdf_path = tmp_path / "submission.hdf5"

    write_table(frame, orc_path)
    write_table(frame, hdf_path)

    assert pd.read_orc(orc_path).to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}
    assert pd.read_hdf(hdf_path).to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


def test_write_table_writes_pickle_and_xml(tmp_path) -> None:
    frame = pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]})
    pickle_path = tmp_path / "submission.pkl.gz"
    xml_path = tmp_path / "submission.xml"

    write_table(frame, pickle_path)
    write_table(frame, xml_path)

    assert pd.read_pickle(pickle_path).to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}
    assert pd.read_xml(xml_path, parser="etree").to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


def test_write_table_writes_avro(tmp_path) -> None:
    path = tmp_path / "submission.avro"

    write_table(pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}), path)

    loaded = read_solver_table(path)
    assert loaded.to_dict(orient="list") == {"id": [1, 2], "target": [0.2, 0.8]}


def test_write_fold_intermediate_artifacts_expands_tiny_public_sample(tmp_path) -> None:
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.5, 0.5]})
    test = pd.DataFrame({"id": [11, 12, 13, 14], "feature": [1.0, 2.0, 3.0, 4.0]})
    result = PipelineResult(
        name="fold-ready",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)},
        oof_predictions_by_fold={"fold_1": np.array([0.2], dtype=np.float64)},
        valid_indices_by_fold={"fold_1": np.array([0])},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target",
    )

    assert records[0]["status"] == "available"
    submission = pd.read_csv(tmp_path / "submission_fold-ready_fold1.csv")
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [11, 12, 13, 14]
    assert np.allclose(submission["target"], [0.2, 0.4, 0.6, 0.8])


def test_write_fold_intermediate_artifacts_preserves_multi_target_sample_columns(tmp_path) -> None:
    sample = pd.DataFrame({"id": [10, 20], "target_a": [0.0, 0.0], "target_b": [0.5, 0.5]})
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="multi-target",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([0.25, 0.75], dtype=np.float64)},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="target_a",
    )

    assert records[0]["status"] == "available"
    submission = pd.read_csv(tmp_path / "submission_multi-target_fold1.csv")
    assert list(submission.columns) == ["id", "target_a", "target_b"]
    assert submission["target_a"].tolist() == [0.25, 0.75]
    assert submission["target_b"].tolist() == [0.5, 0.5]


def test_write_fold_intermediate_artifacts_expands_probability_matrix_columns(tmp_path) -> None:
    sample = pd.DataFrame(
        {
            "id": [10, 20],
            "class_bird": [1 / 3, 1 / 3],
            "class_cat": [1 / 3, 1 / 3],
            "class_dog": [1 / 3, 1 / 3],
        }
    )
    test = pd.DataFrame({"id": [10, 20], "feature": [1.0, 2.0]})
    result = PipelineResult(
        name="class-proba",
        oof_preds=np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]], dtype=np.float64),
        test_preds=np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={},
        metadata={"kind": "single"},
        test_predictions_by_fold={"fold_1": np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 1.0]], dtype=np.float64)},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )

    records = write_fold_intermediate_artifacts(
        output_dirs=[tmp_path],
        result=result,
        sample_submission=sample,
        test_df=test,
        id_col="id",
        target_col="label",
    )

    assert records[0]["status"] == "available"
    submission = pd.read_csv(tmp_path / "submission_class-proba_fold1.csv")
    probability_cols = ["class_bird", "class_cat", "class_dog"]
    assert list(submission.columns) == ["id", *probability_cols]
    assert submission["id"].tolist() == [10, 20]
    assert "label" not in submission.columns
    assert np.allclose(submission[probability_cols].sum(axis=1), [1.0, 1.0])
    assert np.allclose(submission.loc[0, probability_cols].to_numpy(dtype=float), [0.2, 0.3, 0.5])


def test_train_xgb_model_forwards_sample_weight(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeXGBClassifier:
        best_iteration = 12

        def __init__(self, **params):
            captured["params"] = params

        def fit(self, x_train, y_train, **kwargs):
            captured["x_train"] = x_train
            captured["y_train"] = y_train
            captured["fit_kwargs"] = kwargs
            return self

    monkeypatch.setattr(tabular_ensemble, "XGBClassifier", FakeXGBClassifier)

    x_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_train = np.array([0, 1, 0], dtype=np.int8)
    x_valid = pd.DataFrame({"feature": [4.0, 5.0]})
    y_valid = np.array([1, 0], dtype=np.int8)
    sample_weight = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    _, meta = train_xgb_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model_seed=42,
        params_override={},
        sample_weight=sample_weight,
    )

    fit_kwargs = captured["fit_kwargs"]
    assert isinstance(fit_kwargs, dict)
    assert np.array_equal(fit_kwargs["sample_weight"], sample_weight)
    assert fit_kwargs["eval_set"] == [(x_valid, y_valid)]
    assert meta["best_iteration"] == 12


def test_maybe_apply_pseudo_labels_extends_sample_weight(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class BaseModel:
        def predict_proba(self, frame):
            if frame["feature"].iloc[0] < 5.0:
                probs = np.array([0.2, 0.8], dtype=np.float64)
            else:
                probs = np.array([0.995, 0.005], dtype=np.float64)
            return np.column_stack([1.0 - probs, probs])

    class PseudoModel:
        def predict_proba(self, frame):
            if frame["feature"].iloc[0] < 5.0:
                probs = np.array([0.2, 0.8], dtype=np.float64)
            else:
                probs = np.array([0.99, 0.01], dtype=np.float64)
            return np.column_stack([1.0 - probs, probs])

    def fake_train_xgb_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model_seed,
        params_override,
        sample_weight=None,
    ):
        captured["x_train"] = x_train
        captured["y_train"] = y_train
        captured["sample_weight"] = sample_weight
        return PseudoModel(), {"device": "cpu", "best_iteration": 3}

    monkeypatch.setattr(tabular_ensemble, "train_xgb_model", fake_train_xgb_model)

    x_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = np.array([0, 1], dtype=np.int8)
    x_valid = pd.DataFrame({"feature": [3.0, 4.0]})
    y_valid = np.array([0, 1], dtype=np.int8)
    x_test = pd.DataFrame({"feature": [5.0, 6.0]})
    sample_weight = np.array([1.5, 2.5], dtype=np.float32)

    _, _, pl_log, pl_meta = maybe_apply_pseudo_labels(
        model=BaseModel(),
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        model_seed=42,
        threshold=0.99,
        enabled=True,
        params_override={},
        sample_weight=sample_weight,
    )

    assert np.array_equal(captured["sample_weight"], np.array([1.5, 2.5, 1.0, 1.0], dtype=np.float32))
    assert captured["y_train"].tolist() == [0, 1, 1, 0]
    assert pl_log["candidate_count"] == 2
    assert pl_meta["pseudo_model_device"] == "cpu"


def test_train_catboost_model_logs_gpu_fallback(monkeypatch, capsys) -> None:
    init_task_types: list[str] = []

    class FakeCatBoostClassifier:
        def __init__(self, **params):
            self.params = params
            init_task_types.append(str(params["task_type"]))

        def fit(self, x_train, y_train, **kwargs):
            if str(self.params["task_type"]).upper() == "GPU":
                raise RuntimeError("CUDA init failed")
            return self

        def get_best_iteration(self):
            return 17

    monkeypatch.setattr(tabular_ensemble, "CatBoostClassifier", FakeCatBoostClassifier)
    monkeypatch.setattr(tabular_ensemble, "PREFER_CUDA", True)

    x_train = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1.0, 2.0, 3.0]})
    y_train = np.array([0, 1, 0], dtype=np.int8)
    x_valid = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    y_valid = np.array([1, 0], dtype=np.int8)

    _, meta = train_catboost_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model_seed=42,
        params_override={},
        cat_features=["cat"],
        sample_weight=None,
    )

    captured = capsys.readouterr()
    assert init_task_types == ["GPU", "CPU"]
    assert meta["device"] == "cpu"
    assert meta["fallback_reason"] == "CUDA init failed"
    assert "CatBoost GPU failed; retrying on CPU: RuntimeError: CUDA init failed" in captured.out
