from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.compute import Compute
from kagglebot.solver.initial_model import (
    _sibling_tabular_artifact_path,
    _tabular_holdout_indices,
    train_evaluate_and_predict,
)
from kagglebot.solver.io import load_competition_data, read_table


def _write_zip_parquet(path: Path, member_name: str, frame: pd.DataFrame) -> None:
    payload = io.BytesIO()
    frame.to_parquet(payload, index=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member_name, payload.getvalue())


def test_initial_model_trains_local_tabular_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "num": [float(idx) for idx in range(30)],
            "cat": ["a" if idx % 2 else "b" for idx in range(30)],
            "target": [idx % 2 for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "num": [1.5, 2.5, 3.5], "cat": ["a", "b", "a"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert outcome.model_name == "local_tabular_baseline"
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [100, 101, 102]
    assert len(submission) == 3


def test_initial_model_trains_from_zip_wrapped_parquet_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_zip_parquet(
        data_dir / "train.parquet.zip",
        "nested/train.parquet",
        pd.DataFrame(
            {
                "id": range(30),
                "num": [float(idx) for idx in range(30)],
                "cat": ["a" if idx % 2 else "b" for idx in range(30)],
                "target": [idx % 2 for idx in range(30)],
            }
        ),
    )
    _write_zip_parquet(
        data_dir / "test.parquet.zip",
        "nested/test.parquet",
        pd.DataFrame({"id": [100, 101, 102], "num": [1.5, 2.5, 3.5], "cat": ["a", "b", "a"]}),
    )
    _write_zip_parquet(
        data_dir / "sample_submission.parquet.zip",
        "nested/sample_submission.parquet",
        pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}),
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = read_table(outcome.submission_path)
    assert outcome.model_name == "local_tabular_baseline"
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [100, 101, 102]
    assert len(submission) == 3


def test_initial_model_tabular_baseline_respects_compressed_output_suffix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv.gz",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.submission_path.name == "submission.csv.gz"
    submission = read_table(outcome.submission_path)
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [100, 101]


def test_initial_model_tabular_baseline_uses_manifest_fallback_for_non_tabular_output_suffix(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "answers.nii.gz",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.submission_path == tmp_path / "answers.tabular.csv"
    assert not (tmp_path / "answers.nii.gz").exists()
    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [100, 101]
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "tabular"
    assert manifest["submission_path"] == "answers.tabular.csv"
    assert manifest["requested_output_path"] == "answers.nii.gz"


def test_initial_model_tabular_baseline_respects_zstd_pickle_sample_and_output(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_pickle(data_dir / "sample_submission.pkl.zst")

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.pkl.zst",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.submission_path.name == "submission.pkl.zst"
    submission = read_table(outcome.submission_path)
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [100, 101]


def test_initial_model_tabular_baseline_respects_html_sample_and_output(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_html(
        data_dir / "sample_submission.html",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.html",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.submission_path.name == "submission.html"
    submission = read_table(outcome.submission_path)
    assert list(submission.columns) == ["id", "target"]
    assert submission["id"].tolist() == [100, 101]


def test_initial_model_tabular_baseline_ignores_missing_target_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    targets = [idx % 2 for idx in range(40)]
    for idx in range(0, 40, 7):
        targets[idx] = None
    pd.DataFrame(
        {
            "id": range(40),
            "feature": [float(idx) for idx in range(40)],
            "segment": ["a" if idx % 2 else "b" for idx in range(40)],
            "target": targets,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0], "segment": ["a", "b", "a"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert submission["id"].tolist() == [100, 101, 102]
    assert outcome.model_summary["target_summaries"]["target"]["model"] == "logistic_regression"


def test_initial_model_tabular_baseline_uses_sample_weight_column(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = 40
    pd.DataFrame(
        {
            "id": range(rows),
            "feature": [float(idx) for idx in range(rows)],
            "sample_weight": [1.0 if idx % 3 else 3.0 for idx in range(rows)],
            "target": [idx % 2 for idx in range(rows)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "feature": [1.0, 2.0, 3.0],
            "sample_weight": [1.0, 1.0, 1.0],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    target_summary = outcome.model_summary["target_summaries"]["target"]
    assert target_summary["model"] == "logistic_regression"
    assert target_summary["features"] == 1
    assert target_summary["sample_weight_column"] == "sample_weight"


def test_initial_model_tabular_baseline_uses_group_holdout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    group_values = [f"p{idx // 3}" for idx in range(30)]
    pd.DataFrame(
        {
            "id": range(30),
            "patient_id": group_values,
            "feature": [float(idx) for idx in range(30)],
            "target": [idx % 2 for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "patient_id": ["p100", "p101"],
            "feature": [1.0, 2.0],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    data = load_competition_data(data_dir)
    train_idx, valid_idx = _tabular_holdout_indices(
        train=data.train,
        primary_target=data.target_column,
        task=data.task,
        seed=7,
        holdout_frac=0.2,
        group_column=data.group_column,
    )

    train_groups = set(data.train.iloc[train_idx]["patient_id"].astype(str))
    valid_groups = set(data.train.iloc[valid_idx]["patient_id"].astype(str))
    assert data.group_column == "patient_id"
    assert train_groups.isdisjoint(valid_groups)

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.model_summary["group_column"] == "patient_id"
    assert outcome.model_summary["split_strategy"] == "group_shuffle_split"


def test_initial_model_tabular_baseline_uses_timeseries_holdout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "date_block_num": list(range(20)),
            "feature": [float(idx) for idx in range(20)],
            "target": [float(idx * 2) for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "date_block_num": [20, 21],
            "feature": [20.0, 21.0],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)
    train_idx, valid_idx = _tabular_holdout_indices(
        train=data.train,
        primary_target=data.target_column,
        task=data.task,
        seed=7,
        holdout_frac=0.2,
        time_column=data.time_column,
    )

    assert data.time_column == "date_block_num"
    assert data.train.iloc[train_idx]["date_block_num"].max() < data.train.iloc[valid_idx]["date_block_num"].min()

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.model_summary["time_column"] == "date_block_num"
    assert outcome.model_summary["split_strategy"] == "timeseries_holdout"


def test_initial_model_adds_calendar_features_for_datetime_columns(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=24, freq="D")
    pd.DataFrame(
        {
            "id": range(24),
            "date": dates.strftime("%Y-%m-%d"),
            "feature": [float(idx) for idx in range(24)],
            "target": [float(idx * 2) for idx in range(24)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "date": pd.date_range("2024-01-25", periods=2, freq="D").strftime("%Y-%m-%d"),
            "feature": [24.0, 25.0],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    temporal_features = outcome.model_summary["target_summaries"]["target"]["temporal_calendar_features"]
    assert "__time_date_year" in temporal_features
    assert "__time_date_dayofweek" in temporal_features
    assert "__time_date_year" in outcome.model_summary["temporal_calendar_feature_columns"]
    assert "__time_date_dayofweek" in outcome.model_summary["temporal_calendar_feature_columns"]
    assert outcome.model_summary["target_summaries"]["target"]["features"] > 2


def test_initial_model_tabular_baseline_aligns_sample_order_by_id(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [idx % 2 for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [12, 11], "feature": [12.0, 11.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": ["11", "12"], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path, dtype={"id": str})
    assert submission["id"].tolist() == ["11", "12"]


def test_initial_model_aligns_composite_sample_id_from_test_columns(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = list(range(30))
    pd.DataFrame(
        {
            "user_id": [f"u{idx}" for idx in rows],
            "item_id": [f"i{idx}" for idx in rows],
            "feature": [float(idx) for idx in rows],
            "target": [float(idx * 10) for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "feature": [1.0, 2.0],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"row_id": ["u2:i2", "u1:i1"], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert submission["row_id"].tolist() == ["u2:i2", "u1:i1"]
    assert submission.loc[0, "target"] > submission.loc[1, "target"]


def test_initial_model_tabular_baseline_preserves_leading_zero_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(
        "id,feature,target\n001,1.0,0\n002,2.0,1\n003,3.0,0\n004,4.0,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n006,6.0\n005,5.0\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n005,0\n006,0\n", encoding="utf-8")

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.25,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path, dtype={"id": str})
    assert submission["id"].tolist() == ["005", "006"]


def test_initial_model_trains_probability_column_tabular_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    labels = ["bird", "cat", "dog"] * 12
    pd.DataFrame(
        {
            "id": range(len(labels)),
            "feature": [float(idx) for idx in range(len(labels))],
            "label": labels,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(data_dir / "test.csv", index=False)
    probability_cols = ["class_bird", "class_cat", "class_dog"]
    pd.DataFrame(
        {
            "id": [100, 101],
            "class_bird": [1 / 3, 1 / 3],
            "class_cat": [1 / 3, 1 / 3],
            "class_dog": [1 / 3, 1 / 3],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="logloss",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", *probability_cols]
    assert submission["id"].tolist() == [100, 101]
    assert np.allclose(submission[probability_cols].sum(axis=1), [1.0, 1.0])


def test_initial_model_maps_suffix_probability_columns_by_label_name(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    labels = ["bird"] * 20 + ["cat"] * 20 + ["dog"] * 20
    features = [float(-10 + idx * 0.05) for idx in range(20)]
    features += [float(idx * 0.05) for idx in range(20)]
    features += [float(10 + idx * 0.05) for idx in range(20)]
    pd.DataFrame(
        {
            "id": range(len(labels)),
            "feature": features,
            "label": labels,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [-10.0, 0.0, 10.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    probability_cols = ["dog_probability", "cat_probability", "bird_probability"]
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "dog_probability": [1 / 3, 1 / 3, 1 / 3],
            "cat_probability": [1 / 3, 1 / 3, 1 / 3],
            "bird_probability": [1 / 3, 1 / 3, 1 / 3],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="logloss",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", *probability_cols]
    assert submission[probability_cols].idxmax(axis=1).tolist() == [
        "bird_probability",
        "cat_probability",
        "dog_probability",
    ]
    assert np.allclose(submission[probability_cols].sum(axis=1), [1.0, 1.0, 1.0])


def test_initial_model_uses_probabilities_for_auc_metric_with_integer_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "target": [0 if idx < 20 else 1 for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [5.0, 20.0, 35.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="auc",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "target"]
    assert submission["target"].between(0.0, 1.0).all()
    assert not set(submission["target"].round(8)).issubset({0.0, 1.0})
    assert outcome.model_summary["prediction_kind_by_target"] == {"target": "class"}
    assert outcome.model_summary["target_summaries"]["target"]["prediction_kind"] == "probability"


def test_initial_model_uses_probabilities_for_probability_named_binary_target(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "isFraud": [0 if idx < 20 else 1 for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [5.0, 20.0, 35.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "isFraud": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "isFraud"]
    assert submission["isFraud"].between(0.0, 1.0).all()
    assert not set(submission["isFraud"].round(8)).issubset({0.0, 1.0})
    assert outcome.model_summary["prediction_kind_by_target"] == {"isFraud": "probability"}


def test_initial_model_trains_multi_label_column_tabular_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = list(range(36))
    labels = ["cat dog", "dog bird", "cat bird"] * 12
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "labels": labels,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(data_dir / "test.csv", index=False)
    label_cols = ["cat", "dog", "bird"]
    pd.DataFrame(
        {
            "id": [100, 101],
            "cat": [0.0, 0.0],
            "dog": [0.0, 0.0],
            "bird": [0.0, 0.0],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", *label_cols]
    assert submission["id"].tolist() == [100, 101]
    assert outcome.model_summary["prediction_kind_by_target"] == {"labels": "multi_label_columns"}
    assert set(outcome.model_summary["target_summaries"]["labels"]["prediction_columns"]) == set(label_cols)
    assert ((submission[label_cols] >= 0.0) & (submission[label_cols] <= 1.0)).all().all()
    assert not np.allclose(submission[label_cols].sum(axis=1), [1.0, 1.0])


def test_initial_model_text_target_uses_tfidf_nearest_neighbor(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["hello world", "good night", "green apple", "blue ocean"],
            "translation": ["alpha one", "beta two", "gamma three", "delta four"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "prompt": ["good night", "green apple"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "translation": ["", ""]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="text_similarity",
        direction="maximize",
        holdout_frac=0.25,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert submission["translation"].tolist() == ["beta two", "gamma three"]
    summary = outcome.model_summary["target_summaries"]["translation"]
    assert summary["model"] == "tfidf_nearest_neighbor"
    assert summary["text_feature_columns"] == ["prompt"]


def test_initial_model_natural_language_target_uses_text_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["red apple", "blue ocean", "green forest", "yellow flower"],
            "target": [
                "write a concise answer about red apples",
                "write a concise answer about blue oceans",
                "write a concise answer about green forests",
                "write a concise answer about yellow flowers",
            ],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "prompt": ["blue ocean", "green forest"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": ["placeholder", "placeholder"]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="text_similarity",
        direction="maximize",
        holdout_frac=0.25,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert submission["target"].tolist() == [
        "write a concise answer about blue oceans",
        "write a concise answer about green forests",
    ]
    assert outcome.model_summary["task_by_target"] == {"target": "text"}
    assert outcome.model_summary["prediction_kind_by_target"] == {"target": "text"}
    summary = outcome.model_summary["target_summaries"]["target"]
    assert summary["model"] == "tfidf_nearest_neighbor"
    assert summary["text_feature_columns"] == ["prompt"]


def test_initial_model_multi_label_target_uses_text_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["red item", "blue item", "green item", "yellow item"],
            "labels": ["cat dog", "dog bird", "cat bird", "cat dog bird"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "prompt": ["blue item", "green item"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "labels": ["cat dog", "cat dog"]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="text_similarity",
        direction="maximize",
        holdout_frac=0.25,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert submission["labels"].tolist() == ["dog bird", "cat bird"]
    summary = outcome.model_summary["target_summaries"]["labels"]
    assert outcome.model_summary["prediction_kind_by_target"] == {"labels": "text"}
    assert summary["model"] == "tfidf_nearest_neighbor"
    assert summary["text_feature_columns"] == ["prompt"]


def test_initial_model_expands_regression_to_quantile_columns(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx * 2) for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "p10": [0.0, 0.0], "p50": [0.0, 0.0], "p90": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="pinball_loss",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "p10", "p50", "p90"]
    assert submission["id"].tolist() == [100, 101]
    assert (submission["p10"] <= submission["p50"]).all()
    assert (submission["p50"] <= submission["p90"]).all()
    summary = outcome.model_summary["target_summaries"]["target"]
    assert summary["expanded_prediction_kind"] == "quantile_columns"
    assert summary["expanded_prediction_columns"] == ["p10", "p50", "p90"]


def test_initial_model_expands_regression_to_prediction_interval_columns(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx * 2) for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "lower": [0.0, 0.0], "upper": [1.0, 1.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="interval_score",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "lower", "upper"]
    assert submission["id"].tolist() == [100, 101]
    assert (submission["lower"] <= submission["upper"]).all()
    summary = outcome.model_summary["target_summaries"]["target"]
    assert summary["expanded_prediction_kind"] == "prediction_interval_columns"
    assert summary["expanded_prediction_columns"] == ["lower", "upper"]


def test_initial_model_writes_unlabeled_anomaly_score_submission(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "amount": [10.5, 11.2, 980.0],
            "velocity": [0.1, 0.2, 9.8],
            "country": ["JP", "US", "BR"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [4, 5],
            "amount": [12.0, 1200.0],
            "velocity": [0.1, 12.5],
            "country": ["JP", "US"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [4, 5], "anomaly_score": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="auc",
        direction="maximize",
        holdout_frac=0.25,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "anomaly_score"]
    assert submission["id"].tolist() == [4, 5]
    assert submission["anomaly_score"].between(0.0, 1.0).all()
    assert submission.loc[1, "anomaly_score"] > submission.loc[0, "anomaly_score"]
    summary = outcome.model_summary["target_summaries"]["anomaly_score"]
    assert outcome.model_summary["task_by_target"] == {"anomaly_score": "unsupervised"}
    assert summary["model"] == "robust_unsupervised_anomaly_score"


def test_initial_model_writes_survival_event_time_as_single_risk_score(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "efs": [idx % 2 for idx in range(30)],
            "efs_time": [float(30 - idx) for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "prediction": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="concordance_index",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "prediction"]
    assert submission["id"].tolist() == [100, 101]
    assert "efs" not in submission.columns
    assert "efs_time" not in submission.columns
    assert submission["prediction"].between(0.0, 1.0).all()
    assert "prediction" in outcome.model_summary["target_summaries"]
    assert outcome.model_summary["target_summaries"]["prediction"]["model"] == "survival_risk_score"


def test_initial_model_treats_learning_to_rank_relevance_as_continuous_score(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "query_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "document_id": ["d1", "d2", "d3", "d4", "d5", "d6"],
            "bm25": [3.1, 1.2, 0.4, 2.8, 1.7, 0.2],
            "doc_length": [120, 240, 80, 150, 90, 300],
            "relevance": [3, 1, 0, 2, 1, 0],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [7, 8],
            "query_id": ["q3", "q3"],
            "document_id": ["d7", "d8"],
            "bm25": [2.1, 0.9],
            "doc_length": [140, 210],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [7, 8], "relevance": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="ndcg",
        direction="maximize",
        holdout_frac=0.25,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "relevance"]
    assert submission["id"].tolist() == [7, 8]
    assert outcome.model_summary["task_by_target"] == {"relevance": "regression"}
    assert outcome.model_summary["prediction_kind_by_target"] == {"relevance": "continuous"}
    assert outcome.model_summary["target_summaries"]["relevance"]["model"] == "ridge"
    assert pd.api.types.is_float_dtype(submission["relevance"])


def test_initial_model_treats_ordinal_target_as_rounded_score(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(1, 21),
            "feature": [float(idx) for idx in range(20)],
            "severity": [idx % 5 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [101, 102, 103], "feature": [20.0, 21.0, 22.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [101, 102, 103], "severity": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="quadratic_weighted_kappa",
        direction="maximize",
        holdout_frac=0.25,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "severity"]
    assert submission["id"].tolist() == [101, 102, 103]
    assert outcome.model_summary["task_by_target"] == {"severity": "regression"}
    assert outcome.model_summary["prediction_kind_by_target"] == {"severity": "ordinal"}
    assert set(submission["severity"]).issubset({0, 1, 2, 3, 4})
    assert pd.api.types.is_integer_dtype(submission["severity"])


def test_initial_model_treats_named_low_cardinality_numeric_target_as_regression(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "sales": [idx % 4 for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [40.0, 41.0, 42.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "sales": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "sales"]
    assert outcome.model_summary["task_by_target"] == {"sales": "regression"}
    assert outcome.model_summary["prediction_kind_by_target"] == {"sales": "continuous"}
    assert outcome.model_summary["target_summaries"]["sales"]["model"] == "ridge"
    assert "target_transform" not in outcome.model_summary["target_summaries"]["sales"]
    assert pd.api.types.is_float_dtype(submission["sales"])


def test_initial_model_clips_bounded_regression_predictions(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "conversion_rate": [idx / 29.0 for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [100.0, 200.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "conversion_rate": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "conversion_rate"]
    assert outcome.model_summary["task_by_target"] == {"conversion_rate": "regression"}
    assert outcome.model_summary["prediction_kind_by_target"] == {"conversion_rate": "continuous"}
    assert submission["conversion_rate"].between(0.0, 1.0).all()
    assert submission["conversion_rate"].max() == 1.0


def test_initial_model_clips_count_regression_predictions_non_negative(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "count": [30 - idx for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [100.0, 200.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "count": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="rmsle",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "count"]
    assert outcome.model_summary["task_by_target"] == {"count": "regression"}
    assert outcome.model_summary["prediction_kind_by_target"] == {"count": "continuous"}
    assert outcome.model_summary["target_summaries"]["count"]["model"] == "ridge_log1p"
    assert outcome.model_summary["target_summaries"]["count"]["target_transform"] == "log1p"
    assert outcome.model_summary["target_summaries"]["count"]["inverse_transform"] == "expm1"
    assert (submission["count"] >= 0.0).all()
    assert submission["count"].min() == 0.0


def test_initial_model_clips_positive_skew_regression_predictions_non_negative(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    prices = [8000, 5000, 170, 160, 150, 140, 130, 120, 110, 100]
    pd.DataFrame(
        {
            "id": list(range(len(prices))),
            "feature": [float(idx) for idx in range(len(prices))],
            "SalePrice": prices,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [100.0, 200.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "SalePrice": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="rmsle",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path)
    assert list(submission.columns) == ["id", "SalePrice"]
    assert outcome.model_summary["task_by_target"] == {"SalePrice": "regression"}
    assert outcome.model_summary["prediction_kind_by_target"] == {"SalePrice": "continuous"}
    assert outcome.model_summary["target_summaries"]["SalePrice"]["model"] == "ridge_log1p"
    assert outcome.model_summary["target_summaries"]["SalePrice"]["target_transform"] == "log1p"
    assert outcome.model_summary["target_summaries"]["SalePrice"]["inverse_transform"] == "expm1"
    assert (submission["SalePrice"] >= 0.0).all()
    assert submission["SalePrice"].min() == 0.0


def test_initial_model_routes_rle_segmentation_to_empty_mask_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": ["train_1", "train_2", "train_3"],
            "image_path": ["train/a.png", "train/b.png", "train/c.png"],
            "EncodedPixels": ["1 2 10 3", "", "4 1 20 2"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": ["test_1", "test_2"],
            "image_path": ["test/a.png", "test/b.png"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": ["test_1", "test_2"], "EncodedPixels": ["", ""]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="dice",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    submission = pd.read_csv(outcome.submission_path, keep_default_na=False)
    assert outcome.model_name == "rle_empty_mask_baseline"
    assert list(submission.columns) == ["id", "EncodedPixels"]
    assert submission["id"].tolist() == ["test_1", "test_2"]
    assert submission["EncodedPixels"].tolist() == ["-", "-"]


def test_initial_model_rle_segmentation_respects_jsonl_output_suffix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": ["train_1", "train_2", "train_3"],
            "image_path": ["train/a.png", "train/b.png", "train/c.png"],
            "EncodedPixels": ["1 2 10 3", "", "4 1 20 2"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": ["test_1", "test_2"], "image_path": ["test/a.png", "test/b.png"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": ["test_1", "test_2"], "EncodedPixels": ["", ""]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=tmp_path / "submission.jsonl",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=7,
        score_source="holdout",
        metric="dice",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.submission_path.name == "submission.jsonl"
    submission = read_table(outcome.submission_path)
    assert submission["id"].tolist() == ["test_1", "test_2"]
    assert submission["EncodedPixels"].tolist() == ["-", "-"]


def test_sibling_tabular_artifact_path_preserves_compound_suffix(tmp_path: Path) -> None:
    assert _sibling_tabular_artifact_path(tmp_path / "submission.csv.gz", stem="validation_submission").name == (
        "validation_submission.csv.gz"
    )
    assert _sibling_tabular_artifact_path(tmp_path / "submission.jsonl.zst", stem="validation_submission").name == (
        "validation_submission.jsonl.zst"
    )


def test_sibling_tabular_artifact_path_ignores_non_tabular_compound_suffix(tmp_path: Path) -> None:
    assert _sibling_tabular_artifact_path(tmp_path / "answers.nii.gz", stem="validation_submission").name == (
        "validation_submission.csv"
    )
    assert _sibling_tabular_artifact_path(tmp_path / "answers.zarr", stem="validation_submission").name == (
        "validation_submission.csv"
    )
