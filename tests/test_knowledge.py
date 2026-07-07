"""Tests for knowledge base tagging and search."""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
import tarfile
import zipfile

import pandas as pd
import pytest
import zstandard as zstd

import kagglebot.knowledge as knowledge_mod
from kagglebot.knowledge import (
    build_dataset_profile,
    build_plan_and_initial_prompt,
    derive_problem_types,
    ensure_taxonomy,
    format_error_fix_insights,
    knowledge_search,
    load_taxonomy,
    record_competition_profile,
    record_error_fix_insight,
    record_improvement,
    record_iteration,
    record_problem_type_insight,
    record_research_artifacts,
    record_run,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
    resolve_research_artifacts,
)
from kagglebot.knowledge.repositories import InsightRepository
from kagglebot.knowledge.skill_registry import record_skill_evaluation, search_skills, upsert_skill
from kagglebot.knowledge_context import (
    load_problem_type_knowledge_text,
    refresh_knowledge_hints,
    resolve_problem_types_from_profile,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths


def test_knowledge_search_orders_by_overlap(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    taxonomy = ensure_taxonomy(knowledge_paths)

    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="comp-a",
        competition_url=None,
        profile={"metric": "accuracy", "task": "classification", "tags": ["tabular", "binary"]},
    )
    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="comp-b",
        competition_url=None,
        profile={"metric": "rmse", "task": "regression", "tags": ["tabular"]},
    )

    results = knowledge_search(knowledge_paths, ["tabular", "binary"], limit=5)
    assert results[0]["slug"] == "comp-a"


def test_load_taxonomy_yaml(tmp_path) -> None:
    content = """
data_modality:
  - tabular
  - text
aliases:
  bin: binary
"""
    path = tmp_path / "taxonomy.yml"
    path.write_text(content, encoding="utf-8")
    data = load_taxonomy(path)
    assert "tabular" in data["tags"]
    assert data["aliases"]["bin"] == "binary"


def test_build_dataset_profile_samples_oversized_tables(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,feature,target\n1,10,0\n2,20,1\n3,30,0\n4,40,1\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n5,50\n6,60\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n5,0\n6,0\n", encoding="utf-8")

    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "1")
    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_rows"] == 4
    sampling = profile["profile_sampling"]
    assert sampling["enabled"] is True
    assert sampling["train"] is True
    assert sampling["test"] is True
    assert sampling["sample_submission"] is True


def test_build_dataset_profile_handles_zip_wrapped_parquet_tables(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in {
        "train.parquet.zip": pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0], "target": [0, 1, 0]}),
        "test.parquet.zip": pd.DataFrame({"id": [4, 5], "feature": [40.0, 50.0]}),
        "sample_submission.parquet.zip": pd.DataFrame({"id": [4, 5], "target": [0, 0]}),
    }.items():
        payload = io.BytesIO()
        frame.to_parquet(payload, index=False)
        with zipfile.ZipFile(data_dir / name, "w") as archive:
            archive.writestr(name.removesuffix(".zip"), payload.getvalue())

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_rows"] == 3
    assert profile["test_rows"] == 2
    assert profile["target_column"] == "target"
    assert profile["id_column"] == "id"
    assert "feature" in profile["numeric_columns"]


def test_build_dataset_profile_samples_oversized_semicolon_txt_tables(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.txt").write_text("id;feature;target\n1;10;0\n2;20;1\n3;30;0\n4;40;1\n", encoding="utf-8")
    (data_dir / "test.txt").write_text("id;feature\n5;50\n6;60\n", encoding="utf-8")
    (data_dir / "sample_submission.txt").write_text("id;target\n5;0\n6;0\n", encoding="utf-8")

    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "1")
    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_rows"] == 4
    assert profile["target_column"] == "target"
    assert profile["id_column"] == "id"


def test_build_dataset_profile_handles_semicolon_csv_tables(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id;feature;target\n1;10;0\n2;20;1\n3;30;0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id;feature\n4;40\n5;50\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id;target\n4;0\n5;0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["id_column"] == "id"
    assert profile["target_column"] == "target"
    assert "feature" in profile["numeric_columns"]


def test_build_dataset_profile_handles_single_label_probability_columns(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,feature,label\n1,0.1,cat\n2,0.2,dog\n3,0.3,bird\n4,0.4,cat\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n5,0.5\n6,0.6\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text(
        "id,class_bird,class_cat,class_dog\n5,0.333,0.333,0.334\n6,0.333,0.333,0.334\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_column"] == "label"
    assert profile["target_columns"] == ["label"]
    assert profile["prediction_kind_by_target"] == {"label": "probability_columns"}
    assert profile["target_semantics"] == "classification"
    assert profile["metric"] == "logloss"


def test_build_dataset_profile_detects_space_delimited_multi_label_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,feature,labels\n1,0.1,cat dog\n2,0.2,dog bird\n3,0.3,cat bird\n4,0.4,cat dog bird\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n5,0.5\n6,0.6\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,labels\n5,\n6,\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["task"] == "classification"
    assert profile["target_semantics"] == "multi_label"
    assert profile["target_semantics_by_target"] == {"labels": "multi_label"}
    assert profile["metric"] == "f1"
    assert "multi_label" in profile["tags"]


def test_build_dataset_profile_detects_delimited_multi_label_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        'id,feature,target\n1,0.1,"cat,dog"\n2,0.2,"dog,bird"\n3,0.3,"cat,bird"\n',
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n4,0.5\n5,0.6\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,\n5,\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_semantics"] == "multi_label"
    assert "multi_label" in profile["tags"]


def test_build_dataset_profile_detects_multi_label_indicator_targets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train_rows = ["id,comment_text,length," + ",".join(label_columns)]
    for index in range(1, 13):
        labels = [str(int((index + label_index) % 4 == 0)) for label_index, _ in enumerate(label_columns)]
        train_rows.append(f"{index},comment {index},{20 + index}," + ",".join(labels))
    (data_dir / "train.csv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text(
        "id,comment_text,length\n13,heldout one,33\n14,heldout two,34\n", encoding="utf-8"
    )
    (data_dir / "sample_submission.csv").write_text(
        "id," + ",".join(label_columns) + "\n13,0,0,0,0,0,0\n14,0,0,0,0,0,0\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == label_columns
    assert profile["task"] == "classification"
    assert profile["target_semantics"] == "multi_label"
    assert profile["target_semantics_by_target"] == {label: "multi_label" for label in label_columns}
    assert profile["metric"] == "f1"
    assert "multi_label" in profile["tags"]
    assert "multi_label" in derive_problem_types(profile)


def test_build_dataset_profile_does_not_treat_multiword_single_label_as_multi_label(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,feature,label\n1,0.1,very good\n2,0.2,very bad\n3,0.3,very good\n4,0.4,very bad\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n5,0.5\n6,0.6\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,label\n5,\n6,\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_semantics"] == "classification"
    assert "multi_label" not in profile["tags"]


def test_build_dataset_profile_detects_ordinal_classification_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,feature,severity\n1,0.1,0\n2,0.2,1\n3,0.3,2\n4,0.4,3\n5,0.5,4\n6,0.6,2\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n7,0.7\n8,0.8\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,severity\n7,0\n8,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["task"] == "classification"
    assert profile["target_semantics"] == "ordinal_classification"
    assert profile["target_semantics_by_target"] == {"severity": "ordinal_classification"}
    assert profile["metric"] == "quadratic_weighted_kappa"
    assert "ordinal_classification" in profile["tags"]
    assert "ordinal_classification" in derive_problem_types(profile)


def test_build_dataset_profile_detects_count_regression_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_rows = ["id,feature,count"]
    for index in range(1, 31):
        train_rows.append(f"{index},{index * 0.1:.2f},{index - 1}")
    (data_dir / "train.csv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n31,3.10\n32,3.20\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,count\n31,0\n32,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["count"]
    assert profile["task"] == "regression"
    assert profile["target_semantics"] == "count_regression"
    assert profile["target_semantics_by_target"] == {"count": "count_regression"}
    assert profile["metric"] == "rmsle"
    assert "count_regression" in profile["tags"]
    assert "count_regression" in derive_problem_types(profile)


def test_build_dataset_profile_detects_bounded_regression_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,feature,conversion_rate\n"
        "1,0.1,0.00\n"
        "2,0.2,0.25\n"
        "3,0.3,0.50\n"
        "4,0.4,0.75\n"
        "5,0.5,0.00\n"
        "6,0.6,0.25\n"
        "7,0.7,0.50\n"
        "8,0.8,0.75\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n9,0.9\n10,1.0\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,conversion_rate\n9,0.0\n10,0.0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["conversion_rate"]
    assert profile["task"] == "regression"
    assert profile["target_semantics"] == "bounded_regression"
    assert profile["target_semantics_by_target"] == {"conversion_rate": "bounded_regression"}
    assert profile["metric"] == "rmse"
    assert "bounded_regression" in profile["tags"]
    assert "bounded_regression" in derive_problem_types(profile)


def test_build_dataset_profile_detects_positive_skew_regression_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prices = [100, 110, 120, 130, 140, 150, 160, 170, 5000, 8000]
    rows = ["id,feature,SalePrice"]
    for index, price in enumerate(prices, start=1):
        rows.append(f"{index},{index * 0.1:.2f},{price}")
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n11,1.1\n12,1.2\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,SalePrice\n11,0\n12,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["SalePrice"]
    assert profile["task"] == "regression"
    assert profile["target_semantics"] == "positive_skew_regression"
    assert profile["target_semantics_by_target"] == {"SalePrice": "positive_skew_regression"}
    assert profile["metric"] == "rmsle"
    assert "positive_skew_regression" in profile["tags"]
    assert "positive_skew_regression" in derive_problem_types(profile)


def test_build_dataset_profile_detects_multi_output_regression_targets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_rows = ["id,feature,y1,y2"]
    for index in range(1, 26):
        train_rows.append(f"{index},{index * 0.1:.2f},{index * 1.5:.2f},{index * 2.5:.2f}")
    (data_dir / "train.csv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n26,2.60\n27,2.70\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,y1,y2\n26,0.0,0.0\n27,0.0,0.0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["y1", "y2"]
    assert profile["task"] == "regression"
    assert profile["target_semantics"] == "multi_output_regression"
    assert profile["target_semantics_by_target"] == {"y1": "regression", "y2": "regression"}
    assert profile["metric"] == "rmse"
    assert "multi_output" in profile["tags"]


def test_build_dataset_profile_detects_coordinate_regression_targets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_rows = ["id,feature,x,y,z"]
    for index in range(1, 31):
        train_rows.append(f"{index},{index * 0.1:.2f},{index * 1.5:.2f},{index * 2.5:.2f},{index * 3.5:.2f}")
    (data_dir / "train.csv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n31,3.10\n32,3.20\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,x,y,z\n31,0.0,0.0,0.0\n32,0.0,0.0,0.0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["x", "y", "z"]
    assert profile["task"] == "regression"
    assert profile["target_semantics"] == "coordinate_regression"
    assert profile["target_semantics_by_target"] == {
        "x": "coordinate_regression",
        "y": "coordinate_regression",
        "z": "coordinate_regression",
    }
    assert profile["metric"] == "rmse"
    assert "coordinate_regression" in profile["tags"]
    assert "coordinate_regression" in derive_problem_types(profile)


def test_build_dataset_profile_detects_quantile_submission_columns(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = ["id,feature,target"]
    for index in range(1, 26):
        rows.append(f"{index},{index * 0.1:.2f},{index * 1.5:.2f}")
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n26,2.60\n27,2.70\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text(
        "id,p10,p50,p90\n26,0.0,0.0,0.0\n27,0.0,0.0,0.0\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["target"]
    assert profile["target_semantics"] == "quantile_regression"
    assert profile["target_semantics_by_target"] == {"target": "quantile_regression"}
    assert profile["metric"] == "pinball_loss"
    assert "quantile_regression" in profile["tags"]
    assert "quantile_regression" in derive_problem_types(profile)


def test_build_dataset_profile_detects_prediction_interval_submission_columns(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = ["id,feature,target"]
    for index in range(1, 26):
        rows.append(f"{index},{index * 0.1:.2f},{index * 1.5:.2f}")
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n26,2.60\n27,2.70\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text(
        "id,lower,upper\n26,0.0,1.0\n27,0.0,1.0\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_semantics"] == "prediction_interval"
    assert profile["target_semantics_by_target"] == {"target": "prediction_interval"}
    assert profile["metric"] == "interval_score"
    assert "prediction_interval" in profile["tags"]


def test_build_dataset_profile_detects_multi_target_classification_targets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,feature,target_a,target_b\n1,0.1,1,0\n2,0.2,0,1\n3,0.3,1,1\n4,0.4,0,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n5,0.5\n6,0.6\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target_a,target_b\n5,0,0\n6,0,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["target_a", "target_b"]
    assert profile["task"] == "classification"
    assert profile["target_semantics"] == "multi_target_classification"
    assert profile["target_semantics_by_target"] == {"target_a": "classification", "target_b": "classification"}
    assert "multi_target" in profile["tags"]


def test_build_dataset_profile_detects_pairwise_preference_features(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,prompt,model_a,model_b,winner_model_a,winner_model_b,winner_tie\n"
        "1,hello,alpha,beta,1,0,0\n"
        "2,world,beta,gamma,0,1,0\n"
        "3,test,alpha,gamma,0,0,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,prompt,model_a,model_b\n4,question,alpha,beta\n5,answer,beta,gamma\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text(
        "id,winner_model_a,winner_model_b,winner_tie\n4,0.33,0.33,0.34\n5,0.33,0.33,0.34\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["winner_model_a", "winner_model_b", "winner_tie"]
    assert profile["target_semantics"] == "pairwise"
    assert "pairwise" in profile["tags"]


def test_build_dataset_profile_detects_learning_to_rank_features(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,query_id,document_id,bm25,doc_length,relevance\n"
        "1,q1,d1,3.1,120,3\n"
        "2,q1,d2,1.2,240,1\n"
        "3,q1,d3,0.4,80,0\n"
        "4,q2,d4,2.8,150,2\n"
        "5,q2,d5,1.7,90,1\n"
        "6,q2,d6,0.2,300,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,query_id,document_id,bm25,doc_length\n7,q3,d7,2.1,140\n8,q3,d8,0.9,210\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,relevance\n7,0\n8,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_semantics"] == "learning_to_rank"
    assert profile["target_semantics_by_target"] == {"relevance": "learning_to_rank"}
    assert profile["metric"] == "ndcg"
    assert "learning_to_rank" in profile["tags"]
    assert "learning_to_rank" in derive_problem_types(profile)


def test_build_dataset_profile_handles_unlabeled_anomaly_detection_layout(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,amount,velocity,country\n1,10.5,0.1,JP\n2,11.2,0.2,US\n3,980.0,9.8,BR\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,amount,velocity,country\n4,12.0,0.1,JP\n5,1200.0,12.5,US\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,anomaly_score\n4,0.0\n5,0.0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["task"] == "unsupervised"
    assert profile["target_column"] == "anomaly_score"
    assert profile["target_semantics"] == "anomaly_detection"
    assert profile["target_semantics_by_target"] == {"anomaly_score": "anomaly_detection"}
    assert profile["metric"] == "auc"
    assert "unsupervised" in profile["tags"]
    assert "anomaly_detection" in profile["tags"]
    assert "anomaly_detection" in derive_problem_types(profile)


def test_build_dataset_profile_detects_ctr_user_item_interactions(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,user_id,ad_id,device,clicked\n1,u1,a1,mobile,1\n2,u1,a2,desktop,0\n3,u2,a1,mobile,0\n4,u3,a3,desktop,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,user_id,ad_id,device\n5,u1,a3,mobile\n6,u4,a2,desktop\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,clicked\n5,0.5\n6,0.5\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_semantics"] == "ctr"
    assert profile["target_semantics_by_target"] == {"clicked": "ctr"}
    assert profile["metric"] == "logloss"
    assert "ctr" in profile["tags"]
    assert "ctr" in derive_problem_types(profile)


def test_build_dataset_profile_detects_recommender_rating_interactions(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = ["id,user_id,item_id,context_feature,rating"]
    for index in range(1, 31):
        rows.append(f"{index},u{index % 5},i{index % 7},{index * 0.2:.1f},{1.0 + index * 0.13:.2f}")
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text(
        "id,user_id,item_id,context_feature\n31,u1,i3,6.2\n32,u2,i4,6.4\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,rating\n31,3.0\n32,3.0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["task"] == "regression"
    assert profile["target_semantics"] == "recommender"
    assert profile["target_semantics_by_target"] == {"rating": "recommender"}
    assert profile["metric"] == "rmse"
    assert "recommender" in profile["tags"]
    assert "recommender" in derive_problem_types(profile)


def test_build_dataset_profile_detects_survival_event_time_targets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = ["id,feature,efs,efs_time"]
    for index in range(1, 26):
        rows.append(f"{index},{index * 0.1:.2f},{index % 2},{index * 1.7:.2f}")
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n5,0.5\n6,0.6\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,prediction\n5,0.0\n6,0.0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_columns"] == ["efs", "efs_time"]
    assert profile["task"] == "mixed"
    assert profile["target_semantics"] == "survival"
    assert profile["target_semantics_by_target"] == {"efs": "classification", "efs_time": "regression"}
    assert profile["metric"] == "concordance_index"
    assert "survival" in profile["tags"]


def test_build_dataset_profile_handles_text_submission_columns(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,prompt,translation\n1,a,alpha one\n2,b,beta two\n3,c,gamma three\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,prompt\n4,d\n5,e\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,translation\n4,\n5,\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["target_column"] == "translation"
    assert profile["task"] == "text"
    assert profile["task_by_target"] == {"translation": "text"}
    assert profile["target_semantics"] == "text_generation"
    assert profile["target_semantics_by_target"] == {"translation": "text_generation"}
    assert profile["prediction_kind_by_target"] == {"translation": "text"}
    assert profile["metric"] == "text_similarity"
    assert "text" in profile["tags"]
    assert "text_generation" in profile["tags"]


def test_build_dataset_profile_detects_short_text_classification_feature(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,review,label\n"
        "1,Great acting and pacing,1\n"
        "2,Flat plot but nice cast,0\n"
        "3,Loved the ending,1\n"
        "4,Too slow for me,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,review\n5,Funny and sharp\n6,Weak script overall\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,label\n5,0\n6,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "text"
    assert profile["task"] == "classification"
    assert "text" in profile["tags"]


def test_build_dataset_profile_detects_multimodal_asset_and_text_features(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,image_path,question,label\n"
        "1,train/a.jpg,Is this product damaged?,1\n"
        "2,train/b.jpg,Does the label mention organic ingredients?,0\n"
        "3,train/c.jpg,Is the package open or sealed?,1\n"
        "4,train/d.jpg,What condition is shown in the image?,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,image_path,question\n"
        "5,test/e.jpg,Is the product visibly damaged?\n"
        "6,test/f.jpg,Does the package show a readable label?\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,label\n5,0\n6,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "multimodal"
    assert profile["task"] == "classification"
    assert "multimodal" in profile["tags"]
    assert "multimodal" in derive_problem_types(profile)


def test_build_dataset_profile_does_not_treat_short_code_column_as_text(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,comment_code,feature,target\n1,A1,0.4,0\n2,B2,0.2,1\n3,C3,0.8,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,comment_code,feature\n4,D4,0.7\n5,E5,0.1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "tabular"


def test_build_dataset_profile_detects_audio_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    audio_dir = data_dir / "audio" / "train"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,filename,target\n1,a.wav,0\n2,b.wav,1\n3,c.wav,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,filename\n4,d.wav\n5,e.wav\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")
    (audio_dir / "a.wav").write_bytes(b"RIFF")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "audio"
    assert "audio" in profile["tags"]


def test_build_dataset_profile_detects_non_tabular_image_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "case_001.jpg").write_bytes(b"jpeg")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "non_tabular_data"
    assert profile["modality"] == "image"
    assert profile["task"] == "image"
    assert profile["metric"] == "unknown"
    assert "image" in profile["tags"]


def test_build_dataset_profile_detects_non_tabular_audio_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    audio_dir = data_dir / "audio" / "test"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "clip_001.wav").write_bytes(b"RIFF")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "non_tabular_data"
    assert profile["modality"] == "audio"
    assert profile["task"] == "audio"
    assert "audio" in profile["tags"]


def test_build_dataset_profile_detects_image_from_filename_suffix_values(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,filename,target\n1,train/a.jxl,0\n2,train/b.exr,1\n3,train/c.webp,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,filename\n4,test/d.jxl\n5,test/e.exr\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "image"
    assert "image" in profile["tags"]


def test_build_dataset_profile_detects_text_from_document_filename_suffix_values(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,document_path,target\n1,docs/a.pdf,0\n2,docs/b.epub,1\n3,docs/c.tex.gz,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,document_path\n4,docs/d.odt\n5,docs/e.pptx\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "text"
    assert "text" in profile["tags"]


def test_build_dataset_profile_detects_signal_from_waveform_filename_suffix_values(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,waveform_path,target\n1,signals/a.edf,0\n2,signals/b.hea.gz,1\n3,signals/c.nwb,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,waveform_path\n4,signals/d.edf\n5,signals/e.hea.gz\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "signal"
    assert "signal" in profile["tags"]


def test_build_dataset_profile_detects_object_detection_submission_semantics(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,image_path,prediction_string\n1,train/a.jpg,0 0.9 0.5 0.5 0.2 0.2\n2,train/b.jpg,-\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,image_path\n3,test/c.jpg\n4,test/d.jpg\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,prediction_string\n3,-\n4,-\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "image"
    assert profile["target_semantics"] == "object_detection"
    assert profile["metric"] == "map"
    assert "object_detection" in profile["tags"]


def test_build_dataset_profile_detects_segmentation_submission_semantics(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,image_path,EncodedPixels\n1,train/a.png,1 3 10 2\n2,train/b.png,\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,image_path\n3,test/c.png\n4,test/d.png\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,EncodedPixels\n3,\n4,\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "image"
    assert profile["target_semantics"] == "segmentation"
    assert profile["metric"] == "dice"
    assert "segmentation" in profile["tags"]


def test_build_dataset_profile_detects_annotation_only_detection_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    annotations_dir = data_dir / "annotations"
    labels_dir = data_dir / "labels"
    annotations_dir.mkdir(parents=True)
    labels_dir.mkdir()
    (annotations_dir / "instances_train.json").write_text(
        '{"images": [], "annotations": [], "categories": []}',
        encoding="utf-8",
    )
    (labels_dir / "image_001.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "non_tabular_data"
    assert profile["modality"] == "image"
    assert profile["task"] == "image"
    assert "image" in profile["tags"]


def test_build_dataset_profile_detects_signal_only_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "record.edf").write_bytes(b"edf")
    (data_dir / "record.hea.gz").write_bytes(b"compressed header")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "non_tabular_data"
    assert profile["modality"] == "signal"
    assert profile["task"] == "signal"
    assert "signal" in profile["tags"]


def test_build_dataset_profile_does_not_infer_asset_from_generic_filename_without_suffix(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,filename,feature,target\n1,case_001,0.4,0\n2,case_002,0.2,1\n3,case_003,0.8,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,filename,feature\n4,case_004,0.7\n5,case_005,0.1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "tabular"


@pytest.mark.parametrize(
    ("column_name", "expected_modality"),
    [
        ("audio_path", "audio"),
        ("clip_filename", "video"),
        ("scan_path", "medical_imaging"),
        ("array_file", "array"),
        ("lidar_file", "point_cloud"),
        ("pdf_file", "text"),
        ("annotation_path", "annotation"),
    ],
)
def test_build_dataset_profile_detects_asset_reference_modality_from_column_name(
    tmp_path,
    column_name: str,
    expected_modality: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        f"id,{column_name},feature,target\n1,case_001,0.4,0\n2,case_002,0.2,1\n3,case_003,0.8,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        f"id,{column_name},feature\n4,case_004,0.7\n5,case_005,0.1\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == expected_modality
    assert expected_modality in profile["tags"]


def test_build_dataset_profile_detects_medical_imaging_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    scan_dir = data_dir / "scans" / "train"
    scan_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,scan_path,target\n1,a.nii.gz,0\n2,b.nii.gz,1\n3,c.nii.gz,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,scan_path\n4,d.nii.gz\n5,e.nii.gz\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")
    (scan_dir / "a.nii.gz").write_bytes(b"scan")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "medical_imaging"
    assert "medical_imaging" in profile["tags"]


def test_build_dataset_profile_detects_point_cloud_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    point_dir = data_dir / "points" / "train"
    point_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,filename,target\n1,a.ply,0\n2,b.ply,1\n3,c.ply,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,filename\n4,d.ply\n5,e.ply\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")
    (point_dir / "a.ply").write_bytes(b"ply")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "point_cloud"
    assert "point_cloud" in profile["tags"]


def test_build_dataset_profile_detects_numpy_array_assets(tmp_path) -> None:
    data_dir = tmp_path / "data"
    array_dir = data_dir / "arrays" / "train"
    array_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,filename,target\n1,a.npy,0\n2,b.npy,1\n3,c.npy,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,filename\n4,d.npy\n5,e.npy\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")
    (array_dir / "a.npy").write_bytes(b"\x93NUMPY")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "array"
    assert "array" in profile["tags"]


def test_build_dataset_profile_extracts_top_level_zip(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_dir / "competition.zip", "w") as archive:
        archive.writestr("train.csv", "id,feature,target\n1,10,0\n2,20,1\n3,30,0\n")
        archive.writestr("test.csv", "id,feature\n4,40\n5,50\n")
        archive.writestr("sample_submission.csv", "id,target\n4,0\n5,0\n")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.csv"
    assert profile["sample_submission_file"] == "sample_submission.csv"
    assert (data_dir / "train.csv").exists()


def test_build_dataset_profile_extracts_top_level_tgz(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(data_dir / "competition.tgz", "w:gz") as archive:
        for name, text in {
            "train.csv": "id,feature,target\n1,10,0\n2,20,1\n3,30,0\n",
            "test.csv": "id,feature\n4,40\n5,50\n",
            "sample_submission.csv": "id,target\n4,0\n5,0\n",
        }.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.csv"
    assert profile["sample_submission_file"] == "sample_submission.csv"
    assert (data_dir / "train.csv").exists()


def test_build_dataset_profile_extracts_nested_archives(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_zip = io.BytesIO()
    with zipfile.ZipFile(train_zip, "w") as archive:
        archive.writestr("train.csv", "id,feature,target\n1,10,0\n2,20,1\n3,30,0\n")
    test_zip = io.BytesIO()
    with zipfile.ZipFile(test_zip, "w") as archive:
        archive.writestr("test.csv", "id,feature\n4,40\n5,50\n")
    with zipfile.ZipFile(data_dir / "competition.zip", "w") as archive:
        archive.writestr("train.zip", train_zip.getvalue())
        archive.writestr("test.zip", test_zip.getvalue())
        archive.writestr("sample_submission.csv", "id,target\n4,0\n5,0\n")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.csv"
    assert profile["sample_submission_file"] == "sample_submission.csv"
    assert (data_dir / "train.csv").exists()


def test_build_dataset_profile_reads_xlsx_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd = pytest.importorskip("pandas")
    pd.DataFrame({"id": [1, 2, 3], "feature": [10, 20, 30], "target": [0, 1, 0]}).to_excel(
        data_dir / "train.xlsx",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_excel(data_dir / "test.xlsx", index=False)
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_excel(
        data_dir / "sample_submission.xlsx",
        index=False,
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.xlsx"
    assert profile["sample_submission_file"] == "sample_submission.xlsx"
    assert profile["target_column"] == "target"


def test_build_dataset_profile_reads_csv_gz_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / "train.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n1,10,0\n2,20,1\n3,30,0\n")
    with gzip.open(data_dir / "test.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n4,40\n5,50\n")
    with gzip.open(data_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n4,0\n5,0\n")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.csv.gz"
    assert profile["sample_submission_file"] == "sample_submission.csv.gz"
    assert profile["target_column"] == "target"
    assert profile["file_extension_counts"] == {".csv.gz": 3}


def test_build_dataset_profile_reads_jsonl_gz_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / "train.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in (
            {"id": 1, "feature": 10, "target": 0},
            {"id": 2, "feature": 20, "target": 1},
            {"id": 3, "feature": 30, "target": 0},
        ):
            handle.write(json.dumps(row) + "\n")
    with gzip.open(data_dir / "test.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in ({"id": 4, "feature": 40}, {"id": 5, "feature": 50}):
            handle.write(json.dumps(row) + "\n")
    with gzip.open(data_dir / "sample_submission.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in ({"id": 4, "target": 0}, {"id": 5, "target": 0}):
            handle.write(json.dumps(row) + "\n")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.jsonl.gz"
    assert profile["sample_submission_file"] == "sample_submission.jsonl.gz"
    assert profile["target_column"] == "target"


def test_build_dataset_profile_samples_oversized_ndjson_zst_tables(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor()
    payloads = {
        "train.ndjson.zst": "\n".join(
            [
                json.dumps({"id": 1, "feature": 10, "target": 0}),
                json.dumps({"id": 2, "feature": 20, "target": 1}),
                json.dumps({"id": 3, "feature": 30, "target": 0}),
            ]
        )
        + "\n",
        "test.ndjson.zst": "\n".join(
            [
                json.dumps({"id": 4, "feature": 40}),
                json.dumps({"id": 5, "feature": 50}),
            ]
        )
        + "\n",
        "sample_submission.ndjson.zst": "\n".join(
            [
                json.dumps({"id": 4, "target": 0}),
                json.dumps({"id": 5, "target": 0}),
            ]
        )
        + "\n",
    }
    for name, text in payloads.items():
        (data_dir / name).write_bytes(compressor.compress(text.encode("utf-8")))

    read_calls = []
    original_read_table = knowledge_mod.read_table

    def spy_read_table(path, *, nrows=None):
        read_calls.append((path.name, nrows))
        return original_read_table(path, nrows=nrows)

    monkeypatch.setattr(knowledge_mod, "read_table", spy_read_table)
    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "1")
    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.ndjson.zst"
    assert profile["sample_submission_file"] == "sample_submission.ndjson.zst"
    assert profile["train_rows"] == 3
    sampling = profile["profile_sampling"]
    assert sampling["enabled"] is True
    assert sampling["train"] is True
    assert sampling["test"] is True
    assert sampling["sample_submission"] is True
    assert ("train.ndjson.zst", knowledge_mod._PROFILE_SAMPLE_ROWS) in read_calls  # noqa: SLF001
    assert ("test.ndjson.zst", knowledge_mod._PROFILE_SAMPLE_ROWS) in read_calls  # noqa: SLF001
    assert ("sample_submission.ndjson.zst", knowledge_mod._PROFILE_SAMPLE_ROWS) in read_calls  # noqa: SLF001


def test_build_dataset_profile_reads_sqlite_tables(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "competition.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE train (id INTEGER, feature INTEGER, target INTEGER)")
        conn.executemany(
            "INSERT INTO train VALUES (?, ?, ?)",
            [(1, 10, 0), (2, 20, 1), (3, 30, 0)],
        )
        conn.execute("CREATE TABLE test (id INTEGER, feature INTEGER)")
        conn.executemany("INSERT INTO test VALUES (?, ?)", [(4, 40), (5, 50)])
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target INTEGER)")
        conn.executemany("INSERT INTO sample_submission VALUES (?, ?)", [(4, 0), (5, 0)])

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "competition__train.csv"
    assert profile["sample_submission_file"] == "competition__sample_submission.csv"
    assert profile["target_column"] == "target"


def test_profile_max_table_bytes_env_uses_shared_number_parsing(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "nan")
    assert knowledge_mod._profile_max_table_bytes() == 256 * 1024 * 1024  # noqa: SLF001

    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "0")
    assert knowledge_mod._profile_max_table_bytes() == 256 * 1024 * 1024  # noqa: SLF001

    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "1024")
    assert knowledge_mod._profile_max_table_bytes() == 1024  # noqa: SLF001


def test_build_dataset_profile_handles_json_list_features(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.json").write_text(
        json.dumps(
            [
                {"id": "a", "grid": [[1, 2], [3, 4]], "target": 0},
                {"id": "b", "grid": [[4, 3], [2, 1]], "target": 1},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "test.json").write_text(
        json.dumps(
            [
                {"id": "c", "grid": [[1, 1], [1, 1]]},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "sample_submission.json").write_text(
        json.dumps(
            [
                {"id": "c", "target": 0},
            ]
        ),
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["file_extension_counts"] == {".json": 3}


def test_build_dataset_profile_does_not_treat_times_suffix_as_timeseries(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,Total asset turnover rate (Times),target\n1,0.1,0\n2,0.2,1\n3,0.3,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,Total asset turnover rate (Times)\n4,0.4\n5,0.5\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "tabular"


def test_build_dataset_profile_detects_repeated_entity_group_column(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,patient_id,feature,target\n1,p1,0.1,0\n2,p1,0.2,1\n3,p2,0.3,0\n4,p2,0.4,1\n5,p3,0.5,0\n6,p3,0.6,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,patient_id,feature\n7,p4,0.7\n8,p5,0.8\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n7,0\n8,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["group_column_hint"] == "patient_id"
    assert profile["split_strategy_hint"] == "group_kfold"
    assert "grouped" in profile["tags"]
    assert "grouped" in derive_problem_types(profile)


def test_build_dataset_profile_detects_sample_weight_column(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,feature,sample_weight,target\n1,0.1,1.0,0\n2,0.2,2.0,1\n3,0.3,0.5,0\n4,0.4,3.0,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n5,0.5\n6,0.6\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n5,0\n6,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["sample_weight_column_hint"] == "sample_weight"
    assert profile["sample_weight_summary"]["max"] == 3.0
    assert "sample_weighted" in profile["tags"]
    assert "sample_weighted" in derive_problem_types(profile)
    assert "sample_weight" in profile["train_only_columns"]
    assert "sample_weight" not in profile["numeric_columns"]


def test_build_dataset_profile_detects_timeseries_from_datetime_feature(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,store_id,sale_date,feature,target\n1,A,2024-01-01,10,1.0\n2,A,2024-01-02,11,1.1\n3,B,2024-01-01,8,0.8\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,store_id,sale_date,feature\n4,A,2024-01-03,12\n5,B,2024-01-02,9\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "timeseries"
    assert profile["target_semantics"] == "forecasting"
    assert profile["target_semantics_by_target"] == {"target": "forecasting"}
    assert profile["metric"] == "rmse"
    assert profile["split_strategy_hint"] == "timeseries_split"
    assert "timeseries" in profile["tags"]
    assert "forecasting" in profile["tags"]
    assert "forecasting" in derive_problem_types(profile)


def test_build_dataset_profile_detects_timeseries_from_future_ordinal_feature(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,item_id,date_block_num,feature,target\n1,A,0,10,1.0\n2,A,1,11,1.1\n3,A,2,12,1.2\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,item_id,date_block_num,feature\n4,A,3,13\n5,A,4,14\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "timeseries"
    assert profile["target_semantics"] == "forecasting"
    assert profile["target_semantics_by_target"] == {"target": "forecasting"}
    assert profile["split_strategy_hint"] == "timeseries_split"
    assert "forecasting" in profile["tags"]


def test_build_dataset_profile_detects_geospatial_from_lat_lon_features(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,latitude,longitude,feature,target\n1,35.68,139.76,10,1.0\n2,34.69,135.50,11,1.1\n3,43.06,141.35,12,1.2\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,latitude,longitude,feature\n4,33.59,130.40,13\n5,35.18,136.90,14\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "geospatial"
    assert "geospatial" in profile["tags"]


def test_build_dataset_profile_detects_geospatial_from_wkt_geometry(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,geometry,feature,target\n"
        '1,"POINT (139.76 35.68)",10,1.0\n'
        '2,"POINT (135.50 34.69)",11,1.1\n'
        '3,"POINT (141.35 43.06)",12,1.2\n',
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        'id,geometry,feature\n4,"POINT (130.40 33.59)",13\n5,"POINT (136.90 35.18)",14\n',
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "geospatial"


def test_build_dataset_profile_detects_compressed_geojson_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor()
    train_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 1,
                "properties": {"feature": 10, "target": 1.0},
                "geometry": {"type": "Point", "coordinates": [139.76, 35.68]},
            },
            {
                "type": "Feature",
                "id": 2,
                "properties": {"feature": 11, "target": 1.1},
                "geometry": {"type": "Point", "coordinates": [135.50, 34.69]},
            },
        ],
    }
    test_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 3,
                "properties": {"feature": 12},
                "geometry": {"type": "Point", "coordinates": [130.40, 33.59]},
            }
        ],
    }
    (data_dir / "train.geojson.zst").write_bytes(compressor.compress(json.dumps(train_payload).encode("utf-8")))
    (data_dir / "test.geojson.zst").write_bytes(compressor.compress(json.dumps(test_payload).encode("utf-8")))
    (data_dir / "sample_submission.csv").write_text("id,target\n3,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_file"] == "train.geojson.zst"
    assert profile["test_file"] == "test.geojson.zst"
    assert profile["modality"] == "geospatial"
    assert "geospatial" in profile["tags"]
    assert "geospatial" in derive_problem_types(profile)


def test_derive_problem_types_preserves_geospatial_tag() -> None:
    assert "geospatial" in derive_problem_types({"tags": ["geospatial"]})


def test_build_dataset_profile_detects_graph_from_edge_columns(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,source_node,target_node,weight,label\n1,A,B,0.4,1\n2,B,C,0.2,0\n3,C,D,0.8,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,source_node,target_node,weight\n4,D,E,0.7\n5,E,F,0.1\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,label\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "graph"
    assert "graph" in profile["tags"]


def test_build_dataset_profile_detects_graph_from_edge_index_column(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        'id,edge_index,feature,target\n1,"0 1",0.4,1\n2,"1 2",0.2,0\n3,"2 3",0.8,1\n',
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text('id,edge_index,feature\n4,"3 4",0.7\n5,"4 5",0.1\n', encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "graph"


def test_build_dataset_profile_detects_bio_from_smiles_column(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,smiles,feature,target\n1,CCO,0.4,1\n2,CN(C)C=O,0.2,0\n3,c1ccccc1,0.8,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,smiles,feature\n4,CCN,0.7\n5,O=C=O,0.1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "bio"
    assert "bio" in profile["tags"]


def test_build_dataset_profile_detects_bio_from_molecule_filename_suffix_values(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,molecule_path,target\n1,molecules/a.smi,0\n2,molecules/b.smiles.gz,1\n3,molecules/c.inchi,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,molecule_path\n4,molecules/d.smi\n5,molecules/e.selfies\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "bio"
    assert "bio" in profile["tags"]


def test_build_dataset_profile_detects_bio_from_protein_sequence(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,protein_sequence,feature,target\n"
        "1,ACDEFGHIKLMNPQRSTVWY,0.4,1\n"
        "2,MNPQRSTVWYACDEFGHIKL,0.2,0\n"
        "3,GHIKLMNPQRSTVWYACDEF,0.8,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,protein_sequence,feature\n4,ACDEFGHIKLMNPQRS,0.7\n5,MNPQRSTVWYACDEF,0.1\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "bio"


def test_build_dataset_profile_does_not_treat_generic_protein_sequence_as_rna(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,sequence,feature,target\n"
        "1,ACDEFGHIKLMNPQRSTVWY,0.4,1\n"
        "2,MNPQRSTVWYACDEFGHIKL,0.2,0\n"
        "3,GHIKLMNPQRSTVWYACDEF,0.8,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,sequence,feature\n4,ACDEFGHIKLMNPQRS,0.7\n5,MNPQRSTVWYACDEF,0.1\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "bio"


def test_build_dataset_profile_detects_rna_from_sequence_column(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,rna_sequence,feature,target\n1,ACGUACGUACGU,0.4,1\n2,GGCAUUGCAUUG,0.2,0\n3,UUGGCCAAGGUU,0.8,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,rna_sequence,feature\n4,ACGUACGUAA,0.7\n5,GGCAUUGCAA,0.1\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "rna"
    assert "rna" in profile["tags"]


def test_build_dataset_profile_handles_march_mania_submission_only_format(tmp_path) -> None:
    data_dir = tmp_path / "march-machine-learning-mania-2026" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "MRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,1101,70,1102,65,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,3101,80,3102,72,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "MNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,1101,75,1102,70,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,3101,77,3102,71,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "AnswerTemplate.csv").write_text(
        "ID,Pred\n2026_1101_1102,0.5\n2026_3101_3102,0.5\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["sample_submission_file"] == "AnswerTemplate.csv"
    assert profile["id_column"] == "ID"
    assert profile["target_column"] == "Pred"
    assert profile["task"] == "classification"
    assert profile["target_semantics"] == "pairwise"
    assert profile["metric"] == "brier_score"
    assert profile["split_strategy_hint"] == "group_kfold"
    assert "binary" in profile["tags"]
    assert "pairwise" in profile["tags"]


def test_build_dataset_profile_does_not_treat_idless_submission_as_pairwise_profile(tmp_path) -> None:
    data_dir = tmp_path / "march-machine-learning-mania-2026" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "MRegularSeasonCompactResults.csv",
        "WRegularSeasonCompactResults.csv",
        "MNCAATourneyCompactResults.csv",
        "WNCAATourneyCompactResults.csv",
    ):
        (data_dir / name).write_text(
            "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,1101,70,1102,65,N,0\n",
            encoding="utf-8",
        )
    (data_dir / "sample_submission.csv").write_text(
        "target,score\n0,0.5\n1,0.5\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] != "ok"


def test_build_plan_and_initial_prompt_handles_unknown_train_dimensions(tmp_path) -> None:
    data_dir = tmp_path / "march-machine-learning-mania-2026" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "MRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,1101,70,1102,65,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,3101,80,3102,72,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "MNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,1101,75,1102,70,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,3101,77,3102,71,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "SampleSubmissionStage1.csv").write_text(
        "ID,Pred\n2026_1101_1102,0.5\n2026_3101_3102,0.5\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    prompt = build_plan_and_initial_prompt(
        slug="march-machine-learning-mania-2026",
        rules_url="https://www.kaggle.com/competitions/march-machine-learning-mania-2026",
        profile=profile,
        taxonomy={},
        similar_improvements=[],
    )

    assert "**Dataset**: train table unavailable; sample/test view: 2 rows × 2 columns" in prompt


def test_build_plan_and_initial_prompt_uses_profile_sample_submission_filename() -> None:
    profile = {
        "task": "classification",
        "metric": "accuracy",
        "sample_submission_file": "sample_submission.jsonl",
        "tags": ["tabular", "classification"],
    }

    prompt = build_plan_and_initial_prompt(
        slug="demo",
        rules_url="https://www.kaggle.com/competitions/demo/rules",
        profile=profile,
        taxonomy={},
        similar_improvements=[],
    )

    assert "artifacts/demo/context/sample_submission.jsonl" in prompt
    assert "artifacts/demo/context/sample_submission_head.csv" in prompt
    assert "artifacts/demo/context/sample_submission.csv" not in prompt
    assert "the sample submission file" in prompt


def test_build_plan_and_initial_prompt_uses_text_sample_head_suffix() -> None:
    profile = {
        "task": "classification",
        "metric": "accuracy",
        "sample_submission_file": "sample_submission.tsv.gz",
        "tags": ["tabular", "classification"],
    }

    prompt = build_plan_and_initial_prompt(
        slug="demo",
        rules_url="https://www.kaggle.com/competitions/demo/rules",
        profile=profile,
        taxonomy={},
        similar_improvements=[],
    )

    assert "artifacts/demo/context/sample_submission.tsv.gz" in prompt
    assert "artifacts/demo/context/sample_submission_head.tsv" in prompt
    assert "artifacts/demo/context/sample_submission_head.csv" not in prompt


def test_build_plan_and_initial_prompt_uses_sample_glob_when_profile_filename_missing() -> None:
    prompt = build_plan_and_initial_prompt(
        slug="demo",
        rules_url="https://www.kaggle.com/competitions/demo/rules",
        profile={"task": "classification", "metric": "accuracy", "tags": ["tabular", "classification"]},
        taxonomy={},
        similar_improvements=[],
    )

    assert "artifacts/demo/context/sample_submission.* or the detected sample-submission alias" in prompt
    assert "artifacts/demo/context/sample_submission_head.* or the detected sample-submission preview" in prompt
    assert "artifacts/demo/context/sample_submission.csv" not in prompt


def test_build_plan_and_initial_prompt_includes_target_semantics() -> None:
    prompt = build_plan_and_initial_prompt(
        slug="demo",
        rules_url="https://www.kaggle.com/competitions/demo/rules",
        profile={
            "task": "classification",
            "target_semantics": "multi_label",
            "metric": "f1",
            "tags": ["tabular", "classification", "multi_label"],
        },
        taxonomy={},
        similar_improvements=[],
    )

    assert "**Task**: classification" in prompt
    assert "**Target semantics**: multi_label" in prompt


def test_initial_prompt_includes_current_winner_mode_directives() -> None:
    profile = {
        "task": "classification",
        "metric": "accuracy",
        "rows": 10,
        "cols": 3,
        "target": "label",
        "tags": ["tabular", "classification"],
    }

    prompt = knowledge_mod.build_plan_and_initial_prompt(
        slug="demo",
        rules_url="https://www.kaggle.com/competitions/demo/rules",
        profile=profile,
        taxonomy={},
        similar_improvements=[],
        self_improvement_context="Keep high-ceiling candidates.",
    )

    assert "## System Self-Improvement Directives" in prompt
    assert "Keep high-ceiling candidates." in prompt
    assert '"target_medal": "winner"' in prompt
    assert '"target_rank_percentile": 0.001' in prompt
    assert "time_budget_min" not in prompt


def test_problem_type_insight_record_and_resolve(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    profile = {"modality": "tabular", "task": "regression", "tags": ["tabular", "regression"]}
    problem_types = derive_problem_types(profile)
    assert "tabular:regression" in problem_types

    record_problem_type_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        run_id="run-1",
        iteration=1,
        problem_types=problem_types,
        why_poor="Model was underfit with weak features and high validation error.",
        how_improved="Added CatBoost and richer feature engineering with longer training.",
        delta_offline=0.12,
        outcome_bucket="good",
        submission_score=0.8123,
    )

    insights = resolve_problem_type_insights(knowledge_paths, ["tabular:regression"], limit=5)
    assert insights
    first = insights[0]
    assert first["problem_type"] == "tabular:regression"
    assert first["cause_category"] != ""
    assert first["fix_category"] != ""
    assert first["outcome_bucket"] == "good"
    assert first["submission_score"] == 0.8123


def test_error_fix_insight_record_and_resolve(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    problem_types = ["tabular:binary", "tabular"]

    record_error_fix_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-b",
        run_id="run-2",
        iteration=2,
        problem_types=problem_types,
        error_message="ModuleNotFoundError: No module named 'featurewiz'",
        fix_summary="Removed featurewiz import and added sklearn fallback.",
        resolved=True,
        outcome_bucket="low",
        submission_score=0.731,
    )

    insights = resolve_error_fix_insights(knowledge_paths, ["tabular:binary"], limit=5)
    assert insights
    first = insights[0]
    assert first["error_category"] == "dependency_missing"
    assert bool(first["resolved"]) is True
    assert first["outcome_bucket"] == "low"
    assert first["submission_score"] == 0.731

    rendered = format_error_fix_insights(insights, limit=5)
    assert "dependency_missing" in rendered
    assert "featurewiz" in rendered


def test_load_problem_type_knowledge_text_renders_shared_context(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(
        json.dumps({"modality": "tabular", "task": "binary", "tags": ["classification"]}),
        encoding="utf-8",
    )
    problem_types = ["tabular:binary", "tabular", "binary", "classification"]
    record_problem_type_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        run_id="run-1",
        iteration=1,
        problem_types=problem_types,
        why_poor="Validation did not match public leaderboard.",
        how_improved="Use stratified folds and check prediction distribution.",
        delta_offline=0.012,
        outcome_bucket="high",
        submission_score=0.88,
    )
    record_error_fix_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        run_id="run-1",
        iteration=1,
        problem_types=problem_types,
        error_message="Submission Scoring Error: incorrect format",
        fix_summary="Align columns and ids to sample_submission.",
        resolved=True,
        outcome_bucket="high",
        submission_score=0.88,
    )
    record_research_artifacts(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        problem_types=problem_types,
        research_sources_jsonl='{"url":"https://example.com"}',
        research_summary_md="Research summary.",
    )
    upsert_skill(
        knowledge_paths=knowledge_paths,
        skill_id="tabular_binary_oof_blend",
        title="Tabular Binary OOF Blend",
        summary="Use OOF predictions and a small blend when tabular binary public gap persists.",
        body="Build leak-free stratified OOF predictions, then blend diverse GBDT families.",
        tags=["tabular", "binary", "online_far_from_top1"],
        problem_types=["tabular:binary", "tabular", "binary"],
        status="active",
        source="test",
    )

    text = load_problem_type_knowledge_text(
        dataset_profile_path=profile_path,
        knowledge_paths=knowledge_paths,
        include_research=True,
    )

    assert "Problem-type knowledge" in text
    assert "Error-fix knowledge" in text
    assert "Cross-competition research artifacts" in text
    assert "Reusable Kaggle skills" in text
    assert "tabular_binary_oof_blend" in text
    assert "Submission Scoring Error" in text


def test_skill_registry_versions_searches_and_records_fitness(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    first = upsert_skill(
        knowledge_paths=knowledge_paths,
        skill_id="submit_failure_recovery",
        title="Submit Failure Recovery",
        summary="Recover from submit failures with safe retry classification.",
        body="Classify error, preserve artifacts, retry only when allowed.",
        tags=["submit_failed", "submission"],
        problem_types=["submission"],
        status="candidate",
        source="test",
    )
    second = upsert_skill(
        knowledge_paths=knowledge_paths,
        skill_id="submit_failure_recovery",
        title="Submit Failure Recovery",
        summary="Recover from submit failures with safe retry classification.",
        body="Classify error, preserve artifacts, retry only when allowed, then record outcome.",
        tags=["submit_failed", "submission"],
        problem_types=["submission"],
        status="active",
        source="test",
    )
    record_skill_evaluation(
        knowledge_paths=knowledge_paths,
        skill_id="submit_failure_recovery",
        outcome="recovered",
        slug="demo",
        run_id="run-1",
        submit_recovered=True,
    )

    assert first["version"] == 1
    assert second["version"] == 2
    assert (tmp_path / "knowledge" / "skills" / "submit_failure_recovery.md").exists()
    skills = search_skills(
        knowledge_paths=knowledge_paths,
        problem_types=["submission"],
        query="submit failure retry",
    )
    assert skills[0]["skill_id"] == "submit_failure_recovery"
    assert skills[0]["usage_count"] == 1
    assert skills[0]["success_count"] == 1


def test_load_problem_type_knowledge_text_can_skip_research(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps({"modality": "tabular", "task": "binary"}), encoding="utf-8")

    text = load_problem_type_knowledge_text(
        dataset_profile_path=profile_path,
        knowledge_paths=knowledge_paths,
        include_research=False,
    )

    assert "No prior problem-type insights available." in text
    assert "No prior error-fix insights available." in text
    assert "Cross-competition research artifacts" not in text


def test_resolve_research_artifacts_ignores_invalid_problem_types_json(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    record_research_artifacts(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        problem_types=["tabular", "binary"],
        research_sources_jsonl='{"url":"https://example.com"}',
        research_summary_md="Research summary.",
    )
    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            "UPDATE competition_research SET problem_types_json = ? WHERE slug = ?",
            ("{", "comp-a"),
        )

    records = resolve_research_artifacts(knowledge_paths=knowledge_paths, problem_types=["tabular"])

    assert records
    assert records[0]["problem_types"] == []


def test_resolve_problem_types_from_profile_handles_json_profile(tmp_path) -> None:
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps({"modality": "tabular", "task": "regression"}), encoding="utf-8")

    assert resolve_problem_types_from_profile(dataset_profile_path=profile_path) == [
        "tabular:regression",
        "tabular",
        "regression",
    ]


def test_resolve_problem_types_from_profile_handles_invalid_profile(tmp_path) -> None:
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    assert resolve_problem_types_from_profile(dataset_profile_path=profile_path) == ["unknown"]


def test_refresh_knowledge_hints_writes_similar_competition_context(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path / "knowledge")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(json.dumps({"tags": ["tabular", "binary"]}), encoding="utf-8")
    taxonomy = ensure_taxonomy(knowledge_paths)
    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="prior-comp",
        competition_url=None,
        profile={"tags": ["tabular", "binary"], "metric": "auc"},
    )
    record_run(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        slug="prior-comp",
        compute="local",
        goal_metric="auc",
        goal_score=0.8,
        direction="maximize",
    )
    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        summary="Stratified folds improved public score.",
        delta_offline=0.02,
    )

    refresh_knowledge_hints(paths=paths, knowledge_paths=knowledge_paths)

    hints = paths.knowledge_hints_path.read_text(encoding="utf-8")
    assert "Similar competitions and what improved score" in hints
    assert "prior-comp" in hints
    assert "Stratified folds improved public score." in hints
    assert "No self-improvement context available yet." in hints


def test_knowledge_classifies_external_signal_and_online_mismatch(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_error_fix_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-c",
        run_id="run-3",
        iteration=1,
        problem_types=["tabular:binary"],
        error_message="ORIG_proba constant_fallback: original_data_found=false and reference inputs missing.",
        fix_summary="Recover original data from reference inputs and disable constant_fallback.",
        resolved=False,
        outcome_bucket="unknown",
        submission_score=None,
    )
    error_insights = resolve_error_fix_insights(knowledge_paths, ["tabular:binary"], limit=5)
    assert error_insights[0]["error_category"] == "external_signal_missing"

    record_problem_type_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-c",
        run_id="run-3",
        iteration=2,
        problem_types=["tabular:binary"],
        why_poor="Offline improved but public leaderboard regressed, indicating an online mismatch.",
        how_improved="Ban same-family-only tuning and increase model-family diversity with blending.",
        delta_offline=None,
        outcome_bucket="low",
        submission_score=0.812,
    )
    problem_insights = resolve_problem_type_insights(knowledge_paths, ["tabular:binary"], limit=5)
    assert problem_insights[0]["cause_category"] == "online_mismatch"
    assert problem_insights[0]["fix_category"] == "model_diversification"


def test_record_iteration_upserts_on_duplicate_run_iteration(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_iteration(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    record_iteration(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_iteration_falls_back_to_update_on_unique_violation(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    state = {"insert_calls": 0, "raised": False}

    class _FlakyConnection:
        def __init__(self, path, state_map):
            self._conn = sqlite3.connect(path)
            self._state = state_map

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def execute(self, sql, params=()):
            if "INSERT INTO iterations" in sql:
                self._state["insert_calls"] += 1
                if self._state["insert_calls"] == 2 and not self._state["raised"]:
                    self._state["raised"] = True
                    raise sqlite3.IntegrityError("UNIQUE constraint failed: iterations.run_id, iterations.iter")
            return self._conn.execute(sql, params)

    repo = InsightRepository(
        knowledge_paths,
        ensure_db=knowledge_mod._ensure_db,
        connect=lambda path: _FlakyConnection(path, state),
    )

    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert state["raised"] is True
    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_iteration_falls_back_to_update_on_unique_violation_code_name(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    state = {"insert_calls": 0, "raised": False}

    class _FlakyConnection:
        def __init__(self, path, state_map):
            self._conn = sqlite3.connect(path)
            self._state = state_map

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def execute(self, sql, params=()):
            if "INSERT INTO iterations" in sql:
                self._state["insert_calls"] += 1
                if self._state["insert_calls"] == 2 and not self._state["raised"]:
                    self._state["raised"] = True
                    exc = sqlite3.IntegrityError("constraint failed")
                    exc.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_UNIQUE
                    exc.sqlite_errorname = "SQLITE_CONSTRAINT_UNIQUE"
                    raise exc
            return self._conn.execute(sql, params)

    repo = InsightRepository(
        knowledge_paths,
        ensure_db=knowledge_mod._ensure_db,
        connect=lambda path: _FlakyConnection(path, state),
    )

    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert state["raised"] is True
    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_improvement_upserts_on_duplicate_run_iteration(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=2,
        summary="first",
        delta_offline=0.01,
    )
    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=2,
        summary="second",
        delta_offline=0.08,
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT summary, delta_offline
            FROM improvements
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 2),
        ).fetchone()

    assert row == ("second", 0.08)
