from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from kagglebot.local_kernel_drift_guard import (
    build_zero_overlap_drift_guard_payload,
    infer_target_column_from_frames,
    prepare_zero_overlap_drift_guard,
)


def test_infer_target_column_from_frames_uses_train_only_column() -> None:
    assert (
        infer_target_column_from_frames(train_columns=["id", "feature", "target"], test_columns=["id", "feature"])
        == "target"
    )
    assert infer_target_column_from_frames(train_columns=["id", "feature"], test_columns=["id", "feature"]) is None


def test_build_zero_overlap_drift_guard_payload_detects_high_risk_feature() -> None:
    train = pd.DataFrame(
        {
            "id": ["A", "B", "C", "D", "E", "F"],
            "risk_cat": ["x", "x", "x", "y", "y", "y"],
            "safe_cat": ["same", "same", "same", "same", "same", "same"],
            "target": [1, 1, 1, 0, 0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "id": ["T1", "T2", "T3"],
            "risk_cat": ["u", "u", "v"],
            "safe_cat": ["same", "same", "same"],
        }
    )

    payload = build_zero_overlap_drift_guard_payload(
        train_df=train,
        test_df=test,
        target_col="target",
        id_col="id",
    )

    assert payload["enabled"] is True
    assert payload["reason"] == "zero_overlap_high_drift_detected"
    assert payload["drop_columns"] == ["risk_cat"]
    assert payload["stats"] == {
        "categorical_checked": 2,
        "zero_overlap_checked": 1,
        "zero_overlap_ratio": 0.5,
    }


def test_build_zero_overlap_drift_guard_payload_handles_missing_target() -> None:
    train = pd.DataFrame({"feature": ["a", "b"]})
    test = pd.DataFrame({"feature": ["c", "d"]})

    payload = build_zero_overlap_drift_guard_payload(
        train_df=train,
        test_df=test,
        target_col=None,
        id_col=None,
    )

    assert payload["enabled"] is False
    assert payload["reason"] == "missing_target_column"
    assert payload["drop_columns"] == []


def test_prepare_zero_overlap_drift_guard_respects_disabled_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD", "0")
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id\n2\n", encoding="utf-8")
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir()

    assert prepare_zero_overlap_drift_guard(base_dir=tmp_path, slug="demo", context_dir=context_dir) is None
    assert not (context_dir / "zero_overlap_drift_guard.json").exists()


def test_prepare_zero_overlap_drift_guard_writes_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD", raising=False)
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text(
        "id,risk_cat,target\nA,x,1\nB,x,1\nC,y,0\nD,y,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,risk_cat\nT1,u\nT2,v\n",
        encoding="utf-8",
    )
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir()
    (context_dir / "dataset_profile.json").write_text(
        json.dumps({"target_column": "target", "id_column": "id"}),
        encoding="utf-8",
    )

    path = prepare_zero_overlap_drift_guard(base_dir=tmp_path, slug="demo", context_dir=context_dir)

    assert path == context_dir / "zero_overlap_drift_guard.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["target_column"] == "target"
    assert payload["id_column"] == "id"
    first_bytes = path.read_bytes()

    repeated_path = prepare_zero_overlap_drift_guard(base_dir=tmp_path, slug="demo", context_dir=context_dir)

    assert repeated_path == path
    assert path.read_bytes() == first_bytes
    assert "generated_at_epoch" not in payload


def test_prepare_zero_overlap_drift_guard_reads_tsv_train_test(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD", raising=False)
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.tsv").write_text(
        "id\trisk_cat\ttarget\nA\tx\t1\nB\tx\t1\nC\ty\t0\nD\ty\t0\n",
        encoding="utf-8",
    )
    (data_dir / "test.tsv").write_text(
        "id\trisk_cat\nT1\tu\nT2\tv\n",
        encoding="utf-8",
    )
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir()
    (context_dir / "dataset_profile.json").write_text(
        json.dumps({"target_column": "target", "id_column": "id"}),
        encoding="utf-8",
    )

    path = prepare_zero_overlap_drift_guard(base_dir=tmp_path, slug="demo", context_dir=context_dir)

    assert path == context_dir / "zero_overlap_drift_guard.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["drop_columns"] == ["risk_cat"]


def test_prepare_zero_overlap_drift_guard_reads_compressed_and_excel_train_test(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD", raising=False)
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    with gzip.open(data_dir / "train.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,risk_cat,target\nA,x,1\nB,x,1\nC,y,0\nD,y,0\n")
    pd.DataFrame({"id": ["T1", "T2"], "risk_cat": ["u", "v"]}).to_excel(
        data_dir / "test.xlsx",
        index=False,
    )
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir()
    (context_dir / "dataset_profile.json").write_text(
        json.dumps({"target_column": "target", "id_column": "id"}),
        encoding="utf-8",
    )

    path = prepare_zero_overlap_drift_guard(base_dir=tmp_path, slug="demo", context_dir=context_dir)

    assert path == context_dir / "zero_overlap_drift_guard.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["drop_columns"] == ["risk_cat"]


def test_prepare_zero_overlap_drift_guard_reads_role_alias_train_test_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD", raising=False)
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "risk_cat": ["x", "x", "y", "y"],
            "target": [1, 1, 0, 0],
        }
    ).to_parquet(data_dir / "TrainingSet.parquet", index=False)
    pd.DataFrame({"id": ["T1", "T2"], "risk_cat": ["u", "v"]}).to_json(
        data_dir / "PublicTest.jsonl",
        orient="records",
        lines=True,
    )
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir()
    (context_dir / "dataset_profile.json").write_text(
        json.dumps({"target_column": "target", "id_column": "id"}),
        encoding="utf-8",
    )

    path = prepare_zero_overlap_drift_guard(base_dir=tmp_path, slug="demo", context_dir=context_dir)

    assert path == context_dir / "zero_overlap_drift_guard.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["drop_columns"] == ["risk_cat"]
