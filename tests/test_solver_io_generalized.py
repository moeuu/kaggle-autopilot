from __future__ import annotations

import bz2
import gzip
import io
import json
import sqlite3
import tarfile
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyreadr
import pyreadstat
import pytest
import zstandard as zstd
from scipy.io import savemat

from kagglebot.solver.io import (
    _finalize_table_frame,
    _resolve_label_join_columns,
    _synthesize_train_test_from_assets,
    ensure_sample_submission,
    find_competition_files,
    infer_submission_layout,
    load_competition_data,
    read_table,
    write_submission,
    write_table,
)


def _dense_sparse_columns(frame: pd.DataFrame) -> pd.DataFrame:
    dense = frame.copy()
    for column in dense.columns:
        if isinstance(dense[column].dtype, pd.SparseDtype):
            dense[column] = dense[column].sparse.to_dense()
    return dense


def _write_zip_text(path: Path, member: str, text: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, text)


def _write_zip_parquet(path: Path, member: str, frame: pd.DataFrame) -> None:
    payload = io.BytesIO()
    frame.to_parquet(payload, index=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload.getvalue())


def _write_zip_excel(path: Path, member: str, frame: pd.DataFrame) -> None:
    payload = io.BytesIO()
    frame.to_excel(payload, index=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload.getvalue())


def _write_zip_stata(path: Path, member: str, frame: pd.DataFrame) -> None:
    payload = io.BytesIO()
    frame.to_stata(payload, write_index=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload.getvalue())


def _write_netcdf_table(path: Path, columns: dict[str, np.ndarray]) -> None:
    from scipy.io import netcdf_file

    row_count = len(next(iter(columns.values())))
    with netcdf_file(path, mode="w") as dataset:
        dataset.createDimension("row", row_count)
        for name, values in columns.items():
            array = np.asarray(values)
            if array.ndim == 1:
                variable = dataset.createVariable(name, array.dtype.char, ("row",))
                variable[:] = array
                continue
            dataset.createDimension(f"{name}_width", array.shape[1])
            variable = dataset.createVariable(name, array.dtype.char, ("row", f"{name}_width"))
            variable[:] = array


def _write_fits_table(path: Path, columns: dict[str, np.ndarray]) -> None:
    from astropy.io import fits

    fits_columns = []
    for name, values in columns.items():
        array = np.asarray(values)
        if array.dtype.kind in {"f", "c"}:
            fmt = "D"
        elif array.dtype.kind in {"i", "u", "b"}:
            fmt = "K"
        else:
            width = max(len(str(value)) for value in array.tolist()) if array.size else 1
            fmt = f"{max(width, 1)}A"
            array = array.astype(f"S{max(width, 1)}")
        fits_columns.append(fits.Column(name=name, array=array, format=fmt))
    fits.BinTableHDU.from_columns(fits_columns).writeto(path, overwrite=True)


def _write_h5ad_table(path: Path, *, ids: np.ndarray, features: np.ndarray, target: np.ndarray | None = None) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset("id", data=ids)
        if target is not None:
            obs.create_dataset("target", data=target)
        cell_type = obs.create_group("cell_type")
        cell_type.create_dataset("codes", data=np.arange(len(ids)) % 2)
        cell_type.create_dataset("categories", data=np.array([b"a", b"b"]))
        var = handle.create_group("var")
        var.create_dataset("_index", data=np.array([f"gene_{idx}".encode() for idx in range(features.shape[1])]))
        handle.create_dataset("X", data=features)


def _write_loom_table(path: Path, *, ids: np.ndarray, features: np.ndarray, target: np.ndarray | None = None) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("matrix", data=features.T)
        col_attrs = handle.create_group("col_attrs")
        col_attrs.create_dataset("id", data=ids)
        if target is not None:
            col_attrs.create_dataset("target", data=target)
        row_attrs = handle.create_group("row_attrs")
        row_attrs.create_dataset("Gene", data=np.array([f"gene_{idx}".encode() for idx in range(features.shape[1])]))


def _write_geopackage_table(
    path: Path,
    rows: list[tuple[int, float, int | None, bytes]],
    *,
    table: str = "train",
) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                identifier TEXT,
                description TEXT,
                last_change TEXT,
                min_x REAL,
                min_y REAL,
                max_x REAL,
                max_y REAL,
                srs_id INTEGER
            )
            """
        )
        conn.execute("CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER)")
        conn.execute(
            "INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES (?, 'attributes', ?, 0)",
            (table, table),
        )
        conn.execute(f'CREATE TABLE "{table}" (id INTEGER, feature REAL, target INTEGER, geom BLOB)')
        conn.executemany(f'INSERT INTO "{table}" VALUES (?, ?, ?, ?)', rows)


def _write_dbf_table(path: Path, rows: list[tuple[int, float, int | None, str]]) -> None:
    fields = [
        ("id", "N", 10, 0),
        ("feature", "N", 18, 4),
        ("target", "N", 10, 0),
        ("zone", "C", 12, 0),
    ]
    header_length = 32 + (32 * len(fields)) + 1
    record_length = 1 + sum(field[2] for field in fields)
    header = bytearray(32)
    header[0] = 3
    header[4:8] = len(rows).to_bytes(4, "little")
    header[8:10] = header_length.to_bytes(2, "little")
    header[10:12] = record_length.to_bytes(2, "little")
    payload = bytearray(header)
    for name, field_type, length, decimals in fields:
        descriptor = bytearray(32)
        descriptor[: len(name)] = name.encode("ascii")
        descriptor[11] = ord(field_type)
        descriptor[16] = length
        descriptor[17] = decimals
        payload.extend(descriptor)
    payload.append(0x0D)
    for row in rows:
        values = {
            "id": "" if row[0] is None else str(row[0]),
            "feature": "" if row[1] is None else f"{row[1]:.4f}",
            "target": "" if row[2] is None else str(row[2]),
            "zone": row[3],
        }
        payload.append(0x20)
        for name, field_type, length, _decimals in fields:
            raw = str(values[name]).encode("ascii")
            payload.extend(raw.rjust(length, b" ") if field_type == "N" else raw.ljust(length, b" "))
    payload.append(0x1A)
    path.write_bytes(bytes(payload))


def _kml_payload(rows: list[tuple[int, float, int | None]]) -> str:
    placemarks = []
    for row_id, feature, target in rows:
        data = [
            f'<Data name="id"><value>{row_id}</value></Data>',
            f'<Data name="feature"><value>{feature}</value></Data>',
        ]
        if target is not None:
            data.append(f'<Data name="target"><value>{target}</value></Data>')
        placemarks.append(
            "<Placemark>"
            f"<name>row-{row_id}</name>"
            f"<ExtendedData>{''.join(data)}</ExtendedData>"
            f"<Point><coordinates>{feature},{row_id},0</coordinates></Point>"
            "</Placemark>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"{''.join(placemarks)}"
        "</Document></kml>"
    )


def _write_kmz(path: Path, kml_text: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("doc.kml", kml_text)


def test_infer_submission_layout_supports_multi_target() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "f1": [10, 20, 30],
            "target_a": [0, 1, 0],
            "target_b": [0.2, 0.4, 0.6],
        }
    )
    test = pd.DataFrame({"id": [4, 5], "f1": [40, 50]})
    sample = pd.DataFrame({"id": [4, 5], "target_a": [0, 0], "target_b": [0.0, 0.0]})

    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)

    assert id_col == "id"
    assert target_cols == ["target_a", "target_b"]
    assert feature_cols == ["f1"]


def test_read_table_flattens_geojson_feature_collection(tmp_path) -> None:
    path = tmp_path / "features.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": "cell_001",
                        "type": "Feature",
                        "properties": {"score": 0.1, "zone": "a"},
                        "geometry": {"type": "Point", "coordinates": [139.0, 35.0]},
                    },
                    {
                        "id": "cell_002",
                        "type": "Feature",
                        "properties": {"score": 0.2, "zone": "b"},
                        "geometry": {"type": "Point", "coordinates": [140.0, 36.0]},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    frame = read_table(path)

    assert list(frame.columns) == ["score", "zone", "id", "geometry"]
    assert frame["id"].tolist() == ["cell_001", "cell_002"]
    assert frame["score"].tolist() == [0.1, 0.2]
    assert '"Point"' in frame["geometry"].iloc[0]


def test_read_table_accepts_nrows_for_text_tables(tmp_path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("id;feature;target\n1;10;0\n2;20;1\n3;30;0\n", encoding="utf-8")

    frame = read_table(path, nrows=2)

    assert frame.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}


def test_read_table_accepts_nrows_for_json_lines_tables(tmp_path) -> None:
    path = tmp_path / "train.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in (
            {"id": 1, "feature": 10, "target": 0},
            {"id": 2, "feature": 20, "target": 1},
            {"id": 3, "feature": 30, "target": 0},
        ):
            handle.write(json.dumps(row) + "\n")

    frame = read_table(path, nrows=2)

    assert frame.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}


def test_read_table_stabilizes_problematic_columns(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            {
                "columns": ["id", "", None, "score", "score"],
                "data": [[1, "a", 10, 0.1, 0.2], [2, "b", 20, 0.3, 0.4]],
            }
        ),
        encoding="utf-8",
    )

    full = read_table(path)
    head = read_table(path, nrows=1)

    assert list(full.columns) == ["id", "column_2", "column_3", "score", "score_1"]
    assert list(head.columns) == ["id", "column_2", "column_3", "score", "score_1"]
    assert head.to_dict("records") == [{"id": 1, "column_2": "a", "column_3": 10, "score": 0.1, "score_1": 0.2}]

    csv_path = tmp_path / "blank_header.csv"
    csv_path.write_text("id,,target\n1,abc,0\n", encoding="utf-8")
    csv_frame = read_table(csv_path)

    assert list(csv_frame.columns) == ["id", "column_2", "target"]


def test_write_table_stabilizes_problematic_columns_without_mutating_input(tmp_path) -> None:
    frame = pd.DataFrame([[1, "a", 0.1, 0.2]], columns=["id", "", "score", "score"])
    path = tmp_path / "submission.csv"

    written = write_table(frame, path)
    loaded = read_table(written)

    assert list(loaded.columns) == ["id", "column_2", "score", "score_1"]
    assert list(frame.columns) == ["id", "", "score", "score"]


def test_table_column_finalize_flattens_multiindex_columns() -> None:
    frame = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("fold", "a"), ("Unnamed: 1_level_0", None)]),
    )

    normalized = _finalize_table_frame(frame)

    assert list(normalized.columns) == ["fold_a", "column_2"]


def test_infer_submission_layout_supports_survival_event_time_targets() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "f1": [10, 20, 30],
            "efs": [1, 0, 1],
            "efs_time": [5.0, 8.0, 2.0],
        }
    )
    test = pd.DataFrame({"id": [4, 5], "f1": [40, 50]})
    sample = pd.DataFrame({"id": [4, 5], "prediction": [0.0, 0.0]})

    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)

    assert id_col == "id"
    assert target_cols == ["efs", "efs_time"]
    assert feature_cols == ["f1"]


def test_infer_submission_layout_supports_single_label_probability_columns() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "f1": [10, 20, 30],
            "label": ["cat", "dog", "bird"],
        }
    )
    test = pd.DataFrame({"id": [4, 5], "f1": [40, 50]})
    sample = pd.DataFrame(
        {
            "id": [4, 5],
            "class_bird": [1 / 3, 1 / 3],
            "class_cat": [1 / 3, 1 / 3],
            "class_dog": [1 / 3, 1 / 3],
        }
    )

    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)

    assert id_col == "id"
    assert target_cols == ["label"]
    assert feature_cols == ["f1"]


def test_load_competition_data_marks_probability_column_submission(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "f1": [10, 20, 30], "label": ["cat", "dog", "bird"]}).to_csv(
        data_dir / "train.csv",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "f1": [40, 50]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame(
        {
            "id": [4, 5],
            "class_bird": [1 / 3, 1 / 3],
            "class_cat": [1 / 3, 1 / 3],
            "class_dog": [1 / 3, 1 / 3],
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "label"
    assert data.target_columns == ["label"]
    assert data.task == "classification"
    assert data.prediction_kind == "probability_columns"


def test_load_competition_data_marks_suffix_probability_column_submission(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "f1": [10, 20, 30], "label": ["cat", "dog", "bird"]}).to_csv(
        data_dir / "train.csv",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "f1": [40, 50]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame(
        {
            "id": [4, 5],
            "dog_probability": [1 / 3, 1 / 3],
            "cat_probability": [1 / 3, 1 / 3],
            "bird_probability": [1 / 3, 1 / 3],
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "label"
    assert data.target_columns == ["label"]
    assert data.task == "classification"
    assert data.prediction_kind == "probability_columns"


def test_load_competition_data_detects_future_time_column(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(6),
            "store_id": ["a", "a", "b", "b", "c", "c"],
            "date_block_num": [0, 1, 2, 3, 4, 5],
            "feature": [10, 11, 12, 13, 14, 15],
            "target": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "store_id": ["a", "b"],
            "date_block_num": [6, 7],
            "feature": [16, 17],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.time_column == "date_block_num"


def test_load_competition_data_marks_probability_named_binary_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "isFraud": [0, 1] * 6,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [10.0, 11.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "isFraud": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.task_by_target == {"isFraud": "classification"}
    assert data.prediction_kind_by_target == {"isFraud": "probability"}


def test_load_competition_data_preserves_leading_zero_ids(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(
        "id,feature,target\n001,10,0\n002,20,1\n003,30,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n004,40\n005,50\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n004,0\n005,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.id_column == "id"
    assert data.test["id"].tolist() == ["004", "005"]
    assert data.sample["id"].tolist() == ["004", "005"]


def test_load_competition_data_aligns_train_test_column_case_to_sample(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("ID,feature,Target\n1,10,0\n2,20,1\n3,30,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("ID,feature\n4,40\n5,50\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.target_columns == ["target"]
    assert data.feature_columns == ["feature"]
    assert list(data.train.columns) == ["id", "feature", "target"]
    assert list(data.test.columns) == ["id", "feature"]


def test_load_competition_data_aligns_feature_column_case_between_train_test(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("id,Feature,target\n1,10,0\n2,20,1\n3,30,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n4,40\n5,50\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.feature_columns == ["Feature"]
    assert list(data.train.columns) == ["id", "Feature", "target"]
    assert list(data.test.columns) == ["id", "Feature"]


def test_read_table_sniffs_semicolon_csv(tmp_path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("id;feature;target\n1;10;0\n2;20;1\n", encoding="utf-8")

    frame = read_table(path)

    assert list(frame.columns) == ["id", "feature", "target"]
    assert frame["feature"].tolist() == [10, 20]


def test_read_table_sniffs_space_delimited_dat(tmp_path) -> None:
    path = tmp_path / "train.dat"
    path.write_text("id feature target\n1 10 0\n2 20 1\n", encoding="utf-8")

    frame = read_table(path)

    assert list(frame.columns) == ["id", "feature", "target"]
    assert frame["feature"].tolist() == [10, 20]


def test_read_table_supports_fixed_width_inputs(tmp_path) -> None:
    path = tmp_path / "train.fwf"
    path.write_text("id feature target\n1  10      0\n2  20      1\n", encoding="utf-8")

    frame = read_table(path)

    assert list(frame.columns) == ["id", "feature", "target"]
    assert frame["feature"].tolist() == [10, 20]


def test_find_competition_files_prefers_feature_pair_over_train_labels(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train_labels.csv").write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    (data_dir / "train_features.csv").write_text("id,f1,f2\n1,10,100\n2,20,200\n", encoding="utf-8")
    (data_dir / "test_features.csv").write_text("id,f1,f2\n3,30,300\n4,40,400\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n3,0\n4,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    assert train_path.name == "train_features.csv"
    assert test_path.name == "test_features.csv"


def test_train_test_schema_scoring_disables_text_type_inference(tmp_path, monkeypatch) -> None:
    import kagglebot.solver.io as solver_io

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    for path in (train_path, test_path, sample_path):
        path.write_text("row_id,feature,target\n001,1,0\n", encoding="utf-8")

    calls: list[tuple[Path, object, int | None]] = []

    def fake_read_text(path, *, sep=None, dtype=None, nrows=None):  # noqa: ANN001, ARG001
        calls.append((path, dtype, nrows))
        return pd.DataFrame({"row_id": ["001"], "feature": ["1"], "target": ["0"]})

    monkeypatch.setattr(solver_io, "_read_text_tabular_frame", fake_read_text)

    score = solver_io._train_test_pair_score(
        train_path=train_path,
        test_path=test_path,
        sample_path=sample_path,
    )

    assert score > 0
    assert calls == [
        (train_path, str, 5),
        (test_path, str, 5),
        (sample_path, str, 5),
    ]


def test_load_competition_data_merges_separate_train_labels(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train_labels.csv").write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    (data_dir / "train_features.csv").write_text("id,f1,f2\n1,10,100\n2,20,200\n", encoding="utf-8")
    (data_dir / "test_features.csv").write_text("id,f1,f2\n3,30,300\n4,40,400\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n3,0\n4,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.id_column == "id"
    assert data.feature_columns == ["f1", "f2"]
    assert data.train["target"].tolist() == [0, 1]


def test_resolve_label_join_columns_ignores_idless_sample_prediction_column() -> None:
    train = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "f1": [10, 20]})
    test = pd.DataFrame({"row_id": [3, 4], "target": [0, 0], "f1": [30, 40]})
    sample = pd.DataFrame({"target": [0, 0]})
    labels = pd.DataFrame({"row_id": [1, 2], "target": [1, 0]})

    assert _resolve_label_join_columns(train=train, test=test, sample=sample, labels=labels) == ("row_id", "row_id")


def test_load_competition_data_merges_ground_truth_labels(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ground_truth.csv").write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    (data_dir / "train_features.csv").write_text("id,f1,f2\n1,10,100\n2,20,200\n", encoding="utf-8")
    (data_dir / "test_features.csv").write_text("id,f1,f2\n3,30,300\n4,40,400\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n3,0\n4,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.id_column == "id"
    assert data.feature_columns == ["f1", "f2"]
    assert data.train["target"].tolist() == [0, 1]


def test_load_competition_data_merges_labels_case_insensitively(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ground_truth.csv").write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    (data_dir / "train_features.csv").write_text("ID,f1,f2\n1,10,100\n2,20,200\n", encoding="utf-8")
    (data_dir / "test_features.csv").write_text("ID,f1,f2\n3,30,300\n4,40,400\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("ID,Target\n3,0\n4,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.target_column == "Target"
    assert data.id_column == "ID"
    assert data.feature_columns == ["f1", "f2"]
    assert data.train["Target"].tolist() == [0, 1]


def test_load_competition_data_merges_labels_across_join_dtypes(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ground_truth.csv").write_text("id,target\n1,0\n2,1\nextra,0\n", encoding="utf-8")
    (data_dir / "train_features.csv").write_text("id,f1,f2\n1,10,100\n2,20,200\n", encoding="utf-8")
    (data_dir / "test_features.csv").write_text("id,f1,f2\n3,30,300\n4,40,400\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n3,0\n4,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.id_column == "id"
    assert data.feature_columns == ["f1", "f2"]
    assert data.train["target"].tolist() == [0, 1]


def test_load_competition_data_merges_labels_with_id_alias_columns(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ground_truth.csv").write_text("filename,target\nimg_1,0\nimg_2,1\n", encoding="utf-8")
    (data_dir / "train_features.csv").write_text("image_id,f1,f2\nimg_1,10,100\nimg_2,20,200\n", encoding="utf-8")
    (data_dir / "test_features.csv").write_text("image_id,f1,f2\nimg_3,30,300\nimg_4,40,400\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("image_id,target\nimg_3,0\nimg_4,0\n", encoding="utf-8")

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.id_column == "image_id"
    assert data.feature_columns == ["f1", "f2"]
    assert data.train["target"].tolist() == [0, 1]


def test_infer_submission_layout_excludes_train_only_columns() -> None:
    train = pd.DataFrame(
        {
            "oare_id": ["x1", "x2", "x3"],
            "transliteration": ["a", "b", "c"],
            "translation": ["A", "B", "C"],
        }
    )
    test = pd.DataFrame({"id": [10, 11], "transliteration": ["d", "e"]})
    sample = pd.DataFrame({"id": [10, 11], "translation": ["", ""]})

    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)

    assert id_col == "id"
    assert target_cols == ["translation"]
    assert feature_cols == ["transliteration"]


def test_load_competition_data_marks_text_submission_kind(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "prompt": ["a", "b", "c"],
            "translation": ["alpha one", "beta two", "gamma three"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [4, 5], "prompt": ["d", "e"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [4, 5], "translation": ["", ""]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "translation"
    assert data.task == "text"
    assert data.task_by_target == {"translation": "text"}
    assert data.prediction_kind == "text"


def test_load_competition_data_marks_natural_language_target_as_text(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["a", "b", "c", "d"],
            "target": [
                "write a concise answer about red apples",
                "write a concise answer about blue oceans",
                "write a concise answer about green forests",
                "write a concise answer about yellow flowers",
            ],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "prompt": ["b", "c"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "target": ["placeholder", "placeholder"]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.task == "text"
    assert data.task_by_target == {"target": "text"}
    assert data.prediction_kind == "text"


def test_load_competition_data_marks_delimited_multi_label_target_as_text(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["red", "blue", "green", "yellow"],
            "labels": ["cat dog", "dog bird", "cat bird", "cat dog bird"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "prompt": ["blue", "green"]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "labels": ["cat dog", "cat dog"]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "labels"
    assert data.task == "text"
    assert data.task_by_target == {"labels": "text"}
    assert data.prediction_kind == "text"


def test_load_competition_data_marks_quantile_submission_kind(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx * 2) for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [20.0, 21.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "p10": [0.0, 0.0], "p50": [0.0, 0.0], "p90": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.task == "regression"
    assert data.prediction_kind == "quantile_columns"


def test_load_competition_data_marks_prediction_interval_submission_kind(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx * 2) for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [20.0, 21.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "lower": [0.0, 0.0], "upper": [1.0, 1.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.task == "regression"
    assert data.prediction_kind == "prediction_interval_columns"


def test_load_competition_data_handles_unlabeled_anomaly_score_submission(tmp_path) -> None:
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

    data = load_competition_data(data_dir)

    assert data.target_column == "anomaly_score"
    assert data.target_columns == ["anomaly_score"]
    assert data.task == "unsupervised"
    assert data.task_by_target == {"anomaly_score": "unsupervised"}
    assert data.prediction_kind == "probability"


def test_load_competition_data_handles_unlabeled_anomaly_score_submission_without_id(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "amount": [10.5, 11.2, 980.0],
            "velocity": [0.1, 0.2, 9.8],
            "country": ["JP", "US", "BR"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "amount": [12.0, 1200.0],
            "velocity": [0.1, 12.5],
            "country": ["JP", "US"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"anomaly_score": [0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    data = load_competition_data(data_dir)

    assert data.id_column is None
    assert data.target_column == "anomaly_score"
    assert data.target_columns == ["anomaly_score"]
    assert data.feature_columns == ["amount", "velocity", "country"]
    assert data.task == "unsupervised"
    assert data.prediction_kind == "probability"


def test_load_competition_data_marks_learning_to_rank_target_continuous(tmp_path) -> None:
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

    data = load_competition_data(data_dir)

    assert data.target_column == "relevance"
    assert data.task == "regression"
    assert data.task_by_target == {"relevance": "regression"}
    assert data.prediction_kind == "continuous"


def test_load_competition_data_marks_ordinal_target(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(1, 11),
            "feature": [float(idx) for idx in range(10)],
            "severity": [0, 1, 2, 3, 4, 2, 1, 3, 4, 0],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [11, 12], "feature": [10.0, 11.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [11, 12], "severity": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "severity"
    assert data.task == "regression"
    assert data.task_by_target == {"severity": "regression"}
    assert data.prediction_kind == "ordinal"


def test_load_competition_data_treats_named_numeric_target_as_regression(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "sales": [0, 1, 2, 3] * 3,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [10.0, 11.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "sales": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.task_by_target == {"sales": "regression"}
    assert data.prediction_kind_by_target == {"sales": "continuous"}


def test_load_competition_data_treats_count_target_as_continuous(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "count": [idx for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "count": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "count"
    assert data.task_by_target == {"count": "regression"}
    assert data.prediction_kind_by_target == {"count": "continuous"}


def test_load_competition_data_treats_bounded_rate_target_as_continuous(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "conversion_rate": [0.0, 0.25, 0.5, 0.75] * 3,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [12.0, 13.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "conversion_rate": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "conversion_rate"
    assert data.task_by_target == {"conversion_rate": "regression"}
    assert data.prediction_kind_by_target == {"conversion_rate": "continuous"}


def test_load_competition_data_marks_multi_label_column_submission(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(8),
            "feature": [float(idx) for idx in range(8)],
            "labels": [
                "cat dog",
                "dog bird",
                "cat bird",
                "cat dog",
                "dog bird",
                "cat bird",
                "cat dog",
                "dog bird",
            ],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "cat": [0.0, 0.0], "dog": [0.0, 0.0], "bird": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_columns == ["labels"]
    assert data.task_by_target == {"labels": "classification"}
    assert data.prediction_kind_by_target == {"labels": "multi_label_columns"}


def test_write_submission_multitarget_aligns_by_id(tmp_path) -> None:
    sample = pd.DataFrame({"id": [11, 12], "target_a": [0, 0], "target_b": [0.0, 0.0]})
    test = pd.DataFrame({"id": [12, 11], "f": [1, 2]})
    preds = {"target_a": np.array([1, 0]), "target_b": np.array([0.9, 0.1])}

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="id",
        target_columns=["target_a", "target_b"],
        output_path=tmp_path / "submission.csv",
    )
    frame = pd.read_csv(out)
    assert frame["target_a"].tolist() == [0, 1]
    assert frame["target_b"].tolist() == [0.1, 0.9]


def test_write_submission_aligns_id_values_across_string_numeric_dtypes(tmp_path) -> None:
    sample = pd.DataFrame({"id": ["11", "12"], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [12, 11], "f": [1, 2]})
    preds = np.array([0.9, 0.1])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="id",
        target_column="target",
        output_path=tmp_path / "submission.csv",
    )

    frame = pd.read_csv(out, dtype={"id": str})
    assert frame["id"].tolist() == ["11", "12"]
    assert frame["target"].tolist() == [0.1, 0.9]


def test_write_submission_without_id_uses_row_order(tmp_path) -> None:
    sample = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    test = pd.DataFrame({"f": [1, 2, 3]})
    preds = np.array([0.2, 0.4, 0.6])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column=None,
        target_column="prediction",
        output_path=tmp_path / "submission.csv",
    )
    frame = pd.read_csv(out)
    assert frame["prediction"].tolist() == [0.2, 0.4, 0.6]


def test_write_submission_without_id_expands_tiny_sample_to_test_rows(tmp_path) -> None:
    sample = pd.DataFrame({"target": [0.0, 0.0]})
    test = pd.DataFrame({"feature_1": [0.1, 0.2, 0.3]})
    preds = np.array([0.1, 0.2, 0.3])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column=None,
        target_column="target",
        output_path=tmp_path / "submission.csv",
    )

    frame = pd.read_csv(out)
    assert list(frame.columns) == ["target"]
    assert frame["target"].tolist() == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("suffix", [".xls", ".xlsm", ".xlsx", ".ods"])
def test_write_table_writes_excel_from_suffix(tmp_path, suffix: str) -> None:
    path = tmp_path / f"artifact{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)

    assert written == path
    assert pd.read_excel(path).to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_uses_pyxlsb_engine_for_xlsb_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.xlsb"
    path.write_bytes(b"xlsb")

    def fake_read_excel(read_path: Path, *args, **kwargs) -> pd.DataFrame:
        assert Path(read_path) == path
        assert kwargs["engine"] == "pyxlsb"
        return pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    frame = read_table(path)

    assert frame.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_supports_svmlight_sparse_inputs(tmp_path) -> None:
    path = tmp_path / "train.svmlight"
    path.write_text("1 1:0.5 3:1.5\n0 2:2.5\n", encoding="utf-8")

    frame = _dense_sparse_columns(read_table(path))

    assert frame.to_dict("list") == {
        "target": [1.0, 0.0],
        "feature_1": [0.5, 0.0],
        "feature_2": [0.0, 2.5],
        "feature_3": [1.5, 0.0],
    }


def test_read_and_write_table_supports_zstd_compressed_csv(tmp_path) -> None:
    path = tmp_path / "artifact.csv.zst"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_write_table_uses_tabular_fallback_for_non_tabular_artifact_suffix(tmp_path) -> None:
    requested = tmp_path / "answers.nii.gz"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, requested)

    assert written == tmp_path / "answers.tabular.csv"
    assert not requested.exists()
    assert pd.read_csv(written).to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "tabular"
    assert manifest["submission_path"] == "answers.tabular.csv"
    assert manifest["requested_output_path"] == "answers.nii.gz"


@pytest.mark.parametrize(
    ("submission_filename", "expected_name"),
    [
        ("submission.tsv", "answers.tabular.tsv"),
        ("submission.jsonl.zst", "answers.tabular.jsonl.zst"),
    ],
)
def test_write_table_non_tabular_fallback_uses_configured_tabular_suffix(
    tmp_path, monkeypatch, submission_filename: str, expected_name: str
) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", submission_filename)
    requested = tmp_path / "answers.nii.gz"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, requested)

    assert written == tmp_path / expected_name
    assert not requested.exists()
    assert read_table(written).to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == expected_name
    assert manifest["requested_output_path"] == "answers.nii.gz"


def test_write_table_non_tabular_fallback_ignores_configured_non_tabular_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.nii.gz")
    requested = tmp_path / "answers.zarr"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, requested)

    assert written == tmp_path / "answers.tabular.csv"
    assert pd.read_csv(written).to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "answers.tabular.csv"
    assert manifest["requested_output_path"] == "answers.zarr"


def test_write_table_non_tabular_fallback_uses_sample_submission_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.nii.gz")
    monkeypatch.setenv("KAGGLEBOT_SAMPLE_SUBMISSION_PATH", "/kaggle/input/demo/sample_submission.tsv.zst")
    requested = tmp_path / "answers.nii.gz"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, requested)

    assert written == tmp_path / "answers.tabular.tsv.zst"
    assert read_table(written).to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "answers.tabular.tsv.zst"
    assert manifest["requested_output_path"] == "answers.nii.gz"


def test_write_table_non_tabular_fallback_uses_requested_stem_to_avoid_collisions(tmp_path) -> None:
    frame = pd.DataFrame({"id": [1], "target": [0.1]})

    first = write_table(frame, tmp_path / "mask.nii.gz")
    second = write_table(frame, tmp_path / "scores.zarr")

    assert first == tmp_path / "mask.tabular.csv"
    assert second == tmp_path / "scores.tabular.csv"
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "scores.tabular.csv"
    assert manifest["requested_output_path"] == "scores.zarr"


def test_read_and_write_table_supports_compressed_tsv(tmp_path) -> None:
    path = tmp_path / "artifact.tsv.gz"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_and_write_table_supports_structured_json_suffixes(tmp_path) -> None:
    suffixes = [
        ".json",
        ".json.gz",
        ".json.zst",
        ".jsonl",
        ".jsonl.bz2",
        ".jsonl.gz",
        ".jsonl.xz",
        ".jsonl.zst",
        ".jsonlines",
        ".jsonlines.bz2",
        ".jsonlines.gz",
        ".jsonlines.xz",
        ".jsonlines.zst",
        ".ndjson",
        ".ndjson.bz2",
        ".ndjson.gz",
        ".ndjson.xz",
        ".ndjson.zst",
    ]
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})
    for suffix in suffixes:
        path = tmp_path / f"artifact{suffix}"

        written = write_table(frame, path)
        loaded = read_table(written)

        assert written == path
        assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_and_write_table_supports_yaml_suffixes(tmp_path) -> None:
    pytest.importorskip("yaml")
    suffixes = [".yaml", ".yml", ".yaml.bz2", ".yaml.gz", ".yaml.xz", ".yml.zst"]
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})
    for suffix in suffixes:
        path = tmp_path / f"artifact{suffix}"

        written = write_table(frame, path)
        loaded = read_table(written)

        assert written == path
        assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_supports_wrapped_json_records(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {"id": 1, "feature": 10, "target": 0},
                    {"id": 2, "feature": 20, "target": 1},
                ],
                "metadata": {"source": "wrapped"},
            }
        ),
        encoding="utf-8",
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".pkl.gz", ".pkl.zst"])
def test_read_and_write_table_supports_pickle(tmp_path, suffix: str) -> None:
    path = tmp_path / f"artifact{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_and_write_table_supports_stata(tmp_path) -> None:
    path = tmp_path / "artifact.dta"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_supports_single_member_zip_stata(tmp_path: Path) -> None:
    path = tmp_path / "artifact.dta.zip"
    _write_zip_stata(path, "nested/artifact.dta", pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}))

    loaded = read_table(path)
    head = read_table(path, nrows=1)

    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    assert head.to_dict("list") == {"id": [1], "target": [0.1]}


@pytest.mark.parametrize(
    ("suffix", "reader_name"),
    [
        (".sas7bdat", "read_sas"),
        (".sav", "read_spss"),
    ],
)
def test_read_table_passes_zip_wrapped_sas_spss_payload_to_pandas_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    reader_name: str,
) -> None:
    path = tmp_path / f"artifact{suffix}.zip"
    payload = b"binary-table"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"nested/artifact{suffix}", payload)

    def fake_reader(read_path, *args, **kwargs):  # noqa: ANN001
        del args
        if reader_name == "read_sas":
            assert kwargs == {"format": "sas7bdat"}
        else:
            assert kwargs == {}
        assert read_path.read() == payload
        return pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    monkeypatch.setattr(pd, reader_name, fake_reader)

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_and_write_table_supports_xml(tmp_path) -> None:
    suffixes = [".xml", ".xml.bz2", ".xml.gz", ".xml.xz", ".xml.zst"]
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})
    for suffix in suffixes:
        path = tmp_path / f"artifact{suffix}"

        written = write_table(frame, path)
        loaded = read_table(written)

        assert written == path
        assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_and_write_table_supports_html(tmp_path) -> None:
    suffixes = [".html", ".htm", ".html.bz2", ".html.gz", ".html.xz", ".html.zst"]
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})
    for suffix in suffixes:
        path = tmp_path / f"artifact{suffix}"

        written = write_table(frame, path)
        loaded = read_table(written)

        assert written == path
        assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_preserves_header_only_html_schema(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.html"
    pd.DataFrame(columns=["id", "target"]).to_html(path, index=False)

    loaded = read_table(path)

    assert loaded.empty
    assert loaded.columns.tolist() == ["id", "target"]


def test_read_table_supports_single_member_zip_csv(tmp_path: Path) -> None:
    path = tmp_path / "artifact.csv.zip"
    _write_zip_text(path, "artifact.csv", "id,target\n1,0.1\n2,0.2\n")

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_supports_single_member_zip_parquet(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet.zip"
    _write_zip_parquet(path, "nested/artifact.parquet", pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}))

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_supports_single_member_zip_excel(tmp_path: Path) -> None:
    path = tmp_path / "artifact.xlsx.zip"
    _write_zip_excel(path, "nested/artifact.xlsx", pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}))

    loaded = read_table(path)
    head = read_table(path, nrows=1)

    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    assert head.to_dict("list") == {"id": [1], "target": [0.1]}


def test_find_competition_files_supports_zip_wrapped_csv_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_zip_text(data_dir / "train.csv.zip", "train.csv", "id,feature,target\n1,10,0\n2,20,1\n")
    _write_zip_text(data_dir / "test.csv.zip", "test.csv", "id,feature\n3,30\n4,40\n")
    _write_zip_text(data_dir / "sample_submission.csv.zip", "sample_submission.csv", "id,target\n3,0\n4,0\n")

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.csv.zip"
    assert test_path.name == "test.csv.zip"
    assert sample_path.name == "sample_submission.csv.zip"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [3, 4], "feature": [30, 40]}
    assert sample.to_dict("list") == {"id": [3, 4], "target": [0, 0]}


def test_find_competition_files_supports_zip_wrapped_parquet_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_zip_parquet(
        data_dir / "train.parquet.zip",
        "train.parquet",
        pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}),
    )
    _write_zip_parquet(
        data_dir / "test.parquet.zip",
        "test.parquet",
        pd.DataFrame({"id": [3, 4], "feature": [30, 40]}),
    )
    _write_zip_parquet(
        data_dir / "sample_submission.parquet.zip",
        "sample_submission.parquet",
        pd.DataFrame({"id": [3, 4], "target": [0, 0]}),
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.parquet.zip"
    assert test_path.name == "test.parquet.zip"
    assert sample_path.name == "sample_submission.parquet.zip"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [3, 4], "feature": [30, 40]}
    assert sample.to_dict("list") == {"id": [3, 4], "target": [0, 0]}


def test_find_competition_files_supports_zip_wrapped_excel_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_zip_excel(
        data_dir / "train.xlsx.zip",
        "train.xlsx",
        pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}),
    )
    _write_zip_excel(
        data_dir / "test.xlsx.zip",
        "test.xlsx",
        pd.DataFrame({"id": [3, 4], "feature": [30, 40]}),
    )
    _write_zip_excel(
        data_dir / "sample_submission.xlsx.zip",
        "sample_submission.xlsx",
        pd.DataFrame({"id": [3, 4], "target": [0, 0]}),
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.xlsx.zip"
    assert test_path.name == "test.xlsx.zip"
    assert sample_path.name == "sample_submission.xlsx.zip"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [3, 4], "feature": [30, 40]}
    assert sample.to_dict("list") == {"id": [3, 4], "target": [0, 0]}


@pytest.mark.parametrize(
    ("suffix", "separator"),
    [
        (".tab", "\t"),
        (".psv", "|"),
        (".tab.gz", "\t"),
        (".psv.zst", "|"),
    ],
)
def test_read_and_write_table_supports_tab_and_pipe_text_suffixes(
    tmp_path: Path,
    suffix: str,
    separator: str,
) -> None:
    path = tmp_path / f"artifact{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    if suffix.endswith(".zst"):
        raw = zstd.ZstdDecompressor().decompress(path.read_bytes()).decode("utf-8")
    elif suffix.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raw = path.read_text(encoding="utf-8")
    assert raw.splitlines()[0] == separator.join(["id", "target"])


@pytest.mark.parametrize("suffix", [".parquet", ".parq", ".pq"])
def test_read_and_write_table_supports_parquet_aliases(tmp_path, suffix: str) -> None:
    path = tmp_path / f"artifact{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


@pytest.mark.parametrize("suffix", [".feather", ".ftr", ".arrow", ".ipc"])
def test_read_and_write_table_supports_arrow_ipc(tmp_path, suffix: str) -> None:
    path = tmp_path / f"artifact{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_and_write_table_supports_avro(tmp_path) -> None:
    path = tmp_path / "artifact.avro"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2], "label": ["a", "b"]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2], "label": ["a", "b"]}


def test_load_competition_data_supports_avro_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_table(
        pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0], "target": [0, 1, 0]}),
        data_dir / "train.avro",
    )
    write_table(pd.DataFrame({"id": [4, 5], "feature": [40.0, 50.0]}), data_dir / "test.avro")
    write_table(pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}), data_dir / "sample_submission.avro")

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.avro"
    assert test_path.name == "test.avro"
    assert sample_path.name == "sample_submission.avro"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.test["id"].tolist() == [4, 5]
    assert data.sample["target"].tolist() == [0.0, 0.0]


def test_read_and_write_table_supports_orc(tmp_path) -> None:
    path = tmp_path / "artifact.orc"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


@pytest.mark.parametrize("suffix", [".h5", ".hdf", ".hdf5"])
def test_read_and_write_table_supports_hdf(tmp_path, suffix: str) -> None:
    path = tmp_path / f"artifact{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    written = write_table(frame, path)
    loaded = read_table(written)

    assert written == path
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_read_table_supports_native_hdf5_dataset_with_column_attrs(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "artifact.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("table", data=np.array([[1.0, 10.0], [2.0, 20.0]]))
        dataset.attrs["columns"] = np.array([b"id", b"feature"])

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1.0, 2.0], "feature": [10.0, 20.0]}


def test_read_table_supports_native_hdf5_group_column_datasets(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "artifact.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("records")
        group.create_dataset("id", data=np.array([1, 2]))
        group.create_dataset("target", data=np.array([b"a", b"b"]))

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "target": ["a", "b"]}


def test_read_table_supports_h5ad_obs_and_x_features(tmp_path) -> None:
    path = tmp_path / "train.h5ad"
    _write_h5ad_table(
        path,
        ids=np.array([1, 2]),
        features=np.array([[10.0, 0.5], [20.0, 0.75]]),
        target=np.array([0, 1]),
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {
        "id": [1, 2],
        "target": [0, 1],
        "cell_type": ["a", "b"],
        "gene_0": [10.0, 20.0],
        "gene_1": [0.5, 0.75],
    }


def test_read_table_supports_loom_col_attrs_and_matrix_features(tmp_path) -> None:
    path = tmp_path / "train.loom"
    _write_loom_table(
        path,
        ids=np.array([1, 2]),
        features=np.array([[10.0, 0.5], [20.0, 0.75]]),
        target=np.array([0, 1]),
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {
        "id": [1, 2],
        "target": [0, 1],
        "gene_0": [10.0, 20.0],
        "gene_1": [0.5, 0.75],
    }


def test_read_table_supports_geopackage_attribute_tables(tmp_path) -> None:
    path = tmp_path / "train.gpkg"
    _write_geopackage_table(
        path,
        [
            (1, 10.0, 0, b"\x47\x50\x00\x01"),
            (2, 20.0, 1, b"\x47\x50\x00\x02"),
        ],
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {
        "id": [1, 2],
        "feature": [10.0, 20.0],
        "target": [0, 1],
        "geom": ["47500001", "47500002"],
    }


def test_read_table_supports_shapefile_attribute_tables(tmp_path) -> None:
    shp_path = tmp_path / "train.shp"
    shp_path.write_bytes(b"")
    _write_dbf_table(
        tmp_path / "train.dbf",
        [
            (1, 10.0, 0, "north"),
            (2, 20.5, 1, "south"),
        ],
    )

    loaded = read_table(shp_path)

    assert loaded.to_dict("list") == {
        "id": [1, 2],
        "feature": [10.0, 20.5],
        "target": [0, 1],
        "zone": ["north", "south"],
    }


def test_read_table_supports_kml_placemark_tables(tmp_path) -> None:
    path = tmp_path / "train.kml"
    path.write_text(_kml_payload([(1, 10.0, 0), (2, 20.5, 1)]), encoding="utf-8")

    loaded = read_table(path)

    assert loaded["id"].tolist() == ["1", "2"]
    assert loaded["feature"].tolist() == ["10.0", "20.5"]
    assert loaded["target"].tolist() == ["0", "1"]
    assert loaded["geometry_type"].tolist() == ["Point", "Point"]
    assert loaded["coordinates"].tolist() == ["10.0,1,0", "20.5,2,0"]


def test_read_table_supports_compressed_kml_placemark_tables(tmp_path) -> None:
    path = tmp_path / "train.kml.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(_kml_payload([(1, 10.0, 0), (2, 20.5, 1)]))

    loaded = read_table(path)

    assert loaded["id"].tolist() == ["1", "2"]
    assert loaded["target"].tolist() == ["0", "1"]


def test_read_table_supports_kmz_placemark_tables(tmp_path) -> None:
    path = tmp_path / "train.kmz"
    _write_kmz(path, _kml_payload([(1, 10.0, 0), (2, 20.5, 1)]))

    loaded = read_table(path)

    assert loaded["id"].tolist() == ["1", "2"]
    assert loaded["target"].tolist() == ["0", "1"]


def test_read_table_supports_netcdf_table_variables(tmp_path) -> None:
    path = tmp_path / "features.nc"
    _write_netcdf_table(
        path,
        {
            "id": np.array([1, 2], dtype=np.int32),
            "features": np.array([[10.0, 0.5], [20.0, 0.75]], dtype=np.float64),
            "target": np.array([0, 1], dtype=np.int32),
        },
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {
        "id": [1, 2],
        "features_0": [10.0, 20.0],
        "features_1": [0.5, 0.75],
        "target": [0, 1],
    }


def test_read_table_supports_cdf_table_variables(tmp_path) -> None:
    path = tmp_path / "features.cdf"
    _write_netcdf_table(
        path,
        {
            "id": np.array([1, 2], dtype=np.int32),
            "feature": np.array([10.0, 20.0], dtype=np.float64),
        },
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "feature": [10.0, 20.0]}


def test_read_table_supports_hdf5_backed_nc4_table_variables(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "features.nc4"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("id", data=np.array([1, 2], dtype=np.int32))
        handle.create_dataset("feature", data=np.array([10.0, 20.0], dtype=np.float64))

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "feature": [10.0, 20.0]}


def test_read_table_supports_fits_binary_tables(tmp_path) -> None:
    path = tmp_path / "features.fits"
    _write_fits_table(
        path,
        {
            "id": np.array([1, 2], dtype=np.int64),
            "feature": np.array([10.0, 20.0], dtype=np.float64),
            "label": np.array(["a", "b"]),
        },
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "feature": [10.0, 20.0], "label": ["a", "b"]}


def test_read_table_supports_gzipped_fits_binary_tables(tmp_path) -> None:
    path = tmp_path / "features.fits.gz"
    _write_fits_table(
        path,
        {
            "id": np.array([1, 2], dtype=np.int64),
            "feature": np.array([10.0, 20.0], dtype=np.float64),
        },
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "feature": [10.0, 20.0]}


def test_read_table_supports_numpy_2d_arrays(tmp_path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.array([[1.0, 10.0], [2.0, 20.0]]))

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"features_0": [1.0, 2.0], "features_1": [10.0, 20.0]}


def test_read_table_supports_numpy_2d_array_column_sidecars(tmp_path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.array([[1.0, 10.0], [2.0, 20.0]]))
    (tmp_path / "features_columns.txt").write_text("id\nfeature\n", encoding="utf-8")

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1.0, 2.0], "feature": [10.0, 20.0]}


def test_read_table_supports_numpy_json_schema_column_sidecars(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(path, values=np.array([[1.0, 10.0], [2.0, 20.0]]))
    (tmp_path / "features.schema.json").write_text(
        json.dumps({"fields": [{"name": "id"}, {"name": "feature"}]}),
        encoding="utf-8",
    )

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1.0, 2.0], "feature": [10.0, 20.0]}


def test_read_table_supports_numpy_archive_column_arrays(tmp_path) -> None:
    path = tmp_path / "train.npz"
    np.savez(path, id=np.array([1, 2]), feature=np.array([10.0, 20.0]), target=np.array([0, 1]))

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "feature": [10.0, 20.0], "target": [0, 1]}


def test_read_table_supports_structured_numpy_arrays(tmp_path) -> None:
    path = tmp_path / "train.npy"
    records = np.array([(1, 10.0, b"a"), (2, 20.0, b"b")], dtype=[("id", "i8"), ("feature", "f8"), ("label", "S1")])
    np.save(path, records)

    loaded = read_table(path)

    assert loaded.to_dict("list") == {"id": [1, 2], "feature": [10.0, 20.0], "label": ["a", "b"]}


def test_find_competition_files_supports_zstd_compressed_csvs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    compressor = zstd.ZstdCompressor()
    for name, text in {
        "train.csv.zst": "id,feature,target\n1,10,0\n2,20,1\n",
        "test.csv.zst": "id,feature\n3,30\n4,40\n",
        "sample_submission.csv.zst": "id,target\n3,0\n4,0\n",
    }.items():
        (data_dir / name).write_bytes(compressor.compress(text.encode("utf-8")))

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.csv.zst"
    assert test_path.name == "test.csv.zst"
    assert sample_path.name == "sample_submission.csv.zst"


def test_find_competition_files_supports_ndjson_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.ndjson").write_text(
        '{"id":1,"feature":10,"target":0}\n{"id":2,"feature":20,"target":1}\n',
        encoding="utf-8",
    )
    (data_dir / "test.ndjson").write_text(
        '{"id":4,"feature":40}\n{"id":5,"feature":50}\n',
        encoding="utf-8",
    )
    (data_dir / "sample_submission.ndjson").write_text(
        '{"id":4,"target":0}\n{"id":5,"target":0}\n',
        encoding="utf-8",
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.ndjson"
    assert test_path.name == "test.ndjson"
    assert sample_path.name == "sample_submission.ndjson"


def test_load_competition_data_supports_wrapped_json_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text(
        json.dumps(
            {
                "data": [
                    {"id": 1, "feature": 10, "target": 0},
                    {"id": 2, "feature": 20, "target": 1},
                    {"id": 3, "feature": 30, "target": 0},
                ],
                "metadata": {"split": "train"},
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "test.json").write_text(
        json.dumps(
            {
                "records": [
                    {"id": 4, "feature": 40},
                    {"id": 5, "feature": 50},
                ],
                "count": 2,
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "sample_submission.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"id": 4, "target": 0},
                    {"id": 5, "target": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    data = load_competition_data(data_dir)

    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.feature_columns == ["feature"]
    assert data.train["target"].tolist() == [0, 1, 0]
    assert data.test["id"].tolist() == [4, 5]


def test_load_competition_data_supports_html_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [10, 20, 30],
            "target": [0, 1, 0],
        }
    ).to_html(data_dir / "train.html", index=False)
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_html(data_dir / "test.html", index=False)
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_html(
        data_dir / "sample_submission.html",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.feature_columns == ["feature"]
    assert data.train["target"].tolist() == [0, 1, 0]
    assert data.test["id"].tolist() == [4, 5]
    assert data.sample["id"].tolist() == [4, 5]


def test_load_competition_data_supports_geojson_train_test_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": f"train_{idx}",
                        "type": "Feature",
                        "properties": {"id": idx, "feature": float(idx), "target": idx % 2},
                        "geometry": {"type": "Point", "coordinates": [float(idx), float(idx + 1)]},
                    }
                    for idx in range(20)
                ],
            },
        ),
        encoding="utf-8",
    )
    (data_dir / "test.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": f"test_{idx}",
                        "type": "Feature",
                        "properties": {"id": 100 + idx, "feature": float(idx)},
                        "geometry": {"type": "Point", "coordinates": [float(idx), float(idx + 2)]},
                    }
                    for idx in range(3)
                ],
            },
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.geojson"
    assert test_path.name == "test.geojson"
    assert sample_path.name == "sample_submission.csv"
    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [float(idx) for idx in range(20)]
    assert data.test["id"].tolist() == [100, 101, 102]
    assert "geometry" in data.train.columns


def test_find_competition_files_supports_pickle_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}).to_pickle(data_dir / "train.pkl")
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_pickle(data_dir / "test.pkl")
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_pickle(data_dir / "sample_submission.pkl")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.pkl"
    assert test_path.name == "test.pkl"
    assert sample_path.name == "sample_submission.pkl"


def test_load_competition_data_supports_yaml_inputs(tmp_path) -> None:
    pytest.importorskip("yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_table(
        pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}),
        data_dir / "train.yaml",
    )
    write_table(pd.DataFrame({"id": [4, 5], "feature": [40, 50]}), data_dir / "test.yaml")
    write_table(pd.DataFrame({"id": [4, 5], "target": [0, 0]}), data_dir / "sample_submission.yaml")

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.yaml"
    assert test_path.name == "test.yaml"
    assert sample_path.name == "sample_submission.yaml"
    assert data.target_column == "target"
    assert data.test["id"].tolist() == [4, 5]


def test_find_competition_files_supports_stata_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}).to_stata(
        data_dir / "train.dta",
        write_index=False,
    )
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_stata(data_dir / "test.dta", write_index=False)
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_stata(
        data_dir / "sample_submission.dta",
        write_index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.dta"
    assert test_path.name == "test.dta"
    assert sample_path.name == "sample_submission.dta"


def test_find_competition_files_supports_zip_wrapped_stata_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_zip_stata(
        data_dir / "train.dta.zip",
        "train.dta",
        pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}),
    )
    _write_zip_stata(
        data_dir / "test.dta.zip",
        "test.dta",
        pd.DataFrame({"id": [4, 5], "feature": [40, 50]}),
    )
    _write_zip_stata(
        data_dir / "sample_submission.dta.zip",
        "sample_submission.dta",
        pd.DataFrame({"id": [4, 5], "target": [0, 0]}),
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.dta.zip"
    assert test_path.name == "test.dta.zip"
    assert sample_path.name == "sample_submission.dta.zip"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


@pytest.mark.parametrize("suffix", [".sas7bdat", ".xpt", ".xport"])
def test_find_competition_files_supports_sas_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frames = {
        f"train{suffix}": pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}),
        f"test{suffix}": pd.DataFrame({"id": [4, 5], "feature": [40, 50]}),
        f"sample_submission{suffix}": pd.DataFrame({"id": [4, 5], "target": [0, 0]}),
    }
    for name in frames:
        (data_dir / name).write_bytes(b"sas")

    def fake_read_sas(path, *args, **kwargs):
        del args, kwargs
        return frames[path.name].copy()

    monkeypatch.setattr(pd, "read_sas", fake_read_sas)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == f"train{suffix}"
    assert test_path.name == f"test{suffix}"
    assert sample_path.name == f"sample_submission{suffix}"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


@pytest.mark.parametrize("suffix", [".sav", ".zsav"])
def test_find_competition_files_supports_spss_inputs(tmp_path, suffix: str) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frames = {
        f"train{suffix}": pd.DataFrame({"id": [1, 2], "feature": [10.0, 20.0], "target": [0.0, 1.0]}),
        f"test{suffix}": pd.DataFrame({"id": [4, 5], "feature": [40.0, 50.0]}),
        f"sample_submission{suffix}": pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}),
    }
    for name, frame in frames.items():
        pyreadstat.write_sav(frame, data_dir / name, compress=suffix == ".zsav")

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == f"train{suffix}"
    assert test_path.name == f"test{suffix}"
    assert sample_path.name == f"sample_submission{suffix}"
    assert train.to_dict("list") == {"id": [1.0, 2.0], "feature": [10.0, 20.0], "target": [0.0, 1.0]}
    assert test.to_dict("list") == {"id": [4.0, 5.0], "feature": [40.0, 50.0]}
    assert sample.to_dict("list") == {"id": [4.0, 5.0], "target": [0.0, 0.0]}


def test_find_competition_files_supports_matlab_column_variable_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    savemat(data_dir / "train.mat", {"id": [1, 2], "feature": [10, 20], "target": [0, 1]})
    savemat(data_dir / "test.mat", {"id": [4, 5], "feature": [40, 50]})
    savemat(data_dir / "sample_submission.mat", {"id": [4, 5], "target": [0, 0]})

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.mat"
    assert test_path.name == "test.mat"
    assert sample_path.name == "sample_submission.mat"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


def test_read_table_supports_matlab_matrix_input(tmp_path) -> None:
    path = tmp_path / "features.mat"
    savemat(path, {"features": [[1, 10], [2, 20]]})

    frame = read_table(path)

    assert frame.to_dict("list") == {"features_0": [1, 2], "features_1": [10, 20]}


def test_find_competition_files_supports_arff_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.arff").write_text(
        """
@RELATION train
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
1,10,no
2,20,yes
""".strip(),
        encoding="utf-8",
    )
    (data_dir / "test.arff").write_text(
        """
@RELATION test
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@DATA
4,40
5,50
""".strip(),
        encoding="utf-8",
    )
    (data_dir / "sample_submission.arff").write_text(
        """
@RELATION sample_submission
@ATTRIBUTE id NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
4,no
5,no
""".strip(),
        encoding="utf-8",
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.arff"
    assert test_path.name == "test.arff"
    assert sample_path.name == "sample_submission.arff"
    assert train.to_dict("list") == {"id": [1.0, 2.0], "feature": [10.0, 20.0], "target": ["no", "yes"]}
    assert test.to_dict("list") == {"id": [4.0, 5.0], "feature": [40.0, 50.0]}
    assert sample.to_dict("list") == {"id": [4.0, 5.0], "target": ["no", "no"]}


def test_find_competition_files_supports_compressed_arff_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payloads = {
        "train.arff.gz": """
@RELATION train
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
1,10,no
2,20,yes
""",
        "test.arff.gz": """
@RELATION test
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@DATA
4,40
5,50
""",
        "sample_submission.arff.gz": """
@RELATION sample_submission
@ATTRIBUTE id NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
4,no
5,no
""",
    }
    for name, payload in payloads.items():
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            handle.write(payload.strip())

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.arff.gz"
    assert test_path.name == "test.arff.gz"
    assert sample_path.name == "sample_submission.arff.gz"
    assert train.to_dict("list") == {"id": [1.0, 2.0], "feature": [10.0, 20.0], "target": ["no", "yes"]}
    assert test.to_dict("list") == {"id": [4.0, 5.0], "feature": [40.0, 50.0]}
    assert sample.to_dict("list") == {"id": [4.0, 5.0], "target": ["no", "no"]}


def test_find_competition_files_supports_compressed_html_table_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frames = {
        "train.html": pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}),
        "test.htm": pd.DataFrame({"id": [4, 5], "feature": [40, 50]}),
        "sample_submission.html.zst": pd.DataFrame({"id": [4, 5], "target": [0, 0]}),
    }
    for name, frame in frames.items():
        html = frame.to_html(index=False)
        path = data_dir / name
        if name.endswith(".zst"):
            path.write_bytes(zstd.ZstdCompressor().compress(html.encode("utf-8")))
        else:
            path.write_text(html, encoding="utf-8")

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.html"
    assert test_path.name == "test.htm"
    assert sample_path.name == "sample_submission.html.zst"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


def test_find_competition_files_supports_xml_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}).to_xml(
        data_dir / "train.xml",
        index=False,
        parser="etree",
    )
    (data_dir / "test.xml.zst").write_bytes(
        zstd.ZstdCompressor().compress(
            pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_xml(index=False, parser="etree").encode("utf-8")
        )
    )
    (data_dir / "sample_submission.xml.zst").write_bytes(
        zstd.ZstdCompressor().compress(
            pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_xml(index=False, parser="etree").encode("utf-8")
        )
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.xml"
    assert test_path.name == "test.xml.zst"
    assert sample_path.name == "sample_submission.xml.zst"
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


def test_find_competition_files_supports_feather_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}).to_feather(data_dir / "train.feather")
    pd.DataFrame({"id": [3, 4], "feature": [30, 40]}).to_feather(data_dir / "test.feather")
    pd.DataFrame({"id": [3, 4], "target": [0, 0]}).to_feather(data_dir / "sample_submission.feather")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.feather"
    assert test_path.name == "test.feather"
    assert sample_path.name == "sample_submission.feather"


def test_find_competition_files_supports_orc_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}).to_orc(
        data_dir / "train.orc",
        index=False,
    )
    pd.DataFrame({"id": [3, 4], "feature": [30, 40]}).to_orc(data_dir / "test.orc", index=False)
    pd.DataFrame({"id": [3, 4], "target": [0, 0]}).to_orc(
        data_dir / "sample_submission.orc",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.orc"
    assert test_path.name == "test.orc"
    assert sample_path.name == "sample_submission.orc"


def test_find_competition_files_supports_hdf_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}).to_hdf(
        data_dir / "train.h5",
        key="train",
        mode="w",
        format="table",
        index=False,
    )
    pd.DataFrame({"id": [3, 4], "feature": [30, 40]}).to_hdf(
        data_dir / "test.h5",
        key="test",
        mode="w",
        format="table",
        index=False,
    )
    pd.DataFrame({"id": [3, 4], "target": [0, 0]}).to_hdf(
        data_dir / "sample_submission.hdf5",
        key="sample_submission",
        mode="w",
        format="table",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.h5"
    assert test_path.name == "test.h5"
    assert sample_path.name == "sample_submission.hdf5"


def test_load_competition_data_supports_native_hdf5_inputs(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with h5py.File(data_dir / "train.h5", "w") as handle:
        group = handle.create_group("train")
        group.create_dataset("id", data=np.array([1, 2, 3]))
        group.create_dataset("feature", data=np.array([10.0, 20.0, 30.0]))
        group.create_dataset("target", data=np.array([0, 1, 0]))
    with h5py.File(data_dir / "test.h5", "w") as handle:
        group = handle.create_group("test")
        group.create_dataset("id", data=np.array([4, 5]))
        group.create_dataset("feature", data=np.array([40.0, 50.0]))
    with h5py.File(data_dir / "sample_submission.hdf5", "w") as handle:
        group = handle.create_group("submission")
        group.create_dataset("id", data=np.array([4, 5]))
        group.create_dataset("target", data=np.array([0.0, 0.0]))

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.h5"
    assert test_path.name == "test.h5"
    assert sample_path.name == "sample_submission.hdf5"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.test["id"].tolist() == [4, 5]
    assert data.sample["target"].tolist() == [0.0, 0.0]


def test_load_competition_data_supports_h5ad_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_h5ad_table(
        data_dir / "train.h5ad",
        ids=np.array([1, 2, 3]),
        features=np.array([[10.0, 0.1], [20.0, 0.2], [30.0, 0.3]]),
        target=np.array([0, 1, 0]),
    )
    _write_h5ad_table(
        data_dir / "test.h5ad",
        ids=np.array([4, 5]),
        features=np.array([[40.0, 0.4], [50.0, 0.5]]),
    )
    pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.h5ad"
    assert test_path.name == "test.h5ad"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.feature_columns == ["cell_type", "gene_0", "gene_1"]
    assert data.test["id"].tolist() == [4, 5]
    assert data.train["target"].tolist() == [0, 1, 0]


def test_load_competition_data_supports_geopackage_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_geopackage_table(
        data_dir / "train.gpkg",
        [
            (1, 10.0, 0, b"\x47\x50\x00\x01"),
            (2, 20.0, 1, b"\x47\x50\x00\x02"),
            (3, 30.0, 0, b"\x47\x50\x00\x03"),
        ],
    )
    _write_geopackage_table(
        data_dir / "test.gpkg",
        [
            (4, 40.0, None, b"\x47\x50\x00\x04"),
            (5, 50.0, None, b"\x47\x50\x00\x05"),
        ],
        table="test",
    )
    pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.gpkg"
    assert test_path.name == "test.gpkg"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.test["geom"].tolist() == ["47500004", "47500005"]


def test_load_competition_data_supports_shapefile_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.shp").write_bytes(b"")
    (data_dir / "test.shp").write_bytes(b"")
    _write_dbf_table(
        data_dir / "train.dbf",
        [
            (1, 10.0, 0, "north"),
            (2, 20.0, 1, "south"),
            (3, 30.0, 0, "east"),
        ],
    )
    _write_dbf_table(
        data_dir / "test.dbf",
        [
            (4, 40.0, None, "west"),
            (5, 50.0, None, "north"),
        ],
    )
    pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.shp"
    assert test_path.name == "test.shp"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.test["zone"].tolist() == ["west", "north"]


def test_load_competition_data_supports_kmz_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_kmz(data_dir / "train.kmz", _kml_payload([(1, 10.0, 0), (2, 20.0, 1), (3, 30.0, 0)]))
    _write_kmz(data_dir / "test.kmz", _kml_payload([(4, 40.0, None), (5, 50.0, None)]))
    pd.DataFrame({"id": ["4", "5"], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.kmz"
    assert test_path.name == "test.kmz"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.test["id"].tolist() == ["4", "5"]


def test_load_competition_data_supports_compressed_kml_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with gzip.open(data_dir / "train.kml.gz", "wt", encoding="utf-8") as handle:
        handle.write(_kml_payload([(1, 10.0, 0), (2, 20.0, 1), (3, 30.0, 0)]))
    with gzip.open(data_dir / "test.kml.gz", "wt", encoding="utf-8") as handle:
        handle.write(_kml_payload([(4, 40.0, None), (5, 50.0, None)]))
    pd.DataFrame({"id": ["4", "5"], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.kml.gz"
    assert test_path.name == "test.kml.gz"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.test["id"].tolist() == ["4", "5"]


def test_load_competition_data_supports_numpy_archive_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(
        data_dir / "train.npz", id=np.array([1, 2, 3]), feature=np.array([10.0, 20.0, 30.0]), target=np.array([0, 1, 0])
    )
    np.savez(data_dir / "test.npz", id=np.array([4, 5]), feature=np.array([40.0, 50.0]))
    pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.npz"
    assert test_path.name == "test.npz"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.test["id"].tolist() == [4, 5]
    assert data.train["target"].tolist() == [0, 1, 0]


def test_load_competition_data_supports_netcdf_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_netcdf_table(
        data_dir / "train.nc",
        {
            "id": np.array([1, 2, 3], dtype=np.int32),
            "feature": np.array([10.0, 20.0, 30.0], dtype=np.float64),
            "target": np.array([0, 1, 0], dtype=np.int32),
        },
    )
    _write_netcdf_table(
        data_dir / "test.nc",
        {
            "id": np.array([4, 5], dtype=np.int32),
            "feature": np.array([40.0, 50.0], dtype=np.float64),
        },
    )
    pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.nc"
    assert test_path.name == "test.nc"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.feature_columns == ["feature"]
    assert data.test["id"].tolist() == [4, 5]
    assert data.train["target"].tolist() == [0, 1, 0]


def test_load_competition_data_supports_fits_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_fits_table(
        data_dir / "train.fits",
        {
            "id": np.array([1, 2, 3], dtype=np.int64),
            "feature": np.array([10.0, 20.0, 30.0], dtype=np.float64),
            "target": np.array([0, 1, 0], dtype=np.int64),
        },
    )
    _write_fits_table(
        data_dir / "test.fits",
        {
            "id": np.array([4, 5], dtype=np.int64),
            "feature": np.array([40.0, 50.0], dtype=np.float64),
        },
    )
    pd.DataFrame({"id": [4, 5], "target": [0.0, 0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.fits"
    assert test_path.name == "test.fits"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.feature_columns == ["feature"]
    assert data.test["id"].tolist() == [4, 5]
    assert data.train["target"].tolist() == [0, 1, 0]


def test_load_competition_data_supports_numpy_matrix_column_sidecars(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "train.npy", np.array([[1.0, 10.0, 0.0], [2.0, 20.0, 1.0], [3.0, 30.0, 0.0]]))
    np.save(data_dir / "test.npy", np.array([[4.0, 40.0], [5.0, 50.0]]))
    (data_dir / "train_columns.txt").write_text("id\nfeature\ntarget\n", encoding="utf-8")
    (data_dir / "test_columns.txt").write_text("id\nfeature\n", encoding="utf-8")
    pd.DataFrame({"id": [4.0, 5.0], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)
    data = load_competition_data(data_dir)

    assert train_path.name == "train.npy"
    assert test_path.name == "test.npy"
    assert sample_path.name == "sample_submission.csv"
    assert data.id_column == "id"
    assert data.target_column == "target"
    assert data.feature_columns == ["feature"]
    assert data.test["id"].tolist() == [4.0, 5.0]


def test_write_submission_accepts_2d_prediction_matrix(tmp_path) -> None:
    sample = pd.DataFrame({"row_id": [1, 2], "y1": [0.0, 0.0], "y2": [0.0, 0.0]})
    test = pd.DataFrame({"row_id": [1, 2], "f": [5, 6]})
    preds = np.array([[0.3, 0.7], [0.4, 0.6]])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="row_id",
        target_columns=["y1", "y2"],
        output_path=tmp_path / "submission.csv",
    )
    frame = pd.read_csv(out)
    assert frame["y1"].tolist() == [0.3, 0.4]
    assert frame["y2"].tolist() == [0.7, 0.6]


def test_write_submission_expands_single_label_to_probability_columns(tmp_path) -> None:
    sample = pd.DataFrame(
        {
            "id": [1, 2],
            "class_bird": [1 / 3, 1 / 3],
            "class_cat": [1 / 3, 1 / 3],
            "class_dog": [1 / 3, 1 / 3],
        }
    )
    test = pd.DataFrame({"id": [2, 1], "f": [6, 5]})
    preds = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="id",
        target_column="label",
        output_path=tmp_path / "submission.csv",
    )

    frame = pd.read_csv(out)
    assert list(frame.columns) == ["id", "class_bird", "class_cat", "class_dog"]
    assert frame["id"].tolist() == [1, 2]
    assert np.allclose(
        frame[["class_bird", "class_cat", "class_dog"]].to_numpy(),
        np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]),
    )


def test_write_submission_aligns_composite_sample_id_from_test_columns(tmp_path) -> None:
    sample = pd.DataFrame({"row_id": ["u2_i2", "u1_i1"], "target": [0.0, 0.0]})
    test = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i2"], "feature": [1.0, 2.0]})
    preds = np.array([0.1, 0.9])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="row_id",
        target_column="target",
        output_path=tmp_path / "submission.csv",
    )

    frame = pd.read_csv(out)
    assert frame["row_id"].tolist() == ["u2_i2", "u1_i1"]
    assert frame["target"].tolist() == [0.9, 0.1]


@pytest.mark.parametrize(
    ("sample_ids", "expected"),
    [
        (["u2:i2", "u1:i1"], ["u2:i2", "u1:i1"]),
        (["u2.i2", "u1.i1"], ["u2.i2", "u1.i1"]),
        (["u2|i2", "u1|i1"], ["u2|i2", "u1|i1"]),
        (["u2i2", "u1i1"], ["u2i2", "u1i1"]),
    ],
)
def test_write_submission_aligns_composite_sample_id_separator_variants(
    tmp_path,
    sample_ids: list[str],
    expected: list[str],
) -> None:
    sample = pd.DataFrame({"row_id": sample_ids, "target": [0.0, 0.0]})
    test = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i2"], "feature": [1.0, 2.0]})
    preds = np.array([0.1, 0.9])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="row_id",
        target_column="target",
        output_path=tmp_path / "submission.csv",
    )

    frame = pd.read_csv(out)
    assert frame["row_id"].tolist() == expected
    assert frame["target"].tolist() == [0.9, 0.1]


def test_write_submission_expands_tiny_public_sample_to_test_ids(tmp_path) -> None:
    sample = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "winner_model_a": [1 / 3, 1 / 3, 1 / 3],
            "winner_model_b": [1 / 3, 1 / 3, 1 / 3],
            "winner_tie": [1 / 3, 1 / 3, 1 / 3],
        }
    )
    test = pd.DataFrame({"id": [201, 202, 203, 204], "prompt": ["a", "b", "c", "d"]})
    preds = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.4, 0.3, 0.3],
        ]
    )

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="id",
        target_columns=["winner_model_a", "winner_model_b", "winner_tie"],
        output_path=tmp_path / "submission.csv",
    )

    frame = pd.read_csv(out)
    assert list(frame.columns) == list(sample.columns)
    assert frame["id"].tolist() == [201, 202, 203, 204]
    assert np.allclose(frame[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(), preds)


def test_ensure_sample_submission_discovers_tsv_sample(tmp_path) -> None:
    sample_path = tmp_path / "SampleSubmission.tsv"
    sample_path.write_text("id\tprediction\n1\t0\n", encoding="utf-8")
    (tmp_path / "train.csv").write_text("id,feature,target\n1,2,0\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("id,feature\n2,3\n", encoding="utf-8")

    assert ensure_sample_submission(tmp_path) == sample_path


def test_ensure_sample_submission_discovers_jsonl_context_sample(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    sample_path = context_dir / "SampleSubmission.jsonl"
    sample_path.write_text('{"id": 1, "prediction": 0}\n', encoding="utf-8")

    assert ensure_sample_submission(data_dir) == sample_path


def test_ensure_sample_submission_synthesizes_from_markdown_table_and_test_csv(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("row_id,feature,target\n1,10,0\n2,20,1\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("row_id,feature\n101,30\n102,40\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "\n".join(
            [
                "# Submission Format",
                "",
                "| row_id | prediction |",
                "| --- | --- |",
                "| 101 | 0 |",
            ]
        ),
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    assert sample_path.name == "sample_submission_synth.csv"
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "prediction"]
    assert sample["row_id"].astype(str).tolist() == ["101", "102"]
    assert sample["prediction"].tolist() == [0, 0]


def test_ensure_sample_submission_synthesizes_from_header_prose_and_test_csv(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "test.csv").write_text("row_id,feature\n101,30\n102,40\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nThe CSV header: row_id,prediction.\n",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "prediction"]
    assert sample["row_id"].astype(str).tolist() == ["101", "102"]


def test_ensure_sample_submission_synthesizes_compressed_sample_from_format_hint(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "test.csv").write_text("row_id,feature\n101,30\n102,40\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nUpload `submission.csv.gz` with columns row_id,prediction.\n",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    assert sample_path.name == "sample_submission_synth.csv.gz"
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "prediction"]
    assert sample["row_id"].astype(str).tolist() == ["101", "102"]


def test_ensure_sample_submission_synthesizes_pickle_zst_sample_from_format_hint(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "test.csv").write_text("row_id,feature\n101,30\n102,40\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nUpload `submission.pkl.zst` with columns row_id,prediction.\n",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    assert sample_path.name == "sample_submission_synth.pkl.zst"
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "prediction"]
    assert sample["row_id"].astype(str).tolist() == ["101", "102"]


def test_ensure_sample_submission_synthesizes_html_sample_from_format_hint(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "test.csv").write_text("row_id,feature\n101,30\n102,40\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nUpload `submission.html` with columns row_id,prediction.\n",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    assert sample_path.name == "sample_submission_synth.html"
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "prediction"]
    assert sample["row_id"].astype(str).tolist() == ["101", "102"]


@pytest.mark.parametrize(
    ("description", "expected_name"),
    [
        ("Participants must upload an HDF5 file with columns row_id,prediction.", "sample_submission_synth.hdf5"),
        ("Participants must upload an ORC file with columns row_id,prediction.", "sample_submission_synth.orc"),
        (
            "Participants must upload an xz-compressed YAML file with columns row_id,prediction.",
            "sample_submission_synth.yaml.xz",
        ),
        (
            "Participants must upload a zstd-compressed NDJSON file with columns row_id,prediction.",
            "sample_submission_synth.ndjson.zst",
        ),
        (
            "Participants must upload a bzip2-compressed HTML file with columns row_id,prediction.",
            "sample_submission_synth.html.bz2",
        ),
        (
            "Participants must upload an xz-compressed PSV file with columns row_id,prediction.",
            "sample_submission_synth.psv.xz",
        ),
        (
            "Participants must upload a zstd-compressed TAB file with columns row_id,prediction.",
            "sample_submission_synth.tab.zst",
        ),
    ],
)
def test_ensure_sample_submission_synthesizes_structured_sample_from_prose_format_hint(
    tmp_path,
    description: str,
    expected_name: str,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "test.csv").write_text("row_id,feature\n101,30\n102,40\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\n\n{description}\n",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    assert sample_path.name == expected_name
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "prediction"]
    assert sample["row_id"].astype(str).tolist() == ["101", "102"]


def test_ensure_sample_submission_synthesizes_from_roleless_eval_table(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "features_labeled.csv").write_text(
        "row_id,feature,SalePrice\n1,10,100000\n2,20,120000\n",
        encoding="utf-8",
    )
    (data_dir / "features_eval.csv").write_text("row_id,feature\n101,30\n102,40\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nSubmit a CSV file with columns `row_id` and `SalePrice`.",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "SalePrice"]
    assert sample["row_id"].astype(str).tolist() == ["101", "102"]
    assert sample["SalePrice"].tolist() == [0, 0]


def test_ensure_sample_submission_synthesizes_from_submit_id_table(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "submit_ids.csv").write_text("row_id\ncase_b\ncase_a\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nSubmit a CSV file with columns `row_id` and `target`.",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert list(sample.columns) == ["row_id", "target"]
    assert sample["row_id"].astype(str).tolist() == ["case_b", "case_a"]
    assert sample["target"].tolist() == [0, 0]


def test_ensure_sample_submission_synthesizes_from_test_images_dir(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    train_images = data_dir / "train_images"
    test_images = data_dir / "test_images"
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (train_images / "train_1.jpg").write_bytes(b"train")
    (test_images / "case_b.jpg").write_bytes(b"test")
    (test_images / "case_a.jpg").write_bytes(b"test")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nSubmit a CSV file with columns `image_id` and `target`.",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert list(sample.columns) == ["image_id", "target"]
    assert sample["image_id"].astype(str).tolist() == ["case_a.jpg", "case_b.jpg"]
    assert sample["target"].tolist() == [0, 0]


def test_ensure_sample_submission_synthesizes_from_test_audio_dir(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    train_audio = data_dir / "audio" / "train"
    test_audio = data_dir / "audio" / "test"
    train_audio.mkdir(parents=True)
    test_audio.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (train_audio / "train_1.wav").write_bytes(b"train")
    (test_audio / "clip_b.wav").write_bytes(b"test")
    (test_audio / "clip_a.wav").write_bytes(b"test")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nSubmit a CSV file with columns `audio_id` and `target`.",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert list(sample.columns) == ["audio_id", "target"]
    assert sample["audio_id"].astype(str).tolist() == ["clip_a.wav", "clip_b.wav"]
    assert sample["target"].tolist() == [0, 0]


def test_ensure_sample_submission_synthesizes_from_eval_images_dir(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    eval_images = data_dir / "eval_images"
    eval_images.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (eval_images / "scan_2.png").write_bytes(b"eval")
    (eval_images / "scan_1.png").write_bytes(b"eval")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nSubmit a CSV file with columns `image_id` and `prediction`.",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert list(sample.columns) == ["image_id", "prediction"]
    assert sample["image_id"].astype(str).tolist() == ["scan_1.png", "scan_2.png"]
    assert sample["prediction"].tolist() == [0, 0]


def test_ensure_sample_submission_uses_stem_ids_when_format_examples_do(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    test_images = data_dir / "test_images"
    test_images.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (test_images / "case_002.jpg").write_bytes(b"test")
    (test_images / "case_001.jpg").write_bytes(b"test")
    (context_dir / "submission_format.md").write_text(
        "\n".join(
            [
                "## Submission Format",
                "```csv",
                "image_id,target",
                "case_000,0",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert list(sample.columns) == ["image_id", "target"]
    assert sample["image_id"].astype(str).tolist() == ["case_001", "case_002"]


def test_ensure_sample_submission_uses_relative_asset_ids_when_format_examples_do(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    image_dir = data_dir / "images" / "test"
    image_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (image_dir / "abc.jpg").write_bytes(b"image")
    (image_dir / "def.jpg").write_bytes(b"image")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\n| image_id | label |\n| --- | --- |\n| images/test/abc.jpg | 0 |\n",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert sample["image_id"].tolist() == ["images/test/abc.jpg", "images/test/def.jpg"]


def test_ensure_sample_submission_uses_role_relative_asset_ids_when_format_examples_do(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    image_dir = data_dir / "images" / "test"
    image_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (image_dir / "abc.jpg").write_bytes(b"image")
    (image_dir / "def.jpg").write_bytes(b"image")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\n| image_id | label |\n| --- | --- |\n| test/abc.jpg | 0 |\n",
        encoding="utf-8",
    )

    sample_path = ensure_sample_submission(data_dir)

    assert sample_path is not None
    sample = read_table(sample_path)
    assert sample["image_id"].tolist() == ["test/abc.jpg", "test/def.jpg"]


def test_find_competition_files_uses_synthesized_sample_from_test_features(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    context_dir = tmp_path / "demo" / "context"
    data_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (data_dir / "train_labels.csv").write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    (data_dir / "train_features.csv").write_text("id,f1\n1,10\n2,20\n", encoding="utf-8")
    (data_dir / "test_features.csv").write_text("id,f1\n9,90\n10,100\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n\nSubmit a CSV file with columns `id` and `target`.",
        encoding="utf-8",
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train_features.csv"
    assert test_path.name == "test_features.csv"
    sample = read_table(sample_path)
    assert list(sample.columns) == ["id", "target"]
    assert sample["id"].astype(str).tolist() == ["9", "10"]


def test_find_competition_files_synthesizes_asset_tables_from_compressed_labels(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "img_1.jpg").write_bytes(b"train")
    (data_dir / "img_2.jpg").write_bytes(b"test")
    with gzip.open(data_dir / "train_labels.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\nimg_1,1\n")
    (data_dir / "sample_submission.csv").write_text("id,target\nimg_2,0\n", encoding="utf-8")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train_synth.csv"
    assert test_path.name == "test_synth.csv"
    assert sample_path.name == "sample_submission.csv"
    train = read_table(train_path)
    test = read_table(test_path)
    assert train["id"].tolist() == ["img_1"]
    assert test["id"].tolist() == ["img_2"]
    assert train["asset_path"].str.endswith("img_1.jpg").tolist() == [True]
    assert test["asset_path"].str.endswith("img_2.jpg").tolist() == [True]


def test_synthesize_asset_tables_ignores_idless_sample_prediction_column(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "img_1.jpg").write_bytes(b"train")
    (data_dir / "img_2.jpg").write_bytes(b"test")
    labels = pd.DataFrame({"target": ["img_1"], "label": [1]})
    labels.to_csv(data_dir / "train_labels.csv", index=False)
    sample_path = data_dir / "sample_submission.csv"
    pd.DataFrame({"target": ["img_2"], "score": [0.0]}).to_csv(sample_path, index=False)

    assert _synthesize_train_test_from_assets(data_dir, sample_path) is None


def test_find_competition_files_synthesizes_asset_tables_from_annotations(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_images = data_dir / "images" / "train"
    test_images = data_dir / "images" / "test"
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (train_images / "img_1.jpg").write_bytes(b"train")
    (test_images / "img_2.jpg").write_bytes(b"test")
    (data_dir / "annotations.csv").write_text("image_id,target\nimg_1,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("image_id,target\nimg_2,0\n", encoding="utf-8")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train_synth.csv"
    assert test_path.name == "test_synth.csv"
    assert sample_path.name == "sample_submission.csv"
    train = read_table(train_path)
    test = read_table(test_path)
    assert train["image_id"].tolist() == ["img_1"]
    assert test["image_id"].tolist() == ["img_2"]
    assert train["asset_path"].str.endswith("images/train/img_1.jpg").tolist() == [True]
    assert test["asset_path"].str.endswith("images/test/img_2.jpg").tolist() == [True]


def test_find_competition_files_synthesizes_asset_tables_with_label_id_alias(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_images = data_dir / "images" / "train"
    test_images = data_dir / "images" / "test"
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (train_images / "img_1.jpg").write_bytes(b"train")
    (test_images / "img_2.jpg").write_bytes(b"test")
    (data_dir / "annotations.csv").write_text("target,filename\n1,img_1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("image_id,target\nimg_2,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["filename"].tolist() == ["img_1"]
    assert test["image_id"].tolist() == ["img_2"]
    assert train["asset_path"].str.endswith("images/train/img_1.jpg").tolist() == [True]
    assert test["asset_path"].str.endswith("images/test/img_2.jpg").tolist() == [True]


def test_find_competition_files_synthesizes_asset_tables_from_excel_annotations(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_images = data_dir / "images" / "train"
    test_images = data_dir / "images" / "test"
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (train_images / "img_1.jpg").write_bytes(b"train")
    (test_images / "img_2.jpg").write_bytes(b"test")
    pd.DataFrame({"image_id": ["img_1"], "target": [1]}).to_excel(data_dir / "annotations.xlsx", index=False)
    (data_dir / "sample_submission.csv").write_text("image_id,target\nimg_2,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["image_id"].tolist() == ["img_1"]
    assert test["image_id"].tolist() == ["img_2"]
    assert train["asset_path"].str.endswith("images/train/img_1.jpg").tolist() == [True]
    assert test["asset_path"].str.endswith("images/test/img_2.jpg").tolist() == [True]


def test_find_competition_files_tries_next_asset_label_candidate(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_images = data_dir / "images" / "train"
    test_images = data_dir / "images" / "test"
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (train_images / "img_1.jpg").write_bytes(b"train")
    (test_images / "img_2.jpg").write_bytes(b"test")
    (data_dir / "annotations.csv").write_text("image_id,target\nmissing,1\n", encoding="utf-8")
    (data_dir / "ground_truth.csv").write_text("image_id,target\nimg_1,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("image_id,target\nimg_2,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["image_id"].tolist() == ["img_1"]
    assert test["image_id"].tolist() == ["img_2"]
    assert train["asset_path"].str.endswith("images/train/img_1.jpg").tolist() == [True]
    assert test["asset_path"].str.endswith("images/test/img_2.jpg").tolist() == [True]


def test_find_competition_files_synthesizes_asset_tables_from_video_files(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "clip_train.mp4").write_bytes(b"train-video")
    (data_dir / "clip_test.mp4").write_bytes(b"test-video")
    (data_dir / "train_labels.csv").write_text("id,target\nclip_train,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\nclip_test,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["asset_path"].str.endswith("clip_train.mp4").tolist() == [True]
    assert test["asset_path"].str.endswith("clip_test.mp4").tolist() == [True]


def test_find_competition_files_synthesizes_asset_tables_from_nifti_stems(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "case_train.nii.gz").write_bytes(b"train-scan")
    (data_dir / "case_test.nii.gz").write_bytes(b"test-scan")
    (data_dir / "train_labels.csv").write_text("id,target\ncase_train,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\ncase_test,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["asset_path"].str.endswith("case_train.nii.gz").tolist() == [True]
    assert test["asset_path"].str.endswith("case_test.nii.gz").tolist() == [True]


def test_find_competition_files_synthesizes_dicom_asset_tables_with_series_id_alias(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_dicom = data_dir / "dicom" / "train"
    test_dicom = data_dir / "dicom" / "test"
    train_dicom.mkdir(parents=True)
    test_dicom.mkdir(parents=True)
    (train_dicom / "case_train.dcm").write_bytes(b"train-dicom")
    (test_dicom / "case_test.dcm").write_bytes(b"test-dicom")
    (data_dir / "annotations.csv").write_text("id,target\ncase_train,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("series_id,target\ncase_test,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["id"].tolist() == ["case_train"]
    assert test["series_id"].tolist() == ["case_test"]
    assert train["asset_path"].str.endswith("dicom/train/case_train.dcm").tolist() == [True]
    assert test["asset_path"].str.endswith("dicom/test/case_test.dcm").tolist() == [True]


def test_find_competition_files_synthesizes_asset_tables_from_outcomes(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_audio = data_dir / "audio" / "train"
    test_audio = data_dir / "audio" / "test"
    train_audio.mkdir(parents=True)
    test_audio.mkdir(parents=True)
    (train_audio / "clip_train.wav").write_bytes(b"train-audio")
    (test_audio / "clip_test.wav").write_bytes(b"test-audio")
    (data_dir / "outcomes.csv").write_text("audio_id,target\nclip_train,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("audio_id,target\nclip_test,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["audio_id"].tolist() == ["clip_train"]
    assert test["audio_id"].tolist() == ["clip_test"]
    assert train["asset_path"].str.endswith("audio/train/clip_train.wav").tolist() == [True]
    assert test["asset_path"].str.endswith("audio/test/clip_test.wav").tolist() == [True]


def test_find_competition_files_synthesizes_asset_tables_from_zarr_directories(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    train_store = data_dir / "sample_train.zarr"
    test_store = data_dir / "sample_test.zarr"
    train_store.mkdir()
    test_store.mkdir()
    (train_store / ".zarray").write_text("{}", encoding="utf-8")
    (test_store / ".zarray").write_text("{}", encoding="utf-8")
    (data_dir / "train_labels.csv").write_text("id,target\nsample_train,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\nsample_test,0\n", encoding="utf-8")

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["asset_path"].str.endswith("sample_train.zarr").tolist() == [True]
    assert test["asset_path"].str.endswith("sample_test.zarr").tolist() == [True]


def test_find_competition_files_synthesizes_asset_tables_from_relative_asset_ids(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_images = data_dir / "images" / "train"
    test_images = data_dir / "images" / "test"
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (train_images / "case_1.jpg").write_bytes(b"train")
    (test_images / "case_2.jpg").write_bytes(b"test")
    (data_dir / "train_labels.csv").write_text(
        "path,target\nimages/train/case_1.jpg,1\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text(
        "path,target\nimages/test/case_2.jpg,0\n",
        encoding="utf-8",
    )

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["asset_path"].str.endswith("images/train/case_1.jpg").tolist() == [True]
    assert test["asset_path"].str.endswith("images/test/case_2.jpg").tolist() == [True]


def test_find_competition_files_resolves_duplicate_asset_stems_by_split(tmp_path) -> None:
    data_dir = tmp_path / "demo" / "data"
    train_images = data_dir / "images" / "training"
    test_images = data_dir / "images" / "public"
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (train_images / "Case_001.JPG").write_bytes(b"train")
    (test_images / "Case_001.JPG").write_bytes(b"test")
    (data_dir / "train_labels.csv").write_text(
        "image_id,target\ncase_001,1\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text(
        "image_id,target\npublic/case_001,0\n",
        encoding="utf-8",
    )

    train_path, test_path, _ = find_competition_files(data_dir)

    train = read_table(train_path)
    test = read_table(test_path)
    assert train["image_id"].tolist() == ["case_001"]
    assert test["image_id"].tolist() == ["public/case_001"]
    assert train["asset_path"].str.endswith("images/training/Case_001.JPG").tolist() == [True]
    assert test["asset_path"].str.endswith("images/public/Case_001.JPG").tolist() == [True]


def test_find_competition_files_extracts_top_level_zip(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with zipfile.ZipFile(data_dir / "competition.zip", "w") as archive:
        archive.writestr("train.csv", "id,feature,target\n1,10,0\n2,20,1\n")
        archive.writestr("test.csv", "id,feature\n3,30\n4,40\n")
        archive.writestr("sample_submission.csv", "id,target\n3,0\n4,0\n")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"
    assert sample_path.name == "sample_submission.csv"


def test_find_competition_files_extracts_top_level_tgz(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with tarfile.open(data_dir / "competition.tgz", "w:gz") as archive:
        for name, text in {
            "train.csv": "id,feature,target\n1,10,0\n2,20,1\n",
            "test.csv": "id,feature\n3,30\n4,40\n",
            "sample_submission.csv": "id,target\n3,0\n4,0\n",
        }.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"
    assert sample_path.name == "sample_submission.csv"


def test_find_competition_files_extracts_top_level_tar_zst(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        for name, text in {
            "train.csv": "id,feature,target\n1,10,0\n2,20,1\n",
            "test.csv": "id,feature\n3,30\n4,40\n",
            "sample_submission.csv": "id,target\n3,0\n4,0\n",
        }.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    (data_dir / "competition.tar.zst").write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"
    assert sample_path.name == "sample_submission.csv"


def test_find_competition_files_extracts_nested_archives(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_zip = io.BytesIO()
    with zipfile.ZipFile(train_zip, "w") as archive:
        archive.writestr("train.csv", "id,feature,target\n1,10,0\n2,20,1\n")
    test_zip = io.BytesIO()
    with zipfile.ZipFile(test_zip, "w") as archive:
        archive.writestr("test.csv", "id,feature\n3,30\n4,40\n")
    with zipfile.ZipFile(data_dir / "competition.zip", "w") as archive:
        archive.writestr("train.zip", train_zip.getvalue())
        archive.writestr("test.zip", test_zip.getvalue())
        archive.writestr("sample_submission.csv", "id,target\n3,0\n4,0\n")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"
    assert sample_path.name == "sample_submission.csv"


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm"])
def test_load_competition_data_reads_excel_files(tmp_path, suffix: str) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "feature": [10, 20, 30], "target": [0, 1, 0]}).to_excel(
        data_dir / f"train{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_excel(data_dir / f"test{suffix}", index=False)
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_excel(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [10, 20, 30]
    assert data.sample["id"].tolist() == [4, 5]


def test_load_competition_data_reads_ods_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "feature": [10, 20, 30], "target": [0, 1, 0]}).to_excel(
        data_dir / "train.ods",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_excel(data_dir / "test.ods", index=False)
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_excel(
        data_dir / "sample_submission.ods",
        index=False,
    )

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [10, 20, 30]
    assert data.sample["id"].tolist() == [4, 5]


def test_find_competition_files_supports_xlsb_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frames = {
        "train.xlsb": pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}),
        "test.xlsb": pd.DataFrame({"id": [4, 5], "feature": [40, 50]}),
        "sample_submission.xlsb": pd.DataFrame({"id": [4, 5], "target": [0, 0]}),
    }
    for name in frames:
        (data_dir / name).write_bytes(b"xlsb")

    def fake_read_excel(path: Path, *args, **kwargs) -> pd.DataFrame:
        assert kwargs["engine"] == "pyxlsb"
        return frames[Path(path).name].copy()

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.xlsb"
    assert test_path.name == "test.xlsb"
    assert sample_path.name == "sample_submission.xlsb"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


def test_find_competition_files_supports_compressed_svmlight_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.libsvm").write_text("1 1:0.5 3:1.5\n0 2:2.5\n", encoding="utf-8")
    (data_dir / "test.svm.zst").write_bytes(
        zstd.ZstdCompressor().compress(b"0 1:3.5 3:4.5\n0 2:5.5\n"),
    )
    pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = _dense_sparse_columns(read_table(train_path))
    test = _dense_sparse_columns(read_table(test_path))
    sample = read_table(sample_path)

    assert train_path.name == "train.libsvm"
    assert test_path.name == "test.svm.zst"
    assert sample_path.name == "sample_submission.csv"
    assert train.to_dict("list") == {
        "target": [1.0, 0.0],
        "feature_1": [0.5, 0.0],
        "feature_2": [0.0, 2.5],
        "feature_3": [1.5, 0.0],
    }
    assert test.to_dict("list") == {
        "feature_1": [3.5, 0.0],
        "feature_2": [0.0, 5.5],
        "feature_3": [4.5, 0.0],
    }
    assert sample.to_dict("list") == {"id": [1, 2], "target": [0, 0]}


def test_find_competition_files_supports_space_delimited_dat_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.dat").write_text("id feature target\n1 10 0\n2 20 1\n", encoding="utf-8")
    with gzip.open(data_dir / "test.dat.gz", "wt", encoding="utf-8") as handle:
        handle.write("id feature\n4 40\n5 50\n")
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.dat"
    assert test_path.name == "test.dat.gz"
    assert sample_path.name == "sample_submission.csv"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


def test_find_competition_files_supports_fixed_width_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.fwf").write_text("id feature target\n1  10      0\n2  20      1\n", encoding="utf-8")
    with gzip.open(data_dir / "test.fwf.gz", "wt", encoding="utf-8") as handle:
        handle.write("id feature\n4  40\n5  50\n")
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    assert train_path.name == "train.fwf"
    assert test_path.name == "test.fwf.gz"
    assert sample_path.name == "sample_submission.csv"
    assert train.to_dict("list") == {"id": [1, 2], "feature": [10, 20], "target": [0, 1]}
    assert test.to_dict("list") == {"id": [4, 5], "feature": [40, 50]}
    assert sample.to_dict("list") == {"id": [4, 5], "target": [0, 0]}


def test_load_competition_data_reads_csv_gz_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with gzip.open(data_dir / "train.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n1,10,0\n2,20,1\n3,30,0\n")
    with gzip.open(data_dir / "test.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n4,40\n5,50\n")
    with gzip.open(data_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n4,0\n5,0\n")

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [10, 20, 30]
    assert data.sample["id"].tolist() == [4, 5]


def test_find_competition_files_prefers_compressed_canonical_train_test_pair(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("train.csv.gz", "train_features.csv.gz"):
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            handle.write("id,feature,target\n1,10,0\n2,20,1\n")
    for name in ("test.csv.gz", "test_features.csv.gz"):
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            handle.write("id,feature\n4,40\n5,50\n")
    with gzip.open(data_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n4,0\n5,0\n")

    train_path, test_path, _ = find_competition_files(data_dir)

    assert train_path.name == "train.csv.gz"
    assert test_path.name == "test.csv.gz"


def test_find_competition_files_avoids_test_substring_false_positive(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [10, 20, 30],
            "target": [0, 1, 0],
        }
    ).to_csv(data_dir / "TrainingSet.csv", index=False)
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_csv(
        data_dir / "PublicTest.csv",
        index=False,
    )
    pd.DataFrame({"id": [99], "feature": [999], "target": [1]}).to_csv(
        data_dir / "contest.csv",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_csv(
        data_dir / "SampleSubmission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "TrainingSet.csv"
    assert test_path.name == "PublicTest.csv"
    assert sample_path.name == "SampleSubmission.csv"


def test_find_competition_files_infers_roleless_train_test_by_schema(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "feature": [10, 20, 30, 40],
            "SalePrice": [100_000, 120_000, 140_000, 160_000],
        }
    ).to_csv(data_dir / "features_labeled.csv", index=False)
    pd.DataFrame({"row_id": [101, 102], "feature": [50, 60]}).to_csv(
        data_dir / "features_eval.csv",
        index=False,
    )
    pd.DataFrame({"row_id": [101, 102], "SalePrice": [0, 0]}).to_csv(
        data_dir / "SampleSubmission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "features_labeled.csv"
    assert test_path.name == "features_eval.csv"
    assert sample_path.name == "SampleSubmission.csv"


def test_find_competition_files_accepts_scoring_features_as_test(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [10, 20, 30],
            "target": [0, 1, 0],
        }
    ).to_csv(data_dir / "training_data.csv", index=False)
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_csv(
        data_dir / "scoring_features.csv",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "training_data.csv"
    assert test_path.name == "scoring_features.csv"
    assert sample_path.name == "sample_submission.csv"


def test_find_competition_files_accepts_validation_features_as_test(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [10, 20, 30],
            "target": [0, 1, 0],
        }
    ).to_csv(data_dir / "training_data.csv", index=False)
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_csv(
        data_dir / "validation_features.csv",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert train_path.name == "training_data.csv"
    assert test_path.name == "validation_features.csv"
    assert sample_path.name == "sample_submission.csv"


def test_find_competition_files_does_not_treat_public_train_as_test(tmp_path) -> None:
    data_dir = tmp_path / "data"
    public_dir = data_dir / "public"
    data_dir.mkdir()
    public_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [10, 20, 30],
            "target": [0, 1, 0],
        }
    ).to_csv(public_dir / "train.csv", index=False)
    pd.DataFrame({"id": [4, 5], "feature": [40, 50]}).to_csv(
        public_dir / "features.csv",
        index=False,
    )
    pd.DataFrame({"id": [4, 5], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    train_path, test_path, _ = find_competition_files(data_dir)

    assert train_path.name == "train.csv"
    assert test_path.name == "features.csv"


def test_load_competition_data_reads_csv_bz2_files(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with bz2.open(data_dir / "train.csv.bz2", "wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n1,10,0\n2,20,1\n3,30,0\n")
    with bz2.open(data_dir / "test.csv.bz2", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n4,40\n5,50\n")
    with bz2.open(data_dir / "sample_submission.csv.bz2", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n4,0\n5,0\n")

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [10, 20, 30]
    assert data.sample["id"].tolist() == [4, 5]


def test_load_competition_data_reads_sqlite_tables(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "competition.sqlite"
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

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [10, 20, 30]
    assert data.test["id"].tolist() == [4, 5]
    assert data.sample["id"].tolist() == [4, 5]
    assert (data_dir / ".kagglebot_cache" / "sqlite" / "competition__train.csv").exists()


def test_read_table_reads_duckdb_table(tmp_path: Path) -> None:
    path = tmp_path / "train.duckdb"
    conn = duckdb.connect(str(path))
    try:
        conn.execute("CREATE TABLE train (id INTEGER, feature INTEGER, target INTEGER)")
        conn.execute("INSERT INTO train VALUES (1, 10, 0), (2, 20, 1), (3, 30, 0)")
    finally:
        conn.close()

    frame = read_table(path, nrows=2)

    assert frame["feature"].tolist() == [10, 20]
    assert frame["target"].tolist() == [0, 1]


def test_read_table_reads_rds_table(tmp_path: Path) -> None:
    path = tmp_path / "train.rds"
    pyreadr.write_rds(path, pd.DataFrame({"id": [1, 2, 3], "feature": [10, 20, 30], "target": [0, 1, 0]}))

    frame = read_table(path, nrows=2)

    assert frame["feature"].tolist() == [10, 20]
    assert frame["target"].tolist() == [0, 1]


def test_load_competition_data_reads_duckdb_tables(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "competition.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE train (id INTEGER, feature INTEGER, target INTEGER)")
        conn.execute("INSERT INTO train VALUES (1, 10, 0), (2, 20, 1), (3, 30, 0)")
        conn.execute("CREATE TABLE test (id INTEGER, feature INTEGER)")
        conn.execute("INSERT INTO test VALUES (4, 40), (5, 50)")
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target INTEGER)")
        conn.execute("INSERT INTO sample_submission VALUES (4, 0), (5, 0)")
    finally:
        conn.close()

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [10, 20, 30]
    assert data.test["id"].tolist() == [4, 5]
    assert data.sample["id"].tolist() == [4, 5]
    assert (data_dir / ".kagglebot_cache" / "duckdb" / "competition__train.csv").exists()


def test_load_competition_data_reads_rds_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pyreadr.write_rds(
        data_dir / "train.rds", pd.DataFrame({"id": [1, 2, 3], "feature": [10, 20, 30], "target": [0, 1, 0]})
    )
    pyreadr.write_rds(data_dir / "test.rds", pd.DataFrame({"id": [4, 5], "feature": [40, 50]}))
    pyreadr.write_rds(data_dir / "sample_submission.rds", pd.DataFrame({"id": [4, 5], "target": [0, 0]}))

    data = load_competition_data(data_dir)

    assert data.target_column == "target"
    assert data.train["feature"].tolist() == [10, 20, 30]
    assert data.test["id"].tolist() == [4, 5]
    assert data.sample["id"].tolist() == [4, 5]


def test_write_submission_respects_tsv_output_suffix(tmp_path) -> None:
    sample = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [1, 2], "f": [5, 6]})

    out = write_submission(
        sample=sample,
        test=test,
        preds=np.array([0.25, 0.75]),
        id_column="id",
        target_column="target",
        output_path=tmp_path / "submission.tsv",
    )

    assert out.suffix == ".tsv"
    assert out.read_text(encoding="utf-8").splitlines()[0] == "id\ttarget"
    frame = pd.read_csv(out, sep="\t")
    assert frame["target"].tolist() == [0.25, 0.75]


@pytest.mark.parametrize(
    "suffix",
    [
        ".jsonl",
        ".jsonl.bz2",
        ".jsonl.xz",
        ".jsonl.zst",
        ".jsonlines",
        ".jsonlines.bz2",
        ".jsonlines.xz",
        ".jsonlines.zst",
        ".ndjson",
        ".ndjson.bz2",
        ".ndjson.xz",
        ".ndjson.zst",
    ],
)
def test_write_submission_respects_json_lines_output_suffix(tmp_path, suffix: str) -> None:
    sample = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
    test = pd.DataFrame({"id": [1, 2], "f": [5, 6]})

    out = write_submission(
        sample=sample,
        test=test,
        preds=np.array([0.25, 0.75]),
        id_column="id",
        target_column="target",
        output_path=tmp_path / f"submission{suffix}",
    )

    frame = read_table(out)
    assert list(frame.columns) == ["id", "target"]
    assert frame["target"].tolist() == [0.25, 0.75]
