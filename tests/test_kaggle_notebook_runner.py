from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import runpy
import sqlite3
import tarfile
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import py7zr
import pyreadr
import pyreadstat
import pytest
import zstandard as zstd
from scipy.io import savemat

from kagglebot.asset_modality import (
    ASSET_COLLECTION_DIR_NAMES,
    ASSET_COMPRESSION_SUFFIXES,
    AUDIO_SUFFIXES,
    BIO_FASTQ_BASE_SUFFIXES,
    BIO_MOL_STRUCTURE_BASE_SUFFIXES,
    BIO_PDB_STRUCTURE_BASE_SUFFIXES,
    BIO_SEQUENCE_BASE_SUFFIXES,
    DATA_ASSET_SUFFIXES,
    DICOM_IMAGE_BASE_SUFFIXES,
    DICOM_IMAGE_SUFFIXES,
    DIRECTORY_ARRAY_SUFFIXES,
    DOCUMENT_HTML_BASE_SUFFIXES,
    DOCUMENT_TEXT_METADATA_SUFFIXES,
    GRAPH_EDGE_LIST_BASE_SUFFIXES,
    GRAPH_XML_BASE_SUFFIXES,
    IMAGE_SUFFIXES,
    MEDICAL_HEADER_IMAGE_BASE_SUFFIXES,
    MEDICAL_HEADER_IMAGE_SUFFIXES,
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_FILENAMES,
    MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES,
    MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES,
    NIFTI_IMAGE_BASE_SUFFIXES,
    NIFTI_IMAGE_SUFFIXES,
    POINT_CLOUD_TEXT_METADATA_SUFFIXES,
    VIDEO_SUFFIXES,
)
from kagglebot.baseline_tokens import (
    ASSET_LABEL_TABLE_TOKENS,
    FILE_REFERENCE_NAME_TOKENS,
    ID_LIKE_COLUMN_NAMES,
    RLE_SEGMENTATION_COLUMN_TOKENS,
    TEXT_PREDICTION_NAME_TOKENS,
)
from kagglebot.kernel_sources import KernelSourceConfig
from kagglebot.role_tokens import ROLE_TRAILING_PREFIXES, TEST_DIRECT_ROLE_ALIASES
from kagglebot.runners import kaggle_notebook
from kagglebot.runners.kaggle_notebook import (
    KERNEL_TEMPLATE,
    _wait_for_kernel,
    build_kernel_metadata,
    find_submission_file,
)
from kagglebot.sample_name_aliases import SAMPLE_COMPACT_NAME_ALIASES, SAMPLE_OUTPUT_NAME_TOKENS
from kagglebot.solver.io import read_table, write_table
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_SUFFIXES,
    NON_TABULAR_SUBMISSION_SUFFIXES,
    ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES,
)
from kagglebot.submission_output_naming import CONFIGURED_TEMPLATE_STEMS
from kagglebot.submission_sample_discovery import (
    ROLE_ALIASES,
    ROLE_SUFFIXES,
    TABULAR_INPUT_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES,
)


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


def _write_rdata_table(path: Path, frame: pd.DataFrame) -> None:
    if path.suffix.lower() == ".rds":
        pyreadr.write_rds(path, frame)
        return
    pyreadr.write_rdata(path, frame, df_name="dataset")


def _write_h5ad_table(path: Path, *, ids: np.ndarray, features: np.ndarray, target: np.ndarray | None = None) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset("id", data=ids)
        if target is not None:
            obs.create_dataset("target", data=target)
        var = handle.create_group("var")
        var.create_dataset("_index", data=np.array([f"gene_{idx}".encode() for idx in range(features.shape[1])]))
        handle.create_dataset("X", data=features)


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
            "id": str(row[0]),
            "feature": f"{row[1]:.4f}",
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


def test_build_kernel_metadata_uses_plan_driven_sources() -> None:
    metadata = build_kernel_metadata(
        kaggle_username="user",
        kernel_slug="demo-kernel",
        title="demo kernel",
        competition_slug="demo",
        accelerator="gpu",
        enable_internet=False,
        source_config=KernelSourceConfig(
            dataset_sources=("alice/demo-dataset",),
            kernel_sources=("bob/demo-kernel",),
            model_sources=("carol/demo-model/PyTorch/default/1",),
        ),
    )

    assert metadata["competition_sources"] == ["demo"]
    assert metadata["dataset_sources"] == ["alice/demo-dataset"]
    assert metadata["kernel_sources"] == ["bob/demo-kernel"]
    assert metadata["model_sources"] == ["carol/demo-model/PyTorch/default/1"]


def test_kernel_template_aligns_train_test_column_case_to_sample() -> None:
    assert "def align_train_test_column_case_to_sample(" in KERNEL_TEMPLATE
    assert "train, test = align_train_test_column_case_to_sample(train, test, sample)" in KERNEL_TEMPLATE
    assert "sample_by_lower = {str(col).lower(): str(col) for col in sample.columns}" in KERNEL_TEMPLATE
    assert "def align_test_column_case_to_train(" in KERNEL_TEMPLATE
    assert "test = align_test_column_case_to_train(train, test)" in KERNEL_TEMPLATE


def test_render_kernel_main_injects_shared_tabular_suffixes() -> None:
    script = kaggle_notebook.render_kernel_main("demo", "cpu")

    compile(script, "main.py", "exec")
    assert "__TABULAR_" not in script
    assert "__SQLITE_" not in script
    assert "__CODE_FENCE_LANG_TO_SUFFIX_JSON__" not in script
    assert "__TABULAR_SUBMISSION_TOKEN_PATTERNS_JSON__" not in script
    assert "__SUBMISSION_TOKEN_PATTERN_SPECS_JSON__" not in script
    assert "__COMPRESSION_TOKEN_PATTERN_SPECS_JSON__" not in script
    assert "__COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES_JSON__" not in script
    assert "__CONFIGURED_TEMPLATE_STEMS_JSON__" not in script
    assert '"ndjson": ".ndjson"' in script
    assert '"nd-json": ".ndjson"' in script
    assert '"json-lines": ".jsonl"' in script
    assert '"jsonlines": ".jsonlines"' in script
    assert '"yaml": ".yaml"' in script
    assert '"html": ".html"' in script
    assert '"hdf5": ".hdf5"' in script
    assert '"\\\\bhdf5\\\\b|\\\\bhdf\\\\b", ".hdf5"' in script
    assert '"\\\\byaml\\\\b|\\\\byml\\\\b", ".yaml"' in script
    assert '"\\\\bzstd\\\\b|\\\\bzstandard\\\\b|\\\\.zst\\\\b", ".zst"' in script
    assert '"(?<![A-Za-z0-9])(?:epub)(?![A-Za-z0-9.])", ".epub"' in script
    assert '"(?<![A-Za-z0-9])(?:latex)(?![A-Za-z0-9.])", ".tex"' in script
    assert '".tex"' in script
    assert '".smiles"' in script
    for stem in CONFIGURED_TEMPLATE_STEMS:
        assert f'"{stem}"' in script
    for suffix in (
        ".db",
        ".sqlite",
        ".sqlite3",
        ".ndjson.zst",
        ".jsonlines.zst",
        ".sas7bdat",
        ".sav",
        ".mat",
        ".nc",
        ".netcdf",
        ".cdf",
        ".nc4",
        ".fits",
        ".fit",
        ".fts",
        ".fits.gz",
        ".npy",
        ".npz",
        ".arff",
        ".html.zst",
        ".xml.zst",
        ".geojson.zst",
        ".avro",
        ".yaml.zst",
        ".yml",
        ".ods",
        ".xlsb",
        ".h5ad",
        ".loom",
        ".gpkg",
        ".geopackage",
        ".shp",
        ".dbf",
        ".kml",
        ".kml.gz",
        ".kmz",
        ".svm.zst",
        ".dat.gz",
        ".fwf.gz",
        ".duckdb",
        ".ddb",
        ".rds",
        ".rda",
        ".rdata",
    ):
        assert suffix in TABULAR_INPUT_SUFFIXES
        assert f'"{suffix}"' in script
    assert "GEOJSON_FILE_SUFFIXES = set([" in script
    for suffix_group in (
        POINT_CLOUD_TEXT_METADATA_SUFFIXES,
        GRAPH_XML_BASE_SUFFIXES,
        GRAPH_EDGE_LIST_BASE_SUFFIXES,
        DOCUMENT_HTML_BASE_SUFFIXES,
        DOCUMENT_TEXT_METADATA_SUFFIXES,
        BIO_SEQUENCE_BASE_SUFFIXES,
        BIO_FASTQ_BASE_SUFFIXES,
        BIO_PDB_STRUCTURE_BASE_SUFFIXES,
        BIO_MOL_STRUCTURE_BASE_SUFFIXES,
    ):
        for suffix in suffix_group:
            assert f'"{suffix}"' in script
    assert ".dta" in TABULAR_SUBMISSION_SUFFIXES
    assert '".dta"' in script
    assert ".avro" in TABULAR_SUBMISSION_SUFFIXES
    assert '".avro"' in script
    assert ".npy" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".npz" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".nc" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".netcdf" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".cdf" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".nc4" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".fits" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".fits.gz" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".h5ad" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".loom" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".gpkg" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".geopackage" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".shp" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".dbf" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".kml" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".kmz" not in TABULAR_SUBMISSION_SUFFIXES


def test_render_kernel_main_uses_limited_head_reads_for_pair_scoring() -> None:
    script = kaggle_notebook.render_kernel_main("demo", "cpu")

    compile(script, "main.py", "exec")
    assert "def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:" in script
    assert "def read_text_tabular_frame(" in script
    assert "nrows: int | None = None" in script
    assert "read_table(train_path).head(5)" not in script
    assert "read_table(test_path).head(5)" not in script
    assert "read_table(sample_path).head(5)" not in script
    assert "train_head = read_table(train_path, nrows=5)" in script
    assert "test_head = read_table(test_path, nrows=5)" in script
    assert "sample_head = read_table(sample_path, nrows=5)" in script


def test_rendered_kernel_read_table_accepts_nrows_for_text_inputs(tmp_path: Path) -> None:
    namespace: dict[str, object] = {}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    read_table = namespace["read_table"]

    csv_path = tmp_path / "sample_submission.csv"
    csv_path.write_text("id;target\n1;0.1\n2;0.2\n3;0.3\n", encoding="utf-8")
    jsonl_path = tmp_path / "sample_submission.jsonl.gz"
    with gzip.open(jsonl_path, "wt", encoding="utf-8") as handle:
        for row in (
            {"id": 1, "target": 0.1},
            {"id": 2, "target": 0.2},
            {"id": 3, "target": 0.3},
        ):
            handle.write(json.dumps(row) + "\n")

    csv_frame = read_table(csv_path, nrows=2)
    jsonl_frame = read_table(jsonl_path, nrows=2)

    assert csv_frame.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}
    assert jsonl_frame.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_rendered_kernel_read_table_stabilizes_problematic_columns(tmp_path: Path) -> None:
    namespace: dict[str, object] = {}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    read_table = namespace["read_table"]
    finalize_table_frame = namespace["finalize_table_frame"]

    json_path = tmp_path / "records.json"
    json_path.write_text(
        json.dumps(
            {
                "columns": ["id", "", None, "score", "score"],
                "data": [[1, "a", 10, 0.1, 0.2]],
            }
        ),
        encoding="utf-8",
    )
    frame = read_table(json_path)

    assert list(frame.columns) == ["id", "column_2", "column_3", "score", "score_1"]

    csv_path = tmp_path / "blank_header.csv"
    csv_path.write_text("id,,target\n1,abc,0\n", encoding="utf-8")
    csv_frame = read_table(csv_path)

    assert list(csv_frame.columns) == ["id", "column_2", "target"]

    multi_index_frame = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("fold", "a"), ("Unnamed: 1_level_0", None)]),
    )
    normalized = finalize_table_frame(multi_index_frame)

    assert list(normalized.columns) == ["fold_a", "column_2"]


def test_rendered_kernel_write_table_stabilizes_problematic_columns(tmp_path: Path) -> None:
    namespace: dict[str, object] = {}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    write_table = namespace["write_table"]

    path = tmp_path / "submission.csv"
    frame = pd.DataFrame([[1, "-", 0.1, 0.2]], columns=["id", "", "score", "score"])

    write_table(frame, path)
    loaded = pd.read_csv(path)

    assert list(loaded.columns) == ["id", "column_2", "score", "score_1"]
    assert list(frame.columns) == ["id", "", "score", "score"]


def test_rendered_kernel_reads_nifti_file_reference_metadata(tmp_path: Path) -> None:
    import nibabel as nib

    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    nifti_metadata = namespace["nifti_metadata"]

    path = tmp_path / "scan.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.diag([1.5, 2.5, 3.5, 1.0]))
    nib.save(image, path)

    assert nifti_metadata(path) == (4.0, 5.0, 6.0, 1.5, 2.5, 3.5)

    compressed_path = tmp_path / "scan.nii.xz"
    with lzma.open(compressed_path, "wb") as handle:
        handle.write(image.to_bytes())

    assert nifti_metadata(compressed_path) == (4.0, 5.0, 6.0, 1.5, 2.5, 3.5)


def test_rendered_kernel_reads_dicom_file_reference_metadata(tmp_path: Path) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    dicom_metadata = namespace["dicom_metadata"]

    path = tmp_path / "slice.dcm"
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Rows = 128
    dataset.Columns = 256
    dataset.PixelSpacing = [0.7, 0.8]
    dataset.InstanceNumber = 3
    pydicom.dcmwrite(str(path), dataset)

    assert dicom_metadata(path) == (128.0, 256.0, 0.7, 0.8, 3.0)

    compressed_path = tmp_path / "slice.dcm.gz"
    with gzip.open(compressed_path, "wb") as handle:
        handle.write(path.read_bytes())

    assert dicom_metadata(compressed_path) == (128.0, 256.0, 0.7, 0.8, 3.0)

    ima_path = tmp_path / "slice.ima.gz"
    with gzip.open(ima_path, "wb") as handle:
        handle.write(path.read_bytes())

    assert dicom_metadata(ima_path) == (128.0, 256.0, 0.7, 0.8, 3.0)


def test_rendered_kernel_reads_nrrd_and_metaimage_header_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    build_file_asset_index = namespace["build_file_asset_index"]
    file_metadata_frame = namespace["file_metadata_frame"]
    medical_header_metadata = namespace["medical_header_metadata"]

    nrrd_path = tmp_path / "scan.nrrd.zst"
    nrrd_payload = "\n".join(
        [
            "NRRD0005",
            "type: float",
            "dimension: 3",
            "sizes: 4 5 6",
            "spacings: 1.5 2.5 3.5",
            "encoding: raw",
            "",
        ]
    ).encode("utf-8")
    nrrd_path.write_bytes(zstd.ZstdCompressor().compress(nrrd_payload))
    nhdr_path = tmp_path / "scan.nhdr"
    nhdr_path.write_bytes(nrrd_payload)

    mha_path = tmp_path / "volume.mha.gz"
    mha_payload = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "DimSize = 7 8 9",
            "ElementSpacing = 0.7 0.8 0.9",
            "ElementType = MET_FLOAT",
            "ElementDataFile = LOCAL",
            "",
        ]
    ).encode("utf-8")
    with gzip.open(mha_path, "wb") as handle:
        handle.write(mha_payload)

    assert medical_header_metadata(nrrd_path) == (4.0, 5.0, 6.0, 1.5, 2.5, 3.5)
    assert medical_header_metadata(nhdr_path) == (4.0, 5.0, 6.0, 1.5, 2.5, 3.5)
    assert medical_header_metadata(mha_path) == (7.0, 8.0, 9.0, 0.7, 0.8, 0.9)

    asset_index = build_file_asset_index([tmp_path])
    metadata = file_metadata_frame(pd.Series(["scan"]), asset_index, split="test", prefix="__file_scan")

    assert metadata.loc[0, "__file_scan_suffix"] == ".nrrd.zst"
    assert metadata.loc[0, "__file_scan_medical_dim_0"] == 4.0
    assert metadata.loc[0, "__file_scan_medical_spacing_2"] == 3.5


def test_rendered_kernel_reads_image_file_reference_metadata(tmp_path: Path) -> None:
    from PIL import Image

    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    image_metadata = namespace["image_metadata"]
    mask_image_metadata = namespace["mask_image_metadata"]
    image_intensity_metadata = namespace["image_intensity_metadata"]
    image_frame_count = namespace["image_frame_count"]
    build_file_asset_index = namespace["build_file_asset_index"]
    file_metadata_frame = namespace["file_metadata_frame"]

    path = tmp_path / "tile.png"
    Image.new("RGB", (20, 10), color=(1, 2, 3)).save(path)
    stack_path = tmp_path / "stack.tif"
    first_frame = Image.fromarray(np.zeros((4, 5), dtype=np.uint8))
    second_frame = Image.fromarray(np.ones((4, 5), dtype=np.uint8) * 3)
    first_frame.save(stack_path, save_all=True, append_images=[second_frame])
    mask_path = tmp_path / "mask.png"
    mask_array = np.array(
        [
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(mask_array).save(mask_path)

    assert image_metadata(path) == (20.0, 10.0, 3.0, 200.0, 2.0)
    assert image_frame_count(path) == 1.0
    assert image_frame_count(stack_path) == 2.0
    mean_intensity, std_intensity, nonzero_fraction = image_intensity_metadata(path)
    assert mean_intensity == pytest.approx(2.0)
    assert std_intensity == pytest.approx(0.0)
    assert nonzero_fraction == 1.0
    assert mask_image_metadata(mask_path) == (4.0, pytest.approx(4.0 / 12.0), 2.0)
    asset_index = build_file_asset_index([tmp_path])
    metadata = file_metadata_frame(pd.Series(["mask.png"]), asset_index, split="train", prefix="__file_mask")
    stack_metadata = file_metadata_frame(pd.Series(["stack.tif"]), asset_index, split="train", prefix="__file_image")
    assert stack_metadata.loc[0, "__file_image_image_frames"] == 2.0
    assert metadata.loc[0, "__file_mask_image_mean_intensity"] == pytest.approx(float(mask_array.mean()))
    assert metadata.loc[0, "__file_mask_image_std_intensity"] == pytest.approx(float(mask_array.std()))
    assert metadata.loc[0, "__file_mask_image_nonzero_fraction"] == pytest.approx(4.0 / 12.0)
    assert metadata.loc[0, "__file_mask_mask_nonzero_pixels"] == 4.0
    assert metadata.loc[0, "__file_mask_mask_coverage"] == pytest.approx(4.0 / 12.0)
    assert metadata.loc[0, "__file_mask_mask_labels"] == 2.0


def test_rendered_kernel_reads_audio_file_reference_metadata(tmp_path: Path) -> None:
    import soundfile as sf

    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    audio_metadata = namespace["audio_metadata"]

    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros((8000, 2), dtype=np.float32), 8000)

    assert audio_metadata(path) == (1.0, 8000.0, 2.0, 8000.0)


def test_rendered_kernel_reads_video_file_reference_metadata(tmp_path: Path) -> None:
    import cv2

    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    video_metadata = namespace["video_metadata"]

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment")
    for _ in range(3):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    width, height, fps, frames, seconds = video_metadata(path)

    assert width == 32.0
    assert height == 24.0
    assert fps == pytest.approx(5.0, rel=0.05)
    assert frames == pytest.approx(3.0)
    assert seconds == pytest.approx(0.6, rel=0.1)


def test_rendered_kernel_reads_point_cloud_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    point_cloud_metadata = namespace["point_cloud_metadata"]

    ply_path = tmp_path / "cloud.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 7",
                "property float x",
                "property float y",
                "property float z",
                "element face 2",
                "property list uchar int vertex_indices",
                "end_header",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    obj_path = tmp_path / "mesh.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    xyz_path = tmp_path / "points.xyz"
    xyz_path.write_text("0 0 0\n1 1 1\n2 2 2\n", encoding="utf-8")
    gz_ply_path = tmp_path / "cloud.ply.gz"
    with gzip.open(gz_ply_path, "wt", encoding="ascii") as handle:
        handle.write(ply_path.read_text(encoding="ascii"))
    xz_obj_path = tmp_path / "mesh.obj.xz"
    with lzma.open(xz_obj_path, "wt", encoding="utf-8") as handle:
        handle.write("v 0 0 0\nv 1 0 0\nf 1 2\n")
    bz2_xyz_path = tmp_path / "points.xyz.bz2"
    with bz2.open(bz2_xyz_path, "wt", encoding="utf-8") as handle:
        handle.write("0 0 0\n1 1 1\n")

    assert point_cloud_metadata(ply_path) == (7.0, 2.0)
    assert point_cloud_metadata(obj_path) == (3.0, 1.0)
    xyz_points, xyz_faces = point_cloud_metadata(xyz_path)
    assert xyz_points == 3.0
    assert np.isnan(xyz_faces)
    assert point_cloud_metadata(gz_ply_path) == (7.0, 2.0)
    assert point_cloud_metadata(xz_obj_path) == (2.0, 1.0)
    bz2_xyz_points, bz2_xyz_faces = point_cloud_metadata(bz2_xyz_path)
    assert bz2_xyz_points == 2.0
    assert np.isnan(bz2_xyz_faces)


def test_rendered_kernel_reads_annotation_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    annotation_metadata = namespace["annotation_metadata"]
    build_file_asset_index = namespace["build_file_asset_index"]
    file_metadata_frame = namespace["file_metadata_frame"]

    coco_payload = {
        "images": [{"id": 1}, {"id": 2}],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 20], "segmentation": [[0, 0, 1, 0, 1, 1]]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [3, 4, 5, 6]},
        ],
    }
    json_path = tmp_path / "instances.json"
    json_path.write_text(json.dumps(coco_payload), encoding="utf-8")
    gz_json_path = tmp_path / "instances.json.gz"
    with gzip.open(gz_json_path, "wt", encoding="utf-8") as handle:
        json.dump(coco_payload, handle)
    labelme_path = tmp_path / "labelme.json"
    labelme_path.write_text(
        json.dumps(
            {
                "imagePath": "image_001.png",
                "shapes": [
                    {"label": "scratch", "points": [[0, 0], [1, 0], [1, 1]]},
                    {"label": "dent", "bbox": [0, 0, 2, 2]},
                ],
            }
        ),
        encoding="utf-8",
    )
    yolo_path = tmp_path / "image_001.txt"
    yolo_path.write_text("0 0.5 0.5 0.2 0.3\n2 0.2 0.3 0.1 0.2\n", encoding="utf-8")
    gz_yolo_seg_path = tmp_path / "image_002.txt.gz"
    with gzip.open(gz_yolo_seg_path, "wt", encoding="utf-8") as handle:
        handle.write("1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    plain_text_path = tmp_path / "notes.txt"
    plain_text_path.write_text("not a YOLO label file\nsecond line\n", encoding="utf-8")

    assert annotation_metadata(json_path) == (2.0, 2.0, 2.0, 2.0, 1.0)
    assert annotation_metadata(gz_json_path) == (2.0, 2.0, 2.0, 2.0, 1.0)
    assert annotation_metadata(labelme_path) == (2.0, 1.0, 2.0, 1.0, 1.0)
    yolo_annotations, yolo_images, yolo_categories, yolo_bboxes, yolo_segmentations = annotation_metadata(yolo_path)
    assert yolo_annotations == 2.0
    assert np.isnan(yolo_images)
    assert yolo_categories == 2.0
    assert yolo_bboxes == 2.0
    assert yolo_segmentations == 0.0
    seg_annotations, seg_images, seg_categories, seg_bboxes, seg_segmentations = annotation_metadata(gz_yolo_seg_path)
    assert seg_annotations == 1.0
    assert np.isnan(seg_images)
    assert seg_categories == 1.0
    assert seg_bboxes == 0.0
    assert seg_segmentations == 1.0
    assert all(np.isnan(value) for value in annotation_metadata(plain_text_path))

    asset_index = build_file_asset_index([tmp_path])
    metadata = file_metadata_frame(
        pd.Series(["instances.json"]), asset_index, split="train", prefix="__file_annotation"
    )
    assert metadata.loc[0, "__file_annotation_annotation_count"] == 2.0
    assert metadata.loc[0, "__file_annotation_annotation_bboxes"] == 2.0
    assert metadata.loc[0, "__file_annotation_annotation_segmentations"] == 1.0


def test_rendered_kernel_reads_graph_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    graph_metadata = namespace["graph_metadata"]

    graphml_path = tmp_path / "network.graphml"
    graphml_path.write_text(
        """<graphml><graph><node id="a"/><node id="b"/><edge source="a" target="b"/></graph></graphml>""",
        encoding="utf-8",
    )
    gexf_path = tmp_path / "network.gexf"
    gexf_path.write_text(
        '<gexf><graph><nodes><node id="a"/><node id="b"/></nodes>'
        '<edges><edge source="a" target="b"/></edges></graph></gexf>',
        encoding="utf-8",
    )
    gml_path = tmp_path / "network.gml"
    gml_path.write_text(
        "graph [\n  node [ id 1 ]\n  node [ id 2 ]\n  edge [ source 1 target 2 ]\n]\n", encoding="utf-8"
    )
    mtx_path = tmp_path / "network.mtx"
    mtx_path.write_text("%%MatrixMarket matrix coordinate integer general\n% comment\n4 5 6\n", encoding="utf-8")
    edges_path = tmp_path / "network.edgelist"
    edges_path.write_text("a b\nb c\nc a\n", encoding="utf-8")
    gz_graphml_path = tmp_path / "network.graphml.gz"
    with gzip.open(gz_graphml_path, "wt", encoding="utf-8") as handle:
        handle.write("<graphml><graph><node id='a'/><node id='b'/><edge source='a' target='b'/></graph></graphml>")
    xz_mtx_path = tmp_path / "network.mtx.xz"
    with lzma.open(xz_mtx_path, "wt", encoding="utf-8") as handle:
        handle.write("%%MatrixMarket matrix coordinate integer general\n% comment\n3 7 4\n")
    bz2_edges_path = tmp_path / "network.edges.bz2"
    with bz2.open(bz2_edges_path, "wt", encoding="utf-8") as handle:
        handle.write("u v\nv w\n")

    assert graph_metadata(graphml_path) == (2.0, 1.0)
    assert graph_metadata(gexf_path) == (2.0, 1.0)
    assert graph_metadata(gml_path) == (2.0, 1.0)
    assert graph_metadata(mtx_path) == (5.0, 6.0)
    assert graph_metadata(edges_path) == (3.0, 3.0)
    assert graph_metadata(gz_graphml_path) == (2.0, 1.0)
    assert graph_metadata(xz_mtx_path) == (7.0, 4.0)
    assert graph_metadata(bz2_edges_path) == (3.0, 2.0)


def test_rendered_kernel_reads_geospatial_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    geospatial_metadata = namespace["geospatial_metadata"]

    geojson_path = tmp_path / "parcels.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"id": 2},
                        "geometry": {"type": "LineString", "coordinates": [[3.0, 4.0], [5.0, 6.0]]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    kmz_path = tmp_path / "places.kmz"
    _write_kmz(kmz_path, _kml_payload([(1, 10.0, None), (2, 20.0, None)]))
    gz_kml_path = tmp_path / "places.kml.gz"
    with gzip.open(gz_kml_path, "wt", encoding="utf-8") as handle:
        handle.write(_kml_payload([(1, 3.0, None), (2, 7.0, None)]))

    assert geospatial_metadata(geojson_path) == (2.0, 1.0, 2.0, 5.0, 6.0)
    assert geospatial_metadata(kmz_path) == (2.0, 10.0, 1.0, 20.0, 2.0)
    assert geospatial_metadata(gz_kml_path) == (2.0, 3.0, 1.0, 7.0, 2.0)


def test_rendered_kernel_reads_document_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    document_metadata = namespace["document_metadata"]

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Pages /Count 2 >> endobj\n"
        b"2 0 obj << /Type /Page /Parent 1 0 R >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 1 0 R >> endobj\n"
    )
    docx_path = tmp_path / "notes.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>"
                "<w:p><w:r><w:t>Hello world</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>Second line</w:t></w:r></w:p>"
                "</w:body>"
                "</w:document>"
            ),
        )
    markdown_path = tmp_path / "brief.md"
    markdown_text = "# Title\n\nAlpha beta gamma.\n"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    text_path = tmp_path / "transcript.txt"
    transcript_text = "Hello world\n\nSecond line\n"
    text_path.write_text(transcript_text, encoding="utf-8")
    gz_text_path = tmp_path / "transcript.txt.gz"
    gz_transcript_text = "Compressed transcript\n\nThird line\n"
    with gzip.open(gz_text_path, "wt", encoding="utf-8") as handle:
        handle.write(gz_transcript_text)
    subtitle_path = tmp_path / "captions.srt"
    subtitle_text = "Hello subtitle\n\nSecond subtitle\n"
    subtitle_path.write_text(subtitle_text, encoding="utf-8")
    vtt_path = tmp_path / "captions.vtt"
    vtt_text = "WEBVTT\n\nHello caption\n\nSecond caption\n"
    vtt_path.write_text(vtt_text, encoding="utf-8")
    gz_subtitle_path = tmp_path / "captions.srt.gz"
    gz_subtitle_text = "Compressed subtitle\n\nSecond subtitle\n"
    with gzip.open(gz_subtitle_path, "wt", encoding="utf-8") as handle:
        handle.write(gz_subtitle_text)
    gz_html_path = tmp_path / "brief.html.gz"
    with gzip.open(gz_html_path, "wt", encoding="utf-8") as handle:
        handle.write("<html><body><p>Hello <b>compressed</b></p><script>ignored()</script></body></html>")

    pdf_pages, pdf_chars, pdf_words, pdf_paragraphs = document_metadata(pdf_path)
    assert pdf_pages == 2.0
    assert np.isnan(pdf_chars)
    assert np.isnan(pdf_words)
    assert np.isnan(pdf_paragraphs)
    docx_pages, docx_chars, docx_words, docx_paragraphs = document_metadata(docx_path)
    assert np.isnan(docx_pages)
    assert docx_chars == 23.0
    assert docx_words == 4.0
    assert docx_paragraphs == 2.0
    md_pages, md_chars, md_words, md_paragraphs = document_metadata(markdown_path)
    assert np.isnan(md_pages)
    assert md_chars == float(len(markdown_text))
    assert md_words == 4.0
    assert md_paragraphs == 2.0
    text_pages, text_chars, text_words, text_paragraphs = document_metadata(text_path)
    assert np.isnan(text_pages)
    assert text_chars == float(len(transcript_text))
    assert text_words == 4.0
    assert text_paragraphs == 2.0
    gz_pages, gz_chars, gz_words, gz_paragraphs = document_metadata(gz_text_path)
    assert np.isnan(gz_pages)
    assert gz_chars == float(len(gz_transcript_text))
    assert gz_words == 4.0
    assert gz_paragraphs == 2.0
    subtitle_pages, subtitle_chars, subtitle_words, subtitle_paragraphs = document_metadata(subtitle_path)
    assert np.isnan(subtitle_pages)
    assert subtitle_chars == float(len(subtitle_text))
    assert subtitle_words == 4.0
    assert subtitle_paragraphs == 2.0
    vtt_pages, vtt_chars, vtt_words, vtt_paragraphs = document_metadata(vtt_path)
    assert np.isnan(vtt_pages)
    assert vtt_chars == float(len(vtt_text))
    assert vtt_words == 5.0
    assert vtt_paragraphs == 3.0
    gz_subtitle_pages, gz_subtitle_chars, gz_subtitle_words, gz_subtitle_paragraphs = document_metadata(
        gz_subtitle_path
    )
    assert np.isnan(gz_subtitle_pages)
    assert gz_subtitle_chars == float(len(gz_subtitle_text))
    assert gz_subtitle_words == 4.0
    assert gz_subtitle_paragraphs == 2.0
    gz_html_pages, gz_html_chars, gz_html_words, gz_html_paragraphs = document_metadata(gz_html_path)
    assert np.isnan(gz_html_pages)
    assert gz_html_chars > 0
    assert gz_html_words == 2.0
    assert gz_html_paragraphs == 1.0


def test_rendered_kernel_reads_array_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    array_metadata = namespace["array_metadata"]

    npy_path = tmp_path / "embedding.npy"
    np.save(npy_path, np.zeros((2, 3, 4), dtype=np.float32))
    npz_path = tmp_path / "features.npz"
    np.savez(npz_path, small=np.zeros((2,), dtype=np.float32), large=np.zeros((4, 5), dtype=np.int16))
    zarr_path = tmp_path / "tiles.zarr"
    zarr_path.mkdir()
    (zarr_path / ".zarray").write_text(json.dumps({"shape": [6, 7], "chunks": [3, 7]}), encoding="utf-8")
    ome_zarr_path = tmp_path / "labels.ome.zarr"
    (ome_zarr_path / "0").mkdir(parents=True)
    (ome_zarr_path / "zarr.json").write_text(
        json.dumps({"attributes": {"ome": {"version": "0.5"}}}),
        encoding="utf-8",
    )
    (ome_zarr_path / "0" / "zarr.json").write_text(
        json.dumps({"shape": [8, 9, 10], "data_type": "uint16"}),
        encoding="utf-8",
    )
    n5_path = tmp_path / "volumes.n5"
    n5_path.mkdir()
    (n5_path / "attributes.json").write_text(
        json.dumps({"dimensions": [11, 12, 13], "dataType": "uint16"}),
        encoding="utf-8",
    )

    assert array_metadata(npy_path) == (1.0, 3.0, 2.0, 3.0, 4.0, 24.0)
    npz_count, npz_ndim, npz_dim_0, npz_dim_1, npz_dim_2, npz_elements = array_metadata(npz_path)
    assert npz_count == 2.0
    assert npz_ndim == 2.0
    assert npz_dim_0 == 4.0
    assert npz_dim_1 == 5.0
    assert np.isnan(npz_dim_2)
    assert npz_elements == 20.0
    zarr_count, zarr_ndim, zarr_dim_0, zarr_dim_1, zarr_dim_2, zarr_elements = array_metadata(zarr_path)
    assert zarr_count == 1.0
    assert zarr_ndim == 2.0
    assert zarr_dim_0 == 6.0
    assert zarr_dim_1 == 7.0
    assert np.isnan(zarr_dim_2)
    assert zarr_elements == 42.0
    ome_count, ome_ndim, ome_dim_0, ome_dim_1, ome_dim_2, ome_elements = array_metadata(ome_zarr_path)
    assert ome_count == 1.0
    assert ome_ndim == 3.0
    assert ome_dim_0 == 8.0
    assert ome_dim_1 == 9.0
    assert ome_dim_2 == 10.0
    assert ome_elements == 720.0
    n5_count, n5_ndim, n5_dim_0, n5_dim_1, n5_dim_2, n5_elements = array_metadata(n5_path)
    assert n5_count == 1.0
    assert n5_ndim == 3.0
    assert n5_dim_0 == 11.0
    assert n5_dim_1 == 12.0
    assert n5_dim_2 == 13.0
    assert n5_elements == 1716.0


def test_rendered_kernel_reads_scientific_array_file_reference_metadata(tmp_path: Path) -> None:
    from astropy.io import fits

    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    array_metadata = namespace["array_metadata"]

    netcdf_path = tmp_path / "sensor.nc"
    _write_netcdf_table(
        netcdf_path,
        {
            "series": np.zeros((2, 3), dtype=np.float32),
            "id": np.array([1, 2], dtype=np.int32),
        },
    )
    fits_path = tmp_path / "image.fits"
    fits.PrimaryHDU(np.zeros((4, 5), dtype=np.float32)).writeto(fits_path)
    gz_fits_path = tmp_path / "image.fits.gz"
    fits.PrimaryHDU(np.zeros((3, 4, 5), dtype=np.float32)).writeto(gz_fits_path)
    mat_path = tmp_path / "matrix.mat"
    savemat(mat_path, {"matrix": np.zeros((6, 7), dtype=np.float64)})
    h5ad_path = tmp_path / "cells.h5ad"
    _write_h5ad_table(
        h5ad_path,
        ids=np.array([b"cell_1", b"cell_2"]),
        features=np.zeros((2, 3), dtype=np.float32),
    )

    nc_count, nc_ndim, nc_dim_0, nc_dim_1, nc_dim_2, nc_elements = array_metadata(netcdf_path)
    assert nc_count == 2.0
    assert nc_ndim == 2.0
    assert nc_dim_0 == 2.0
    assert nc_dim_1 == 3.0
    assert np.isnan(nc_dim_2)
    assert nc_elements == 6.0
    fits_count, fits_ndim, fits_dim_0, fits_dim_1, fits_dim_2, fits_elements = array_metadata(fits_path)
    assert fits_count == 1.0
    assert fits_ndim == 2.0
    assert fits_dim_0 == 4.0
    assert fits_dim_1 == 5.0
    assert np.isnan(fits_dim_2)
    assert fits_elements == 20.0
    gz_fits_count, gz_fits_ndim, gz_fits_dim_0, gz_fits_dim_1, gz_fits_dim_2, gz_fits_elements = array_metadata(
        gz_fits_path
    )
    assert gz_fits_count == 1.0
    assert gz_fits_ndim == 3.0
    assert gz_fits_dim_0 == 3.0
    assert gz_fits_dim_1 == 4.0
    assert gz_fits_dim_2 == 5.0
    assert gz_fits_elements == 60.0
    mat_count, mat_ndim, mat_dim_0, mat_dim_1, mat_dim_2, mat_elements = array_metadata(mat_path)
    assert mat_count == 1.0
    assert mat_ndim == 2.0
    assert mat_dim_0 == 6.0
    assert mat_dim_1 == 7.0
    assert np.isnan(mat_dim_2)
    assert mat_elements == 42.0
    h5ad_count, h5ad_ndim, h5ad_dim_0, h5ad_dim_1, h5ad_dim_2, h5ad_elements = array_metadata(h5ad_path)
    assert h5ad_count >= 3.0
    assert h5ad_ndim == 2.0
    assert h5ad_dim_0 == 2.0
    assert h5ad_dim_1 == 3.0
    assert np.isnan(h5ad_dim_2)
    assert h5ad_elements == 6.0


def test_rendered_kernel_reads_model_artifact_file_reference_metadata(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper
    from safetensors.numpy import save_file as save_safetensors

    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    build_file_asset_index = namespace["build_file_asset_index"]
    file_metadata_frame = namespace["file_metadata_frame"]
    model_artifact_metadata = namespace["model_artifact_metadata"]
    model_sidecar_metadata = namespace["model_sidecar_metadata"]

    onnx_path = tmp_path / "model.onnx"
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [2], [1.0, 2.0])
    node = helper.make_node("Add", ["input", "weight"], ["output"])
    graph = helper.make_graph([node], "demo", [input_info], [output_info], [weight])
    onnx.save(helper.make_model(graph), onnx_path)
    safetensors_path = tmp_path / "weights.safetensors"
    save_safetensors(
        {
            "layer.weight": np.zeros((2, 3), dtype=np.float32),
            "layer.bias": np.zeros((3,), dtype=np.float32),
        },
        safetensors_path,
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1234},
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "model": {"vocab": {"hello": 0, "world": 1}},
                "added_tokens": [{"id": 2, "content": "<pad>"}],
            }
        ),
        encoding="utf-8",
    )
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("hello\nworld\n\n<pad>\n", encoding="utf-8")
    adapter_config_path = tmp_path / "adapter_config.json"
    adapter_config_path.write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "target_modules": ["q_proj", "v_proj"]}),
        encoding="utf-8",
    )

    assert model_artifact_metadata(onnx_path) == (1.0, 1.0, 1.0, 1.0, 2.0)
    nodes, inputs, outputs, tensors, parameters = model_artifact_metadata(safetensors_path)
    assert np.isnan(nodes)
    assert np.isnan(inputs)
    assert np.isnan(outputs)
    assert tensors == 2.0
    assert parameters == 9.0
    index_nodes, index_inputs, index_outputs, index_tensors, index_parameters = model_artifact_metadata(index_path)
    assert np.isnan(index_nodes)
    assert np.isnan(index_inputs)
    assert np.isnan(index_outputs)
    assert index_tensors == 2.0
    assert index_parameters == 1234.0
    sidecar_keys, sidecar_vocab, sidecar_added_tokens = model_sidecar_metadata(tokenizer_path)
    assert sidecar_keys == 2.0
    assert sidecar_vocab == 2.0
    assert sidecar_added_tokens == 1.0
    vocab_keys, vocab_size, vocab_added_tokens = model_sidecar_metadata(vocab_path)
    assert np.isnan(vocab_keys)
    assert vocab_size == 3.0
    assert np.isnan(vocab_added_tokens)
    adapter_keys, adapter_vocab, adapter_added_tokens = model_sidecar_metadata(adapter_config_path)
    assert adapter_keys == 3.0
    assert np.isnan(adapter_vocab)
    assert np.isnan(adapter_added_tokens)

    asset_index = build_file_asset_index([tmp_path])
    metadata = file_metadata_frame(
        pd.Series(["model.safetensors.index.json", "tokenizer.json", "vocab.txt", "adapter_config.json"]),
        asset_index,
        split="test",
        prefix="__file_model",
    )
    assert metadata.loc[0, "__file_model_suffix"] == ".safetensors.index.json"
    assert metadata.loc[0, "__file_model_model_tensors"] == 2.0
    assert metadata.loc[0, "__file_model_model_parameters"] == 1234.0
    assert metadata.loc[1, "__file_model_model_sidecar_vocab"] == 2.0
    assert metadata.loc[1, "__file_model_model_sidecar_added_tokens"] == 1.0
    assert metadata.loc[2, "__file_model_model_sidecar_vocab"] == 3.0
    assert metadata.loc[3, "__file_model_model_sidecar_keys"] == 3.0


def test_rendered_kernel_reads_bio_sequence_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    bio_sequence_metadata = namespace["bio_sequence_metadata"]

    fasta_path = tmp_path / "proteins.fa.gz"
    with gzip.open(fasta_path, "wt", encoding="utf-8") as handle:
        handle.write(">a\nACDE\nFG\n>b\nMNPQ\n")
    fastq_path = tmp_path / "reads.fastq"
    fastq_path.write_text("@r1\nACGT\n+\n!!!!\n@r2\nACGTA\n+\n!!!!!\n", encoding="utf-8")
    xz_fasta_path = tmp_path / "genome.fna.xz"
    with lzma.open(xz_fasta_path, "wt", encoding="utf-8") as handle:
        handle.write(">chr1\nACGTAC\n>chr2\nTT\n")
    bz2_fastq_path = tmp_path / "reads.fastq.bz2"
    with bz2.open(bz2_fastq_path, "wt", encoding="utf-8") as handle:
        handle.write("@r1\nACG\n+\n!!!\n@r2\nACGTAC\n+\n!!!!!!\n")

    assert bio_sequence_metadata(fasta_path) == (2.0, 10.0, 4.0, 5.0, 6.0)
    assert bio_sequence_metadata(fastq_path) == (2.0, 9.0, 4.0, 4.5, 5.0)
    assert bio_sequence_metadata(xz_fasta_path) == (2.0, 8.0, 2.0, 4.0, 6.0)
    assert bio_sequence_metadata(bz2_fastq_path) == (2.0, 9.0, 3.0, 4.5, 6.0)


def test_rendered_kernel_reads_bio_structure_file_reference_metadata(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    bio_structure_metadata = namespace["bio_structure_metadata"]

    pdb_path = tmp_path / "protein.pdb"
    pdb_text = (
        "ATOM      1  N   ALA A   1      11.104  13.207   9.120  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  13.100   9.000  1.00 20.00           C\n"
        "HETATM    3  O   HOH A   2      13.000  13.500   9.500  1.00 20.00           O\n"
    )
    pdb_path.write_text(pdb_text, encoding="utf-8")
    sdf_path = tmp_path / "molecule.sdf"
    sdf_text = (
        "\n".join(
            [
                "demo",
                "  kagglebot",
                "",
                "  3  2  0  0  0  0            999 V2000",
                "    0.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "    1.0    0.0    0.0 O   0  0  0  0  0  0  0  0  0  0  0  0",
                "    0.0    1.0    0.0 N   0  0  0  0  0  0  0  0  0  0  0  0",
                "  1  2  1  0  0  0  0",
                "  2  3  1  0  0  0  0",
                "M  END",
                "$$$$",
            ]
        )
        + "\n"
    )
    sdf_path.write_text(sdf_text, encoding="utf-8")
    gz_pdb_path = tmp_path / "protein.pdb.gz"
    with gzip.open(gz_pdb_path, "wt", encoding="utf-8") as handle:
        handle.write(pdb_text)
    xz_sdf_path = tmp_path / "molecule.sdf.xz"
    with lzma.open(xz_sdf_path, "wt", encoding="utf-8") as handle:
        handle.write(sdf_text)

    atoms, residues, bonds, molecules = bio_structure_metadata(pdb_path)
    assert atoms == 3.0
    assert residues == 2.0
    assert np.isnan(bonds)
    assert molecules == 1.0
    sdf_atoms, sdf_residues, sdf_bonds, sdf_molecules = bio_structure_metadata(sdf_path)
    assert sdf_atoms == 3.0
    assert np.isnan(sdf_residues)
    assert sdf_bonds == 2.0
    assert sdf_molecules == 1.0
    gz_atoms, gz_residues, gz_bonds, gz_molecules = bio_structure_metadata(gz_pdb_path)
    assert gz_atoms == 3.0
    assert gz_residues == 2.0
    assert np.isnan(gz_bonds)
    assert gz_molecules == 1.0
    xz_atoms, xz_residues, xz_bonds, xz_molecules = bio_structure_metadata(xz_sdf_path)
    assert xz_atoms == 3.0
    assert np.isnan(xz_residues)
    assert xz_bonds == 2.0
    assert xz_molecules == 1.0


@pytest.mark.parametrize(
    ("format_text", "expected_suffix"),
    [
        ("## Submission Format\nParticipants must upload an HDF5 file with columns id,target.", ".hdf5"),
        ("## Submission Format\nParticipants must upload an ORC file with columns id,target.", ".orc"),
        ("## Submission Format\nParticipants must upload a Stata file with columns id,target.", ".dta"),
        ("## Submission Format\nParticipants must upload a Pickle file with columns id,target.", ".pkl"),
        ("## Submission Format\nParticipants must upload a zstd-compressed JSON Lines file.", ".jsonl.zst"),
        ("## Submission Format\nParticipants must upload an EPUB document for scoring.", ".csv"),
        ("## Submission Format\nParticipants must upload a zstd-compressed LaTeX file.", ".csv"),
    ],
)
def test_rendered_kernel_synthesized_sample_suffix_uses_shared_tabular_token_patterns(
    format_text: str,
    expected_suffix: str,
) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)

    assert namespace["synthesized_sample_suffix"](format_text) == expected_suffix


def test_render_kernel_main_injects_shared_asset_suffixes(tmp_path: Path) -> None:
    script = kaggle_notebook.render_kernel_main("demo", "cpu")
    namespace = {"__name__": "kagglebot_test"}

    compile(script, "main.py", "exec")
    exec(script, namespace)
    assert "__DATA_ASSET_SUFFIXES_JSON__" not in script
    assert "__NON_TABULAR_SUBMISSION_SUFFIXES_JSON__" not in script
    assert "__MODEL_ARTIFACT_COMPOUND_SUFFIXES_JSON__" not in script
    assert "__MODEL_ARTIFACT_FILENAMES_JSON__" not in script
    assert "__MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES_JSON__" not in script
    assert "__MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES_JSON__" not in script
    assert "__ASSET_COMPRESSION_SUFFIXES_JSON__" not in script
    assert "__TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES_JSON__" not in script
    assert "__TABULAR_TSV_LIKE_SUFFIX_PREFIXES_JSON__" not in script
    assert "TABULAR_TSV_LIKE_SUFFIX_PREFIXES" not in script
    assert "__ASSET_COLLECTION_DIR_NAMES_JSON__" not in script
    assert "__ARCHIVE_SUFFIXES_JSON__" not in script
    assert "__ARCHIVE_SUBMISSION_SUFFIXES_JSON__" not in script
    assert "__ZSTD_TAR_ARCHIVE_SUFFIXES_JSON__" not in script
    assert "__DIRECTORY_ASSET_SUFFIXES_JSON__" not in script
    assert "__IMAGE_SUFFIXES_JSON__" not in script
    assert "__AUDIO_SUFFIXES_JSON__" not in script
    assert "__VIDEO_SUFFIXES_JSON__" not in script
    assert "__DICOM_BASE_SUFFIXES_JSON__" not in script
    assert "__DICOM_SUFFIXES_JSON__" not in script
    assert "__NIFTI_BASE_SUFFIXES_JSON__" not in script
    assert "__NIFTI_SUFFIXES_JSON__" not in script
    assert "__MEDICAL_HEADER_BASE_SUFFIXES_JSON__" not in script
    assert "__MEDICAL_HEADER_SUFFIXES_JSON__" not in script
    assert "__ROLE_SUFFIXES_JSON__" not in script
    assert "__ROLE_ALIASES_JSON__" not in script
    assert "__TEST_DIRECT_ROLE_ALIASES_JSON__" not in script
    assert "__ROLE_TRAILING_PREFIXES_JSON__" not in script
    assert "__SAMPLE_OUTPUT_NAME_TOKENS_JSON__" not in script
    assert "__SAMPLE_COMPACT_NAME_ALIASES_JSON__" not in script
    assert "__FILE_REFERENCE_NAME_TOKENS_JSON__" not in script
    assert "__TEXT_PREDICTION_NAME_TOKENS_JSON__" not in script
    assert "__RLE_SEGMENTATION_COLUMN_TOKENS_JSON__" not in script
    assert "__ID_LIKE_COLUMN_NAMES_JSON__" not in script
    assert "__ASSET_LABEL_TABLE_TOKENS_JSON__" not in script
    for suffix in (
        ".nii.gz",
        ".wav",
        ".mp4",
        ".abc",
        ".blend",
        ".glb",
        ".mat",
        ".fits.gz",
        ".onnx",
        ".zarr",
        ".ply.gz",
        ".obj.xz",
        ".xyz.bz2",
        ".usda.gz",
        ".usdz",
        ".kml.gz",
        ".srt",
        ".vtt",
        ".srt.gz",
        ".html.gz",
        ".fastq.bz2",
        ".fna.xz",
        ".pdb.gz",
        ".sdf.xz",
        ".graphml.gz",
        ".mtx.xz",
        ".edges.bz2",
        ".safetensors",
    ):
        assert suffix in NON_TABULAR_SUBMISSION_SUFFIXES
        assert f'"{suffix}"' in script
    assert namespace["FILE_ASSET_SUFFIXES"] == set(DATA_ASSET_SUFFIXES) | set(MODEL_ARTIFACT_COMPOUND_SUFFIXES)
    assert namespace["SUBMISSION_FILE_ASSET_SUFFIXES"] == set(NON_TABULAR_SUBMISSION_SUFFIXES)
    assert namespace["MODEL_ARTIFACT_COMPOUND_SUFFIXES"] == set(MODEL_ARTIFACT_COMPOUND_SUFFIXES)
    assert namespace["MODEL_ARTIFACT_FILENAMES"] == set(MODEL_ARTIFACT_FILENAMES)
    assert namespace["MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES"] == set(MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES)
    assert namespace["MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES"] == set(MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES)
    assert namespace["file_asset_suffix"](Path("model.safetensors.index.json")) == ".safetensors.index.json"
    assert namespace["file_asset_stem"](Path("model.safetensors.index.json")) == "model"
    assert namespace["is_file_asset_path"](Path("tokenizer.model")) is False
    tokenizer_model = tmp_path / "tokenizer.model"
    tokenizer_model.write_text("tokenizer", encoding="utf-8")
    assert namespace["is_file_asset_path"](tokenizer_model) is True
    assert ".jp2" in IMAGE_SUFFIXES
    assert ".jxl" in IMAGE_SUFFIXES
    assert '".jp2"' in script
    assert '".jxl"' in script
    assert namespace["AUDIO_FILE_SUFFIXES"] == set(AUDIO_SUFFIXES)
    assert namespace["VIDEO_FILE_SUFFIXES"] == set(VIDEO_SUFFIXES)
    assert namespace["DICOM_FILE_BASE_SUFFIXES"] == set(DICOM_IMAGE_BASE_SUFFIXES)
    assert namespace["DICOM_FILE_SUFFIXES"] == set(DICOM_IMAGE_SUFFIXES)
    assert namespace["NIFTI_FILE_BASE_SUFFIXES"] == set(NIFTI_IMAGE_BASE_SUFFIXES)
    assert namespace["NIFTI_FILE_SUFFIXES"] == set(NIFTI_IMAGE_SUFFIXES)
    assert namespace["MEDICAL_HEADER_FILE_BASE_SUFFIXES"] == set(MEDICAL_HEADER_IMAGE_BASE_SUFFIXES)
    assert namespace["MEDICAL_HEADER_FILE_SUFFIXES"] == set(MEDICAL_HEADER_IMAGE_SUFFIXES)
    assert namespace["ZSTD_TAR_ARCHIVE_SUFFIXES"] == set(ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES)
    assert "ZSTD_TAR_ARCHIVE_SUFFIXES = set([" in script
    assert '".mid"' in script
    assert '".opus"' in script
    assert '".mpg"' in script
    assert '".webm"' in script
    assert "ROLE_SUFFIXES = set([" in script
    for role_suffix in ROLE_SUFFIXES:
        assert f'"{role_suffix}"' in script
    for token in FILE_REFERENCE_NAME_TOKENS + TEXT_PREDICTION_NAME_TOKENS:
        assert f'"{token}"' in script
    for token in RLE_SEGMENTATION_COLUMN_TOKENS:
        assert f'"{token}"' in script
    assert namespace["FILE_REFERENCE_NAME_TOKENS"] == tuple(FILE_REFERENCE_NAME_TOKENS)
    assert namespace["TEXT_PREDICTION_NAME_TOKENS"] == tuple(TEXT_PREDICTION_NAME_TOKENS)
    assert namespace["RLE_SEGMENTATION_COLUMN_TOKENS"] == set(RLE_SEGMENTATION_COLUMN_TOKENS)
    assert namespace["ID_LIKE_COLUMN_NAMES"] == set(ID_LIKE_COLUMN_NAMES)
    assert namespace["ASSET_LABEL_TABLE_TOKENS"] == tuple(ASSET_LABEL_TABLE_TOKENS)
    assert namespace["ASSET_COMPRESSION_SUFFIXES"] == tuple(ASSET_COMPRESSION_SUFFIXES)
    assert namespace["compression_suffix_for"](".jsonl.zst") == ".zst"
    assert namespace["compression_suffix_for"](".csv") is None
    assert namespace["ASSET_COLLECTION_DIR_NAMES"] == set(ASSET_COLLECTION_DIR_NAMES)
    assert namespace["TEST_DIRECT_ROLE_ALIASES"] == set(TEST_DIRECT_ROLE_ALIASES)
    assert namespace["ROLE_TRAILING_PREFIXES"] == set(ROLE_TRAILING_PREFIXES)
    assert namespace["SAMPLE_OUTPUT_NAME_TOKENS"] == set(SAMPLE_OUTPUT_NAME_TOKENS)
    assert namespace["SAMPLE_COMPACT_NAME_ALIASES"] == set(SAMPLE_COMPACT_NAME_ALIASES)
    assert namespace["configured_submission_filename_is_template"]("sample_solution.csv")
    assert namespace["configured_submission_filename_is_template"]("solutions_template.jsonl")
    assert "for suffix in sorted(FILE_ASSET_SUFFIXES, key=len, reverse=True):" in script
    assert "def archive_output_suffix(path: Path) -> str:" in script
    assert "archive_path.suffix.lower()" not in script
    assert "def is_file_asset_path(path: Path) -> bool:" in script
    assert {".n5", ".ome.zarr", ".zarr"} <= DIRECTORY_ARRAY_SUFFIXES
    assert 'DIRECTORY_ASSET_SUFFIXES = set([".n5", ".ome.zarr", ".zarr"])' in script
    for suffix in ARCHIVE_SUBMISSION_SUFFIXES:
        assert f'"{suffix}"' in script
    assert namespace["ARCHIVE_OUTPUT_SUFFIXES"] == set(ARCHIVE_SUBMISSION_SUFFIXES)
    assert "path.is_dir() and suffix in DIRECTORY_ASSET_SUFFIXES" in script
    assert "file_asset_suffix(path) not in IMAGE_FILE_SUFFIXES" in script


def test_rendered_kernel_uses_shared_tab_delimited_suffixes_for_text_defaults() -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)

    assert namespace["default_delimited_text_separator"](".tab") == "\t"
    assert namespace["default_delimited_text_separator"](".txt.zst") == "\t"
    assert namespace["default_delimited_text_separator"](".psv") == "|"
    assert namespace["default_delimited_text_separator"](".csv.gz") == ","


@pytest.mark.parametrize(
    "column",
    [
        "protein_id",
        "molecule_id",
        "sequence_id",
        "geo_id",
        "geojson_id",
        "network_id",
        "embedding_id",
        "matrix_id",
        "mask_id",
        "annotation_id",
        "bbox_id",
        "segmentation_id",
        "text_id",
        "transcript_id",
        "caption_id",
        "prompt_id",
        "subtitle_id",
        "transcription_id",
    ],
)
def test_rendered_kernel_detects_stem_only_asset_reference_column_names(column: str) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    looks_like_file_reference_column = namespace["looks_like_file_reference_column"]

    frame = pd.DataFrame({column: ["case_001", "case_002"]})

    assert looks_like_file_reference_column(frame, column)


def test_rendered_kernel_resolves_asset_split_alias_directories(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    build_file_asset_index = namespace["build_file_asset_index"]
    resolve_asset_path = namespace["resolve_asset_path"]

    root = tmp_path / "input"
    train_asset = root / "images" / "training" / "case_train.png"
    eval_asset = root / "images" / "eval" / "case_eval.png"
    public_asset = root / "images" / "public" / "case_public.png"
    train_protein = root / "proteins" / "training" / "shared_case.pdb"
    eval_protein = root / "proteins" / "eval" / "shared_case.pdb"
    train_mask = root / "masks" / "training" / "mask_case.png"
    eval_mask = root / "masks" / "eval" / "mask_case.png"
    train_text = root / "texts" / "training" / "shared_doc.txt"
    eval_text = root / "texts" / "eval" / "shared_doc.txt"
    train_subtitle = root / "subtitles" / "training" / "clip_caption.vtt"
    eval_subtitle = root / "subtitles" / "eval" / "clip_caption.vtt"
    raw_train = root / "raw_bucket" / "training" / "Case_Duplicate.JPG"
    raw_public = root / "raw_bucket" / "public" / "Case_Duplicate.JPG"
    train_asset.parent.mkdir(parents=True)
    eval_asset.parent.mkdir(parents=True)
    public_asset.parent.mkdir(parents=True)
    train_protein.parent.mkdir(parents=True)
    eval_protein.parent.mkdir(parents=True)
    train_mask.parent.mkdir(parents=True)
    eval_mask.parent.mkdir(parents=True)
    train_text.parent.mkdir(parents=True)
    eval_text.parent.mkdir(parents=True)
    train_subtitle.parent.mkdir(parents=True)
    eval_subtitle.parent.mkdir(parents=True)
    raw_train.parent.mkdir(parents=True)
    raw_public.parent.mkdir(parents=True)
    train_asset.write_bytes(b"train")
    eval_asset.write_bytes(b"eval")
    public_asset.write_bytes(b"public")
    train_protein.write_text("ATOM train\n", encoding="utf-8")
    eval_protein.write_text("ATOM eval\n", encoding="utf-8")
    train_mask.write_bytes(b"train-mask")
    eval_mask.write_bytes(b"eval-mask")
    train_text.write_text("train transcript\n", encoding="utf-8")
    eval_text.write_text("eval transcript\n", encoding="utf-8")
    train_subtitle.write_text("WEBVTT\n\ntrain caption\n", encoding="utf-8")
    eval_subtitle.write_text("WEBVTT\n\neval caption\n", encoding="utf-8")
    raw_train.write_bytes(b"raw train")
    raw_public.write_bytes(b"raw public")
    asset_index = build_file_asset_index([root])

    assert resolve_asset_path("case_train", asset_index, split="train") == train_asset
    assert resolve_asset_path("case_eval", asset_index, split="test") == eval_asset
    assert resolve_asset_path("case_public", asset_index, split="test") == public_asset
    assert resolve_asset_path("shared_case", asset_index, split="train") == train_protein
    assert resolve_asset_path("shared_case", asset_index, split="test") == eval_protein
    assert resolve_asset_path("mask_case", asset_index, split="train") == train_mask
    assert resolve_asset_path("mask_case", asset_index, split="test") == eval_mask
    assert resolve_asset_path("shared_doc", asset_index, split="train") == train_text
    assert resolve_asset_path("shared_doc", asset_index, split="test") == eval_text
    assert resolve_asset_path("clip_caption", asset_index, split="train") == train_subtitle
    assert resolve_asset_path("clip_caption", asset_index, split="test") == eval_subtitle
    assert resolve_asset_path("case_duplicate", asset_index, split="train") == raw_train
    assert resolve_asset_path("case_duplicate", asset_index, split="test") == raw_public
    assert resolve_asset_path("public/case_duplicate", asset_index, split="test") == raw_public


def test_rendered_kernel_resolve_label_join_columns_ignores_idless_sample_prediction_column() -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    resolve_label_join_columns = namespace["resolve_label_join_columns"]

    train = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "f1": [10, 20]})
    test = pd.DataFrame({"row_id": [3, 4], "target": [0, 0], "f1": [30, 40]})
    sample = pd.DataFrame({"target": [0, 0]})
    labels = pd.DataFrame({"row_id": [1, 2], "target": [1, 0]})

    assert resolve_label_join_columns(train, test, sample, labels) == ("row_id", "row_id")


def test_rendered_kernel_synthesize_asset_tables_ignores_idless_sample_prediction_column(tmp_path: Path) -> None:
    namespace = {"__name__": "kagglebot_test"}
    exec(kaggle_notebook.render_kernel_main("demo", "cpu"), namespace)
    synthesize_train_test_from_assets = namespace["synthesize_train_test_from_assets"]

    data_dir = tmp_path / "input" / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "img_1.jpg").write_bytes(b"train")
    (data_dir / "img_2.jpg").write_bytes(b"test")
    labels_path = data_dir / "train_labels.csv"
    sample_path = data_dir / "sample_submission.csv"
    pd.DataFrame({"target": ["img_1"], "label": [1]}).to_csv(labels_path, index=False)
    pd.DataFrame({"target": ["img_2"], "score": [0.0]}).to_csv(sample_path, index=False)
    namespace["INPUT_ROOT"] = data_dir
    namespace["SYNTHETIC_TABLE_DIR"] = working_dir / "synthetic_tables"

    assert synthesize_train_test_from_assets([labels_path, sample_path], labels_path, sample_path) is None


@pytest.mark.parametrize(
    "filename",
    ["submission.xlsx", "submission.xlsm", "submission.ods", "submission.orc", "submission.hdf", "submission.hdf5"],
)
def test_render_kernel_main_uses_tabular_format_default_submission_filename(filename: str) -> None:
    script = kaggle_notebook.render_kernel_main(
        "demo",
        "cpu",
        default_submission_filename=filename,
    )

    compile(script, "main.py", "exec")
    assert f'default_name = "{filename}"' in script


def test_render_kernel_main_uses_non_tabular_default_submission_filename() -> None:
    script = kaggle_notebook.render_kernel_main(
        "demo",
        "cpu",
        default_submission_filename="submission.onnx",
    )

    compile(script, "main.py", "exec")
    assert 'default_name = "submission.onnx"' in script


def test_render_kernel_main_uses_explicit_non_submission_default_filename() -> None:
    script = kaggle_notebook.render_kernel_main(
        "demo",
        "cpu",
        default_submission_filename="answers.nii.gz",
    )

    compile(script, "main.py", "exec")
    assert 'default_name = "answers.nii.gz"' in script


def test_render_kernel_main_escapes_default_submission_filename_literal() -> None:
    script = kaggle_notebook.render_kernel_main(
        "demo",
        "cpu",
        default_submission_filename='answers "quoted".nii.gz',
    )

    compile(script, "main.py", "exec")
    assert 'default_name = "answers \\"quoted\\".nii.gz"' in script


@pytest.mark.parametrize(
    "filename",
    [
        "submission.xlsx",
        "submission.xlsm",
        "submission.ods",
        "submission.orc",
        "submission.hdf",
        "submission.hdf5",
        "submission.db",
        "submission.sqlite",
        "submission.sqlite3",
    ],
)
def test_default_submission_filename_from_format_uses_tabular_suffix(tmp_path: Path, filename: str) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        f"## Submission Format\nSubmit a tabular file named `{filename}`.\n",
        encoding="utf-8",
    )

    assert kaggle_notebook._default_submission_filename_from_format(format_path) == filename


@pytest.mark.parametrize(
    ("description", "filename"),
    [
        ("Submit a zstd-compressed NDJSON file with columns row_id,target.", "submission.ndjson.zst"),
        ("Submit a bzip2-compressed HTML file with columns row_id,target.", "submission.html.bz2"),
    ],
)
def test_default_submission_filename_from_format_uses_compressed_tabular_keywords(
    tmp_path: Path,
    description: str,
    filename: str,
) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(f"## Submission Format\n{description}\n", encoding="utf-8")

    assert kaggle_notebook._default_submission_filename_from_format(format_path) == filename


@pytest.mark.parametrize(
    ("template_name", "default_name"),
    [
        ("sample_submission.tsv", "submission.tsv"),
        ("sample-submission.csv.gz", "submission.csv.gz"),
        ("submission_template.jsonl.zst", "submission.jsonl.zst"),
    ],
)
def test_safe_default_submission_filename_normalizes_template_name_suffix(
    template_name: str,
    default_name: str,
) -> None:
    assert kaggle_notebook._safe_default_submission_filename(template_name) == default_name


def test_render_kernel_main_normalizes_template_default_submission_suffix() -> None:
    script = kaggle_notebook.render_kernel_main(
        "demo",
        "cpu",
        default_submission_filename="sample_submission.tsv",
    )

    compile(script, "main.py", "exec")
    assert 'default_name = "submission.tsv"' in script


def test_default_submission_filename_from_format_uses_explicit_non_tabular_filename(tmp_path: Path) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        "## Submission Format\nUpload a single file named `answers.nii.gz`.\n",
        encoding="utf-8",
    )

    assert kaggle_notebook._default_submission_filename_from_format(format_path) == "answers.nii.gz"


def test_default_submission_filename_from_format_uses_explicit_directory_array_filename(tmp_path: Path) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        "## Submission Format\nThe required output is a Zarr store called predictions.zarr.\n",
        encoding="utf-8",
    )

    assert kaggle_notebook._default_submission_filename_from_format(format_path) == "predictions.zarr"


@pytest.mark.parametrize(
    ("description", "filename"),
    [
        ("Participants must upload an EPUB document for scoring.", "submission.epub"),
        ("Participants must upload a zstd-compressed LaTeX file for scoring.", "submission.tex.zst"),
        ("Participants must upload COCO annotations for scoring.", "submission.json"),
        ("Participants must upload a VCF variant file for scoring.", "submission.vcf"),
        ("Participants must upload a cryo-EM MRC volume for scoring.", "submission.mrc"),
        ("Participants must upload an HGT elevation raster for scoring.", "submission.hgt"),
        ("Participants must upload a BrainVision header file for scoring.", "submission.vhdr"),
        ("Participants must upload JSON-LD linked data for scoring.", "submission.jsonld"),
        ("Participants must upload a SQLite database for scoring.", "submission.sqlite"),
        ("Participants must upload a SQLite3 database for scoring.", "submission.sqlite3"),
        ("Participants must upload a TensorRT engine for scoring.", "submission.engine"),
        ("Participants must upload an RKNN edge model for scoring.", "submission.rknn"),
        ("Participants must upload a safetensors model file for scoring.", "submission.safetensors"),
    ],
)
def test_default_submission_filename_from_format_uses_non_tabular_prose_suffix(
    tmp_path: Path,
    description: str,
    filename: str,
) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(f"## Submission Format\n{description}\n", encoding="utf-8")

    assert kaggle_notebook._default_submission_filename_from_format(format_path) == filename


def test_render_kernel_main_writes_h5_as_hdf() -> None:
    script = kaggle_notebook.render_kernel_main("demo", "cpu")

    compile(script, "main.py", "exec")
    assert "TABULAR_PARQUET_SUFFIXES" in script
    assert '".parq"' in script
    assert '".pq"' in script
    assert '".ftr"' in script
    assert '".tab"' in script
    assert '".psv"' in script
    assert '".csv.zip"' in script
    assert "def select_zip_tabular_member" in script
    assert "TABULAR_HDF_SUFFIXES" in script
    assert '".h5"' in script
    assert '".hdf"' in script
    assert '".hdf5"' in script
    assert "def read_native_hdf_table(path: Path)" in script
    assert "TABULAR_ANNDATA_SUFFIXES" in script
    assert '".h5ad"' in script
    assert "def read_h5ad_tabular_frame(path: Path)" in script
    assert "TABULAR_LOOM_SUFFIXES" in script
    assert '".loom"' in script
    assert "def read_loom_tabular_frame(path: Path)" in script
    assert "TABULAR_GEOPACKAGE_SUFFIXES" in script
    assert '".gpkg"' in script
    assert "def read_geopackage_tabular_frame(path: Path" in script
    assert "TABULAR_SHAPEFILE_SUFFIXES" in script
    assert '".shp"' in script
    assert "def read_shapefile_tabular_frame(path: Path" in script
    assert "TABULAR_KML_SUFFIXES" in script
    assert '".kmz"' in script
    assert "def read_kml_tabular_frame(path: Path" in script


def test_render_kernel_main_reads_legacy_scientific_table_inputs() -> None:
    script = kaggle_notebook.render_kernel_main("demo", "cpu")

    compile(script, "main.py", "exec")
    assert "TABULAR_SAS_SUFFIXES" in script
    assert '".sas7bdat"' in script
    assert '".xpt"' in script
    assert '".xport"' in script
    assert "pd.read_sas(path, format=sas_format_for_suffix(suffix))" in script
    assert "def sas_format_for_suffix(suffix: str) -> str | None:" in script
    assert "TABULAR_SPSS_SUFFIXES" in script
    assert '".sav"' in script
    assert '".zsav"' in script
    assert "pd.read_spss(path)" in script
    assert "TABULAR_MATLAB_SUFFIXES" in script
    assert '".mat"' in script
    assert "def read_mat_tabular_frame(path: Path)" in script
    assert "from scipy.io import loadmat" in script
    assert "MATLAB file does not contain table-like arrays" in script
    assert "TABULAR_NETCDF_SUFFIXES" in script
    assert '".nc"' in script
    assert '".netcdf"' in script
    assert '".cdf"' in script
    assert '".nc4"' in script
    assert "def read_netcdf_tabular_frame(path: Path)" in script
    assert "from scipy.io import netcdf_file" in script
    assert "TABULAR_FITS_SUFFIXES" in script
    assert '".fits"' in script
    assert '".fit"' in script
    assert '".fts"' in script
    assert '".fits.gz"' in script
    assert "def read_fits_tabular_frame(path: Path)" in script
    assert "from astropy.io import fits" in script
    assert "TABULAR_NUMPY_SUFFIXES" in script
    assert '".npy"' in script
    assert '".npz"' in script
    assert "def read_numpy_tabular_frame(path: Path)" in script
    assert "def numpy_column_names_for_path(path: Path | None, width: int)" in script
    assert "TABULAR_ARFF_SUFFIXES" in script
    assert '".arff"' in script
    assert '".arff.gz"' in script
    assert "def read_arff_tabular_frame(path: Path)" in script
    assert "from scipy.io import arff" in script
    assert "arff.loadarff(handle)" in script
    assert "TABULAR_HTML_SUFFIX_PREFIXES" in script
    assert '".html"' in script
    assert '".htm"' in script
    assert "def read_html_tabular_frame(path: Path)" in script
    assert "pd.read_html" in script
    assert "table.shape[1] > 0" in script
    assert "not table.empty and table.shape[1] > 0" not in script
    assert "frame.to_html(index=False)" in script
    assert '".avro"' in script
    assert "def read_avro_table(path: Path)" in script
    assert "def write_avro_table(frame: pd.DataFrame, path: Path)" in script
    assert "TABULAR_EXCEL_SUFFIXES" in script
    assert '".xls"' in script
    assert '".xlsm"' in script
    assert '".xlsx"' in script
    assert '".ods"' in script
    assert "TABULAR_EXCEL_INPUT_ONLY_SUFFIXES" in script
    assert '".xlsb"' in script
    assert 'engine="pyxlsb"' in script
    assert "TABULAR_SVMLIGHT_SUFFIX_PREFIXES" in script
    assert '".svm"' in script
    assert '".svmlight"' in script
    assert '".libsvm"' in script
    assert "def read_svmlight_tabular_frame(path: Path)" in script
    assert "load_svmlight_file" in script
    assert "TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES" in script
    assert '".fwf"' in script
    assert '".fixed"' in script
    assert '".fixedwidth"' in script
    assert "def read_fixed_width_tabular_frame(path: Path)" in script
    assert "pd.read_fwf(StringIO(handle.read()))" in script
    assert 'return r"\\s+"' in script


def test_default_submission_filename_from_format_uses_archive_suffix(tmp_path: Path) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        "## Submission Format\nSubmit a submission.tar.xz archive containing model weights.\n",
        encoding="utf-8",
    )

    assert kaggle_notebook._default_submission_filename_from_format(format_path) == "submission.tar.xz"


def test_render_kernel_main_recognizes_inference_role_aliases() -> None:
    script = kaggle_notebook.render_kernel_main("demo", "cpu")
    namespace = {"__name__": "kagglebot_test"}

    compile(script, "main.py", "exec")
    for token in ('"validation"', '"holdout"', '"unlabeled"', '"predict"', '"score"'):
        assert token in script
    exec(script, namespace)
    assert namespace["role_aliases"]("test") == set(ROLE_ALIASES["test"])
    assert namespace["role_aliases"]("train") == set(ROLE_ALIASES["train"])
    assert namespace["path_mentions_role"](Path("testset.csv"), "test")
    assert namespace["path_mentions_role"](Path("trainset.csv"), "train")
    assert not namespace["path_mentions_role"](Path("trainset.csv"), "test")


def test_wait_for_kernel_pushes_cpu_stop_marker_after_failed_gpu_run(monkeypatch, tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    logs_dir = tmp_path / "logs"
    kernel_dir.mkdir()
    logs_dir.mkdir()
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/kernel",
                "title": "kernel",
                "code_file": "main.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "competition_sources": ["demo"],
                "dataset_sources": [],
                "kernel_sources": [],
                "model_sources": [],
                "keywords": [],
            }
        ),
        encoding="utf-8",
    )
    pushed: list[Path] = []
    monkeypatch.setattr(
        kaggle_notebook.kaggle_cli,
        "kernels_status",
        lambda *_args,
        **_kwargs: 'owner/kernel has status "KernelWorkerStatus.ERROR"\nFailure message: "Your notebook failed"',
    )
    monkeypatch.setattr(
        kaggle_notebook.kaggle_cli,
        "kernels_push",
        lambda kernel_path, **_kwargs: pushed.append(Path(kernel_path)) or "pushed",
    )

    with pytest.raises(RuntimeError, match="Kernel run failed"):
        _wait_for_kernel("owner/kernel", logs_dir=logs_dir, slug="demo", kernel_dir=kernel_dir)

    assert pushed == [tmp_path / "kernel-stop"]
    metadata = json.loads((tmp_path / "kernel-stop" / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert (logs_dir / "kernel_stop.log").exists()


def test_generated_kernel_expands_tiny_public_sample_to_test_ids(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102, 103, 104], "feature": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101, 102, 103, 104]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_extracts_nested_input_archives(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    train_zip = io.BytesIO()
    with zipfile.ZipFile(train_zip, "w") as archive:
        archive.writestr(
            "train.csv",
            "id,feature,target\n1,0.0,0\n2,1.0,1\n3,2.0,0\n4,3.0,1\n5,4.0,0\n6,5.0,1\n",
        )
    test_zip = io.BytesIO()
    with zipfile.ZipFile(test_zip, "w") as archive:
        archive.writestr("test.csv", "id,feature\n100,0.5\n101,4.5\n")
    with zipfile.ZipFile(data_dir / "competition.zip", "w") as archive:
        archive.writestr("train.zip", train_zip.getvalue())
        archive.writestr("test.zip", test_zip.getvalue())
        archive.writestr("sample_submission.csv", "id,target\n100,0\n101,0\n")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert (working_dir / "extracted_input" / "train.csv").exists()
    assert (working_dir / "extracted_input" / "test.csv").exists()
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_extracts_tar_xz_input_archive(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with tarfile.open(data_dir / "competition.tar.xz", "w:xz") as archive:
        for name, payload in {
            "train.csv": b"id,feature,target\n1,0.0,0\n2,1.0,1\n3,2.0,0\n4,3.0,1\n5,4.0,0\n6,5.0,1\n",
            "test.csv": b"id,feature\n100,0.5\n101,4.5\n",
            "sample_submission.csv": b"id,target\n100,0\n101,0\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert (working_dir / "extracted_input" / "train.csv").exists()
    assert (working_dir / "extracted_input" / "test.csv").exists()


def test_generated_kernel_extracts_tar_zst_input_archive(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        for name, payload in {
            "train.csv": b"id,feature,target\n1,0.0,0\n2,1.0,1\n3,2.0,0\n4,3.0,1\n5,4.0,0\n6,5.0,1\n",
            "test.csv": b"id,feature\n100,0.5\n101,4.5\n",
            "sample_submission.csv": b"id,target\n100,0\n101,0\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    (data_dir / "competition.tar.zst").write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert (working_dir / "extracted_input" / "train.csv").exists()
    assert (working_dir / "extracted_input" / "test.csv").exists()


def test_generated_kernel_extracts_7z_input_archive(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    source_dir = tmp_path / "source"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    source_dir.mkdir()
    (source_dir / "train.csv").write_text(
        "id,feature,target\n1,0.0,0\n2,1.0,1\n3,2.0,0\n4,3.0,1\n5,4.0,0\n6,5.0,1\n",
        encoding="utf-8",
    )
    (source_dir / "test.csv").write_text("id,feature\n100,0.5\n101,4.5\n", encoding="utf-8")
    (source_dir / "sample_submission.csv").write_text("id,target\n100,0\n101,0\n", encoding="utf-8")
    with py7zr.SevenZipFile(data_dir / "competition.7z", "w") as archive:
        archive.write(source_dir / "train.csv", "train.csv")
        archive.write(source_dir / "test.csv", "test.csv")
        archive.write(source_dir / "sample_submission.csv", "sample_submission.csv")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert (working_dir / "extracted_input" / "train.csv").exists()
    assert (working_dir / "extracted_input" / "test.csv").exists()


def test_generated_kernel_rejects_duplicate_archive_targets(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with zipfile.ZipFile(data_dir / "competition.zip", "w") as archive:
        archive.writestr("train.csv", "id,feature,target\n1,0,0\n")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("train.csv", "id,feature,target\n2,1,1\n")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate archive member target"):
        runpy.run_path(str(kernel_path), run_name="__main__")


def test_generated_kernel_template_supports_rar_input_archives() -> None:
    assert "def safe_extract_rar(" in KERNEL_TEMPLATE
    assert '".rar"' in KERNEL_TEMPLATE
    assert "safe_extract_rar(archive_path, EXTRACTED_INPUT_ROOT" in KERNEL_TEMPLATE


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm"])
def test_generated_kernel_reads_excel_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_excel(data_dir / f"train{suffix}", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_excel(
        data_dir / f"test{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_excel(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_excel(working_dir / f"submission{suffix}")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_reads_ods_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_excel(data_dir / "train.ods", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_excel(
        data_dir / "test.ods",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_excel(
        data_dir / "sample_submission.ods",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_excel(working_dir / "submission.ods")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


@pytest.mark.parametrize("suffix", [".html", ".htm"])
def test_generated_kernel_reads_html_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_html(data_dir / f"train{suffix}", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_html(
        data_dir / f"test{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_html(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_html(working_dir / f"submission{suffix}")[0]
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


@pytest.mark.parametrize("suffix", [".parquet", ".parq", ".pq"])
def test_generated_kernel_reads_parquet_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_parquet(data_dir / f"train{suffix}", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_parquet(
        data_dir / f"test{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_parquet(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_parquet(working_dir / f"submission{suffix}")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_reads_zip_wrapped_csv_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with zipfile.ZipFile(data_dir / "train.csv.zip", "w") as archive:
        rows = "\n".join(f"{idx},{float(idx)},{idx % 2}" for idx in range(12))
        archive.writestr("train.csv", f"id,feature,target\n{rows}\n")
    with zipfile.ZipFile(data_dir / "test.csv.zip", "w") as archive:
        archive.writestr("test.csv", "id,feature\n100,1.0\n101,2.0\n")
    with zipfile.ZipFile(data_dir / "sample_submission.csv.zip", "w") as archive:
        archive.writestr("sample_submission.csv", "id,target\n100,0\n101,0\n")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_reads_zip_wrapped_parquet_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_zip_parquet(
        data_dir / "train.parquet.zip",
        "nested/train.parquet",
        pd.DataFrame(
            {
                "id": list(range(12)),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            }
        ),
    )
    _write_zip_parquet(
        data_dir / "test.parquet.zip",
        "nested/test.parquet",
        pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}),
    )
    _write_zip_parquet(
        data_dir / "sample_submission.parquet.zip",
        "nested/sample_submission.parquet",
        pd.DataFrame({"id": [100, 101], "target": [0, 0]}),
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_reads_zip_wrapped_excel_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_zip_excel(
        data_dir / "train.xlsx.zip",
        "nested/train.xlsx",
        pd.DataFrame(
            {
                "id": list(range(12)),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            }
        ),
    )
    _write_zip_excel(
        data_dir / "test.xlsx.zip",
        "nested/test.xlsx",
        pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}),
    )
    _write_zip_excel(
        data_dir / "sample_submission.xlsx.zip",
        "nested/sample_submission.xlsx",
        pd.DataFrame({"id": [100, 101], "target": [0, 0]}),
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_reads_zip_wrapped_stata_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_zip_stata(
        data_dir / "train.dta.zip",
        "nested/train.dta",
        pd.DataFrame(
            {
                "id": list(range(12)),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            }
        ),
    )
    _write_zip_stata(
        data_dir / "test.dta.zip",
        "nested/test.dta",
        pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}),
    )
    _write_zip_stata(
        data_dir / "sample_submission.dta.zip",
        "nested/sample_submission.dta",
        pd.DataFrame({"id": [100, 101], "target": [0, 0]}),
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_reads_csv_gz_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with gzip.open(data_dir / "train.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n")
        for idx in range(12):
            handle.write(f"{idx},{float(idx)},{idx % 2}\n")
    with gzip.open(data_dir / "test.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n100,1.0\n101,2.0\n")
    with gzip.open(data_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n100,0\n101,0\n")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv.gz")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


@pytest.mark.parametrize("suffix", [".fwf", ".fixed", ".fixedwidth"])
def test_generated_kernel_reads_fixed_width_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / f"train{suffix}").write_text(
        "id   feature   target\n" + "\n".join(f"{idx:<4}{float(idx):<8.1f}{idx % 2}" for idx in range(12)) + "\n",
        encoding="utf-8",
    )
    (data_dir / f"test{suffix}").write_text("id   feature\n100  1.0\n101  2.0\n", encoding="utf-8")
    (data_dir / f"sample_submission{suffix}").write_text("id   target\n100  0\n101  0\n", encoding="utf-8")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


@pytest.mark.parametrize("suffix", [".svm", ".svmlight", ".libsvm"])
def test_generated_kernel_reads_svmlight_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / f"train{suffix}").write_text(
        "\n".join(
            [
                "0 1:0.0 2:1.0",
                "1 1:1.0 2:0.0",
                "0 1:0.1 2:1.1",
                "1 1:1.1 2:0.1",
                "0 1:0.2 2:1.2",
                "1 1:1.2 2:0.2",
                "0 1:0.3 2:1.3",
                "1 1:1.3 2:0.3",
                "0 1:0.4 2:1.4",
                "1 1:1.4 2:0.4",
                "0 1:0.5 2:1.5",
                "1 1:1.5 2:0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / f"test{suffix}").write_text(
        "0 1:0.05 2:1.05\n0 1:1.45 2:0.45\n0 1:0.75 2:0.80\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("target\n0\n0\n", encoding="utf-8")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(submission) == 3
    assert list(submission.columns) == ["target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


def test_generated_kernel_reads_compressed_arff_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    train_rows = "\n".join(f"{idx},{float(idx)},{'yes' if idx % 2 else 'no'}" for idx in range(12))
    payloads = {
        "train.arff.gz": f"""
@RELATION train
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@ATTRIBUTE target {{no,yes}}
@DATA
{train_rows}
""",
        "test.arff.gz": """
@RELATION test
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@DATA
100,1.0
101,2.0
""",
        "sample_submission.arff.gz": """
@RELATION sample_submission
@ATTRIBUTE id NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
100,no
101,no
""",
    }
    for name, payload in payloads.items():
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            handle.write(payload.strip())

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100.0, 101.0]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] in {"accuracy", "leave_one_out_exact_match"}


def test_generated_kernel_reads_matlab_column_variable_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    savemat(
        data_dir / "train.mat",
        {
            "id": list(range(12)),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        },
    )
    savemat(data_dir / "test.mat", {"id": [100, 101], "feature": [1.0, 2.0]})
    savemat(data_dir / "sample_submission.mat", {"id": [100, 101], "target": [0, 0]})

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


@pytest.mark.parametrize("suffix", [".sav", ".zsav"])
def test_generated_kernel_reads_spss_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pyreadstat.write_sav(
        pd.DataFrame(
            {
                "id": range(12),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            }
        ),
        data_dir / f"train{suffix}",
    )
    pyreadstat.write_sav(pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}), data_dir / f"test{suffix}")
    pyreadstat.write_sav(
        pd.DataFrame({"id": [100, 101], "target": [0.0, 0.0]}),
        data_dir / f"sample_submission{suffix}",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100.0, 101.0]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] in {"accuracy", "leave_one_out_exact_match"}


@pytest.mark.parametrize("suffix", [".xpt", ".xport"])
def test_generated_kernel_reads_sas_xport_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pyreadstat.write_xport(
        pd.DataFrame(
            {
                "id": range(1, 13),
                "feature": [float(idx) for idx in range(1, 13)],
                "target": [idx % 2 for idx in range(1, 13)],
            }
        ),
        data_dir / f"train{suffix}",
        file_format_version=5,
    )
    pyreadstat.write_xport(
        pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}),
        data_dir / f"test{suffix}",
        file_format_version=5,
    )
    pyreadstat.write_xport(
        pd.DataFrame({"id": [100, 101], "target": [0.0, 0.0]}),
        data_dir / f"sample_submission{suffix}",
        file_format_version=5,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100.0, 101.0]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] in {"accuracy", "leave_one_out_exact_match"}


def test_generated_kernel_reads_stata_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_stata(data_dir / "train.dta", write_index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_stata(
        data_dir / "test.dta",
        write_index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0.0, 0.0]}).to_stata(
        data_dir / "sample_submission.dta",
        write_index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_stata(working_dir / "submission.dta")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["target_metrics"]["target"]["metric"] == "accuracy"


@pytest.mark.parametrize("sample_name", ["sample_predictions.csv", "prediction_template.csv", "AnswerTemplate.csv"])
def test_generated_kernel_accepts_sample_submission_aliases(tmp_path: Path, sample_name: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / sample_name,
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    namespace = runpy.run_path(str(kernel_path), run_name="__main__")

    _, _, sample_path = namespace["pick_files"](namespace["find_tabular_files"](data_dir))
    assert sample_path.name == sample_name
    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_accepts_eval_features_as_test_table(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(
        data_dir / "eval_features.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    namespace = runpy.run_path(str(kernel_path), run_name="__main__")

    _, test_path, _ = namespace["pick_files"](namespace["find_tabular_files"](data_dir))
    assert test_path.name == "eval_features.csv"
    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_does_not_treat_public_train_as_test(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    public_dir = data_dir / "public"
    working_dir = tmp_path / "working"
    public_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(12),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_csv(public_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(
        public_dir / "features.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    namespace = runpy.run_path(str(kernel_path), run_name="__main__")

    train_path, test_path, _ = namespace["pick_files"](namespace["find_tabular_files"](data_dir))
    assert train_path.name == "train.csv"
    assert test_path.name == "features.csv"
    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]


def test_generated_kernel_prefers_compressed_canonical_train_test_pair(tmp_path: Path) -> None:
    pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    for name, target_offset in (("train.csv.gz", 0), ("train_features.csv.gz", 1)):
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            handle.write("id,feature,target\n")
            for idx in range(12):
                handle.write(f"{idx},{float(idx)},{(idx + target_offset) % 2}\n")
    for name in ("test.csv.gz", "test_features.csv.gz"):
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            handle.write("id,feature\n100,1.0\n101,2.0\n")
    with gzip.open(data_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n100,0\n101,0\n")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    namespace = runpy.run_path(str(kernel_path), run_name="__main__")

    train_path, test_path, _ = namespace["pick_files"](namespace["find_tabular_files"](data_dir))
    assert train_path.name == "train.csv.gz"
    assert test_path.name == "test.csv.gz"
    assert (working_dir / "submission.csv.gz").exists()


def test_generated_kernel_avoids_test_substring_false_positive(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "TrainingSet.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(
        data_dir / "PublicTest.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": [200, 201],
            "feature": [9.0, 10.0],
            "target": [1, 0],
        }
    ).to_csv(data_dir / "contest.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "SampleSubmission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    namespace = runpy.run_path(str(kernel_path), run_name="__main__")

    train_path, test_path, sample_path = namespace["pick_files"](namespace["find_tabular_files"](data_dir))
    assert train_path.name == "TrainingSet.csv"
    assert test_path.name == "PublicTest.csv"
    assert sample_path.name == "SampleSubmission.csv"
    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]


def test_generated_kernel_accepts_validation_features_as_test(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "training_data.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(
        data_dir / "validation_features.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    namespace = runpy.run_path(str(kernel_path), run_name="__main__")

    train_path, test_path, sample_path = namespace["pick_files"](namespace["find_tabular_files"](data_dir))
    assert train_path.name == "training_data.csv"
    assert test_path.name == "validation_features.csv"
    assert sample_path.name == "sample_submission.csv"
    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]


def test_generated_kernel_preserves_leading_zero_ids(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "train.csv").write_text(
        "id,feature,target\n001,1.0,0\n002,2.0,1\n003,3.0,0\n004,4.0,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,feature\n006,6.0\n005,5.0\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n005,0\n006,0\n", encoding="utf-8")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv", dtype={"id": str})
    assert submission["id"].tolist() == ["005", "006"]


def test_generated_kernel_reads_csv_xz_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with lzma.open(data_dir / "train.csv.xz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n")
        for idx in range(12):
            handle.write(f"{idx},{float(idx)},{idx % 2}\n")
    with lzma.open(data_dir / "test.csv.xz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n100,1.0\n101,2.0\n")
    with lzma.open(data_dir / "sample_submission.csv.xz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n100,0\n101,0\n")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv.xz")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_feather_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": list(range(12)),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_feather(data_dir / "train.feather")
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_feather(data_dir / "test.feather")
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_feather(data_dir / "sample_submission.feather")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_feather(working_dir / "submission.feather")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("suffix", [".ftr", ".arrow", ".ipc"])
def test_generated_kernel_reads_arrow_ipc_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": list(range(12)),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_feather(data_dir / f"train{suffix}")
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_feather(data_dir / f"test{suffix}")
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_feather(data_dir / f"sample_submission{suffix}")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_feather(working_dir / f"submission{suffix}")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_avro_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    write_table(
        pd.DataFrame(
            {
                "id": list(range(12)),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            },
        ),
        data_dir / "train.avro",
    )
    write_table(pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}), data_dir / "test.avro")
    write_table(pd.DataFrame({"id": [100, 101], "target": [0, 0]}), data_dir / "sample_submission.avro")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = read_table(working_dir / "submission.avro")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".pkl.gz", ".pkl.zst"])
def test_generated_kernel_reads_pickle_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": list(range(12)),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_pickle(data_dir / f"train{suffix}")
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_pickle(data_dir / f"test{suffix}")
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_pickle(data_dir / f"sample_submission{suffix}")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_pickle(working_dir / f"submission{suffix}")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_orc_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": list(range(12)),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_orc(data_dir / "train.orc", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_orc(data_dir / "test.orc", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_orc(
        data_dir / "sample_submission.orc",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_orc(working_dir / "submission.orc")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_hdf_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": list(range(12)),
            "feature": [float(idx) for idx in range(12)],
            "target": [idx % 2 for idx in range(12)],
        }
    ).to_hdf(data_dir / "train.h5", key="train", mode="w", format="table", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_hdf(
        data_dir / "test.h5",
        key="test",
        mode="w",
        format="table",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_hdf(
        data_dir / "sample_submission.hdf5",
        key="sample_submission",
        mode="w",
        format="table",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_hdf(working_dir / "submission.hdf5")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_native_hdf5_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    h5py = pytest.importorskip("h5py")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with h5py.File(data_dir / "train.h5", "w") as handle:
        group = handle.create_group("train")
        group.create_dataset("id", data=list(range(12)))
        group.create_dataset("feature", data=[float(idx) for idx in range(12)])
        group.create_dataset("target", data=[idx % 2 for idx in range(12)])
    with h5py.File(data_dir / "test.h5", "w") as handle:
        group = handle.create_group("test")
        group.create_dataset("id", data=[100, 101])
        group.create_dataset("feature", data=[1.0, 2.0])
    with h5py.File(data_dir / "sample_submission.hdf5", "w") as handle:
        group = handle.create_group("submission")
        group.create_dataset("id", data=[100, 101])
        group.create_dataset("target", data=[0, 0])

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_hdf(working_dir / "submission.hdf5")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_h5ad_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_h5ad_table(
        data_dir / "train.h5ad",
        ids=np.arange(12),
        features=np.array([[float(idx), float(idx) / 10.0] for idx in range(12)]),
        target=np.array([idx % 2 for idx in range(12)]),
    )
    _write_h5ad_table(
        data_dir / "test.h5ad",
        ids=np.array([100, 101]),
        features=np.array([[1.0, 0.1], [2.0, 0.2]]),
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_geopackage_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_geopackage_table(
        data_dir / "train.gpkg",
        [
            (1, 10.0, 0, b"\x47\x50\x00\x01"),
            (2, 20.0, 1, b"\x47\x50\x00\x02"),
            (3, 30.0, 0, b"\x47\x50\x00\x03"),
            (4, 40.0, 1, b"\x47\x50\x00\x04"),
        ],
    )
    _write_geopackage_table(
        data_dir / "test.gpkg",
        [
            (100, 1.0, None, b"\x47\x50\x00\x64"),
            (101, 2.0, None, b"\x47\x50\x00\x65"),
        ],
        table="test",
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_shapefile_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "train.shp").write_bytes(b"")
    (data_dir / "test.shp").write_bytes(b"")
    _write_dbf_table(
        data_dir / "train.dbf",
        [
            (1, 10.0, 0, "north"),
            (2, 20.0, 1, "south"),
            (3, 30.0, 0, "east"),
            (4, 40.0, 1, "west"),
        ],
    )
    _write_dbf_table(
        data_dir / "test.dbf",
        [
            (100, 1.0, None, "north"),
            (101, 2.0, None, "south"),
        ],
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_kmz_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_kmz(data_dir / "train.kmz", _kml_payload([(1, 10.0, 0), (2, 20.0, 1), (3, 30.0, 0), (4, 40.0, 1)]))
    _write_kmz(data_dir / "test.kmz", _kml_payload([(100, 1.0, None), (101, 2.0, None)]))
    pd.DataFrame({"id": ["100", "101"], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_compressed_kml_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with gzip.open(data_dir / "train.kml.gz", "wt", encoding="utf-8") as handle:
        handle.write(_kml_payload([(1, 10.0, 0), (2, 20.0, 1), (3, 30.0, 0), (4, 40.0, 1)]))
    with gzip.open(data_dir / "test.kml.gz", "wt", encoding="utf-8") as handle:
        handle.write(_kml_payload([(100, 1.0, None), (101, 2.0, None)]))
    pd.DataFrame({"id": ["100", "101"], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_numpy_archive_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    np.savez(
        data_dir / "train.npz",
        id=np.arange(12),
        feature=np.array([float(idx) for idx in range(12)]),
        target=np.array([idx % 2 for idx in range(12)]),
    )
    np.savez(data_dir / "test.npz", id=np.array([100, 101]), feature=np.array([1.0, 2.0]))
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("suffix", [".nc", ".netcdf", ".cdf"])
def test_generated_kernel_reads_netcdf_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_netcdf_table(
        data_dir / f"train{suffix}",
        {
            "id": np.arange(12, dtype=np.int32),
            "feature": np.array([float(idx) for idx in range(12)], dtype=np.float64),
            "target": np.array([idx % 2 for idx in range(12)], dtype=np.int32),
        },
    )
    _write_netcdf_table(
        data_dir / f"test{suffix}",
        {
            "id": np.array([100, 101], dtype=np.int32),
            "feature": np.array([1.0, 2.0], dtype=np.float64),
        },
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_hdf5_backed_nc4_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    h5py = pytest.importorskip("h5py")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with h5py.File(data_dir / "train.nc4", "w") as handle:
        handle.create_dataset("id", data=np.arange(12, dtype=np.int32))
        handle.create_dataset("feature", data=np.array([float(idx) for idx in range(12)], dtype=np.float64))
        handle.create_dataset("target", data=np.array([idx % 2 for idx in range(12)], dtype=np.int32))
    with h5py.File(data_dir / "test.nc4", "w") as handle:
        handle.create_dataset("id", data=np.array([100, 101], dtype=np.int32))
        handle.create_dataset("feature", data=np.array([1.0, 2.0], dtype=np.float64))
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("suffix", [".fits", ".fit", ".fts", ".fits.gz"])
def test_generated_kernel_reads_fits_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_fits_table(
        data_dir / f"train{suffix}",
        {
            "id": np.arange(12, dtype=np.int64),
            "feature": np.array([float(idx) for idx in range(12)], dtype=np.float64),
            "target": np.array([idx % 2 for idx in range(12)], dtype=np.int64),
        },
    )
    _write_fits_table(
        data_dir / f"test{suffix}",
        {
            "id": np.array([100, 101], dtype=np.int64),
            "feature": np.array([1.0, 2.0], dtype=np.float64),
        },
    )
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_numpy_matrix_column_sidecars(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    np.save(
        data_dir / "train.npy",
        np.array([[float(idx), float(idx * 10), float(idx % 2)] for idx in range(12)]),
    )
    np.save(data_dir / "test.npy", np.array([[100.0, 1.0], [101.0, 2.0]]))
    (data_dir / "train_columns.txt").write_text("id\nfeature\ntarget\n", encoding="utf-8")
    (data_dir / "test_columns.txt").write_text("id\nfeature\n", encoding="utf-8")
    pd.DataFrame({"id": [100.0, 101.0], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100.0, 101.0]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_csv_zst_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    compressor = zstd.ZstdCompressor()
    train_rows = ["id,feature,target\n", *[f"{idx},{float(idx)},{idx % 2}\n" for idx in range(12)]]
    (data_dir / "train.csv.zst").write_bytes(compressor.compress("".join(train_rows).encode("utf-8")))
    (data_dir / "test.csv.zst").write_bytes(compressor.compress(b"id,feature\n100,1.0\n101,2.0\n"))
    (data_dir / "sample_submission.csv.zst").write_bytes(compressor.compress(b"id,target\n100,0\n101,0\n"))

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv.zst")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize(
    "suffix",
    [".jsonl.bz2", ".jsonl.xz", ".jsonl.zst", ".jsonlines.bz2", ".jsonlines.zst", ".ndjson.xz", ".ndjson.zst"],
)
def test_generated_kernel_reads_compressed_json_lines_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    train_rows = [json.dumps({"id": idx, "feature": float(idx), "target": idx % 2}) + "\n" for idx in range(12)]
    test_rows = [
        json.dumps({"id": 100, "feature": 1.0}) + "\n",
        json.dumps({"id": 101, "feature": 2.0}) + "\n",
    ]
    sample_rows = [
        json.dumps({"id": 100, "target": 0}) + "\n",
        json.dumps({"id": 101, "target": 0}) + "\n",
    ]

    def write_json_lines(path: Path, rows: list[str]) -> None:
        payload = "".join(rows).encode("utf-8")
        if path.name.endswith(".bz2"):
            with bz2.open(path, "wb") as handle:
                handle.write(payload)
        elif path.name.endswith(".xz"):
            with lzma.open(path, "wb") as handle:
                handle.write(payload)
        else:
            path.write_bytes(zstd.ZstdCompressor().compress(payload))

    write_json_lines(data_dir / f"train{suffix}", train_rows)
    write_json_lines(data_dir / f"test{suffix}", test_rows)
    write_json_lines(data_dir / f"sample_submission{suffix}", sample_rows)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_json(working_dir / f"submission{suffix}", lines=True)
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("suffix", [".yaml", ".yml.zst"])
def test_generated_kernel_reads_yaml_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")
    yaml = pytest.importorskip("yaml")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()

    def write_yaml(path: Path, frame) -> None:
        payload = yaml.safe_dump(frame.to_dict(orient="records"), sort_keys=False).encode("utf-8")
        if path.name.endswith(".zst"):
            path.write_bytes(zstd.ZstdCompressor().compress(payload))
        else:
            path.write_bytes(payload)

    write_yaml(
        data_dir / f"train{suffix}",
        pd.DataFrame(
            {
                "id": list(range(12)),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            }
        ),
    )
    write_yaml(data_dir / f"test{suffix}", pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}))
    write_yaml(data_dir / f"sample_submission{suffix}", pd.DataFrame({"id": [100, 101], "target": [0, 0]}))

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    output_path = working_dir / f"submission{suffix}"
    payload = output_path.read_bytes()
    if suffix.endswith(".zst"):
        payload = zstd.ZstdDecompressor().decompress(payload)
    submission = pd.DataFrame(yaml.safe_load(payload.decode("utf-8")))
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("suffix", [".yaml.xz", ".xml.bz2", ".html.bz2", ".psv.xz", ".tab.zst"])
def test_generated_kernel_reads_compressed_structured_tabular_inputs(tmp_path: Path, suffix: str) -> None:
    pd = pytest.importorskip("pandas")
    if suffix.startswith(".yaml"):
        pytest.importorskip("yaml")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    write_table(
        pd.DataFrame(
            {
                "id": list(range(12)),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            }
        ),
        data_dir / f"train{suffix}",
    )
    write_table(pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}), data_dir / f"test{suffix}")
    write_table(pd.DataFrame({"id": [100, 101], "target": [0, 0]}), data_dir / f"sample_submission{suffix}")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = read_table(working_dir / f"submission{suffix}")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_sqlite_table_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with sqlite3.connect(data_dir / "competition.sqlite") as conn:
        conn.execute("CREATE TABLE train (id INTEGER, feature REAL, target INTEGER)")
        conn.executemany(
            "INSERT INTO train VALUES (?, ?, ?)",
            [(idx, float(idx), idx % 2) for idx in range(12)],
        )
        conn.execute("CREATE TABLE test (id INTEGER, feature REAL)")
        conn.executemany("INSERT INTO test VALUES (?, ?)", [(100, 1.0), (101, 2.0)])
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target INTEGER)")
        conn.executemany("INSERT INTO sample_submission VALUES (?, ?)", [(100, 0), (101, 0)])

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


def test_rendered_kernel_reads_duckdb_table_inputs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    conn = duckdb.connect(str(data_dir / "competition.duckdb"))
    try:
        conn.execute("CREATE TABLE train (id INTEGER, feature DOUBLE, target INTEGER)")
        conn.execute(
            "INSERT INTO train SELECT i, CAST(i AS DOUBLE), i % 2 FROM range(12) tbl(i)",
        )
        conn.execute("CREATE TABLE test (id INTEGER, feature DOUBLE)")
        conn.execute("INSERT INTO test VALUES (100, 1.0), (101, 2.0)")
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target INTEGER)")
        conn.execute("INSERT INTO sample_submission VALUES (100, 0), (101, 0)")
    finally:
        conn.close()

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("suffix", [".rds", ".rda", ".rdata"])
def test_rendered_kernel_reads_rdata_table_inputs(tmp_path: Path, suffix: str) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_rdata_table(
        data_dir / f"train{suffix}",
        pd.DataFrame(
            {
                "id": list(range(12)),
                "feature": [float(idx) for idx in range(12)],
                "target": [idx % 2 for idx in range(12)],
            }
        ),
    )
    _write_rdata_table(data_dir / f"test{suffix}", pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}))
    _write_rdata_table(data_dir / f"sample_submission{suffix}", pd.DataFrame({"id": [100, 101], "target": [0, 0]}))

    script = (
        kaggle_notebook.render_kernel_main("demo", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize(
    ("filename", "reader"),
    [
        ("submission.tsv", lambda path, pd: pd.read_csv(path, sep="\t")),
        ("submission.tab", lambda path, pd: pd.read_csv(path, sep="\t")),
        ("submission.psv", lambda path, pd: pd.read_csv(path, sep="|")),
        ("submission.jsonl", lambda path, pd: pd.read_json(path, lines=True)),
        ("submission.ndjson", lambda path, pd: pd.read_json(path, lines=True)),
        ("submission.xlsx", lambda path, pd: pd.read_excel(path)),
        ("submission.pkl", lambda path, pd: pd.read_pickle(path)),
        ("submission.pkl.gz", lambda path, pd: pd.read_pickle(path)),
        ("submission.dta", lambda path, pd: pd.read_stata(path)),
        ("submission.html", lambda path, pd: pd.read_html(path)[0]),
        (
            "submission.html.zst",
            lambda path, pd: pd.read_html(
                io.StringIO(zstd.ZstdDecompressor().decompress(path.read_bytes()).decode("utf-8"))
            )[0],
        ),
        ("submission.xml", lambda path, pd: pd.read_xml(path, parser="etree")),
        (
            "submission.xml.zst",
            lambda path, pd: pd.read_xml(
                io.BytesIO(zstd.ZstdDecompressor().decompress(path.read_bytes())),
                parser="etree",
            ),
        ),
        ("submission.csv.gz", lambda path, pd: pd.read_csv(path)),
        ("submission.jsonl.gz", lambda path, pd: pd.read_json(path, lines=True)),
        ("predictions.csv.gz", lambda path, pd: pd.read_csv(path)),
    ],
)
def test_generated_kernel_respects_non_csv_submission_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    reader,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", filename)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission_path = working_dir / filename
    assert submission_path.exists()
    submission = reader(submission_path, pd)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize(
    ("requested_name", "fallback_name"),
    [
        ("answers.nii.gz", "answers.tabular.csv"),
        ("submission.tar.gz", "submission.tabular.csv"),
    ],
)
def test_generated_kernel_does_not_write_csv_payload_to_non_tabular_submission_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_name: str,
    fallback_name: str,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", requested_name)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    requested_path = working_dir / requested_name
    fallback_path = working_dir / fallback_name
    manifest_path = working_dir / "submission_manifest.json"
    assert not requested_path.exists()
    assert fallback_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "tabular"
    assert manifest["submission_path"] == fallback_name
    assert manifest["requested_output_path"] == requested_name
    submission = pd.read_csv(fallback_path)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_non_tabular_fallback_uses_sample_tabular_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.tsv",
        sep="\t",
        index=False,
    )
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "answers.nii.gz")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    requested_path = working_dir / "answers.nii.gz"
    fallback_path = working_dir / "answers.tabular.tsv"
    manifest_path = working_dir / "submission_manifest.json"
    assert not requested_path.exists()
    assert fallback_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "tabular"
    assert manifest["submission_path"] == "answers.tabular.tsv"
    assert manifest["requested_output_path"] == "answers.nii.gz"
    submission = pd.read_csv(fallback_path, sep="\t")
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


@pytest.mark.parametrize("configured_name", ["sample_submission.csv", "sample-submission.csv.gz", "metrics.json"])
def test_generated_kernel_rejects_configured_template_or_reserved_submission_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_name: str,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    with gzip.open(data_dir / "sample_submission.tsv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n100\t0\n101\t0\n102\t0\n")
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", configured_name)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission_path = working_dir / "submission.tsv.gz"
    assert submission_path.exists()
    if configured_name == "metrics.json":
        metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "score" in metrics
    else:
        assert not (working_dir / configured_name).exists()
    submission = pd.read_csv(submission_path, sep="\t")
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_prefers_usable_jsonl_sample_over_header_only_csv(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n", encoding="utf-8")
    (data_dir / "SampleSubmission.jsonl").write_text(
        '{"id":100,"target":0}\n{"id":101,"target":0}\n{"id":102,"target":0}\n',
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_json(working_dir / "submission.jsonl", lines=True)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_synthesizes_sample_from_submission_format_and_test_table(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "submission_format.md").write_text(
        "Upload `submission.csv` with columns row_id,target.\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / "sample_submission_synth.csv"
    assert synthesized.exists()
    synthesized_sample = pd.read_csv(synthesized)
    assert synthesized_sample["row_id"].tolist() == [100, 101, 102]
    assert list(synthesized_sample.columns) == ["row_id", "target"]

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["row_id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


def test_generated_kernel_synthesizes_sample_from_overview_submission_section(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "overview.md").write_text(
        "# Data\n\n"
        "| field | meaning |\n"
        "| --- | --- |\n"
        "| feature | input value |\n\n"
        "# Submission Format\n\n"
        "Submit a CSV with header:\n\n"
        "```csv\n"
        "row_id,target\n"
        "100,0\n"
        "```\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / "sample_submission_synth.csv"
    assert synthesized.exists()
    synthesized_sample = pd.read_csv(synthesized)
    assert synthesized_sample["row_id"].tolist() == [100, 101, 102]
    assert list(synthesized_sample.columns) == ["row_id", "target"]

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["row_id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


def test_generated_kernel_synthesizes_jsonl_sample_from_context_suffix(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "overview.md").write_text(
        "# Submission\n\nSubmit `submission.jsonl` with columns row_id,target.\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / "sample_submission_synth.jsonl"
    assert synthesized.exists()
    synthesized_sample = pd.read_json(synthesized, lines=True)
    assert synthesized_sample["row_id"].tolist() == [100, 101, 102]
    assert list(synthesized_sample.columns) == ["row_id", "target"]

    submission = pd.read_json(working_dir / "submission.jsonl", lines=True)
    assert submission["row_id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


def test_generated_kernel_synthesizes_compressed_jsonl_sample_from_context_suffix(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "overview.md").write_text(
        "# Submission\n\nSubmit `submission.jsonl.gz` with columns row_id,target.\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / "sample_submission_synth.jsonl.gz"
    assert synthesized.exists()
    synthesized_sample = pd.read_json(synthesized, lines=True)
    assert synthesized_sample["row_id"].tolist() == [100, 101, 102]
    assert list(synthesized_sample.columns) == ["row_id", "target"]

    submission = pd.read_json(working_dir / "submission.jsonl.gz", lines=True)
    assert submission["row_id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


def test_generated_kernel_synthesizes_compressed_csv_zst_sample_from_context_suffix(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("zstandard")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "overview.md").write_text(
        "# Submission\n\nSubmit `submission.csv.zst` with columns row_id,target.\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / "sample_submission_synth.csv.zst"
    assert synthesized.exists()
    synthesized_sample = pd.read_csv(synthesized)
    assert synthesized_sample["row_id"].tolist() == [100, 101, 102]
    assert list(synthesized_sample.columns) == ["row_id", "target"]

    submission = pd.read_csv(working_dir / "submission.csv.zst")
    assert submission["row_id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Submit an xz-compressed YAML file with columns row_id,target.", ".yaml.xz"),
        ("Submit a zstd-compressed NDJSON file with columns row_id,target.", ".ndjson.zst"),
        ("Submit a bzip2-compressed HTML file with columns row_id,target.", ".html.bz2"),
        ("Submit an xz-compressed PSV file with columns row_id,target.", ".psv.xz"),
        ("Submit a zstd-compressed TAB file with columns row_id,target.", ".tab.zst"),
    ],
)
def test_generated_kernel_synthesizes_compressed_structured_sample_from_context_keywords(
    tmp_path: Path,
    description: str,
    suffix: str,
) -> None:
    pd = pytest.importorskip("pandas")
    if suffix.endswith(".zst"):
        pytest.importorskip("zstandard")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "overview.md").write_text(f"# Submission\n\n{description}\n", encoding="utf-8")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / f"sample_submission_synth{suffix}"
    assert synthesized.exists()
    synthesized_sample = read_table(synthesized)
    assert synthesized_sample["row_id"].astype(int).tolist() == [100, 101, 102]
    assert list(synthesized_sample.columns) == ["row_id", "target"]

    submission = read_table(working_dir / f"submission{suffix}")
    assert submission["row_id"].astype(int).tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


def test_generated_kernel_synthesizes_jsonl_sample_from_json_object_example(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "overview.md").write_text(
        "# Submission\n\n"
        "Submit `submission.jsonl` with one JSON object per prediction:\n\n"
        "```json\n"
        "{\n"
        '  "row_id": 100,\n'
        '  "target": 0\n'
        "}\n"
        "```\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / "sample_submission_synth.jsonl"
    assert synthesized.exists()
    synthesized_sample = pd.read_json(synthesized, lines=True)
    assert synthesized_sample["row_id"].tolist() == [100, 101, 102]
    assert list(synthesized_sample.columns) == ["row_id", "target"]

    submission = pd.read_json(working_dir / "submission.jsonl", lines=True)
    assert submission["row_id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


def test_generated_kernel_does_not_use_input_suffix_for_synthesized_sample(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"row_id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "overview.md").write_text(
        "# Submission\n\n"
        "The training data may be provided as train.parquet in some mirrors.\n"
        "Submit a CSV with columns row_id,target.\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = working_dir / "synthetic_tables" / "sample_submission_synth.csv"
    assert synthesized.exists()
    assert not (working_dir / "synthetic_tables" / "sample_submission_synth.parquet").exists()

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["row_id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["row_id", "target"]


def test_generated_kernel_reads_wrapped_json_tables(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "train.json").write_text(
        json.dumps(
            {
                "data": [{"id": idx, "feature": float(idx), "target": idx % 2} for idx in range(20)],
                "metadata": {"split": "train"},
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "test.json").write_text(
        json.dumps(
            {
                "records": [
                    {"id": 100, "feature": 1.0},
                    {"id": 101, "feature": 2.0},
                    {"id": 102, "feature": 3.0},
                ],
                "count": 3,
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "sample_submission.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"id": 100, "target": 0},
                    {"id": 101, "target": 0},
                    {"id": 102, "target": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission_path = working_dir / "submission.json"
    assert submission_path.exists()
    submission = pd.read_json(submission_path)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_geojson_feature_tables(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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
            }
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
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission_path = working_dir / "submission.csv"
    assert submission_path.exists()
    submission = pd.read_csv(submission_path)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_reads_compressed_geojson_feature_tables(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    compressor = zstd.ZstdCompressor()
    train_payload = {
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
    }
    test_payload = {
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
    }
    (data_dir / "train.geojson.zst").write_bytes(compressor.compress(json.dumps(train_payload).encode("utf-8")))
    (data_dir / "test.geojson.zst").write_bytes(compressor.compress(json.dumps(test_payload).encode("utf-8")))
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission_path = working_dir / "submission.csv"
    assert submission_path.exists()
    submission = pd.read_csv(submission_path)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_derives_submission_filename_from_jsonl_sample_without_env(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (data_dir / "SampleSubmission.jsonl").write_text(
        '{"id":100,"target":0}\n{"id":101,"target":0}\n{"id":102,"target":0}\n',
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission_path = working_dir / "submission.jsonl"
    assert submission_path.exists()
    assert not (working_dir / "submission.csv").exists()
    submission = pd.read_json(submission_path, lines=True)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_derives_submission_filename_from_pickle_sample_without_env(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_pickle(
        data_dir / "SampleSubmission.pkl.gz",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission_path = working_dir / "submission.pkl.gz"
    assert submission_path.exists()
    assert not (working_dir / "submission.csv").exists()
    submission = pd.read_pickle(submission_path)
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_honors_sample_submission_stage_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "ID": range(30),
            "feature": [float(idx) for idx in range(30)],
            "Pred": [idx % 2 for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"ID": ["S1_A", "S1_B"], "feature": [1.0, 2.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (data_dir / "sample_submission.csv").write_text("ID,Pred\n", encoding="utf-8")
    pd.DataFrame({"ID": ["S1_A", "S1_B"], "Pred": [0, 0]}).to_csv(
        data_dir / "SampleSubmissionStage1.csv",
        index=False,
    )
    pd.DataFrame({"ID": ["S2_A", "S2_B"], "Pred": [0, 0]}).to_csv(
        data_dir / "SampleSubmissionStage2.csv",
        index=False,
    )
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_STAGE", "1")

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["ID"].tolist() == ["S1_A", "S1_B"]
    assert list(submission.columns) == ["ID", "Pred"]


def test_generated_kernel_handles_singleton_minority_class(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(25),
            "feature": [float(idx) for idx in range(25)],
            "target": [1] + [0 for _ in range(24)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["task"] == "classification"


def test_generated_kernel_uses_timeseries_holdout_for_future_time_column(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(24),
            "date_block_num": list(range(24)),
            "feature": [float(idx) for idx in range(24)],
            "target": [float(idx * 1.5) for idx in range(24)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "date_block_num": [24, 25, 26],
            "feature": [24.0, 25.0, 26.0],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "target": [0.0, 0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["time_column"] == "date_block_num"
    assert metrics["split_strategy"] == "timeseries_holdout"
    assert metrics["target_metrics"]["target"]["split_strategy"] == "timeseries_holdout"


def test_generated_kernel_adds_calendar_features_for_datetime_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(24),
            "date": pd.date_range("2024-01-01", periods=24, freq="D").strftime("%Y-%m-%d"),
            "feature": [float(idx) for idx in range(24)],
            "target": [float(idx * 1.5) for idx in range(24)],
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    temporal_features = metrics["temporal_calendar_feature_columns"]
    assert "__time_date_year" in temporal_features
    assert "__time_date_dayofweek" in temporal_features
    assert metrics["target_metrics"]["target"]["features"] > 2


def test_generated_kernel_handles_constant_classification_target(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": range(12), "feature": [float(idx) for idx in range(12)], "target": [0] * 12}).to_csv(
        data_dir / "train.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["target"].tolist() == [0, 0, 0]
    assert metrics["metric"] == "accuracy"


def test_generated_kernel_aligns_composite_sample_id_from_test_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["row_id"].tolist() == ["u2:i2", "u1:i1"]
    assert submission.loc[0, "target"] > submission.loc[1, "target"]


def test_generated_kernel_fits_all_multi_target_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target_class": [idx % 2 for idx in range(30)],
            "target_value": [float(idx) * 1.5 for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [100.0, 101.0, 102.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "target_class": [0, 0, 0],
            "target_value": [0.0, 0.0, 0.0],
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "target_class", "target_value"]
    assert submission["id"].tolist() == [100, 101, 102]
    assert not submission["target_class"].isna().any()
    assert not submission["target_value"].isna().any()
    assert submission["target_value"].tolist() != [0.0, 0.0, 0.0]
    assert set(metrics["target_metrics"]) == {"target_class", "target_value"}
    assert metrics["target_metrics"]["target_class"]["metric"] == "accuracy"
    assert metrics["target_metrics"]["target_value"]["metric"] == "rmse"


def test_generated_kernel_expands_single_label_to_class_probability_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    labels = ["cat", "dog", "bird"] * 12
    pd.DataFrame(
        {
            "id": range(len(labels)),
            "feature": [float(idx) for idx in range(len(labels))],
            "label": labels,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [100.0, 101.0, 102.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "class_bird": [1 / 3, 1 / 3, 1 / 3],
            "class_cat": [1 / 3, 1 / 3, 1 / 3],
            "class_dog": [1 / 3, 1 / 3, 1 / 3],
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    probability_cols = ["class_bird", "class_cat", "class_dog"]
    assert list(submission.columns) == ["id", *probability_cols]
    assert submission["id"].tolist() == [100, 101, 102]
    assert "label" not in submission.columns
    assert metrics["target_metrics"]["label"]["prediction_kind"] == "probability_columns"
    assert metrics["target_metrics"]["label"]["probability_columns"] == probability_cols
    row_sums = submission[probability_cols].sum(axis=1).round(6).tolist()
    assert row_sums == [1.0, 1.0, 1.0]


def test_generated_kernel_maps_suffix_probability_columns_by_label_name(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "dog_probability": [1 / 3, 1 / 3, 1 / 3],
            "cat_probability": [1 / 3, 1 / 3, 1 / 3],
            "bird_probability": [1 / 3, 1 / 3, 1 / 3],
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    probability_cols = ["dog_probability", "cat_probability", "bird_probability"]
    assert list(submission.columns) == ["id", *probability_cols]
    assert submission[probability_cols].idxmax(axis=1).tolist() == [
        "bird_probability",
        "cat_probability",
        "dog_probability",
    ]
    assert metrics["target_metrics"]["label"]["prediction_kind"] == "probability_columns"
    assert metrics["target_metrics"]["label"]["probability_columns"] == probability_cols
    row_sums = submission[probability_cols].sum(axis=1).round(6).tolist()
    assert row_sums == [1.0, 1.0, 1.0]


def test_generated_kernel_uses_probability_for_probability_named_binary_target(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "isFraud"]
    assert submission["isFraud"].between(0.0, 1.0).all()
    assert not set(submission["isFraud"].round(8)).issubset({0.0, 1.0})
    assert metrics["target_metrics"]["isFraud"]["prediction_kind"] == "probability"


def test_generated_kernel_expands_multi_label_target_to_label_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    rows = list(range(36))
    labels = ["cat dog", "dog bird", "cat bird"] * 12
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "labels": labels,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [100.0, 101.0, 102.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "cat": [0.0, 0.0, 0.0],
            "dog": [0.0, 0.0, 0.0],
            "bird": [0.0, 0.0, 0.0],
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    label_cols = ["cat", "dog", "bird"]
    assert list(submission.columns) == ["id", *label_cols]
    assert submission["id"].tolist() == [100, 101, 102]
    assert metrics["target_metrics"]["labels"]["prediction_kind"] == "multi_label_columns"
    assert set(metrics["target_metrics"]["labels"]["prediction_columns"]) == set(label_cols)
    assert ((submission[label_cols] >= 0.0) & (submission[label_cols] <= 1.0)).all().all()
    assert submission[label_cols].sum(axis=1).round(6).tolist() != [1.0, 1.0, 1.0]


def test_generated_kernel_expands_single_regression_label_to_quantile_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "target": [float(idx * 2 + (idx % 5)) for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [100.0, 101.0, 102.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "p10": [0.0, 0.0, 0.0],
            "p50": [0.0, 0.0, 0.0],
            "p90": [0.0, 0.0, 0.0],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "p10", "p50", "p90"]
    assert submission["id"].tolist() == [100, 101, 102]
    assert "target" not in submission.columns
    assert metrics["target_metrics"]["target"]["prediction_kind"] == "quantile_columns"
    assert metrics["target_metrics"]["target"]["expanded_prediction_columns"] == ["p10", "p50", "p90"]
    assert (submission["p10"] <= submission["p50"]).all()
    assert (submission["p50"] <= submission["p90"]).all()
    assert submission[["p10", "p50", "p90"]].nunique().min() > 1


def test_generated_kernel_expands_single_regression_label_to_interval_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "target": [float(idx * 3 - (idx % 7)) for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [100.0, 101.0, 102.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "lower": [0.0, 0.0, 0.0],
            "upper": [0.0, 0.0, 0.0],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "lower", "upper"]
    assert submission["id"].tolist() == [100, 101, 102]
    assert "target" not in submission.columns
    assert metrics["target_metrics"]["target"]["prediction_kind"] == "prediction_interval_columns"
    assert metrics["target_metrics"]["target"]["expanded_prediction_columns"] == ["lower", "upper"]
    assert (submission["lower"] <= submission["upper"]).all()
    assert submission[["lower", "upper"]].nunique().min() > 1


def test_generated_kernel_expands_single_regression_label_to_continuous_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "target": [float(idx * 1.5 + (idx % 3)) for idx in rows],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [100.0, 101.0, 102.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": [100, 101, 102],
            "prediction_a": [0.0, 0.0, 0.0],
            "prediction_b": [0.0, 0.0, 0.0],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "prediction_a", "prediction_b"]
    assert submission["id"].tolist() == [100, 101, 102]
    assert "target" not in submission.columns
    assert metrics["target_metrics"]["target"]["prediction_kind"] == "continuous_columns"
    assert metrics["target_metrics"]["target"]["expanded_prediction_columns"] == ["prediction_a", "prediction_b"]
    assert submission["prediction_a"].tolist() == submission["prediction_b"].tolist()
    assert submission["prediction_a"].nunique() > 1


def test_generated_kernel_writes_unlabeled_anomaly_score_submission(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "anomaly_score"]
    assert submission["id"].tolist() == [4, 5]
    assert submission["anomaly_score"].between(0.0, 1.0).all()
    assert submission.loc[1, "anomaly_score"] > submission.loc[0, "anomaly_score"]
    assert metrics["target_metrics"]["anomaly_score"]["task"] == "unsupervised"
    assert metrics["target_metrics"]["anomaly_score"]["model_kind"] == "robust_unsupervised_anomaly_score"


def test_generated_kernel_writes_unlabeled_anomaly_score_submission_without_id(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["anomaly_score"]
    assert len(submission) == 2
    assert submission["anomaly_score"].between(0.0, 1.0).all()
    assert submission.loc[1, "anomaly_score"] > submission.loc[0, "anomaly_score"]
    assert metrics["target_metrics"]["anomaly_score"]["task"] == "unsupervised"
    assert metrics["target_metrics"]["anomaly_score"]["model_kind"] == "robust_unsupervised_anomaly_score"


def test_generated_kernel_writes_survival_event_time_as_single_risk_score(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "prediction"]
    assert submission["id"].tolist() == [100, 101]
    assert "efs" not in submission.columns
    assert "efs_time" not in submission.columns
    assert submission["prediction"].between(0.0, 1.0).all()
    assert metrics["target_metrics"]["prediction"]["task"] == "survival"
    assert metrics["target_metrics"]["prediction"]["model_kind"] == "survival_risk_score"


def test_generated_kernel_treats_learning_to_rank_relevance_as_continuous_score(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "relevance"]
    assert submission["id"].tolist() == [7, 8]
    assert pd.api.types.is_float_dtype(submission["relevance"])
    assert metrics["target_metrics"]["relevance"]["task"] == "regression"
    assert metrics["target_metrics"]["relevance"]["prediction_kind"] == "continuous"


def test_generated_kernel_treats_ordinal_target_as_rounded_score(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "severity"]
    assert submission["id"].tolist() == [101, 102, 103]
    assert set(submission["severity"]).issubset({0, 1, 2, 3, 4})
    assert pd.api.types.is_integer_dtype(submission["severity"])
    assert metrics["target_metrics"]["severity"]["task"] == "regression"
    assert metrics["target_metrics"]["severity"]["prediction_kind"] == "ordinal"


def test_generated_kernel_treats_named_low_cardinality_numeric_target_as_regression(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "sales"]
    assert pd.api.types.is_float_dtype(submission["sales"])
    assert metrics["target_metrics"]["sales"]["task"] == "regression"
    assert metrics["target_metrics"]["sales"]["prediction_kind"] == "continuous"
    assert metrics["target_metrics"]["sales"]["target_transform"] is None


def test_generated_kernel_clips_bounded_regression_predictions(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "conversion_rate"]
    assert submission["conversion_rate"].between(0.0, 1.0).all()
    assert submission["conversion_rate"].max() == 1.0
    assert metrics["target_metrics"]["conversion_rate"]["task"] == "regression"
    assert metrics["target_metrics"]["conversion_rate"]["target_transform"] is None


def test_generated_kernel_uses_log1p_for_count_regression(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "count"]
    assert (submission["count"] >= 0.0).all()
    assert submission["count"].min() == 0.0
    assert metrics["target_metrics"]["count"]["task"] == "regression"
    assert metrics["target_metrics"]["count"]["metric"] == "rmsle"
    assert metrics["target_metrics"]["count"]["target_transform"] == "log1p"
    assert metrics["target_metrics"]["count"]["inverse_transform"] == "expm1"


def test_generated_kernel_uses_log1p_for_positive_skew_regression(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "SalePrice"]
    assert (submission["SalePrice"] >= 0.0).all()
    assert submission["SalePrice"].min() == 0.0
    assert metrics["target_metrics"]["SalePrice"]["task"] == "regression"
    assert metrics["target_metrics"]["SalePrice"]["metric"] == "rmsle"
    assert metrics["target_metrics"]["SalePrice"]["target_transform"] == "log1p"
    assert metrics["target_metrics"]["SalePrice"]["inverse_transform"] == "expm1"


def test_generated_kernel_handles_text_target_submission(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["a", "b", "c", "d"],
            "translation": ["alpha one", "beta two", "alpha one", "gamma three"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "prompt": ["b", "d"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "translation": ["", ""]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "translation"]
    assert submission["id"].tolist() == [100, 101]
    assert submission["translation"].tolist() == ["beta two", "gamma three"]
    assert metrics["target_metrics"]["translation"]["prediction_kind"] == "text"
    assert metrics["target_metrics"]["translation"]["model_kind"] == "tfidf_nearest_neighbor"
    assert metrics["target_metrics"]["translation"]["text_feature_columns"] == ["prompt"]


def test_generated_kernel_handles_natural_language_target_submission(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "target"]
    assert submission["target"].tolist() == [
        "write a concise answer about blue oceans",
        "write a concise answer about green forests",
    ]
    assert metrics["target_metrics"]["target"]["task"] == "text"
    assert metrics["target_metrics"]["target"]["prediction_kind"] == "text"
    assert metrics["target_metrics"]["target"]["model_kind"] == "tfidf_nearest_neighbor"
    assert metrics["target_metrics"]["target"]["text_feature_columns"] == ["prompt"]


def test_generated_kernel_handles_delimited_multi_label_target_submission(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
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

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "labels"]
    assert submission["id"].tolist() == [100, 101]
    assert submission["labels"].tolist() == ["dog bird", "cat bird"]
    assert metrics["target_metrics"]["labels"]["prediction_kind"] == "text"
    assert metrics["target_metrics"]["labels"]["model_kind"] == "tfidf_nearest_neighbor"


def test_generated_kernel_synthesizes_text_sample_with_string_defaults(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    context_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["a", "b", "c", "d"],
            "translation": ["alpha one", "beta two", "alpha one", "gamma three"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "prompt": ["b", "d"]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    (context_dir / "submission_format.md").write_text(
        "CSV header: id,translation\nThe translation column is a string.\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized = pd.read_csv(working_dir / "synthetic_tables" / "sample_submission_synth.csv")
    assert synthesized["translation"].tolist() == ["-", "-"]

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "translation"]
    assert submission["id"].tolist() == [100, 101]
    assert submission["translation"].tolist() == ["beta two", "gamma three"]
    assert metrics["target_metrics"]["translation"]["prediction_kind"] == "text"
    assert metrics["target_metrics"]["translation"]["model_kind"] == "tfidf_nearest_neighbor"


def test_generated_kernel_routes_rle_segmentation_to_empty_mask_baseline(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": ["train_1", "train_2", "train_3"],
            "image_path": ["train_1.png", "train_2.png", "train_3.png"],
            "EncodedPixels": ["1 2 10 3", "", "4 1 20 2"],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": ["test_1", "test_2"],
            "image_path": ["test_1.png", "test_2.png"],
        }
    ).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": ["test_1", "test_2"], "EncodedPixels": ["", ""]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv", keep_default_na=False)
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert list(submission.columns) == ["id", "EncodedPixels"]
    assert submission["id"].tolist() == ["test_1", "test_2"]
    assert submission["EncodedPixels"].tolist() == ["-", "-"]
    assert metrics["task"] == "segmentation"
    assert metrics["metric"] == "segmentation_rle_placeholder"
    assert metrics["target_metrics"]["EncodedPixels"]["prediction_kind"] == "rle"
    assert metrics["target_metrics"]["EncodedPixels"]["model_kind"] == "rle_empty_mask_baseline"


def test_generated_kernel_adds_file_reference_metadata_features(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    image_train_dir = data_dir / "images" / "train"
    image_test_dir = data_dir / "images" / "test"
    image_train_dir.mkdir(parents=True)
    image_test_dir.mkdir(parents=True)
    working_dir.mkdir()

    rows = []
    for idx in range(30):
        filename = f"train_{idx}.jpg"
        target = int(idx >= 15)
        size = 20 if target == 0 else 400
        (image_train_dir / filename).write_bytes(b"x" * size)
        rows.append({"id": idx, "filename": filename, "target": target})
    pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
    (image_test_dir / "test_small.jpg").write_bytes(b"x" * 20)
    (image_test_dir / "test_large.jpg").write_bytes(b"x" * 400)
    pd.DataFrame(
        {
            "id": [100, 101],
            "filename": ["test_small.jpg", "test_large.jpg"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert submission["target"].tolist() == [0, 1]
    assert "__file_filename_bytes" in metrics["file_reference_feature_columns"]
    assert "__file_filename_suffix" in metrics["file_reference_feature_columns"]


def test_generated_kernel_adds_document_file_reference_metadata_features(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    train_dir = data_dir / "documents" / "train"
    test_dir = data_dir / "documents" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    working_dir.mkdir()

    rows = []
    for idx in range(30):
        filename = f"train_{idx}.md.gz"
        target = int(idx >= 15)
        words = "short text" if target == 0 else "long document text with many repeated words"
        with gzip.open(train_dir / filename, "wt", encoding="utf-8") as handle:
            handle.write(words)
        rows.append({"id": idx, "document_path": filename, "target": target})
    pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
    with gzip.open(test_dir / "test_short.md.gz", "wt", encoding="utf-8") as handle:
        handle.write("short text")
    with gzip.open(test_dir / "test_long.md.gz", "wt", encoding="utf-8") as handle:
        handle.write("long document text with many repeated words")
    pd.DataFrame(
        {
            "id": [100, 101],
            "document_path": ["test_short.md.gz", "test_long.md.gz"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert "__file_document_path_document_words" in metrics["file_reference_feature_columns"]
    assert "__file_document_path_document_paragraphs" in metrics["file_reference_feature_columns"]
    assert metrics["file_reference_feature_summary"]["__file_document_path_suffix"]["train"] == {".md.gz": 30}
    assert metrics["file_reference_feature_summary"]["__file_document_path_suffix"]["test"] == {".md.gz": 2}


def test_generated_kernel_resolves_scan_file_reference_metadata_features(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    train_dir = data_dir / "scans" / "train"
    test_dir = data_dir / "scans" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    working_dir.mkdir()

    rows = []
    for idx in range(30):
        filename = f"train_{idx}.nii.gz"
        target = int(idx >= 15)
        size = 20 if target == 0 else 400
        (train_dir / filename).write_bytes(b"x" * size)
        rows.append({"id": idx, "scan_path": f"train_{idx}", "target": target})
    pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
    (test_dir / "test_small.nii.gz").write_bytes(b"x" * 20)
    (test_dir / "test_large.nii.gz").write_bytes(b"x" * 400)
    pd.DataFrame(
        {
            "id": [100, 101],
            "scan_path": ["test_small", "test_large"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert submission["target"].tolist() == [0, 1]
    assert "__file_scan_path_bytes" in metrics["file_reference_feature_columns"]
    assert "__file_scan_path_suffix" in metrics["file_reference_feature_columns"]
    assert metrics["file_reference_feature_summary"]["__file_scan_path_suffix"]["train"] == {".nii.gz": 30}
    assert metrics["file_reference_feature_summary"]["__file_scan_path_suffix"]["test"] == {".nii.gz": 2}


def test_generated_kernel_resolves_dicom_series_file_reference_metadata_features(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    train_dir = data_dir / "dicom" / "train"
    test_dir = data_dir / "dicom" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    working_dir.mkdir()

    rows = []
    for idx in range(30):
        series_id = f"series_{idx}"
        target = int(idx >= 15)
        size = 20 if target == 0 else 400
        (train_dir / f"{series_id}.dcm").write_bytes(b"x" * size)
        rows.append({"id": idx, "series_id": series_id, "target": target})
    pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
    (test_dir / "test_small.dcm").write_bytes(b"x" * 20)
    (test_dir / "test_large.dcm").write_bytes(b"x" * 400)
    pd.DataFrame(
        {
            "id": [100, 101],
            "series_id": ["test_small", "test_large"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert submission["target"].tolist() == [0, 1]
    assert "__file_series_id_bytes" in metrics["file_reference_feature_columns"]
    assert "__file_series_id_suffix" in metrics["file_reference_feature_columns"]
    assert metrics["file_reference_feature_summary"]["__file_series_id_suffix"]["train"] == {".dcm": 30}
    assert metrics["file_reference_feature_summary"]["__file_series_id_suffix"]["test"] == {".dcm": 2}


def test_generated_kernel_synthesizes_asset_tables_when_test_csv_missing(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    train_dir = data_dir / "images" / "train"
    test_dir = data_dir / "images" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    working_dir.mkdir()

    rows = []
    for idx in range(30):
        image_id = f"train_{idx}"
        target = int(idx >= 15)
        size = 20 if target == 0 else 400
        (train_dir / f"{image_id}.jpg").write_bytes(b"x" * size)
        rows.append({"id": image_id, "target": target})
    pd.DataFrame(rows).to_csv(data_dir / "train_labels.csv", index=False)
    (test_dir / "test_small.jpg").write_bytes(b"x" * 20)
    (test_dir / "test_large.jpg").write_bytes(b"x" * 400)
    pd.DataFrame({"id": ["test_small", "test_large"], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == ["test_small", "test_large"]
    assert submission["target"].tolist() == [0, 1]
    assert (working_dir / "synthetic_tables" / "train_synth.csv").exists()
    assert (working_dir / "synthetic_tables" / "test_synth.csv").exists()
    assert "__file_asset_path_bytes" in metrics["file_reference_feature_columns"]
    assert metrics["file_reference_feature_summary"]["__file_asset_path_suffix"]["test"] == {".jpg": 2}


def test_generated_kernel_synthesizes_sample_and_asset_tables_when_sample_missing(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    context_dir = data_dir / "context"
    working_dir = tmp_path / "working"
    train_dir = data_dir / "images" / "train"
    test_dir = data_dir / "images" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    working_dir.mkdir()

    rows = []
    for idx in range(30):
        image_id = f"train_{idx}"
        target = int(idx >= 15)
        size = 20 if target == 0 else 400
        (train_dir / f"{image_id}.jpg").write_bytes(b"x" * size)
        rows.append({"id": image_id, "target": target})
    pd.DataFrame(rows).to_csv(data_dir / "train_labels.csv", index=False)
    (test_dir / "test_small.jpg").write_bytes(b"x" * 20)
    (test_dir / "test_large.jpg").write_bytes(b"x" * 400)
    (context_dir / "submission_format.md").write_text(
        "Submit a CSV like:\n\n```csv\nid,target\nimages/test/test_small.jpg,0\n```\n",
        encoding="utf-8",
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    synthesized_sample = pd.read_csv(working_dir / "synthetic_tables" / "sample_submission_synth.csv")
    assert synthesized_sample["id"].tolist() == [
        "images/test/test_large.jpg",
        "images/test/test_small.jpg",
    ]
    assert list(synthesized_sample.columns) == ["id", "target"]

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [
        "images/test/test_large.jpg",
        "images/test/test_small.jpg",
    ]
    assert submission["target"].tolist() == [1, 0]
    assert (working_dir / "synthetic_tables" / "train_synth.csv").exists()
    assert (working_dir / "synthetic_tables" / "test_synth.csv").exists()
    assert "__file_asset_path_bytes" in metrics["file_reference_feature_columns"]


def test_generated_kernel_resolves_array_file_reference_metadata_features(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    train_dir = data_dir / "arrays" / "train"
    test_dir = data_dir / "arrays" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    working_dir.mkdir()

    rows = []
    for idx in range(30):
        filename = f"train_{idx}.npy"
        target = int(idx >= 15)
        size = 20 if target == 0 else 400
        (train_dir / filename).write_bytes(b"x" * size)
        rows.append({"id": idx, "array_path": f"train_{idx}", "target": target})
    pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
    (test_dir / "test_small.npy").write_bytes(b"x" * 20)
    (test_dir / "test_large.npy").write_bytes(b"x" * 400)
    pd.DataFrame(
        {
            "id": [100, 101],
            "array_path": ["test_small", "test_large"],
        }
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101]
    assert submission["target"].tolist() == [0, 1]
    assert "__file_array_path_bytes" in metrics["file_reference_feature_columns"]
    assert "__file_array_path_suffix" in metrics["file_reference_feature_columns"]
    assert metrics["file_reference_feature_summary"]["__file_array_path_suffix"]["train"] == {".npy": 30}
    assert metrics["file_reference_feature_summary"]["__file_array_path_suffix"]["test"] == {".npy": 2}


def test_generated_kernel_merges_separate_train_labels(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": range(30), "target": [idx % 2 for idx in range(30)]}).to_csv(
        data_dir / "train_labels.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "segment": ["a" if idx % 2 else "b" for idx in range(30)],
        }
    ).to_csv(data_dir / "train_features.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0], "segment": ["a", "b", "a"]}).to_csv(
        data_dir / "test_features.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    assert submission["id"].tolist() == [100, 101, 102]
    assert list(submission.columns) == ["id", "target"]
    assert metrics["task"] == "classification"


def test_generated_kernel_regression_uses_rmse_without_squared_argument(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx) * 1.5 for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0.0, 0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    submission = pd.read_csv(working_dir / "submission.csv")
    assert metrics["metric"] == "rmse"
    assert submission["id"].tolist() == [100, 101, 102]


def test_runner_submission_discovery_uses_fold_intermediate_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fold1 = output_dir / "submission_model_fold1.csv"
    fold2 = output_dir / "nested" / "submission_model_fold2.csv"
    fold2.parent.mkdir()
    fold1.write_text("id,target\n1,0.1\n", encoding="utf-8")
    fold2.write_text("id,target\n1,0.2\n", encoding="utf-8")

    assert find_submission_file(output_dir) == fold2
