from __future__ import annotations

import gzip
import io
import json
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

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_submit_wrapper import count_tabular_data_rows_at_most, render_submission_kernel_script
from kagglebot.solver.io import read_table, write_table
from kagglebot.submission_extension_hints import ARCHIVE_SUBMISSION_SUFFIXES, ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES
from kagglebot.test_table_aliases import STRONG_TEST_TABLE_TOKENS, TEST_TABLE_STEMS, WEAK_TEST_TABLE_TOKENS


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
    table: str = "test",
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


class _FakeRarMember:
    def __init__(self, filename: str, *, is_dir: bool = False, is_file: bool = True) -> None:
        self.filename = filename
        self._is_dir = is_dir
        self._is_file = is_file

    def is_dir(self) -> bool:
        return self._is_dir

    def is_file(self) -> bool:
        return self._is_file

    def is_symlink(self) -> bool:
        return False

    def needs_password(self) -> bool:
        return False


class _FakeRarFile:
    members_by_name: dict[str, list[_FakeRarMember]] = {}

    def __init__(self, path: Path) -> None:
        self.members = self.members_by_name[Path(path).name]

    def __enter__(self) -> _FakeRarFile:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def infolist(self) -> list[_FakeRarMember]:
        return self.members


def _write_valid_7z(path: Path) -> None:
    payload = path.with_name(f"{path.stem}_payload.txt")
    payload.write_text("ok\n", encoding="utf-8")
    with py7zr.SevenZipFile(path, "w") as archive:
        archive.write(payload, "predictions/a.txt")
    payload.unlink()


def test_render_submission_kernel_script_preserves_non_csv_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.tsv"
    submission_path.write_text("id\ttarget\n1\t0.1\n", encoding="utf-8")

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.tsv"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".tsv"' in script
    assert "unable to read submission.csv" not in script
    assert "submission.csv has no data rows" not in script
    assert "submission.csv contains empty values" not in script
    assert "__SAMPLE_OUTPUT_NAME_TOKENS__" not in script
    assert "__SAMPLE_COMPACT_NAME_ALIASES__" not in script
    assert "__TEST_TABLE_STEMS__" not in script
    assert "__STRONG_TEST_TABLE_TOKENS__" not in script
    assert "__WEAK_TEST_TABLE_TOKENS__" not in script
    assert "__TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES__" not in script
    assert "SAMPLE_OUTPUT_NAME_TOKENS = set((" in script
    assert f"TEST_TABLE_STEMS = {TEST_TABLE_STEMS!r}" in script
    assert "'answertemplate'" in script
    assert repr(tuple(sorted(STRONG_TEST_TABLE_TOKENS))) in script
    assert repr(tuple(sorted(WEAK_TEST_TABLE_TOKENS))) in script
    assert "TABULAR_ANNDATA_SUFFIXES" in script
    assert "'.h5ad'" in script
    assert "TABULAR_SVMLIGHT_SUFFIX_PREFIXES" in script
    assert "TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES" in script
    assert "TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES" in script
    assert "TABULAR_NETCDF_SUFFIXES" in script
    assert "'.cdf'" in script
    assert "'.nc4'" in script
    assert "TABULAR_FITS_SUFFIXES" in script
    assert "'.fits'" in script
    assert "'.fits.gz'" in script
    assert "TABULAR_LOOM_SUFFIXES" in script
    assert "'.loom'" in script
    assert "TABULAR_GEOPACKAGE_SUFFIXES" in script
    assert "'.gpkg'" in script
    assert "'.geopackage'" in script
    assert "TABULAR_SHAPEFILE_SUFFIXES" in script
    assert "'.shp'" in script
    assert "'.dbf'" in script
    assert "TABULAR_KML_SUFFIXES" in script
    assert "'.kml'" in script
    assert "'.kml.gz'" in script
    assert "'.kmz'" in script
    assert "TABULAR_PARQUET_SUFFIXES" in script
    assert "'.parq'" in script
    assert "'.pq'" in script
    assert "'.ftr'" in script
    assert "'.tab'" in script
    assert "'.psv'" in script
    assert "'.csv.zip'" in script
    assert "def _select_zip_tabular_member" in script
    assert "TABULAR_NUMPY_SUFFIXES" in script
    assert "def _read_svmlight_tabular_frame(path: Path)" in script
    assert "def _read_fixed_width_tabular_frame(path: Path)" in script
    assert "def _read_netcdf_tabular_frame(path: Path)" in script
    assert "def _read_fits_tabular_frame(path: Path)" in script
    assert "def _read_h5ad_tabular_frame(path: Path)" in script
    assert "def _read_loom_tabular_frame(path: Path)" in script
    assert "def _read_geopackage_tabular_frame(path: Path)" in script
    assert "def _read_shapefile_tabular_frame(path: Path)" in script
    assert "def _read_kml_tabular_frame(path: Path)" in script
    assert "return _decompress_compressed_payload(path.read_bytes(), suffix)" in script
    assert "def _read_numpy_tabular_frame(path: Path)" in script
    assert "def _numpy_column_names_for_path(path: Path | None, width: int)" in script


def test_rendered_submission_kernel_preserves_pipe_delimited_output_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission.psv"
    submission_path.write_text("id|target\n1|0.1\n", encoding="utf-8")
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.psv", sep="|")
    assert output.to_dict("list") == {"id": [1], "target": [0.1]}


def test_render_submission_kernel_script_keeps_header_only_html_tables_readable(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.html"
    pd.DataFrame(columns=["id", "target"]).to_html(submission_path, index=False)

    script = render_submission_kernel_script(submission_path)

    assert "table.shape[1] > 0" in script
    assert "not table.empty and table.shape[1] > 0" not in script


def test_render_submission_kernel_script_preserves_compressed_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.1\n")

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.csv.gz"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".csv.gz"' in script


def test_render_submission_kernel_script_preserves_zstd_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv.zst"
    submission_path.write_bytes(zstd.ZstdCompressor().compress(b"id,target\n1,0.1\n"))

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.csv.zst"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".csv.zst"' in script
    assert "__ASSET_COMPRESSION_SUFFIXES__" not in script
    assert "ASSET_COMPRESSION_SUFFIXES = (" in script
    assert "def _compression_suffix_for" in script
    assert "def _decompress_compressed_payload" in script
    assert "def _compress_payload_for_suffix" in script


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm", ".ods"])
def test_render_submission_kernel_script_preserves_excel_output_name(tmp_path: Path, suffix: str) -> None:
    submission_path = tmp_path / f"submission{suffix}"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_excel(submission_path, index=False)

    script = render_submission_kernel_script(submission_path)

    assert f'SUBMISSION_OUTPUT_NAME = "submission{suffix}"' in script
    assert f'SUBMISSION_INPUT_SUFFIX = "{suffix}"' in script
    assert "def _read_native_hdf_table(path: Path)" in script


def test_render_submission_kernel_script_preserves_orc_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.orc"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_orc(submission_path, index=False)

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.orc"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".orc"' in script


def test_render_submission_kernel_script_preserves_avro_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.avro"
    write_table(pd.DataFrame({"id": [1], "target": [0.1]}), submission_path)

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.avro"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".avro"' in script
    assert "def _read_avro_table(path: Path)" in script
    assert "def _write_avro_payload(frame, buffer)" in script


@pytest.mark.parametrize("suffix", [".hdf", ".hdf5"])
def test_render_submission_kernel_script_preserves_hdf_output_name(tmp_path: Path, suffix: str) -> None:
    submission_path = tmp_path / f"submission{suffix}"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_hdf(
        submission_path,
        key="submission",
        mode="w",
        format="table",
        index=False,
    )

    script = render_submission_kernel_script(submission_path)

    assert f'SUBMISSION_OUTPUT_NAME = "submission{suffix}"' in script
    assert f'SUBMISSION_INPUT_SUFFIX = "{suffix}"' in script


def test_render_submission_kernel_script_preserves_pickle_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.pkl"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_pickle(submission_path)

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.pkl"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".pkl"' in script


def test_render_submission_kernel_script_preserves_compressed_pickle_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.pkl.zst"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_pickle(submission_path)

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.pkl.zst"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".pkl.zst"' in script
    assert "'.pickle.zst'" in script


def test_render_submission_kernel_script_preserves_stata_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.dta"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_stata(submission_path, write_index=False)

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.dta"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".dta"' in script


def test_render_submission_kernel_script_preserves_xml_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.xml"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_xml(submission_path, index=False, parser="etree")

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.xml"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".xml"' in script


def test_render_submission_kernel_script_preserves_archive_output_names(tmp_path: Path) -> None:
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("predictions/a.txt", "ok\n")
    tar_path = tmp_path / "submission.tar.xz"
    payload = b"ok\n"
    with tarfile.open(tar_path, "w:xz") as archive:
        info = tarfile.TarInfo("predictions/a.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    tar_zst_path = tmp_path / "submission.tar.zst"
    with tar_zst_path.open("wb") as raw:
        with zstd.ZstdCompressor(level=9).stream_writer(raw) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                info = tarfile.TarInfo("predictions/a.txt")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    zip_script = render_submission_kernel_script(zip_path)
    tar_script = render_submission_kernel_script(tar_path)
    tar_zst_script = render_submission_kernel_script(tar_zst_path)

    assert 'SUBMISSION_OUTPUT_NAME = "submission.zip"' in zip_script
    assert 'SUBMISSION_INPUT_SUFFIX = ".zip"' in zip_script
    assert 'SUBMISSION_OUTPUT_NAME = "submission.tar.xz"' in tar_script
    assert 'SUBMISSION_INPUT_SUFFIX = ".tar.xz"' in tar_script
    assert 'SUBMISSION_OUTPUT_NAME = "submission.tar.zst"' in tar_zst_script
    assert 'SUBMISSION_INPUT_SUFFIX = ".tar.zst"' in tar_zst_script


@pytest.mark.parametrize(("name", "suffix"), [("submission.7z", ".7z"), ("submission.rar", ".rar")])
def test_render_submission_kernel_script_preserves_external_archive_output_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    suffix: str,
) -> None:
    submission_path = tmp_path / name
    if suffix == ".7z":
        _write_valid_7z(submission_path)
    else:
        submission_path.write_bytes(b"prebuilt rar archive")
        _FakeRarFile.members_by_name = {name: [_FakeRarMember("predictions/a.txt")]}
        monkeypatch.setattr("kagglebot.submission_service.rarfile.RarFile", _FakeRarFile)

    script = render_submission_kernel_script(submission_path)

    assert f'SUBMISSION_OUTPUT_NAME = "{name}"' in script
    assert f'SUBMISSION_INPUT_SUFFIX = "{suffix}"' in script
    assert f'"{suffix}"' in script


def test_render_submission_kernel_script_rejects_invalid_external_archive(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.7z"
    submission_path.write_bytes(b"not-a-7z")

    with pytest.raises(KernelFailedError, match="Refusing to embed invalid external archive submission"):
        render_submission_kernel_script(submission_path)


def test_rendered_submission_kernel_archives_directory_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    submission_path = tmp_path / "submission.zarr"
    (submission_path / "arrays").mkdir(parents=True)
    (submission_path / "empty_group").mkdir()
    (submission_path / ".zgroup").write_text("{}", encoding="utf-8")
    (submission_path / "arrays" / "0").write_bytes(b"chunk")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.zarr.zip"
    assert written.exists()
    with zipfile.ZipFile(written, "r") as archive:
        infos = archive.infolist()
        members = sorted(info.filename for info in infos if not info.is_dir())
        dirs = sorted(info.filename for info in infos if info.is_dir())
        assert archive.read("arrays/0") == b"chunk"
    assert members == [".zgroup", "arrays/0"]
    assert "empty_group/" in dirs


def test_render_submission_kernel_script_preserves_unknown_single_file_output_name(tmp_path: Path) -> None:
    submission_path = tmp_path / "predictions.bin"
    submission_path.write_bytes(b"\x00\x01predictions")

    script = render_submission_kernel_script(submission_path)

    assert 'SUBMISSION_OUTPUT_NAME = "predictions.bin"' in script
    assert 'SUBMISSION_INPUT_SUFFIX = ".bin"' in script
    assert "submission.csv" not in script


@pytest.mark.parametrize(
    ("name", "suffix"),
    [
        ("answers.nii.gz", ".nii.gz"),
        ("model.safetensors.index.json", ".safetensors.index.json"),
        ("predictions.npy", ".npy"),
        ("results.npz", ".npz"),
    ],
)
def test_render_submission_kernel_script_preserves_non_tabular_single_file_suffix(
    tmp_path: Path,
    name: str,
    suffix: str,
) -> None:
    submission_path = tmp_path / name
    submission_path.write_bytes(b"single-file-payload")

    script = render_submission_kernel_script(submission_path)

    assert f'SUBMISSION_OUTPUT_NAME = "{name}"' in script
    assert f'SUBMISSION_INPUT_SUFFIX = "{suffix}"' in script
    assert "NON_TABULAR_OUTPUT_SUFFIXES = " in script
    assert repr(suffix) in script
    assert "submission.csv" not in script


def test_render_submission_kernel_script_is_valid_python(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.tsv"
    submission_path.write_text("id\ttarget\n1\t0.1\n", encoding="utf-8")

    script = render_submission_kernel_script(submission_path)

    compile(script, "submission_kernel.py", "exec")


def test_render_submission_kernel_script_escapes_output_name_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    submission_path = tmp_path / 'submission "quoted".csv'
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"

    script = render_submission_kernel_script(submission_path)
    script_path.write_text(script, encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    compile(script, "submission_kernel.py", "exec")
    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / submission_path.name
    assert written.exists()
    assert pd.read_csv(written).to_dict("list") == {"id": [1], "target": [0.1]}


def test_render_submission_kernel_script_injects_shared_archive_suffixes(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.tsv"
    submission_path.write_text("id\ttarget\n1\t0.1\n", encoding="utf-8")

    script = render_submission_kernel_script(submission_path)

    assert "__ARCHIVE_OUTPUT_SUFFIXES__" not in script
    assert "__EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES__" not in script
    assert "__ZSTD_TAR_ARCHIVE_SUFFIXES__" not in script
    for suffix in ARCHIVE_SUBMISSION_SUFFIXES:
        assert repr(suffix) in script
    for suffix in ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES:
        assert repr(suffix) in script
    assert "def _archive_output_suffix(path: Path) -> str:" in script
    assert "archive_path.suffix.lower()" not in script


def test_render_submission_kernel_script_injects_duckdb_suffixes(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    script = render_submission_kernel_script(submission_path)

    assert "__DUCKDB_TABULAR_SUFFIXES__" not in script
    assert "'.duckdb'" in script
    assert "def _read_duckdb_table(path: Path):" in script


def test_render_submission_kernel_script_injects_rdata_suffixes(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    script = render_submission_kernel_script(submission_path)

    assert "__TABULAR_RDATA_SUFFIXES__" not in script
    assert "'.rds'" in script
    assert "'.rda'" in script
    assert "'.rdata'" in script
    assert "def _read_rdata_tabular_frame(path: Path):" in script


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("submission.zip", "zip"),
        ("submission.tar.xz", "tar_xz"),
        ("submission.tar.zst", "tar_zst"),
        ("submission.7z", "external"),
        ("submission.rar", "external"),
    ],
)
def test_rendered_submission_kernel_writes_archive_submission_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mode: str,
) -> None:
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    submission_path = tmp_path / name
    if mode == "zip":
        with zipfile.ZipFile(submission_path, "w") as archive:
            archive.writestr("predictions/a.txt", "ok\n")
    elif mode == "tar_xz":
        payload = b"ok\n"
        with tarfile.open(submission_path, "w:xz") as archive:
            info = tarfile.TarInfo("predictions/a.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    elif mode == "tar_zst":
        payload = b"ok\n"
        with submission_path.open("wb") as raw:
            with zstd.ZstdCompressor(level=9).stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    info = tarfile.TarInfo("predictions/a.txt")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
    else:
        if name.endswith(".7z"):
            _write_valid_7z(submission_path)
        else:
            submission_path.write_bytes(b"prebuilt external rar archive")
            _FakeRarFile.members_by_name = {name: [_FakeRarMember("predictions/a.txt")]}
            monkeypatch.setattr("kagglebot.submission_service.rarfile.RarFile", _FakeRarFile)
    original_bytes = submission_path.read_bytes()
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / name
    assert written.exists()
    assert written.read_bytes() == original_bytes


@pytest.mark.parametrize("mode", ["zip", "tar_xz"])
def test_rendered_submission_kernel_rejects_duplicate_archive_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    submission_path = tmp_path / ("submission.zip" if mode == "zip" else "submission.tar.xz")
    if mode == "zip":
        with zipfile.ZipFile(submission_path, "w") as archive:
            archive.writestr("predictions/a.txt", "first\n")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("predictions/a.txt", "second\n")
    else:
        with tarfile.open(submission_path, "w:xz") as archive:
            for payload in (b"first\n", b"second\n"):
                info = tarfile.TarInfo("predictions/a.txt")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    with pytest.raises(RuntimeError, match="duplicate archive member"):
        runpy.run_path(str(script_path), run_name="__main__")


def test_rendered_submission_kernel_writes_unknown_single_file_submission_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    submission_path = tmp_path / "predictions.bin"
    submission_path.write_bytes(b"\x00\x01predictions")
    original_bytes = submission_path.read_bytes()
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "predictions.bin"
    assert written.exists()
    assert written.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "name",
    [
        "answers.nii.gz",
        "predictions.npy",
        "submission.fasta.gz",
        "submission.graphml.bz2",
        "submission.ply.zst",
        "submission.xgb",
        "submission.cbm",
        "submission.npz",
    ],
)
def test_rendered_submission_kernel_writes_non_tabular_single_file_submission_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    submission_path = tmp_path / name
    submission_path.write_bytes(b"single-file-payload")
    original_bytes = submission_path.read_bytes()
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / name
    assert written.exists()
    assert written.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("suffix", "sample_payload", "test_payload", "submission_payload"),
    [
        (
            ".csv",
            "id;target\n1;0.0\n2;0.0\n",
            "id;feature\n1;10\n2;20\n",
            "id;target\n1;0.1\n2;0.2\n",
        ),
        (
            ".tsv",
            "id\ttarget\n1\t0.0\n2\t0.0\n",
            "id\tfeature\n1\t10\n2\t20\n",
            "id\ttarget\n1\t0.1\n2\t0.2\n",
        ),
        (
            ".jsonl",
            '{"id":1,"target":0.0}\n{"id":2,"target":0.0}\n',
            '{"id":1,"feature":10}\n{"id":2,"feature":20}\n',
            '{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n',
        ),
        (
            ".jsonlines",
            '{"id":1,"target":0.0}\n{"id":2,"target":0.0}\n',
            '{"id":1,"feature":10}\n{"id":2,"feature":20}\n',
            '{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n',
        ),
        (
            ".ndjson",
            '{"id":1,"target":0.0}\n{"id":2,"target":0.0}\n',
            '{"id":1,"feature":10}\n{"id":2,"feature":20}\n',
            '{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n',
        ),
    ],
)
def test_rendered_submission_kernel_writes_and_validates_non_csv_tabular_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    sample_payload: str,
    test_payload: str,
    submission_payload: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / f"sample_submission{suffix}").write_text(sample_payload, encoding="utf-8")
    (data_dir / f"test{suffix}").write_text(test_payload, encoding="utf-8")
    submission_path = tmp_path / f"submission{suffix}"
    submission_path.write_text(submission_payload, encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / f"submission{suffix}"
    assert written.exists()
    if suffix in {".jsonl", ".jsonlines", ".ndjson"}:
        output = pd.read_json(written, lines=True)
    elif suffix == ".tsv":
        output = pd.read_csv(written, sep="\t")
    else:
        output = pd.read_csv(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_reads_wrapped_json_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.json").write_text(
        '{"records":[{"id":1,"target":0.0},{"id":2,"target":0.0}]}',
        encoding="utf-8",
    )
    (data_dir / "test.json").write_text(
        '{"data":[{"id":1,"feature":10},{"id":2,"feature":20}]}',
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        '{"rows":[{"id":1,"target":0.1},{"id":2,"target":0.2}]}',
        encoding="utf-8",
    )
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.json"
    assert written.exists()
    output = pd.read_json(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_uses_validation_features_as_test_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    (data_dir / "validation_features.csv").write_text("id,feature\n3,30\n2,20\n1,10\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n3,0.3\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [3, 2, 1]
    assert output["target"].tolist() == [0.3, 0.2, 0.1]


def test_rendered_submission_kernel_aligns_file_path_id_alias_without_test_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text(
        "file_path,target\ntest/a.png,0.0\ntest/b.png,0.0\ntest/c.png,0.0\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text(
        "file_path,target\ntest/c.png,0.3\ntest/a.png,0.1\ntest/b.png,0.2\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["file_path"].tolist() == ["test/a.png", "test/b.png", "test/c.png"]
    assert output["target"].tolist() == [0.1, 0.2, 0.3]


def test_rendered_submission_kernel_writes_pickle_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_pickle(data_dir / "sample_submission.pkl")
    pd.DataFrame({"id": [1, 2], "feature": [10, 20]}).to_pickle(data_dir / "test.pkl")
    submission_path = tmp_path / "submission.pkl"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_pickle(submission_path)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.pkl"
    assert written.exists()
    output = pd.read_pickle(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_writes_avro_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}), data_dir / "sample_submission.avro")
    write_table(pd.DataFrame({"id": [1, 2], "feature": [10, 20]}), data_dir / "test.avro")
    submission_path = tmp_path / "submission.avro"
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}), submission_path)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.avro"
    assert written.exists()
    output = read_table(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_writes_compressed_pickle_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_pickle(data_dir / "sample_submission.pkl.zst")
    pd.DataFrame({"id": [1, 2], "feature": [10, 20]}).to_pickle(data_dir / "test.pkl.zst")
    submission_path = tmp_path / "submission.pkl.zst"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_pickle(submission_path)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.pkl.zst"
    assert written.exists()
    output = pd.read_pickle(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_writes_stata_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_stata(
        data_dir / "sample_submission.dta",
        write_index=False,
    )
    pd.DataFrame({"id": [1, 2], "feature": [10, 20]}).to_stata(data_dir / "test.dta", write_index=False)
    submission_path = tmp_path / "submission.dta"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_stata(submission_path, write_index=False)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.dta"
    assert written.exists()
    output = pd.read_stata(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_aligns_against_stata_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_stata(
        data_dir / "sample_submission.dta",
        write_index=False,
    )
    pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}).to_stata(
        data_dir / "test.dta",
        write_index=False,
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_avro_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}), data_dir / "sample_submission.avro")
    write_table(pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}), data_dir / "test.avro")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".xml", ".xml.zst"])
def test_rendered_submission_kernel_writes_xml_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()

    def write_xml_frame(path: Path, frame: pd.DataFrame) -> None:
        payload = frame.to_xml(index=False, parser="etree").encode("utf-8")
        if path.name.endswith(".zst"):
            path.write_bytes(zstd.ZstdCompressor().compress(payload))
        else:
            path.write_bytes(payload)

    write_xml_frame(data_dir / f"sample_submission{suffix}", pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}))
    write_xml_frame(data_dir / f"test{suffix}", pd.DataFrame({"id": [1, 2], "feature": [10, 20]}))
    submission_path = tmp_path / f"submission{suffix}"
    write_xml_frame(submission_path, pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}))
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / f"submission{suffix}"
    assert written.exists()
    if suffix.endswith(".zst"):
        output = pd.read_xml(io.BytesIO(zstd.ZstdDecompressor().decompress(written.read_bytes())), parser="etree")
    else:
        output = pd.read_xml(written, parser="etree")
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_writes_orc_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_orc(
        data_dir / "sample_submission.orc",
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "feature": [10, 20]}).to_orc(data_dir / "test.orc", index=False)
    submission_path = tmp_path / "submission.orc"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_orc(submission_path, index=False)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.orc"
    assert written.exists()
    output = pd.read_orc(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_writes_hdf5_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_hdf(
        data_dir / "sample_submission.hdf5",
        key="sample_submission",
        mode="w",
        format="table",
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "feature": [10, 20]}).to_hdf(
        data_dir / "test.hdf5",
        key="test",
        mode="w",
        format="table",
        index=False,
    )
    submission_path = tmp_path / "submission.hdf5"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_hdf(
        submission_path,
        key="submission",
        mode="w",
        format="table",
        index=False,
    )
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.hdf5"
    assert written.exists()
    output = pd.read_hdf(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_aligns_against_native_hdf5_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with h5py.File(data_dir / "sample_submission.hdf5", "w") as handle:
        group = handle.create_group("submission")
        group.create_dataset("id", data=[1, 2])
        group.create_dataset("target", data=[0.0, 0.0])
    with h5py.File(data_dir / "test.hdf5", "w") as handle:
        group = handle.create_group("test")
        group.create_dataset("id", data=[1, 2, 3])
        group.create_dataset("feature", data=[10.0, 20.0, 30.0])
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_numpy_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    np.savez(data_dir / "test.npz", id=np.array([1, 2, 3]), feature=np.array([10.0, 20.0, 30.0]))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".nc", ".netcdf", ".cdf"])
def test_rendered_submission_kernel_aligns_against_netcdf_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    _write_netcdf_table(
        data_dir / f"test{suffix}",
        {
            "id": np.array([1, 2, 3], dtype=np.int32),
            "feature": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        },
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".fits", ".fit", ".fts", ".fits.gz"])
def test_rendered_submission_kernel_aligns_against_fits_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    _write_fits_table(
        data_dir / f"test{suffix}",
        {
            "id": np.array([1, 2, 3], dtype=np.int64),
            "feature": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        },
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_hdf5_backed_nc4_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    with h5py.File(data_dir / "test.nc4", "w") as handle:
        handle.create_dataset("id", data=np.array([1, 2, 3], dtype=np.int32))
        handle.create_dataset("feature", data=np.array([10.0, 20.0, 30.0], dtype=np.float64))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_h5ad_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    _write_h5ad_table(
        data_dir / "test.h5ad",
        ids=np.array([1, 2, 3]),
        features=np.array([[10.0, 0.1], [20.0, 0.2], [30.0, 0.3]]),
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_geopackage_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    _write_geopackage_table(
        data_dir / "test.gpkg",
        [
            (1, 10.0, None, b"\x47\x50\x00\x01"),
            (2, 20.0, None, b"\x47\x50\x00\x02"),
            (3, 30.0, None, b"\x47\x50\x00\x03"),
        ],
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_shapefile_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    (data_dir / "test.shp").write_bytes(b"")
    _write_dbf_table(
        data_dir / "test.dbf",
        [
            (1, 10.0, None, "north"),
            (2, 20.0, None, "south"),
            (3, 30.0, None, "east"),
        ],
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_kmz_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    _write_kmz(data_dir / "test.kmz", _kml_payload([(1, 10.0, None), (2, 20.0, None), (3, 30.0, None)]))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_compressed_kml_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    with gzip.open(data_dir / "test.kml.gz", "wt", encoding="utf-8") as handle:
        handle.write(_kml_payload([(1, 10.0, None), (2, 20.0, None), (3, 30.0, None)]))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_numpy_matrix_sidecar_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    np.save(data_dir / "test.npy", np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    (data_dir / "test_columns.txt").write_text("id\nfeature\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1.0, 2.0, 3.0]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_writes_compressed_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with gzip.open(data_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.0\n2,0.0\n")
    with gzip.open(data_dir / "test.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n1,10\n2,20\n")
    submission_path = tmp_path / "submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.1\n2,0.2\n")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.csv.gz"
    assert written.exists()
    output = pd.read_csv(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_stabilizes_problematic_tabular_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    working_dir = tmp_path / "working"
    input_root.mkdir()
    working_dir.mkdir()
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    namespace = runpy.run_path(str(script_path), run_name="__main__")

    raw = pd.DataFrame([[1, "x", 0.1, 0.2]], columns=["id", "", "score", "score"])
    payload = namespace["_frame_to_submission_bytes"](raw)
    output = pd.read_csv(io.BytesIO(payload))
    assert list(output.columns) == ["id", "column_2", "score", "score_1"]
    assert list(raw.columns) == ["id", "", "score", "score"]

    embedded = namespace["_read_embedded_submission"](b"id,,target\n1,x,0.1\n")
    assert list(embedded.columns) == ["id", "column_2", "target"]


def test_rendered_submission_kernel_writes_txt_submission_with_tab_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    working_dir = tmp_path / "working"
    input_root.mkdir()
    working_dir.mkdir()
    submission_path = tmp_path / "submission.txt"
    submission_path.write_text("id\ttarget\n1\t0.1\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    namespace = runpy.run_path(str(script_path), run_name="__main__")
    payload = namespace["_frame_to_submission_bytes"](pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}))

    assert payload.decode("utf-8").splitlines()[0] == "id\ttarget"
    output = pd.read_csv(io.BytesIO(payload), sep="\t")
    assert output.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_rendered_submission_kernel_writes_zstd_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    compressor = zstd.ZstdCompressor()
    (data_dir / "sample_submission.csv.zst").write_bytes(compressor.compress(b"id,target\n1,0.0\n2,0.0\n"))
    (data_dir / "test.csv.zst").write_bytes(compressor.compress(b"id,feature\n1,10\n2,20\n"))
    submission_path = tmp_path / "submission.csv.zst"
    submission_path.write_bytes(compressor.compress(b"id,target\n1,0.1\n2,0.2\n"))
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.csv.zst"
    assert written.exists()
    output = pd.read_csv(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_extracts_archived_sample_for_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        for name, payload in {
            "sample_submission.csv": b"id,target\n1,0\n2,0\n",
            "test.csv": b"id,feature\n1,10\n2,20\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    (data_dir / "competition.tar.zst").write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.9\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.csv"
    output = pd.read_csv(written)
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.9, 0.9]
    assert (working_dir / "extracted_input" / "sample_submission.csv").exists()
    assert (working_dir / "extracted_input" / "test.csv").exists()


def test_rendered_submission_kernel_rejects_duplicate_archive_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with zipfile.ZipFile(data_dir / "competition.zip", "w") as archive:
        archive.writestr("sample_submission.csv", "id,target\n1,0\n")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("sample_submission.csv", "id,target\n2,0\n")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.9\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    with pytest.raises(ValueError, match="duplicate archive member target"):
        runpy.run_path(str(script_path), run_name="__main__")


def test_rendered_submission_kernel_extracts_7z_sample_for_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    source_dir = tmp_path / "source"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    source_dir.mkdir()
    (source_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n", encoding="utf-8")
    (source_dir / "test.csv").write_text("id,feature\n1,10\n2,20\n", encoding="utf-8")
    with py7zr.SevenZipFile(data_dir / "competition.7z", "w") as archive:
        archive.write(source_dir / "sample_submission.csv", "sample_submission.csv")
        archive.write(source_dir / "test.csv", "test.csv")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.9\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / "submission.csv"
    output = pd.read_csv(written)
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.9, 0.9]
    assert (working_dir / "extracted_input" / "sample_submission.csv").exists()
    assert (working_dir / "extracted_input" / "test.csv").exists()


@pytest.mark.parametrize("suffix", [".jsonl.bz2", ".jsonlines.xz", ".ndjson.zst"])
def test_rendered_submission_kernel_writes_compressed_json_lines_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}), data_dir / f"sample_submission{suffix}")
    write_table(pd.DataFrame({"id": [1, 2], "feature": [10, 20]}), data_dir / f"test{suffix}")
    submission_path = tmp_path / f"submission{suffix}"
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}), submission_path)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / f"submission{suffix}"
    assert written.exists()
    output = read_table(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


@pytest.mark.parametrize("suffix", [".yaml.xz", ".xml.bz2", ".html.bz2", ".psv.xz", ".tab.zst"])
def test_rendered_submission_kernel_writes_compressed_structured_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    if suffix.startswith(".yaml"):
        pytest.importorskip("yaml")
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}), data_dir / f"sample_submission{suffix}")
    write_table(pd.DataFrame({"id": [1, 2], "feature": [10, 20]}), data_dir / f"test{suffix}")
    submission_path = tmp_path / f"submission{suffix}"
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}), submission_path)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / f"submission{suffix}"
    assert written.exists()
    output = read_table(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_rendered_submission_kernel_reads_excel_sample_and_compressed_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_excel(
        data_dir / "sample_submission.xlsx",
        index=False,
    )
    with gzip.open(data_dir / "test.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n1,10\n2,20\n")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm", ".ods"])
def test_rendered_submission_kernel_aligns_against_spreadsheet_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_excel(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}).to_excel(
        data_dir / f"test{suffix}",
        index=False,
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".parquet", ".parq", ".pq"])
def test_rendered_submission_kernel_aligns_against_parquet_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_parquet(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}).to_parquet(
        data_dir / f"test{suffix}",
        index=False,
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_zip_wrapped_csv_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with zipfile.ZipFile(data_dir / "sample_submission.csv.zip", "w") as archive:
        archive.writestr("sample_submission.csv", "id,target\n1,0.0\n2,0.0\n")
    with zipfile.ZipFile(data_dir / "test.csv.zip", "w") as archive:
        archive.writestr("test.csv", "id,feature\n1,10.0\n2,20.0\n3,30.0\n")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_zip_wrapped_parquet_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    sample_payload = io.BytesIO()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_parquet(sample_payload, index=False)
    with zipfile.ZipFile(data_dir / "sample_submission.parquet.zip", "w") as archive:
        archive.writestr("nested/sample_submission.parquet", sample_payload.getvalue())
    test_payload = io.BytesIO()
    pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}).to_parquet(test_payload, index=False)
    with zipfile.ZipFile(data_dir / "test.parquet.zip", "w") as archive:
        archive.writestr("nested/test.parquet", test_payload.getvalue())
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".ftr", ".arrow", ".ipc"])
def test_rendered_submission_kernel_aligns_against_arrow_ipc_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_feather(
        data_dir / f"sample_submission{suffix}",
    )
    pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}).to_feather(
        data_dir / f"test{suffix}",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".pkl.gz", ".pkl.zst"])
def test_rendered_submission_kernel_aligns_against_pickle_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_pickle(
        data_dir / f"sample_submission{suffix}",
    )
    pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}).to_pickle(
        data_dir / f"test{suffix}",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".yaml", ".yml.zst"])
def test_rendered_submission_kernel_aligns_against_yaml_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    yaml = pytest.importorskip("yaml")
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()

    def write_yaml(path: Path, frame: pd.DataFrame) -> None:
        payload = yaml.safe_dump(frame.to_dict(orient="records"), sort_keys=False).encode("utf-8")
        if path.name.endswith(".zst"):
            path.write_bytes(zstd.ZstdCompressor().compress(payload))
        else:
            path.write_bytes(payload)

    write_yaml(data_dir / f"sample_submission{suffix}", pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}))
    write_yaml(data_dir / f"test{suffix}", pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_to_eval_features_when_test_file_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    (data_dir / "eval_features.csv").write_text("id,feature\n1,10\n2,20\n3,30\n4,40\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3, 4]
    assert output["target"].tolist() == [0.1, 0.2, 0.15, 0.15]


def test_rendered_submission_kernel_aligns_to_geojson_eval_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    (data_dir / "eval.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": idx, "feature": float(idx)},
                        "geometry": {"type": "Point", "coordinates": [float(idx), float(idx + 1)]},
                    }
                    for idx in (1, 2, 3, 4)
                ],
            }
        ),
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3, 4]
    assert output["target"].tolist() == [0.1, 0.2, 0.15, 0.15]


def test_rendered_submission_kernel_aligns_to_compressed_geojson_eval_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    eval_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": idx, "feature": float(idx)},
                "geometry": {"type": "Point", "coordinates": [float(idx), float(idx + 1)]},
            }
            for idx in (1, 2, 3, 4)
        ],
    }
    (data_dir / "eval.geojson.zst").write_bytes(
        zstd.ZstdCompressor().compress(json.dumps(eval_payload).encode("utf-8"))
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3, 4]
    assert output["target"].tolist() == [0.1, 0.2, 0.15, 0.15]


def test_rendered_submission_kernel_finds_camel_case_public_test_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "SampleSubmission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    (data_dir / "PublicTest.csv").write_text("id,feature\n1,10\n2,20\n3,30\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_sqlite_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    with sqlite3.connect(data_dir / "sample_submission.sqlite") as conn:
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target REAL)")
        conn.executemany("INSERT INTO sample_submission VALUES (?, ?)", [(1, 0.0), (2, 0.0)])
    with sqlite3.connect(data_dir / "test.sqlite") as conn:
        conn.execute("CREATE TABLE test (id INTEGER, feature REAL)")
        conn.executemany("INSERT INTO test VALUES (?, ?)", [(1, 10.0), (2, 20.0), (3, 30.0)])
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_duckdb_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    conn = duckdb.connect(str(data_dir / "sample_submission.duckdb"))
    try:
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target DOUBLE)")
        conn.execute("INSERT INTO sample_submission VALUES (1, 0.0), (2, 0.0)")
    finally:
        conn.close()
    conn = duckdb.connect(str(data_dir / "test.duckdb"))
    try:
        conn.execute("CREATE TABLE test (id INTEGER, feature DOUBLE)")
        conn.execute("INSERT INTO test VALUES (1, 10.0), (2, 20.0), (3, 30.0)")
    finally:
        conn.close()
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".rds", ".rda", ".rdata"])
def test_rendered_submission_kernel_aligns_against_rdata_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    _write_rdata_table(data_dir / f"sample_submission{suffix}", pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}))
    _write_rdata_table(data_dir / f"test{suffix}", pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_matlab_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    savemat(data_dir / "sample_submission.mat", {"id": [1, 2], "target": [0.0, 0.0]})
    savemat(data_dir / "test.mat", {"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]})
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".sav", ".zsav"])
def test_rendered_submission_kernel_aligns_against_spss_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pyreadstat.write_sav(
        pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}),
        data_dir / f"sample_submission{suffix}",
    )
    pyreadstat.write_sav(
        pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}),
        data_dir / f"test{suffix}",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".xpt", ".xport"])
def test_rendered_submission_kernel_aligns_against_sas_xport_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pyreadstat.write_xport(
        pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}),
        data_dir / f"sample_submission{suffix}",
        file_format_version=5,
    )
    pyreadstat.write_xport(
        pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}),
        data_dir / f"test{suffix}",
        file_format_version=5,
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".html", ".htm"])
def test_rendered_submission_kernel_aligns_against_html_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_html(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [1, 2, 3], "feature": [10.0, 20.0, 30.0]}).to_html(
        data_dir / f"test{suffix}",
        index=False,
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".fwf", ".fixed", ".fixedwidth"])
def test_rendered_submission_kernel_aligns_against_fixed_width_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / f"sample_submission{suffix}").write_text("id target\n1  0.0\n2  0.0\n", encoding="utf-8")
    (data_dir / f"test{suffix}").write_text("id feature\n1  10.0\n2  20.0\n3  30.0\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".svm", ".svmlight", ".libsvm"])
def test_rendered_submission_kernel_aligns_against_svmlight_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("target\n0.0\n0.0\n", encoding="utf-8")
    (data_dir / f"test{suffix}").write_text(
        "0 1:0.05 2:1.05\n0 1:1.45 2:0.45\n0 1:0.75 2:0.80\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("target\n0.1\n0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert list(output.columns) == ["target"]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


def test_rendered_submission_kernel_aligns_against_compressed_arff_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    payloads = {
        "sample_submission.arff.gz": """
@RELATION sample_submission
@ATTRIBUTE id NUMERIC
@ATTRIBUTE target NUMERIC
@DATA
1,0.0
2,0.0
""",
        "test.arff.gz": """
@RELATION test
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@DATA
1,10.0
2,20.0
3,30.0
""",
    }
    for name, payload in payloads.items():
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            handle.write(payload.strip())
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    output = pd.read_csv(working_dir / "submission.csv")
    assert output["id"].tolist() == [1, 2, 3]
    assert output["target"].tolist() == [0.1, 0.2, 0.15]


@pytest.mark.parametrize("suffix", [".html", ".htm"])
def test_rendered_submission_kernel_writes_html_tabular_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_html(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "feature": [10.0, 20.0]}).to_html(data_dir / f"test{suffix}", index=False)
    submission_path = tmp_path / f"submission{suffix}"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_html(submission_path, index=False)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / f"submission{suffix}"
    assert written.exists()
    output = pd.read_html(written)[0]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm"])
def test_rendered_submission_kernel_writes_excel_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_excel(
        data_dir / f"sample_submission{suffix}",
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "feature": [10, 20]}).to_excel(data_dir / f"test{suffix}", index=False)
    submission_path = tmp_path / f"submission{suffix}"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_excel(submission_path, index=False)
    script_path = tmp_path / "submission_kernel.py"
    script_path.write_text(render_submission_kernel_script(submission_path), encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_SLUG", "demo")
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    runpy.run_path(str(script_path), run_name="__main__")

    written = working_dir / f"submission{suffix}"
    assert written.exists()
    output = pd.read_excel(written)
    assert list(output.columns) == ["id", "target"]
    assert output["id"].tolist() == [1, 2]
    assert output["target"].tolist() == [0.1, 0.2]


def test_count_tabular_data_rows_at_most_supports_compressed_and_excel(tmp_path: Path) -> None:
    csv_gz = tmp_path / "submission.csv.gz"
    with gzip.open(csv_gz, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.1\n2,0.2\n")
    xlsx = tmp_path / "submission.xlsx"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_excel(xlsx, index=False)

    assert count_tabular_data_rows_at_most(csv_gz, limit=2) is True
    assert count_tabular_data_rows_at_most(csv_gz, limit=1) is False
    assert count_tabular_data_rows_at_most(xlsx, limit=2) is True
    assert count_tabular_data_rows_at_most(xlsx, limit=1) is False


@pytest.mark.parametrize("suffix", [".jsonl.bz2", ".jsonlines.xz", ".ndjson.zst"])
def test_count_tabular_data_rows_at_most_supports_compressed_json_lines_aliases(
    tmp_path: Path,
    suffix: str,
) -> None:
    submission_path = tmp_path / f"submission{suffix}"
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}), submission_path)

    assert count_tabular_data_rows_at_most(submission_path, limit=2) is True
    assert count_tabular_data_rows_at_most(submission_path, limit=1) is False


@pytest.mark.parametrize("suffix", [".yaml.xz", ".xml.bz2", ".html.bz2", ".psv.xz", ".tab.zst"])
def test_count_tabular_data_rows_at_most_supports_compressed_structured_tabular_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    if suffix.startswith(".yaml"):
        pytest.importorskip("yaml")
    submission_path = tmp_path / f"submission{suffix}"
    write_table(pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}), submission_path)

    assert count_tabular_data_rows_at_most(submission_path, limit=2) is True
    assert count_tabular_data_rows_at_most(submission_path, limit=1) is False


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5", ".pkl.zst"])
def test_count_tabular_data_rows_at_most_supports_binary_tabular_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    submission_path = tmp_path / f"submission{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})
    if suffix == ".orc":
        frame.to_orc(submission_path, index=False)
    elif suffix in {".hdf", ".hdf5"}:
        frame.to_hdf(submission_path, key="submission", mode="w", format="table", index=False)
    else:
        frame.to_pickle(submission_path)

    assert count_tabular_data_rows_at_most(submission_path, limit=2) is True
    assert count_tabular_data_rows_at_most(submission_path, limit=1) is False


def test_count_tabular_data_rows_at_most_supports_zip_wrapped_parquet(tmp_path: Path) -> None:
    payload = io.BytesIO()
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_parquet(payload, index=False)
    submission_path = tmp_path / "submission.parquet.zip"
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr("nested/submission.parquet", payload.getvalue())

    assert count_tabular_data_rows_at_most(submission_path, limit=3) is True
    assert count_tabular_data_rows_at_most(submission_path, limit=2) is False
