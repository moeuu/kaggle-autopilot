from __future__ import annotations

import gzip
import io
import json
import sqlite3
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import pyreadr
import pyreadstat
import pytest
import zstandard as zstd
from scipy.io import savemat

from kagglebot.submission_sample_discovery import (
    DUCKDB_TABULAR_SUFFIXES,
    TABULAR_ARFF_SUFFIXES,
    TABULAR_ARROW_IPC_SUFFIXES,
    TABULAR_EXCEL_INPUT_ONLY_SUFFIXES,
    TABULAR_EXCEL_SUFFIXES,
    TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES,
    TABULAR_GEOJSON_SUFFIXES,
    TABULAR_HDF_SUFFIXES,
    TABULAR_HTML_SUFFIX_PREFIXES,
    TABULAR_INPUT_SUFFIXES,
    TABULAR_MATLAB_SUFFIXES,
    TABULAR_PARQUET_SUFFIXES,
    TABULAR_RDATA_SUFFIXES,
    TABULAR_SAS_SUFFIXES,
    TABULAR_SPSS_SUFFIXES,
    TABULAR_STATA_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES_LENGTH_ORDERED,
    TABULAR_SVMLIGHT_SUFFIX_PREFIXES,
    _finalize_tabular_frame,
    _is_json_table_suffix,
    _is_yaml_table_suffix,
    _read_structured_tabular_frame,
    _read_structured_tabular_sample,
    default_delimited_text_separator,
    find_usable_sample_submissions,
    is_json_lines_tabular_suffix,
    is_psv_like_tabular_suffix,
    is_tab_delimited_tabular_suffix,
    is_tabular_data_path,
    is_tsv_like_tabular_suffix,
    is_txt_like_tabular_suffix,
    open_tabular_text,
    path_mentions_role,
    preferred_rowless_tabular_sample_suffix,
    preferred_tabular_submission_suffix,
    sample_name_score,
    select_sample_submission_path,
    tabular_data_row_count_capped,
    tabular_file_has_data_rows,
    tabular_file_has_two_or_more_columns,
    tabular_suffix,
)


def test_open_tabular_text_reads_zstd_compressed_text(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.csv.zst"
    path.write_bytes(zstd.ZstdCompressor().compress(b"id,target\n1,0.1\n"))

    with open_tabular_text(path) as handle:
        assert handle.read() == "id,target\n1,0.1\n"


def test_open_tabular_text_reads_single_member_zip_tabular_text(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.csv.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/sample_submission.csv", "id,target\n1,0.1\n")

    with open_tabular_text(path) as handle:
        assert handle.read() == "id,target\n1,0.1\n"

    assert tabular_suffix(path) == ".csv.zip"
    assert ".csv.zip" in TABULAR_INPUT_SUFFIXES
    assert ".csv.zip" not in TABULAR_SUBMISSION_SUFFIXES


def test_tabular_suffixes_include_zip_wrapped_binary_tables() -> None:
    assert {".parquet.zip", ".feather.zip", ".avro.zip", ".orc.zip"} <= TABULAR_INPUT_SUFFIXES
    assert not ({".parquet.zip", ".feather.zip", ".avro.zip", ".orc.zip"} & TABULAR_SUBMISSION_SUFFIXES)
    assert tabular_suffix(Path("train.parquet.zip")) == ".parquet.zip"
    assert tabular_suffix(Path("sample_submission.feather.zip")) == ".feather.zip"


def test_tabular_submission_suffixes_length_ordered_prefers_compound_suffixes() -> None:
    suffixes = TABULAR_SUBMISSION_SUFFIXES_LENGTH_ORDERED

    assert suffixes == tuple(sorted(TABULAR_SUBMISSION_SUFFIXES, key=len, reverse=True))
    assert suffixes.index(".jsonl.zst") < suffixes.index(".jsonl")
    assert suffixes.index(".csv.gz") < suffixes.index(".csv")


def test_tabular_data_row_count_capped_counts_zip_wrapped_parquet_rows(tmp_path: Path) -> None:
    payload = io.BytesIO()
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_parquet(payload, index=False)
    path = tmp_path / "sample_submission.parquet.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/sample_submission.parquet", payload.getvalue())

    assert tabular_data_row_count_capped(path, cap=10) == 3
    assert tabular_data_row_count_capped(path, cap=2) == 3


def test_tabular_suffixes_include_tab_and_psv_text_variants() -> None:
    assert {".tab", ".psv", ".tab.gz", ".psv.zst", ".tab.zip", ".psv.zip"} <= TABULAR_INPUT_SUFFIXES
    assert {".tab", ".psv", ".tab.gz", ".psv.zst"} <= TABULAR_SUBMISSION_SUFFIXES
    assert ".tab.zip" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".psv.zip" not in TABULAR_SUBMISSION_SUFFIXES
    assert is_tsv_like_tabular_suffix(".tab")
    assert is_tsv_like_tabular_suffix(".tab.gz")
    assert is_tab_delimited_tabular_suffix(".tab")
    assert is_tab_delimited_tabular_suffix(".tab.gz")
    assert not is_tab_delimited_tabular_suffix(".txt")
    assert is_txt_like_tabular_suffix(".txt")
    assert is_txt_like_tabular_suffix(".txt.zst")
    assert is_psv_like_tabular_suffix(".psv")
    assert is_psv_like_tabular_suffix(".psv.zip")
    assert default_delimited_text_separator(".tab.gz") == "\t"
    assert default_delimited_text_separator(".txt.zst") == "\t"
    assert default_delimited_text_separator(".psv.zip") == "|"
    assert default_delimited_text_separator(".csv") == ","
    assert tabular_suffix(Path("train.tab.zip")) == ".tab.zip"
    assert tabular_suffix(Path("test.psv.zst")) == ".psv.zst"


def test_tabular_suffixes_include_duckdb_inputs_only() -> None:
    assert {".duckdb", ".ddb"} <= TABULAR_INPUT_SUFFIXES
    assert {".duckdb", ".ddb"} == set(DUCKDB_TABULAR_SUFFIXES)
    assert ".duckdb" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".ddb" not in TABULAR_SUBMISSION_SUFFIXES
    assert preferred_tabular_submission_suffix([".duckdb"]) == ".csv"
    assert tabular_suffix(Path("train.duckdb")) == ".duckdb"


def test_tabular_suffixes_include_rdata_inputs_only() -> None:
    assert {".rds", ".rda", ".rdata"} <= TABULAR_INPUT_SUFFIXES
    assert {".rds", ".rda", ".rdata"} == set(TABULAR_RDATA_SUFFIXES)
    assert ".rds" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".rdata" not in TABULAR_SUBMISSION_SUFFIXES
    assert preferred_tabular_submission_suffix([".rds"]) == ".csv"
    assert tabular_suffix(Path("train.rdata")) == ".rdata"


def test_tabular_suffixes_include_orc() -> None:
    assert ".orc" in TABULAR_INPUT_SUFFIXES
    assert ".orc" in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("sample_submission.orc")) == ".orc"


def test_tabular_suffixes_include_avro() -> None:
    assert ".avro" in TABULAR_INPUT_SUFFIXES
    assert ".avro" in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("sample_submission.avro")) == ".avro"


def test_tabular_suffixes_include_compressed_xml_submissions() -> None:
    assert ".xml.zst" in TABULAR_INPUT_SUFFIXES
    assert ".xml.zst" in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("sample_submission.xml.zst")) == ".xml.zst"


def test_tabular_suffixes_include_compressed_core_families() -> None:
    for base_suffix in (".csv", ".tsv", ".psv", ".txt", ".json", ".jsonl", ".pkl", ".pickle", ".html"):
        for compression in (".gz", ".bz2", ".xz", ".zst"):
            suffix = f"{base_suffix}{compression}"
            assert suffix in TABULAR_INPUT_SUFFIXES
            assert tabular_suffix(Path(f"sample_submission{suffix}")) == suffix


def test_tabular_suffixes_include_compressed_input_only_families() -> None:
    for base_suffix in (".geojson", ".dat", ".fwf", ".fixed", ".fixedwidth", ".svm", ".svmlight", ".libsvm"):
        for compression in (".gz", ".bz2", ".xz", ".zst"):
            suffix = f"{base_suffix}{compression}"
            assert suffix in TABULAR_INPUT_SUFFIXES
            assert suffix not in TABULAR_SUBMISSION_SUFFIXES
            assert tabular_suffix(Path(f"train{suffix}")) == suffix

    for suffix in (".fits.gz", ".fit.xz", ".fts.zst", ".arff.bz2"):
        assert suffix in TABULAR_INPUT_SUFFIXES
        assert suffix not in TABULAR_SUBMISSION_SUFFIXES
        assert tabular_suffix(Path(f"train{suffix}")) == suffix


def test_tabular_suffixes_include_geojson() -> None:
    assert ".geojson" in TABULAR_INPUT_SUFFIXES
    assert ".geojson.zst" in TABULAR_INPUT_SUFFIXES
    assert {".geojson", ".geojson.gz", ".geojson.bz2", ".geojson.xz", ".geojson.zst"} == TABULAR_GEOJSON_SUFFIXES
    assert ".geojson" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".geojson.zst" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("features.geojson")) == ".geojson"
    assert tabular_suffix(Path("sample_submission.geojson.zst")) == ".geojson.zst"


def test_structured_object_suffix_checks_follow_compressed_tabular_suffixes() -> None:
    for suffix in (".json", ".geojson"):
        assert _is_json_table_suffix(suffix)
        for compression in (".gz", ".bz2", ".xz", ".zst"):
            assert _is_json_table_suffix(f"{suffix}{compression}")
            assert f"{suffix}{compression}" in TABULAR_INPUT_SUFFIXES

    for suffix in (".yaml", ".yml"):
        assert _is_yaml_table_suffix(suffix)
        for compression in (".gz", ".bz2", ".xz", ".zst"):
            assert _is_yaml_table_suffix(f"{suffix}{compression}")
            assert f"{suffix}{compression}" in TABULAR_INPUT_SUFFIXES

    assert not _is_json_table_suffix(".jsonl")
    assert not _is_yaml_table_suffix(".json")


def test_tabular_suffixes_include_geopackage_inputs() -> None:
    assert ".gpkg" in TABULAR_INPUT_SUFFIXES
    assert ".geopackage" in TABULAR_INPUT_SUFFIXES
    assert ".gpkg" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".geopackage" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("features.gpkg")) == ".gpkg"
    assert tabular_suffix(Path("features.geopackage")) == ".geopackage"


def test_tabular_suffixes_include_shapefile_attribute_inputs() -> None:
    assert ".shp" in TABULAR_INPUT_SUFFIXES
    assert ".dbf" in TABULAR_INPUT_SUFFIXES
    assert ".shp" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".dbf" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("features.shp")) == ".shp"
    assert tabular_suffix(Path("features.dbf")) == ".dbf"


def test_tabular_suffixes_include_kml_inputs() -> None:
    assert ".kml" in TABULAR_INPUT_SUFFIXES
    assert ".kml.gz" in TABULAR_INPUT_SUFFIXES
    assert ".kmz" in TABULAR_INPUT_SUFFIXES
    assert ".kml" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".kml.gz" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".kmz" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("features.kml")) == ".kml"
    assert tabular_suffix(Path("features.kml.gz")) == ".kml.gz"
    assert tabular_suffix(Path("features.kmz")) == ".kmz"


def test_tabular_suffixes_include_hdf_inputs_without_h5_submissions() -> None:
    assert ".h5" in TABULAR_INPUT_SUFFIXES
    assert ".hdf" in TABULAR_INPUT_SUFFIXES
    assert ".hdf5" in TABULAR_INPUT_SUFFIXES
    assert ".h5" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".hdf" in TABULAR_SUBMISSION_SUFFIXES
    assert ".hdf5" in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("train.h5")) == ".h5"
    assert tabular_suffix(Path("sample_submission.hdf")) == ".hdf"
    assert tabular_suffix(Path("sample_submission.hdf5")) == ".hdf5"


def test_tabular_suffixes_include_numpy_inputs_without_numpy_submissions() -> None:
    assert ".npy" in TABULAR_INPUT_SUFFIXES
    assert ".npz" in TABULAR_INPUT_SUFFIXES
    assert ".npy" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".npz" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("train.npy")) == ".npy"
    assert tabular_suffix(Path("test.npz")) == ".npz"


def test_preferred_tabular_submission_suffix_normalizes_and_skips_unsupported_suffixes() -> None:
    assert preferred_tabular_submission_suffix(["onnx", "hdf5"]) == ".hdf5"
    assert preferred_tabular_submission_suffix([".mat", ".orc"]) == ".orc"
    assert preferred_tabular_submission_suffix([".html.zst", ".csv"]) == ".html.zst"
    assert preferred_tabular_submission_suffix([".sqlite3"]) == ".csv"


def test_preferred_rowless_tabular_sample_suffix_skips_structured_suffixes() -> None:
    assert preferred_rowless_tabular_sample_suffix(["jsonl", "hdf5", "feather"]) == ".feather"
    assert preferred_rowless_tabular_sample_suffix([".json.zst", ".pkl.zst"]) == ".pkl.zst"
    assert preferred_rowless_tabular_sample_suffix([".sqlite3", ".ndjson", ".orc"]) == ".csv"


def test_tabular_suffix_helpers_cover_structured_reader_groups() -> None:
    assert {".arrow", ".feather", ".ftr", ".ipc"} <= TABULAR_ARROW_IPC_SUFFIXES
    assert {".parquet", ".parq", ".pq"} <= TABULAR_PARQUET_SUFFIXES
    assert {".arff", ".arff.gz", ".arff.zst"} <= TABULAR_ARFF_SUFFIXES
    assert {".xlsb"} <= TABULAR_EXCEL_INPUT_ONLY_SUFFIXES
    assert {".ods", ".xls", ".xlsm", ".xlsx"} <= TABULAR_EXCEL_SUFFIXES
    assert (".fwf", ".fixed", ".fixedwidth") == TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES
    assert {".h5", ".hdf", ".hdf5"} <= TABULAR_HDF_SUFFIXES
    assert (".html", ".htm") == TABULAR_HTML_SUFFIX_PREFIXES
    assert {".mat"} <= TABULAR_MATLAB_SUFFIXES
    assert {".sas7bdat", ".xpt", ".xport"} <= TABULAR_SAS_SUFFIXES
    assert {".sav", ".zsav"} <= TABULAR_SPSS_SUFFIXES
    assert {".dta"} <= TABULAR_STATA_SUFFIXES
    assert {".sas7bdat.zip", ".xpt.zip", ".xport.zip", ".sav.zip", ".zsav.zip", ".dta.zip"} <= TABULAR_INPUT_SUFFIXES
    assert (".svm", ".svmlight", ".libsvm") == TABULAR_SVMLIGHT_SUFFIX_PREFIXES

    for suffix in (".jsonl", ".jsonl.gz", ".jsonlines", ".jsonlines.zst", ".ndjson", ".ndjson.zst"):
        assert is_json_lines_tabular_suffix(suffix)
    assert not is_json_lines_tabular_suffix(".json")

    for suffix in (".tsv", ".tsv.gz", ".txt", ".txt.zst"):
        assert is_tsv_like_tabular_suffix(suffix)
    assert not is_tsv_like_tabular_suffix(".csv")


@pytest.mark.parametrize("suffix", [".sas7bdat", ".xpt", ".xport"])
def test_tabular_suffixes_include_sas_inputs_without_submissions(suffix: str) -> None:
    assert suffix in TABULAR_INPUT_SUFFIXES
    assert suffix not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path(f"train{suffix}")) == suffix


def test_find_usable_sample_submissions_supports_sas_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample_submission.xpt"
    path.write_bytes(b"sas-xport")

    def fake_read_sas(read_path: Path, *args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        assert Path(read_path) == path
        return pd.DataFrame({"id": [1, 2], "target": [0, 0]})

    monkeypatch.setattr(pd, "read_sas", fake_read_sas)

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


@pytest.mark.parametrize("suffix", [".sav", ".zsav"])
def test_tabular_suffixes_include_spss_inputs_without_submissions(suffix: str) -> None:
    assert suffix in TABULAR_INPUT_SUFFIXES
    assert suffix not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path(f"train{suffix}")) == suffix


@pytest.mark.parametrize("suffix", [".sav", ".zsav"])
def test_find_usable_sample_submissions_supports_spss_inputs(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"sample_submission{suffix}"
    pyreadstat.write_sav(
        pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}),
        path,
        compress=suffix == ".zsav",
    )

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_tabular_suffixes_include_matlab_inputs_without_submissions() -> None:
    assert ".mat" in TABULAR_INPUT_SUFFIXES
    assert ".mat" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("train.mat")) == ".mat"


def test_find_usable_sample_submissions_supports_matlab_column_variables(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.mat"
    savemat(path, {"id": [1, 2], "target": [0.0, 0.0]})

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_tabular_suffixes_include_arff_inputs_without_submissions() -> None:
    assert ".arff" in TABULAR_INPUT_SUFFIXES
    assert ".arff.gz" in TABULAR_INPUT_SUFFIXES
    assert ".arff" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".arff.gz" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("train.arff")) == ".arff"
    assert tabular_suffix(Path("train.arff.gz")) == ".arff.gz"


def test_is_tabular_data_path_alias_matches_supported_tabular_inputs() -> None:
    assert is_tabular_data_path(Path("train.parquet"))
    assert is_tabular_data_path(Path("features.fwf"))
    assert is_tabular_data_path(Path("labels.csv.zst"))
    assert not is_tabular_data_path(Path("image.png"))


def test_tabular_suffixes_include_html_inputs_and_submissions() -> None:
    assert ".html" in TABULAR_INPUT_SUFFIXES
    assert ".htm" in TABULAR_INPUT_SUFFIXES
    assert ".html.zst" in TABULAR_INPUT_SUFFIXES
    assert ".html" in TABULAR_SUBMISSION_SUFFIXES
    assert ".htm" in TABULAR_SUBMISSION_SUFFIXES
    assert ".html.zst" in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("train.html")) == ".html"
    assert tabular_suffix(Path("sample_submission.html.zst")) == ".html.zst"


def test_tabular_suffixes_include_open_document_spreadsheets() -> None:
    assert ".ods" in TABULAR_INPUT_SUFFIXES
    assert ".ods" in TABULAR_SUBMISSION_SUFFIXES
    assert ".ods.zip" in TABULAR_INPUT_SUFFIXES
    assert ".ods.zip" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("sample_submission.ods")) == ".ods"
    assert tabular_suffix(Path("sample_submission.ods.zip")) == ".ods.zip"


def test_tabular_suffixes_include_macro_enabled_excel_workbooks() -> None:
    assert ".xlsm" in TABULAR_INPUT_SUFFIXES
    assert ".xlsm" in TABULAR_SUBMISSION_SUFFIXES
    assert ".xlsx.zip" in TABULAR_INPUT_SUFFIXES
    assert ".xlsx.zip" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("sample_submission.xlsm")) == ".xlsm"
    assert tabular_suffix(Path("sample_submission.xlsx.zip")) == ".xlsx.zip"


def test_tabular_suffixes_include_xlsb_inputs_without_submissions() -> None:
    assert ".xlsb" in TABULAR_INPUT_SUFFIXES
    assert ".xlsb.zip" in TABULAR_INPUT_SUFFIXES
    assert ".xlsb" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".xlsb.zip" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("train.xlsb")) == ".xlsb"
    assert tabular_suffix(Path("train.xlsb.zip")) == ".xlsb.zip"


def test_tabular_suffixes_include_dat_inputs_without_submissions() -> None:
    assert ".dat" in TABULAR_INPUT_SUFFIXES
    assert ".dat.gz" in TABULAR_INPUT_SUFFIXES
    assert ".dat" not in TABULAR_SUBMISSION_SUFFIXES
    assert ".dat.gz" not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path("train.dat.gz")) == ".dat.gz"


@pytest.mark.parametrize("suffix", [".fwf", ".fixed", ".fixedwidth", ".fwf.gz"])
def test_tabular_suffixes_include_fixed_width_inputs_without_submissions(suffix: str) -> None:
    assert suffix in TABULAR_INPUT_SUFFIXES
    assert suffix not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path(f"train{suffix}")) == suffix


@pytest.mark.parametrize("suffix", [".svm", ".svmlight", ".libsvm", ".svm.zst"])
def test_tabular_suffixes_include_svmlight_inputs_without_submissions(suffix: str) -> None:
    assert suffix in TABULAR_INPUT_SUFFIXES
    assert suffix not in TABULAR_SUBMISSION_SUFFIXES
    assert tabular_suffix(Path(f"train{suffix}")) == suffix


def test_find_usable_sample_submissions_supports_arff_inputs(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.arff"
    path.write_text(
        """
@RELATION sample_submission
@ATTRIBUTE id NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
1,no
2,yes
""".strip(),
        encoding="utf-8",
    )

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_find_usable_sample_submissions_supports_compressed_arff_inputs(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.arff.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            """
@RELATION sample_submission
@ATTRIBUTE id NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
1,no
2,yes
""".strip()
        )

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_find_usable_sample_submissions_supports_ods_tables(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.ods"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_excel(path, index=False)

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_find_usable_sample_submissions_supports_jsonl_zst(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.jsonl.zst"
    payload = b'{"id":1,"target":0.0}\n{"id":2,"target":0.0}\n'
    path.write_bytes(zstd.ZstdCompressor().compress(payload))

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_find_usable_sample_submissions_supports_xlsb_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample_submission.xlsb"
    path.write_bytes(b"xlsb")

    def fake_read_excel(read_path: Path, *args, **kwargs) -> pd.DataFrame:
        assert Path(read_path) == path
        assert kwargs["engine"] == "pyxlsb"
        return pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_find_usable_sample_submissions_supports_compressed_html_tables(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.html.zst"
    html = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_html(index=False)
    path.write_bytes(zstd.ZstdCompressor().compress(html.encode("utf-8")))

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5"])
def test_binary_submission_samples_count_rows_and_columns(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"sample_submission{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    if suffix == ".orc":
        frame.to_orc(path, index=False)
    else:
        frame.to_hdf(path, key="submission", mode="w", format="table", index=False)

    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_native_hdf5_submission_samples_count_rows_and_columns(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sample_submission.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("submission")
        group.create_dataset("id", data=[1, 2])
        group.create_dataset("target", data=[0.0, 1.0])

    candidates = find_usable_sample_submissions(tmp_path)

    assert candidates == [path]
    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_tabular_data_row_count_capped_counts_sqlite_rows(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE predictions (id INTEGER, target REAL)")
        conn.executemany(
            "INSERT INTO predictions VALUES (?, ?)",
            [(1, 0.1), (2, 0.2), (3, 0.3)],
        )

    assert tabular_data_row_count_capped(path, cap=10) == 3
    assert tabular_data_row_count_capped(path, cap=2) == 3


def test_tabular_data_row_count_capped_counts_duckdb_rows(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.duckdb"
    conn = duckdb.connect(str(path))
    try:
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target DOUBLE)")
        conn.execute("INSERT INTO sample_submission VALUES (1, 0.1), (2, 0.2), (3, 0.3)")
    finally:
        conn.close()

    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_file_has_data_rows(path)
    assert tabular_data_row_count_capped(path, cap=10) == 3
    assert tabular_data_row_count_capped(path, cap=2) == 3


def test_tabular_data_row_count_capped_counts_rds_rows(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.rds"
    pyreadr.write_rds(path, pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}))

    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_file_has_data_rows(path)
    assert tabular_data_row_count_capped(path, cap=10) == 3
    assert tabular_data_row_count_capped(path, cap=2) == 3


def test_wrapped_json_tables_count_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.json"
    path.write_text(
        '{"records":[{"id":1,"target":0.1},{"id":2,"target":0.2}]}',
        encoding="utf-8",
    )

    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_compressed_wrapped_json_tables_count_rows(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.json.zst"
    path.write_bytes(
        zstd.ZstdCompressor().compress(
            b'{"data":[{"id":1,"target":0.1},{"id":2,"target":0.2},{"id":3,"target":0.3}]}',
        ),
    )

    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=2) == 3


def test_geojson_feature_collection_counts_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": "cell_001",
                        "type": "Feature",
                        "properties": {"target": 0.1},
                        "geometry": {"type": "Point", "coordinates": [139.0, 35.0]},
                    },
                    {
                        "id": "cell_002",
                        "type": "Feature",
                        "properties": {"target": 0.2},
                        "geometry": {"type": "Point", "coordinates": [140.0, 36.0]},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_geopackage_attribute_table_counts_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.gpkg"
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
            "INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) "
            "VALUES ('sample_submission', 'attributes', 'sample_submission', 0)"
        )
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target REAL, geom BLOB)")
        conn.executemany(
            "INSERT INTO sample_submission VALUES (?, ?, ?)",
            [(1, 0.1, b"\x47\x50\x00\x01"), (2, 0.2, b"\x47\x50\x00\x02")],
        )

    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_shapefile_attribute_table_counts_rows_and_columns(tmp_path: Path) -> None:
    shp_path = tmp_path / "sample_submission.shp"
    dbf_path = tmp_path / "sample_submission.dbf"
    shp_path.write_bytes(b"")
    fields = [
        ("id", "N", 10, 0),
        ("target", "N", 12, 3),
    ]
    rows = [(1, 0.1), (2, 0.2)]
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
        payload.append(0x20)
        payload.extend(str(row[0]).encode("ascii").rjust(10, b" "))
        payload.extend(f"{row[1]:.3f}".encode("ascii").rjust(12, b" "))
    payload.append(0x1A)
    dbf_path.write_bytes(bytes(payload))

    assert tabular_file_has_data_rows(shp_path)
    assert tabular_file_has_two_or_more_columns(shp_path)
    assert tabular_data_row_count_capped(shp_path, cap=10) == 2


def test_kmz_placemark_table_counts_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.kmz"
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        '<Placemark><ExtendedData><Data name="id"><value>1</value></Data>'
        '<Data name="target"><value>0.1</value></Data></ExtendedData>'
        "<Point><coordinates>139,35,0</coordinates></Point></Placemark>"
        '<Placemark><ExtendedData><Data name="id"><value>2</value></Data>'
        '<Data name="target"><value>0.2</value></Data></ExtendedData>'
        "<Point><coordinates>140,36,0</coordinates></Point></Placemark>"
        "</Document></kml>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("doc.kml", kml)

    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_compressed_kml_placemark_table_counts_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.kml.gz"
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        '<Placemark><ExtendedData><Data name="id"><value>1</value></Data>'
        '<Data name="target"><value>0.1</value></Data></ExtendedData>'
        "<Point><coordinates>139,35,0</coordinates></Point></Placemark>"
        '<Placemark><ExtendedData><Data name="id"><value>2</value></Data>'
        '<Data name="target"><value>0.2</value></Data></ExtendedData>'
        "<Point><coordinates>140,36,0</coordinates></Point></Placemark>"
        "</Document></kml>"
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(kml)

    assert tabular_file_has_data_rows(path)
    assert tabular_file_has_two_or_more_columns(path)
    assert tabular_data_row_count_capped(path, cap=10) == 2


def test_sample_name_score_recognizes_common_sample_aliases() -> None:
    assert sample_name_score(Path("sample_submission.csv")) == 3
    assert sample_name_score(Path("SampleSubmission.csv")) == 3
    assert sample_name_score(Path("submission_sample.csv")) == 2
    assert sample_name_score(Path("sample_predictions.csv")) == 2
    assert sample_name_score(Path("example_submission.csv")) == 2
    assert sample_name_score(Path("prediction_template.csv")) == 2
    assert sample_name_score(Path("AnswerTemplate.csv")) == 2
    assert sample_name_score(Path("submission.csv")) == 1
    assert sample_name_score(Path("template.csv")) == 0
    assert sample_name_score(Path("train.csv")) == 0


def test_find_usable_sample_submissions_includes_prediction_aliases(tmp_path: Path) -> None:
    (tmp_path / "sample_predictions.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (tmp_path / "train.csv").write_text("id,target\n1,1\n", encoding="utf-8")

    candidates = find_usable_sample_submissions(tmp_path)

    assert [path.name for path in candidates] == ["sample_predictions.csv"]


def test_find_usable_sample_submissions_includes_output_templates(tmp_path: Path) -> None:
    (tmp_path / "prediction_template.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (tmp_path / "template.csv").write_text("id,value\n1,x\n", encoding="utf-8")

    candidates = find_usable_sample_submissions(tmp_path)

    assert [path.name for path in candidates] == ["prediction_template.csv"]


def test_structured_discovery_readers_stabilize_problematic_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.json"
    path.write_text(
        json.dumps(
            {
                "columns": ["id", "", None, "score", "score"],
                "data": [[1, "a", 10, 0.1, 0.2], [2, "b", 20, 0.3, 0.4]],
            }
        ),
        encoding="utf-8",
    )

    sample = _read_structured_tabular_sample(path)
    frame = _read_structured_tabular_frame(path)

    assert list(sample.columns) == ["id", "column_2", "column_3", "score", "score_1"]
    assert list(frame.columns) == ["id", "column_2", "column_3", "score", "score_1"]


def test_discovery_column_finalize_flattens_multiindex_columns() -> None:
    frame = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("fold", "a"), ("Unnamed: 1_level_0", None)]),
    )

    normalized = _finalize_tabular_frame(frame)

    assert list(normalized.columns) == ["fold_a", "column_2"]


def test_select_sample_submission_prefers_canonical_over_alias(tmp_path: Path) -> None:
    alias = tmp_path / "sample_predictions.csv"
    canonical = tmp_path / "sample_submission.csv"
    alias.write_text("id,target\n1,0\n2,0\n", encoding="utf-8")
    canonical.write_text("id,target\n1,0\n", encoding="utf-8")

    selected = select_sample_submission_path([alias, canonical])

    assert selected == canonical


def test_path_mentions_role_recognizes_inference_test_aliases() -> None:
    assert path_mentions_role(Path("eval_features.csv"), "test")
    assert path_mentions_role(Path("validation_features.csv"), "test")
    assert path_mentions_role(Path("holdout_features.parquet"), "test")
    assert path_mentions_role(Path("unlabeled_records.jsonl"), "test")
    assert path_mentions_role(Path("scoring_features.parquet"), "test")
    assert path_mentions_role(Path("inference.jsonl"), "test")
    assert path_mentions_role(Path("leaderboard.tsv"), "test")
    assert path_mentions_role(Path("final_features.csv"), "test")
    assert path_mentions_role(Path("blind_records.csv"), "test")
    assert path_mentions_role(Path("challenge_features.parquet"), "test")
    assert path_mentions_role(Path("public_train.csv"), "train")
    assert not path_mentions_role(Path("public_train.csv"), "test")
    assert path_mentions_role(Path("final_train.csv"), "train")
    assert not path_mentions_role(Path("final_train.csv"), "test")
    assert path_mentions_role(Path("public/features.csv"), "test")
    assert path_mentions_role(Path("public/train.csv"), "train")
    assert not path_mentions_role(Path("public/train.csv"), "test")


def test_path_mentions_role_recognizes_asset_modality_suffixes() -> None:
    assert path_mentions_role(Path("train_audio.csv"), "train")
    assert path_mentions_role(Path("test_audio.csv"), "test")
    assert path_mentions_role(Path("train_scans.parquet"), "train")
    assert path_mentions_role(Path("test_videos.jsonl"), "test")
    assert path_mentions_role(Path("train_dicom.csv"), "train")
    assert path_mentions_role(Path("test_series.parquet"), "test")
    assert path_mentions_role(Path("validation_studies.csv"), "test")
