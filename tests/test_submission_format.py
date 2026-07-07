from __future__ import annotations

import pytest

from kagglebot.bootstrap import _extract_usable_submission_section
from kagglebot.submission_format import extract_submission_section, parse_submission_format


def test_extract_submission_section_skips_submission_code_requirements_heading() -> None:
    markdown = (
        "## Foundational Rules\n\n"
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, not a CSV header.\n"
    )
    assert extract_submission_section(markdown) is None


def test_extract_submission_section_prefers_real_submission_block() -> None:
    markdown = (
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, not a CSV header.\n\n"
        "## Submission\n\n"
        "```csv\n"
        "id,prediction\n"
        "```\n"
    )
    section = extract_submission_section(markdown)
    assert section is not None
    hint = parse_submission_format(section)
    assert hint.columns == ["id", "prediction"]
    assert hint.expected_suffixes and hint.expected_suffixes[0] == ".csv"


def test_extract_usable_submission_section_rejects_rules_text() -> None:
    markdown = (
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, not a CSV header.\n"
    )
    assert _extract_usable_submission_section(markdown) is None


def test_parse_submission_format_prefers_zip_when_rules_require_zip_file() -> None:
    markdown = (
        "## Submission Format\n\n"
        "You must submit a ZIP file containing per-record predictions.\n"
        "```csv\nid,prediction\n```\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes is not None
    assert hint.expected_suffixes[0] == ".zip"
    assert ".csv" in hint.expected_suffixes
    assert hint.artifact_class == "multi_file_zip"


def test_parse_submission_format_reads_bullet_column_definitions() -> None:
    markdown = (
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename\n"
        "* `Category`: The predicted class\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.columns == ["Id", "Category"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_bold_bullet_column_definitions() -> None:
    markdown = (
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "- **Id**: The filename\n"
        "- **Category**: The predicted class\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.columns == ["Id", "Category"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_required_columns_from_prose() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Participants should submit their files in CSV format. "
        "Each submission must include the columns KEEP, ASSOCIATION, and DIFF. "
        "Use `; ` to separate multiple codes.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.columns == ["KEEP", "ASSOCIATION", "DIFF"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_header_columns_from_prose() -> None:
    markdown = "## Submission Format\n\nThe CSV header: image_id,prediction_string.\n"

    hint = parse_submission_format(markdown)

    assert hint.columns == ["image_id", "prediction_string"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_header_row_is_columns_from_prose() -> None:
    markdown = "## Submission Format\n\nThe required header row is row_id,score.\n"

    hint = parse_submission_format(markdown)

    assert hint.columns == ["row_id", "score"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_columns_from_jsonl_object_example() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Submit `submission.jsonl` with one JSON object per line:\n\n"
        "```json\n"
        "{\n"
        '  "row_id": 100,\n'
        '  "target": 0\n'
        "}\n"
        "```\n"
    )

    hint = parse_submission_format(markdown)

    assert hint.columns == ["row_id", "target"]
    assert hint.expected_suffixes is not None
    assert hint.expected_suffixes[0] == ".jsonl"
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_explicit_compressed_csv_suffix() -> None:
    markdown = "## Submission Format\n\nYou must upload a file named `submission.csv.gz` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".csv.gz"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_explicit_zstd_compressed_csv_suffix() -> None:
    markdown = "## Submission Format\n\nYou must upload a file named `submission.csv.zst` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".csv.zst"]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".jsonl", ".jsonl"),
        (".jsonl.gz", ".jsonl.gz"),
        (".jsonl.zst", ".jsonl.zst"),
        (".jsonlines", ".jsonlines"),
        (".jsonlines.zst", ".jsonlines.zst"),
        (".ndjson", ".ndjson"),
        (".ndjson.zst", ".ndjson.zst"),
    ],
)
def test_parse_submission_format_classifies_json_lines_suffix_as_tabular_without_columns(
    suffix: str,
    expected: str,
) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [expected]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize("suffix", [".json", ".json.zst", ".txt"])
def test_parse_submission_format_keeps_ambiguous_json_and_txt_single_file_without_columns(suffix: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "single_file"


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".json", ".json"),
        (".json.gz", ".json.gz"),
        (".json.zst", ".json.zst"),
        (".jsonl.gz", ".jsonl.gz"),
        (".jsonl.zst", ".jsonl.zst"),
        (".ndjson.zst", ".ndjson.zst"),
    ],
)
def test_parse_submission_format_reads_explicit_compressed_json_suffix(suffix: str, expected: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [expected]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_preserves_ndjson_code_fence_language() -> None:
    markdown = """## Submission Format

Submit one newline-delimited JSON object per prediction:

```ndjson
{"id": 1, "target": 0}
```
"""

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".ndjson"]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("json-lines", ".jsonl"),
        ("nd-json", ".ndjson"),
        ("html", ".html"),
    ],
)
def test_parse_submission_format_reads_code_fence_language_aliases(language: str, expected: str) -> None:
    markdown = f"""## Submission Format

Submit predictions in this format:

```{language}
id,target
1,0
```
"""

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [expected]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Upload an NDJSON file for scoring.", ".ndjson"),
        ("Upload a JSONLines file for scoring.", ".jsonlines"),
        ("Upload a JSON Lines file for scoring.", ".jsonl"),
    ],
)
def test_parse_submission_format_preserves_specific_json_lines_keyword(description: str, expected: str) -> None:
    markdown = f"## Submission Format\n\n{description}\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [expected]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_inferrs_compressed_json_lines_keyword() -> None:
    cases = [
        ("Upload a gzip-compressed NDJSON file for scoring.", ".ndjson.gz"),
        ("Upload a bzip2-compressed NDJSON file for scoring.", ".ndjson.bz2"),
        ("Upload an xz-compressed NDJSON file for scoring.", ".ndjson.xz"),
        ("Upload a zstd-compressed NDJSON file for scoring.", ".ndjson.zst"),
        ("Upload a gzip-compressed JSONLines file for scoring.", ".jsonlines.gz"),
        ("Upload a bzip2-compressed JSONLines file for scoring.", ".jsonlines.bz2"),
        ("Upload an xz-compressed JSONLines file for scoring.", ".jsonlines.xz"),
        ("Upload a zstd-compressed JSONLines file for scoring.", ".jsonlines.zst"),
        ("Upload a zstd-compressed JSON Lines file for scoring.", ".jsonl.zst"),
    ]
    for description, expected in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [expected], description
        assert hint.artifact_class == "tabular", description


def test_parse_submission_format_inferrs_compressed_structured_tabular_keyword() -> None:
    cases = [
        ("Upload a gzip-compressed YAML file for scoring.", ".yaml.gz"),
        ("Upload an xz-compressed YML file for scoring.", ".yaml.xz"),
        ("Upload a zstd-compressed XML file for scoring.", ".xml.zst"),
        ("Upload a bzip2-compressed HTML file with columns id,target for scoring.", ".html.bz2"),
        ("Upload an xz-compressed HTM file with columns id,target for scoring.", ".html.xz"),
        ("Upload a gzip-compressed comma-separated values file for scoring.", ".csv.gz"),
        ("Upload a zstd-compressed semicolon-delimited file for scoring.", ".csv.zst"),
        ("Upload a gzip-compressed PSV file for scoring.", ".psv.gz"),
        ("Upload a zstd-compressed TAB file for scoring.", ".tab.zst"),
    ]
    for description, expected in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [expected], description
        assert hint.artifact_class == "tabular", description


@pytest.mark.parametrize("suffix", [".xls", ".xlsm", ".xlsx", ".ods"])
def test_parse_submission_format_reads_explicit_excel_suffix(suffix: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize("suffix", [".parq", ".pq"])
def test_parse_submission_format_reads_explicit_parquet_alias_suffix(suffix: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize("suffix", [".tab", ".psv"])
def test_parse_submission_format_reads_explicit_delimited_text_alias_suffix(suffix: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize("suffix", [".feather", ".ftr", ".arrow", ".ipc"])
def test_parse_submission_format_reads_explicit_arrow_ipc_suffix(suffix: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_explicit_orc_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `submission.orc` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".orc"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_explicit_avro_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `submission.avro` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".avro"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_explicit_hdf5_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `submission.hdf5` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".hdf5"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_classifies_explicit_hdf5_model_as_single_file() -> None:
    markdown = "## Submission Format\n\nUpload `submission.hdf5` as an HDF5 model file for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".hdf5"]
    assert hint.artifact_class == "single_file"


@pytest.mark.parametrize("suffix", [".html", ".htm", ".html.zst"])
def test_parse_submission_format_reads_explicit_html_tabular_suffix_when_columns_are_named(suffix: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_explicit_matlab_single_file_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `submission.mat` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".mat"]
    assert hint.artifact_class == "single_file"


def test_parse_submission_format_inferrs_matlab_single_file_from_tokens() -> None:
    markdown = "## Submission Format\n\nSubmit a MATLAB file for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".mat"]
    assert hint.artifact_class == "single_file"


def test_parse_submission_format_reads_explicit_stata_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `submission.dta` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".dta"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_explicit_compressed_pickle_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `submission.pkl.zst` with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".pkl.zst"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_classifies_pickle_suffix_as_tabular_without_columns() -> None:
    markdown = "## Submission Format\n\nUpload `submission.pkl.zst` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".pkl.zst"]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Upload a pickle model file for scoring.", ".pkl"),
        ("Upload a zstd-compressed pickle model file for scoring.", ".pkl.zst"),
        ("Upload a pickled estimator file for scoring.", ".pkl"),
    ],
)
def test_parse_submission_format_classifies_model_pickle_as_single_file(
    description: str,
    suffix: str,
) -> None:
    markdown = f"## Submission Format\n\n{description}\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "single_file"


def test_parse_submission_format_inferrs_zstd_pickle_from_tokens() -> None:
    markdown = "## Submission Format\n\nUpload a zstd-compressed pickle file for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".pkl.zst"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_ignores_sqlite_as_submission_suffix() -> None:
    markdown = "## Submission Format\n\nThe sample input is `sample_submission.sqlite`; upload predictions normally.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes is None
    assert hint.artifact_class == "unknown"


@pytest.mark.parametrize("suffix", [".db", ".sqlite", ".sqlite3"])
def test_parse_submission_format_reads_explicit_sqlite_submission_suffix(suffix: str) -> None:
    markdown = f"## Submission Format\n\nUpload `submission{suffix}` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "single_file"


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Upload a SQLite database for scoring.", ".sqlite"),
        ("Upload a SQLite3 database for scoring.", ".sqlite3"),
    ],
)
def test_parse_submission_format_infers_sqlite_submission_suffix_from_prose(
    description: str,
    suffix: str,
) -> None:
    markdown = f"## Submission Format\n\n{description}\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "single_file"


@pytest.mark.parametrize(
    "suffix",
    [".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".tzst"],
)
def test_parse_submission_format_reads_explicit_tar_suffix(suffix: str) -> None:
    markdown = f"## Submission Format\n\nYou must upload `submission{suffix}` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "single_file"
    assert hint.artifact_container == "tar"


def test_parse_submission_format_reads_explicit_numpy_single_file_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `predictions.npy` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".npy"]
    assert hint.artifact_class == "single_file"


def test_parse_submission_format_reads_medical_image_suffixes() -> None:
    for suffix in [".nii.gz", ".ome.tif", ".svs", ".ndpi", ".czi", ".mrxs", ".qptiff"]:
        markdown = f"## Submission Format\n\nYou must submit `segmentation{suffix}` for each case bundle.\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], suffix
        assert hint.artifact_class == "single_file", suffix


def test_parse_submission_format_infers_medical_image_suffix_from_prose() -> None:
    cases = [
        ("Participants must upload a MetaImage segmentation file for scoring.", ".mha"),
        ("Participants must upload an NRRD mask for scoring.", ".nrrd"),
        ("Participants must upload a Nearly Raw Raster Data volume for scoring.", ".nrrd"),
        ("Participants must upload an Analyze 7.5 image for scoring.", ".hdr"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], description
        assert hint.artifact_class == "single_file", description


def test_parse_submission_format_infers_numpy_array_from_prose() -> None:
    markdown = "## Submission Format\n\nParticipants must upload a NumPy array file for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".npy"]
    assert hint.artifact_class == "single_file"


def test_parse_submission_format_infers_numpy_archive_from_prose() -> None:
    cases = [
        ("Participants must upload a NumPy archive file for scoring.", ".npz"),
        ("Participants must upload a compressed NumPy archive for scoring.", ".npz"),
        ("Participants must upload a compressed NumPy array for scoring.", ".npz"),
        ("Participants must upload a SciPy sparse matrix archive for scoring.", ".npz"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], description
        assert hint.artifact_class == "single_file", description


def test_parse_submission_format_infers_non_tabular_asset_suffix_from_prose() -> None:
    cases = [
        ("Participants must upload a WebP image file for scoring.", ".webp"),
        ("Participants must upload a CZI microscopy file for scoring.", ".czi"),
        ("Participants must upload an MRXS whole-slide image for scoring.", ".mrxs"),
        ("Participants must upload a QPTIFF image for scoring.", ".qptiff"),
        ("Participants must upload an AVIF image file for scoring.", ".avif"),
        ("Participants must upload a HEIC image file for scoring.", ".heic"),
        ("Participants must upload a HEIF image file for scoring.", ".heif"),
        ("Participants must upload a JPEG XL image file for scoring.", ".jxl"),
        ("Participants must upload a JPEG 2000 image file for scoring.", ".jp2"),
        ("Participants must upload an OpenEXR image file for scoring.", ".exr"),
        ("Participants must upload a Netpbm image file for scoring.", ".pnm"),
        ("Participants must upload a Windows bitmap image for scoring.", ".bmp"),
        ("Participants must upload a Portable Network Graphics image for scoring.", ".png"),
        ("Participants must upload an Advanced Audio Coding file for scoring.", ".aac"),
        ("Participants must upload a Free Lossless Audio Codec file for scoring.", ".flac"),
        ("Participants must upload an MP3 audio file for scoring.", ".mp3"),
        ("Participants must upload an Ogg Vorbis audio file for scoring.", ".ogg"),
        ("Participants must upload an MPEG-4 audio file for scoring.", ".m4a"),
        ("Participants must upload a MIDI audio file for scoring.", ".mid"),
        ("Participants must upload an OPUS audio file for scoring.", ".opus"),
        ("Participants must upload a Windows Media Audio file for scoring.", ".wma"),
        ("Participants must upload an AIFF audio file for scoring.", ".aiff"),
        ("Participants must upload an Audio Video Interleave file for scoring.", ".avi"),
        ("Participants must upload a 3GP video file for scoring.", ".3gp"),
        ("Participants must upload a Flash Video file for scoring.", ".flv"),
        ("Participants must upload a Matroska video file for scoring.", ".mkv"),
        ("Participants must upload an M4V video file for scoring.", ".m4v"),
        ("Participants must upload an MPEG-4 video file for scoring.", ".mp4"),
        ("Participants must upload an MPEG video file for scoring.", ".mpg"),
        ("Participants must upload a QuickTime movie file for scoring.", ".mov"),
        ("Participants must upload a WMV video file for scoring.", ".wmv"),
        ("Participants must upload an E57 point cloud file for scoring.", ".e57"),
        ("Participants must upload an XYZ point cloud file for scoring.", ".xyz"),
        ("Participants must upload a PTS point cloud file for scoring.", ".pts"),
        ("Participants must upload a PTX point cloud file for scoring.", ".ptx"),
        ("Participants must upload an OFF mesh file for scoring.", ".off"),
        ("Participants must upload a GLTF scene file for scoring.", ".gltf"),
        ("Participants must upload a Wavefront OBJ mesh for scoring.", ".obj"),
        ("Participants must upload a COLLADA scene for scoring.", ".dae"),
        ("Participants must upload a binary glTF scene for scoring.", ".glb"),
        ("Participants must upload a VTK legacy mesh file for scoring.", ".vtk"),
        ("Participants must upload a VTK PolyData file for scoring.", ".vtp"),
        ("Participants must upload a VTK unstructured grid file for scoring.", ".vtu"),
        ("Participants must upload a Gmsh mesh file for scoring.", ".msh"),
        ("Participants must upload a Medit mesh file for scoring.", ".mesh"),
        ("Participants must upload an Autodesk FBX scene for scoring.", ".fbx"),
        ("Participants must upload a 3D Studio scene for scoring.", ".3ds"),
        ("Participants must upload a STEP CAD file for scoring.", ".step"),
        ("Participants must upload a STP geometry format file for scoring.", ".stp"),
        ("Participants must upload an IGES CAD model for scoring.", ".iges"),
        ("Participants must upload an IGS geometry file for scoring.", ".igs"),
        ("Participants must upload an IFC BIM model for scoring.", ".ifc"),
        ("Participants must upload a BREP geometry file for scoring.", ".brep"),
        ("Participants must upload a PDF document for scoring.", ".pdf"),
        ("Participants must upload a Word document for scoring.", ".docx"),
        ("Participants must upload an EPUB document for scoring.", ".epub"),
        ("Participants must upload an OpenDocument text file for scoring.", ".odt"),
        ("Participants must upload a PowerPoint file for scoring.", ".pptx"),
        ("Participants must upload a Markdown report for scoring.", ".md"),
        ("Participants must upload a LaTeX file for scoring.", ".tex"),
        ("Participants must upload a reStructuredText file for scoring.", ".rst"),
        ("Participants must upload an AsciiDoc file for scoring.", ".adoc"),
        ("Participants must upload a WebVTT caption file for scoring.", ".vtt"),
        ("Participants must upload a SubRip subtitle file for scoring.", ".srt"),
        ("Participants must upload an HTML file for scoring.", ".html"),
        ("Participants must upload a GeoJSON file for scoring.", ".geojson"),
        ("Participants must upload a GeoJSON Lines file for scoring.", ".geojsonl"),
        ("Participants must upload a GeoJSON text sequence for scoring.", ".geojsonseq"),
        ("Participants must upload a TopoJSON file for scoring.", ".topojson"),
        ("Participants must upload an OpenStreetMap XML file for scoring.", ".osm"),
        ("Participants must upload an OSM PBF file for scoring.", ".osm.pbf"),
        ("Participants must upload an MBTiles file for scoring.", ".mbtiles"),
        ("Participants must upload a PMTiles file for scoring.", ".pmtiles"),
        ("Participants must upload a Mapbox vector tile for scoring.", ".mvt"),
        ("Participants must upload a GeoTIFF file for scoring.", ".tif"),
        ("Participants must upload a Cloud Optimized GeoTIFF for scoring.", ".tif"),
        ("Participants must upload a COG raster for scoring.", ".tif"),
        ("Participants must upload a GeoPackage file for scoring.", ".gpkg"),
        ("Participants must upload an ESRI shapefile for scoring.", ".shp"),
        ("Participants must upload a Keyhole Markup Language file for scoring.", ".kml"),
        ("Participants must upload a zipped KML file for scoring.", ".kmz"),
        ("Participants must upload a GDAL VRT file for scoring.", ".vrt"),
        ("Participants must upload a MapInfo MIF/MID pair for scoring.", ".mif"),
        ("Participants must upload an ENVI raster header for scoring.", ".hdr"),
        ("Participants must upload a Band Interleaved by Line raster for scoring.", ".bil"),
        ("Participants must upload a Band Interleaved by Pixel raster for scoring.", ".bip"),
        ("Participants must upload a Band Sequential raster for scoring.", ".bsq"),
        ("Participants must upload a PDB file for scoring.", ".pdb"),
        ("Participants must upload a macromolecular crystallographic information file for scoring.", ".mmcif"),
        ("Participants must upload an SDF molecule file for scoring.", ".sdf"),
        ("Participants must upload a Tripos MOL2 molecule file for scoring.", ".mol2"),
        ("Participants must upload an MDL molfile for scoring.", ".mol"),
        ("Participants must upload a FASTQ reads file for scoring.", ".fastq"),
        ("Participants must upload a SMILES file for scoring.", ".smiles"),
        ("Participants must upload a Simplified Molecular Input Line Entry System file for scoring.", ".smiles"),
        ("Participants must upload an InChI file for scoring.", ".inchi"),
        ("Participants must upload a SELFIES file for scoring.", ".selfies"),
        ("Participants must upload a reaction file for scoring.", ".rxn"),
        ("Participants must upload a FASTA sequence file for scoring.", ".fasta"),
        ("Participants must upload a GraphML file for scoring.", ".graphml"),
        ("Participants must upload a Graph Exchange XML Format file for scoring.", ".gexf"),
        ("Participants must upload a Graph Modeling Language file for scoring.", ".gml"),
        ("Participants must upload an edge list file for scoring.", ".edgelist"),
        ("Participants must upload a Matrix Market file for scoring.", ".mtx"),
        ("Participants must upload a NetCDF file for scoring.", ".nc"),
        ("Participants must upload a NetCDF-4 file for scoring.", ".nc4"),
        ("Participants must upload a Common Data Format file for scoring.", ".cdf"),
        ("Participants must upload a GRIB file for scoring.", ".grib"),
        ("Participants must upload a FITS file for scoring.", ".fits"),
        ("Participants must upload an AnnData file for scoring.", ".h5ad"),
        ("Participants must upload a H5AD file for scoring.", ".h5ad"),
        ("Participants must upload a Loom file for scoring.", ".loom"),
        ("Participants must upload an OME-Zarr store for scoring.", ".ome.zarr"),
        ("Participants must upload a Zarr store for scoring.", ".zarr"),
        ("Participants must upload an N5 store for scoring.", ".n5"),
        ("Participants must upload a Keras model file for scoring.", ".keras"),
        ("Participants must upload a Keras H5 model file for scoring.", ".h5"),
        ("Participants must upload a Safetensors model file for scoring.", ".safetensors"),
        ("Participants must upload a PyTorch checkpoint for scoring.", ".pth"),
        ("Participants must upload a TorchScript model for scoring.", ".pt"),
        ("Participants must upload a Core ML model file for scoring.", ".mlmodel"),
        ("Participants must upload a Predictive Model Markup Language file for scoring.", ".pmml"),
        ("Participants must upload a SentencePiece model file for scoring.", ".spm"),
        ("Participants must upload a skops model file for scoring.", ".skops"),
        ("Participants must upload an XGBoost UBJSON model file for scoring.", ".ubj"),
        ("Participants must upload an ONNX model file for scoring.", ".onnx"),
        ("Participants must upload a GGUF model file for scoring.", ".gguf"),
        ("Participants must upload a Msgpack model file for scoring.", ".msgpack"),
        ("Participants must upload a TensorFlow Lite model file for scoring.", ".tflite"),
        ("Participants must upload a protobuf model file for scoring.", ".pb"),
        ("Participants must upload a Joblib model file for scoring.", ".joblib"),
        ("Participants must upload a European Data Format signal file for scoring.", ".edf"),
        ("Participants must upload an EDF+ signal file for scoring.", ".edf"),
        ("Participants must upload a Biomedical Data Format signal file for scoring.", ".bdf"),
        ("Participants must upload a WaveForm DataBase header for scoring.", ".hea"),
        ("Participants must upload a WFDB header file for scoring.", ".hea"),
        ("Participants must upload a Neurodata Without Borders file for scoring.", ".nwb"),
        ("Participants must upload a Neurodata Without Borders Neurophysiology file for scoring.", ".nwb"),
        ("Participants must upload a National Instruments TDMS file for scoring.", ".tdms"),
        ("Participants must upload a TDMS signal file for scoring.", ".tdms"),
        ("Participants must upload an Axon Binary File for scoring.", ".abf"),
        ("Participants must upload an Axon Binary Format file for scoring.", ".abf"),
        ("Participants must upload an XGBoost model file for scoring.", ".xgb"),
        ("Participants must upload a CatBoost model file for scoring.", ".cbm"),
        ("Participants must upload a Python script for scoring.", ".py"),
        ("Participants must upload a Jupyter notebook for scoring.", ".ipynb"),
        ("Participants must upload an R script for scoring.", ".r"),
        ("Participants must upload a Julia script for scoring.", ".jl"),
        ("Participants must upload COCO annotations for scoring.", ".json"),
        ("Participants must upload COCO panoptic JSON for scoring.", ".json"),
        ("Participants must upload LabelMe annotations for scoring.", ".json"),
        ("Participants must upload YOLO labels for scoring.", ".txt"),
        ("Participants must upload YOLOv8 annotation text files for scoring.", ".txt"),
        ("Participants must upload JSON-LD linked data for scoring.", ".jsonld"),
        ("Participants must upload Turtle RDF for scoring.", ".ttl"),
        ("Participants must upload N-Triples for scoring.", ".nt"),
        ("Participants must upload N-Quads for scoring.", ".nq"),
        ("Participants must upload RDF XML for scoring.", ".rdf"),
        ("Participants must upload an OWL ontology for scoring.", ".owl"),
        ("Participants must upload TriG RDF for scoring.", ".trig"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix]
        assert hint.artifact_class == "single_file"


def test_parse_submission_format_infers_annotation_tabular_style_suffix_from_prose() -> None:
    cases = [
        ("Participants must upload Pascal VOC annotations for scoring.", ".xml"),
        ("Participants must upload Pascal VOC XML files for scoring.", ".xml"),
        ("Participants must upload Open Images annotations for scoring.", ".csv"),
        ("Participants must upload Open Images CSV files for scoring.", ".csv"),
        ("Participants must upload RLE masks for scoring.", ".csv"),
        ("Participants must upload run length encoding masks for scoring.", ".csv"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], description
        assert hint.artifact_class == "tabular", description


def test_parse_submission_format_infers_compressed_document_suffix_from_prose() -> None:
    cases = [
        ("Participants must upload a gzip-compressed Markdown report for scoring.", ".md.gz"),
        ("Participants must upload a zstd-compressed LaTeX file for scoring.", ".tex.zst"),
        ("Participants must upload a zstd-compressed WebVTT caption file for scoring.", ".vtt.zst"),
        ("Participants must upload an xz-compressed rich text document for scoring.", ".rtf.xz"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], description
        assert hint.artifact_class == "single_file", description


def test_parse_submission_format_infers_compressed_text_like_asset_suffix_from_prose() -> None:
    cases = [
        ("Participants must upload a gzip-compressed FASTA sequence file for scoring.", ".fasta.gz"),
        ("Participants must upload a zstd-compressed FASTQ file for scoring.", ".fastq.zst"),
        ("Participants must upload a gzip-compressed PDB file for scoring.", ".pdb.gz"),
        ("Participants must upload a gzip-compressed SMILES file for scoring.", ".smiles.gz"),
        ("Participants must upload a zstd-compressed SELFIES file for scoring.", ".selfies.zst"),
        ("Participants must upload a bzip2-compressed GraphML file for scoring.", ".graphml.bz2"),
        ("Participants must upload an xz-compressed edge list file for scoring.", ".edgelist.xz"),
        ("Participants must upload a zstd-compressed JSON-LD linked data file for scoring.", ".jsonld.zst"),
        ("Participants must upload a gzip-compressed Turtle RDF file for scoring.", ".ttl.gz"),
        ("Participants must upload a bzip2-compressed N-Triples file for scoring.", ".nt.bz2"),
        ("Participants must upload an xz-compressed N-Quads file for scoring.", ".nq.xz"),
        ("Participants must upload a gzip-compressed RDF XML file for scoring.", ".rdf.gz"),
        ("Participants must upload a zstd-compressed OWL ontology file for scoring.", ".owl.zst"),
        ("Participants must upload a gzip-compressed TriG RDF file for scoring.", ".trig.gz"),
        ("Participants must upload a gzip-compressed MRC cryo-EM map for scoring.", ".mrc.gz"),
        ("Participants must upload a zstd-compressed CCP4 map for scoring.", ".ccp4.zst"),
        ("Participants must upload a zstd-compressed BrainVision header for scoring.", ".vhdr.zst"),
        ("Participants must upload a gzip-compressed MNE FIF file for scoring.", ".fif.gz"),
        ("Participants must upload a gzip-compressed VCF variant file for scoring.", ".vcf.gz"),
        ("Participants must upload a zstd-compressed GFF3 annotation file for scoring.", ".gff3.zst"),
        ("Participants must upload an xz-compressed HGT elevation raster for scoring.", ".hgt.xz"),
        ("Participants must upload a gzip-compressed VTK PolyData file for scoring.", ".vtp.gz"),
        ("Participants must upload a gzip-compressed OFF mesh file for scoring.", ".off.gz"),
        ("Participants must upload a zstd-compressed PLY point cloud file for scoring.", ".ply.zst"),
        ("Participants must upload a gzip-compressed KML file for scoring.", ".kml.gz"),
        ("Participants must upload a zstd-compressed TopoJSON file for scoring.", ".topojson.zst"),
        ("Participants must upload a gzip-compressed GeoJSON text sequence for scoring.", ".geojsonseq.gz"),
        ("Participants must upload an xz-compressed OpenStreetMap XML file for scoring.", ".osm.xz"),
        ("Participants must upload a gzip-compressed FITS file for scoring.", ".fits.gz"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], description
        assert hint.artifact_class == "single_file", description


def test_parse_submission_format_reads_explicit_safetensors_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `submission.safetensors` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".safetensors"]
    assert hint.artifact_class == "single_file"


def test_parse_submission_format_reads_explicit_safetensors_index_suffix() -> None:
    markdown = "## Submission Format\n\nUpload `model.safetensors.index.json` for scoring.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".safetensors.index.json"]
    assert hint.artifact_class == "single_file"


def test_parse_submission_format_infers_model_index_suffix_from_prose() -> None:
    cases = [
        ("Upload a safetensors index JSON file for scoring.", ".safetensors.index.json"),
        ("Upload a PyTorch model bin index JSON file for scoring.", ".bin.index.json"),
        ("Upload a TensorFlow checkpoint index file for scoring.", ".ckpt.index"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], description
        assert hint.artifact_class == "single_file", description


def test_parse_submission_format_infers_model_directory_suffix_from_prose() -> None:
    cases = [
        ("Upload a TensorFlow SavedModel directory for scoring.", ".savedmodel"),
        ("Upload a Hugging Face model directory for scoring.", ".hfmodel"),
        ("Upload an MLflow model directory for scoring.", ".mlflowmodel"),
        ("Upload a Core ML package for scoring.", ".mlpackage"),
        ("Upload a compiled Core ML model package for scoring.", ".mlmodelc"),
        ("Upload a TensorFlow checkpoint directory for scoring.", ".tfcheckpoint"),
    ]
    for description, suffix in cases:
        markdown = f"## Submission Format\n\n{description}\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix], description
        assert hint.artifact_class == "single_file", description


def test_parse_submission_format_reads_explicit_additional_single_file_artifact_suffixes() -> None:
    suffixes = [
        ".gguf",
        ".msgpack",
        ".tflite",
        ".pb",
        ".joblib",
        ".pdf",
        ".docx",
        ".html",
        ".geojson",
        ".gpkg",
        ".pdb",
        ".mmcif",
        ".sdf",
        ".mol2",
        ".fasta",
        ".fastq",
        ".graphml",
        ".gexf",
        ".gml",
        ".mtx",
        ".edgelist",
        ".edges",
        ".nc",
        ".cdf",
        ".grib",
        ".grib2",
        ".grb",
        ".fits",
        ".fit",
        ".fts",
        ".h5ad",
        ".loom",
        ".zarr",
    ]
    for suffix in suffixes:
        markdown = f"## Submission Format\n\nUpload `submission{suffix}` for scoring.\n"

        hint = parse_submission_format(markdown)

        assert hint.expected_suffixes == [suffix]
        assert hint.artifact_class == "single_file"


def test_parse_submission_format_reads_excel_prose() -> None:
    markdown = "## Submission Format\n\nParticipants must submit an Excel file with columns id,target.\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".xlsx"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_reads_gzipped_csv_prose() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Participants are required to submit a gzip-compressed CSV file with columns id,target.\n"
    )

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".csv.gz"]
    assert hint.artifact_class == "tabular"


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Participants must submit comma-separated values with columns id,target.", ".csv"),
        ("Participants must submit a comma delimited file with columns id,target.", ".csv"),
        ("Participants must submit semicolon-separated values with columns id,target.", ".csv"),
        ("Participants must submit a semicolon delimited file with columns id,target.", ".csv"),
    ],
)
def test_parse_submission_format_infers_csv_from_delimited_tabular_prose(
    description: str,
    expected: str,
) -> None:
    markdown = f"## Submission Format\n\n{description}\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [expected]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_prefers_submission_section_over_explanatory_table() -> None:
    markdown = (
        "## Description\n\n"
        "| Column | Meaning |\n"
        "| --- | --- |\n"
        "| KEEP | Primary codes |\n"
        "| ASSOCIATION | Related codes |\n\n"
        "## Submission Format\n\n"
        "Each submission must include the columns KEEP, ASSOCIATION, and DIFF.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.columns == ["KEEP", "ASSOCIATION", "DIFF"]


def test_parse_submission_format_treats_weights_and_inference_script_as_bundle() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Submit a ZIP archive containing model weights (.pt or .pth) and the inference script.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes == [".zip"]
    assert hint.artifact_class == "bundle"


@pytest.mark.parametrize("suffix", [".keras", ".mlmodelc", ".onnx", ".tflite"])
def test_parse_submission_format_treats_model_archive_as_bundle(suffix: str) -> None:
    markdown = (
        "## Submission Format\n\n"
        f"Submit a ZIP archive containing a trained model `{suffix}` and the inference script.\n"
    )

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [".zip"]
    assert hint.artifact_class == "bundle"


def test_parse_submission_format_treats_tar_gz_weights_as_bundle() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Submit a submission.tar.gz archive containing model weights (.pt) and the inference script.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes == [".tar.gz"]
    assert hint.artifact_class == "bundle"
    assert hint.artifact_container == "tar"


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Submit a gzipped tarball containing model weights and the inference script.", ".tar.gz"),
        ("Submit a bzip2-compressed tarball containing model weights and the inference script.", ".tar.bz2"),
        ("Submit an xz-compressed tar archive containing model weights and the inference script.", ".tar.xz"),
        ("Submit a zstd-compressed tarball containing model weights and the inference script.", ".tar.zst"),
        ("Submit a tarball containing model weights and the inference script.", ".tar"),
    ],
)
def test_parse_submission_format_infers_tarball_archive_aliases(
    description: str,
    suffix: str,
) -> None:
    markdown = f"## Submission Format\n\n{description}\n"

    hint = parse_submission_format(markdown)

    assert hint.expected_suffixes == [suffix]
    assert hint.artifact_class == "bundle"
    assert hint.artifact_container == "tar"


def test_parse_submission_format_detects_rar_archive() -> None:
    markdown = (
        "## Submission Format\n\nUpload a single `submission.rar` archive containing the required prediction files.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes == [".rar"]
    assert hint.artifact_class == "single_file"
    assert hint.artifact_container == "rar"


def test_parse_submission_format_ignores_topology_json_noise() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Include the topology JSON description in your documentation.\n"
        "Submit predictions through the official submission channel.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes is None
    assert hint.artifact_class == "unknown"
