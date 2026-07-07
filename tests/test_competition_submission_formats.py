from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kagglebot.competition_submission_formats import (
    CompetitionListing,
    CrawlRecord,
    crawl_competition_submission_format,
    crawl_submission_formats,
    discover_competitions,
    extract_competition_slugs_from_html,
    find_submission_text_block,
    html_to_text,
    infer_submission_mode,
)


class _FakeFetcher:
    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def fetch(self, url: str) -> str:
        return self._pages[url]


class _FakeApi:
    def __init__(self, responses: dict[tuple[str | None, str | None, int, str | None], list[SimpleNamespace]]) -> None:
        self._responses = responses

    def authenticate(self) -> None:
        return None

    def competitions_list(
        self,
        *,
        group: str | None = None,
        category: str | None = None,
        sort_by: str | None = None,
        page: int = 1,
        search: str | None = None,
    ) -> list[SimpleNamespace]:
        return list(self._responses.get((group, category, page, search), []))


def test_extract_competition_slugs_from_landing_html() -> None:
    html = """
    <div>
      <a href="/competitions/titanic">Titanic</a>
      <a href="/competitions/house-prices-advanced-regression-techniques">House Prices</a>
      <a href="/competitions/titanic/overview">Ignore section link</a>
    </div>
    """
    assert extract_competition_slugs_from_html(html) == [
        "house-prices-advanced-regression-techniques",
        "titanic",
    ]


def test_html_to_text_keeps_submission_section_structure() -> None:
    html = """
    <div>
      <h2>Submission File Format:</h2>
      <p>You should submit a csv file.</p>
      <ul>
        <li><code>PassengerId</code></li>
        <li><code>Survived</code></li>
      </ul>
    </div>
    """
    text = html_to_text(html)
    assert "Submission File Format:" in text
    assert "You should submit a csv file." in text
    assert "PassengerId" in text
    assert "Survived" in text


def test_crawl_competition_submission_format_parses_csv_submission() -> None:
    listing = CompetitionListing(
        slug="titanic",
        title="Titanic",
        url="https://www.kaggle.com/competitions/titanic",
        description="",
        category="Getting Started",
        reward="Knowledge",
        evaluation_metric="Accuracy",
        team_count=100,
        max_daily_submissions=10,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission File Format:</h2>
      <p>You should submit a csv file with exactly 418 entries plus a header row.</p>
      <ul>
        <li><code>PassengerId</code></li>
        <li><code>Survived</code></li>
      </ul>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/titanic/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == ".csv file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [".csv"]
    assert record.detected_columns == ["PassengerId", "Survived"]


def test_crawl_competition_submission_format_parses_zstd_csv_submission() -> None:
    listing = CompetitionListing(
        slug="demo-zstd",
        title="Demo Zstd",
        url="https://www.kaggle.com/competitions/demo-zstd",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="LogLoss",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission File Format</h2>
      <p>Upload <code>submission.csv.zst</code> with columns id,target.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/demo-zstd/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [".csv.zst"]


def test_crawl_competition_submission_format_parses_feather_submission() -> None:
    listing = CompetitionListing(
        slug="demo-feather",
        title="Demo Feather",
        url="https://www.kaggle.com/competitions/demo-feather",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="RMSE",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission File Format</h2>
      <p>Upload <code>submission.feather</code> with columns id,target.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/demo-feather/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [".feather"]


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".jsonl", ".jsonl"),
        (".jsonl.zst", ".jsonl.zst"),
        (".jsonlines", ".jsonlines"),
        (".jsonlines.zst", ".jsonlines.zst"),
        (".ndjson", ".ndjson"),
    ],
)
def test_crawl_competition_submission_format_classifies_json_lines_submission_as_tabular(
    suffix: str,
    expected: str,
) -> None:
    listing = CompetitionListing(
        slug="demo-json-lines",
        title="Demo JSON Lines",
        url="https://www.kaggle.com/competitions/demo-json-lines",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="LogLoss",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission File Format</h2>
      <p>Upload <code>submission{suffix}</code> for scoring.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/demo-json-lines/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == f"{expected} file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [expected]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Upload a gzip-compressed NDJSON file for scoring.", ".ndjson.gz"),
        ("Upload a bzip2-compressed NDJSON file for scoring.", ".ndjson.bz2"),
        ("Upload an xz-compressed JSONLines file for scoring.", ".jsonlines.xz"),
        ("Upload a zstd-compressed JSON Lines file for scoring.", ".jsonl.zst"),
    ],
)
def test_crawl_competition_submission_format_detects_compressed_json_lines_keyword(
    description: str,
    expected: str,
) -> None:
    listing = CompetitionListing(
        slug="demo-compressed-json-lines",
        title="Demo Compressed JSON Lines",
        url="https://www.kaggle.com/competitions/demo-compressed-json-lines",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="LogLoss",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission File Format</h2>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher(
            {"https://www.kaggle.com/competitions/demo-compressed-json-lines/overview": overview_html}
        ),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == f"{expected} file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [expected]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Upload a gzip-compressed YAML file for scoring.", ".yaml.gz"),
        ("Upload a zstd-compressed XML file for scoring.", ".xml.zst"),
        ("Upload a bzip2-compressed HTML file with columns id,target for scoring.", ".html.bz2"),
        ("Upload an xz-compressed PSV file for scoring.", ".psv.xz"),
        ("Upload a zstd-compressed TAB file for scoring.", ".tab.zst"),
    ],
)
def test_crawl_competition_submission_format_detects_compressed_structured_tabular_keyword(
    description: str,
    expected: str,
) -> None:
    listing = CompetitionListing(
        slug="demo-compressed-structured-tabular",
        title="Demo Compressed Structured Tabular",
        url="https://www.kaggle.com/competitions/demo-compressed-structured-tabular",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="LogLoss",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission File Format</h2>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher(
            {"https://www.kaggle.com/competitions/demo-compressed-structured-tabular/overview": overview_html}
        ),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == f"{expected} file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [expected]


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Participants must upload a NIfTI file for each case for scoring.", ".nii.gz"),
        ("Participants must upload a file named submission.svs for scoring.", ".svs"),
        ("Participants must upload a file named submission.ome.tif for scoring.", ".ome.tif"),
        ("Participants must upload a JPEG XL image file for scoring.", ".jxl"),
        ("Participants must upload a HEIC image file for scoring.", ".heic"),
        ("Participants must upload an OpenEXR image file for scoring.", ".exr"),
        ("Participants must upload an EPUB document for scoring.", ".epub"),
        ("Participants must upload a PowerPoint file for scoring.", ".pptx"),
        ("Participants must upload a Scalable Vector Graphics file for scoring.", ".svg"),
        ("Participants must upload a gzip-compressed SVG file for scoring.", ".svg.gz"),
        ("Participants must upload a MIDI audio file for scoring.", ".mid"),
        ("Participants must upload an OPUS audio file for scoring.", ".opus"),
        ("Participants must upload an MPEG video file for scoring.", ".mpg"),
        ("Participants must upload an M4V video file for scoring.", ".m4v"),
        ("Participants must upload an E57 point cloud file for scoring.", ".e57"),
        ("Participants must upload an OFF mesh file for scoring.", ".off"),
        ("Participants must upload a MATLAB file for scoring.", ".mat"),
        ("Participants must upload a SMILES file for scoring.", ".smiles"),
        ("Participants must upload an InChI file for scoring.", ".inchi"),
        ("Participants must upload COCO annotations for scoring.", ".json"),
        ("Participants must upload COCO keypoints JSON for scoring.", ".json"),
        ("Participants must upload YOLO labels for scoring.", ".txt"),
        ("Participants must upload YOLOv5 labels for scoring.", ".txt"),
        ("Participants must upload a SQLite database for scoring.", ".sqlite"),
        ("Participants must upload a SQLite3 database for scoring.", ".sqlite3"),
        ("Participants must upload a European Data Format signal file for scoring.", ".edf"),
        ("Participants must upload a WFDB header file for scoring.", ".hea"),
        ("Participants must upload a Neurodata Without Borders file for scoring.", ".nwb"),
    ],
)
def test_crawl_competition_submission_format_detects_non_tabular_keyword_without_heading(
    description: str,
    suffix: str,
) -> None:
    listing = CompetitionListing(
        slug="medical-volume-demo",
        title="Medical Volume Demo",
        url="https://www.kaggle.com/competitions/medical-volume-demo",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Dice",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/medical-volume-demo/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == f"{suffix} single file"
    assert record.artifact_class == "single_file"
    assert record.detected_extensions == [suffix]


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Participants must upload an ORC file for scoring.", ".orc"),
        ("Participants must upload an HDF5 file for scoring.", ".hdf5"),
        ("Participants must upload Pascal VOC annotations for scoring.", ".xml"),
        ("Participants must upload Pascal VOC XML files for scoring.", ".xml"),
        ("Participants must upload Open Images annotations for scoring.", ".csv"),
        ("Participants must upload Open Images CSV files for scoring.", ".csv"),
        ("Participants must upload RLE masks for scoring.", ".csv"),
        ("Participants must upload run length encoding masks for scoring.", ".csv"),
    ],
)
def test_crawl_competition_submission_format_detects_tabular_keyword_without_heading(
    description: str,
    suffix: str,
) -> None:
    listing = CompetitionListing(
        slug="structured-table-demo",
        title="Structured Table Demo",
        url="https://www.kaggle.com/competitions/structured-table-demo",
        description="",
        category="Research",
        reward="",
        evaluation_metric="LogLoss",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/structured-table-demo/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == f"{suffix} file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [suffix]


def test_crawl_competition_submission_format_detects_non_tabular_direct_extension_without_heading() -> None:
    listing = CompetitionListing(
        slug="model-artifact-demo",
        title="Model Artifact Demo",
        url="https://www.kaggle.com/competitions/model-artifact-demo",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Runtime",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <p>The final submission should upload <code>submission.onnx</code> for scoring.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/model-artifact-demo/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == ".onnx single file"
    assert record.artifact_class == "single_file"
    assert record.detected_extensions == [".onnx"]


def test_crawl_competition_submission_format_detects_single_file_artifact_direct_extensions() -> None:
    suffixes = [
        ".safetensors",
        ".xgb",
        ".cbm",
        ".gguf",
        ".msgpack",
        ".tflite",
        ".pb",
        ".joblib",
        ".pdf",
        ".md.gz",
        ".vtt.zst",
        ".geojson",
        ".gpkg",
        ".pdb",
        ".mmcif",
        ".sdf",
        ".fasta",
        ".graphml",
        ".gexf",
        ".edgelist",
        ".nc",
        ".grib2",
        ".fits",
        ".h5ad",
        ".loom",
        ".zarr",
        ".czi",
        ".mrxs",
        ".qptiff",
        ".avif",
        ".heic",
        ".heif",
        ".opus",
        ".aiff",
        ".m4v",
        ".wmv",
        ".e57",
        ".xyz",
        ".pts",
        ".ptx",
        ".off",
    ]
    for suffix in suffixes:
        slug = f"{suffix.lstrip('.')}-artifact-demo"
        listing = CompetitionListing(
            slug=slug,
            title="Model Artifact Demo",
            url=f"https://www.kaggle.com/competitions/{slug}",
            description="",
            category="Research",
            reward="",
            evaluation_metric="Runtime",
            team_count=100,
            max_daily_submissions=5,
            is_kernels_submissions_only=False,
            submissions_disabled=False,
            source="test",
        )
        overview_html = f"""
        <html><body>
          <p>The final submission should upload <code>submission{suffix}</code> for scoring.</p>
        </body></html>
        """

        record = crawl_competition_submission_format(
            listing=listing,
            fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
            fetch_rules_page=False,
        )

        assert record.submission_mode == "direct_file_upload"
        assert record.required_artifact == f"{suffix} single file"
        assert record.artifact_class == "single_file"
        assert record.detected_extensions == [suffix]


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Participants must upload a gzip-compressed FASTA sequence file for scoring.", ".fasta.gz"),
        ("Participants must upload a bzip2-compressed GraphML file for scoring.", ".graphml.bz2"),
        ("Participants must upload a zstd-compressed PLY point cloud file for scoring.", ".ply.zst"),
    ],
)
def test_crawl_competition_submission_format_detects_compressed_text_like_artifact_prose(
    description: str,
    suffix: str,
) -> None:
    slug = "compressed-text-asset-demo"
    listing = CompetitionListing(
        slug=slug,
        title="Compressed Text Asset Demo",
        url=f"https://www.kaggle.com/competitions/{slug}",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Runtime",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission Format:</h2>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == f"{suffix} single file"
    assert record.artifact_class == "single_file"
    assert record.detected_extensions == [suffix]


@pytest.mark.parametrize(
    "description",
    [
        "Participants must upload a NumPy archive file for scoring.",
        "Participants must upload a compressed NumPy array for scoring.",
        "Participants must upload a SciPy sparse matrix archive for scoring.",
    ],
)
def test_crawl_competition_submission_format_detects_numpy_archive_prose(
    description: str,
) -> None:
    slug = "numpy-archive-demo"
    listing = CompetitionListing(
        slug=slug,
        title="NumPy Archive Demo",
        url=f"https://www.kaggle.com/competitions/{slug}",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Runtime",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission Format:</h2>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == ".npz single file"
    assert record.artifact_class == "single_file"
    assert record.detected_extensions == [".npz"]


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Participants must upload an XGBoost model file for scoring.", ".xgb"),
        ("Participants must upload a CatBoost model file for scoring.", ".cbm"),
        ("Participants must upload a pickle model file for scoring.", ".pkl"),
        ("Participants must upload a Python script file for scoring.", ".py"),
        ("Participants must upload an R script file for scoring.", ".r"),
        ("Participants must upload a Julia script file for scoring.", ".jl"),
    ],
)
def test_crawl_competition_submission_format_detects_framework_model_artifact_prose(
    description: str,
    suffix: str,
) -> None:
    slug = "framework-model-demo"
    listing = CompetitionListing(
        slug=slug,
        title="Framework Model Demo",
        url=f"https://www.kaggle.com/competitions/{slug}",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Runtime",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission Format:</h2>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == f"{suffix} single file"
    assert record.artifact_class == "single_file"
    assert record.detected_extensions == [suffix]


def test_crawl_competition_submission_format_detects_hdf5_model_artifact_prose() -> None:
    slug = "hdf5-model-demo"
    listing = CompetitionListing(
        slug=slug,
        title="HDF5 Model Demo",
        url=f"https://www.kaggle.com/competitions/{slug}",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Runtime",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>Participants must upload an HDF5 model file for scoring.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == ".h5 single file"
    assert record.artifact_class == "single_file"
    assert record.detected_extensions == [".h5", ".hdf5"]


def test_crawl_competition_submission_format_filters_topology_json_false_positive() -> None:
    listing = CompetitionListing(
        slug="quantum-quest",
        title="Quantum Quest",
        url="https://www.kaggle.com/competitions/quantum-quest",
        description="",
        category="Research",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>
        Provide the topology JSON in the writeup and package model weights (.pt)
        plus the inference script in a ZIP archive.
      </p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/quantum-quest/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "bundle"
    assert record.required_artifact == "ZIP bundle containing model assets and inference code"
    assert ".json" not in record.detected_extensions


def test_crawl_competition_submission_format_filters_json_metadata_false_positive() -> None:
    listing = CompetitionListing(
        slug="metadata-bundle",
        title="Metadata Bundle",
        url="https://www.kaggle.com/competitions/metadata-bundle",
        description="",
        category="Research",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>
        Include JSON metadata in your methods note, but upload a ZIP archive
        containing model weights and the inference script.
      </p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/metadata-bundle/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "bundle"
    assert record.required_artifact == "ZIP bundle containing model assets and inference code"
    assert record.detected_extensions == [".zip"]


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Upload a safetensors index JSON file for scoring.", ".safetensors.index.json"),
        ("Upload a PyTorch model bin index JSON file for scoring.", ".bin.index.json"),
        ("Upload a TensorFlow checkpoint index file for scoring.", ".ckpt.index"),
    ],
)
def test_crawl_competition_submission_format_detects_model_index_suffix_from_prose(
    description: str,
    suffix: str,
) -> None:
    slug = "model-index-demo"
    listing = CompetitionListing(
        slug=slug,
        title="Model Index Demo",
        url=f"https://www.kaggle.com/competitions/{slug}",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Runtime",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission Format:</h2>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.detected_extensions == [suffix]
    assert record.required_artifact == f"{suffix} single file"
    assert record.artifact_class == "single_file"


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Upload a TensorFlow SavedModel directory for scoring.", ".savedmodel"),
        ("Upload a Hugging Face model directory for scoring.", ".hfmodel"),
        ("Upload an MLflow model directory for scoring.", ".mlflowmodel"),
        ("Upload a Core ML package for scoring.", ".mlpackage"),
        ("Upload a compiled Core ML model package for scoring.", ".mlmodelc"),
        ("Upload a TensorFlow checkpoint directory for scoring.", ".tfcheckpoint"),
    ],
)
def test_crawl_competition_submission_format_detects_model_directory_suffix_from_prose(
    description: str,
    suffix: str,
) -> None:
    slug = "model-directory-demo"
    listing = CompetitionListing(
        slug=slug,
        title="Model Directory Demo",
        url=f"https://www.kaggle.com/competitions/{slug}",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Runtime",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission Format:</h2>
      <p>{description}</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.detected_extensions == [suffix]
    assert record.required_artifact == f"{suffix} single file"
    assert record.artifact_class == "single_file"


@pytest.mark.parametrize("suffix", [".keras", ".mlmodelc", ".onnx", ".tflite"])
def test_crawl_competition_submission_format_describes_model_suffix_archive_as_bundle(suffix: str) -> None:
    listing = CompetitionListing(
        slug="model-bundle-demo",
        title="Model Bundle Demo",
        url="https://www.kaggle.com/competitions/model-bundle-demo",
        description="",
        category="Research",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = f"""
    <html><body>
      <h2>Submission Format:</h2>
      <p>Submit a ZIP archive containing a trained model ({suffix}) and the inference script.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/model-bundle-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "bundle"
    assert record.required_artifact == "ZIP bundle containing model assets and inference code"


def test_crawl_competition_submission_format_describes_tar_model_bundle() -> None:
    listing = CompetitionListing(
        slug="tar-bundle-demo",
        title="Tar Bundle Demo",
        url="https://www.kaggle.com/competitions/tar-bundle-demo",
        description="",
        category="Research",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>Submit a submission.tar.xz archive containing model weights (.pt) and the inference script.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/tar-bundle-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "bundle"
    assert record.artifact_container == "tar"
    assert record.required_artifact == "TAR bundle containing model assets and inference code"
    assert record.detected_extensions == [".tar.xz"]


def test_crawl_competition_submission_format_describes_tar_zst_model_bundle() -> None:
    listing = CompetitionListing(
        slug="tar-zst-bundle-demo",
        title="Tar Zstd Bundle Demo",
        url="https://www.kaggle.com/competitions/tar-zst-bundle-demo",
        description="",
        category="Research",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>Submit a submission.tar.zst archive containing model weights (.pt) and the inference script.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/tar-zst-bundle-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "bundle"
    assert record.artifact_container == "tar"
    assert record.required_artifact == "TAR bundle containing model assets and inference code"
    assert record.detected_extensions == [".tar.zst"]


def test_crawl_competition_submission_format_describes_zstd_tarball_model_bundle() -> None:
    listing = CompetitionListing(
        slug="zstd-tarball-bundle-demo",
        title="Zstd Tarball Bundle Demo",
        url="https://www.kaggle.com/competitions/zstd-tarball-bundle-demo",
        description="",
        category="Research",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>Submit a zstd-compressed tarball containing model weights (.pt) and the inference script.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/zstd-tarball-bundle-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "bundle"
    assert record.artifact_container == "tar"
    assert record.required_artifact == "TAR bundle containing model assets and inference code"
    assert record.detected_extensions == [".tar.zst"]


def test_crawl_competition_submission_format_parses_excel_submission() -> None:
    listing = CompetitionListing(
        slug="spreadsheet-demo",
        title="Spreadsheet Demo",
        url="https://www.kaggle.com/competitions/spreadsheet-demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>You must upload `submission.xlsx` with columns id,target.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/spreadsheet-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == ".xlsx file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [".xlsx"]


def test_crawl_competition_submission_format_detects_compressed_pickle_tabular_submission() -> None:
    listing = CompetitionListing(
        slug="pickle-demo",
        title="Pickle Demo",
        url="https://www.kaggle.com/competitions/pickle-demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>You must upload `submission.pkl.zst` for scoring.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/pickle-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.required_artifact == ".pkl.zst file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [".pkl.zst"]


def test_crawl_competition_submission_format_detects_pickle_xz_extension_from_text() -> None:
    listing = CompetitionListing(
        slug="pickle-xz-demo",
        title="Pickle XZ Demo",
        url="https://www.kaggle.com/competitions/pickle-xz-demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>The required file is named submission.pickle.xz.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/pickle-xz-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.required_artifact == ".pickle.xz file"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [".pickle.xz"]


def test_crawl_competition_submission_format_detects_multi_file_zip() -> None:
    listing = CompetitionListing(
        slug="vesuvius-demo",
        title="Vesuvius Demo",
        url="https://www.kaggle.com/competitions/vesuvius-demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission File Format:</h2>
      <p>You must submit a ZIP file containing one .tif mask per test fragment.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/vesuvius-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.artifact_class == "multi_file_zip"
    assert record.artifact_container == "zip"
    assert record.required_artifact == "ZIP archive containing multiple prediction files"


def test_crawl_competition_submission_format_detects_tar_archive() -> None:
    listing = CompetitionListing(
        slug="archive-demo",
        title="Archive Demo",
        url="https://www.kaggle.com/competitions/archive-demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>You must upload `submission.tar.xz` for scoring.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/archive-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.required_artifact == "TAR archive"
    assert record.artifact_class == "single_file"
    assert record.artifact_container == "tar"
    assert record.detected_extensions == [".tar.xz"]


def test_crawl_competition_submission_format_detects_plain_tar_archive() -> None:
    listing = CompetitionListing(
        slug="archive-demo",
        title="Archive Demo",
        url="https://www.kaggle.com/competitions/archive-demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>You must upload `submission.tar` for scoring.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/archive-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.required_artifact == "TAR archive"
    assert record.artifact_class == "single_file"
    assert record.artifact_container == "tar"
    assert record.detected_extensions == [".tar"]


def test_crawl_competition_submission_format_detects_7z_archive() -> None:
    listing = CompetitionListing(
        slug="archive-demo",
        title="Archive Demo",
        url="https://www.kaggle.com/competitions/archive-demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Format:</h2>
      <p>You must upload `submission.7z` for scoring.</p>
    </body></html>
    """
    fetcher = _FakeFetcher(
        {
            "https://www.kaggle.com/competitions/archive-demo/overview": overview_html,
        }
    )

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=fetcher,  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.required_artifact == "7Z archive"
    assert record.artifact_class == "single_file"
    assert record.artifact_container == "7z"
    assert record.detected_extensions == [".7z"]


def test_find_submission_text_block_returns_empty_without_submission_markers() -> None:
    text = """
    Competition Overview

    This challenge predicts survival on the Titanic.

    Evaluation

    Accuracy is used for ranking.
    """
    assert find_submission_text_block(text) == ""


def test_find_submission_text_block_prefers_submission_file_format_over_generic_submit_copy() -> None:
    text = """
    Make a Submission
    Upload your prediction as a submission on Kaggle and receive an accuracy score.

    Submission File Format:
    You should submit a csv file with exactly 418 entries plus a header row.
    - PassengerId
    - Survived
    """
    section = find_submission_text_block(text)
    assert "Submission File Format" in section
    assert "csv file" in section


def test_infer_submission_mode_prefers_code_competition_flag() -> None:
    listing = CompetitionListing(
        slug="code-comp",
        title="Code Comp",
        url="https://www.kaggle.com/competitions/code-comp",
        description="",
        category="Featured",
        reward="$10,000",
        evaluation_metric="Score",
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=True,
        submissions_disabled=False,
        source="test",
    )
    assert infer_submission_mode(listing, "Submit your notebook output to Kaggle.", []) == (
        "code_competition_runtime_submission"
    )


def test_crawl_submission_formats_writes_summary_json(monkeypatch, tmp_path) -> None:
    listing = CompetitionListing(
        slug="demo",
        title="Demo",
        url="https://www.kaggle.com/competitions/demo",
        description="",
        category="Featured",
        reward="",
        evaluation_metric="Accuracy",
        team_count=10,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    record = CrawlRecord(
        slug="demo",
        title="Demo",
        competition_type="Featured",
        submission_mode="direct_file_upload",
        required_artifact=".csv file",
        artifact_class="tabular",
        artifact_container=None,
        raw_format_text="Submission File Format: csv",
        detected_extensions=[".csv"],
        detected_columns=["id", "target"],
        delimiter=",",
        is_code_competition=False,
        evidence_url="https://www.kaggle.com/competitions/demo/overview",
        evidence_html_snippet="",
        extraction_confidence="high",
        reward="",
        evaluation_metric="Accuracy",
        team_count=10,
        max_daily_submissions=5,
        discovery_source="test",
        crawled_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "kagglebot.competition_submission_formats.discover_competitions",
        lambda **_kwargs: [listing],
    )
    monkeypatch.setattr(
        "kagglebot.competition_submission_formats.crawl_competition_submission_format",
        lambda **_kwargs: record,
    )

    summary = crawl_submission_formats(
        output_dir=tmp_path,
        fetcher=_FakeFetcher({}),  # type: ignore[arg-type]
        max_competitions=1,
        resume=False,
    )

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["competition_count"] == 1
    assert saved["submission_modes"] == {"direct_file_upload": 1}
    assert saved["artifact_classes"] == {"tabular": 1}
    assert saved["extensions"] == {".csv": 1}


def test_discover_competitions_merges_landing_and_api_results(monkeypatch) -> None:
    landing_html = """
    <div>
      <a href="/competitions/titanic"><div>Titanic</div></a>
    </div>
    """
    fetcher = _FakeFetcher({"https://www.kaggle.com/competitions": landing_html})
    fake_api = _FakeApi(
        {
            ("general", None, 1, None): [
                SimpleNamespace(
                    url="https://www.kaggle.com/competitions/titanic",
                    ref="https://www.kaggle.com/competitions/titanic",
                    title="Titanic",
                    description="desc",
                    category="Getting Started",
                    reward="Knowledge",
                    evaluation_metric="Accuracy",
                    team_count=123,
                    max_daily_submissions=10,
                    is_kernels_submissions_only=False,
                    submissions_disabled=False,
                )
            ],
            ("community", None, 1, None): [],
            ("entered", None, 1, None): [],
            (None, "all", 1, None): [],
            (None, "featured", 1, None): [],
            (None, "research", 1, None): [],
            (None, "gettingStarted", 1, None): [],
            (None, "playground", 1, None): [],
            (None, "all", 1, "a"): [],
            (None, "all", 1, "b"): [],
            (None, "all", 1, "c"): [],
        }
    )
    monkeypatch.setattr("kagglebot.competition_submission_formats.KaggleApi", lambda: fake_api)

    listings = discover_competitions(
        fetcher=fetcher,  # type: ignore[arg-type]
        max_prefix_depth=1,
        max_pages_per_search=1,
        search_alphabet="abc",
    )

    assert len(listings) == 1
    assert listings[0].slug == "titanic"
    assert listings[0].team_count == 123
    assert listings[0].category == "Getting Started"
