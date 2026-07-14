from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kagglebot.competition_submission_formats import (
    CompetitionListing,
    CrawlRecord,
    assess_submission_format_support,
    crawl_competition_submission_format,
    crawl_submission_formats,
    discover_competitions,
    discover_entered_competitions,
    enrich_records_with_kaggle_evidence,
    extract_competition_slugs_from_html,
    find_submission_text_block,
    html_to_text,
    infer_submission_mode,
    infer_suffixes_from_keywords,
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


def test_submission_suffix_keywords_ignore_common_prose_tokens() -> None:
    text = "The test set closes at 11:59 PM PT. Winners receive $1,000 USD. See the debugging doc."

    assert infer_suffixes_from_keywords(text) == []


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


def test_crawl_competition_submission_format_infers_csv_from_header_and_rows() -> None:
    listing = CompetitionListing(
        slug="csv-example",
        title="CSV Example",
        url="https://www.kaggle.com/competitions/csv-example",
        description="",
        category="Playground",
        reward="",
        evaluation_metric="AUC",
        team_count=10,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <h2>Submission File</h2>
    <p>All computations used to generate the submission file must be offline.</p>
    <h2>Submission Format</h2>
    <p>link</p>
    <p>keyboard_arrow_up</p>
    <p>The file should contain a header and have the following format:</p>
    <pre>id,Heart Disease
    630000,0.2
    630001,0.3</pre>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/csv-example/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.detected_extensions == [".csv"]
    assert record.detected_columns == ["id", "Heart Disease"]


def test_crawl_competition_submission_format_prefers_named_outer_submission_file() -> None:
    listing = CompetitionListing(
        slug="named-output",
        title="Named Output",
        url="https://www.kaggle.com/competitions/named-output",
        description="",
        category="Research",
        reward="",
        evaluation_metric="Score",
        team_count=10,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <h2>Submission Format</h2>
    <p>Your notebook must output a submission.csv file.</p>
    <p>Run predict.py and save predictions to submission.csv.</p>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/named-output/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.detected_extensions == [".csv"]
    assert record.artifact_class == "tabular"


def test_crawl_competition_submission_format_classifies_external_pull_request() -> None:
    listing = CompetitionListing(
        slug="external-pr",
        title="External PR",
        url="https://www.kaggle.com/competitions/external-pr",
        description="",
        category="Community",
        reward="$1,000",
        evaluation_metric="",
        team_count=1,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <h2>Submission Results</h2>
    <p>Participants must submit their source code by initiating a Pull Request (PR) to the repository.</p>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/external-pr/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "external_repository_submission"
    assert assess_submission_format_support(record) == (False, "external_submission_requires_manual_workflow")


def test_crawl_competition_submission_format_classifies_writeup_rubric() -> None:
    listing = CompetitionListing(
        slug="judged-challenge",
        title="Judged Challenge",
        url="https://www.kaggle.com/competitions/judged-challenge",
        description="",
        category="Community",
        reward="$10,000",
        evaluation_metric="",
        team_count=10,
        max_daily_submissions=1,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Evaluation</h2>
      <p>All submissions are judged based on the following rubric:</p>
      <ul><li>Technical quality (40)</li><li>Impact (60)</li></ul>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/judged-challenge/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "writeup_submission"
    assert record.required_artifact == "Kaggle Writeup"
    assert record.artifact_class == "writeup"
    assert record.extraction_confidence == "high"
    assert "following rubric" in record.raw_format_text


def test_crawl_competition_submission_format_prefers_code_runtime_over_writeup_wording() -> None:
    listing = CompetitionListing(
        slug="code-challenge",
        title="Code Challenge",
        url="https://www.kaggle.com/competitions/code-challenge",
        description="",
        category="Featured",
        reward="$10,000",
        evaluation_metric="Score",
        team_count=10,
        max_daily_submissions=1,
        is_kernels_submissions_only=True,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <p>Submissions must be made through Notebooks and are judged with the following rubric.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/code-challenge/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "code_competition_runtime_submission"
    assert record.artifact_class == "notebook_output"


def test_crawl_competition_submission_format_prefers_explicit_csv_over_rubric_writeup_wording() -> None:
    listing = CompetitionListing(
        slug="csv-with-writeup",
        title="CSV With Writeup",
        url="https://www.kaggle.com/competitions/csv-with-writeup",
        description="",
        category="Community",
        reward="$1,000",
        evaluation_metric="dice",
        team_count=10,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Evaluation</h2><p>Technical quality is reviewed under the judging rubric.</p>
      <h2>Submission File</h2>
      <p>Upload one submission.csv with columns id,prediction.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/csv-with-writeup/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.artifact_class == "tabular"
    assert record.detected_extensions == [".csv"]


def test_crawl_competition_submission_format_prefers_file_over_supplemental_required_writeup() -> None:
    listing = CompetitionListing(
        slug="file-plus-winner-writeup",
        title="File Plus Winner Writeup",
        url="https://www.kaggle.com/competitions/file-plus-winner-writeup",
        description="",
        category="Featured",
        reward="$1,000",
        evaluation_metric="score",
        team_count=10,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission File</h2><p>Upload predictions.csv with columns id,prediction.</p>
      <h2>Prizes</h2><p>Prize winners must submit a solution writeup after the competition.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/file-plus-winner-writeup/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.detected_extensions == [".csv"]


def test_crawl_competition_submission_format_keeps_required_writeup_with_attachment() -> None:
    listing = CompetitionListing(
        slug="writeup-with-zip",
        title="Writeup With Zip",
        url="https://www.kaggle.com/competitions/writeup-with-zip",
        description="",
        category="Community",
        reward="$1,000",
        evaluation_metric="",
        team_count=10,
        max_daily_submissions=1,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Requirements</h2>
      <p>A Kaggle Writeup is required to be eligible. Attach submission.zip to the Writeup.</p>
      <p>To create a new Writeup, click New Writeup. After you save your Writeup, click Submit.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/writeup-with-zip/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "writeup_submission"
    assert record.artifact_class == "writeup"


def test_crawl_competition_submission_format_prefers_executable_notebook_over_output_suffix() -> None:
    listing = CompetitionListing(
        slug="executable-reconstruction",
        title="Executable Reconstruction",
        url="https://www.kaggle.com/competitions/executable-reconstruction",
        description="",
        category="Community",
        reward="$1,000",
        evaluation_metric="",
        team_count=10,
        max_daily_submissions=1,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <h2>Submission Requirements</h2>
      <p>Participants must submit an executable reconstruction algorithm, not a dataset.</p>
      <p>Each submission must be a Kaggle Notebook or script that runs end-to-end.</p>
      <p>The notebook writes reconstruction.npy during evaluation.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({"https://www.kaggle.com/competitions/executable-reconstruction/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "notebook_output_submission"
    assert record.required_artifact == "Kaggle notebook submission"


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


def test_find_submission_text_block_prefers_explicit_archive_over_earlier_sample_mention() -> None:
    text = """
    The agent receives a sample submission while running in the sandbox.

    Evaluation
    You must provide a zip archive named submission.zip containing agent.yaml and prompts/system.md.
    """

    section = find_submission_text_block(text)

    assert "submission.zip" in section


def test_crawl_competition_submission_format_classifies_agent_config_zip_bundle() -> None:
    slug = "agent-config-demo"
    listing = CompetitionListing(
        slug=slug,
        title="Agent Config Demo",
        url=f"https://www.kaggle.com/competitions/{slug}",
        description="",
        category="Playground",
        reward="Swag",
        evaluation_metric="AUC",
        team_count=100,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="test",
    )
    overview_html = """
    <html><body>
      <p>The agent receives a sample submission in the sandbox.</p>
      <h2>Evaluation</h2>
      <p>You must provide a zip archive named submission.zip containing your Agent Config.</p>
      <p>An agent.yaml file must be located at the root of the archive.</p>
    </body></html>
    """

    record = crawl_competition_submission_format(
        listing=listing,
        fetcher=_FakeFetcher({f"https://www.kaggle.com/competitions/{slug}/overview": overview_html}),  # type: ignore[arg-type]
        fetch_rules_page=False,
    )

    assert record.submission_mode == "direct_file_upload"
    assert record.artifact_class == "bundle"
    assert record.required_artifact == "ZIP submission bundle"
    assert ".zip" in record.detected_extensions


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
    assert saved["supported_competition_count"] == 1
    assert saved["review_required_count"] == 0
    assert saved["review_required"] == []


def test_assess_submission_format_support_flags_unknown_direct_upload() -> None:
    record = CrawlRecord(
        slug="unknown",
        title="Unknown",
        competition_type="Community",
        submission_mode="direct_file_upload",
        required_artifact="unknown",
        artifact_class="unknown",
        artifact_container=None,
        raw_format_text="",
        detected_extensions=[],
        detected_columns=[],
        delimiter=None,
        is_code_competition=False,
        evidence_url="https://www.kaggle.com/competitions/unknown/overview",
        evidence_html_snippet="",
        extraction_confidence="low",
        reward="",
        evaluation_metric="",
        team_count=None,
        max_daily_submissions=None,
        discovery_source="test",
        crawled_at="2026-01-01T00:00:00+00:00",
    )

    assert assess_submission_format_support(record) == (False, "unknown_artifact_class")


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


def test_discover_entered_competitions_pages_and_deduplicates() -> None:
    item = SimpleNamespace(
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
    first_page = [item, *[item for _ in range(19)]]
    fake_api = _FakeApi(
        {
            ("entered", None, 1, None): first_page,
            ("entered", None, 2, None): [],
        }
    )

    listings = discover_entered_competitions(api=fake_api, page_limit=3)

    assert [listing.slug for listing in listings] == ["titanic"]


def test_discover_entered_competitions_accepts_kaggle_sdk_2_response_wrapper() -> None:
    item = SimpleNamespace(
        url="https://www.kaggle.com/competitions/wrapped",
        ref="https://www.kaggle.com/competitions/wrapped",
        title="Wrapped",
        description="desc",
        category="Research",
        reward="$1,000",
        evaluation_metric="AUC",
        team_count=20,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
    )

    class WrappedFakeApi(_FakeApi):
        def competitions_list(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(competitions=super().competitions_list(**kwargs))

    fake_api = WrappedFakeApi({("entered", None, 1, None): [item]})

    listings = discover_entered_competitions(api=fake_api, page_limit=3)

    assert [listing.slug for listing in listings] == ["wrapped"]


def test_enrich_records_prefers_accepted_submission_filename() -> None:
    record = CrawlRecord(
        slug="currency-noise",
        title="Currency Noise",
        competition_type="Community",
        submission_mode="direct_file_upload",
        required_artifact=".usd file",
        artifact_class="tabular",
        artifact_container="file",
        raw_format_text="Cash prizes are USD 10,000.",
        detected_extensions=[".usd"],
        detected_columns=[],
        delimiter=None,
        is_code_competition=False,
        evidence_url="https://www.kaggle.com/competitions/currency-noise/overview",
        evidence_html_snippet="",
        extraction_confidence="low",
        reward="$10,000",
        evaluation_metric="RMSE",
        team_count=10,
        max_daily_submissions=5,
        discovery_source="test",
        crawled_at="2026-01-01T00:00:00+00:00",
    )

    class EvidenceApi:
        def authenticate(self) -> None:
            return None

        def competition_submissions(self, slug: str, *, page_size: int):  # noqa: ANN202
            assert slug == "currency-noise"
            assert page_size == 100
            return [SimpleNamespace(file_name="submission.csv", status="SubmissionStatus.COMPLETE")]

        def competition_list_files(self, slug: str, *, page_size: int):  # noqa: ANN202, ARG002
            raise AssertionError("accepted submission evidence should avoid a file-list call")

    [enriched] = enrich_records_with_kaggle_evidence([record], api=EvidenceApi())  # type: ignore[arg-type]

    assert enriched.detected_extensions == [".csv"]
    assert enriched.artifact_class == "tabular"
    assert enriched.required_artifact == ".csv file"
    assert "accepted submission filename" in enriched.raw_format_text


def test_enrich_records_uses_only_top_level_official_sample_submission() -> None:
    record = CrawlRecord(
        slug="sample-evidence",
        title="Sample Evidence",
        competition_type="Community",
        submission_mode="other_or_unknown",
        required_artifact="unknown",
        artifact_class="unknown",
        artifact_container=None,
        raw_format_text="Submission details are in the data files.",
        detected_extensions=[],
        detected_columns=[],
        delimiter=None,
        is_code_competition=False,
        evidence_url="https://www.kaggle.com/competitions/sample-evidence/overview",
        evidence_html_snippet="",
        extraction_confidence="low",
        reward="",
        evaluation_metric="",
        team_count=10,
        max_daily_submissions=5,
        discovery_source="test",
        crawled_at="2026-01-01T00:00:00+00:00",
    )

    class EvidenceApi:
        def authenticate(self) -> None:
            return None

        def competition_submissions(self, slug: str, *, page_size: int):  # noqa: ANN202, ARG002
            return []

        def competition_list_files(self, slug: str, *, page_size: int):  # noqa: ANN202, ARG002
            return SimpleNamespace(
                files=[
                    SimpleNamespace(name="training/example/sample_submission.csv"),
                    SimpleNamespace(name="SampleSubmissionStage1.tsv"),
                ]
            )

    [enriched] = enrich_records_with_kaggle_evidence([record], api=EvidenceApi())  # type: ignore[arg-type]

    assert enriched.submission_mode == "direct_file_upload"
    assert enriched.detected_extensions == [".tsv"]
    assert enriched.artifact_class == "tabular"
