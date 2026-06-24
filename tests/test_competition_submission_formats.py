from __future__ import annotations

import json
from types import SimpleNamespace

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
