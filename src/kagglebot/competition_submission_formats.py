from __future__ import annotations

import base64
import csv
import json
import re
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kaggle.api.kaggle_api_extended import KaggleApi

from kagglebot.asset_modality import (
    DOCUMENT_SUFFIXES,
    archive_container,
    artifact_suffix,
)
from kagglebot.compression_suffixes import strip_compression_suffix
from kagglebot.exec_utils import run_command
from kagglebot.json_utils import append_jsonl_record, load_jsonl_records, write_json_object, write_jsonl_records
from kagglebot.kaggle_api import normalize_competitions_list_response
from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_NOTEBOOK_OUTPUT,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_UNKNOWN,
    ARTIFACT_CLASS_WRITEUP,
)
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_SUFFIXES,
    COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES,
    COMPRESSION_TOKEN_PATTERNS,
    JSON_TEXT_NOISE_CONTEXT_MARKERS,
    MODEL_BUNDLE_MARKERS,
    NON_TABULAR_SUBMISSION_SUFFIXES,
    SUBMISSION_ARTIFACT_KEYWORDS,
    SUBMISSION_TOKEN_PATTERNS,
    drop_shadowed_submission_suffixes,
    submission_extension_pattern,
)
from kagglebot.submission_format import SubmissionFormatHint, extract_submission_section, parse_submission_format
from kagglebot.submission_output_naming import all_submission_output_suffixes
from kagglebot.submission_sample_discovery import TABULAR_SUBMISSION_SUFFIXES
from kagglebot.writeup import infer_deliverable_mode

_DEFAULT_SEARCH_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_DEFAULT_PAGE_SIZE = 20
_SUBMISSION_NOTEBOOK_RE = re.compile(r"\b(submit|submission|upload)\b.{0,80}\b(notebook|code)\b", re.I | re.S)
_EXECUTABLE_NOTEBOOK_SUBMISSION_RE = re.compile(
    r"(?:"
    r"(?:must|shall|required to)\s+submit.{0,160}\b(?:executable|notebook|script|algorithm)\b|"
    r"(?:submission|entry)\s+must.{0,160}\b(?:kaggle\s+)?(?:notebook|script|executable)\b|"
    r"\b(?:kaggle\s+)?notebook\s+or\s+script\b"
    r")",
    re.I | re.S,
)
_CODE_COMPETITION_RE = re.compile(r"\b(code competition|kernel submissions only|notebook submission)\b", re.I)
_BULLET_COLUMN_RE = re.compile(r"^\s*[-*]\s*`?(?P<name>[A-Za-z0-9_. -]+?)`?\s*$")
_SECTION_BLOCK_RE = re.compile(
    r"(?is)"
    r"(submission file(?: format)?|submission format|make a submission|how to submit|submit predictions)"
    r"(.*?)(?=\n[A-Z][^\n]{0,80}\n|\Z)"
)
_KEYWORD_WINDOW_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsubmission file format\b", re.I),
    re.compile(r"\bsubmission file\b", re.I),
    re.compile(r"\bsubmission format\b", re.I),
    re.compile(r"\bsample submission\b", re.I),
    re.compile(r"\bsubmit predictions\b", re.I),
    re.compile(r"\bsubmission[A-Za-z0-9_.-]*\.off\b", re.I),
    re.compile(
        rf"\bsubmit\b.{{0,120}}\b({SUBMISSION_ARTIFACT_KEYWORDS})\b",
        re.I | re.S,
    ),
    re.compile(
        rf"\b(upload|submission)\b.{{0,120}}\b({SUBMISSION_ARTIFACT_KEYWORDS})\b",
        re.I | re.S,
    ),
)
_ARCHIVE_SUBMISSION_SUFFIXES = set(ARCHIVE_SUBMISSION_SUFFIXES)
_TABULAR_SINGLE_FILE_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)
_NON_TABULAR_SINGLE_FILE_SUFFIXES = set(NON_TABULAR_SUBMISSION_SUFFIXES)
_DIRECT_SUBMISSION_EXTENSION_PATTERN = submission_extension_pattern(all_submission_output_suffixes())
_SUPPORTED_SUBMISSION_MODES = {
    "code_competition_runtime_submission",
    "direct_file_upload",
    "notebook_output_submission",
    "writeup_submission",
}
_WRITEUP_EVIDENCE_MARKERS = ("writeup", "judged", "rubric", "panel", "manual grading", "manual review")
_REQUIRED_WRITEUP_RE = re.compile(
    r"(?:"
    r"(?:must|required|requires?|mandatory|need to|to be eligible).{0,100}\bwrite[ -]?ups?\b|"
    r"\bwrite[ -]?ups?\b.{0,100}(?:must|required|mandatory|to be eligible)|"
    r"final submission.{0,120}(?:through|as).{0,40}\bwrite[ -]?ups?\b|"
    r"create (?:a |your )?(?:new )?write[ -]?up.{0,120}\bsubmit\b"
    r")",
    re.I | re.S,
)
_PRIMARY_WRITEUP_SUBMISSION_RE = re.compile(
    r"(?:"
    r"final submission.{0,120}(?:through|as).{0,40}\bwrite[ -]?ups?\b|"
    r"(?:create|new) (?:a |your )?(?:new )?write[ -]?up.{0,160}\bsubmit\b|"
    r"after (?:you )?save (?:your )?write[ -]?up.{0,100}\bsubmit\b|"
    r"complete submission consists of.{0,160}\bwrite[ -]?ups?\b"
    r")",
    re.I | re.S,
)
_PRIMARY_FILE_SUBMISSION_RE = re.compile(
    r"(?:\bsubmission file\b|\bpredictions? must be submitted as\b|\bfile should contain a header\b)",
    re.I,
)
_GENERIC_BUNDLE_MARKERS = ("agent config", "agent.yaml", "system prompts", "custom tools", "skills/")
_EXTERNAL_REPOSITORY_SUBMISSION_RE = re.compile(
    r"\bsubmit\b.{0,100}\bsource code\b.{0,180}\b(?:pull request|\bpr\b)",
    re.I | re.S,
)
_SAMPLE_NOTEBOOK_LINK_RE = re.compile(
    r"sample\s+submission.{0,700}?href=[\"'](?:https?://www\.kaggle\.com)?/code/[^\"']+[\"']",
    re.I | re.S,
)
_SAMPLE_SUBMISSION_NAME_RE = re.compile(
    r"^(?:sample[_ -]?submission|samplesubmission|submission[_ -]?template|answer[_ -]?template)",
    re.I,
)


@dataclass(frozen=True)
class CompetitionListing:
    slug: str
    title: str
    url: str
    description: str
    category: str
    reward: str
    evaluation_metric: str
    team_count: int | None
    max_daily_submissions: int | None
    is_kernels_submissions_only: bool
    submissions_disabled: bool
    source: str


@dataclass(frozen=True)
class CrawlRecord:
    slug: str
    title: str
    competition_type: str
    submission_mode: str
    required_artifact: str
    artifact_class: str
    artifact_container: str | None
    raw_format_text: str
    detected_extensions: list[str]
    detected_columns: list[str]
    delimiter: str | None
    is_code_competition: bool
    evidence_url: str
    evidence_html_snippet: str
    extraction_confidence: str
    reward: str
    evaluation_metric: str
    team_count: int | None
    max_daily_submissions: int | None
    discovery_source: str
    crawled_at: str


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._in_pre = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"br", "hr"}:
            self._chunks.append("\n")
        elif tag in {"p", "div", "section", "article", "ul", "ol", "table", "tr"}:
            self._chunks.append("\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._chunks.append("\n")
        elif tag == "li":
            self._chunks.append("\n- ")
        elif tag == "pre":
            self._in_pre = True
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in {"p", "div", "section", "article", "ul", "ol", "table", "tr"}:
            self._chunks.append("\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._chunks.append("\n")
        elif tag == "pre":
            self._in_pre = False
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        self._chunks.append(data if self._in_pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        text = "".join(self._chunks)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class ChromeDomFetcher:
    def __init__(
        self,
        *,
        chrome_binary: str = "google-chrome",
        virtual_time_budget_ms: int = 15_000,
        timeout_sec: float = 45.0,
    ) -> None:
        self._chrome_binary = chrome_binary
        self._virtual_time_budget_ms = virtual_time_budget_ms
        self._timeout_sec = timeout_sec

    def fetch(self, url: str) -> str:
        result = run_command(
            [
                self._chrome_binary,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                f"--virtual-time-budget={self._virtual_time_budget_ms}",
                "--dump-dom",
                url,
            ],
            timeout=self._timeout_sec,
        )
        if result.returncode != 0:
            raise RuntimeError(f"chrome dump-dom failed for {url}: {result.output[:400]}")
        html = result.stdout.strip()
        if not html:
            raise RuntimeError(f"chrome dump-dom returned empty output for {url}")
        return html


def crawl_submission_formats(
    *,
    output_dir: Path,
    fetcher: ChromeDomFetcher | None = None,
    max_prefix_depth: int = 1,
    max_pages_per_search: int = 2,
    search_alphabet: str = _DEFAULT_SEARCH_ALPHABET,
    max_competitions: int | None = None,
    fetch_rules_pages: bool = True,
    resume: bool = True,
    entered_only: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or ChromeDomFetcher()

    if entered_only:
        listings = discover_entered_competitions()
    else:
        listings = discover_competitions(
            fetcher=fetcher,
            max_prefix_depth=max_prefix_depth,
            max_pages_per_search=max_pages_per_search,
            search_alphabet=search_alphabet,
        )
    if max_competitions is not None:
        listings = listings[:max_competitions]

    raw_path = output_dir / "raw_submission_formats.jsonl"
    listing_slugs = {listing.slug for listing in listings}
    records: list[CrawlRecord] = []
    if resume and raw_path.exists():
        existing_records = [record for record in load_raw_records(raw_path) if record.slug in listing_slugs]
        records.extend(record for record in existing_records if not should_retry_crawl_record(record))
    processed_slugs = {record.slug for record in records}

    for index, listing in enumerate(listings, start=1):
        if listing.slug in processed_slugs:
            continue
        try:
            record = crawl_competition_submission_format(
                listing=listing,
                fetcher=fetcher,
                fetch_rules_page=fetch_rules_pages,
            )
        except Exception as exc:  # noqa: BLE001
            record = CrawlRecord(
                slug=listing.slug,
                title=listing.title,
                competition_type=listing.category or "Unspecified",
                submission_mode="other_or_unknown",
                required_artifact="unknown",
                artifact_class=ARTIFACT_CLASS_UNKNOWN,
                artifact_container=None,
                raw_format_text=str(exc),
                detected_extensions=[],
                detected_columns=[],
                delimiter=None,
                is_code_competition=listing.is_kernels_submissions_only,
                evidence_url=listing.url,
                evidence_html_snippet="",
                extraction_confidence="low",
                reward=listing.reward,
                evaluation_metric=listing.evaluation_metric,
                team_count=listing.team_count,
                max_daily_submissions=listing.max_daily_submissions,
                discovery_source=listing.source,
                crawled_at=datetime.now(UTC).isoformat(),
            )
        append_jsonl(raw_path, asdict(record))
        records.append(record)
        processed_slugs.add(listing.slug)
        print(f"[{index}/{len(listings)}] crawled {listing.slug} -> {record.submission_mode}")

    records = [refine_crawl_record(record) for record in records]
    if entered_only:
        records = enrich_records_with_kaggle_evidence(records)

    write_jsonl(raw_path, [asdict(record) for record in records])
    write_jsonl(output_dir / "discovered_competitions.jsonl", [asdict(item) for item in listings])
    write_csv(output_dir / "normalized_submission_formats.csv", records)
    summary = build_summary(records)
    write_json_object(output_dir / "summary.json", summary)
    return summary


def should_retry_crawl_record(record: CrawlRecord) -> bool:
    return record.submission_mode == "other_or_unknown" or (
        record.submission_mode == "direct_file_upload"
        and (record.artifact_class == ARTIFACT_CLASS_UNKNOWN or not record.detected_extensions)
    )


def discover_entered_competitions(
    *,
    api: KaggleApi | None = None,
    page_limit: int = 100,
    retry_attempts: int = 5,
) -> list[CompetitionListing]:
    api = api or KaggleApi()
    api.authenticate()
    listings: dict[str, CompetitionListing] = {}
    for page in range(1, max(1, page_limit) + 1):
        competitions = _entered_competitions_page_with_retry(
            api,
            page=page,
            retry_attempts=retry_attempts,
        )
        if not competitions:
            break
        for competition in competitions:
            listing = listing_from_api_competition(competition, source="api-group:entered")
            listings[listing.slug] = merge_listing(listings.get(listing.slug), listing)
        if len(competitions) < _DEFAULT_PAGE_SIZE:
            break
    return sorted(listings.values(), key=lambda item: item.slug)


def _entered_competitions_page_with_retry(
    api: KaggleApi,
    *,
    page: int,
    retry_attempts: int,
) -> list[Any]:
    for attempt in range(max(1, retry_attempts)):
        try:
            return normalize_competitions_list_response(
                api.competitions_list(group="entered", page=page, sort_by="latestDeadline")
            )
        except Exception as exc:  # noqa: BLE001
            if "429" not in str(exc) or attempt + 1 >= max(1, retry_attempts):
                raise
            delay_sec = min(60, 5 * (2**attempt))
            print(f"Kaggle API rate limited; retrying entered page {page} in {delay_sec}s")
            time.sleep(delay_sec)
    return []


def discover_competitions(
    *,
    fetcher: ChromeDomFetcher,
    max_prefix_depth: int,
    max_pages_per_search: int,
    search_alphabet: str,
) -> list[CompetitionListing]:
    listings: dict[str, CompetitionListing] = {}

    for listing in discover_from_landing_page(fetcher):
        listings.setdefault(listing.slug, listing)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception:  # noqa: BLE001
        return sorted(listings.values(), key=lambda item: (item.category.lower(), item.slug))

    for group in ("general", "community", "entered"):
        for page in range(1, max_pages_per_search + 1):
            competitions = normalize_competitions_list_response(
                api.competitions_list(group=group, page=page, sort_by="recentlyCreated")
            )
            if not competitions:
                break
            for competition in competitions:
                listing = listing_from_api_competition(competition, source=f"api-group:{group}")
                listings[listing.slug] = merge_listing(listings.get(listing.slug), listing)
            if len(competitions) < _DEFAULT_PAGE_SIZE:
                break

    for category in ("all", "featured", "research", "gettingStarted", "playground"):
        for page in range(1, max_pages_per_search + 1):
            competitions = normalize_competitions_list_response(
                api.competitions_list(category=category, page=page, sort_by="recentlyCreated")
            )
            if not competitions:
                break
            for competition in competitions:
                listing = listing_from_api_competition(competition, source=f"api-category:{category}")
                listings[listing.slug] = merge_listing(listings.get(listing.slug), listing)
            if len(competitions) < _DEFAULT_PAGE_SIZE:
                break

    queue: deque[str] = deque(search_alphabet)
    seen_queries: set[str] = set()
    while queue:
        prefix = queue.popleft()
        if prefix in seen_queries:
            continue
        seen_queries.add(prefix)
        saturated = False
        for page in range(1, max_pages_per_search + 1):
            competitions = normalize_competitions_list_response(
                api.competitions_list(
                    search=prefix,
                    page=page,
                    category="all",
                    sort_by="recentlyCreated",
                )
            )
            if not competitions:
                break
            for competition in competitions:
                listing = listing_from_api_competition(competition, source=f"api-search:{prefix}")
                listings[listing.slug] = merge_listing(listings.get(listing.slug), listing)
            if len(competitions) < _DEFAULT_PAGE_SIZE:
                break
            saturated = True
        if saturated and len(prefix) < max_prefix_depth:
            for char in search_alphabet:
                queue.append(prefix + char)

    return sorted(listings.values(), key=lambda item: (item.category.lower(), item.slug))


def discover_from_landing_page(fetcher: ChromeDomFetcher) -> list[CompetitionListing]:
    html = fetcher.fetch("https://www.kaggle.com/competitions")
    text = html_to_text(html)
    title_map = dict(extract_competition_titles_from_landing_html(html))
    records: list[CompetitionListing] = []
    for slug in extract_competition_slugs_from_html(html):
        records.append(
            CompetitionListing(
                slug=slug,
                title=title_map.get(slug, slug),
                url=f"https://www.kaggle.com/competitions/{slug}",
                description="",
                category=infer_category_from_landing_text(text, slug),
                reward="",
                evaluation_metric="",
                team_count=None,
                max_daily_submissions=None,
                is_kernels_submissions_only=False,
                submissions_disabled=False,
                source="landing-page",
            )
        )
    return records


def crawl_competition_submission_format(
    *,
    listing: CompetitionListing,
    fetcher: ChromeDomFetcher,
    fetch_rules_page: bool,
) -> CrawlRecord:
    overview_url = f"https://www.kaggle.com/competitions/{listing.slug}/overview"
    rules_url = f"https://www.kaggle.com/competitions/{listing.slug}/rules"

    overview_html = fetcher.fetch(overview_url)
    overview_text = html_to_text(overview_html)
    overview_section = find_submission_text_block(overview_text)

    rules_html = ""
    rules_text = ""
    rules_section = ""
    if fetch_rules_page and (listing.is_kernels_submissions_only or not overview_section):
        rules_html = fetcher.fetch(rules_url)
        rules_text = html_to_text(rules_html)
        rules_section = find_submission_text_block(rules_text)

    combined_text = "\n".join((overview_text, rules_text))
    is_external_repository_submission = bool(_EXTERNAL_REPOSITORY_SUBMISSION_RE.search(combined_text))
    preliminary_text = overview_section or rules_section
    preliminary_suffixes = filter_noisy_suffixes(
        preliminary_text,
        normalize_suffixes(
            extract_suffixes_from_text(preliminary_text) or infer_suffixes_from_keywords(preliminary_text)
        ),
    )
    has_primary_file_or_notebook_contract = bool(
        preliminary_suffixes
        or _PRIMARY_FILE_SUBMISSION_RE.search(preliminary_text)
        or _SUBMISSION_NOTEBOOK_RE.search(preliminary_text)
        or _CODE_COMPETITION_RE.search(preliminary_text)
    )
    writeup_required = bool(_REQUIRED_WRITEUP_RE.search(combined_text))
    writeup_is_primary = bool(_PRIMARY_WRITEUP_SUBMISSION_RE.search(combined_text))
    is_writeup = bool(
        not listing.is_kernels_submissions_only
        and not is_external_repository_submission
        and (
            writeup_is_primary
            or (
                not has_primary_file_or_notebook_contract
                and (writeup_required or infer_deliverable_mode(overview_text, rules_text, default="") == "writeup")
            )
        )
    )
    writeup_text = extract_writeup_evidence(overview_text, rules_text) if is_writeup else ""
    chosen_text = overview_section or rules_section or writeup_text
    chosen_url = overview_url if overview_section or writeup_text or not rules_section else rules_url
    if is_writeup:
        hint = SubmissionFormatHint(
            None,
            None,
            None,
            artifact_class=ARTIFACT_CLASS_WRITEUP,
            artifact_container=None,
        )
        expected_suffixes: list[str] = []
        submission_mode = "writeup_submission"
        artifact_class = ARTIFACT_CLASS_WRITEUP
        artifact_container = None
        required_artifact = "Kaggle Writeup"
        confidence = "high"
    elif is_external_repository_submission:
        hint = SubmissionFormatHint(
            None,
            None,
            None,
            artifact_class=ARTIFACT_CLASS_UNKNOWN,
            artifact_container=None,
        )
        expected_suffixes = []
        submission_mode = "external_repository_submission"
        artifact_class = ARTIFACT_CLASS_UNKNOWN
        artifact_container = None
        required_artifact = "external source-code pull request"
        confidence = "high"
    else:
        hint = (
            parse_submission_format(chosen_text)
            if chosen_text
            else SubmissionFormatHint(None, None, None, artifact_class=ARTIFACT_CLASS_UNKNOWN, artifact_container=None)
        )
        expected_suffixes = normalize_suffixes(
            hint.expected_suffixes
            or extract_suffixes_from_text(chosen_text)
            or infer_suffixes_from_keywords(chosen_text)
        )
        expected_suffixes = filter_noisy_suffixes(chosen_text, expected_suffixes)
        named_submission_suffixes = extract_named_submission_suffixes(chosen_text)
        if named_submission_suffixes:
            expected_suffixes = named_submission_suffixes
        csv_columns = infer_csv_example_columns(chosen_text)
        if csv_columns:
            expected_suffixes = [".csv"]
            hint = replace(hint, expected_suffixes=[".csv"], columns=csv_columns, delimiter=",")
        if any(suffix in _ARCHIVE_SUBMISSION_SUFFIXES for suffix in expected_suffixes) and any(
            marker in chosen_text.lower() for marker in MODEL_BUNDLE_MARKERS + _GENERIC_BUNDLE_MARKERS
        ):
            expected_suffixes = [suffix for suffix in expected_suffixes if suffix in _ARCHIVE_SUBMISSION_SUFFIXES]
        if not expected_suffixes and _SAMPLE_NOTEBOOK_LINK_RE.search(overview_html):
            submission_mode = "notebook_output_submission"
        else:
            submission_mode = infer_submission_mode(listing, chosen_text, expected_suffixes)
        artifact_class = infer_artifact_class(listing=listing, hint=hint, text=chosen_text, suffixes=expected_suffixes)
        artifact_container = infer_artifact_container(artifact_class=artifact_class, suffixes=expected_suffixes)
        required_artifact = infer_required_artifact(
            submission_mode=submission_mode,
            suffixes=expected_suffixes,
            artifact_class=artifact_class,
            text=chosen_text,
        )
        confidence = infer_confidence(chosen_text, expected_suffixes, listing.is_kernels_submissions_only)
    evidence_html = overview_html if chosen_url == overview_url else rules_html
    evidence_marker = overview_section or rules_section or chosen_text

    return refine_crawl_record(
        CrawlRecord(
            slug=listing.slug,
            title=listing.title,
            competition_type=listing.category or "Unspecified",
            submission_mode=submission_mode,
            required_artifact=required_artifact,
            artifact_class=artifact_class,
            artifact_container=artifact_container,
            raw_format_text=truncate_text(chosen_text, 4_000),
            detected_extensions=expected_suffixes,
            detected_columns=hint.columns
            or infer_columns_from_bullets(chosen_text, allow_fallback=bool(expected_suffixes)),
            delimiter=hint.delimiter,
            is_code_competition=submission_mode == "code_competition_runtime_submission",
            evidence_url=chosen_url,
            evidence_html_snippet=truncate_text(extract_evidence_snippet(evidence_html, evidence_marker), 1_500),
            extraction_confidence=confidence,
            reward=listing.reward,
            evaluation_metric=listing.evaluation_metric,
            team_count=listing.team_count,
            max_daily_submissions=listing.max_daily_submissions,
            discovery_source=listing.source,
            crawled_at=datetime.now(UTC).isoformat(),
        )
    )


def infer_csv_example_columns(text: str) -> list[str]:
    """Infer CSV from an adjacent header/data example, without treating prose dates as rows."""

    lowered = text.lower()
    if "submission" not in lowered or not any(marker in lowered for marker in ("format", "header", "schema")):
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, header_line in enumerate(lines[:-1]):
        if "," not in header_line:
            continue
        try:
            header = next(csv.reader([header_line]))
        except csv.Error:
            continue
        header = [cell.strip() for cell in header]
        if len(header) < 2 or not all(header) or not any(re.search(r"[A-Za-z_]", cell) for cell in header):
            continue
        for data_line in lines[index + 1 : index + 3]:
            if "," not in data_line:
                continue
            try:
                data = next(csv.reader([data_line]))
            except csv.Error:
                continue
            if len(data) == len(header):
                return header
    return []


def refine_crawl_record(record: CrawlRecord) -> CrawlRecord:
    text = record.raw_format_text
    if _EXTERNAL_REPOSITORY_SUBMISSION_RE.search(text):
        return replace(
            record,
            submission_mode="external_repository_submission",
            required_artifact="external source-code pull request",
            artifact_class=ARTIFACT_CLASS_UNKNOWN,
            artifact_container=None,
            detected_extensions=[],
            extraction_confidence="high",
        )
    if record.submission_mode in {
        "code_competition_runtime_submission",
        "notebook_output_submission",
        "writeup_submission",
        "external_repository_submission",
    }:
        return record

    named_submission_suffixes = extract_named_submission_suffixes(text)
    if named_submission_suffixes:
        return _record_with_authoritative_suffixes(record, named_submission_suffixes)

    csv_columns = infer_csv_example_columns(text)
    if csv_columns:
        return _record_with_authoritative_suffixes(record, [".csv"], columns=csv_columns)

    archive_suffixes = [suffix for suffix in record.detected_extensions if suffix in _ARCHIVE_SUBMISSION_SUFFIXES]
    if archive_suffixes and record.artifact_class in {ARTIFACT_CLASS_BUNDLE, ARTIFACT_CLASS_MULTI_FILE_ZIP}:
        return _record_with_authoritative_suffixes(record, archive_suffixes)
    return record


def enrich_records_with_kaggle_evidence(
    records: list[CrawlRecord],
    *,
    api: KaggleApi | None = None,
    retry_attempts: int = 4,
) -> list[CrawlRecord]:
    """Prefer accepted uploads and official top-level sample files over page-text guesses."""

    candidates = [record for record in records if record.submission_mode in {"direct_file_upload", "other_or_unknown"}]
    if not candidates:
        return records
    api = api or KaggleApi()
    api.authenticate()
    enriched: dict[str, CrawlRecord] = {}
    for record in candidates:
        successful_names = _successful_submission_names(
            api,
            slug=record.slug,
            retry_attempts=retry_attempts,
        )
        successful_suffixes = _suffixes_from_names(successful_names)
        if successful_suffixes:
            enriched[record.slug] = _record_with_authoritative_suffixes(
                record,
                successful_suffixes,
                evidence=f"accepted submission filename(s): {', '.join(successful_names)}",
            )
            continue

        sample_names = _top_level_sample_submission_names(
            api,
            slug=record.slug,
            retry_attempts=retry_attempts,
        )
        sample_suffixes = _suffixes_from_names(sample_names)
        if sample_suffixes:
            enriched[record.slug] = _record_with_authoritative_suffixes(
                record,
                sample_suffixes,
                evidence=f"official sample submission file(s): {', '.join(sample_names)}",
            )
    return [enriched.get(record.slug, record) for record in records]


def _successful_submission_names(api: KaggleApi, *, slug: str, retry_attempts: int) -> list[str]:
    try:
        submissions = _kaggle_call_with_retry(
            lambda: list(api.competition_submissions(slug, page_size=100)),
            label=f"submission history for {slug}",
            retry_attempts=retry_attempts,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Kaggle submission-history evidence unavailable for {slug}: {exc}")
        return []
    names: list[str] = []
    for submission in submissions:
        status = str(_api_attribute(submission, "status") or "").lower()
        if not status.endswith("complete"):
            continue
        name = _api_attribute(submission, "file_name", "fileName", "ref")
        if name and str(name) not in names:
            names.append(str(name))
    return names


def _top_level_sample_submission_names(api: KaggleApi, *, slug: str, retry_attempts: int) -> list[str]:
    try:
        response = _kaggle_call_with_retry(
            lambda: api.competition_list_files(slug, page_size=1000),
            label=f"sample files for {slug}",
            retry_attempts=retry_attempts,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Kaggle sample-file evidence unavailable for {slug}: {exc}")
        return []
    names: list[str] = []
    for item in list(getattr(response, "files", response) or []):
        name = str(_api_attribute(item, "name", "ref") or "").replace("\\", "/")
        if "/" in name or not _SAMPLE_SUBMISSION_NAME_RE.match(name):
            continue
        if name not in names:
            names.append(name)
    return names


def _kaggle_call_with_retry(call: Any, *, label: str, retry_attempts: int) -> Any:
    for attempt in range(max(1, retry_attempts)):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            if "429" not in str(exc) or attempt + 1 >= max(1, retry_attempts):
                raise
            delay_sec = min(60, 5 * (2**attempt))
            print(f"Kaggle API rate limited; retrying {label} in {delay_sec}s")
            time.sleep(delay_sec)
    raise RuntimeError(f"Kaggle API call failed: {label}")


def _api_attribute(item: Any, *names: str) -> Any:
    for name in names:
        value = getattr(item, name, None)
        if value is None:
            value = getattr(item, f"_{name}", None)
        if value is not None:
            return value
    return None


def _suffixes_from_names(names: Iterable[str]) -> list[str]:
    suffixes: list[str] = []
    for name in names:
        suffix = artifact_suffix(Path(name))
        if suffix and suffix not in suffixes:
            suffixes.append(suffix)
    return normalize_suffixes(suffixes)


def _record_with_authoritative_suffixes(
    record: CrawlRecord,
    suffixes: list[str],
    *,
    columns: list[str] | None = None,
    evidence: str = "",
) -> CrawlRecord:
    normalized = normalize_suffixes(suffixes)
    archive_suffixes = [suffix for suffix in normalized if suffix in _ARCHIVE_SUBMISSION_SUFFIXES]
    if archive_suffixes:
        normalized = archive_suffixes
        if record.artifact_class == ARTIFACT_CLASS_BUNDLE:
            artifact_class = ARTIFACT_CLASS_BUNDLE
        elif record.artifact_class == ARTIFACT_CLASS_MULTI_FILE_ZIP:
            artifact_class = ARTIFACT_CLASS_MULTI_FILE_ZIP
        else:
            artifact_class = ARTIFACT_CLASS_SINGLE_FILE
    elif any(_is_tabular_file_submission_suffix(suffix) for suffix in normalized):
        artifact_class = ARTIFACT_CLASS_TABULAR
    else:
        artifact_class = ARTIFACT_CLASS_SINGLE_FILE
    raw_text = record.raw_format_text
    if evidence and evidence not in raw_text:
        raw_text = truncate_text(f"{raw_text}\n\nAudit evidence: {evidence}".strip(), 4_000)
    return replace(
        record,
        submission_mode="direct_file_upload",
        required_artifact=infer_required_artifact(
            "direct_file_upload",
            normalized,
            artifact_class,
            text=raw_text,
        ),
        artifact_class=artifact_class,
        artifact_container=infer_artifact_container(artifact_class=artifact_class, suffixes=normalized),
        raw_format_text=raw_text,
        detected_extensions=normalized,
        detected_columns=columns or record.detected_columns,
        delimiter="," if ".csv" in normalized else record.delimiter,
        extraction_confidence="high" if evidence or columns else record.extraction_confidence,
        discovery_source=(
            f"{record.discovery_source},kaggle-submission-evidence"
            if evidence and "kaggle-submission-evidence" not in record.discovery_source.split(",")
            else record.discovery_source
        ),
    )


def find_submission_text_block(text: str) -> str:
    candidates: list[str] = []
    section = extract_submission_section(text)
    if section and len(section.splitlines()) > 1:
        candidates.append(section)
    for match in _SECTION_BLOCK_RE.finditer(text):
        candidate = match.group(0).strip()
        if len(candidate.splitlines()) > 1:
            candidates.append(candidate)
    keyword_window = extract_keyword_window(text, window=1_200)
    if keyword_window:
        candidates.append(keyword_window)
    preferred = select_best_submission_block(candidates)
    if preferred:
        return preferred
    return extract_keyword_window(text)


def extract_writeup_evidence(*texts: str, max_lines: int = 12) -> str:
    lines: list[str] = []
    for text in texts:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            lowered = line.lower()
            if line and any(marker in lowered for marker in _WRITEUP_EVIDENCE_MARKERS) and line not in lines:
                lines.append(line)
            if len(lines) >= max_lines:
                return "\n".join(lines)
    return "\n".join(lines)


def infer_submission_mode(listing: CompetitionListing, text: str, suffixes: list[str]) -> str:
    if listing.is_kernels_submissions_only:
        return "code_competition_runtime_submission"
    if _EXECUTABLE_NOTEBOOK_SUBMISSION_RE.search(text):
        return "notebook_output_submission"
    if suffixes:
        return "direct_file_upload"
    if _SUBMISSION_NOTEBOOK_RE.search(text) or _CODE_COMPETITION_RE.search(text):
        return "notebook_output_submission"
    return "other_or_unknown"


def infer_artifact_class(
    *,
    listing: CompetitionListing,
    hint: SubmissionFormatHint,
    text: str,
    suffixes: list[str],
) -> str:
    if listing.is_kernels_submissions_only:
        return ARTIFACT_CLASS_NOTEBOOK_OUTPUT
    lowered = text.lower()
    archive_suffixes = [suffix for suffix in suffixes if suffix in _ARCHIVE_SUBMISSION_SUFFIXES]
    if archive_suffixes and any(marker in lowered for marker in MODEL_BUNDLE_MARKERS + _GENERIC_BUNDLE_MARKERS):
        return ARTIFACT_CLASS_BUNDLE
    if hint.artifact_class and hint.artifact_class != ARTIFACT_CLASS_UNKNOWN:
        return hint.artifact_class
    if archive_suffixes:
        if ".zip" in archive_suffixes:
            return ARTIFACT_CLASS_MULTI_FILE_ZIP
        return ARTIFACT_CLASS_SINGLE_FILE
    if suffixes:
        if any(_is_tabular_file_submission_suffix(suffix) for suffix in suffixes):
            return ARTIFACT_CLASS_TABULAR
        return ARTIFACT_CLASS_SINGLE_FILE
    return ARTIFACT_CLASS_UNKNOWN


def infer_artifact_container(*, artifact_class: str, suffixes: list[str]) -> str | None:
    archive_suffixes = [suffix for suffix in suffixes if suffix in _ARCHIVE_SUBMISSION_SUFFIXES]
    if artifact_class in {ARTIFACT_CLASS_BUNDLE, ARTIFACT_CLASS_MULTI_FILE_ZIP}:
        return archive_container(archive_suffixes)
    if artifact_class in {ARTIFACT_CLASS_TABULAR, ARTIFACT_CLASS_SINGLE_FILE, ARTIFACT_CLASS_NOTEBOOK_OUTPUT}:
        container = archive_container(archive_suffixes)
        if container is not None:
            return container
        return "file"
    if archive_suffixes:
        return archive_container(archive_suffixes)
    if suffixes:
        return "file"
    return None


def infer_required_artifact(
    submission_mode: str,
    suffixes: list[str],
    artifact_class: str,
    *,
    text: str = "",
) -> str:
    if submission_mode == "code_competition_runtime_submission":
        return "Kaggle notebook output"
    if submission_mode == "notebook_output_submission":
        return "Kaggle notebook submission"
    if artifact_class == ARTIFACT_CLASS_BUNDLE:
        container = (archive_container(suffixes) or "zip").upper()
        if any(marker in text.lower() for marker in _GENERIC_BUNDLE_MARKERS):
            return f"{container} submission bundle"
        return f"{container} bundle containing model assets and inference code"
    if artifact_class == ARTIFACT_CLASS_MULTI_FILE_ZIP:
        container = (archive_container(suffixes) or "zip").upper()
        return f"{container} archive containing multiple prediction files"
    container = archive_container(suffixes)
    if container is not None:
        return f"{container.upper()} archive"
    if artifact_class == ARTIFACT_CLASS_SINGLE_FILE and suffixes:
        return f"{suffixes[0]} single file"
    if suffixes:
        return f"{suffixes[0]} file"
    return "unknown"


def filter_noisy_suffixes(text: str, suffixes: list[str]) -> list[str]:
    lowered = text.lower()
    if not any(marker in lowered for marker in JSON_TEXT_NOISE_CONTEXT_MARKERS):
        return suffixes
    return [suffix for suffix in suffixes if suffix not in {".json", ".txt"}]


def _base_submission_suffix(suffix: str) -> str:
    return strip_compression_suffix(suffix)


def _is_tabular_file_submission_suffix(suffix: str) -> bool:
    if suffix not in _TABULAR_SINGLE_FILE_SUFFIXES:
        return False
    return _base_submission_suffix(suffix) not in {".json", ".txt"}


def infer_confidence(text: str, suffixes: list[str], kernels_only: bool) -> str:
    lowered = text.lower()
    if kernels_only:
        return "high"
    if suffixes and "submission format" in lowered:
        return "high"
    if suffixes:
        return "medium"
    return "low"


def listing_from_api_competition(competition: Any, *, source: str) -> CompetitionListing:
    url = getattr(competition, "url", None) or getattr(competition, "ref", "")
    slug = competition_slug_from_url(url)
    return CompetitionListing(
        slug=slug,
        title=getattr(competition, "title", "") or slug,
        url=url,
        description=getattr(competition, "description", "") or "",
        category=getattr(competition, "category", "") or "Unspecified",
        reward=getattr(competition, "reward", "") or "",
        evaluation_metric=getattr(competition, "evaluation_metric", "") or "",
        team_count=getattr(competition, "team_count", None),
        max_daily_submissions=getattr(competition, "max_daily_submissions", None),
        is_kernels_submissions_only=bool(getattr(competition, "is_kernels_submissions_only", False)),
        submissions_disabled=bool(getattr(competition, "submissions_disabled", False)),
        source=source,
    )


def merge_listing(existing: CompetitionListing | None, new: CompetitionListing) -> CompetitionListing:
    if existing is None:
        return new
    return CompetitionListing(
        slug=new.slug,
        title=new.title or existing.title,
        url=new.url or existing.url,
        description=new.description or existing.description,
        category=new.category if new.category and new.category != "Unspecified" else existing.category,
        reward=new.reward or existing.reward,
        evaluation_metric=new.evaluation_metric or existing.evaluation_metric,
        team_count=new.team_count if new.team_count is not None else existing.team_count,
        max_daily_submissions=(
            new.max_daily_submissions if new.max_daily_submissions is not None else existing.max_daily_submissions
        ),
        is_kernels_submissions_only=new.is_kernels_submissions_only or existing.is_kernels_submissions_only,
        submissions_disabled=new.submissions_disabled or existing.submissions_disabled,
        source=f"{existing.source},{new.source}" if new.source not in existing.source.split(",") else existing.source,
    )


def competition_slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) >= 2 and segments[0] == "competitions":
        return segments[1].strip().lower()
    raise ValueError(f"Unable to parse competition slug from '{url}'.")


def html_to_text(html: str) -> str:
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def extract_competition_slugs_from_html(html: str) -> list[str]:
    slugs = sorted(set(re.findall(r'href="/competitions/([A-Za-z0-9][A-Za-z0-9._-]*)"', html)))
    return [slug for slug in slugs if slug not in {"overview", "rules", "leaderboard", "discussion", "data"}]


def extract_competition_titles_from_landing_html(html: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r'href="/competitions/(?P<slug>[A-Za-z0-9][A-Za-z0-9._-]*)"[^>]*>.*?<div[^>]*>(?P<title>[^<]+)</div>',
        re.S,
    )
    matches: list[tuple[str, str]] = []
    for match in pattern.finditer(html):
        slug = match.group("slug")
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        if slug and title:
            matches.append((slug, title))
    return matches


def infer_category_from_landing_text(text: str, slug: str) -> str:
    lowered = text.lower()
    slug_text = slug.replace("-", " ").lower()
    for category in ("featured", "research", "community", "getting started", "playground"):
        marker = category if category != "getting started" else "getting started"
        if slug_text in lowered and marker in lowered:
            if category == "getting started":
                return "Getting Started"
            return category.title()
    return "Unspecified"


def extract_suffixes_from_text(text: str) -> list[str]:
    matches = re.findall(
        rf"\.({_DIRECT_SUBMISSION_EXTENSION_PATTERN})\b",
        text,
        flags=re.I,
    )
    return normalize_suffixes([f".{match.lower()}" for match in matches])


def extract_named_submission_suffixes(text: str) -> list[str]:
    matches = re.findall(
        rf"(?<![A-Za-z0-9_])submission\.({_DIRECT_SUBMISSION_EXTENSION_PATTERN})\b",
        text,
        flags=re.I,
    )
    return normalize_suffixes([f".{match.lower()}" for match in matches])


def infer_suffixes_from_keywords(text: str) -> list[str]:
    suffixes: list[str] = []
    for pattern, suffix in SUBMISSION_TOKEN_PATTERNS:
        if pattern.search(text) and suffix not in suffixes:
            suffixes.append(suffix)
    if ".txt" in suffixes and any(suffix in DOCUMENT_SUFFIXES for suffix in suffixes):
        suffixes = [suffix for suffix in suffixes if suffix != ".txt"]
    compression_suffix = compression_suffix_from_keywords(text)
    if compression_suffix is None:
        return suffixes
    compressed = [
        f"{suffix}{compression_suffix}" for suffix in suffixes if suffix in COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES
    ]
    return compressed or suffixes


def compression_suffix_from_keywords(text: str) -> str | None:
    for pattern, suffix in COMPRESSION_TOKEN_PATTERNS:
        if pattern.search(text):
            return suffix
    return None


def infer_columns_from_bullets(text: str, *, allow_fallback: bool = False) -> list[str]:
    if not allow_fallback:
        return []
    columns: list[str] = []
    for line in text.splitlines():
        match = _BULLET_COLUMN_RE.match(line)
        if not match:
            continue
        name = match.group("name").strip()
        if name and name not in columns:
            columns.append(name)
    return columns if len(columns) >= 2 else []


def normalize_suffixes(suffixes: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for suffix in suffixes:
        item = suffix.strip().lower()
        if not item.startswith("."):
            item = f".{item}"
        if item not in ordered:
            ordered.append(item)
    return drop_shadowed_submission_suffixes(ordered)


def extract_evidence_snippet(html: str, marker_text: str) -> str:
    if not marker_text:
        return extract_keyword_window(html_to_text(html))
    plain = html_to_text(html)
    marker = truncate_text(re.sub(r"\s+", " ", marker_text), 200)
    idx = plain.lower().find(marker[:80].lower()) if marker else -1
    if idx < 0:
        return extract_keyword_window(plain)
    start = max(0, idx - 200)
    end = min(len(plain), idx + 1000)
    return plain[start:end]


def extract_keyword_window(text: str, *, window: int = 700) -> str:
    plain = text.strip()
    if not plain:
        return ""
    candidates: list[str] = []
    for pattern in _KEYWORD_WINDOW_PATTERNS:
        for match in pattern.finditer(plain):
            start = max(0, match.start() - 160)
            end = min(len(plain), match.end() + window)
            candidate = plain[start:end].strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return select_best_submission_block(candidates)


def select_best_submission_block(candidates: Iterable[str]) -> str:
    best_block = ""
    best_score = -1
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized:
            continue
        score = 0
        lowered = normalized.lower()
        if "submission file format" in lowered:
            score += 6
        if "sample submission" in lowered:
            score += 4
        if "submission format" in lowered:
            score += 3
        score += len(extract_suffixes_from_text(normalized)) * 3
        score += len(infer_suffixes_from_keywords(normalized)) * 2
        if infer_csv_example_columns(normalized):
            score += 5
        if "too many requests" in lowered:
            score -= 10
        if len(normalized.splitlines()) >= 3:
            score += 1
        if score > best_score:
            best_block = normalized
            best_score = score
    return best_block


def truncate_text(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    append_jsonl_record(path, row, ensure_ascii=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl_records(path, rows, ensure_ascii=True)


def load_raw_records(path: Path) -> list[CrawlRecord]:
    return [CrawlRecord(**row) for row in load_jsonl(path)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_jsonl_records(path)]


def write_csv(path: Path, records: list[CrawlRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slug",
        "title",
        "competition_type",
        "submission_mode",
        "required_artifact",
        "artifact_class",
        "artifact_container",
        "detected_extensions",
        "detected_columns",
        "delimiter",
        "is_code_competition",
        "evidence_url",
        "extraction_confidence",
        "reward",
        "evaluation_metric",
        "team_count",
        "max_daily_submissions",
        "discovery_source",
        "crawled_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["detected_extensions"] = "|".join(record.detected_extensions)
            row["detected_columns"] = "|".join(record.detected_columns)
            writer.writerow({key: row[key] for key in fieldnames})


def build_summary(records: list[CrawlRecord]) -> dict[str, Any]:
    by_mode: dict[str, int] = {}
    by_artifact_class: dict[str, int] = {}
    by_extension: dict[str, int] = {}
    review_required: list[dict[str, str]] = []
    for record in records:
        by_mode[record.submission_mode] = by_mode.get(record.submission_mode, 0) + 1
        by_artifact_class[record.artifact_class] = by_artifact_class.get(record.artifact_class, 0) + 1
        for suffix in record.detected_extensions:
            by_extension[suffix] = by_extension.get(suffix, 0) + 1
        supported, reason = assess_submission_format_support(record)
        if not supported:
            review_required.append({"slug": record.slug, "reason": reason})
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "competition_count": len(records),
        "submission_modes": dict(sorted(by_mode.items())),
        "artifact_classes": dict(sorted(by_artifact_class.items())),
        "extensions": dict(sorted(by_extension.items())),
        "supported_competition_count": len(records) - len(review_required),
        "review_required_count": len(review_required),
        "review_required": review_required,
    }


def assess_submission_format_support(record: CrawlRecord) -> tuple[bool, str]:
    if record.submission_mode == "external_repository_submission":
        return False, "external_submission_requires_manual_workflow"
    if record.submission_mode not in _SUPPORTED_SUBMISSION_MODES:
        return False, f"unsupported_or_unknown_submission_mode:{record.submission_mode}"
    if record.submission_mode in {
        "code_competition_runtime_submission",
        "notebook_output_submission",
        "writeup_submission",
    }:
        return True, "supported"
    if record.artifact_class == ARTIFACT_CLASS_UNKNOWN:
        return False, "unknown_artifact_class"
    if not record.detected_extensions:
        return False, "direct_upload_extension_unknown"
    supported_suffixes = set(all_submission_output_suffixes())
    unsupported = [suffix for suffix in record.detected_extensions if suffix not in supported_suffixes]
    if unsupported:
        return False, f"unsupported_extensions:{'|'.join(unsupported)}"
    return True, "supported"


def extract_build_version_from_client_token(token: str) -> str:
    payload = token.split(".")[1]
    padding = "=" * (-len(payload) % 4)
    data = json.loads(base64.urlsafe_b64decode((payload + padding).encode()).decode())
    return str(data["bld"])
