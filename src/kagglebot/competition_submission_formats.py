from __future__ import annotations

import base64
import csv
import json
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kaggle.api.kaggle_api_extended import KaggleApi

from kagglebot.exec_utils import run_command
from kagglebot.json_utils import write_json_object
from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_NOTEBOOK_OUTPUT,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_UNKNOWN,
)
from kagglebot.submission_format import SubmissionFormatHint, extract_submission_section, parse_submission_format

_DEFAULT_SEARCH_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_DEFAULT_PAGE_SIZE = 20
_SUBMISSION_NOTEBOOK_RE = re.compile(r"\b(submit|submission|upload)\b.{0,80}\b(notebook|code)\b", re.I | re.S)
_CODE_COMPETITION_RE = re.compile(r"\b(code competition|kernel submissions only|notebook submission)\b", re.I)
_BULLET_COLUMN_RE = re.compile(r"^\s*[-*]\s*`?(?P<name>[A-Za-z0-9_. -]+?)`?\s*$")
_SECTION_BLOCK_RE = re.compile(
    r"(?is)"
    r"(submission file format|submission format|make a submission|how to submit|submit predictions)"
    r"(.*?)(?=\n[A-Z][^\n]{0,80}\n|\Z)"
)
_KEYWORD_WINDOW_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsubmission file format\b", re.I),
    re.compile(r"\bsubmission format\b", re.I),
    re.compile(r"\bsample submission\b", re.I),
    re.compile(r"\bsubmit predictions\b", re.I),
    re.compile(r"\bsubmit\b.{0,120}\b(csv|zip|jsonl|json|tsv|parquet|txt|notebook|code)\b", re.I | re.S),
    re.compile(r"\b(upload|submission)\b.{0,120}\b(csv|zip|jsonl|json|tsv|parquet|txt|notebook|code)\b", re.I | re.S),
)
_KEYWORD_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bzip(?:ped)?\b", re.I), ".zip"),
    (re.compile(r"\bjsonl\b|\bndjson\b", re.I), ".jsonl"),
    (re.compile(r"\bparquet\b", re.I), ".parquet"),
    (re.compile(r"\btsv\b|\btab[-\s]*separated\b", re.I), ".tsv"),
    (re.compile(r"\bcsv\b", re.I), ".csv"),
    (re.compile(r"\bjson\b", re.I), ".json"),
    (re.compile(r"\btxt\b|\btext\s+file\b", re.I), ".txt"),
)
_NOISY_SUFFIX_CONTEXT_MARKERS = (
    "topology json",
    "metadata json",
    "example json",
    "sample json",
    "description json",
    "text description",
    "txt description",
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or ChromeDomFetcher()

    listings = discover_competitions(
        fetcher=fetcher,
        max_prefix_depth=max_prefix_depth,
        max_pages_per_search=max_pages_per_search,
        search_alphabet=search_alphabet,
    )
    if max_competitions is not None:
        listings = listings[:max_competitions]

    raw_path = output_dir / "raw_submission_formats.jsonl"
    processed_slugs = load_processed_slugs(raw_path) if resume else set()
    records: list[CrawlRecord] = []
    if resume and raw_path.exists():
        records.extend(load_raw_records(raw_path))

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

    write_jsonl(output_dir / "discovered_competitions.jsonl", [asdict(item) for item in listings])
    write_csv(output_dir / "normalized_submission_formats.csv", records)
    summary = build_summary(records)
    write_json_object(output_dir / "summary.json", summary)
    return summary


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
            competitions = api.competitions_list(group=group, page=page, sort_by="recentlyCreated")
            if not competitions:
                break
            for competition in competitions:
                listing = listing_from_api_competition(competition, source=f"api-group:{group}")
                listings[listing.slug] = merge_listing(listings.get(listing.slug), listing)
            if len(competitions) < _DEFAULT_PAGE_SIZE:
                break

    for category in ("all", "featured", "research", "gettingStarted", "playground"):
        for page in range(1, max_pages_per_search + 1):
            competitions = api.competitions_list(category=category, page=page, sort_by="recentlyCreated")
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
            competitions = api.competitions_list(
                search=prefix,
                page=page,
                category="all",
                sort_by="recentlyCreated",
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

    chosen_text = overview_section or rules_section
    chosen_url = overview_url if overview_section or not rules_section else rules_url
    hint = (
        parse_submission_format(chosen_text)
        if chosen_text
        else SubmissionFormatHint(None, None, None, artifact_class=ARTIFACT_CLASS_UNKNOWN, artifact_container=None)
    )
    expected_suffixes = normalize_suffixes(
        hint.expected_suffixes or extract_suffixes_from_text(chosen_text) or infer_suffixes_from_keywords(chosen_text)
    )
    expected_suffixes = filter_noisy_suffixes(chosen_text, expected_suffixes)
    submission_mode = infer_submission_mode(listing, chosen_text, expected_suffixes)
    artifact_class = infer_artifact_class(listing=listing, hint=hint, text=chosen_text, suffixes=expected_suffixes)
    artifact_container = infer_artifact_container(artifact_class=artifact_class, suffixes=expected_suffixes)
    required_artifact = infer_required_artifact(
        submission_mode=submission_mode,
        suffixes=expected_suffixes,
        artifact_class=artifact_class,
    )
    confidence = infer_confidence(chosen_text, expected_suffixes, listing.is_kernels_submissions_only)
    evidence_html = overview_html if chosen_url == overview_url else rules_html
    evidence_marker = overview_section or rules_section or chosen_text

    return CrawlRecord(
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


def find_submission_text_block(text: str) -> str:
    candidates: list[str] = []
    section = extract_submission_section(text)
    if section and len(section.splitlines()) > 1:
        candidates.append(section)
    for match in _SECTION_BLOCK_RE.finditer(text):
        candidate = match.group(0).strip()
        if len(candidate.splitlines()) > 1:
            candidates.append(candidate)
    preferred = select_best_submission_block(candidates)
    if preferred:
        return preferred
    return extract_keyword_window(text)


def infer_submission_mode(listing: CompetitionListing, text: str, suffixes: list[str]) -> str:
    if listing.is_kernels_submissions_only:
        return "code_competition_runtime_submission"
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
    if hint.artifact_class and hint.artifact_class != ARTIFACT_CLASS_UNKNOWN:
        return hint.artifact_class
    lowered = text.lower()
    if ".zip" in suffixes:
        if any(marker in lowered for marker in ("weights", "inference script", ".pt", ".pth", ".ckpt", ".h5")):
            return ARTIFACT_CLASS_BUNDLE
        return ARTIFACT_CLASS_MULTI_FILE_ZIP
    if suffixes:
        if any(suffix in {".csv", ".tsv", ".parquet"} for suffix in suffixes):
            return ARTIFACT_CLASS_TABULAR
        return ARTIFACT_CLASS_SINGLE_FILE
    return ARTIFACT_CLASS_UNKNOWN


def infer_artifact_container(*, artifact_class: str, suffixes: list[str]) -> str | None:
    if artifact_class in {ARTIFACT_CLASS_BUNDLE, ARTIFACT_CLASS_MULTI_FILE_ZIP}:
        return "zip"
    if artifact_class in {ARTIFACT_CLASS_TABULAR, ARTIFACT_CLASS_SINGLE_FILE, ARTIFACT_CLASS_NOTEBOOK_OUTPUT}:
        return "file"
    if ".zip" in suffixes:
        return "zip"
    if suffixes:
        return "file"
    return None


def infer_required_artifact(submission_mode: str, suffixes: list[str], artifact_class: str) -> str:
    if submission_mode == "code_competition_runtime_submission":
        return "Kaggle notebook output"
    if submission_mode == "notebook_output_submission":
        return "Kaggle notebook submission"
    if artifact_class == ARTIFACT_CLASS_BUNDLE:
        return "ZIP bundle containing model assets and inference code"
    if artifact_class == ARTIFACT_CLASS_MULTI_FILE_ZIP:
        return "ZIP archive containing multiple prediction files"
    if artifact_class == ARTIFACT_CLASS_SINGLE_FILE and suffixes:
        return f"{suffixes[0]} single file"
    if ".zip" in suffixes:
        return "ZIP archive"
    if suffixes:
        return f"{suffixes[0]} file"
    return "unknown"


def filter_noisy_suffixes(text: str, suffixes: list[str]) -> list[str]:
    lowered = text.lower()
    if not any(marker in lowered for marker in _NOISY_SUFFIX_CONTEXT_MARKERS):
        return suffixes
    return [suffix for suffix in suffixes if suffix not in {".json", ".txt"}]


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
    matches = re.findall(r"\.(csv|tsv|txt|parquet|jsonl|json|zip)\b", text, flags=re.I)
    return normalize_suffixes([f".{match.lower()}" for match in matches])


def infer_suffixes_from_keywords(text: str) -> list[str]:
    suffixes: list[str] = []
    for pattern, suffix in _KEYWORD_SUFFIX_PATTERNS:
        if pattern.search(text) and suffix not in suffixes:
            suffixes.append(suffix)
    return suffixes


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
    return ordered


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
    for pattern in _KEYWORD_WINDOW_PATTERNS:
        match = pattern.search(plain)
        if not match:
            continue
        start = max(0, match.start() - 160)
        end = min(len(plain), match.end() + window)
        return plain[start:end].strip()
    return ""


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_processed_slugs(path: Path) -> set[str]:
    return {row["slug"] for row in load_jsonl(path) if isinstance(row.get("slug"), str)}


def load_raw_records(path: Path) -> list[CrawlRecord]:
    return [CrawlRecord(**row) for row in load_jsonl(path)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    for record in records:
        by_mode[record.submission_mode] = by_mode.get(record.submission_mode, 0) + 1
        by_artifact_class[record.artifact_class] = by_artifact_class.get(record.artifact_class, 0) + 1
        for suffix in record.detected_extensions:
            by_extension[suffix] = by_extension.get(suffix, 0) + 1
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "competition_count": len(records),
        "submission_modes": dict(sorted(by_mode.items())),
        "artifact_classes": dict(sorted(by_artifact_class.items())),
        "extensions": dict(sorted(by_extension.items())),
    }


def extract_build_version_from_client_token(token: str) -> str:
    payload = token.split(".")[1]
    padding = "=" * (-len(payload) % 4)
    data = json.loads(base64.urlsafe_b64decode((payload + padding).encode()).decode())
    return str(data["bld"])
