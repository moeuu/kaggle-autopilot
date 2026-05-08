from __future__ import annotations

import html
import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from kagglebot.bootstrap_reference_inputs import stage_reference_notebook_inputs
from kagglebot.competition import rules_url_for_slug
from kagglebot.competition_policy import NotebookSelectionPolicy, load_competition_policy
from kagglebot.kaggle_api import (
    DownloadProgressCallback,
    download_competition,
    download_dataset,
    kernels_pull,
    list_competition_kernels,
)
from kagglebot.knowledge import (
    build_improve_template,
    build_kernel_fix_template,
    build_plan_and_initial_prompt,
    ensure_taxonomy,
    record_competition_profile,
    resolve_similar_improvements,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.solver.io import ensure_sample_submission
from kagglebot.submission_format import (
    extract_submission_section,
    load_submission_format_hint,
    parse_submission_format,
)
from kagglebot.types import BootstrapMeta, PlanConfig, RulesInfo
from kagglebot.validators import safe_extract_zip


def bootstrap_competition(
    *,
    slug: str,
    competition_url: str | None,
    paths: CompetitionPaths,
    knowledge_paths: KnowledgePaths,
    rules_source: str = "none",
    rules_file: Path | None = None,
    download: bool = False,
    quiet: bool = True,
    force: bool = False,
    dry_run: bool = False,
    download_progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    """
    Prepare local workspace directories and write meta + plan files.
    Does not join competitions or accept rules automatically.
    """
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.prompts_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.submissions_dir.mkdir(parents=True, exist_ok=True)
    paths.kernels_dir.mkdir(parents=True, exist_ok=True)

    rules_url = rules_url_for_slug(slug)
    rules_info = RulesInfo(url=rules_url, source=rules_source, file=str(rules_file) if rules_file else None)
    meta = BootstrapMeta(
        slug=slug,
        created_at=datetime.now(UTC).isoformat(),
        rules=rules_info,
    )
    _write_json(paths.meta_path, meta.to_dict(), force=force)
    paths.rules_url_path.write_text(rules_url + "\n", encoding="utf-8")

    _capture_rules(paths, rules_source=rules_source, rules_file=rules_file, rules_url=rules_url, dry_run=dry_run)
    if not paths.rules_md_path.exists():
        _write_rules_markdown(
            paths,
            f"Rules content not provided. See: {rules_url}\n",
        )
    if not paths.overview_md_path.exists():
        _write_overview_markdown(paths, f"Overview content not provided. See: {rules_url}\n")
    if not paths.data_md_path.exists():
        _write_data_markdown(paths, f"Data content not provided. See: {rules_url}\n")
    if not paths.submission_format_md_path.exists():
        _write_submission_format_markdown(paths, f"Submission format not provided. See: {rules_url}\n")
    competition_base_url = f"https://www.kaggle.com/competitions/{slug}"
    if not paths.code_md_path.exists():
        _write_code_markdown(paths, f"Code content not provided. See: {competition_base_url}/code\n")
    if not paths.models_md_path.exists():
        _write_models_markdown(paths, f"Models content not provided. See: {competition_base_url}/models\n")
    if not paths.discussion_md_path.exists():
        _write_discussion_markdown(paths, f"Discussion content not provided. See: {competition_base_url}/discussions\n")
    _write_plan(paths, force=force)

    if download:
        if dry_run:
            paths.context_dir.joinpath("download_skipped.txt").write_text(
                "DRY RUN: download skipped.\n", encoding="utf-8"
            )
        else:
            download_competition(
                slug,
                paths.data_dir,
                force=True,
                quiet=quiet,
                progress_callback=download_progress_callback,
            )
            _unzip_downloads(paths.data_dir)
            if rules_source == "url":
                _capture_rules(
                    paths,
                    rules_source=rules_source,
                    rules_file=rules_file,
                    rules_url=rules_url,
                    dry_run=dry_run,
                )

    profile = _write_dataset_profile(paths)
    _cache_sample_submission(paths)
    _mirror_sample_submission_to_data(paths)
    stage_reference_notebook_inputs(
        paths=paths,
        slug=slug,
        download=download,
        quiet=quiet,
        dry_run=dry_run,
        download_competition_fn=download_competition,
        download_dataset_fn=download_dataset,
        download_kernel_fn=kernels_pull,
    )

    taxonomy = ensure_taxonomy(knowledge_paths)
    similar = resolve_similar_improvements(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        tags=profile.get("tags", []),
    )
    _write_knowledge_hints(paths, similar)
    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug=slug,
        competition_url=competition_url,
        profile=profile,
    )

    _write_prompts(
        paths=paths,
        slug=slug,
        rules_url=rules_url,
        profile=profile,
        taxonomy=taxonomy,
        similar_improvements=similar,
    )

    return paths.meta_path


def _write_json(path: Path, payload: dict[str, object], *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_rules(
    paths: CompetitionPaths,
    *,
    rules_source: str,
    rules_file: Path | None,
    rules_url: str,
    dry_run: bool,
) -> None:
    if rules_source == "none":
        return
    if rules_source in {"url", "fetch"}:
        if dry_run:
            (paths.context_dir / "rules_url_skipped.txt").write_text(
                "DRY RUN: rules download skipped.\n", encoding="utf-8"
            )
            return
        pages = _fetch_competition_pages(slug=paths.slug, rules_url=rules_url)
        if pages:
            rules_md = _select_page_markdown(pages, names=("rules",), keywords=("rule",))
            overview_md = _build_overview_markdown(pages)
            data_md = _select_page_markdown(
                pages,
                names=("data-description", "data"),
                keywords=("data", "dataset"),
            )
            if rules_md:
                _write_rules_markdown(paths, _normalize_page_markdown(rules_md))
            normalized_overview = ""
            normalized_data = ""
            if overview_md:
                normalized_overview = _normalize_page_markdown(overview_md)
                _write_overview_markdown(paths, normalized_overview)
            if data_md:
                normalized_data = _normalize_page_markdown(data_md)
                _write_data_markdown(paths, normalized_data)

            submission_section = None
            if normalized_data:
                submission_section = _extract_usable_submission_section(normalized_data)
            if not submission_section and normalized_overview:
                submission_section = _extract_usable_submission_section(normalized_overview)
            if not submission_section:
                for page in pages:
                    content = page.get("content")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    section = _extract_usable_submission_section(_normalize_page_markdown(content))
                    if section:
                        submission_section = section
                        break
            if submission_section:
                _write_submission_format_markdown(paths, submission_section)
        community_snapshots = _fetch_competition_community_snapshots(paths=paths, slug=paths.slug)
        if "code" in community_snapshots:
            _write_code_markdown(paths, community_snapshots["code"])
        if "models" in community_snapshots:
            _write_models_markdown(paths, community_snapshots["models"])
        if "discussion" in community_snapshots:
            _write_discussion_markdown(paths, community_snapshots["discussion"])
        return
    if rules_source != "file":
        raise ValueError(f"Unknown rules source: {rules_source}")
    if not rules_file:
        raise ValueError("--rules-file is required when --rules-source file.")
    if dry_run:
        (paths.context_dir / "rules_file_skipped.txt").write_text("DRY RUN: rules file skipped.\n", encoding="utf-8")
        return
    _write_rules_from_file(paths, rules_file, rules_url=rules_url)


class _RulesHtmlToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._lines: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
        if tag == "li":
            self._current.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if data:
            self._current.append(data)

    def _flush(self) -> None:
        if not self._current:
            return
        line = " ".join(" ".join(self._current).split())
        if line:
            self._lines.append(line)
        self._current = []

    def to_markdown(self) -> str:
        self._flush()
        return "\n".join(self._lines).strip()


def _extract_usable_submission_section(markdown: str) -> str | None:
    section = extract_submission_section(markdown)
    if not section:
        return None
    hint = parse_submission_format(section)
    if hint.columns:
        return section
    return None


def _write_rules_from_file(paths: CompetitionPaths, rules_file: Path, *, rules_url: str) -> None:
    suffix = rules_file.suffix.lower()
    text = rules_file.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        paths.rules_html_path.write_text(text, encoding="utf-8")
        parser = _RulesHtmlToMarkdown()
        parser.feed(text)
        md_text = parser.to_markdown()
        _write_rules_markdown(paths, md_text or f"Rules URL: {rules_url}")
        return
    _write_rules_markdown(paths, text or f"Rules URL: {rules_url}")


def _fetch_competition_pages(*, slug: str, rules_url: str, timeout: int = 10) -> list[dict[str, object]]:
    competition_url = (
        f"https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName={slug}"
    )
    payload = _fetch_json_with_retry(competition_url, timeout=timeout)
    if payload is None:
        return []
    competition_id = None
    if isinstance(payload, dict):
        competition_id = payload.get("id") or payload.get("competition", {}).get("id")
    if not competition_id:
        return []

    pages_url = f"https://www.kaggle.com/api/i/competitions.PageService/ListPages?competitionId={competition_id}"
    pages_payload = _fetch_json_with_retry(pages_url, timeout=timeout)
    if pages_payload is None:
        return []
    pages = pages_payload.get("pages", []) if isinstance(pages_payload, dict) else []
    if not isinstance(pages, list):
        return []
    return [page for page in pages if isinstance(page, dict)]


_TAB_URL_SUFFIXES = {
    "code": ("code",),
    "models": ("models", "model"),
    "discussion": ("discussions", "discussion"),
}

_CODE_NOTEBOOK_LIMIT_ENV = "KAGGLEBOT_CODE_NOTEBOOK_LIMIT"
_CODE_SCORE_DIRECTION_ENV = "KAGGLEBOT_CODE_SCORE_DIRECTION"
_DISCUSSION_THREAD_LIMIT_ENV = "KAGGLEBOT_DISCUSSION_THREAD_LIMIT"
_CODE_NOTEBOOK_DEFAULT_DOWNLOAD = 50
_CODE_NOTEBOOK_MAX_DOWNLOAD = 50


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag.lower() != "a":
            return
        href_value = None
        for key, value in attrs:
            if str(key).lower() == "href" and value:
                href_value = str(value)
                break
        if href_value:
            self._href = href_value
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href and data:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a":
            return
        if self._href:
            label = " ".join(" ".join(self._text_parts).split()).strip()
            self.links.append({"href": self._href, "text": label})
        self._href = None
        self._text_parts = []


def _read_int_env(name: str, *, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _fetch_competition_community_snapshots(*, paths: CompetitionPaths, slug: str, timeout: int = 10) -> dict[str, str]:
    snapshots: dict[str, str] = {}
    code_snapshot = _fetch_and_download_competition_code(paths=paths, slug=slug, timeout=timeout)
    if code_snapshot:
        snapshots["code"] = code_snapshot
    models_snapshot = _fetch_competition_tab_snapshot(slug=slug, tab="models", timeout=timeout)
    if models_snapshot:
        snapshots["models"] = models_snapshot
    discussion_snapshot = _fetch_and_download_competition_discussions(paths=paths, slug=slug, timeout=timeout)
    if discussion_snapshot:
        snapshots["discussion"] = discussion_snapshot
    return snapshots


def _fetch_competition_tab_snapshot(*, slug: str, tab: str, timeout: int = 10) -> str | None:
    suffixes = _TAB_URL_SUFFIXES.get(tab, (tab,))
    if not suffixes:
        return None
    base_url = f"https://www.kaggle.com/competitions/{slug}"
    for suffix in suffixes:
        url = f"{base_url}/{suffix}"
        page_html = _fetch_text_with_retry(url, timeout=timeout)
        if not page_html:
            continue
        extracted = _extract_text_snapshot_from_html(page_html, max_lines=180, max_chars=12000)
        if not extracted:
            continue
        title = tab.replace("_", " ").title()
        fetched_at = datetime.now(UTC).isoformat()
        return (
            f"# {title} Snapshot\n\n"
            f"Source URL: {url}\n"
            f"Fetched at: {fetched_at}\n\n"
            f"## Extracted Content\n\n{extracted}\n"
        )
    return None


_KAGGLE_LINK_RE = re.compile(r"^https?://(?:www\.)?kaggle\.com(?P<path>/.*)$", re.IGNORECASE)
_CODE_HREF_RE = re.compile(r"^/code/(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)(?:/.*)?$")
_SCORE_PATTERNS = (
    re.compile(r"(?:public\s+score|private\s+score|leaderboard\s+score|lb|score)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.I),
    re.compile(r"(?:cv|oof)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.I),
)
_KERNEL_ROW_SCORE_KEYS = (
    "score",
    "publicscore",
    "privatescore",
    "leaderboardscore",
    "bestscore",
)
_TITLE_SCORE_HINT_RE = re.compile(
    r"(?:public\s+lb|private\s+lb|public\s+score|private\s+score|lb|score)\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
    re.I,
)
_SCORE_DIRECTION_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbrier\b", re.IGNORECASE), "minimize"),
    (re.compile(r"\blog\s*loss\b|\blogloss\b|\bcross[-\s]?entropy\b", re.IGNORECASE), "minimize"),
    (re.compile(r"\brmse\b|\brmsle\b|\bmae\b|\bmape\b|\bmse\b", re.IGNORECASE), "minimize"),
    (re.compile(r"\bauc\b|\broc[-\s]?auc\b", re.IGNORECASE), "maximize"),
    (re.compile(r"\baccuracy\b|\bf1\b|\bprecision\b|\brecall\b|\br2\b", re.IGNORECASE), "maximize"),
)
_LEAKY_NOTEBOOK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bleak\b", re.IGNORECASE),
    re.compile(r"\bperfect\s+score\b", re.IGNORECASE),
    re.compile(r"\bguaranteed\b", re.IGNORECASE),
    re.compile(r"\bpublic\s*lb\b", re.IGNORECASE),
    re.compile(r"\bstage\s*1\b", re.IGNORECASE),
    re.compile(r"\b0(?:\.0+)?\s*(?:lb|score)\b", re.IGNORECASE),
)


def _normalize_kaggle_href(href: str) -> str:
    match = _KAGGLE_LINK_RE.match(href.strip())
    path = match.group("path") if match else href.strip()
    if not path.startswith("/"):
        path = "/" + path
    path = path.split("#", 1)[0]
    path = path.split("?", 1)[0]
    return path


def _extract_code_candidates(*, html_text: str) -> list[dict[str, object]]:
    parser = _AnchorCollector()
    parser.feed(html_text)
    parser.close()

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for source_order, link in enumerate(parser.links):
        href = _normalize_kaggle_href(link.get("href", ""))
        match = _CODE_HREF_RE.match(href)
        if not match:
            continue
        kernel_id = f"{match.group('user')}/{match.group('slug')}"
        if kernel_id in seen:
            continue
        seen.add(kernel_id)
        score = _infer_score_near_href(html_text=html_text, href=href)
        title = (link.get("text") or "").strip() or match.group("slug").replace("-", " ")
        entries.append(
            {
                "kernel_id": kernel_id,
                "title": title,
                "url": f"https://www.kaggle.com{href}",
                "score": score,
                "total_votes": 0,
                "last_run_time": None,
                "source_order": source_order,
            }
        )
    return entries


def _sort_code_candidates(
    candidates: list[dict[str, object]],
    *,
    score_direction: str,
    selection_policy: NotebookSelectionPolicy | None = None,
) -> list[dict[str, object]]:
    """Sort notebook candidates by score, with votes/recency fallbacks."""

    minimize_score = score_direction == "minimize"

    def key(candidate: dict[str, object]) -> tuple[int, float, int, float, int]:
        raw_score = candidate.get("score")
        has_score = isinstance(raw_score, (int, float))
        score_value = float(raw_score) if has_score else 0.0
        score_boost = _candidate_keyword_boost(candidate, selection_policy)
        adjusted_score = score_value - score_boost if minimize_score else score_value + score_boost
        if has_score:
            score_key = adjusted_score if minimize_score else -adjusted_score
        else:
            score_key = score_boost if minimize_score else -score_boost
        vote_key = -_parse_vote_count(candidate.get("total_votes"))
        recency_key = -_parse_last_run_epoch(candidate.get("last_run_time"))
        order = int(candidate.get("source_order") or 0)
        return (0 if has_score else 1, score_key, vote_key, recency_key, order)

    return sorted(candidates, key=key)


def _is_likely_leak_or_placeholder_notebook(candidate: dict[str, object]) -> bool:
    """Heuristic filter for leak-style or placeholder leaderboard notebooks."""
    text_parts = [
        str(candidate.get("kernel_id") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("summary") or ""),
    ]
    text = " ".join(text_parts)
    if any(pattern.search(text) for pattern in _LEAKY_NOTEBOOK_PATTERNS):
        return True
    raw_score = candidate.get("score")
    if isinstance(raw_score, (int, float)) and abs(float(raw_score)) <= 1e-12:
        lowered = text.lower()
        if "lb" in lowered or "score" in lowered:
            return True
    return False


def _select_required_reference_notebook(entries: list[dict[str, object]]) -> dict[str, object] | None:
    """Pick a mandatory reference notebook, preferring non-leak candidates."""
    if not entries:
        return None
    for entry in entries:
        if not _is_likely_leak_or_placeholder_notebook(entry):
            return entry
    return entries[0]


def _candidate_text(candidate: dict[str, object]) -> str:
    return " ".join(
        [
            str(candidate.get("kernel_id") or ""),
            str(candidate.get("title") or ""),
            str(candidate.get("summary") or ""),
        ]
    ).lower()


def _candidate_keyword_boost(candidate: dict[str, object], selection_policy: NotebookSelectionPolicy | None) -> float:
    if selection_policy is None:
        return 0.0
    text = _candidate_text(candidate)
    boost = 0.0
    for keyword, weight in selection_policy.keyword_boosts.items():
        clean = keyword.strip().lower()
        if clean and clean in text:
            boost += float(weight)
    return boost


def _candidate_keyword_hits(candidate: dict[str, object], keywords: tuple[str, ...]) -> tuple[int, list[str]]:
    text = _candidate_text(candidate)
    hits = [keyword for keyword in keywords if keyword.strip() and keyword.strip().lower() in text]
    return len(hits), hits


def _select_keyword_reference_notebook(
    entries: list[dict[str, object]],
    *,
    keywords: tuple[str, ...],
    excluded_kernel_ids: set[str] | None = None,
) -> dict[str, object] | None:
    excluded = excluded_kernel_ids or set()
    ranked: list[tuple[int, int, float, int, dict[str, object]]] = []
    for order, entry in enumerate(entries):
        kernel_id = str(entry.get("kernel_id") or "").strip()
        if not kernel_id or kernel_id in excluded or _is_likely_leak_or_placeholder_notebook(entry):
            continue
        hit_count, _hits = _candidate_keyword_hits(entry, keywords)
        if hit_count <= 0:
            continue
        ranked.append(
            (
                hit_count,
                _parse_vote_count(entry.get("total_votes")),
                _parse_last_run_epoch(entry.get("last_run_time")),
                -order,
                entry,
            )
        )
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][-1]


def _select_reference_notebooks(
    entries: list[dict[str, object]],
    *,
    selection_policy: NotebookSelectionPolicy | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, str | None]:
    required = _select_required_reference_notebook(entries)
    ensemble: dict[str, object] | None = None
    selection_reason: str | None = None
    if selection_policy is None:
        return required, None, None

    policy_required = _select_keyword_reference_notebook(
        entries,
        keywords=selection_policy.required_reference_keywords,
    )
    if policy_required is not None:
        if required is None or policy_required != required:
            selection_reason = "Policy keyword match prioritized a competition-specific execution baseline."
        required = policy_required

    required_kernel_id = str(required.get("kernel_id") or "").strip() if isinstance(required, dict) else ""
    ensemble = _select_keyword_reference_notebook(
        entries,
        keywords=selection_policy.ensemble_reference_keywords,
        excluded_kernel_ids={required_kernel_id} if required_kernel_id else set(),
    )
    return required, ensemble, selection_reason


def _parse_vote_count(raw: object) -> int:
    """Parse notebook vote count with resilient fallback."""
    if isinstance(raw, int):
        return max(raw, 0)
    if isinstance(raw, float):
        return max(int(raw), 0)
    if not isinstance(raw, str):
        return 0
    text = raw.strip().replace(",", "")
    if not text:
        return 0
    try:
        return max(int(text), 0)
    except ValueError:
        return 0


def _parse_last_run_epoch(raw: object) -> float:
    """Convert last-run timestamps to epoch seconds for stable sorting."""
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str) and raw.strip():
        text = raw.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return 0.0
    else:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _parse_score_value(raw: object) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_kernel_row_score(row: dict[str, str]) -> float | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for key in _KERNEL_ROW_SCORE_KEYS:
        score = _parse_score_value(lowered.get(key))
        if score is not None:
            return score
    title = str(lowered.get("title") or "")
    match = _TITLE_SCORE_HINT_RE.search(title)
    if match:
        score = _parse_score_value(match.group(1))
        if score is not None:
            return score
    return None


def _list_competition_code_candidates_from_cli(*, slug: str) -> list[dict[str, object]]:
    try:
        rows = list_competition_kernels(
            slug,
            page=1,
            page_size=200,
            sort_by="scoreDescending",
            kernel_type="notebook",
            dry_run=False,
        )
    except Exception:  # noqa: BLE001
        return []

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        kernel_id = str(row.get("ref") or "").strip()
        if not kernel_id or "/" not in kernel_id:
            continue
        if kernel_id in seen:
            continue
        seen.add(kernel_id)
        entries.append(
            {
                "kernel_id": kernel_id,
                "title": str(row.get("title") or kernel_id),
                "url": f"https://www.kaggle.com/code/{kernel_id}",
                "score": _parse_kernel_row_score(row),
                "total_votes": _parse_vote_count(row.get("totalVotes")),
                "last_run_time": str(row.get("lastRunTime") or ""),
                "source_order": index,
            }
        )
    return entries


def _resolve_code_score_direction(*, paths: CompetitionPaths) -> str:
    """Resolve notebook score direction: minimize or maximize."""
    explicit = os.getenv(_CODE_SCORE_DIRECTION_ENV, "").strip().lower()
    if explicit in {"minimize", "maximize"}:
        return explicit

    spec_direction = _read_direction_from_json(paths.context_dir / "evaluation_spec.json", ("direction",))
    if spec_direction is not None:
        return spec_direction

    plan_direction = _read_direction_from_json(paths.plan_path, ("target_direction", "direction"))
    if plan_direction is not None:
        return plan_direction

    inferred = _infer_direction_from_context_files(paths)
    if inferred is not None:
        return inferred
    return "maximize"


def _read_direction_from_json(path: Path, keys: tuple[str, ...]) -> str | None:
    """Read direction from a JSON file if present and valid."""
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in keys:
        raw = str(payload.get(key) or "").strip().lower()
        if raw in {"minimize", "maximize"}:
            return raw
    return None


def _infer_direction_from_context_files(paths: CompetitionPaths) -> str | None:
    """Infer score direction from competition context text."""
    text = "\n".join(
        [
            paths.rules_md_path.read_text(encoding="utf-8", errors="ignore") if paths.rules_md_path.exists() else "",
            paths.overview_md_path.read_text(encoding="utf-8", errors="ignore")
            if paths.overview_md_path.exists()
            else "",
            paths.data_md_path.read_text(encoding="utf-8", errors="ignore") if paths.data_md_path.exists() else "",
        ]
    ).strip()
    if not text:
        return None

    lowered = text.lower()
    if re.search(r"\blower\b[^.\n]*\bbetter\b|\bminimi[sz]e\b", lowered):
        return "minimize"
    if re.search(r"\bhigher\b[^.\n]*\bbetter\b|\bmaximi[sz]e\b", lowered):
        return "maximize"

    earliest: tuple[int, str] | None = None
    for pattern, direction in _SCORE_DIRECTION_HINTS:
        match = pattern.search(text)
        if not match:
            continue
        found = (match.start(), direction)
        if earliest is None or found[0] < earliest[0]:
            earliest = found
    if earliest is not None:
        return earliest[1]
    return None


def _infer_score_near_href(*, html_text: str, href: str) -> float | None:
    index = html_text.find(href)
    if index < 0:
        encoded = href.replace("/", r"\/")
        index = html_text.find(encoded)
    if index < 0:
        return None
    windows = [
        html_text[index : min(len(html_text), index + 1800)],
        html_text[max(0, index - 900) : min(len(html_text), index + 2500)],
    ]
    for window in windows:
        window_text = _extract_text_snapshot_from_html(window, max_lines=80, max_chars=2200)
        for pattern in _SCORE_PATTERNS:
            match = pattern.search(window_text)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if abs(value) <= 1000:
                return value
    return None


def _safe_kernel_dir_name(kernel_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", kernel_id.replace("/", "__")).strip("_")


def _choose_notebook_source_file(directory: Path) -> Path | None:
    for suffix in (".ipynb", ".py"):
        files = sorted(directory.glob(f"*{suffix}"))
        if files:
            return files[0]
    return None


def _truncate_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _normalize_cell_source(source: object) -> str:
    if isinstance(source, list):
        return "".join(str(chunk) for chunk in source)
    if isinstance(source, str):
        return source
    return ""


def _summarize_notebook_source(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return ""
        cells = payload.get("cells", [])
        if not isinstance(cells, list):
            return ""
        excerpts: list[str] = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            cell_type = str(cell.get("cell_type") or "")
            source_text = _normalize_cell_source(cell.get("source"))
            compact = _truncate_text(source_text, 240)
            if not compact:
                continue
            lowered = compact.lower()
            if cell_type == "markdown" or any(
                token in lowered
                for token in ("train", "fold", "augment", "ensemble", "pretrained", "score", "metric", "model")
            ):
                excerpts.append(compact)
            if len(excerpts) >= 8:
                break
        if not excerpts:
            return ""
        return "\n".join(f"- {line}" for line in excerpts)

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    excerpts: list[str] = []
    for raw_line in content.splitlines():
        compact = raw_line.strip()
        if not compact:
            continue
        lowered = compact.lower()
        if any(token in lowered for token in ("train", "fold", "augment", "ensemble", "pretrained", "score", "metric")):
            excerpts.append(_truncate_text(compact, 200))
        if len(excerpts) >= 8:
            break
    if not excerpts:
        return ""
    return "\n".join(f"- {line}" for line in excerpts)


def _fetch_and_download_competition_code(*, paths: CompetitionPaths, slug: str, timeout: int) -> str | None:
    code_url = f"https://www.kaggle.com/competitions/{slug}/code"
    html_text = _fetch_text_with_retry(code_url, timeout=timeout) or ""
    competition_policy = load_competition_policy(paths)
    selection_policy = competition_policy.notebook_selection if competition_policy.active else None
    candidates = _list_competition_code_candidates_from_cli(slug=slug)
    candidate_source = "kaggle kernels list --competition --sort-by scoreDescending"
    if not candidates:
        if not html_text:
            return _fetch_competition_tab_snapshot(slug=slug, tab="code", timeout=timeout)
        candidates = _extract_code_candidates(html_text=html_text)
        candidate_source = "competition code page HTML"

    score_direction = _resolve_code_score_direction(paths=paths)
    candidates = _sort_code_candidates(
        candidates,
        score_direction=score_direction,
        selection_policy=selection_policy,
    )
    limit = _read_int_env(
        _CODE_NOTEBOOK_LIMIT_ENV,
        default=_CODE_NOTEBOOK_DEFAULT_DOWNLOAD,
        min_value=1,
        max_value=_CODE_NOTEBOOK_MAX_DOWNLOAD,
    )
    selected = candidates[:limit]

    paths.code_notebooks_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for candidate in selected:
        kernel_id = str(candidate.get("kernel_id") or "").strip()
        if not kernel_id:
            continue
        notebook_dir = paths.code_notebooks_dir / _safe_kernel_dir_name(kernel_id)
        notebook_dir.mkdir(parents=True, exist_ok=True)
        pull_error = ""
        try:
            kernels_pull(kernel_id, notebook_dir, slug=slug, dry_run=False, metadata=True)
        except Exception as exc:  # noqa: BLE001
            pull_error = str(exc)

        source_file = _choose_notebook_source_file(notebook_dir)
        summary = _summarize_notebook_source(source_file)
        if summary:
            (notebook_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")

        entries.append(
            {
                "kernel_id": kernel_id,
                "title": str(candidate.get("title") or kernel_id),
                "url": str(candidate.get("url") or ""),
                "score": candidate.get("score"),
                "total_votes": _parse_vote_count(candidate.get("total_votes")),
                "last_run_time": str(candidate.get("last_run_time") or ""),
                "local_dir": str(notebook_dir),
                "source_file": str(source_file) if source_file else None,
                "summary": summary,
                "download_error": pull_error,
            }
        )

    required_entry, ensemble_entry, policy_selection_reason = _select_reference_notebooks(
        entries,
        selection_policy=selection_policy,
    )
    index_payload = {
        "source_url": code_url,
        "candidate_source": candidate_source,
        "score_direction": score_direction,
        "fetched_at": datetime.now(UTC).isoformat(),
        "notebook_count": len(entries),
        "top_kernel_id": str(entries[0]["kernel_id"]) if entries else "",
        "required_reference_kernel_id": str(required_entry.get("kernel_id") or "") if required_entry else "",
        "ensemble_reference_kernel_id": str(ensemble_entry.get("kernel_id") or "") if ensemble_entry else "",
        "policy_tags": list(competition_policy.archetype_tags),
        "required_capabilities": list(competition_policy.required_capabilities),
        "notebooks": entries,
    }
    paths.code_notebooks_index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    lines = [
        "# Code Notebook Snapshot",
        "",
        f"Source URL: {code_url}",
        f"Candidate source: {candidate_source}",
        f"Score direction: {score_direction}",
        f"Fetched at: {index_payload['fetched_at']}",
        "",
    ]
    if entries:
        top_entry = entries[0]
        top_score = top_entry.get("score")
        top_score_text = f"{float(top_score):.6f}" if isinstance(top_score, (int, float)) else "unknown"
        required_is_top = bool(required_entry) and required_entry == top_entry
        required_score = required_entry.get("score") if required_entry else None
        required_score_text = f"{float(required_score):.6f}" if isinstance(required_score, (int, float)) else "unknown"
        lines.extend(
            [
                "## Top-ranked Notebook (Raw ranking)",
                f"- kernel_id: {top_entry['kernel_id']}",
                f"- title: {top_entry['title']}",
                f"- notebook_score: {top_score_text}",
                f"- total_votes: {int(top_entry.get('total_votes') or 0)}",
                "",
            ]
        )
        if required_entry:
            lines.extend(
                [
                    "## Required Reference Notebook (Execution baseline)",
                    f"- kernel_id: {required_entry['kernel_id']}",
                    f"- title: {required_entry['title']}",
                    f"- notebook_score: {required_score_text}",
                    f"- total_votes: {int(required_entry.get('total_votes') or 0)}",
                    (
                        "- selection_reason: Top-ranked notebook is usable."
                        if required_is_top
                        else (
                            f"- selection_reason: {policy_selection_reason}"
                            if policy_selection_reason
                            else (
                                "- selection_reason: Top-ranked notebook appears leak-like/placeholder; "
                                "using the highest-ranked non-leak notebook."
                            )
                        )
                    ),
                    "- instruction: Treat this notebook as a mandatory baseline reference before adding improvements.",
                    "",
                ]
            )
        if ensemble_entry:
            ensemble_score = ensemble_entry.get("score")
            ensemble_score_text = (
                f"{float(ensemble_score):.6f}" if isinstance(ensemble_score, (int, float)) else "unknown"
            )
            lines.extend(
                [
                    "## Ensemble Reference Notebook (Blend blueprint)",
                    f"- kernel_id: {ensemble_entry['kernel_id']}",
                    f"- title: {ensemble_entry['title']}",
                    f"- notebook_score: {ensemble_score_text}",
                    f"- total_votes: {int(ensemble_entry.get('total_votes') or 0)}",
                    "- instruction: Treat this notebook as an ensemble/feature blueprint "
                    "after the execution baseline is reproduced.",
                    "",
                ]
            )

        lines.append("## Downloaded Notebooks")
        lines.append("")
        for index, entry in enumerate(entries, start=1):
            score = entry.get("score")
            score_text = f"{float(score):.6f}" if isinstance(score, (int, float)) else "unknown"
            lines.extend(
                [
                    f"### {index}. {entry['title']}",
                    f"- kernel_id: {entry['kernel_id']}",
                    f"- notebook_score: {score_text}",
                    f"- total_votes: {int(entry.get('total_votes') or 0)}",
                    f"- url: {entry['url']}",
                    f"- local_dir: {entry['local_dir']}",
                ]
            )
            if entry.get("source_file"):
                lines.append(f"- source_file: {entry['source_file']}")
            if entry.get("download_error"):
                lines.append(f"- download_error: {entry['download_error']}")
            summary = str(entry.get("summary") or "").strip()
            if summary:
                lines.append("Notebook summary excerpt:")
                lines.append(summary)
            lines.append("")
    else:
        lines.append("No downloadable competition notebooks were discovered from the code tab.")
        lines.append("")

    raw_snapshot = _extract_text_snapshot_from_html(html_text, max_lines=140, max_chars=8000) if html_text else ""
    if raw_snapshot:
        lines.append("## Code Tab Text Snapshot")
        lines.append("")
        lines.append(raw_snapshot)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _extract_discussion_candidates(*, slug: str, html_text: str) -> list[dict[str, str]]:
    parser = _AnchorCollector()
    parser.feed(html_text)
    parser.close()
    pattern = re.compile(rf"^/competitions/{re.escape(slug)}/discussion/(?P<thread_id>\d+)(?:/.*)?$", re.IGNORECASE)
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in parser.links:
        href = _normalize_kaggle_href(link.get("href", ""))
        match = pattern.match(href)
        if not match:
            continue
        thread_id = match.group("thread_id")
        if thread_id in seen:
            continue
        seen.add(thread_id)
        title = (link.get("text") or "").strip() or f"Discussion {thread_id}"
        entries.append(
            {
                "thread_id": thread_id,
                "title": title,
                "href": href,
                "url": f"https://www.kaggle.com{href}",
            }
        )
    return entries


_HTML_TITLE_RE = re.compile(r"<title>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL)


def _extract_html_title(html_text: str) -> str:
    match = _HTML_TITLE_RE.search(html_text)
    if not match:
        return ""
    title = html.unescape(match.group("title"))
    title = " ".join(title.split()).strip()
    return re.sub(r"\s*\|\s*Kaggle\s*$", "", title, flags=re.IGNORECASE).strip()


def _fetch_competition_topics_from_api(*, slug: str, timeout: int) -> list[dict[str, object]]:
    competition_payload = _fetch_json_with_retry(
        f"https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName={slug}",
        timeout=timeout,
    )
    if not competition_payload:
        return []
    forum_id = competition_payload.get("forumId")
    if not isinstance(forum_id, int):
        return []
    topics_payload = _fetch_json_with_retry(
        f"https://www.kaggle.com/api/i/discussions.DiscussionsService/GetTopicListByForumId?forumId={forum_id}",
        timeout=timeout,
    )
    if not topics_payload:
        return []
    topics = topics_payload.get("topics", [])
    if not isinstance(topics, list):
        return []
    rows: list[dict[str, object]] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("id")
        if not isinstance(topic_id, int):
            continue
        rows.append(topic)
    return rows


def _fetch_and_download_competition_discussions(*, paths: CompetitionPaths, slug: str, timeout: int) -> str | None:
    listing_url = f"https://www.kaggle.com/competitions/{slug}/discussions"
    api_topics = _fetch_competition_topics_from_api(slug=slug, timeout=timeout)
    html_text = _fetch_text_with_retry(listing_url, timeout=timeout) or ""

    candidates: list[dict[str, str]]
    candidate_source: str
    if api_topics:
        candidates = []
        seen: set[str] = set()
        for topic in api_topics:
            topic_id = str(topic.get("id") or "").strip()
            if not topic_id or topic_id in seen:
                continue
            seen.add(topic_id)
            title = str(topic.get("title") or topic.get("subject") or f"Discussion {topic_id}")
            href = f"/competitions/{slug}/discussion/{topic_id}"
            candidates.append(
                {
                    "thread_id": topic_id,
                    "title": title,
                    "href": href,
                    "url": f"https://www.kaggle.com{href}",
                }
            )
        candidate_source = "discussions.DiscussionsService/GetTopicListByForumId"
    else:
        if not html_text:
            return _fetch_competition_tab_snapshot(slug=slug, tab="discussion", timeout=timeout)
        candidates = _extract_discussion_candidates(slug=slug, html_text=html_text)
        candidate_source = "competition discussions page HTML"

    limit = _read_int_env(_DISCUSSION_THREAD_LIMIT_ENV, default=20, min_value=1, max_value=100)
    selected = candidates[:limit]

    paths.discussion_threads_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for candidate in selected:
        thread_url = str(candidate.get("url") or "")
        thread_html = _fetch_text_with_retry(thread_url, timeout=timeout)
        if not thread_html:
            rows.append(
                {
                    "thread_id": candidate.get("thread_id"),
                    "title": candidate.get("title"),
                    "url": thread_url,
                    "local_path": None,
                    "excerpt": "",
                    "download_error": "Unable to fetch thread page.",
                }
            )
            continue

        thread_title = _extract_html_title(thread_html) or str(candidate.get("title") or "Discussion")
        thread_text = _extract_text_snapshot_from_html(thread_html, max_lines=500, max_chars=30000)
        thread_file = paths.discussion_threads_dir / f"{candidate['thread_id']}.md"
        thread_file.write_text(
            "\n".join(
                [
                    f"# {thread_title}",
                    "",
                    f"Source URL: {thread_url}",
                    "",
                    "## Extracted Content",
                    "",
                    thread_text or "Thread text extraction returned empty content.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        excerpt = _truncate_text(thread_text, 360) if thread_text else ""
        rows.append(
            {
                "thread_id": candidate.get("thread_id"),
                "title": thread_title,
                "url": thread_url,
                "local_path": str(thread_file),
                "excerpt": excerpt,
                "download_error": "",
            }
        )

    index_payload = {
        "source_url": listing_url,
        "candidate_source": candidate_source,
        "fetched_at": datetime.now(UTC).isoformat(),
        "thread_count": len(rows),
        "threads": rows,
    }
    paths.discussion_threads_index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    lines = [
        "# Discussion Threads Snapshot",
        "",
        f"Source URL: {listing_url}",
        f"Candidate source: {candidate_source}",
        f"Fetched at: {index_payload['fetched_at']}",
        "",
    ]
    if rows:
        lines.append("## Thread Digest")
        lines.append("")
        for index, row in enumerate(rows, start=1):
            lines.extend(
                [
                    f"### {index}. {row['title']}",
                    f"- thread_id: {row['thread_id']}",
                    f"- url: {row['url']}",
                    f"- local_path: {row['local_path'] or 'unavailable'}",
                ]
            )
            if row.get("download_error"):
                lines.append(f"- download_error: {row['download_error']}")
            excerpt = str(row.get("excerpt") or "").strip()
            if excerpt:
                lines.append(f"- excerpt: {excerpt}")
            lines.append("")
    else:
        lines.append("No discussion threads were discovered from the discussions tab.")
        lines.append("")

    listing_snapshot = _extract_text_snapshot_from_html(html_text, max_lines=120, max_chars=6000) if html_text else ""
    if listing_snapshot:
        lines.append("## Discussion Tab Text Snapshot")
        lines.append("")
        lines.append(listing_snapshot)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _fetch_text_with_retry(url: str, *, timeout: int, attempts: int = 3) -> str | None:
    for attempt in range(attempts):
        try:
            return _fetch_text(url, timeout=timeout)
        except (urllib.error.URLError, OSError, TimeoutError, UnicodeDecodeError):
            if attempt < attempts - 1:
                time.sleep(1.0)
                continue
            return None
    return None


def _fetch_text(url: str, *, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "kagglebot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        encoding = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(encoding, errors="ignore")


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _extract_text_snapshot_from_html(html_text: str, *, max_lines: int, max_chars: int) -> str:
    if not html_text.strip():
        return ""
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html_text)
    without_comments = _HTML_COMMENT_RE.sub(" ", without_scripts)
    stripped = _HTML_TAG_RE.sub("\n", without_comments)
    plain = html.unescape(stripped)

    lines: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for raw_line in plain.splitlines():
        compact = " ".join(raw_line.split()).strip()
        if len(compact) < 4:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        bullet = f"- {compact}"
        lines.append(bullet)
        total_chars += len(bullet) + 1
        if len(lines) >= max_lines or total_chars >= max_chars:
            break

    if not lines:
        return ""
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n... (truncated)"
    return text


def _select_page_markdown(
    pages: list[dict[str, object]],
    *,
    names: tuple[str, ...],
    keywords: tuple[str, ...],
) -> str | None:
    name_set = {name.lower() for name in names}
    for page in pages:
        name = str(page.get("name") or "").lower()
        if name in name_set:
            content = page.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    for page in pages:
        name = str(page.get("name") or "").lower()
        if any(keyword in name for keyword in keywords):
            content = page.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    for page in pages:
        title = str(page.get("title") or "").lower()
        if any(keyword in title for keyword in keywords):
            content = page.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _build_overview_markdown(pages: list[dict[str, object]]) -> str | None:
    excluded = {"rules", "data-description", "data"}
    sections: list[str] = []
    for page in pages:
        name = str(page.get("name") or "").strip()
        if not name:
            continue
        if name.lower() in excluded:
            continue
        content = page.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        title = str(page.get("title") or name).strip()
        body = _normalize_page_markdown(content)
        sections.append(f"## {title}\n\n{body}")
    if not sections:
        return None
    return "\n\n".join(sections).strip()


def _looks_like_html(text: str) -> bool:
    lowered = text.lower()
    return any(
        tag in lowered
        for tag in (
            "<p",
            "<div",
            "<br",
            "<li",
            "<h1",
            "<h2",
            "<h3",
            "<ul",
            "<ol",
            "<table",
            "<span",
            "<a ",
            "<strong",
            "<em",
        )
    )


def _normalize_page_markdown(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if _looks_like_html(cleaned):
        parser = _RulesHtmlToMarkdown()
        parser.feed(cleaned)
        converted = parser.to_markdown()
        if converted:
            if len(cleaned) < 400 or len(converted) >= int(len(cleaned) * 0.6):
                return converted
    return cleaned


def _fetch_json_with_retry(url: str, *, timeout: int, attempts: int = 3) -> dict[str, object] | None:
    for attempt in range(attempts):
        try:
            return _fetch_json(url, timeout=timeout)
        except (urllib.error.URLError, ValueError, json.JSONDecodeError):
            if attempt < attempts - 1:
                time.sleep(1.0)
                continue
            return None
    return None


def _fetch_json(url: str, *, timeout: int) -> dict[str, object]:
    req = urllib.request.Request(url, headers={"User-Agent": "kagglebot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("Unexpected JSON payload.")
    return payload


def _write_rules_markdown(paths: CompetitionPaths, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        normalized = "Rules content unavailable."
    if not normalized.endswith("\n"):
        normalized += "\n"
    paths.rules_md_path.write_text(normalized, encoding="utf-8")


def _write_overview_markdown(paths: CompetitionPaths, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        normalized = "Overview content unavailable."
    if not normalized.endswith("\n"):
        normalized += "\n"
    paths.overview_md_path.write_text(normalized, encoding="utf-8")


def _write_data_markdown(paths: CompetitionPaths, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        normalized = "Data content unavailable."
    if not normalized.endswith("\n"):
        normalized += "\n"
    paths.data_md_path.write_text(normalized, encoding="utf-8")


def _write_submission_format_markdown(paths: CompetitionPaths, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        normalized = "Submission format unavailable."
    if not normalized.endswith("\n"):
        normalized += "\n"
    paths.submission_format_md_path.write_text(normalized, encoding="utf-8")


def _write_code_markdown(paths: CompetitionPaths, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        normalized = "Code content unavailable."
    if not normalized.endswith("\n"):
        normalized += "\n"
    paths.code_md_path.write_text(normalized, encoding="utf-8")


def _write_models_markdown(paths: CompetitionPaths, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        normalized = "Models content unavailable."
    if not normalized.endswith("\n"):
        normalized += "\n"
    paths.models_md_path.write_text(normalized, encoding="utf-8")


def _write_discussion_markdown(paths: CompetitionPaths, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        normalized = "Discussion content unavailable."
    if not normalized.endswith("\n"):
        normalized += "\n"
    paths.discussion_md_path.write_text(normalized, encoding="utf-8")


def _pages_need_refresh(paths: CompetitionPaths, rules_url: str) -> bool:
    def needs_refresh(path: Path, markers: tuple[str, ...]) -> bool:
        if not path.exists():
            return True
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return True
        return any(text.startswith(marker) for marker in markers)

    if needs_refresh(paths.rules_md_path, ("Rules content not provided.", f"Rules URL: {rules_url}")):
        return True
    if needs_refresh(paths.overview_md_path, ("Overview content not provided.",)):
        return True
    if needs_refresh(paths.data_md_path, ("Data content not provided.",)):
        return True
    if needs_refresh(paths.submission_format_md_path, ("Submission format not provided.",)):
        return True
    if needs_refresh(paths.code_md_path, ("Code content not provided.",)):
        return True
    if needs_refresh(paths.models_md_path, ("Models content not provided.",)):
        return True
    if needs_refresh(paths.discussion_md_path, ("Discussion content not provided.",)):
        return True
    return False


def _write_plan(paths: CompetitionPaths, *, force: bool) -> None:
    plan = PlanConfig()
    _write_json(paths.plan_path, plan.to_dict(), force=force)


def _write_prompts(
    *,
    paths: CompetitionPaths,
    slug: str,
    rules_url: str,
    profile: dict[str, object],
    taxonomy: dict[str, object],
    similar_improvements: list[dict[str, object]],
) -> None:
    initial_prompt = build_plan_and_initial_prompt(
        slug=slug,
        rules_url=rules_url,
        profile=profile,
        taxonomy=taxonomy,
        similar_improvements=similar_improvements,
    )
    improve_template = build_improve_template()
    kernel_fix_template = build_kernel_fix_template()
    paths.codex_plan_and_implement_prompt.write_text(initial_prompt, encoding="utf-8")
    paths.codex_improve_template.write_text(improve_template, encoding="utf-8")
    paths.codex_kernel_fix_template.write_text(kernel_fix_template, encoding="utf-8")


def _write_dataset_profile(paths: CompetitionPaths) -> dict[str, object]:
    from kagglebot.knowledge import build_dataset_profile

    profile = build_dataset_profile(paths.data_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def _cache_sample_submission(paths: CompetitionPaths) -> None:
    format_hint = load_submission_format_hint(paths.submission_format_md_path)
    candidate = ensure_sample_submission(paths.data_dir)
    if candidate is None or not candidate.exists():
        if paths.sample_submission_path.exists() and not paths.sample_submission_head_path.exists():
            _write_sample_head(paths.sample_submission_path, paths.sample_submission_head_path)
        return
    if candidate.suffix.lower() == ".csv":
        shutil.copy2(candidate, paths.sample_submission_path)
    else:
        try:
            frame = _read_table(candidate, format_hint=format_hint)
        except Exception:
            return
        frame.to_csv(paths.sample_submission_path, index=False)
    _write_sample_head(paths.sample_submission_path, paths.sample_submission_head_path)


def _mirror_sample_submission_to_data(paths: CompetitionPaths) -> None:
    source = paths.sample_submission_path
    destination = paths.data_dir / "sample_submission.csv"
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if source.read_bytes() == destination.read_bytes():
                return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _find_tabular_files(root: Path) -> list[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def _read_table(path: Path, *, format_hint=None):
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - optional dependency in tests
        raise
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix == ".tsv":
        return _read_delimited_table(path, sep="\t", format_hint=format_hint)
    if suffix == ".txt":
        hint_sep = getattr(format_hint, "delimiter", None) if format_hint is not None else None
        return _read_delimited_table(path, sep=hint_sep or "\t", format_hint=format_hint)
    if suffix == ".csv":
        return _read_delimited_table(path, sep=",", format_hint=format_hint)
    return pd.read_csv(path)


def _read_delimited_table(path: Path, *, sep: str, format_hint=None):
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - optional dependency in tests
        raise
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        pass
    columns = getattr(format_hint, "columns", None) if format_hint is not None else None
    expected_cols = len(columns) if columns else _infer_column_count(path, sep)
    if expected_cols is None:
        return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
    names = columns or [f"col{i}" for i in range(expected_cols)]
    filtered = _filter_delimited_text(path, sep=sep, expected_cols=expected_cols)
    if filtered is None:
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            names=names,
            usecols=list(range(expected_cols)),
            engine="python",
            on_bad_lines="skip",
        )
    else:
        df = pd.read_csv(
            io.StringIO(filtered),
            sep=sep,
            header=None,
            names=names,
            engine="python",
        )
    if columns and not df.empty and list(df.iloc[0].astype(str)) == columns:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _infer_column_count(path: Path, sep: str, max_lines: int = 200) -> int | None:
    from collections import Counter

    counts: Counter[int] = Counter()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                counts[len(line.rstrip("\n").split(sep))] += 1
                if sum(counts.values()) >= max_lines:
                    break
    except OSError:
        return None
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _filter_delimited_text(path: Path, *, sep: str, expected_cols: int, max_lines: int | None = None) -> str | None:
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if len(line.rstrip("\n").split(sep)) != expected_cols:
                    continue
                lines.append(line)
                if max_lines is not None and len(lines) >= max_lines:
                    break
    except OSError:
        return None
    if not lines:
        return None
    return "".join(lines)


def _write_sample_head(sample_path: Path, head_path: Path, rows: int = 5) -> None:
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - optional dependency in tests
        return
    try:
        sample = pd.read_csv(sample_path, nrows=rows)
    except Exception:
        return
    head_path.write_text(sample.head(rows).to_csv(index=False), encoding="utf-8")


def _write_knowledge_hints(paths: CompetitionPaths, similar: list[dict[str, object]]) -> None:
    lines = ["# Knowledge Hints", ""]
    if not similar:
        lines.append("No similar competitions found in knowledge base.")
    else:
        lines.append("Similar competitions and what improved score:")
        lines.append("")
        for item in similar:
            slug = item.get("slug", "unknown")
            overlap = item.get("overlap", 0)
            summary = item.get("summary", "No summary recorded.")
            lines.append(f"- {slug} ({overlap} tag overlap): {summary}")
    paths.knowledge_hints_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unzip_downloads(data_dir: Path) -> None:
    for zip_path in data_dir.glob("*.zip"):
        safe_extract_zip(zip_path, data_dir)
