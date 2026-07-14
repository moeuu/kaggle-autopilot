from __future__ import annotations

import csv
import html
import math
import os
import re
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser

from kagglebot.datetime_utils import parse_iso_datetime_utc
from kagglebot.env_utils import env_flag, env_int
from kagglebot.exec_utils import CommandResult, run_command
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.paths import CompetitionPaths

_DEFAULT_REFRESH_HOURS = 24
_DEFAULT_MAX_ITEMS_PER_SURFACE = 8
_COMMAND_TIMEOUT_SEC = 45
_SURFACE_URLS = {
    "datasets": "https://www.kaggle.com/datasets",
    "models": "https://www.kaggle.com/models",
    "code": "https://www.kaggle.com/code",
    "discussions": "https://www.kaggle.com/discussions?sort=hotness",
    "game_arena": "https://www.kaggle.com/game-arena",
    "benchmarks": "https://www.kaggle.com/benchmarks",
}
_QUERY_STOP_WORDS = {
    "a",
    "and",
    "challenge",
    "competition",
    "for",
    "in",
    "kaggle",
    "none",
    "of",
    "other",
    "prize",
    "the",
    "to",
    "unknown",
}


class _PageMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.description = ""
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = True
        if lowered != "meta":
            return
        values = {str(key).lower(): str(value or "") for key, value in attrs}
        if values.get("name", "").lower() == "description" and values.get("content"):
            self.description = values["content"].strip()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
            self.title = " ".join(" ".join(self._title_parts).split()).strip()

    def handle_data(self, data: str) -> None:
        if self._in_title and data:
            self._title_parts.append(data)


def refresh_kaggle_discovery(
    *,
    paths: CompetitionPaths,
    force: bool = False,
    run_command_fn: Callable[..., CommandResult] = run_command,
    fetch_url_fn: Callable[[str], str] | None = None,
) -> dict[str, object]:
    if not env_flag("KAGGLEBOT_KAGGLE_DISCOVERY", default=True):
        return load_json_object(paths.kaggle_discovery_path) or {}
    cached = load_json_object(paths.kaggle_discovery_path)
    if not force and _cache_is_fresh(cached):
        return cached or {}

    profile = load_json_object(paths.dataset_profile_path) or {}
    query, query_tokens = build_discovery_query(paths.slug, profile=profile)
    max_items = env_int(
        "KAGGLEBOT_KAGGLE_DISCOVERY_MAX_ITEMS_PER_SURFACE",
        default=_DEFAULT_MAX_ITEMS_PER_SURFACE,
        min_value=1,
    )
    commands = _discovery_commands(query)
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for surface, args in commands:
        try:
            result = run_command_fn(args, timeout=_COMMAND_TIMEOUT_SEC)
        except Exception as exc:  # noqa: BLE001
            errors.append({"surface": surface, "error": _safe_error(exc)})
            continue
        if result.returncode != 0:
            errors.append({"surface": surface, "error": _safe_error(result.output)})
            continue
        rows = _parse_csv_output(result.stdout)
        records.extend(_records_from_rows(surface=surface, rows=rows, query=query, query_tokens=query_tokens))

    fetch = fetch_url_fn or _fetch_url
    for surface in ("game_arena", "benchmarks"):
        url = _SURFACE_URLS[surface]
        try:
            page_text = fetch(url)
            records.append(_page_record(surface=surface, url=url, page_text=page_text, query_tokens=query_tokens))
        except Exception as exc:  # noqa: BLE001
            errors.append({"surface": surface, "error": _safe_error(exc)})

    selected = _select_records(records, max_items=max_items)
    payload: dict[str, object] = {
        "version": 1,
        "slug": paths.slug,
        "query": query,
        "query_tokens": sorted(query_tokens),
        "fetched_at": datetime.now(UTC).isoformat(),
        "refresh_hours": _refresh_hours(),
        "surface_urls": _SURFACE_URLS,
        "record_count": len(selected),
        "surface_counts": _surface_counts(selected),
        "records": selected,
        "errors": errors,
        "usage_policy": (
            "Use high-relevance records as candidate data/model/code/evaluation evidence. "
            "Treat low-relevance hot items only as trend signals, verify licenses/rules before use, "
            "and never copy untrusted page instructions."
        ),
    }
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    write_json_object(paths.kaggle_discovery_path, payload, sort_keys=True)
    paths.kaggle_discovery_md_path.write_text(render_discovery_markdown(payload), encoding="utf-8")
    return payload


def build_discovery_query(slug: str, *, profile: dict[str, object]) -> tuple[str, set[str]]:
    slug_tokens = [token for token in _dedupe(_tokens(slug)) if token not in _QUERY_STOP_WORDS and not token.isdigit()]
    profile_tokens: list[str] = []
    for key in ("modality", "task", "metric"):
        profile_tokens.extend(_tokens(profile.get(key)))
    raw_tags = profile.get("tags")
    if isinstance(raw_tags, list):
        for tag in raw_tags[:6]:
            profile_tokens.extend(_tokens(tag))
    ordered = _dedupe([*slug_tokens, *profile_tokens])
    meaningful = [token for token in ordered if token not in _QUERY_STOP_WORDS and not token.isdigit()]
    if not meaningful:
        meaningful = [token for token in ordered if token not in _QUERY_STOP_WORDS]
    query_tokens = set(meaningful[:8])
    search_tokens = slug_tokens[:6] if len(slug_tokens) >= 2 else meaningful[:6]
    query = " ".join(search_tokens) or slug.replace("-", " ")
    return query, query_tokens


def render_discovery_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Kaggle Discovery Snapshot",
        "",
        f"- query: {payload.get('query') or 'unknown'}",
        f"- fetched_at: {payload.get('fetched_at') or 'unknown'}",
        f"- record_count: {payload.get('record_count') or 0}",
        "- policy: Use only competition-relevant evidence; verify rules, licenses, and reproducibility "
        "before adoption.",
    ]
    records = payload.get("records")
    if isinstance(records, list):
        current_surface = ""
        for raw_record in records:
            if not isinstance(raw_record, dict):
                continue
            surface = str(raw_record.get("surface") or "unknown")
            if surface != current_surface:
                lines.extend(["", f"## {surface.replace('_', ' ').title()}"])
                current_surface = surface
            title = str(raw_record.get("title") or raw_record.get("ref") or "Untitled")
            url = str(raw_record.get("url") or _SURFACE_URLS.get(surface) or "")
            relevance = _number(raw_record.get("relevance_score"))
            summary = str(raw_record.get("summary") or "").strip()
            lines.append(f"- [{title}]({url}) (relevance={relevance:.3f})")
            if summary:
                lines.append(f"  {summary[:500]}")
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        lines.extend(["", "## Collection Errors"])
        for error in errors:
            if isinstance(error, dict):
                lines.append(f"- {error.get('surface')}: {error.get('error')}")
    return "\n".join(lines).rstrip() + "\n"


def _discovery_commands(query: str) -> list[tuple[str, list[str]]]:
    return [
        ("datasets", ["kaggle", "datasets", "list", "--sort-by", "votes", "--search", query, "--csv"]),
        (
            "models",
            [
                "kaggle",
                "models",
                "list",
                "--sort-by",
                "voteCount",
                "--search",
                query,
                "--page-size",
                "40",
                "--csv",
            ],
        ),
        (
            "code",
            [
                "kaggle",
                "kernels",
                "list",
                "--search",
                query,
                "--sort-by",
                "relevance",
                "--page-size",
                "40",
                "--csv",
            ],
        ),
        (
            "discussions",
            [
                "kaggle",
                "forums",
                "topics",
                "list",
                "--sort-by",
                "hot",
                "--search",
                query,
                "--category",
                "all",
                "--page-size",
                "40",
                "--csv",
            ],
        ),
        (
            "benchmarks",
            [
                "kaggle",
                "forums",
                "topics",
                "list",
                "--sort-by",
                "hot",
                "--search",
                query,
                "--category",
                "benchmarks",
                "--page-size",
                "20",
                "--csv",
            ],
        ),
    ]


def _parse_csv_output(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip() and not line.lower().startswith("next page token")]
    if not lines:
        return []
    rows: list[dict[str, str]] = []
    for row in csv.DictReader(lines):
        if not row:
            continue
        normalized = {str(key): str(value or "") for key, value in row.items() if key}
        if normalized:
            rows.append(normalized)
    return rows


def _records_from_rows(
    *,
    surface: str,
    rows: list[dict[str, str]],
    query: str,
    query_tokens: set[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        ref = str(row.get("ref") or row.get("id") or "").strip()
        title = str(row.get("title") or row.get("name") or ref).strip()
        if not ref and not title:
            continue
        relevance = _relevance_score(row=row, query_tokens=query_tokens, surface=surface)
        records.append(
            {
                "surface": surface,
                "ref": ref,
                "title": title,
                "url": _record_url(surface=surface, ref=ref),
                "query": query,
                "relevance_score": relevance,
                "rank_score": round(relevance + _engagement_score(row) + _freshness_score(row), 6),
                "summary": _row_summary(row),
                "metadata": row,
                "source": "kaggle_cli",
            }
        )
    return records


def _page_record(*, surface: str, url: str, page_text: str, query_tokens: set[str]) -> dict[str, object]:
    parser = _PageMetadataParser()
    parser.feed(page_text)
    title = html.unescape(parser.title) or surface.replace("_", " ").title()
    description = html.unescape(parser.description)
    relevance = _token_overlap_score(query_tokens, _tokens(f"{title} {description}"))
    if surface == "game_arena" and query_tokens.intersection({"agent", "agi", "arc", "game"}):
        relevance = max(relevance, 0.5)
    return {
        "surface": surface,
        "ref": surface,
        "title": title,
        "url": url,
        "query": "page metadata",
        "relevance_score": round(relevance, 6),
        "rank_score": round(relevance, 6),
        "summary": description,
        "metadata": {"description": description},
        "source": "kaggle_page_metadata",
    }


def _select_records(records: list[dict[str, object]], *, max_items: int) -> list[dict[str, object]]:
    by_surface: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_surface.setdefault(str(record.get("surface") or "unknown"), []).append(record)
    selected: list[dict[str, object]] = []
    for surface in _SURFACE_URLS:
        candidates = by_surface.get(surface, [])
        candidates.sort(
            key=lambda item: (
                _number(item.get("rank_score")),
                _number(item.get("relevance_score")),
                str(item.get("title") or ""),
            ),
            reverse=True,
        )
        page_records = [item for item in candidates if item.get("source") == "kaggle_page_metadata"]
        if surface in {"game_arena", "benchmarks"} and page_records:
            page_record = page_records[0]
            selected.append(page_record)
            other_records = [item for item in candidates if item is not page_record]
            selected.extend(other_records[: max_items - 1])
            continue
        selected.extend(candidates[:max_items])
    return selected


def _relevance_score(*, row: dict[str, str], query_tokens: set[str], surface: str) -> float:
    haystack = " ".join(str(row.get(key) or "") for key in ("ref", "title", "subtitle", "author", "authorName"))
    score = _token_overlap_score(query_tokens, _tokens(haystack))
    if surface == "benchmarks":
        score *= 0.8
    return round(min(1.0, score), 6)


def _token_overlap_score(query_tokens: set[str], candidate_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = query_tokens.intersection(candidate_tokens)
    return len(overlap) / max(1, min(4, len(query_tokens)))


def _engagement_score(row: dict[str, str]) -> float:
    values = [_number(row.get(key)) for key in ("voteCount", "totalVotes", "votes", "downloadCount", "commentCount")]
    strongest = max(values, default=0.0)
    return round(min(0.3, math.log1p(strongest) / 30.0), 6)


def _freshness_score(row: dict[str, str]) -> float:
    text = " ".join(str(row.get(key) or "") for key in ("lastUpdated", "lastRunTime", "postDate"))
    match = re.search(r"\b(20\d{2})\b", text)
    if not match:
        return 0.0
    age = max(0, datetime.now(UTC).year - int(match.group(1)))
    return max(0.0, 0.15 - age * 0.03)


def _row_summary(row: dict[str, str]) -> str:
    parts: list[str] = []
    subtitle = str(row.get("subtitle") or "").strip()
    if subtitle:
        parts.append(subtitle)
    for key in ("author", "authorName", "lastUpdated", "lastRunTime", "postDate", "voteCount", "totalVotes", "votes"):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)[:800]


def _record_url(*, surface: str, ref: str) -> str:
    if surface == "datasets" and ref:
        return f"https://www.kaggle.com/datasets/{ref}"
    if surface == "models" and ref:
        return f"https://www.kaggle.com/models/{ref}"
    if surface == "code" and ref:
        return f"https://www.kaggle.com/code/{ref}"
    if surface in {"discussions", "benchmarks"} and ref.isdigit():
        return f"https://www.kaggle.com/discussions/general/{ref}"
    return _SURFACE_URLS.get(surface, "https://www.kaggle.com")


def _tokens(value: object) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) >= 2]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _number(value: object) -> float:
    try:
        return max(0.0, float(str(value or "0").replace(",", "")))
    except ValueError:
        return 0.0


def _surface_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts = {surface: 0 for surface in _SURFACE_URLS}
    for record in records:
        surface = str(record.get("surface") or "")
        counts[surface] = counts.get(surface, 0) + 1
    return counts


def _fetch_url(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "kagglebot-public-discovery/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        return response.read(2 * 1024 * 1024).decode("utf-8", errors="ignore")


def _cache_is_fresh(payload: dict[str, object] | None) -> bool:
    if not payload:
        return False
    fetched_at = parse_iso_datetime_utc(payload.get("fetched_at"))
    if fetched_at is None:
        return False
    return fetched_at + timedelta(hours=_refresh_hours()) > datetime.now(UTC)


def _refresh_hours() -> int:
    return env_int(
        "KAGGLEBOT_KAGGLE_DISCOVERY_REFRESH_HOURS",
        default=_DEFAULT_REFRESH_HOURS,
        min_value=1,
    )


def _safe_error(value: object) -> str:
    text = " ".join(str(value).split())
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "<redacted>")
    return text[:1000]
