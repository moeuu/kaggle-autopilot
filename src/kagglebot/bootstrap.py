from __future__ import annotations

import io
import json
import shutil
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from kagglebot.competition import rules_url_for_slug
from kagglebot.kaggle_api import download_competition
from kagglebot.knowledge import (
    build_improve_template,
    build_kernel_fix_template,
    build_plan_and_initial_prompt,
    ensure_taxonomy,
    record_competition_profile,
    resolve_similar_improvements,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.submission_format import extract_submission_section, load_submission_format_hint
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
    _write_plan(paths, force=force)

    if download:
        if dry_run:
            paths.context_dir.joinpath("download_skipped.txt").write_text(
                "DRY RUN: download skipped.\n", encoding="utf-8"
            )
        else:
            download_competition(slug, paths.data_dir, force=True, quiet=quiet)
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
                submission_section = extract_submission_section(normalized_data)
            if not submission_section and normalized_overview:
                submission_section = extract_submission_section(normalized_overview)
            if not submission_section:
                for page in pages:
                    content = page.get("content")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    section = extract_submission_section(_normalize_page_markdown(content))
                    if section:
                        submission_section = section
                        break
            if submission_section:
                _write_submission_format_markdown(paths, submission_section)
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
    if paths.sample_submission_path.exists():
        if not paths.sample_submission_head_path.exists():
            _write_sample_head(paths.sample_submission_path, paths.sample_submission_head_path)
        return
    tabular = _find_tabular_files(paths.data_dir)
    if not tabular:
        return
    for path in tabular:
        if "sample_submission" in path.name.lower():
            if path.suffix.lower() == ".csv":
                shutil.copy2(path, paths.sample_submission_path)
            else:
                try:
                    frame = _read_table(path, format_hint=format_hint)
                except Exception:
                    return
                frame.to_csv(paths.sample_submission_path, index=False)
            _write_sample_head(paths.sample_submission_path, paths.sample_submission_head_path)
            return


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
        sample = pd.read_csv(sample_path)
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
