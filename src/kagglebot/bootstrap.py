from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.competition import rules_url_for_slug
from kagglebot.kaggle_api import download_competition
from kagglebot.knowledge import (
    build_improve_template,
    build_plan_and_baseline_prompt,
    ensure_taxonomy,
    record_competition_profile,
    resolve_similar_improvements,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.types import BootstrapMeta, PlanConfig, RulesInfo
from kagglebot.validators import safe_extract_zip


def bootstrap_competition(
    *,
    slug: str,
    competition_url: str | None,
    paths: CompetitionPaths,
    knowledge_paths: KnowledgePaths,
    rules_source: str = "url",
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
    _write_plan(paths, force=force)

    if download:
        if dry_run:
            paths.context_dir.joinpath("download_skipped.txt").write_text(
                "DRY RUN: download skipped.\n", encoding="utf-8"
            )
        else:
            download_competition(slug, paths.data_dir, force=force, quiet=quiet)
            _unzip_downloads(paths.data_dir)

    profile = _write_dataset_profile(paths)
    _cache_sample_submission(paths)

    taxonomy = ensure_taxonomy(knowledge_paths)
    similar = resolve_similar_improvements(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        tags=profile.get("tags", []),
    )
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
    if rules_source == "url":
        return
    if rules_source == "file":
        if not rules_file:
            raise ValueError("--rules-file is required when --rules-source file.")
        dest = paths.context_dir / f"rules{rules_file.suffix}"
        shutil.copy2(rules_file, dest)
        return
    if rules_source == "fetch":
        if dry_run:
            (paths.context_dir / "fetch_skipped.txt").write_text("DRY RUN: rules fetch skipped.\n", encoding="utf-8")
            return
        try:
            with urllib.request.urlopen(rules_url, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            (paths.context_dir / "rules.html").write_text(html, encoding="utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            (paths.context_dir / "fetch_error.txt").write_text(str(exc), encoding="utf-8")
        return
    raise ValueError(f"Unknown rules source: {rules_source}")


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
    baseline = build_plan_and_baseline_prompt(
        slug=slug,
        rules_url=rules_url,
        profile=profile,
        taxonomy=taxonomy,
        similar_improvements=similar_improvements,
    )
    improve_template = build_improve_template()
    paths.codex_plan_and_baseline_prompt.write_text(baseline, encoding="utf-8")
    paths.codex_improve_template.write_text(improve_template, encoding="utf-8")


def _write_dataset_profile(paths: CompetitionPaths) -> dict[str, object]:
    from kagglebot.knowledge import build_dataset_profile

    profile = build_dataset_profile(paths.data_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def _cache_sample_submission(paths: CompetitionPaths) -> None:
    if paths.sample_submission_path.exists():
        return
    csvs = sorted([p for p in paths.data_dir.rglob("*.csv") if p.is_file()])
    if not csvs:
        return
    for path in csvs:
        if "sample_submission" in path.name.lower():
            shutil.copy2(path, paths.sample_submission_path)
            return


def _unzip_downloads(data_dir: Path) -> None:
    for zip_path in data_dir.glob("*.zip"):
        safe_extract_zip(zip_path, data_dir)
