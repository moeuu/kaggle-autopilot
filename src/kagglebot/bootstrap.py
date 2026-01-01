from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.competition import rules_url_for_slug
from kagglebot.kaggle_api import download_competition
from kagglebot.paths import CompetitionPaths, repo_root
from kagglebot.types import BootstrapMeta, PlanConfig, RulesInfo
from kagglebot.validators import safe_extract_zip


def bootstrap_competition(
    *,
    slug: str,
    force: bool = False,
    root: Path | None = None,
    rules_source: str = "url",
    rules_file: Path | None = None,
    download: bool = False,
    quiet: bool = True,
) -> Path:
    """
    Prepare local workspace directories and write meta + plan files.
    Does not join competitions or accept rules automatically.
    """
    base_root = root if root is not None else repo_root()
    paths = CompetitionPaths(slug=slug, repo_root=base_root)

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

    _capture_rules(paths, rules_source=rules_source, rules_file=rules_file, rules_url=rules_url)
    _write_plan(paths, force=force)
    _write_prompts(paths, slug=slug)
    _write_dataset_summary(paths)

    if download:
        download_competition(slug, paths.data_dir, force=force, quiet=quiet)
        _unzip_downloads(paths.data_dir)

    return paths.meta_path


def _write_json(path: Path, payload: dict[str, object], *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_rules(paths: CompetitionPaths, *, rules_source: str, rules_file: Path | None, rules_url: str) -> None:
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


def _write_prompts(paths: CompetitionPaths, *, slug: str) -> None:
    codex_prompt = paths.prompts_dir / "codex.md"
    claude_prompt = paths.prompts_dir / "claude.md"

    codex_prompt.write_text(_render_prompt(slug, agent_name="Codex"), encoding="utf-8")
    claude_prompt.write_text(_render_prompt(slug, agent_name="Claude"), encoding="utf-8")


def _render_prompt(slug: str, *, agent_name: str) -> str:
    return (
        f"# Kagglebot {agent_name} Prompt\n\n"
        f"Competition slug: {slug}\n\n"
        "You are implementing a competition baseline in this repo.\n\n"
        "Instructions:\n"
        "1) Read artifacts/<slug>/context/meta.json and plan.json.\n"
        "2) Read artifacts/<slug>/context/dataset_summary.txt.\n"
        "3) Refer to artifacts/<slug>/context/rules_url.txt (or rules.* if provided).\n"
        "4) Decide missing plan values (time_budget_min, seed, kernel_name, internet, accelerator, kaggle_username).\n"
        "5) Implement a robust baseline under kagglebot/solver/.\n"
        "6) Ensure tests pass (uv run pytest -q).\n"
    )


def _write_dataset_summary(paths: CompetitionPaths) -> None:
    data_dir = paths.data_dir
    summary_path = paths.dataset_summary_path
    if not data_dir.exists():
        summary_path.write_text("Data directory not found.\n", encoding="utf-8")
        return

    lines = ["Dataset summary:"]
    csvs = sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])
    if not csvs:
        lines.append("No CSV files found.")
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for path in csvs:
        try:
            import pandas as pd

            df = pd.read_csv(path, nrows=5)
            lines.append(f"- {path.name}: columns={list(df.columns)}")
        except Exception:
            lines.append(f"- {path.name}: unable to read columns")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unzip_downloads(data_dir: Path) -> None:
    for zip_path in data_dir.glob("*.zip"):
        safe_extract_zip(zip_path, data_dir)
