from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kagglebot.agents.claude_runner import run_claude
from kagglebot.agents.codex_runner import run_codex
from kagglebot.exceptions import KaggleBotError
from kagglebot.exec_utils import run_command
from kagglebot.paths import CompetitionPaths
from kagglebot.validators import scan_text_for_secrets


@dataclass(frozen=True)
class StrategyConfig:
    slug: str
    competition_url: str | None
    compute: str
    accelerator: str
    internet: str
    run_id: str
    dry_run: bool
    repo_root: Path


def run_strategy_pipeline(*, paths: CompetitionPaths, config: StrategyConfig) -> None:
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
    _ensure_context_materials(paths)

    brief_dir = paths.context_agent_dir / "brief"
    claude_dir = paths.context_agent_dir / "claude"
    implement_dir = paths.context_agent_dir / "implement"
    brief_dir.mkdir(parents=True, exist_ok=True)
    claude_dir.mkdir(parents=True, exist_ok=True)
    implement_dir.mkdir(parents=True, exist_ok=True)

    _run_codex_brief(paths, config, brief_dir)
    _run_claude_strategy(paths, config, claude_dir)
    _run_codex_implement(paths, config, implement_dir)


def _run_codex_brief(paths: CompetitionPaths, config: StrategyConfig, output_dir: Path) -> None:
    template = _load_template("codex_brief.md")
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "competition_url": config.competition_url or "unknown",
            "overview_md": str(paths.overview_md_path),
            "data_md": str(paths.data_md_path),
            "rules_md": str(paths.rules_md_path),
            "rules_url": str(paths.rules_url_path),
            "dataset_profile": str(paths.dataset_profile_path),
            "sample_submission_head": str(paths.sample_submission_head_path),
            "top1_public": str(paths.top1_public_path),
            "brief_md": str(paths.context_agent_dir / "brief_for_claude.md"),
            "brief_json": str(paths.context_agent_dir / "brief_for_claude.json"),
        },
    )
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    before = _git_status_paths(config.repo_root)
    result = run_codex(prompt_path, output_dir, dry_run=config.dry_run)
    if result.returncode != 0:
        raise KaggleBotError(f"Codex brief failed: {result.stderr}")
    after = _git_status_paths(config.repo_root)
    _assert_no_new_changes(before, after)

    _ensure_brief_outputs(paths.context_agent_dir, result)


def _run_claude_strategy(paths: CompetitionPaths, config: StrategyConfig, output_dir: Path) -> None:
    template = _load_template("claude_strategy.md")
    brief_md_path = paths.context_agent_dir / "brief_for_claude.md"
    brief_json_path = paths.context_agent_dir / "brief_for_claude.json"
    brief_content = brief_md_path.read_text(encoding="utf-8") if brief_md_path.exists() else ""
    brief_json_content = brief_json_path.read_text(encoding="utf-8") if brief_json_path.exists() else "{}"
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "compute": config.compute,
            "accelerator": config.accelerator,
            "internet": config.internet,
            "brief_md": str(brief_md_path),
            "brief_json": str(brief_json_path),
            "brief_content": brief_content,
            "brief_json_content": brief_json_content,
            "overview_md": str(paths.overview_md_path),
            "data_md": str(paths.data_md_path),
            "rules_md": str(paths.rules_md_path),
            "dataset_profile": str(paths.dataset_profile_path),
            "sample_submission_head": str(paths.sample_submission_head_path),
            "top1_public": str(paths.top1_public_path),
        },
    )
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    result = run_claude(prompt_path, output_dir, dry_run=config.dry_run)
    if result.returncode != 0:
        raise KaggleBotError(f"Claude strategy failed: {result.stderr}")

    transcript_path = paths.context_agent_dir / "claude_transcript.txt"
    transcript_path.write_text(result.stdout, encoding="utf-8")
    _write_claude_outputs(paths.context_agent_dir, result.stdout)


def _run_codex_implement(paths: CompetitionPaths, config: StrategyConfig, output_dir: Path) -> None:
    instructions_path = paths.context_agent_dir / "codex_implementation_instructions.md"
    if not instructions_path.exists():
        raise KaggleBotError("Missing codex_implementation_instructions.md from Claude.")
    prompt_text = instructions_path.read_text(encoding="utf-8")
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    result = run_codex(prompt_path, output_dir, dry_run=config.dry_run)
    if result.returncode != 0:
        raise KaggleBotError(f"Codex implementation failed: {result.stderr}")


def _ensure_context_materials(paths: CompetitionPaths) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    rules_url = paths.rules_url_path.read_text(encoding="utf-8").strip() if paths.rules_url_path.exists() else ""
    if not paths.dataset_profile_path.exists():
        from kagglebot.knowledge import build_dataset_profile

        profile = build_dataset_profile(paths.data_dir)
        paths.dataset_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    if not paths.overview_md_path.exists():
        _write_markdown(paths.overview_md_path, f"Overview content not provided. See: {rules_url}\n")
    if not paths.data_md_path.exists():
        _write_markdown(paths.data_md_path, _build_data_markdown(paths))
    if not paths.rules_md_path.exists():
        _write_markdown(paths.rules_md_path, f"Rules content not provided. See: {rules_url}\n")
    if not paths.sample_submission_head_path.exists() and paths.sample_submission_path.exists():
        _write_markdown(paths.sample_submission_head_path, paths.sample_submission_path.read_text(encoding="utf-8"))


def _build_data_markdown(paths: CompetitionPaths) -> str:
    if not paths.data_dir.exists():
        return "Data directory missing.\n"
    entries: list[str] = []
    for path in sorted(paths.data_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(paths.data_dir)
        size_kb = path.stat().st_size / 1024
        entries.append(f"- {rel} ({size_kb:.1f} KB)")
        if len(entries) >= 50:
            entries.append("- ... (truncated)")
            break
    if not entries:
        return "No data files found.\n"
    return "Data files:\n" + "\n".join(entries) + "\n"


def _ensure_brief_outputs(agent_dir: Path, result) -> None:  # noqa: ANN001
    brief_md = agent_dir / "brief_for_claude.md"
    brief_json = agent_dir / "brief_for_claude.json"
    if not brief_md.exists():
        content = ""
        if hasattr(result, "last_message_path") and result.last_message_path.exists():
            content = result.last_message_path.read_text(encoding="utf-8")
        _write_markdown(brief_md, content or "Brief not generated by Codex.\n")
    if not brief_json.exists():
        payload = {"status": "missing", "note": "Codex brief JSON not generated."}
        brief_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_claude_outputs(agent_dir: Path, text: str) -> None:
    sections = _split_sections(text)
    _write_markdown(agent_dir / "claude_strategy.md", sections["strategy"])
    _write_markdown(agent_dir / "codex_implementation_instructions.md", sections["instructions"])
    _write_markdown(agent_dir / "references.md", sections["references"])


def _split_sections(text: str) -> dict[str, str]:
    markers = {
        "strategy": "===CLAUDE_STRATEGY===",
        "instructions": "===CODEX_IMPLEMENTATION_INSTRUCTIONS===",
        "references": "===REFERENCES===",
    }
    current = None
    sections: dict[str, list[str]] = {"strategy": [], "instructions": [], "references": []}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == markers["strategy"]:
            current = "strategy"
            continue
        if stripped == markers["instructions"]:
            current = "instructions"
            continue
        if stripped == markers["references"]:
            current = "references"
            continue
        if current:
            sections[current].append(line)
    if not any(sections.values()):
        return {"strategy": text.strip(), "instructions": text.strip(), "references": ""}
    rendered = {key: "\n".join(value).strip() for key, value in sections.items()}
    if not rendered["instructions"]:
        rendered["instructions"] = rendered["strategy"]
    return rendered


def _load_template(name: str) -> str:
    base_dir = Path(__file__).resolve().parents[1] / "prompts" / "templates"
    return (base_dir / name).read_text(encoding="utf-8")


def _render_template(template: str, mapping: dict[str, str]) -> str:
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _git_status_paths(repo_root: Path) -> list[str]:
    result = run_command(["git", "status", "--porcelain"], cwd=repo_root)
    if result.returncode != 0:
        return []
    paths = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            paths.append(parts[1])
    return paths


def _assert_no_new_changes(before: list[str], after: list[str]) -> None:
    before_set = set(before)
    new = [path for path in after if path not in before_set]
    if new:
        raise KaggleBotError(f"Codex brief modified files unexpectedly: {new}")


def _write_markdown(path: Path, text: str) -> None:
    normalized = text.strip()
    if not normalized.endswith("\n"):
        normalized += "\n"
    path.write_text(normalized, encoding="utf-8")


def _assert_no_secrets(text: str) -> None:
    matches = scan_text_for_secrets(text)
    if matches:
        raise KaggleBotError(f"Secret pattern detected in prompt text: {matches}")
