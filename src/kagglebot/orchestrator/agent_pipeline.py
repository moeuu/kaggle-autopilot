from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kagglebot.agents.claude_runner import run_claude
from kagglebot.agents.codex_runner import run_codex
from kagglebot.exceptions import KaggleBotError
from kagglebot.paths import CompetitionPaths
from kagglebot.validators import scan_text_for_secrets


@dataclass(frozen=True)
class AgentPipelineConfig:
    slug: str
    competition_url: str | None
    compute: str
    accelerator: str
    internet: str
    run_id: str
    dry_run: bool
    repo_root: Path


def run_agent_pipeline(*, paths: CompetitionPaths, config: AgentPipelineConfig) -> None:
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
    _ensure_context_materials(paths)

    brief_dir = paths.context_agent_dir / "brief"
    claude_dir = paths.context_agent_dir / "claude"
    implement_dir = paths.context_agent_dir / "implement"
    brief_dir.mkdir(parents=True, exist_ok=True)
    claude_dir.mkdir(parents=True, exist_ok=True)
    implement_dir.mkdir(parents=True, exist_ok=True)

    brief_path = _run_codex_brief(paths, config, brief_dir)
    instructions_path = _run_claude_strategy(paths, config, claude_dir, brief_path)
    _run_codex_kernel_implementation(paths, config, implement_dir, instructions_path)


def _run_codex_brief(paths: CompetitionPaths, config: AgentPipelineConfig, output_dir: Path) -> Path:
    template = _load_template("codex_brief.md")
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "competition_url": config.competition_url or "unknown",
            "overview_content": _read_text(paths.overview_md_path),
            "data_content": _read_text(paths.data_md_path),
            "rules_content": _read_text(paths.rules_md_path),
            "rules_url": _read_text(paths.rules_url_path),
            "dataset_profile": _read_text(paths.dataset_profile_path),
            "sample_submission": _read_text(paths.sample_submission_path),
        },
    )
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    before = _snapshot_tree(config.repo_root)
    result = run_codex(prompt_path, output_dir, dry_run=config.dry_run)
    after = _snapshot_tree(config.repo_root)
    _enforce_allowlist_changes(
        root=config.repo_root,
        before=before,
        after=after,
        allowed_prefixes=[paths.context_agent_dir],
        stage="codex_brief",
    )
    if result.returncode != 0:
        raise KaggleBotError(f"Codex brief failed: {result.stderr}")

    brief_path = paths.context_agent_dir / "brief_for_claude.md"
    brief_text = _read_text(result.last_message_path).strip()
    if not brief_text:
        brief_text = "Brief not generated."
    brief_path.write_text(brief_text + "\n", encoding="utf-8")
    return brief_path


def _run_claude_strategy(
    paths: CompetitionPaths,
    config: AgentPipelineConfig,
    output_dir: Path,
    brief_path: Path,
) -> Path:
    template = _load_template("claude_strategy.md")
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "compute": config.compute,
            "accelerator": config.accelerator,
            "internet": config.internet,
            "brief_content": _read_text(brief_path),
            "overview_content": _read_text(paths.overview_md_path),
            "data_content": _read_text(paths.data_md_path),
            "rules_content": _read_text(paths.rules_md_path),
            "dataset_profile": _read_text(paths.dataset_profile_path),
            "sample_submission": _read_text(paths.sample_submission_path),
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

    strategy_text, instructions_text = _split_claude_output(result.stdout)
    strategy_path = paths.context_agent_dir / "claude_strategy.md"
    instructions_path = paths.context_agent_dir / "codex_instructions.md"
    strategy_path.write_text(strategy_text + "\n", encoding="utf-8")
    instructions_path.write_text(instructions_text + "\n", encoding="utf-8")
    return instructions_path


def _run_codex_kernel_implementation(
    paths: CompetitionPaths,
    config: AgentPipelineConfig,
    output_dir: Path,
    instructions_path: Path,
) -> None:
    kernel_dir = paths.kernel_source_dir
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        kernel_path.write_text("# kagglebot kernel\n", encoding="utf-8")

    template = _load_template("codex_kernel_impl.md")
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "kernel_dir": str(kernel_dir),
            "kernel_path": str(kernel_path),
            "instructions": _read_text(instructions_path),
        },
    )
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    before = _snapshot_tree(config.repo_root)
    result = run_codex(prompt_path, output_dir, dry_run=config.dry_run)
    after = _snapshot_tree(config.repo_root)
    _enforce_allowlist_changes(
        root=config.repo_root,
        before=before,
        after=after,
        allowed_prefixes=[paths.kernel_source_dir, paths.runs_dir, paths.context_dir],
        stage="codex_kernel_implementation",
    )
    if result.returncode != 0:
        raise KaggleBotError(f"Codex kernel implementation failed: {result.stderr}")


def _ensure_context_materials(paths: CompetitionPaths) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    rules_url = _read_text(paths.rules_url_path)
    if not rules_url:
        rules_url = f"https://www.kaggle.com/competitions/{paths.slug}/rules"
        paths.rules_url_path.write_text(rules_url + "\n", encoding="utf-8")

    if not paths.dataset_profile_path.exists():
        from kagglebot.knowledge import build_dataset_profile

        profile = build_dataset_profile(paths.data_dir)
        paths.dataset_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    if not paths.overview_md_path.exists():
        paths.overview_md_path.write_text(f"Overview not available. See {rules_url}\n", encoding="utf-8")
    if not paths.data_md_path.exists():
        paths.data_md_path.write_text(f"Data description not available. See {rules_url}\n", encoding="utf-8")
    if not paths.rules_md_path.exists():
        paths.rules_md_path.write_text(f"Rules not available. See {rules_url}\n", encoding="utf-8")

    if not paths.sample_submission_path.exists():
        from kagglebot.solver.io import find_competition_files

        try:
            _, _, sample_path = find_competition_files(paths.data_dir)
            paths.sample_submission_path.write_text(sample_path.read_text(encoding="utf-8"), encoding="utf-8")
        except FileNotFoundError:
            paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")


def _load_template(name: str) -> str:
    base_dir = Path(__file__).resolve().parents[1] / "prompts"
    return (base_dir / name).read_text(encoding="utf-8")


def _render_template(template: str, mapping: dict[str, str]) -> str:
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _split_claude_output(text: str) -> tuple[str, str]:
    strategy_marker = "===STRATEGY==="
    instructions_marker = "===CODEX_INSTRUCTIONS==="
    strategy_lines: list[str] = []
    instructions_lines: list[str] = []
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == strategy_marker:
            current = "strategy"
            continue
        if stripped == instructions_marker:
            current = "instructions"
            continue
        if current == "strategy":
            strategy_lines.append(line)
        elif current == "instructions":
            instructions_lines.append(line)
    strategy_text = "\n".join(strategy_lines).strip()
    instructions_text = "\n".join(instructions_lines).strip()
    if not strategy_text:
        strategy_text = text.strip()
    if not instructions_text:
        instructions_text = text.strip()
    return strategy_text, instructions_text


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        stat = path.stat()
        snapshot[rel] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _enforce_allowlist_changes(
    *,
    root: Path,
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
    allowed_prefixes: list[Path],
    stage: str,
) -> None:
    allowed = []
    for prefix in allowed_prefixes:
        try:
            rel = prefix.relative_to(root).as_posix()
        except ValueError:
            continue
        allowed.append(rel.rstrip("/") + "/")
    changed = _diff_snapshots(before, after)
    unauthorized = [path for path in changed if not _is_allowed(path, allowed)]
    if unauthorized:
        raise KaggleBotError(f"Agent write-guard failed in {stage}: {unauthorized}")


def _diff_snapshots(
    before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]
) -> list[str]:
    changed: list[str] = []
    for path, meta in after.items():
        if before.get(path) != meta:
            changed.append(path)
    for path in before:
        if path not in after:
            changed.append(path)
    return sorted(set(changed))


def _is_allowed(path: str, allowed_prefixes: list[str]) -> bool:
    for prefix in allowed_prefixes:
        if path.startswith(prefix):
            return True
    return False


def _assert_no_secrets(text: str) -> None:
    matches = scan_text_for_secrets(text)
    if matches:
        raise KaggleBotError(f"Secret pattern detected in prompt text: {matches}")
