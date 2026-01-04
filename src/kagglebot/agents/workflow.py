"""
Three-stage agent workflow orchestration with safety guardrails.

Implements the Codex → Claude → Codex pipeline:
1. Codex: Extract brief from context files
2. Claude: Generate strategy + implementation instructions
3. Codex: Implement kernel following Claude's instructions

All stages enforce write allowlists and detect unauthorized file modifications.
"""

from __future__ import annotations

from pathlib import Path

from kagglebot.agents.allowlist import WriteAllowlist
from kagglebot.agents.base import (
    parse_claude_strategy_output,
    render_prompt_template,
    verify_outputs_exist,
)
from kagglebot.agents.claude_runner import run_claude
from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.exceptions import (
    AgentOutputError,
    AllowlistViolationError,
)
from kagglebot.agents.snapshot import FileSnapshot


def run_agent_pipeline(
    slug: str,
    artifacts_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Path]:
    """
    Run the three-stage agent pipeline for a competition.

    Stage 1 (Codex brief):
        - Reads context files
        - Writes brief.md and brief.json to context/agent/
        - Allowlist: context/agent/brief.*

    Stage 2 (Claude strategy):
        - Reads brief + context files
        - Writes strategy.md, codex_instructions.md, references.md to context/agent/
        - Allowlist: context/agent/*.md

    Stage 3 (Codex implementation):
        - Reads codex_instructions.md
        - Writes kernel.py and supporting files to kernel/
        - Allowlist: kernel/**

    Args:
        slug: Competition slug (e.g., "titanic")
        artifacts_dir: Path to artifacts/ directory
        dry_run: If True, skip actual agent execution

    Returns:
        Dict mapping stage names to output paths:
        {
            "brief": Path("artifacts/slug/context/agent/brief.json"),
            "strategy": Path("artifacts/slug/context/agent/strategy.md"),
            "kernel": Path("artifacts/slug/kernel/kernel.py"),
        }

    Raises:
        AllowlistViolationError: If agent modifies forbidden files
        AgentOutputError: If agent fails to create required outputs
        AgentExecutionError: If agent process fails
    """
    competition_dir = artifacts_dir / slug
    context_dir = competition_dir / "context"
    agent_dir = context_dir / "agent"
    kernel_dir = competition_dir / "kernel"
    runs_dir = competition_dir / "runs"

    # Ensure directories exist
    agent_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Stage 1: Codex Brief Extraction
    # ========================================================================
    print("=" * 60)
    print("STAGE 1: Codex Brief Extraction")
    print("=" * 60)

    brief_output = _run_codex_brief(
        slug=slug,
        competition_dir=competition_dir,
        dry_run=dry_run,
    )

    # ========================================================================
    # Stage 2: Claude Strategy + Implementation Plan
    # ========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Claude Strategy Generation")
    print("=" * 60)

    strategy_output = _run_claude_strategy(
        slug=slug,
        competition_dir=competition_dir,
        dry_run=dry_run,
    )

    # ========================================================================
    # Stage 3: Codex Kernel Implementation
    # ========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Codex Kernel Implementation")
    print("=" * 60)

    kernel_output = _run_codex_implementation(
        slug=slug,
        competition_dir=competition_dir,
        dry_run=dry_run,
    )

    print("\n" + "=" * 60)
    print("AGENT PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "brief": brief_output,
        "strategy": strategy_output,
        "kernel": kernel_output,
    }


def _run_codex_brief(
    slug: str,
    competition_dir: Path,
    dry_run: bool,
) -> Path:
    """Stage 1: Codex extracts brief from context files."""
    context_dir = competition_dir / "context"
    agent_dir = context_dir / "agent"
    runs_dir = competition_dir / "runs"

    # Build prompt
    prompt_template = Path(__file__).parent.parent / "prompts/templates/codex_brief.md"
    variables = {
        "overview_md": str(context_dir / "overview.md"),
        "data_md": str(context_dir / "data.md"),
        "rules_md": str(context_dir / "rules.md"),
        "dataset_profile": str(context_dir / "dataset_profile.json"),
        "sample_submission_head": str(context_dir / "sample_submission_head.txt"),
        "top1_public": str(context_dir / "top1_public.json"),
        "rules_url": f"https://www.kaggle.com/competitions/{slug}/rules",
        "brief_md": str(agent_dir / "brief.md"),
        "brief_json": str(agent_dir / "brief.json"),
    }
    prompt_text = render_prompt_template(prompt_template, variables)
    prompt_path = runs_dir / "01_codex_brief_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # Create allowlist (ONLY agent/ writable)
    allowlist = WriteAllowlist(base_dir=competition_dir)
    allowlist.allow("context/agent/brief.md")
    allowlist.allow("context/agent/brief.json")

    # Pre-snapshot
    snapshot_pre = FileSnapshot.create(competition_dir)

    # Execute Codex
    print("Running Codex brief extraction...")
    output_dir = runs_dir / "01_codex_brief"
    run_codex(prompt_path, output_dir, dry_run=dry_run)

    # Post-snapshot and enforce allowlist
    snapshot_post = FileSnapshot.create(competition_dir)
    violations = snapshot_pre.diff(snapshot_post, allowlist)
    if violations:
        raise AllowlistViolationError(violations)

    # Verify outputs exist
    if not dry_run:
        verify_outputs_exist(agent_dir, ["brief.md", "brief.json"])

    print(f"✓ Brief created: {agent_dir / 'brief.json'}")
    return agent_dir / "brief.json"


def _run_claude_strategy(
    slug: str,
    competition_dir: Path,
    dry_run: bool,
) -> Path:
    """Stage 2: Claude generates strategy + implementation instructions."""
    context_dir = competition_dir / "context"
    agent_dir = context_dir / "agent"
    runs_dir = competition_dir / "runs"

    # Read brief content
    brief_md = (agent_dir / "brief.md").read_text(encoding="utf-8")
    brief_json = (agent_dir / "brief.json").read_text(encoding="utf-8")

    # Build prompt
    prompt_template = Path(__file__).parent.parent / "prompts/templates/claude_strategy.md"
    variables = {
        "brief_content": brief_md,
        "brief_json_content": brief_json,
        "brief_md": str(agent_dir / "brief.md"),
        "brief_json": str(agent_dir / "brief.json"),
        "overview_md": str(context_dir / "overview.md"),
        "data_md": str(context_dir / "data.md"),
        "rules_md": str(context_dir / "rules.md"),
        "dataset_profile": str(context_dir / "dataset_profile.json"),
        "sample_submission_head": str(context_dir / "sample_submission_head.txt"),
        "top1_public": str(context_dir / "top1_public.json"),
        "compute": "Kaggle Kernels",
        "accelerator": "GPU T4 (16GB)",
        "internet": "enabled",
    }
    prompt_text = render_prompt_template(prompt_template, variables)
    prompt_path = runs_dir / "02_claude_strategy_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # Create allowlist (ONLY agent/*.md writable)
    allowlist = WriteAllowlist(base_dir=competition_dir)
    allowlist.allow("context/agent/strategy.md")
    allowlist.allow("context/agent/codex_instructions.md")
    allowlist.allow("context/agent/references.md")

    # Pre-snapshot
    snapshot_pre = FileSnapshot.create(competition_dir)

    # Execute Claude
    print("Running Claude strategy generation...")
    output_dir = runs_dir / "02_claude_strategy"
    result = run_claude(prompt_path, output_dir, dry_run=dry_run)

    if not dry_run:
        # Parse delimited output
        parsed = parse_claude_strategy_output(result.stdout)

        # Write parsed sections to files
        (agent_dir / "strategy.md").write_text(parsed.strategy, encoding="utf-8")
        (agent_dir / "codex_instructions.md").write_text(parsed.codex_instructions, encoding="utf-8")
        (agent_dir / "references.md").write_text(parsed.references, encoding="utf-8")

    # Post-snapshot and enforce allowlist
    snapshot_post = FileSnapshot.create(competition_dir)
    violations = snapshot_pre.diff(snapshot_post, allowlist)
    if violations:
        raise AllowlistViolationError(violations)

    # Verify outputs exist
    if not dry_run:
        verify_outputs_exist(agent_dir, ["strategy.md", "codex_instructions.md", "references.md"])

    print(f"✓ Strategy created: {agent_dir / 'strategy.md'}")
    return agent_dir / "strategy.md"


def _run_codex_implementation(
    slug: str,
    competition_dir: Path,
    dry_run: bool,
) -> Path:
    """Stage 3: Codex implements kernel following Claude's instructions."""
    agent_dir = competition_dir / "context" / "agent"
    kernel_dir = competition_dir / "kernel"
    runs_dir = competition_dir / "runs"

    # Read Claude's instructions
    codex_instructions = (agent_dir / "codex_instructions.md").read_text(encoding="utf-8")

    # Build prompt
    prompt_template = Path(__file__).parent.parent / "prompts/templates/codex_implement_from_claude.md"
    variables = {
        "claude_instructions": codex_instructions,
    }
    prompt_text = render_prompt_template(prompt_template, variables)
    prompt_path = runs_dir / "03_codex_implementation_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # Create allowlist (ONLY kernel/** writable)
    allowlist = WriteAllowlist(base_dir=competition_dir)
    allowlist.allow("kernel/**")

    # Pre-snapshot
    snapshot_pre = FileSnapshot.create(competition_dir)

    # Execute Codex
    print("Running Codex kernel implementation...")
    output_dir = runs_dir / "03_codex_implementation"
    run_codex(prompt_path, output_dir, dry_run=dry_run)

    # Post-snapshot and enforce allowlist
    snapshot_post = FileSnapshot.create(competition_dir)
    violations = snapshot_pre.diff(snapshot_post, allowlist)
    if violations:
        raise AllowlistViolationError(violations)

    # Verify kernel.py exists
    kernel_path = kernel_dir / "kernel.py"
    if not dry_run and not kernel_path.exists():
        raise AgentOutputError(f"Missing required output: {kernel_path}")

    print(f"✓ Kernel created: {kernel_path}")
    return kernel_path
