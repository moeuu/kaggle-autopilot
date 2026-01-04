"""
Base agent workflow utilities and interfaces.

Provides common functions for agent execution, including:
- Delimiter-based output parsing
- Prompt template rendering
- Workflow orchestration helpers
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.agents.exceptions import DelimiterParseError


@dataclass(frozen=True)
class ClaudeStrategyOutput:
    """Parsed output from Claude strategy agent."""

    strategy: str
    codex_instructions: str
    references: str
    raw_output: str


def parse_claude_strategy_output(output: str) -> ClaudeStrategyOutput:
    """
    Parse Claude's delimited strategy output.

    Expected format:
        ===CLAUDE_STRATEGY===
        <strategy content>
        ===CODEX_IMPLEMENTATION_INSTRUCTIONS===
        <implementation instructions>
        ===REFERENCES===
        <references>

    Args:
        output: Raw stdout from Claude agent.

    Returns:
        ClaudeStrategyOutput with parsed sections.

    Raises:
        DelimiterParseError: If required sections are missing or malformed.
    """
    markers = {
        "strategy": "===CLAUDE_STRATEGY===",
        "instructions": "===CODEX_IMPLEMENTATION_INSTRUCTIONS===",
        "references": "===REFERENCES===",
    }
    current: str | None = None
    sections: dict[str, list[str]] = {"strategy": [], "instructions": [], "references": []}
    seen = {"strategy": False, "instructions": False, "references": False}
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == markers["strategy"]:
            current = "strategy"
            seen["strategy"] = True
            continue
        if stripped == markers["instructions"]:
            current = "instructions"
            seen["instructions"] = True
            continue
        if stripped == markers["references"]:
            current = "references"
            seen["references"] = True
            continue
        if current:
            sections[current].append(line)

    if not seen["strategy"]:
        raise DelimiterParseError("Missing ===CLAUDE_STRATEGY=== section")
    if not seen["instructions"]:
        raise DelimiterParseError("Missing ===CODEX_IMPLEMENTATION_INSTRUCTIONS=== section")

    strategy = "\n".join(sections["strategy"]).strip()
    instructions = "\n".join(sections["instructions"]).strip()
    references = "\n".join(sections["references"]).strip()
    return ClaudeStrategyOutput(
        strategy=strategy,
        codex_instructions=instructions,
        references=references,
        raw_output=output,
    )


def render_prompt_template(template_path: Path, variables: dict[str, str]) -> str:
    """
    Render a prompt template with variable substitution.

    Replaces {{variable_name}} placeholders with values from variables dict.

    Args:
        template_path: Path to template file.
        variables: Dict mapping variable names to their values.

    Returns:
        Rendered prompt string.

    Raises:
        FileNotFoundError: If template_path doesn't exist.
        KeyError: If template references undefined variable.
    """
    template_text = template_path.read_text(encoding="utf-8")

    # Find all {{variable}} placeholders
    placeholders = re.findall(r"\{\{(\w+)\}\}", template_text)

    # Verify all variables are provided
    missing = [p for p in placeholders if p not in variables]
    if missing:
        raise KeyError(f"Template requires undefined variables: {missing}")

    # Perform substitution
    rendered = template_text
    for name, value in variables.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)

    return rendered


def verify_outputs_exist(output_dir: Path, required_files: list[str]) -> None:
    """
    Verify that required output files were created by agent.

    Args:
        output_dir: Directory where outputs should be located.
        required_files: List of relative file paths that must exist.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    missing = []
    for file_path in required_files:
        full_path = output_dir / file_path
        if not full_path.exists():
            missing.append(file_path)

    if missing:
        raise FileNotFoundError(f"Agent failed to create required outputs: {', '.join(missing)}")
