"""
Agent framework for autonomous Kaggle kernel implementation.

This package provides:
- WriteAllowlist: File write restriction enforcement
- FileSnapshot: Change detection for file modifications
- Agent runners: Codex and Claude execution with safety guardrails
- Exceptions: Agent-specific errors with dedicated exit codes
"""

from __future__ import annotations

from kagglebot.agents.allowlist import WriteAllowlist
from kagglebot.agents.base import (
    ClaudeStrategyOutput,
    parse_claude_strategy_output,
    render_prompt_template,
    verify_outputs_exist,
)
from kagglebot.agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentOutputError,
    AgentTimeoutError,
    AllowlistViolationError,
    DelimiterParseError,
)
from kagglebot.agents.snapshot import FileSnapshot, FileState
from kagglebot.agents.workflow import run_agent_pipeline

__all__ = [
    # Core infrastructure
    "WriteAllowlist",
    "FileSnapshot",
    "FileState",
    # Base utilities
    "ClaudeStrategyOutput",
    "parse_claude_strategy_output",
    "render_prompt_template",
    "verify_outputs_exist",
    # Workflow
    "run_agent_pipeline",
    # Exceptions
    "AgentError",
    "AllowlistViolationError",
    "AgentOutputError",
    "AgentTimeoutError",
    "DelimiterParseError",
    "AgentExecutionError",
]
