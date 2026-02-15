"""
Agent framework for autonomous Kaggle kernel implementation.

This package provides:
- WriteAllowlist: File write restriction enforcement
- FileSnapshot: Change detection for file modifications
- Exceptions: Agent-specific errors with dedicated exit codes
"""

from __future__ import annotations

from kagglebot.agents.allowlist import WriteAllowlist
from kagglebot.agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentOutputError,
    AgentTimeoutError,
    AllowlistViolationError,
    DelimiterParseError,
)
from kagglebot.agents.snapshot import FileSnapshot, FileState

__all__ = [
    # Core infrastructure
    "WriteAllowlist",
    "FileSnapshot",
    "FileState",
    # Exceptions
    "AgentError",
    "AllowlistViolationError",
    "AgentOutputError",
    "AgentTimeoutError",
    "DelimiterParseError",
    "AgentExecutionError",
]
