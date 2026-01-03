"""
Agent-specific exceptions with dedicated exit codes.

Exit code convention:
- 0: Success
- 1-19: General errors
- 20-29: Agent safety violations
- 30-39: Agent execution errors
"""

from __future__ import annotations

from pathlib import Path


class AgentError(Exception):
    """Base exception for all agent errors."""

    exit_code: int = 1


class AllowlistViolationError(AgentError):
    """
    Agent attempted to modify files outside allowed paths.

    Exit code: 20
    """

    exit_code = 20

    def __init__(self, violations: set[Path] | str):
        if isinstance(violations, set):
            paths = "\n  ".join(str(p) for p in sorted(violations))
            msg = f"Agent modified forbidden files:\n  {paths}"
        else:
            msg = str(violations)
        super().__init__(msg)


class AgentOutputError(AgentError):
    """
    Agent failed to produce required output files.

    Exit code: 21
    """

    exit_code = 21


class AgentTimeoutError(AgentError):
    """
    Agent execution exceeded time limit.

    Exit code: 22
    """

    exit_code = 22


class DelimiterParseError(AgentError):
    """
    Failed to parse delimited sections from agent output.

    Exit code: 23
    """

    exit_code = 23


class AgentExecutionError(AgentError):
    """
    Agent process failed with non-zero exit code.

    Exit code: 30
    """

    exit_code = 30

    def __init__(self, returncode: int, stderr: str = ""):
        msg = f"Agent exited with code {returncode}"
        if stderr:
            msg += f"\nStderr:\n{stderr}"
        super().__init__(msg)
        self.returncode = returncode
