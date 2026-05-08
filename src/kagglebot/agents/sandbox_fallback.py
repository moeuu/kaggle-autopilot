from __future__ import annotations

import os
from collections.abc import Sequence

_SANDBOX_FAILURE_LINE_PATTERNS = (
    "bwrap:",
    "failed rtm_newaddr",
    "sandbox setup failed",
    "sandbox startup failed",
    "failed to start sandbox",
)
_DEFAULT_AGENT_SANDBOX_MODE = "permissive"
_SUPPORTED_AGENT_SANDBOX_MODES = {"permissive", "fallback", "workspace-write"}


def resolve_agent_sandbox_mode() -> str:
    raw = os.environ.get("KAGGLEBOT_AGENT_SANDBOX_MODE", _DEFAULT_AGENT_SANDBOX_MODE)
    normalized = str(raw or "").strip().lower()
    if normalized in _SUPPORTED_AGENT_SANDBOX_MODES:
        return normalized
    return _DEFAULT_AGENT_SANDBOX_MODE


def append_sandbox_args(
    args: list[str],
    supported_flags: set[str],
    *,
    sandbox_mode: str = "workspace-write",
    dangerously_bypass: bool = False,
    include_full_auto: bool = True,
) -> None:
    if dangerously_bypass and "--dangerously-bypass-approvals-and-sandbox" in supported_flags:
        args.append("--dangerously-bypass-approvals-and-sandbox")
        return
    if include_full_auto and "--full-auto" in supported_flags:
        args.append("--full-auto")
    if "--sandbox" in supported_flags or "-s" in supported_flags:
        args += ["--sandbox", sandbox_mode]


def detect_sandbox_startup_failure(*texts: str) -> str | None:
    combined = "\n".join(text for text in texts if text).strip()
    if not combined:
        return None
    lowered = combined.lower()
    if any(pattern in lowered for pattern in _SANDBOX_FAILURE_LINE_PATTERNS):
        return _first_relevant_line(combined.splitlines())
    if "sandbox" in lowered and "operation not permitted" in lowered:
        return _first_relevant_line(combined.splitlines())
    return None


def _first_relevant_line(lines: Sequence[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= 240:
            return stripped
        return stripped[:237] + "..."
    return "sandbox startup failure"
