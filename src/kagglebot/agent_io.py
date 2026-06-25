from __future__ import annotations

import builtins
from pathlib import Path

from rich import print

_AGENT_CAPACITY_MARKERS = (
    "selected model is at capacity",
    "model is at capacity",
    "please try a different model",
)


class TrainingLiveStdout:
    """Render a single carriage-return live line while preserving regular log lines."""

    def __init__(self, base_stream) -> None:
        self._base_stream = base_stream
        self._last_live_text = ""
        self._live_active = False

    def render_live(self, text: str) -> None:
        self._last_live_text = text
        self._base_stream.write(f"\r{text}")
        self._live_active = True

    def finish_live(self, text: str) -> None:
        self._last_live_text = text
        self._base_stream.write(f"\r{text}\n")
        self._live_active = False

    def write(self, s: str) -> int:
        if not s:
            return 0
        interrupted = False
        if self._live_active and any(ch not in "\r" for ch in s):
            self._base_stream.write("\n")
            self._live_active = False
            interrupted = True
        written = self._base_stream.write(s)
        if interrupted and s.endswith("\n") and self._last_live_text:
            self._base_stream.write(f"\r{self._last_live_text}")
            self._live_active = True
        return written

    def flush(self) -> None:
        self._base_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._base_stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self._base_stream, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._base_stream, "errors", None)

    def fileno(self) -> int:
        return self._base_stream.fileno()


def print_agent_prompt(*, log_alias: str, prompt_path: Path, prompt_text: str) -> None:
    print(f"[cyan]{log_alias} prompt[/cyan]: {prompt_path}")
    builtins.print(prompt_text.rstrip())
    builtins.print("")


def read_agent_response(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").rstrip()


def print_agent_response(*, log_alias: str, response_path: Path, response_text: str) -> None:
    print(f"[cyan]{log_alias} response[/cyan]: {response_path}")
    builtins.print(response_text)
    builtins.print("")


def log_codex_sandbox_fallback(*, stage_label: str, result: object) -> None:
    if not bool(getattr(result, "used_sandbox_fallback", False)):
        return
    excerpt = str(getattr(result, "sandbox_failure_excerpt", "")).strip()
    detail = f": {excerpt}" if excerpt else ""
    print(f"[yellow]{stage_label}[/yellow]: codex sandbox fallback used{detail}")


def tail_for_prompt(text: str, *, max_chars: int = 6000) -> str:
    normalized = (text or "").replace("\r", "\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


def is_agent_capacity_failure(result: object, response: str) -> bool:
    haystack = "\n".join(
        str(part or "")
        for part in (
            getattr(result, "stdout", ""),
            getattr(result, "stderr", ""),
            response,
        )
    ).lower()
    return any(marker in haystack for marker in _AGENT_CAPACITY_MARKERS)


def agent_failure_detail(result: object, response: str) -> str:
    detail = "\n".join(
        part
        for part in (
            f"returncode={getattr(result, 'returncode', 'unknown')}",
            f"stderr={getattr(result, 'stderr', '')}",
            f"response={response}",
            f"transcript_tail={str(getattr(result, 'stdout', ''))[-6000:]}",
        )
        if part
    )
    return tail_for_prompt(detail, max_chars=8000)


def append_fix_retry_feedback(
    *,
    base_prompt: str,
    stage_label: str,
    codex_pass: int,
    failure_text: str,
) -> str:
    clipped = tail_for_prompt(failure_text, max_chars=6000)
    if not clipped:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        f"## Retry Feedback (pass {codex_pass})\n\n"
        f"The previous {stage_label} pass did not fully resolve the issue.\n"
        "Apply additional minimal edits focused on the remaining failure below.\n\n"
        "```\n"
        f"{clipped}\n"
        "```\n"
    )
