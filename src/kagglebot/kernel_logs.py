from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rich import print

from kagglebot.logging_utils import truncate_lines


@dataclass
class KernelLogState:
    seen_lines: dict[Path, int] = field(default_factory=dict)
    seen_json: dict[Path, int] = field(default_factory=dict)
    seen_size: dict[Path, int] = field(default_factory=dict)
    last_log_at: float | None = None
    last_heartbeat: float = 0.0


def log_candidates(output_dir: Path) -> list[Path]:
    candidates = []
    for name in ("stdout.txt", "stderr.txt", "output.log", "log.txt", "logs.txt"):
        path = output_dir / name
        if path.exists():
            candidates.append(path)
    candidates.extend(sorted(output_dir.rglob("*.log")))
    return candidates


def print_kernel_logs(output_dir: Path, state: KernelLogState) -> bool:
    printed = False
    for path in log_candidates(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        size = len(text)
        prev_size = state.seen_size.get(path, 0)
        if size < prev_size:
            state.seen_lines[path] = 0
            state.seen_json[path] = 0
        state.seen_size[path] = size

        json_events = parse_json_log(text)
        if json_events is not None:
            last = state.seen_json.get(path, 0)
            if len(json_events) <= last:
                continue
            new_events = json_events[last:]
            state.seen_json[path] = len(json_events)
            formatted = format_log_events(new_events)
            if not formatted:
                continue
            print(f"[cyan]kernel log[/cyan]: {path.name}")
            print(truncate_lines("\n".join(formatted), max_lines=5))
            printed = True
            continue

        lines = text.splitlines()
        last = state.seen_lines.get(path, 0)
        if len(lines) <= last:
            continue
        new_lines = lines[last:]
        state.seen_lines[path] = len(lines)
        print(f"[cyan]kernel log[/cyan]: {path.name}")
        print(truncate_lines("\n".join(new_lines), max_lines=5))
        printed = True
    return printed


def detect_failure_in_logs(output_dir: Path) -> str | None:
    for path in log_candidates(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Traceback (most recent call last)" not in text:
            continue
        tail = collect_log_tail_from_text(path, text)
        if tail:
            return tail
        return f"{path.name}\nTraceback detected"
    return None


def collect_log_tail(output_dir: Path, max_lines: int = 50) -> str | None:
    candidates = log_candidates(output_dir)
    if not candidates:
        return None
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Traceback (most recent call last)" in text:
            return collect_log_tail_from_text(path, text, max_lines=max_lines)
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Error" in text or "Exception" in text:
            return collect_log_tail_from_text(path, text, max_lines=max_lines)
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tail = collect_log_tail_from_text(path, text, max_lines=max_lines)
        if tail:
            return tail
    return None


def collect_log_tail_from_text(path: Path, text: str, max_lines: int = 50) -> str | None:
    json_events = parse_json_log(text)
    if json_events is not None:
        formatted = format_log_events(json_events)
        if not formatted:
            return None
        start = find_error_marker_index(formatted)
        if start is None:
            start = max(len(formatted) - max_lines, 0)
        else:
            if len(formatted) - start > max_lines:
                start = max(len(formatted) - max_lines, start)
        tail = "\n".join(formatted[start:])
        return f"{path.name}\n{tail}".strip()
    lines = text.splitlines()
    if not lines:
        return None
    start = find_error_marker_index(lines)
    if start is None:
        start = max(len(lines) - max_lines, 0)
    else:
        if len(lines) - start > max_lines:
            start = max(len(lines) - max_lines, start)
    tail = "\n".join(lines[start:])
    return f"{path.name}\n{tail}".strip()


def find_error_marker_index(lines: list[str]) -> int | None:
    markers = ("Traceback", "Error", "Exception")
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if any(marker in line for marker in markers):
            return idx
    return None


def parse_json_log(text: str) -> list[dict[str, object]] | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("logs"), list):
            return [item for item in payload["logs"] if isinstance(item, dict)]
        return None
    return None


def format_log_events(events: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        data = event.get("data")
        if not isinstance(data, str) or not data:
            continue
        stream = event.get("stream_name")
        prefix = f"[{stream}] " if isinstance(stream, str) and stream else ""
        for line in data.splitlines():
            lines.append(f"{prefix}{line}")
    return lines
