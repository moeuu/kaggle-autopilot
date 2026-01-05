from __future__ import annotations


def truncate_lines(
    text: str,
    max_lines: int = 5,
    suffix: str = " ... (truncated)",
    max_chars: int = 2000,
) -> str:
    if max_lines <= 0 or max_chars <= 0:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if suffix:
            lines[-1] = _append_suffix(lines[-1], suffix)
    trimmed: list[str] = []
    for line in lines:
        if len(line) > max_chars:
            line = _append_suffix(line[:max_chars].rstrip(), suffix)
        trimmed.append(line)
    return "\n".join(trimmed)


def _append_suffix(line: str, suffix: str) -> str:
    if not suffix:
        return line
    if line.endswith(suffix):
        return line
    return f"{line}{suffix}"
