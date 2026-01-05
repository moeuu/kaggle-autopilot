from __future__ import annotations


def truncate_lines(text: str, max_lines: int = 5, suffix: str = " ... (truncated)") -> str:
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    trimmed = lines[:max_lines]
    if suffix:
        trimmed[-1] = f"{trimmed[-1]}{suffix}"
    return "\n".join(trimmed)
