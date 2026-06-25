from __future__ import annotations

import ast
import re
from pathlib import Path

from kagglebot.exceptions import KernelFailedError

INVALID_KERNEL_SOURCE_RE = re.compile(
    r"The following are not valid (?P<kind>dataset|kernel|model) sources and could not be added to the kernel:\s*"
    r"(?P<items>\[[^\]]*\])",
    re.IGNORECASE,
)


def extract_invalid_kernel_push_sources(output: str) -> dict[str, list[str]]:
    invalid_sources: dict[str, list[str]] = {}
    if not output:
        return invalid_sources

    for match in INVALID_KERNEL_SOURCE_RE.finditer(output):
        kind = str(match.group("kind") or "").strip().lower()
        raw_items = str(match.group("items") or "").strip()
        if not kind or not raw_items:
            continue
        try:
            parsed_items = ast.literal_eval(raw_items)
        except (SyntaxError, ValueError):
            parsed_items = []
        if not isinstance(parsed_items, list):
            continue
        cleaned = [str(item).strip() for item in parsed_items if str(item).strip()]
        if cleaned:
            invalid_sources.setdefault(kind, []).extend(cleaned)

    for kind, refs in list(invalid_sources.items()):
        invalid_sources[kind] = list(dict.fromkeys(refs))
    return invalid_sources


def raise_for_invalid_kernel_push_sources(output: str, *, kernel_dir: Path) -> None:
    invalid_sources = extract_invalid_kernel_push_sources(output)
    if not invalid_sources:
        return
    details = ", ".join(f"{kind}={','.join(refs)}" for kind, refs in sorted(invalid_sources.items()) if refs)
    raise KernelFailedError(
        "Kaggle kernel push rejected source references: "
        f"{details}. Fix {kernel_dir / 'kernel-metadata.json'} before retrying."
    )
