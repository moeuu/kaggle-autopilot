from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

_GENERATED_RUNTIME_DIR_NAMES = frozenset(
    {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        "venv",
    }
)


def iter_named_output_paths(root: Path, filename: str) -> Iterator[Path]:
    """Yield exact-name output candidates without descending into runtimes."""

    if not root.is_dir():
        return
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        matching_dirs = [name for name in dirnames if name == filename]
        dirnames[:] = [name for name in dirnames if not _is_generated_runtime_dir(name)]
        for name in matching_dirs:
            yield current_path / name
        if filename in filenames:
            yield current_path / filename


def _is_generated_runtime_dir(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in _GENERATED_RUNTIME_DIR_NAMES or normalized.endswith("_runtime_site")
