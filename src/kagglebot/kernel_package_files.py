from __future__ import annotations

import shutil
from pathlib import Path

from kagglebot.artifact_io import copy_artifact_if_needed

_GENERATED_CACHE_DIR_NAMES = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})
_GENERATED_CACHE_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})


def copy_kernel_sources(source_dir: Path, dest_dir: Path) -> None:
    for path in source_dir.iterdir():
        if _is_generated_kernel_output_dir(path):
            continue
        dest_path = dest_dir / path.name
        if path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(
                path,
                dest_path,
                ignore=shutil.ignore_patterns(
                    *_GENERATED_CACHE_DIR_NAMES,
                    "*.pyc",
                    "*.pyo",
                    ".DS_Store",
                ),
            )
        elif path.is_file():
            if path.suffix.lower() in _GENERATED_CACHE_FILE_SUFFIXES or path.name == ".DS_Store":
                continue
            copy_artifact_if_needed(source=path, destination=dest_path)


def remove_generated_kernel_cache_files(kernel_dir: Path) -> None:
    """Remove local tooling/bytecode caches before fingerprinting and pushing."""
    if not kernel_dir.exists():
        return
    for path in sorted(kernel_dir.rglob("*"), key=lambda candidate: len(candidate.parts), reverse=True):
        if path.is_dir() and path.name in _GENERATED_CACHE_DIR_NAMES:
            shutil.rmtree(path)
            continue
        if path.is_file() and (path.suffix.lower() in _GENERATED_CACHE_FILE_SUFFIXES or path.name == ".DS_Store"):
            path.unlink()


def _is_generated_kernel_output_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name.lower()
    return name in {"output", "outputs", "__pycache__"} or name.startswith(
        ("output-", "outputs-", "output_", "outputs_")
    )


def copy_competition_external_assets(*, base_dir: Path, slug: str, kernel_dir: Path) -> None:
    external_dir = base_dir / slug / "external"
    if not external_dir.exists():
        return
    for path in external_dir.iterdir():
        if not path.is_file():
            continue
        copy_artifact_if_needed(source=path, destination=kernel_dir / path.name)


def copy_shared_kernel_runtime_modules(kernel_dir: Path) -> None:
    runtime_dir = Path(__file__).resolve().parent / "kernel_runtime"
    if not runtime_dir.exists():
        return
    for path in sorted(runtime_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        copy_artifact_if_needed(source=path, destination=kernel_dir / path.name)


def sync_plan_snapshot(*, plan_path: Path, targets: list[Path]) -> None:
    if not plan_path.exists():
        return
    for target in targets:
        if target.resolve() == plan_path.resolve():
            continue
        copy_artifact_if_needed(source=plan_path, destination=target)
