from __future__ import annotations

import shutil
from pathlib import Path

from kagglebot.kernel_outputs import copy_artifact_if_needed


def copy_kernel_sources(source_dir: Path, dest_dir: Path) -> None:
    for path in source_dir.iterdir():
        if path.name in {"output", "outputs", "__pycache__"}:
            continue
        dest_path = dest_dir / path.name
        if path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(path, dest_path)
        elif path.is_file():
            if path.suffix == ".pyc":
                continue
            copy_artifact_if_needed(source=path, destination=dest_path)


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
