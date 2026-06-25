from __future__ import annotations

import shutil
from pathlib import Path

from kagglebot.paths import CompetitionPaths

BEST_KERNEL_SNAPSHOT_FILENAME = "best_kernel.py"


def best_kernel_snapshot_path(run_dir: Path) -> Path:
    return run_dir / BEST_KERNEL_SNAPSHOT_FILENAME


def capture_best_kernel_snapshot(*, paths: CompetitionPaths, run_dir: Path) -> bool:
    kernel_path = paths.kernel_source_dir / "kernel.py"
    if not kernel_path.exists():
        return False
    snapshot_path = best_kernel_snapshot_path(run_dir)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(kernel_path, snapshot_path)
    except OSError:
        return False
    return True


def ensure_best_kernel_snapshot(*, paths: CompetitionPaths, run_dir: Path) -> None:
    snapshot_path = best_kernel_snapshot_path(run_dir)
    if snapshot_path.exists():
        return
    capture_best_kernel_snapshot(paths=paths, run_dir=run_dir)


def restore_best_kernel_snapshot(*, paths: CompetitionPaths, run_dir: Path) -> bool:
    snapshot_path = best_kernel_snapshot_path(run_dir)
    kernel_path = paths.kernel_source_dir / "kernel.py"
    if not snapshot_path.exists():
        return False
    try:
        shutil.copy2(snapshot_path, kernel_path)
    except OSError:
        return False
    return True
