from __future__ import annotations

from pathlib import Path

from kagglebot.kernel_snapshot import (
    best_kernel_snapshot_path,
    capture_best_kernel_snapshot,
    ensure_best_kernel_snapshot,
    restore_best_kernel_snapshot,
)
from kagglebot.paths import CompetitionPaths


def test_capture_and_restore_best_kernel_snapshot(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    kernel_path = paths.kernel_source_dir / "kernel.py"
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text("print('best')\n", encoding="utf-8")

    assert capture_best_kernel_snapshot(paths=paths, run_dir=run_dir)
    assert best_kernel_snapshot_path(run_dir).read_text(encoding="utf-8") == "print('best')\n"

    kernel_path.write_text("print('regressed')\n", encoding="utf-8")

    assert restore_best_kernel_snapshot(paths=paths, run_dir=run_dir)
    assert kernel_path.read_text(encoding="utf-8") == "print('best')\n"


def test_capture_best_kernel_snapshot_returns_false_when_kernel_missing(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")

    assert not capture_best_kernel_snapshot(paths=paths, run_dir=paths.run_dir("run-1"))


def test_ensure_best_kernel_snapshot_does_not_overwrite_existing_snapshot(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    snapshot_path = best_kernel_snapshot_path(run_dir)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text("print('original snapshot')\n", encoding="utf-8")
    kernel_path = paths.kernel_source_dir / "kernel.py"
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text("print('new kernel')\n", encoding="utf-8")

    ensure_best_kernel_snapshot(paths=paths, run_dir=run_dir)

    assert snapshot_path.read_text(encoding="utf-8") == "print('original snapshot')\n"


def test_restore_best_kernel_snapshot_returns_false_when_snapshot_missing(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")

    assert not restore_best_kernel_snapshot(paths=paths, run_dir=paths.run_dir("run-1"))
