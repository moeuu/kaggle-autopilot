from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.hashing import sha256_path, sha256_text
from kagglebot.json_utils import load_json_object, write_json_object

_RESUME_DIRNAME = "durable_kernel_state"
_CHECKPOINT_DIR = Path("outputs/checkpoints")
_SHARED_STATE_DIRS = (
    Path("kernel_output/cache"),
    Path("kernel_output/adapters"),
    Path("models"),
)


def _has_entries(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True) == right.resolve(strict=True)
    except OSError:
        return False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _merge_tree_by_move(source: Path, destination: Path) -> None:
    """Merge a stopped stage into durable storage without copying large files."""
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir() and not child.is_symlink() and target.is_dir() and not target.is_symlink():
            _merge_tree_by_move(child, target)
            continue
        if target.exists() or target.is_symlink():
            try:
                if os.path.samefile(child, target):
                    _remove_path(child)
                    continue
            except OSError:
                pass
            _remove_path(target)
        os.replace(child, target)
    source.rmdir()


def preserve_local_kernel_shared_state(*, kernel_stage_dir: Path, durable_root: Path) -> list[Path]:
    """Move reusable same-run caches out of a disposable local-kernel stage.

    These paths are deliberately limited to caches, downloaded models, and
    adapter state. Generated kernels remain responsible for content-addressing
    and validating anything they reuse after a source repair.
    """
    preserved: list[Path] = []
    shared_root = durable_root / "shared"
    for relative in _SHARED_STATE_DIRS:
        source = kernel_stage_dir / relative
        destination = shared_root / relative
        if source.is_symlink() and _same_path(source, destination):
            preserved.append(destination)
            continue
        if not _has_entries(source):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _merge_tree_by_move(source, destination)
        else:
            os.replace(source, destination)
        preserved.append(destination)
    if preserved:
        write_json_object(
            shared_root / "shared_state_manifest.json",
            {
                "schema_version": 1,
                "preserved_at": datetime.now(UTC).isoformat(),
                "paths": [str(path.relative_to(shared_root)) for path in preserved],
                "restore_policy": "same_run_iteration_content_addressed_cache",
            },
            sort_keys=True,
        )
    return preserved


def restore_local_kernel_shared_state(*, kernel_stage_dir: Path, durable_root: Path) -> list[Path]:
    """Link reusable state into a recreated stage, including after source fixes."""
    restored: list[Path] = []
    shared_root = durable_root / "shared"
    manifest = load_json_object(shared_root / "shared_state_manifest.json")
    declared = set(manifest.get("paths", [])) if manifest is not None else set()
    for relative in _SHARED_STATE_DIRS:
        if str(relative) not in declared:
            continue
        source = shared_root / relative
        destination = kernel_stage_dir / relative
        if not _has_entries(source):
            continue
        if destination.exists() or destination.is_symlink():
            if _same_path(destination, source):
                restored.append(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=True)
        restored.append(destination)
    return restored


def _source_fingerprint(kernel_stage_dir: Path) -> tuple[str, dict[str, str]] | None:
    kernel_path = kernel_stage_dir / "kernel.py"
    plan_path = kernel_stage_dir / "plan.json"
    if not kernel_path.is_file():
        return None
    hashes = {"kernel.py": sha256_path(kernel_path)}
    if plan_path.is_file():
        hashes["plan.json"] = sha256_path(plan_path)
    fingerprint = sha256_text("\n".join(f"{name}={hashes[name]}" for name in sorted(hashes)))
    return fingerprint, hashes


def durable_state_root(*, base_dir: Path, slug: str, run_id: str, iteration: int) -> Path:
    return base_dir / slug / "runs" / run_id / f"iter-{iteration}" / _RESUME_DIRNAME


def preserve_local_kernel_checkpoints(
    *,
    kernel_stage_dir: Path,
    durable_root: Path,
) -> Path | None:
    """Move staged checkpoints outside the disposable kernel directory.

    The runner recreates the staging directory before every local attempt. A
    source fingerprint is retained so checkpoints are only restored into the
    exact kernel and plan that produced them.
    """
    preserve_local_kernel_shared_state(kernel_stage_dir=kernel_stage_dir, durable_root=durable_root)
    checkpoint_dir = kernel_stage_dir / _CHECKPOINT_DIR
    fingerprint_row = _source_fingerprint(kernel_stage_dir)
    if fingerprint_row is None or not checkpoint_dir.is_dir() or not any(checkpoint_dir.rglob("*")):
        return None
    fingerprint, source_hashes = fingerprint_row
    snapshot_dir = durable_root / fingerprint
    durable_checkpoints = snapshot_dir / "checkpoints"
    durable_checkpoints.parent.mkdir(parents=True, exist_ok=True)
    if durable_checkpoints.exists():
        shutil.copytree(checkpoint_dir, durable_checkpoints, dirs_exist_ok=True)
        shutil.rmtree(checkpoint_dir)
    else:
        checkpoint_dir.replace(durable_checkpoints)
    manifest = {
        "schema_version": 1,
        "source_fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "preserved_at": datetime.now(UTC).isoformat(),
        "checkpoint_tree_sha256": sha256_path(durable_checkpoints),
        "restore_policy": "exact_staged_kernel_and_plan",
    }
    write_json_object(snapshot_dir / "resume_manifest.json", manifest, sort_keys=True)
    return snapshot_dir


def restore_local_kernel_checkpoints(
    *,
    kernel_stage_dir: Path,
    durable_root: Path,
) -> Path | None:
    """Restore checkpoints only when staged source and plan match exactly."""
    fingerprint_row = _source_fingerprint(kernel_stage_dir)
    if fingerprint_row is None:
        return None
    fingerprint, source_hashes = fingerprint_row
    snapshot_dir = durable_root / fingerprint
    manifest_path = snapshot_dir / "resume_manifest.json"
    durable_checkpoints = snapshot_dir / "checkpoints"
    manifest = load_json_object(manifest_path)
    if manifest is None or not durable_checkpoints.is_dir():
        return None
    if manifest.get("source_fingerprint") != fingerprint or manifest.get("source_hashes") != source_hashes:
        return None
    expected_tree_hash = str(manifest.get("checkpoint_tree_sha256") or "")
    if not expected_tree_hash or sha256_path(durable_checkpoints) != expected_tree_hash:
        return None
    destination = kernel_stage_dir / _CHECKPOINT_DIR
    shutil.copytree(durable_checkpoints, destination, dirs_exist_ok=True)
    return destination
