from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.hashing import sha256_path, sha256_text
from kagglebot.json_utils import load_json_object, write_json_object

_RESUME_DIRNAME = "durable_kernel_state"
_CHECKPOINT_DIR = Path("outputs/checkpoints")


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
