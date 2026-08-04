from __future__ import annotations

import shutil
from pathlib import Path

from kagglebot.local_kernel_resume import (
    preserve_local_kernel_checkpoints,
    preserve_local_kernel_shared_state,
    restore_local_kernel_checkpoints,
    restore_local_kernel_shared_state,
)


def _stage(root: Path, *, kernel: str = "print('v1')", plan: str = '{"epochs": 50}') -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.py").write_text(kernel, encoding="utf-8")
    (root / "plan.json").write_text(plan, encoding="utf-8")
    return root


def test_preserve_and_restore_exact_source_checkpoints(tmp_path: Path) -> None:
    stage = _stage(tmp_path / "stage")
    checkpoint = stage / "outputs" / "checkpoints" / "fold0" / "step-001000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    snapshot = preserve_local_kernel_checkpoints(
        kernel_stage_dir=stage,
        durable_root=tmp_path / "durable",
    )

    assert snapshot is not None
    assert not checkpoint.exists()
    recreated = _stage(tmp_path / "recreated")
    restored = restore_local_kernel_checkpoints(
        kernel_stage_dir=recreated,
        durable_root=tmp_path / "durable",
    )
    assert restored is not None
    assert (restored / "fold0" / "step-001000.pt").read_bytes() == b"checkpoint"


def test_restore_rejects_changed_kernel_or_plan(tmp_path: Path) -> None:
    stage = _stage(tmp_path / "stage")
    checkpoint = stage / "outputs" / "checkpoints" / "fold0" / "step-001000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    preserve_local_kernel_checkpoints(kernel_stage_dir=stage, durable_root=tmp_path / "durable")

    changed_kernel = _stage(tmp_path / "changed-kernel", kernel="print('v2')")
    assert (
        restore_local_kernel_checkpoints(
            kernel_stage_dir=changed_kernel,
            durable_root=tmp_path / "durable",
        )
        is None
    )
    changed_plan = _stage(tmp_path / "changed-plan", plan='{"epochs": 1}')
    assert (
        restore_local_kernel_checkpoints(
            kernel_stage_dir=changed_plan,
            durable_root=tmp_path / "durable",
        )
        is None
    )


def test_preserve_is_noop_without_checkpoints(tmp_path: Path) -> None:
    stage = _stage(tmp_path / "stage")
    assert (
        preserve_local_kernel_checkpoints(
            kernel_stage_dir=stage,
            durable_root=tmp_path / "durable",
        )
        is None
    )


def test_shared_state_survives_source_change_and_stage_removal(tmp_path: Path) -> None:
    stage = _stage(tmp_path / "stage")
    cache_file = stage / "kernel_output" / "cache" / "reference" / "part.jsonl"
    model_file = stage / "models" / "model" / "config.json"
    cache_file.parent.mkdir(parents=True)
    model_file.parent.mkdir(parents=True)
    cache_file.write_text('{"image_file":"a.jpg"}\n', encoding="utf-8")
    model_file.write_text("{}", encoding="utf-8")

    preserved = preserve_local_kernel_shared_state(
        kernel_stage_dir=stage,
        durable_root=tmp_path / "durable",
    )

    assert len(preserved) == 2
    shutil.rmtree(stage)
    recreated = _stage(tmp_path / "stage", kernel="print('fixed')")
    restored = restore_local_kernel_shared_state(
        kernel_stage_dir=recreated,
        durable_root=tmp_path / "durable",
    )
    assert len(restored) == 2
    restored_cache = recreated / "kernel_output" / "cache"
    assert restored_cache.is_symlink()
    assert (restored_cache / "reference" / "part.jsonl").read_text(encoding="utf-8").endswith("\n")

    (restored_cache / "reference" / "next.jsonl").write_text("next\n", encoding="utf-8")
    shutil.rmtree(recreated)
    assert (tmp_path / "durable" / "shared" / "kernel_output" / "cache" / "reference" / "next.jsonl").is_file()


def test_checkpoint_preservation_also_preserves_shared_state_without_checkpoint(tmp_path: Path) -> None:
    stage = _stage(tmp_path / "stage")
    cache_file = stage / "kernel_output" / "cache" / "entry.bin"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cache")

    assert preserve_local_kernel_checkpoints(kernel_stage_dir=stage, durable_root=tmp_path / "durable") is None
    restored_stage = _stage(tmp_path / "restored", kernel="print('changed')")
    restored = restore_local_kernel_shared_state(
        kernel_stage_dir=restored_stage,
        durable_root=tmp_path / "durable",
    )
    assert restored
    assert (restored_stage / "kernel_output" / "cache" / "entry.bin").read_bytes() == b"cache"
