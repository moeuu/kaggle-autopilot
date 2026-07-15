from __future__ import annotations

from pathlib import Path

from kagglebot.local_kernel_resume import (
    preserve_local_kernel_checkpoints,
    restore_local_kernel_checkpoints,
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
