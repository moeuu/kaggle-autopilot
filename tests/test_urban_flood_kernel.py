from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_kernel_module():
    kernel_path = Path("artifacts/urban-flood-modelling/kernel/kernel.py")
    spec = importlib.util.spec_from_file_location("urban_flood_kernel", kernel_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promote_flat_full_resolution_for_large_competition_root(tmp_path: Path) -> None:
    mod = _load_kernel_module()
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in mod.FLAT_FULL_REQUIRED_FILES:
        (data_root / name).write_text("x\n", encoding="utf-8")

    resolution = mod.DataResolution(
        dataset_root=data_root,
        sample_submission_root=data_root,
        dataset_kind="sample",
        full_dataset_resolved=False,
        source="sample_fallback",
    )
    sample_meta = mod.SampleSubmissionMeta(
        path=data_root / "sample_submission.csv",
        columns=[],
        row_count=mod.FLAT_FULL_SAMPLE_ROW_COUNT_MIN,
        model_event_ids={1: [101], 2: [202]},
    )

    promoted = mod._promote_flat_full_resolution(
        resolution,
        sample_meta=sample_meta,
        train_events_by_model={1: {101: object()}, 2: {202: object()}},
        test_events_by_model={1: {101: object()}, 2: {202: object()}},
    )

    assert promoted.dataset_kind == "full"
    assert promoted.full_dataset_resolved is True
    assert promoted.source.endswith("+flat_full_root")


def test_promote_flat_full_resolution_keeps_sample_for_small_root(tmp_path: Path) -> None:
    mod = _load_kernel_module()
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in mod.FLAT_FULL_REQUIRED_FILES:
        (data_root / name).write_text("x\n", encoding="utf-8")

    resolution = mod.DataResolution(
        dataset_root=data_root,
        sample_submission_root=data_root,
        dataset_kind="sample",
        full_dataset_resolved=False,
        source="sample_fallback",
    )
    sample_meta = mod.SampleSubmissionMeta(
        path=data_root / "sample_submission.csv",
        columns=[],
        row_count=128,
        model_event_ids={1: [101]},
    )

    promoted = mod._promote_flat_full_resolution(
        resolution,
        sample_meta=sample_meta,
        train_events_by_model={1: {101: object()}},
        test_events_by_model={1: {101: object()}},
    )

    assert promoted == resolution
