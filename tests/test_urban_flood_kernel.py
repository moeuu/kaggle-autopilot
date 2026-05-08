from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.competition_artifact


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


def test_select_results_by_model_prefers_per_node_type_mix() -> None:
    mod = _load_kernel_module()
    result_a = {
        "name": "pipe_a",
        "selected_variant": "pipe_a",
        "metric": 0.30,
        "cv_event_preds": {
            (2, 101, 1): np.asarray([[0.0]], dtype=np.float32),
            (2, 101, 2): np.asarray([[3.0]], dtype=np.float32),
        },
        "cv_event_targets": {
            (2, 101, 1): np.asarray([[0.0]], dtype=np.float32),
            (2, 101, 2): np.asarray([[1.0]], dtype=np.float32),
        },
        "test_event_preds": {
            (2, 101): np.asarray([[0.0, 3.0]], dtype=np.float32),
        },
        "std_lookup": {(2, 1): 1.0, (2, 2): 1.0},
        "seed_selection": {"2": {"selection": "pipe_a"}},
    }
    result_b = {
        "name": "pipe_b",
        "selected_variant": "pipe_b",
        "metric": 0.30,
        "cv_event_preds": {
            (2, 101, 1): np.asarray([[2.0]], dtype=np.float32),
            (2, 101, 2): np.asarray([[1.0]], dtype=np.float32),
        },
        "cv_event_targets": {
            (2, 101, 1): np.asarray([[0.0]], dtype=np.float32),
            (2, 101, 2): np.asarray([[1.0]], dtype=np.float32),
        },
        "test_event_preds": {
            (2, 101): np.asarray([[2.0, 1.0]], dtype=np.float32),
        },
        "std_lookup": {(2, 1): 1.0, (2, 2): 1.0},
        "seed_selection": {"2": {"selection": "pipe_b"}},
    }

    selected = mod._select_results_by_model(
        [result_a, result_b],
        [2],
        {2: {101: SimpleNamespace(template=SimpleNamespace(n_1d=1))}},
    )

    assert selected is not None
    assert selected["selected_variant"] == "per_model_node_pipeline_mix"
    assert selected["selected_pipelines_by_model"]["2"]["pipeline"] == "mixed_by_node_type"
    assert selected["selected_pipelines_by_model_node"]["model_2_node_type_1"]["pipeline"] == "pipe_a"
    assert selected["selected_pipelines_by_model_node"]["model_2_node_type_2"]["pipeline"] == "pipe_b"
    assert np.allclose(selected["test_event_preds"][(2, 101)], np.asarray([[0.0, 1.0]], dtype=np.float32))
