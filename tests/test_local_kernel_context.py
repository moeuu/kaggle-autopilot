from __future__ import annotations

import json
from pathlib import Path

from kagglebot.local_kernel_context import (
    load_dataset_profile_identity,
    stage_local_kernel_context_profile,
    stage_local_kernel_data_dir,
)


def test_load_dataset_profile_identity_ignores_missing_invalid_or_non_object_payload(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    assert load_dataset_profile_identity(context_dir=context_dir) == (None, None)

    profile_path = context_dir / "dataset_profile.json"
    profile_path.write_text("{", encoding="utf-8")
    assert load_dataset_profile_identity(context_dir=context_dir) == (None, None)

    profile_path.write_text("[]", encoding="utf-8")
    assert load_dataset_profile_identity(context_dir=context_dir) == (None, None)

    profile_path.write_text(json.dumps({"target_column": "target", "id_column": "id"}), encoding="utf-8")
    assert load_dataset_profile_identity(context_dir=context_dir) == ("target", "id")


def test_stage_local_kernel_data_dir_replaces_stale_file_target(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "images").mkdir(exist_ok=True)
    (data_dir / "images" / "a.jpg").write_bytes(b"img")

    run_dir = tmp_path / "demo" / "kernels" / "run-stale"
    run_dir.mkdir(parents=True, exist_ok=True)
    stale_target = run_dir / "data"
    stale_target.write_text("stale", encoding="utf-8")

    stage_local_kernel_data_dir(base_dir=tmp_path, slug="demo", run_dir=run_dir)

    assert stale_target.exists()
    assert stale_target.is_dir() or stale_target.is_symlink()
    assert (stale_target / "sample_submission.csv").exists()
    assert (stale_target / "images" / "a.jpg").exists()
    compat_target = tmp_path / "demo" / "artifacts" / "demo" / "data"
    assert compat_target.exists()
    assert compat_target.is_dir() or compat_target.is_symlink()
    assert (compat_target / "sample_submission.csv").exists()
    assert (compat_target / "images" / "a.jpg").exists()


def test_stage_local_kernel_context_profile_copies_dataset_profile(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "dataset_profile.json").write_text('{"target_column": "label"}\n', encoding="utf-8")

    run_dir = tmp_path / "demo" / "kernels" / "run-1"
    stale_target = run_dir / "context" / "dataset_profile.json"
    stale_target.parent.mkdir(parents=True)
    stale_target.write_text("stale", encoding="utf-8")

    stage_local_kernel_context_profile(base_dir=tmp_path, slug="demo", run_dir=run_dir)

    assert stale_target.read_text(encoding="utf-8") == '{"target_column": "label"}\n'


def test_stage_local_kernel_context_profile_noops_when_target_is_source(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True)
    profile_path = context_dir / "dataset_profile.json"
    profile_path.write_text('{"id_column": "id"}\n', encoding="utf-8")

    stage_local_kernel_context_profile(base_dir=tmp_path, slug="demo", run_dir=tmp_path / "demo")

    assert profile_path.read_text(encoding="utf-8") == '{"id_column": "id"}\n'
