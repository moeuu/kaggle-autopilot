from __future__ import annotations

import json
from pathlib import Path

from kagglebot.kernel_metadata import (
    resolve_kernel_slug,
    resolve_submit_kernel_slug,
    sanitize_kernel_slug,
    write_kernel_metadata,
)
from kagglebot.kernel_sources import KernelSourceConfig


def test_sanitize_kernel_slug() -> None:
    assert sanitize_kernel_slug("KaggleBot Titan! 2024") == "kagglebot-titan-2024"


def test_submit_kernel_slug_stays_distinct_when_long_slug_is_truncated() -> None:
    slug = "deep-past-initiative-machine-translation"
    run_id = "20260318T153341Z-52de3f45"
    train_slug = resolve_kernel_slug(None, slug, run_id, 1)
    submit_slug = resolve_submit_kernel_slug(None, slug, run_id, 1)

    assert train_slug == "kagglebot-deep-past-initiative-machine-t-de3f45-i1"
    assert submit_slug.startswith("kagglebot-submit-")
    assert submit_slug != train_slug
    assert len(submit_slug) <= 50


def test_submit_kernel_slug_isolated_by_official_machine_shape() -> None:
    slug = "arc-prize-2026-arc-agi-2"
    run_id = "20260714T155630Z-4d48ff70"

    default_slug = resolve_submit_kernel_slug(None, slug, run_id, 1)
    t4_slug = resolve_submit_kernel_slug(None, slug, run_id, 1, machine_shape="NvidiaTeslaT4")
    p100_slug = resolve_submit_kernel_slug(None, slug, run_id, 1, machine_shape="NvidiaTeslaP100")

    assert "-t4-" in t4_slug
    assert "-p100-" in p100_slug
    assert len({default_slug, t4_slug, p100_slug}) == 3
    assert len(t4_slug) <= 50


def test_training_kernel_slug_isolated_by_official_machine_shape() -> None:
    slug = "arc-prize-2026-arc-agi-3"
    run_id = "20260714T060514Z-134b7f94"

    default_slug = resolve_kernel_slug(None, slug, run_id, 1)
    rtx_slug = resolve_kernel_slug(None, slug, run_id, 1, machine_shape="NvidiaRtxPro6000")

    assert "-rtx-pro-6000-" in rtx_slug
    assert default_slug != rtx_slug
    assert len(rtx_slug) <= 50


def test_write_kernel_metadata_ignores_invalid_existing_metadata(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel-metadata.json").write_text("{", encoding="utf-8")

    write_kernel_metadata(
        kernel_dir=kernel_dir,
        kernel_id="user/demo",
        title="demo",
        code_file="kernel.py",
        kernel_type="script",
        accelerator="gpu",
        enable_internet=False,
        competition_slug="competition",
    )

    payload = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert payload["id"] == "user/demo"
    assert payload["competition_sources"] == ["competition"]
    assert payload["dataset_sources"] == []
    assert payload["kernel_sources"] == []
    assert payload["model_sources"] == []


def test_write_kernel_metadata_uses_source_config(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()

    write_kernel_metadata(
        kernel_dir=kernel_dir,
        kernel_id="user/demo",
        title="demo",
        code_file="kernel.py",
        kernel_type="script",
        accelerator="cpu",
        enable_internet=False,
        competition_slug="competition",
        source_config=KernelSourceConfig(
            dataset_sources=("alice/data",),
            kernel_sources=("bob/kernel",),
            model_sources=("carol/model/PyTorch/default/1",),
        ),
    )

    payload = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert payload["enable_gpu"] is False
    assert payload["enable_tpu"] is False
    assert payload["dataset_sources"] == ["alice/data"]
    assert payload["kernel_sources"] == ["bob/kernel"]
    assert payload["model_sources"] == ["carol/model/PyTorch/default/1"]


def test_write_kernel_metadata_attaches_required_model_sources(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()

    write_kernel_metadata(
        kernel_dir=kernel_dir,
        kernel_id="user/demo",
        title="demo",
        code_file="kernel.py",
        kernel_type="script",
        accelerator="gpu",
        enable_internet=False,
        competition_slug="competition",
        source_config=KernelSourceConfig(
            required_model_sources=("google/gemma-4/Transformers/gemma-4-31b-it/1",),
        ),
    )

    payload = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert payload["model_sources"] == ["google/gemma-4/Transformers/gemma-4-31b-it/1"]
