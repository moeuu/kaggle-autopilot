from __future__ import annotations

import json

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_contracts import (
    enforce_competition_kernel_contract,
    extract_kernel_size_markers,
    requires_bvs_kernel_contract,
    resolve_kernel_contract,
)


def test_requires_bvs_kernel_contract_matches_family_slug() -> None:
    assert requires_bvs_kernel_contract("beyond-visible-spectrum-ai-for-agriculture-2026p2") is True
    assert requires_bvs_kernel_contract("demo") is False
    assert resolve_kernel_contract(slug="demo", policy_contract="bvs") == "bvs_timm_size_ensemble"
    assert resolve_kernel_contract(slug="demo", policy_contract="unknown") is None


def test_extract_kernel_size_markers_reads_load_and_img_sizes() -> None:
    log_text = "\n".join(
        [
            "tri_branch cfg: load_size=224 crop_size=128",
            "rgb cfg: img_size=160",
            "ignored: load_size=bad",
        ]
    )

    assert extract_kernel_size_markers(log_text) == [224, 160]


def test_enforce_competition_kernel_contract_ignores_non_bvs_slug(tmp_path) -> None:
    enforce_competition_kernel_contract(slug="demo", logs_dir=tmp_path, metrics_path=None)


def test_enforce_competition_kernel_contract_rejects_regressed_bvs_payload(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "local_kernel_stdout.log").write_text(
        "falling back to smallspectralencoder for rgb\ntri_branch cfg: load_size=64\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": "resnet50",
                "chosen_pipeline": "tri_branch_convnext_spectral",
                "pipelines": [{"name": "tri_branch_convnext_spectral", "score": 0.68}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError) as excinfo:
        enforce_competition_kernel_contract(
            slug="beyond-visible-spectrum-ai-for-agriculture-2026p2",
            logs_dir=logs_dir,
            metrics_path=metrics_path,
        )

    message = str(excinfo.value)
    assert "BVS kernel contract failed" in message
    assert "load_size below 128" in message
    assert "Weak fallback backbone" in message
    assert "ensemble-based" in message


def test_enforce_competition_kernel_contract_allows_bvs_ensemble_payload(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "local_kernel_stdout.log").write_text("tri_branch cfg: load_size=224\n", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": "convnext_tiny",
                "chosen_pipeline": "ensemble_tri_branch__tabular",
                "pipelines": [
                    {"name": "tri_branch_timm_gated", "score": 0.70},
                    {"name": "tabular_fallback", "score": 0.66},
                    {"name": "ensemble_tri_branch__tabular", "score": 0.72},
                ],
            }
        ),
        encoding="utf-8",
    )

    enforce_competition_kernel_contract(
        slug="beyond-visible-spectrum-ai-for-agriculture-2026p2",
        logs_dir=logs_dir,
        metrics_path=metrics_path,
    )
