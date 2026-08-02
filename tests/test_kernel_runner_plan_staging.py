from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.hashing import sha256_path
from kagglebot.kernel_plan_validation import validate_local_kernel_plan_runtime_hyperparameters
from kagglebot.kernel_runner import _stage_local_kernel_plan_snapshot


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_stage_local_kernel_plan_replaces_stale_snapshot_with_authoritative_plan(tmp_path: Path) -> None:
    source_plan_path = tmp_path / "competition" / "plan.json"
    staged_plan_path = tmp_path / "competition" / "kernels" / "run-1" / "local-iter-2" / "plan.json"
    stale_plan = {
        "stability_seeds": [2024, 777],
        "pipelines": [
            {
                "name": "nested_cross_family_stacker",
                "key_hyperparameters": {"stability_seeds": [2024, 777]},
            }
        ],
    }
    authoritative_plan = {
        "stability_seeds": [2024, 777],
        "runtime_budget": {"stability_seeds": 2},
        "pipelines": [
            {
                "name": "nested_oof_ensemble",
                "key_hyperparameters": {
                    "stability_seed_1": 2024,
                    "stability_seed_2": 777,
                },
            }
        ],
    }
    _write_json(source_plan_path, authoritative_plan)
    _write_json(staged_plan_path, stale_plan)

    _stage_local_kernel_plan_snapshot(
        source_plan_path=source_plan_path,
        targets=[staged_plan_path],
    )

    assert sha256_path(staged_plan_path) == sha256_path(source_plan_path)
    staged = json.loads(staged_plan_path.read_text(encoding="utf-8"))
    assert [pipeline["name"] for pipeline in staged["pipelines"]] == ["nested_oof_ensemble"]
    hyperparameters = staged["pipelines"][0]["key_hyperparameters"]
    assert hyperparameters["stability_seed_1"] == 2024
    assert hyperparameters["stability_seed_2"] == 777
    assert "stability_seeds" not in hyperparameters
    validate_local_kernel_plan_runtime_hyperparameters(staged_plan_path)


def test_stage_local_kernel_plan_hash_mismatch_reports_pair_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_plan_path = tmp_path / "competition" / "plan.json"
    staged_plan_path = tmp_path / "stage" / "plan.json"
    _write_json(
        source_plan_path,
        {"pipelines": [{"name": "nested_oof_ensemble", "key_hyperparameters": {"seed": 2024}}]},
    )

    def copy_corrupt_plan(*, plan_path: Path, targets: list[Path]) -> None:  # noqa: ARG001
        for target in targets:
            _write_json(
                target,
                {"pipelines": [{"name": "corrupt_staged_pipeline", "key_hyperparameters": {}}]},
            )

    monkeypatch.setattr(
        "kagglebot.kernel_runner._kernel_package_files.sync_plan_snapshot",
        copy_corrupt_plan,
    )

    with pytest.raises(KernelFailedError) as exc_info:
        _stage_local_kernel_plan_snapshot(
            source_plan_path=source_plan_path,
            targets=[staged_plan_path],
        )

    message = str(exc_info.value)
    assert f"source_plan={source_plan_path}" in message
    assert f"source_sha256={sha256_path(source_plan_path)}" in message
    assert f"staged_sha256={sha256_path(staged_plan_path)}" in message
    assert "staged_pipeline_names=['corrupt_staged_pipeline']" in message
