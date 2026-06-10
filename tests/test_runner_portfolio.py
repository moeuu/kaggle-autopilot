from __future__ import annotations

import json
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.runners.base import CandidateRunSpec, RunContext
from kagglebot.runners.local_kernel import LocalKernelRunner


def test_local_runner_candidate_batch_writes_candidate_manifests(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    context = RunContext(
        competition="demo",
        slug="demo",
        run_id="run-1",
        paths=paths,
        workdir=tmp_path,
        dry_run=True,
        force=False,
        force_submit=False,
        message="test",
        time_budget_minutes=10,
        cv_folds=3,
        model_names=None,
        use_stacking=False,
        compute="local_gpu",
        accelerator="gpu",
        enable_internet=False,
        kaggle_username=None,
        strict_accelerator=False,
    )
    spec = CandidateRunSpec(
        candidate_id="candidate-a",
        node_id="model_candidate:strong_single",
        category="strong_single",
        method_id="gbdt",
        validation_profile_id="default_cv",
        expected_outputs={"oof": str(tmp_path / "oof.npy"), "test_prediction": str(tmp_path / "test.npy")},
    )

    results = LocalKernelRunner().run_candidate_batch(context, [spec])

    assert results[0].candidate_id == "candidate-a"
    assert results[0].status == "planned"
    manifest = json.loads(results[0].metrics_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert manifest["node_id"] == "model_candidate:strong_single"
    assert manifest["method_id"] == "gbdt"


def test_local_runner_non_dry_run_materializes_candidate_outputs(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    context = RunContext(
        competition="demo",
        slug="demo",
        run_id="run-1",
        paths=paths,
        workdir=tmp_path,
        dry_run=False,
        force=False,
        force_submit=False,
        message="test",
        time_budget_minutes=10,
        cv_folds=3,
        model_names=None,
        use_stacking=False,
        compute="local_gpu",
        accelerator="gpu",
        enable_internet=False,
        kaggle_username=None,
        strict_accelerator=False,
        candidate_budget_minutes=5,
    )
    spec = CandidateRunSpec(
        candidate_id="candidate-b",
        node_id="model_candidate:strong_single",
        category="strong_single",
        method_id="gbdt",
        validation_profile_id="default_cv",
        expected_outputs={
            "oof": str(tmp_path / "candidate-b.oof.npy"),
            "test_prediction": str(tmp_path / "candidate-b.test.npy"),
        },
        dependency_check={"required": ["numpy", "pandas", "scikit-learn"], "optional": ["missing_optional_pkg"]},
    )

    result = LocalKernelRunner().run_one_candidate(context, spec)

    assert result.status == "completed"
    assert result.oof_path is not None and result.oof_path.exists()
    assert result.prediction_path is not None and result.prediction_path.exists()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert metrics["dependency_check"]["status"] == "ok"
    assert "missing_optional_pkg" in metrics["dependency_check"]["optional_missing"]
