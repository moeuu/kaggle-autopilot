from __future__ import annotations

import json
from pathlib import Path

from kagglebot.experiment_executor import GRAPH_EXECUTION_REPORT_FILENAME, execute_experiment_graph
from kagglebot.paths import CompetitionPaths
from kagglebot.runners.base import RunContext
from kagglebot.runners.local_kernel import LocalKernelRunner


def _context(tmp_path: Path, *, dry_run: bool = False) -> RunContext:
    return RunContext(
        competition="demo",
        slug="demo",
        run_id="run-1",
        paths=CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        workdir=tmp_path,
        dry_run=dry_run,
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
        max_candidates_per_iteration=1,
    )


def test_execute_experiment_graph_runs_ready_node_and_updates_graph(tmp_path: Path) -> None:
    context = _context(tmp_path)
    iter_dir = context.paths.iter_dir(context.run_id, 1)
    graph = {
        "version": 1,
        "run_id": context.run_id,
        "iteration": 1,
        "mode": "serial",
        "campaign_id": "demo-run-1",
        "nodes": [
            {
                "node_id": "model_candidate:strong_single",
                "node_type": "model_candidate",
                "candidate_id": "candidate-a",
                "category": "strong_single",
                "status": "ready",
                "method_id": "gbdt",
                "validation_profile_id": "default_cv",
                "expected_outputs": {
                    "oof": str(iter_dir / "candidate-a.oof.npy"),
                    "test_prediction": str(iter_dir / "candidate-a.test.npy"),
                },
                "metadata": {"adapter": "tabular_model_candidate"},
            }
        ],
    }

    result = execute_experiment_graph(
        graph=graph,
        context=context,
        runner=LocalKernelRunner(),
        iter_dir=iter_dir,
    )

    assert result.status == "completed"
    assert result.completed_nodes == ["model_candidate:strong_single"]
    report = json.loads((iter_dir / GRAPH_EXECUTION_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["candidate_results"][0]["candidate_id"] == "candidate-a"
    updated_graph = json.loads((iter_dir / "experiment_graph.json").read_text(encoding="utf-8"))
    assert updated_graph["nodes"][0]["status"] == "completed"
    assert updated_graph["nodes"][0]["evidence"]["decision"] == "adopted"
    assert (context.paths.run_dir(context.run_id) / "candidates" / "candidate-a" / "candidate_manifest.json").exists()


def test_execute_experiment_graph_parallel_runs_all_ready_nodes(tmp_path: Path) -> None:
    context = RunContext(
        competition="demo",
        slug="demo",
        run_id="run-1",
        paths=CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
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
    )
    graph = {
        "version": 1,
        "run_id": context.run_id,
        "iteration": 1,
        "mode": "parallel",
        "nodes": [
            {
                "node_id": "model_candidate:a",
                "node_type": "model_candidate",
                "candidate_id": "a",
                "category": "strong_single",
                "status": "ready",
                "expected_outputs": {},
            },
            {
                "node_id": "feature_probe:b",
                "node_type": "feature_probe",
                "candidate_id": "b",
                "category": "feature_variant",
                "status": "ready",
                "expected_outputs": {},
            },
        ],
    }

    result = execute_experiment_graph(
        graph=graph,
        context=context,
        runner=LocalKernelRunner(),
        iter_dir=context.paths.iter_dir(context.run_id, 1),
    )

    assert result.completed_nodes == ["model_candidate:a", "feature_probe:b"]


def test_execute_experiment_graph_budgeted_selects_highest_priority_ready_node(tmp_path: Path) -> None:
    context = _context(tmp_path)
    graph = {
        "version": 1,
        "run_id": context.run_id,
        "iteration": 1,
        "mode": "budgeted",
        "nodes": [
            {
                "node_id": "model_candidate:low",
                "node_type": "model_candidate",
                "candidate_id": "low",
                "category": "strong_single",
                "status": "ready",
                "expected_outputs": {},
                "runtime": {"execution_priority": 0.2},
            },
            {
                "node_id": "split_probe:high",
                "node_type": "split_probe",
                "candidate_id": "high",
                "category": "validation_variant",
                "status": "ready",
                "expected_outputs": {},
                "runtime": {"execution_priority": 0.9},
            },
        ],
    }

    result = execute_experiment_graph(
        graph=graph,
        context=context,
        runner=LocalKernelRunner(),
        iter_dir=context.paths.iter_dir(context.run_id, 1),
    )

    assert result.selected_nodes == ["split_probe:high"]
    assert result.completed_nodes == ["split_probe:high"]
    assert result.skipped_nodes == ["model_candidate:low"]
