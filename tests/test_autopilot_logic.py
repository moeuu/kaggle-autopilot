"""Tests for autopilot logic functions (met_target, improvement tracking)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from kagglebot.agent_io import TrainingLiveStdout
from kagglebot.leaderboard_policy import (
    resume_best_online_submission_score,
    should_force_major_overhaul_by_rank,
)
from kagglebot.paths import CompetitionPaths
from kagglebot.plan_policy import write_plan_config as _write_plan
from kagglebot.runtime_fixes import infer_column_mapping
from kagglebot.score_progress import effective_best_score_for_progress
from kagglebot.score_utils import should_update_best_score as _update_best_score
from kagglebot.submission_policy import is_top1_tier, meets_target
from kagglebot.types import PlanConfig


class TestMeetsTarget:
    """Test the shared meets_target function for direction-aware comparison."""

    def test_minimize_target_met(self) -> None:
        """For minimize metrics, value <= target should return True."""
        assert meets_target(value=0.4, target=0.5, direction="minimize") is True
        assert meets_target(value=0.5, target=0.5, direction="minimize") is True  # Equal counts as met

    def test_minimize_target_not_met(self) -> None:
        """For minimize metrics, value > target should return False."""
        assert meets_target(value=0.6, target=0.5, direction="minimize") is False
        assert meets_target(value=1.0, target=0.5, direction="minimize") is False

    def test_maximize_target_met(self) -> None:
        """For maximize metrics, value >= target should return True."""
        assert meets_target(value=0.9, target=0.8, direction="maximize") is True
        assert meets_target(value=0.8, target=0.8, direction="maximize") is True  # Equal counts as met

    def test_maximize_target_not_met(self) -> None:
        """For maximize metrics, value < target should return False."""
        assert meets_target(value=0.7, target=0.8, direction="maximize") is False
        assert meets_target(value=0.5, target=0.8, direction="maximize") is False

    def test_edge_cases(self) -> None:
        """Test edge cases like very small/large values."""
        # Very small improvements should still count
        assert meets_target(value=0.50001, target=0.5, direction="minimize") is False
        assert meets_target(value=0.49999, target=0.5, direction="minimize") is True
        assert meets_target(value=0.80001, target=0.8, direction="maximize") is True
        assert meets_target(value=0.79999, target=0.8, direction="maximize") is False

        # Negative values
        assert meets_target(value=-0.1, target=0.0, direction="minimize") is True
        assert meets_target(value=-0.1, target=-0.2, direction="maximize") is True

        # Very large values
        assert meets_target(value=1e6, target=1e5, direction="minimize") is False
        assert meets_target(value=1e6, target=1e5, direction="maximize") is True


class TestUpdateBestScore:
    """Test the _update_best_score function for improvement tracking."""

    def test_first_iteration_always_improves(self) -> None:
        """First iteration (best=None) should always count as improvement."""
        assert _update_best_score(best=None, current=0.5, direction="minimize", min_improvement=0.0) is True
        assert _update_best_score(best=None, current=0.5, direction="maximize", min_improvement=0.0) is True
        assert _update_best_score(best=None, current=0.5, direction="minimize", min_improvement=0.1) is True

    def test_minimize_improvement(self) -> None:
        """For minimize, lower score is better."""
        # Improvement with min_improvement=0.0 (any improvement counts)
        assert _update_best_score(best=0.5, current=0.4, direction="minimize", min_improvement=0.0) is True
        assert _update_best_score(best=0.5, current=0.3, direction="minimize", min_improvement=0.0) is True

        # Equal score counts as "not worse" with min_improvement=0.0 (prevents patience increment)
        assert _update_best_score(best=0.5, current=0.5, direction="minimize", min_improvement=0.0) is True

        # Worse score is not improvement
        assert _update_best_score(best=0.5, current=0.6, direction="minimize", min_improvement=0.0) is False

        # With min_improvement threshold
        assert _update_best_score(best=0.5, current=0.4, direction="minimize", min_improvement=0.1) is True
        assert _update_best_score(best=0.5, current=0.41, direction="minimize", min_improvement=0.1) is False
        assert _update_best_score(best=0.5, current=0.39, direction="minimize", min_improvement=0.1) is True

    def test_maximize_improvement(self) -> None:
        """For maximize, higher score is better."""
        # Improvement with min_improvement=0.0 (any improvement counts)
        assert _update_best_score(best=0.8, current=0.9, direction="maximize", min_improvement=0.0) is True
        assert _update_best_score(best=0.8, current=0.85, direction="maximize", min_improvement=0.0) is True

        # Equal score counts as "not worse" with min_improvement=0.0 (prevents patience increment)
        assert _update_best_score(best=0.8, current=0.8, direction="maximize", min_improvement=0.0) is True

        # Worse score is not improvement
        assert _update_best_score(best=0.8, current=0.7, direction="maximize", min_improvement=0.0) is False

        # With min_improvement threshold
        assert _update_best_score(best=0.8, current=0.9, direction="maximize", min_improvement=0.1) is True
        assert _update_best_score(best=0.8, current=0.89, direction="maximize", min_improvement=0.1) is False
        assert _update_best_score(best=0.8, current=0.91, direction="maximize", min_improvement=0.1) is True

    def test_edge_cases_improvement(self) -> None:
        """Test edge cases for improvement tracking."""
        # Exact threshold boundary (use values that avoid floating point precision issues)
        # For minimize: improvement = best - current = 1.0 - 0.9 = 0.1 >= 0.1 → True
        assert _update_best_score(best=1.0, current=0.9, direction="minimize", min_improvement=0.1) is True
        # For maximize: improvement = current - best = 0.9 - 0.8 = 0.1 >= 0.1 → True
        assert _update_best_score(best=0.8, current=0.9, direction="maximize", min_improvement=0.1) is True

        # Very small improvements
        assert _update_best_score(best=0.5, current=0.499999, direction="minimize", min_improvement=0.0) is True
        assert _update_best_score(best=0.5, current=0.500001, direction="maximize", min_improvement=0.0) is True

        # Negative values
        assert _update_best_score(best=-0.1, current=-0.2, direction="minimize", min_improvement=0.0) is True
        assert _update_best_score(best=-0.2, current=-0.1, direction="maximize", min_improvement=0.0) is True

        # Test exact boundary with clearer values
        # Just meeting threshold should count
        assert _update_best_score(best=0.5, current=0.4, direction="minimize", min_improvement=0.1) is True
        assert _update_best_score(best=0.4, current=0.5, direction="maximize", min_improvement=0.1) is True


def test_resume_best_online_submission_score_ignores_missing_invalid_and_non_object_metrics(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_id = "run-1"

    iter_1 = paths.iter_dir(run_id, 1)
    iter_1.mkdir(parents=True)
    (iter_1 / "metrics.json").write_text(json.dumps({"submission_score": "0.72"}), encoding="utf-8")

    iter_2 = paths.iter_dir(run_id, 2)
    iter_2.mkdir(parents=True)
    (iter_2 / "metrics.json").write_text("{", encoding="utf-8")

    iter_3 = paths.iter_dir(run_id, 3)
    iter_3.mkdir(parents=True)
    (iter_3 / "metrics.json").write_text("[]", encoding="utf-8")

    iter_4 = paths.iter_dir(run_id, 4)
    iter_4.mkdir(parents=True)
    (iter_4 / "metrics.json").write_text(json.dumps({"submission_score": "0.81"}), encoding="utf-8")

    assert (
        resume_best_online_submission_score(
            paths=paths,
            run_id=run_id,
            direction="maximize",
            max_iterations=5,
        )
        == 0.81
    )


class TestPatienceLogic:
    """Integration tests for patience/early stopping logic."""

    def test_patience_stops_after_n_no_improvements(self) -> None:
        """Verify patience counter works correctly."""
        # Simulate iterations with no improvement
        best = 0.5
        patience_counter = 0
        patience = 2
        min_improvement = 0.01

        # Iteration 1: no improvement (0.52 worse than 0.5 for minimize)
        current = 0.52
        if _update_best_score(best, current, "minimize", min_improvement):
            best = current
            patience_counter = 0
        else:
            patience_counter += 1
        assert patience_counter == 1

        # Iteration 2: no improvement (0.51 only 0.01 better, threshold is 0.01, so marginal)
        current = 0.51
        if _update_best_score(best, current, "minimize", min_improvement):
            best = current
            patience_counter = 0
        else:
            patience_counter += 1
        assert patience_counter == 2

        # Should stop now (patience_counter >= patience)
        assert patience_counter >= patience

    def test_patience_resets_on_improvement(self) -> None:
        """Verify patience counter resets when improvement happens."""
        best = 0.5
        patience_counter = 1
        min_improvement = 0.0

        # Improvement happens (0.45 < 0.5 for minimize)
        current = 0.45
        if _update_best_score(best, current, "minimize", min_improvement):
            best = current
            patience_counter = 0
        else:
            patience_counter += 1

        assert patience_counter == 0
        assert best == 0.45


class TestDirectionInference:
    """Test that direction is correctly inferred and used."""

    @pytest.mark.parametrize(
        "metric,expected_direction",
        [
            ("rmse", "minimize"),
            ("mae", "minimize"),
            ("mse", "minimize"),
            ("logloss", "minimize"),
            ("accuracy", "maximize"),
            ("auc_roc", "maximize"),
            ("f1", "maximize"),
            ("r2", "maximize"),
        ],
    )
    def test_metric_direction_mapping(self, metric: str, expected_direction: str) -> None:
        """Verify common metrics have correct direction."""
        from kagglebot.solver.metrics import infer_direction

        assert infer_direction(metric) == expected_direction


class TestTop1Tier:
    """Test the top1-tier heuristic."""

    def test_top1_none(self) -> None:
        assert is_top1_tier(0.5, None, "minimize") is False

    def test_top1_minimize(self) -> None:
        assert is_top1_tier(0.4, 0.5, "minimize") is True
        assert is_top1_tier(0.6, 0.5, "minimize") is False

    def test_top1_maximize(self) -> None:
        assert is_top1_tier(0.9, 0.8, "maximize") is True
        assert is_top1_tier(0.7, 0.8, "maximize") is False


class TestRankDrivenMajorOverhaul:
    def test_rank_policy_forces_major_overhaul_when_percentile_is_poor(self) -> None:
        assert (
            should_force_major_overhaul_by_rank(rank=1300, total_teams=2700, max_percentile=0.35, min_teams=200) is True
        )

    def test_rank_policy_does_not_force_for_good_rank(self) -> None:
        assert (
            should_force_major_overhaul_by_rank(rank=200, total_teams=2700, max_percentile=0.35, min_teams=200) is False
        )

    def test_rank_policy_does_not_force_for_small_competition(self) -> None:
        assert (
            should_force_major_overhaul_by_rank(rank=50, total_teams=120, max_percentile=0.35, min_teams=200) is False
        )


class TestBestScoreOutlierGuard:
    def test_clips_implausible_previous_best_for_maximize_metric(self) -> None:
        effective_best, guard = effective_best_score_for_progress(
            prev_best=0.999511,
            current_score=0.799651,
            top1_score=0.78,
            direction="maximize",
        )
        assert effective_best is not None
        assert effective_best < 0.999511
        assert guard is not None
        assert guard["reason"] == "clip_prev_best_above_top1_band"

    def test_does_not_clip_when_current_score_is_still_in_outlier_band(self) -> None:
        effective_best, guard = effective_best_score_for_progress(
            prev_best=0.999511,
            current_score=0.98,
            top1_score=0.78,
            direction="maximize",
        )
        assert effective_best == 0.999511
        assert guard is None


def test_infer_column_mapping_handles_non_string_group_tokens() -> None:
    columns_by_file = {"train.csv": ["session_id", "target"]}
    groups = [["session_id", None, 123], ["target", object()]]
    mapping = infer_column_mapping(columns_by_file, groups)
    assert mapping["session_id"] == "session_id"
    assert mapping["target"] == "target"


def test_training_live_stdout_separates_live_line_and_logs() -> None:
    buf = io.StringIO()
    stream = TrainingLiveStdout(buf)

    stream.render_live("Training: local_gpu 10s")
    stream.write("[local train] candidate=logreg source=holdout training\n")
    stream.finish_live("Training: local_gpu 11s")

    rendered = buf.getvalue()
    assert "\n[local train] candidate=logreg source=holdout training\n\rTraining: local_gpu 10s" in rendered
    assert rendered.endswith("\rTraining: local_gpu 11s\n")


def test_training_live_stdout_keeps_regular_output_without_live_line() -> None:
    buf = io.StringIO()
    stream = TrainingLiveStdout(buf)
    stream.write("plain log line\n")
    assert buf.getvalue() == "plain log line\n"


def test_write_plan_preserves_extended_strategy_fields(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(
        json.dumps(
            {
                "target_metric": "auc",
                "target_direction": "maximize",
                "target_score": 0.9,
                "pipelines": [{"name": "p1"}],
                "toggles": {"USE_MODEL": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_plan(
        paths,
        PlanConfig(
            target_metric="auc",
            target_direction="maximize",
            target_score=0.92,
            score_source="cv",
            max_iterations=3,
        ),
    )
    payload = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    assert payload["target_score"] == 0.92
    assert payload["pipelines"] == [{"name": "p1"}]
    assert payload["toggles"]["USE_MODEL"] is True
    assert payload["toggles"]["ENABLE_TRAINING"] is True
    assert payload["toggles"]["RUN_VALIDATION_GENERATION"] is True
    assert payload["toggles"]["PACKAGING_ONLY"] is False
