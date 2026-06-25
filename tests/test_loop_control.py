from __future__ import annotations

from kagglebot.loop_control import (
    append_policy_reason,
    decide_no_improve_major_overhaul,
    decide_stagnation_stop,
    decide_terminal_iteration_stop,
    select_stagnation_track,
    update_same_config_streak,
)


def test_update_same_config_streak_increments_matching_hash() -> None:
    state = update_same_config_streak(
        current_config_hash="abc",
        last_config_hash="abc",
        same_config_streak=2,
    )

    assert state.same_config_streak == 3
    assert state.last_config_hash == "abc"


def test_update_same_config_streak_resets_on_new_hash() -> None:
    state = update_same_config_streak(
        current_config_hash="new",
        last_config_hash="old",
        same_config_streak=2,
    )

    assert state.same_config_streak == 0
    assert state.last_config_hash == "new"


def test_select_stagnation_track_prefers_accuracy_frontier_after_candidate_exists() -> None:
    track = select_stagnation_track(
        best_high_potential_score=0.81,
        no_improve_streak=5,
        frontier_no_improve_streak=2,
    )

    assert track.no_improve_streak == 2
    assert track.label == "accuracy frontier"


def test_select_stagnation_track_uses_offline_metric_without_frontier_candidate() -> None:
    track = select_stagnation_track(
        best_high_potential_score=None,
        no_improve_streak=5,
        frontier_no_improve_streak=2,
    )

    assert track.no_improve_streak == 5
    assert track.label == "offline metric"


def test_decide_stagnation_stop_stops_on_no_improvement_patience() -> None:
    decision = decide_stagnation_stop(
        stop_allowed=True,
        no_improve_streak=3,
        no_improve_patience=2,
        stop_min_delta=0.001,
        track_label="offline metric",
        same_config_streak=0,
        same_config_patience=1,
    )

    assert decision.should_stop is True
    assert decision.reason == "offline metric did not improve by >= 0.001000 for 3 consecutive iterations"


def test_decide_stagnation_stop_stops_on_same_config_loop() -> None:
    decision = decide_stagnation_stop(
        stop_allowed=True,
        no_improve_streak=0,
        no_improve_patience=2,
        stop_min_delta=0.001,
        track_label="offline metric",
        same_config_streak=2,
        same_config_patience=2,
    )

    assert decision.should_stop is True
    assert decision.reason == "model/pipeline config hash unchanged for 2 consecutive iterations"


def test_decide_stagnation_stop_keeps_existing_submit_enabled_behavior() -> None:
    decision = decide_stagnation_stop(
        stop_allowed=False,
        no_improve_streak=3,
        no_improve_patience=2,
        stop_min_delta=0.001,
        track_label="accuracy frontier",
        same_config_streak=3,
        same_config_patience=2,
    )

    assert decision.should_stop is False
    assert decision.reason == ""


def test_decide_terminal_iteration_stop_prefers_confirmed_first_place() -> None:
    decision = decide_terminal_iteration_stop(
        confirmed_first_place=True,
        iteration=3,
        max_iterations=3,
        submitted=True,
    )

    assert decision.should_stop is True
    assert decision.status == "submitted"
    assert decision.stop_reason == "submission_rank_1"
    assert "rank reached #1" in decision.message


def test_decide_terminal_iteration_stop_handles_max_iterations_without_reason() -> None:
    decision = decide_terminal_iteration_stop(
        confirmed_first_place=False,
        iteration=3,
        max_iterations=3,
        submitted=False,
    )

    assert decision.should_stop is True
    assert decision.status == "completed"
    assert decision.stop_reason == ""
    assert decision.message == ""


def test_decide_terminal_iteration_stop_can_defer_max_iteration_check() -> None:
    decision = decide_terminal_iteration_stop(
        confirmed_first_place=False,
        iteration=3,
        max_iterations=3,
        submitted=False,
        allow_max_iteration_stop=False,
    )

    assert decision.should_stop is False


def test_decide_no_improve_major_overhaul_forces_when_enabled_and_not_improved() -> None:
    decision = decide_no_improve_major_overhaul(
        force_enabled=True,
        improved=False,
        high_potential_improved=False,
        best_score_guarded=False,
        metric_name="log_loss",
        current_score=1.2345678,
        previous_best_score=1.1,
    )

    assert decision.force_major_overhaul is True
    assert decision.reason == "Offline log_loss did not improve (current=1.234568, best=1.100000)."
    assert decision.skip_message == ""


def test_decide_no_improve_major_overhaul_skips_guarded_best_score() -> None:
    decision = decide_no_improve_major_overhaul(
        force_enabled=True,
        improved=False,
        high_potential_improved=False,
        best_score_guarded=True,
        metric_name="log_loss",
        current_score=1.2,
        previous_best_score=1.0,
    )

    assert decision.force_major_overhaul is False
    assert decision.reason == ""
    assert "previous best was clipped as an outlier" in decision.skip_message


def test_decide_no_improve_major_overhaul_noops_when_disabled_or_high_potential_improved() -> None:
    disabled = decide_no_improve_major_overhaul(
        force_enabled=False,
        improved=False,
        high_potential_improved=False,
        best_score_guarded=False,
        metric_name="auc",
        current_score=0.7,
        previous_best_score=None,
    )
    high_potential = decide_no_improve_major_overhaul(
        force_enabled=True,
        improved=False,
        high_potential_improved=True,
        best_score_guarded=False,
        metric_name="auc",
        current_score=0.7,
        previous_best_score=None,
    )

    assert disabled.force_major_overhaul is False
    assert high_potential.force_major_overhaul is False


def test_append_policy_reason_preserves_existing_reason_order() -> None:
    assert append_policy_reason(None, "new reason") == "new reason"
    assert append_policy_reason("existing reason.", "new reason.") == "existing reason. new reason."
    assert append_policy_reason("existing reason.", "") == "existing reason."
