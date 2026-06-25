from __future__ import annotations

from kagglebot.loop_control import decide_stagnation_stop


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
