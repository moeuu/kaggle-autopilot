from __future__ import annotations

import pytest

from kagglebot.score_utils import best_score, is_better_score, score_gap, should_update_best_score


def test_should_update_best_score_matches_direction_and_thresholds() -> None:
    assert should_update_best_score(None, 0.5, "minimize", 0.1)
    assert should_update_best_score(0.5, 0.4, "minimize", 0.1)
    assert not should_update_best_score(0.5, 0.41, "minimize", 0.1)
    assert should_update_best_score(0.8, 0.9, "maximize", 0.1)
    assert not should_update_best_score(0.8, 0.89, "maximize", 0.1)


def test_should_update_best_score_treats_ties_as_updates_without_min_delta() -> None:
    assert should_update_best_score(0.5, 0.5, "minimize", 0.0)
    assert should_update_best_score(0.5, 0.5, "maximize", 0.0)


def test_is_better_score_matches_direction_and_delta() -> None:
    assert is_better_score(0.4, 0.5, direction="minimize")
    assert not is_better_score(0.49, 0.5, direction="minimize", min_delta=0.02)
    assert is_better_score(0.9, 0.8, direction="maximize")
    assert not is_better_score(0.81, 0.8, direction="maximize", min_delta=0.02)
    assert not is_better_score("bad", 0.8, direction="maximize")


def test_best_score_filters_invalid_values_and_honors_direction() -> None:
    assert best_score(direction="minimize", scores=[0.5, "bad", 0.4, None]) == pytest.approx(0.4)
    assert best_score(direction="maximize", scores=[0.5, "bad", 0.4, None]) == pytest.approx(0.5)
    assert best_score(direction="minimize", scores=["bad", None]) is None


def test_score_gap_is_positive_when_current_is_better_than_reference() -> None:
    assert score_gap(current=0.4, reference=0.5, direction="minimize") == pytest.approx(0.1)
    assert score_gap(current=0.6, reference=0.5, direction="minimize") == pytest.approx(-0.1)
    assert score_gap(current=0.9, reference=0.8, direction="maximize") == pytest.approx(0.1)
    assert score_gap(current=0.7, reference=0.8, direction="maximize") == pytest.approx(-0.1)
    assert score_gap(current="bad", reference=0.8, direction="maximize") is None
