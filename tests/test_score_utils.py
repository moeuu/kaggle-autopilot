from __future__ import annotations

import pytest

from kagglebot.score_utils import score_gap, should_update_best_score


def test_should_update_best_score_matches_direction_and_thresholds() -> None:
    assert should_update_best_score(None, 0.5, "minimize", 0.1)
    assert should_update_best_score(0.5, 0.4, "minimize", 0.1)
    assert not should_update_best_score(0.5, 0.41, "minimize", 0.1)
    assert should_update_best_score(0.8, 0.9, "maximize", 0.1)
    assert not should_update_best_score(0.8, 0.89, "maximize", 0.1)


def test_should_update_best_score_treats_ties_as_updates_without_min_delta() -> None:
    assert should_update_best_score(0.5, 0.5, "minimize", 0.0)
    assert should_update_best_score(0.5, 0.5, "maximize", 0.0)


def test_score_gap_is_positive_when_current_is_better_than_reference() -> None:
    assert score_gap(current=0.4, reference=0.5, direction="minimize") == pytest.approx(0.1)
    assert score_gap(current=0.6, reference=0.5, direction="minimize") == pytest.approx(-0.1)
    assert score_gap(current=0.9, reference=0.8, direction="maximize") == pytest.approx(0.1)
    assert score_gap(current=0.7, reference=0.8, direction="maximize") == pytest.approx(-0.1)
    assert score_gap(current="bad", reference=0.8, direction="maximize") is None
