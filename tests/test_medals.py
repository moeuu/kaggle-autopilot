from __future__ import annotations

import pytest

from kagglebot.medals import (
    DEFAULT_TARGET_MEDAL,
    TARGET_RANK_PERCENTILE_NUMBER_ERROR,
    TARGET_RANK_PERCENTILE_RANGE_ERROR,
    normalize_target_medal,
    normalize_target_rank_percentile,
    validate_target_rank_percentile,
)


def test_medal_defaults_to_winner_rank_band() -> None:
    assert DEFAULT_TARGET_MEDAL == "winner"
    assert normalize_target_medal("WINNER") == "winner"
    assert normalize_target_rank_percentile(None, medal="winner") == pytest.approx(0.001)


def test_rank_percentile_parser_rejects_invalid_values_with_clear_errors() -> None:
    assert validate_target_rank_percentile("not-a-number", medal=None) == (
        None,
        TARGET_RANK_PERCENTILE_NUMBER_ERROR,
    )
    assert validate_target_rank_percentile(2.0, medal=None) == (None, TARGET_RANK_PERCENTILE_RANGE_ERROR)
