"""Tests for competition URL parsing."""

from __future__ import annotations

import pytest

from kagglebot.competition import parse_competition_slug


@pytest.mark.parametrize(
    "value,expected",
    [
        ("titanic", "titanic"),
        ("https://www.kaggle.com/competitions/titanic", "titanic"),
        ("https://www.kaggle.com/competitions/titanic/overview", "titanic"),
        ("https://www.kaggle.com/c/titanic", "titanic"),
        ("https://www.kaggle.com/c/titanic/", "titanic"),
        ("www.kaggle.com/competitions/titanic", "titanic"),
        ("WiDSWorldWide_GlobalDathon26", "widsworldwide_globaldathon26"),
        (
            "https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26",
            "widsworldwide_globaldathon26",
        ),
    ],
)
def test_parse_competition_slug(value: str, expected: str) -> None:
    assert parse_competition_slug(value) == expected


def test_parse_competition_slug_invalid_domain() -> None:
    with pytest.raises(ValueError, match="kaggle.com"):
        parse_competition_slug("https://example.com/competitions/titanic")


@pytest.mark.parametrize("value", ["", "   ", "not/a/slug"])
def test_parse_competition_slug_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_competition_slug(value)
