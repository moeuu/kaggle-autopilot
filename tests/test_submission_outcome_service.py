from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kagglebot.submission.outcome_service import SubmissionOutcomePollingError, SubmissionOutcomeService


def test_submission_outcome_service_extracts_rank_pair_from_row() -> None:
    rows = [
        {
            "description": "demo",
            "status": "complete",
            "publicScore": "0.9532",
            "rank": "1300 / 2700",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=1, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("rank") == 1300
    assert outcome.get("total_teams") == 2700
    assert outcome.get("rank_percentile") == 1300 / 2700
    assert outcome.get("rank_source") == "submission_row"


def test_submission_outcome_service_extracts_rank_and_total_keys() -> None:
    rows = [
        {
            "description": "demo",
            "status": "complete",
            "publicScore": "0.9532",
            "publicLeaderboardRank": "88",
            "publicLeaderboardTotalTeams": "2700",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=1, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("rank") == 88
    assert outcome.get("total_teams") == 2700
    assert outcome.get("rank_percentile") == 88 / 2700


def test_submission_outcome_service_handles_prefixed_error_status_as_terminal() -> None:
    rows = [
        {
            "description": "demo",
            "status": "SubmissionStatus.ERROR",
            "publicScore": "",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=3, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("status") == "error"
    assert outcome.get("score") is None


def test_submission_outcome_service_raises_after_consecutive_fetch_errors() -> None:
    calls = {"count": 0}

    def failing_fetch(_slug: str) -> list[dict[str, str]]:
        calls["count"] += 1
        raise RuntimeError("kaggle api unreachable")

    service = SubmissionOutcomeService(
        fetch_rows=failing_fetch,
        max_attempts=None,
        poll_interval_sec=0.0,
        max_fetch_errors=2,
    )

    with pytest.raises(SubmissionOutcomePollingError) as exc_info:
        service.wait_for_outcome(
            slug="demo",
            message="demo",
            submitted_at=datetime.now(UTC),
        )

    assert calls["count"] == 2
    assert exc_info.value.consecutive_errors == 2
