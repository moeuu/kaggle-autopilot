from __future__ import annotations

from datetime import UTC, datetime

from kagglebot.submission.outcome_service import SubmissionOutcomeService


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
