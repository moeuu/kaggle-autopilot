from __future__ import annotations

import pytest

from kagglebot.submission_history import (
    build_previous_submission_history_payload,
    detect_online_regression_vs_submission_history,
    format_previous_submission_history_for_prompt,
)


def test_build_previous_submission_history_payload_selects_best_and_latest_for_minimize() -> None:
    payload = build_previous_submission_history_payload(
        rows=[
            {
                "description": "latest",
                "publicScore": "10.308",
                "status": "complete",
                "date": "2026-05-22 09:24:24",
            },
            {
                "description": "best",
                "publicScore": "9.600",
                "status": "complete",
                "date": "2026-05-17 03:08:20",
                "rank": "4/200",
            },
        ],
        direction="minimize",
        source="test",
    )

    assert payload["best_score"] == pytest.approx(9.600)
    assert payload["latest_score"] == pytest.approx(10.308)
    assert payload["scored_count"] == 2
    assert payload["best"] is not None
    assert "best" in str(payload["best"])


def test_previous_submission_history_prompt_and_regression_signal() -> None:
    history = {
        "direction": "minimize",
        "best_score": 9.600,
        "best": {"description": "best public", "score": 9.600},
        "recent": [{"submitted_at": "2026-05-22T09:24:24+00:00", "score": 10.308}],
    }

    signal = detect_online_regression_vs_submission_history(
        previous_best_online=10.271,
        current_online=10.308,
        direction="minimize",
        history=history,
    )

    assert signal is not None
    assert signal["previous_best_online"] == pytest.approx(9.600)
    prompt = format_previous_submission_history_for_prompt(history)
    assert "Best historical public score: 9.600000" in prompt
    assert "Do not call a new iteration improved" in prompt
