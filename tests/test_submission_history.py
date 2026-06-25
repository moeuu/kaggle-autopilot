from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.submission_history import (
    build_previous_submission_history_payload,
    detect_online_regression_vs_submission_history,
    format_previous_submission_history_for_prompt,
    load_previous_submission_history,
)


def test_build_previous_submission_history_payload_selects_best_and_latest_for_minimize() -> None:
    payload = build_previous_submission_history_payload(
        rows=[
            {
                "description": "invalid",
                "publicScore": "nan",
                "status": "complete",
                "date": "2026-05-23 09:24:24",
            },
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
    assert payload["count"] == 3
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


def test_load_previous_submission_history_uses_best_public_score(tmp_path: Path) -> None:
    history_path = tmp_path / "context" / "submission_history.json"

    def fake_submissions(slug: str) -> list[dict[str, str]]:
        assert slug == "rogii-demo"
        return [
            {
                "description": "kb run i=2 offline=10.7205",
                "publicScore": "10.308",
                "status": "complete",
                "date": "2026-05-22 09:24:24",
            },
            {
                "description": "kb previous i=5 offline=10.4211",
                "publicScore": "9.600",
                "status": "complete",
                "date": "2026-05-17 03:08:20",
            },
            {
                "description": "kb previous i=1 offline=10.6129",
                "publicScore": "11.504",
                "status": "complete",
                "date": "2026-05-17 02:08:20",
            },
        ]

    history = load_previous_submission_history(
        slug="rogii-demo",
        history_path=history_path,
        direction="minimize",
        dry_run=False,
        fetch_submission_rows=fake_submissions,
    )

    assert history["best_score"] == pytest.approx(9.600)
    assert history["latest_score"] == pytest.approx(10.308)
    assert history["scored_count"] == 3
    assert "previous i=5" in str(history["best"])
    saved = json.loads(history_path.read_text(encoding="utf-8"))
    assert saved["best_score"] == pytest.approx(9.600)


def test_load_previous_submission_history_uses_cache_on_fetch_error(tmp_path: Path) -> None:
    history_path = tmp_path / "context" / "submission_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps({"source": "previous", "best_score": 0.82}),
        encoding="utf-8",
    )
    messages: list[str] = []

    def fail_fetch(slug: str) -> list[dict[str, str]]:
        raise RuntimeError(f"boom {slug}")

    history = load_previous_submission_history(
        slug="demo",
        history_path=history_path,
        direction="maximize",
        dry_run=False,
        fetch_submission_rows=fail_fetch,
        on_message=messages.append,
    )

    assert history["source"] == "previous"
    assert history["best_score"] == pytest.approx(0.82)
    assert history["fetch_error"] == "RuntimeError: boom demo"
    assert messages
