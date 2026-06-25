from __future__ import annotations

from pathlib import Path

from kagglebot.agent_io import (
    agent_failure_detail,
    append_fix_retry_feedback,
    is_agent_capacity_failure,
    read_agent_response,
    tail_for_prompt,
)


def test_read_agent_response_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_agent_response(tmp_path / "missing.txt") == ""


def test_read_agent_response_strips_trailing_whitespace(tmp_path: Path) -> None:
    response_path = tmp_path / "response.txt"
    response_path.write_text("answer\n\n", encoding="utf-8")

    assert read_agent_response(response_path) == "answer"


def test_tail_for_prompt_normalizes_carriage_returns_and_keeps_tail() -> None:
    assert tail_for_prompt(" a\rb\nc ", max_chars=3) == "b\nc"


def test_agent_capacity_failure_detection_and_detail() -> None:
    class DummyResult:
        returncode = 1
        stdout = '{"type":"error","message":"Selected model is at capacity. Please try a different model."}'
        stderr = "stderr text"

    assert is_agent_capacity_failure(DummyResult(), "stale success text")
    detail = agent_failure_detail(DummyResult(), "stale success text")
    assert "returncode=1" in detail
    assert "stderr=stderr text" in detail
    assert "response=stale success text" in detail
    assert "transcript_tail=" in detail


def test_append_fix_retry_feedback_adds_clipped_failure_context() -> None:
    prompt = append_fix_retry_feedback(
        base_prompt="base",
        stage_label="kernel fix",
        codex_pass=2,
        failure_text="x" * 7000,
    )

    assert prompt.startswith("base\n\n## Retry Feedback (pass 2)")
    assert "previous kernel fix pass" in prompt
    assert "x" * 6000 in prompt
    assert "x" * 6001 not in prompt


def test_append_fix_retry_feedback_returns_base_when_failure_empty() -> None:
    assert (
        append_fix_retry_feedback(
            base_prompt="base",
            stage_label="kernel fix",
            codex_pass=2,
            failure_text="",
        )
        == "base"
    )
