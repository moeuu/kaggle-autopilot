from __future__ import annotations

from pathlib import Path

from kagglebot.agent_io import (
    agent_failure_detail,
    append_fix_retry_feedback,
    is_agent_capacity_failure,
    log_codex_sandbox_fallback,
    read_agent_response,
    tail_for_prompt,
    write_agent_prompt,
    write_autofix_error_transcript,
)


def test_read_agent_response_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_agent_response(tmp_path / "missing.txt") == ""


def test_write_agent_prompt_writes_text_and_returns_path(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"

    returned = write_agent_prompt(prompt_path, "hello\n")

    assert returned == prompt_path
    assert prompt_path.read_text(encoding="utf-8") == "hello\n"


def test_write_autofix_error_transcript_writes_attempt_and_latest_alias(tmp_path: Path) -> None:
    error_path = write_autofix_error_transcript(
        autofix_dir=tmp_path,
        attempt=3,
        error_text="boom",
    )

    expected = "autofix_attempt: 3\nboom\n"
    assert error_path == tmp_path / "error-03.txt"
    assert error_path.read_text(encoding="utf-8") == expected
    assert (tmp_path / "error.txt").read_text(encoding="utf-8") == expected


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


def test_log_codex_sandbox_fallback_prints_only_when_used(capsys) -> None:  # noqa: ANN001
    class UnusedResult:
        used_sandbox_fallback = False

    class UsedResult:
        used_sandbox_fallback = True
        sandbox_failure_excerpt = "sandbox denied"

    log_codex_sandbox_fallback(stage_label="kernel fix", result=UnusedResult())
    assert capsys.readouterr().out == ""

    log_codex_sandbox_fallback(stage_label="kernel fix", result=UsedResult())
    out = capsys.readouterr().out
    assert "kernel fix" in out
    assert "codex sandbox fallback used: sandbox denied" in out


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
