from __future__ import annotations

from kagglebot.agents.codex_runner import _format_command_for_log


def test_format_command_for_log_truncates_to_two_lines() -> None:
    command = '/bin/bash -lc "' + "verylongtoken " * 40 + '"'
    first, second = _format_command_for_log(command)

    assert first
    assert second
    assert second.endswith("...")


def test_format_command_for_log_keeps_short_command_on_one_line() -> None:
    first, second = _format_command_for_log("uv run pytest -q")

    assert first == "uv run pytest -q"
    assert second == ""
