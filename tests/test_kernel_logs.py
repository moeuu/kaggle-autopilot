from __future__ import annotations

import json
from pathlib import Path

from kagglebot.kernel_logs import (
    KernelLogState,
    collect_log_tail,
    collect_log_tail_from_text,
    detect_failure_in_logs,
    format_log_events,
    parse_json_log,
    print_kernel_logs,
)


def test_collect_log_tail_prioritizes_traceback(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "output.log").write_text("start\nError: earlier\n", encoding="utf-8")
    traceback_log = output_dir / "kernel.log"
    traceback_log.write_text("line 1\nTraceback (most recent call last)\nValueError: bad\n", encoding="utf-8")

    tail = collect_log_tail(output_dir, max_lines=5)

    assert tail is not None
    assert tail.startswith("kernel.log")
    assert "ValueError: bad" in tail


def test_collect_log_tail_from_json_log() -> None:
    text = json.dumps(
        [
            {"stream_name": "stdout", "data": "setup\n"},
            {"stream_name": "stderr", "data": "Traceback\nRuntimeError: bad\n"},
        ]
    )

    tail = collect_log_tail_from_text(Path("kernel.json.log"), text, max_lines=5)

    assert tail == "kernel.json.log\n[stderr] RuntimeError: bad"


def test_parse_json_log_supports_logs_object() -> None:
    payload = {"logs": [{"stream_name": "stdout", "data": "ok\n"}, "skip"]}
    events = parse_json_log(json.dumps(payload))

    assert events == [{"stream_name": "stdout", "data": "ok\n"}]
    assert format_log_events(events or []) == ["[stdout] ok"]


def test_detect_failure_in_logs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "stderr.txt").write_text(
        "Traceback (most recent call last)\nRuntimeError: failed\n",
        encoding="utf-8",
    )

    failure = detect_failure_in_logs(output_dir)

    assert failure is not None
    assert "RuntimeError: failed" in failure


def test_print_kernel_logs_only_prints_new_content(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    log = output_dir / "stdout.txt"
    log.write_text("first\n", encoding="utf-8")
    state = KernelLogState()

    assert print_kernel_logs(output_dir, state) is True
    assert "first" in capsys.readouterr().out
    assert print_kernel_logs(output_dir, state) is False

    log.write_text("first\nsecond\n", encoding="utf-8")
    assert print_kernel_logs(output_dir, state) is True
    assert "second" in capsys.readouterr().out
