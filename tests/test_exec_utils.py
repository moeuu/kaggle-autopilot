from __future__ import annotations

import sys
import time

from kagglebot import exec_utils
from kagglebot.exec_utils import run_command


def test_run_command_stream_output_does_not_wait_for_inherited_stdout_holders(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(exec_utils, "_STREAM_OUTPUT_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(exec_utils, "_STREAM_OUTPUT_EXIT_PIPE_DRAIN_SEC", 0.05)
    started = time.monotonic()
    result = run_command(
        [
            sys.executable,
            "-c",
            (
                "import subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)']); "
                "print('streamed parent exited', flush=True)"
            ),
        ],
        stream_output=True,
        timeout=2,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0
    assert "streamed parent exited" in result.stdout
    assert elapsed < 1
