from __future__ import annotations

import os
import time
from pathlib import Path

from kagglebot.local_kernel_process import (
    LocalKernelLogFilterState,
    run_local_kernel_once,
    should_suppress_local_kernel_log_line,
)


def test_should_suppress_local_kernel_log_line_filters_fragmentation_and_catboost_noise() -> None:
    state = LocalKernelLogFilterState()
    lines = [
        "/tmp/kernel.py:1036: PerformanceWarning: DataFrame is highly fragmented.\n",
        "  out[ratio_col] = out[t1] / (out[t2].abs() + 1e-6)\n",
        "Default metric period is 5 because BrierScore is/are not implemented for GPU\n",
        "training fold=1\n",
    ]

    suppressed = [should_suppress_local_kernel_log_line(line, state=state) for line in lines]

    assert suppressed == [True, True, True, False]


def test_run_local_kernel_once_does_not_wait_for_inherited_stdout_holders(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        (
            "import subprocess\n"
            "import sys\n"
            "\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
            "print('kernel parent exited', flush=True)\n"
        ),
        encoding="utf-8",
    )

    started = time.monotonic()
    result = run_local_kernel_once(
        kernel_path=kernel_path,
        kernel_stage_dir=tmp_path,
        current_env=os.environ.copy(),
        timeout_sec=5,
        line_callback=None,
        progress_tracker=None,
    )
    elapsed = time.monotonic() - started

    assert result.command_result.returncode == 0
    assert result.command_result.args[1] == "-u"
    assert "kernel parent exited" in result.command_result.stdout
    assert elapsed < 5
