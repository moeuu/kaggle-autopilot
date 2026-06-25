from __future__ import annotations

from pathlib import Path

from kagglebot.local_kernel_duration import (
    append_local_kernel_duration_history,
    estimate_local_kernel_duration_seconds,
)


def test_local_kernel_duration_history_estimate_uses_recent_median(tmp_path: Path) -> None:
    for idx, duration in enumerate([100.0, 120.0, 80.0, 110.0], start=1):
        append_local_kernel_duration_history(
            base_dir=tmp_path,
            slug="demo",
            run_id="run-a",
            iteration=idx,
            duration_sec=duration,
        )

    estimate, samples = estimate_local_kernel_duration_seconds(base_dir=tmp_path, slug="demo")
    assert samples == 4
    assert estimate == 105.0


def test_local_kernel_duration_history_ignores_invalid_rows(tmp_path: Path) -> None:
    history = tmp_path / "demo" / "context" / "local_kernel_duration_history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text(
        "\n".join(
            [
                "{",
                '{"duration_sec": 0}',
                '{"duration_sec": -1}',
                '{"duration_sec": "10"}',
                '{"duration_sec": 30}',
            ]
        ),
        encoding="utf-8",
    )

    estimate, samples = estimate_local_kernel_duration_seconds(base_dir=tmp_path, slug="demo")
    assert samples == 1
    assert estimate == 30.0
