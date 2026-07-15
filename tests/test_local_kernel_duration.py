from __future__ import annotations

from pathlib import Path

from kagglebot.local_kernel_duration import (
    append_local_kernel_duration_history,
    estimate_local_kernel_duration_seconds,
    exact_source_exceeds_timeout,
    exact_source_timeout_count,
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


def test_local_kernel_duration_estimate_only_uses_exact_source_fingerprint(tmp_path: Path) -> None:
    append_local_kernel_duration_history(
        base_dir=tmp_path,
        slug="demo",
        run_id="run-old",
        iteration=1,
        duration_sec=30.0,
        kernel_fingerprint="old-source",
    )
    append_local_kernel_duration_history(
        base_dir=tmp_path,
        slug="demo",
        run_id="run-new",
        iteration=2,
        duration_sec=900.0,
        kernel_fingerprint="new-source",
    )

    estimate, samples = estimate_local_kernel_duration_seconds(
        base_dir=tmp_path,
        slug="demo",
        kernel_fingerprint="new-source",
    )

    assert samples == 1
    assert estimate == 900.0


def test_runtime_preflight_only_rejects_repeated_exact_source_overruns() -> None:
    assert not exact_source_exceeds_timeout(
        estimated_duration_sec=90_000.0,
        sample_count=1,
        timeout_sec=86_400,
    )
    assert exact_source_exceeds_timeout(
        estimated_duration_sec=90_000.0,
        sample_count=2,
        timeout_sec=86_400,
    )
    assert not exact_source_exceeds_timeout(
        estimated_duration_sec=80_000.0,
        sample_count=2,
        timeout_sec=86_400,
    )
    assert exact_source_exceeds_timeout(
        estimated_duration_sec=None,
        sample_count=0,
        timeout_sec=86_400,
        timeout_count=2,
    )


def test_timeout_history_is_separate_from_completed_eta_history(tmp_path: Path) -> None:
    for iteration in (1, 2):
        append_local_kernel_duration_history(
            base_dir=tmp_path,
            slug="demo",
            run_id="run-timeout",
            iteration=iteration,
            duration_sec=86_400.0,
            kernel_fingerprint="same-source",
            outcome="timeout",
        )

    estimate, samples = estimate_local_kernel_duration_seconds(
        base_dir=tmp_path,
        slug="demo",
        kernel_fingerprint="same-source",
    )

    assert estimate is None
    assert samples == 0
    assert (
        exact_source_timeout_count(
            base_dir=tmp_path,
            slug="demo",
            kernel_fingerprint="same-source",
        )
        == 2
    )
