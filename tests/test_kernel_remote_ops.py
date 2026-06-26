from __future__ import annotations

from pathlib import Path

from kagglebot.exceptions import KaggleCliError
from kagglebot.kernel_remote_ops import (
    KernelRegistrationDependencies,
    clear_stale_kernel_output,
    resolve_kernel_id,
    try_fetch_kernel_output,
    wait_for_kernel_registration,
    write_push_log,
)


def test_wait_for_kernel_registration_resolves_by_title_after_status_miss() -> None:
    calls: list[tuple[str, str]] = []

    def kernels_status(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(("status", args[0]))
        raise KaggleCliError("not found", output="not found")

    def kernel_exists(kernel_id: str) -> bool:
        calls.append(("exists", kernel_id))
        return False

    def kernel_id_by_title(kernel_slug: str) -> str | None:
        calls.append(("title", kernel_slug))
        return "user/resolved"

    result = wait_for_kernel_registration(
        "user/original",
        "kernel-slug",
        deps=KernelRegistrationDependencies(
            kernels_status=kernels_status,
            kernel_exists=kernel_exists,
            kernel_id_by_title=kernel_id_by_title,
            sleep=lambda _seconds: None,
        ),
        retries=3,
        sleep_interval=0.0,
    )

    assert result == "user/resolved"
    assert calls == [
        ("status", "user/original"),
        ("exists", "user/original"),
        ("title", "kernel-slug"),
    ]


def test_resolve_kernel_id_keeps_original_on_cli_error() -> None:
    def kernel_id_by_title(_kernel_slug: str) -> str | None:
        raise KaggleCliError("boom")

    assert (
        resolve_kernel_id(
            "user/original",
            "kernel-slug",
            kernel_id_by_title_func=kernel_id_by_title,
        )
        == "user/original"
    )


def test_write_push_log_and_clear_stale_kernel_output(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    write_push_log(logs_dir, 2, " pushed \n")

    assert (logs_dir / "kernel_push-02.txt").read_text(encoding="utf-8") == "pushed\n"

    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "old.txt").write_text("old", encoding="utf-8")
    (output_dir / "submission.csv").write_text("id,target\n", encoding="utf-8")

    clear_stale_kernel_output(output_dir)

    assert list(output_dir.iterdir()) == []


def test_try_fetch_kernel_output_swallows_kaggle_cli_error(tmp_path: Path) -> None:
    calls: list[str] = []

    def kernels_output(kernel_id: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        calls.append(kernel_id)
        raise KaggleCliError("transient")

    try_fetch_kernel_output(
        "user/kernel",
        output_dir=tmp_path,
        slug="demo",
        kernels_output_func=kernels_output,
    )

    assert calls == ["user/kernel"]
