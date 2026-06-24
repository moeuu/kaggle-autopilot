from __future__ import annotations

import hashlib
from pathlib import Path

from kagglebot.submit_retry_policy import (
    compute_submit_code_fingerprint,
    consume_same_submit_fingerprint_retry_allowance,
)


def _sha256_or_none(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compute_submit_code_fingerprint_includes_src_and_kernel_files(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    kernel_root = tmp_path / "kernel"
    src_root.mkdir()
    kernel_root.mkdir()
    (src_root / "module.py").write_text("print('src')\n", encoding="utf-8")
    (kernel_root / "kernel.py").write_text("print('kernel')\n", encoding="utf-8")

    original = compute_submit_code_fingerprint(
        src_root=src_root,
        kernel_source_dir=kernel_root,
        sha256_or_none=_sha256_or_none,
    )
    (kernel_root / "kernel.py").write_text("print('changed')\n", encoding="utf-8")
    changed = compute_submit_code_fingerprint(
        src_root=src_root,
        kernel_source_dir=kernel_root,
        sha256_or_none=_sha256_or_none,
    )

    assert len(original) == 64
    assert changed != original


def test_compute_submit_code_fingerprint_ignores_python_cache_files(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    kernel_root = tmp_path / "kernel"
    cache_dir = src_root / "__pycache__"
    cache_dir.mkdir(parents=True)
    kernel_root.mkdir()
    (src_root / "module.py").write_text("print('src')\n", encoding="utf-8")

    original = compute_submit_code_fingerprint(
        src_root=src_root,
        kernel_source_dir=kernel_root,
        sha256_or_none=_sha256_or_none,
    )
    (cache_dir / "module.cpython-312.pyc").write_bytes(b"cache")
    (src_root / "ignored.pyc").write_bytes(b"cache")
    changed = compute_submit_code_fingerprint(
        src_root=src_root,
        kernel_source_dir=kernel_root,
        sha256_or_none=_sha256_or_none,
    )

    assert changed == original


def test_consume_same_submit_fingerprint_retry_allowance_records_code_change() -> None:
    run_state: dict[str, object] = {
        "last_submit_fingerprint": "error-fp",
        "last_submit_code_fingerprint": "old-code",
    }
    saved: list[dict[str, object]] = []

    allowed = consume_same_submit_fingerprint_retry_allowance(
        run_state=run_state,
        fingerprint="error-fp",
        code_fingerprint="new-code",
        save_run_state=saved.append,
    )

    assert allowed is True
    assert run_state["same_fp_allowance_code_fingerprint"] == "new-code"
    assert run_state["same_fp_allowance_error_fingerprint"] == "error-fp"
    assert saved == [
        {
            "same_fp_allowance_code_fingerprint": "new-code",
            "same_fp_allowance_error_fingerprint": "error-fp",
        }
    ]


def test_consume_same_submit_fingerprint_retry_allowance_only_once_per_code_and_error() -> None:
    run_state: dict[str, object] = {
        "last_submit_fingerprint": "error-fp",
        "last_submit_code_fingerprint": "old-code",
        "same_fp_allowance_code_fingerprint": "new-code",
        "same_fp_allowance_error_fingerprint": "error-fp",
    }

    allowed = consume_same_submit_fingerprint_retry_allowance(
        run_state=run_state,
        fingerprint="error-fp",
        code_fingerprint="new-code",
        save_run_state=lambda updates: None,
    )

    assert allowed is False


def test_consume_same_submit_fingerprint_retry_allowance_rejects_unchanged_code() -> None:
    run_state: dict[str, object] = {
        "last_submit_fingerprint": "error-fp",
        "last_submit_code_fingerprint": "same-code",
    }

    allowed = consume_same_submit_fingerprint_retry_allowance(
        run_state=run_state,
        fingerprint="error-fp",
        code_fingerprint="same-code",
        save_run_state=lambda updates: None,
    )

    assert allowed is False
    assert "same_fp_allowance_code_fingerprint" not in run_state


def test_consume_same_submit_fingerprint_retry_allowance_allows_legacy_state_once() -> None:
    run_state: dict[str, object] = {
        "last_submit_fingerprint": "error-fp",
    }
    saved: list[dict[str, object]] = []

    allowed = consume_same_submit_fingerprint_retry_allowance(
        run_state=run_state,
        fingerprint="error-fp",
        code_fingerprint="new-code",
        save_run_state=saved.append,
    )

    assert allowed is True
    assert saved
