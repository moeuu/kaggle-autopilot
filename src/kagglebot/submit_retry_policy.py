from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path


def compute_submit_code_fingerprint(
    *,
    src_root: Path,
    kernel_source_dir: Path,
    sha256_or_none: Callable[[Path | None], str | None],
) -> str:
    """Compute a stable fingerprint of submit-relevant local code."""
    hasher = hashlib.sha256()
    root_specs = (
        ("src", src_root),
        ("kernel", kernel_source_dir),
    )
    for label, root in root_specs:
        if not root.exists() or not root.is_dir():
            hasher.update(f"{label}:<missing>\n".encode())
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.name
            hasher.update(f"{label}:{rel}\n".encode())
            hasher.update((sha256_or_none(path) or "missing").encode())
            hasher.update(b"\n")
    return hasher.hexdigest()


def consume_same_submit_fingerprint_retry_allowance(
    *,
    run_state: dict[str, object],
    fingerprint: str,
    code_fingerprint: str,
    save_run_state: Callable[[dict[str, object]], None],
) -> bool:
    """Allow one repeated-fingerprint retry after submit-relevant code changes."""
    last_code_fingerprint = str(run_state.get("last_submit_code_fingerprint") or "").strip()
    prior_error_fingerprint = str(
        run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or ""
    ).strip()
    if not code_fingerprint:
        return False

    consumed_code_fingerprint = str(run_state.get("same_fp_allowance_code_fingerprint") or "").strip()
    consumed_error_fingerprint = str(run_state.get("same_fp_allowance_error_fingerprint") or "").strip()
    if consumed_code_fingerprint == code_fingerprint and consumed_error_fingerprint == fingerprint:
        return False

    # Backward compatibility for runs recorded before code_fingerprint tracking existed.
    # In that case we cannot compare "before vs after" code reliably, so allow exactly once.
    if not last_code_fingerprint:
        if not prior_error_fingerprint or prior_error_fingerprint != fingerprint:
            return False
        _record_same_fingerprint_allowance(
            run_state=run_state,
            fingerprint=fingerprint,
            code_fingerprint=code_fingerprint,
            save_run_state=save_run_state,
        )
        return True

    if code_fingerprint == last_code_fingerprint:
        return False

    _record_same_fingerprint_allowance(
        run_state=run_state,
        fingerprint=fingerprint,
        code_fingerprint=code_fingerprint,
        save_run_state=save_run_state,
    )
    return True


def _record_same_fingerprint_allowance(
    *,
    run_state: dict[str, object],
    fingerprint: str,
    code_fingerprint: str,
    save_run_state: Callable[[dict[str, object]], None],
) -> None:
    updates = {
        "same_fp_allowance_code_fingerprint": code_fingerprint,
        "same_fp_allowance_error_fingerprint": fingerprint,
    }
    run_state.update(updates)
    save_run_state(updates)
