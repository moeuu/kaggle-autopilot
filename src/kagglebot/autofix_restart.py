from __future__ import annotations

import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from kagglebot.json_utils import load_json_object_or_empty, write_json_object

RESTART_STATE_FILENAME = "autofix_restart.json"
KERNEL_REGENERATE_MARKER_FILENAME = "kernel_regenerated_once.json"
NO_RESTART_ENV = "KAGGLEBOT_NO_RESTART"
RESUME_RUN_ID_ENV = "KAGGLEBOT_RESUME_RUN_ID"
RESUME_SLUG_ENV = "KAGGLEBOT_RESUME_SLUG"


class SourceReloadLoopError(SystemExit):
    """Stop a stale process when one source generation already requested reload."""


def maybe_restart_for_src_changes(
    *,
    dry_run: bool,
    run_dir: Path,
    repo_root: Path,
    run_id: str,
    slug: str,
    changed: list[str],
    stage: str,
    max_restarts: int,
) -> bool:
    if dry_run:
        return False
    if os.environ.get(NO_RESTART_ENV) == "1":
        return False
    if not any(path.startswith("src/") for path in changed):
        return False

    source_fingerprint = source_tree_fingerprint(repo_root)
    state_path = run_dir / RESTART_STATE_FILENAME
    state = _load_restart_state(state_path)
    stage_family = restart_stage_family(stage)
    counts_by_stage = _restart_counts_by_stage(state)
    stage_count = int(counts_by_stage.get(stage_family, 0))
    raw_source_counts = state.get("counts_by_source")
    counts_by_source = dict(raw_source_counts) if isinstance(raw_source_counts, dict) else {}
    source_count = int(counts_by_source.get(source_fingerprint, 0))
    if source_count >= max(1, max_restarts):
        raise SourceReloadLoopError(
            "Source reload did not advance to the new code generation; refusing to continue in a stale process "
            f"(stage={stage}, source={source_fingerprint[:12]})."
        )

    counts_by_stage[stage_family] = stage_count + 1
    counts_by_source[source_fingerprint] = source_count + 1
    state["counts_by_stage"] = counts_by_stage
    state["counts_by_source"] = counts_by_source
    state["count"] = sum(counts_by_stage.values())
    state["last_stage"] = stage
    state["last_stage_family"] = stage_family
    state["last_source_fingerprint"] = source_fingerprint
    write_json_object(state_path, state)
    print(
        f"[yellow]autofix[/yellow]: src changes detected in {stage}; "
        "current process may be stale, restarting to reload code"
    )
    os.environ[RESUME_RUN_ID_ENV] = run_id
    os.environ[RESUME_SLUG_ENV] = slug
    os.execv(sys.executable, [sys.executable, *sys.argv])
    return True


def source_tree_fingerprint(repo_root: Path) -> str:
    """Hash the complete source generation that an exec restart must reload."""
    source_root = repo_root.resolve() / "src"
    digest = sha256()
    if not source_root.is_dir():
        digest.update(b"missing-src\0")
        return digest.hexdigest()
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(repo_root.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def restart_stage_family(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if not normalized:
        return "unknown"
    return normalized.split("_attempt_", 1)[0]


def kernel_regenerate_marker_path(agent_dir: Path) -> Path:
    return agent_dir / KERNEL_REGENERATE_MARKER_FILENAME


def kernel_regeneration_already_marked(agent_dir: Path, *, kernel_sha256: str | None = None) -> bool:
    marker_path = kernel_regenerate_marker_path(agent_dir)
    if not marker_path.exists():
        return False
    if kernel_sha256 is None:
        return True
    marker = load_json_object_or_empty(marker_path)
    marked_sha = marker.get("kernel_sha_after") or marker.get("kernel_sha_before")
    if not marked_sha:
        # Preserve the bounded behavior of markers created before SHA tracking.
        return True
    return str(marked_sha) == kernel_sha256


def write_kernel_regeneration_marker(
    *,
    agent_dir: Path,
    run_id: str,
    iteration: int,
    attempt: int,
    trigger_reason: str,
    kernel_sha_before: str | None = None,
    kernel_sha_after: str | None = None,
) -> Path:
    agent_dir.mkdir(parents=True, exist_ok=True)
    marker_path = kernel_regenerate_marker_path(agent_dir)
    payload: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "trigger_reason": trigger_reason,
        "attempt": int(attempt),
        "iteration": int(iteration),
        "run_id": run_id,
    }
    if kernel_sha_before is not None:
        payload["kernel_sha_before"] = kernel_sha_before
    if kernel_sha_after is not None:
        payload["kernel_sha_after"] = kernel_sha_after
    write_json_object(marker_path, payload)
    return marker_path


def write_kernel_regeneration_note(
    *,
    agent_dir: Path,
    attempt: int,
    trigger_reason: str,
    error: Exception | None = None,
) -> Path:
    agent_dir.mkdir(parents=True, exist_ok=True)
    note_path = agent_dir / f"kernel_regen_note-{attempt:02d}.txt"
    if error is None:
        text = f"kernel_regen_applied: regeneration fallback succeeded.\ntrigger_reason: {trigger_reason}\n"
    else:
        text = f"kernel_regen_failed: regeneration fallback failed.\ntrigger_reason: {trigger_reason}\nerror: {error}\n"
    note_path.write_text(text, encoding="utf-8")
    return note_path


def maybe_regenerate_kernel_sources_once(
    *,
    dry_run: bool,
    agent_dir: Path,
    run_id: str,
    iteration: int,
    attempt: int,
    trigger_reason: str,
    regenerate_kernel_sources: Callable[[], None],
    get_kernel_sha256: Callable[[], str | None] | None = None,
    on_message: Callable[[str], None] = print,
) -> bool:
    """Regenerate authoritative kernel sources once when fix loops are stuck."""
    if dry_run:
        return False
    kernel_sha_before = get_kernel_sha256() if get_kernel_sha256 is not None else None
    if kernel_regeneration_already_marked(agent_dir, kernel_sha256=kernel_sha_before):
        return False

    write_kernel_regeneration_marker(
        agent_dir=agent_dir,
        run_id=run_id,
        iteration=iteration,
        attempt=attempt,
        trigger_reason=trigger_reason,
        kernel_sha_before=kernel_sha_before,
    )
    on_message(
        "[yellow]kernel fix[/yellow]: unresolved kernel error loop detected; "
        "regenerating kernel sources once before retry."
    )
    try:
        regenerate_kernel_sources()
    except Exception as exc:  # noqa: BLE001
        write_kernel_regeneration_note(
            agent_dir=agent_dir,
            attempt=attempt,
            trigger_reason=trigger_reason,
            error=exc,
        )
        return False
    if get_kernel_sha256 is not None:
        write_kernel_regeneration_marker(
            agent_dir=agent_dir,
            run_id=run_id,
            iteration=iteration,
            attempt=attempt,
            trigger_reason=trigger_reason,
            kernel_sha_before=kernel_sha_before,
            kernel_sha_after=get_kernel_sha256(),
        )
    write_kernel_regeneration_note(
        agent_dir=agent_dir,
        attempt=attempt,
        trigger_reason=trigger_reason,
    )
    return True


def _load_restart_state(path: Path) -> dict[str, object]:
    return load_json_object_or_empty(path)


def _restart_counts_by_stage(state: dict[str, object]) -> dict[str, int]:
    counts_by_stage: dict[str, int] = {}
    raw_counts = state.get("counts_by_stage")
    if isinstance(raw_counts, dict):
        for key, value in raw_counts.items():
            if not isinstance(key, str):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                counts_by_stage[key] = parsed
        return counts_by_stage

    try:
        legacy_count = int(state.get("count", 0))
    except (TypeError, ValueError):
        legacy_count = 0
    if legacy_count > 0:
        legacy_stage = str(state.get("last_stage") or "").strip()
        legacy_family = restart_stage_family(legacy_stage) if legacy_stage else "legacy"
        counts_by_stage[legacy_family] = legacy_count
    return counts_by_stage
