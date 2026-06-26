from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.json_utils import load_json_object_or_empty, write_json_object

RESTART_STATE_FILENAME = "autofix_restart.json"
KERNEL_REGENERATE_MARKER_FILENAME = "kernel_regenerated_once.json"
NO_RESTART_ENV = "KAGGLEBOT_NO_RESTART"
RESUME_RUN_ID_ENV = "KAGGLEBOT_RESUME_RUN_ID"
RESUME_SLUG_ENV = "KAGGLEBOT_RESUME_SLUG"


def maybe_restart_for_src_changes(
    *,
    dry_run: bool,
    run_dir: Path,
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

    state_path = run_dir / RESTART_STATE_FILENAME
    state = _load_restart_state(state_path)
    stage_family = restart_stage_family(stage)
    counts_by_stage = _restart_counts_by_stage(state)
    stage_count = int(counts_by_stage.get(stage_family, 0))
    if stage_count >= max_restarts:
        print(f"[yellow]autofix[/yellow]: src changes detected in {stage}, restart limit reached")
        return False

    counts_by_stage[stage_family] = stage_count + 1
    state["counts_by_stage"] = counts_by_stage
    state["count"] = sum(counts_by_stage.values())
    state["last_stage"] = stage
    state["last_stage_family"] = stage_family
    write_json_object(state_path, state)
    print(
        f"[yellow]autofix[/yellow]: src changes detected in {stage}; "
        "current process may be stale, restarting to reload code"
    )
    os.environ[RESUME_RUN_ID_ENV] = run_id
    os.environ[RESUME_SLUG_ENV] = slug
    os.execv(sys.executable, [sys.executable, *sys.argv])
    return True


def restart_stage_family(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if not normalized:
        return "unknown"
    return normalized.split("_attempt_", 1)[0]


def kernel_regenerate_marker_path(agent_dir: Path) -> Path:
    return agent_dir / KERNEL_REGENERATE_MARKER_FILENAME


def kernel_regeneration_already_marked(agent_dir: Path) -> bool:
    return kernel_regenerate_marker_path(agent_dir).exists()


def write_kernel_regeneration_marker(
    *,
    agent_dir: Path,
    run_id: str,
    iteration: int,
    attempt: int,
    trigger_reason: str,
) -> Path:
    agent_dir.mkdir(parents=True, exist_ok=True)
    marker_path = kernel_regenerate_marker_path(agent_dir)
    write_json_object(
        marker_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "trigger_reason": trigger_reason,
            "attempt": int(attempt),
            "iteration": int(iteration),
            "run_id": run_id,
        },
    )
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
