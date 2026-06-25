from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.scalar_utils import tolerant_finite_float
from kagglebot.score_utils import should_update_best_score
from kagglebot.submission.outcome_service import SubmissionOutcomeService


def build_previous_submission_history_payload(
    *,
    rows: list[dict[str, str]],
    direction: str,
    source: str,
) -> dict[str, object]:
    entries = [_submission_history_entry(row) for row in rows]
    entries = [entry for entry in entries if entry is not None]

    scored_entries = [entry for entry in entries if tolerant_finite_float(entry.get("score")) is not None]
    best_entry: dict[str, object] | None = None
    for entry in scored_entries:
        score = tolerant_finite_float(entry.get("score"))
        best_score = tolerant_finite_float(best_entry.get("score")) if best_entry is not None else None
        if score is not None and should_update_best_score(best_score, score, direction, 0.0):
            best_entry = entry

    def sort_key(entry: dict[str, object]) -> tuple[int, str]:
        submitted_at = str(entry.get("submitted_at") or "")
        return (1 if submitted_at else 0, submitted_at)

    recent_scored = sorted(scored_entries, key=sort_key, reverse=True)
    latest_entry = recent_scored[0] if recent_scored else (entries[0] if entries else None)

    return {
        "source": source,
        "fetched_at": datetime.now(UTC).isoformat(),
        "direction": direction,
        "count": len(rows),
        "scored_count": len(scored_entries),
        "best_score": tolerant_finite_float(best_entry.get("score")) if best_entry is not None else None,
        "best": best_entry,
        "latest_score": tolerant_finite_float(latest_entry.get("score")) if latest_entry is not None else None,
        "latest": latest_entry,
        "recent": recent_scored[:10],
    }


def load_previous_submission_history(
    *,
    slug: str,
    history_path: Path,
    direction: str,
    dry_run: bool,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    on_message: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if dry_run and history_path.exists():
        cached = load_json_object(history_path)
        if cached is not None:
            cached["source"] = str(cached.get("source") or "cache")
            cached["cache_path"] = str(history_path)
            return cached
    if dry_run:
        return _empty_submission_history_payload(
            source="dry_run",
            direction=direction,
            history_path=history_path,
        )

    try:
        rows = fetch_submission_rows(slug)
    except Exception as exc:  # noqa: BLE001
        cached = load_json_object(history_path) if history_path.exists() else None
        if cached is not None:
            cached["source"] = str(cached.get("source") or "cache")
            cached["fetch_error"] = f"{type(exc).__name__}: {exc}"
            cached["cache_path"] = str(history_path)
            _emit(
                on_message,
                "[yellow]submission history[/yellow]: "
                f"failed to refresh Kaggle submissions; using cached {history_path}",
            )
            return cached
        _emit(on_message, f"[yellow]submission history[/yellow]: failed to fetch Kaggle submissions: {exc}")
        return _empty_submission_history_payload(
            source="fetch_error",
            direction=direction,
            history_path=history_path,
            fetch_error=f"{type(exc).__name__}: {exc}",
        )

    payload = build_previous_submission_history_payload(
        rows=rows,
        direction=direction,
        source="kaggle competitions submissions --csv",
    )
    payload["cache_path"] = str(history_path)
    write_json_object(history_path, payload)
    return payload


def detect_online_regression_vs_submission_history(
    *,
    previous_best_online: float | None,
    current_online: float | None,
    direction: str,
    history: dict[str, object] | None,
) -> dict[str, object] | None:
    historical_best = tolerant_finite_float((history or {}).get("best_score"))
    baseline = historical_best if historical_best is not None else previous_best_online
    current = tolerant_finite_float(current_online)
    if baseline is None or current is None:
        return None
    if should_update_best_score(baseline, current, direction, 0.0):
        return None
    best_entry = (history or {}).get("best")
    return {
        "previous_best_online": baseline,
        "current_online": current,
        "direction": direction,
        "best": best_entry if isinstance(best_entry, dict) else None,
        "note": (
            "Current public leaderboard score is worse than the best historical Kaggle submission "
            f"(current={current:.6f}, historical_best={baseline:.6f}, direction={direction}). "
            "Treat the historical submission as the public baseline and require a materially different "
            "model/feature/blend strategy before the next submission."
        ),
    }


def format_previous_submission_history_for_prompt(history: dict[str, object] | None) -> str:
    if not history:
        return ""
    best_score = tolerant_finite_float(history.get("best_score"))
    recent = history.get("recent")
    if best_score is None and not isinstance(recent, list):
        return ""
    lines = [
        "- Always use these Kaggle public submission results as the online baseline for this competition.",
    ]
    if best_score is not None:
        direction = history.get("direction") or "auto"
        lines.append(f"- Best historical public score: {best_score:.6f} (direction={direction}).")
        lines.append("- Do not call a new iteration improved unless its public score beats this historical baseline.")
    best = history.get("best")
    if isinstance(best, dict):
        best_desc = str(best.get("description") or best.get("label") or "").strip()
        if best_desc:
            lines.append(f"- Best submission: {best_desc}")
    if isinstance(recent, list) and recent:
        lines.append("- Recent scored submissions:")
        for item in recent[:5]:
            if not isinstance(item, dict):
                continue
            score = tolerant_finite_float(item.get("score"))
            if score is None:
                continue
            submitted = str(item.get("submitted_at") or "unknown_time")
            desc = str(item.get("description") or item.get("label") or "").strip()
            suffix = f" {desc}" if desc else ""
            lines.append(f"  - {submitted}: public={score:.6f}{suffix}")
    lines.append(
        "- If the latest public score is worse than the historical best, change the approach instead of "
        "continuing same-family tuning."
    )
    return "\n".join(lines)


def _empty_submission_history_payload(
    *,
    source: str,
    direction: str,
    history_path: Path,
    fetch_error: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "fetched_at": datetime.now(UTC).isoformat(),
        "direction": direction,
        "count": 0,
        "scored_count": 0,
        "best_score": None,
        "best": None,
        "latest_score": None,
        "latest": None,
        "recent": [],
        "cache_path": str(history_path),
    }
    if fetch_error is not None:
        payload["fetch_error"] = fetch_error
    return payload


def _emit(on_message: Callable[[str], None] | None, message: str) -> None:
    if on_message is not None:
        on_message(message)


def _submission_history_entry(row: dict[str, str]) -> dict[str, object] | None:
    if not row:
        return None
    score = SubmissionOutcomeService._extract_submission_score(row)
    status = SubmissionOutcomeService._extract_submission_status(row)
    submitted_at = SubmissionOutcomeService._parse_submission_row_time(row)
    rank, total_teams = SubmissionOutcomeService._extract_submission_rank(row)
    description = None
    for key in ("description", "message", "comments", "comment"):
        value = SubmissionOutcomeService._get_row_value_ci(row, key)
        if value and value.strip():
            description = value.strip()
            break
    label = None
    for key in ("fileName", "file_name", "ref", "name", "title"):
        value = SubmissionOutcomeService._get_row_value_ci(row, key)
        if value and value.strip():
            label = value.strip()
            break

    entry: dict[str, object] = {
        "score": score,
        "status": status,
    }
    if submitted_at is not None:
        entry["submitted_at"] = submitted_at.isoformat()
    if description:
        entry["description"] = description
    if label:
        entry["label"] = label
    if rank is not None:
        entry["rank"] = rank
    if total_teams is not None:
        entry["total_teams"] = total_teams
    if rank is not None and total_teams is not None and total_teams > 0:
        entry["rank_percentile"] = rank / total_teams
    return entry
