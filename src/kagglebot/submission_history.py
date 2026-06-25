from __future__ import annotations

from datetime import UTC, datetime

from kagglebot.scalar_utils import parse_finite_float
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

    scored_entries = [entry for entry in entries if _to_float(entry.get("score")) is not None]
    best_entry: dict[str, object] | None = None
    for entry in scored_entries:
        score = _to_float(entry.get("score"))
        best_score = _to_float(best_entry.get("score")) if best_entry is not None else None
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
        "best_score": _to_float(best_entry.get("score")) if best_entry is not None else None,
        "best": best_entry,
        "latest_score": _to_float(latest_entry.get("score")) if latest_entry is not None else None,
        "latest": latest_entry,
        "recent": recent_scored[:10],
    }


def detect_online_regression_vs_submission_history(
    *,
    previous_best_online: float | None,
    current_online: float | None,
    direction: str,
    history: dict[str, object] | None,
) -> dict[str, object] | None:
    historical_best = _to_float((history or {}).get("best_score"))
    baseline = historical_best if historical_best is not None else previous_best_online
    current = _to_float(current_online)
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
    best_score = _to_float(history.get("best_score"))
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
            score = _to_float(item.get("score"))
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


def _to_float(value: object) -> float | None:
    return parse_finite_float(value, allow_commas=True)
