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

    return _build_submission_history_payload(
        entries=entries,
        direction=direction,
        source=source,
        count=len(rows),
    )


def merge_current_submission_outcome(
    *,
    history: dict[str, object] | None,
    outcome: dict[str, object],
    direction: str,
    history_path: Path | None = None,
) -> dict[str, object]:
    """Make a just-polled Kaggle outcome immediately available to improvement planning.

    Kaggle's submissions listing can lag behind the outcome poll.  Merging the exact
    matched row keeps the next improvement decision score-aware even during that lag;
    the next live history refresh remains authoritative.
    """

    raw = outcome.get("raw")
    entry = _submission_history_entry(_normalize_submission_row(raw)) if isinstance(raw, dict) else None
    if entry is None:
        score = tolerant_finite_float(outcome.get("score"))
        status = str(outcome.get("status") or "unknown")
        checked_at = str(outcome.get("checked_at") or datetime.now(UTC).isoformat())
        entry = {"score": score, "status": status, "submitted_at": checked_at}

    previous = history or {}
    previous_entries = _history_entries(previous)
    current_identity = _submission_entry_identity(entry)
    already_present = any(_submission_entry_identity(item) == current_identity for item in previous_entries)
    merged_entries = [entry]
    merged_entries.extend(item for item in previous_entries if _submission_entry_identity(item) != current_identity)
    previous_count = _nonnegative_int(previous.get("count"))
    previous_scored_count = _nonnegative_int(previous.get("scored_count"))
    current_is_scored = tolerant_finite_float(entry.get("score")) is not None
    payload = _build_submission_history_payload(
        entries=merged_entries,
        direction=direction,
        source=_merged_history_source(previous.get("source")),
        count=max(len(merged_entries), previous_count + (0 if already_present else 1)),
        scored_count=max(
            sum(tolerant_finite_float(item.get("score")) is not None for item in merged_entries),
            previous_scored_count + (1 if current_is_scored and not already_present else 0),
        ),
    )
    if history_path is not None:
        payload["cache_path"] = str(history_path)
        write_json_object(history_path, payload)
    elif previous.get("cache_path"):
        payload["cache_path"] = previous["cache_path"]
    return payload


def build_public_score_feedback(history: dict[str, object] | None) -> dict[str, object] | None:
    """Summarize the latest public result against the best earlier scored result."""

    if not history:
        return None
    direction = str(history.get("direction") or "maximize")
    best_score = tolerant_finite_float(history.get("best_score"))
    latest = history.get("latest")
    latest_entry = latest if isinstance(latest, dict) else None
    latest_score = tolerant_finite_float(latest_entry.get("score")) if latest_entry is not None else None
    if latest_score is None:
        return {
            "direction": direction,
            "latest_public_score": None,
            "best_public_score": best_score,
            "prior_best_public_score": _best_scored_entry(history.get("recent"), direction=direction),
            "improvement_delta_vs_prior_best": None,
            "result": "latest_unscored" if latest_entry is not None else "no_submissions",
        }

    latest_identity = _submission_entry_identity(latest_entry)
    earlier_entries = [
        item for item in _dict_items(history.get("recent")) if _submission_entry_identity(item) != latest_identity
    ]
    prior_best = _best_scored_entry(earlier_entries, direction=direction)
    if prior_best is None:
        result = "first_scored_submission"
        delta = None
    else:
        delta = latest_score - prior_best if direction == "maximize" else prior_best - latest_score
        result = "improved" if delta > 0 else "regressed" if delta < 0 else "tied"
    return {
        "direction": direction,
        "latest_public_score": latest_score,
        "best_public_score": best_score,
        "prior_best_public_score": prior_best,
        "improvement_delta_vs_prior_best": delta,
        "result": result,
    }


def _build_submission_history_payload(
    *,
    entries: list[dict[str, object]],
    direction: str,
    source: str,
    count: int,
    scored_count: int | None = None,
) -> dict[str, object]:
    entries = _deduplicate_submission_entries(entries)

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

    recent_entries = sorted(entries, key=sort_key, reverse=True)
    recent_scored = sorted(scored_entries, key=sort_key, reverse=True)
    recent_unscored = [entry for entry in recent_entries if tolerant_finite_float(entry.get("score")) is None]
    latest_entry = recent_entries[0] if recent_entries else None

    return {
        "source": source,
        "fetched_at": datetime.now(UTC).isoformat(),
        "direction": direction,
        "count": count,
        "scored_count": len(scored_entries) if scored_count is None else scored_count,
        "best_score": tolerant_finite_float(best_entry.get("score")) if best_entry is not None else None,
        "best": best_entry,
        "latest_score": tolerant_finite_float(latest_entry.get("score")) if latest_entry is not None else None,
        "latest": latest_entry,
        "recent": recent_scored[:10],
        "recent_unscored": recent_unscored[:10],
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
    feedback = build_public_score_feedback(history)
    if feedback is not None:
        latest_score = tolerant_finite_float(feedback.get("latest_public_score"))
        prior_best = tolerant_finite_float(feedback.get("prior_best_public_score"))
        delta = tolerant_finite_float(feedback.get("improvement_delta_vs_prior_best"))
        result = str(feedback.get("result") or "unknown")
        if latest_score is not None:
            lines.append(f"- Latest public score: {latest_score:.6f}.")
            if prior_best is None:
                lines.append("- This is the first scored submission; use it as the public baseline.")
            elif delta is not None:
                lines.append(
                    f"- Public improvement delta vs prior best: {delta:+.6f} (result={result}; positive means better)."
                )
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
    recent_unscored = history.get("recent_unscored")
    if isinstance(recent_unscored, list) and recent_unscored:
        lines.append("- Recent unscored submissions:")
        for item in recent_unscored[:3]:
            if not isinstance(item, dict):
                continue
            submitted = str(item.get("submitted_at") or "unknown_time")
            status = str(item.get("status") or "unknown")
            desc = str(item.get("description") or item.get("label") or "").strip()
            detail = str(item.get("detail") or "").strip()
            suffix = f" {desc}" if desc else ""
            detail_suffix = f" ({detail})" if detail else ""
            lines.append(f"  - {submitted}: status={status}{suffix}{detail_suffix}")
        lines.append(
            "- Treat recent unscored leaderboard submissions as possible scoring/format failures before retrying."
        )
    lines.append(
        "- If the latest public score is worse than the historical best, change the approach instead of "
        "continuing same-family tuning."
    )
    return "\n".join(lines)


def _normalize_submission_row(raw: dict[object, object]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in raw.items()}


def _history_entries(history: dict[str, object]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for key in ("latest", "best"):
        item = history.get(key)
        if isinstance(item, dict):
            entries.append(dict(item))
    entries.extend(_dict_items(history.get("recent")))
    entries.extend(_dict_items(history.get("recent_unscored")))
    return _deduplicate_submission_entries(entries)


def _dict_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _deduplicate_submission_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for entry in entries:
        identity = _submission_entry_identity(entry)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(entry)
    return unique


def _submission_entry_identity(entry: dict[str, object]) -> tuple[object, ...]:
    submitted_at = str(entry.get("submitted_at") or "").strip()
    description = str(entry.get("description") or "").strip()
    label = str(entry.get("label") or "").strip()
    if submitted_at or description or label:
        return (submitted_at, description, label)
    return (
        tolerant_finite_float(entry.get("score")),
        str(entry.get("status") or "").strip(),
        str(entry.get("detail") or "").strip(),
    )


def _best_scored_entry(value: object, *, direction: str) -> float | None:
    entries = _dict_items(value) if isinstance(value, list) else value
    if not isinstance(entries, list):
        return None
    best: float | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        score = tolerant_finite_float(entry.get("score"))
        if score is not None and should_update_best_score(best, score, direction, 0.0):
            best = score
    return best


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _merged_history_source(value: object) -> str:
    source = str(value or "cache").strip()
    suffix = "live outcome"
    return source if suffix in source else f"{source} + {suffix}"


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
        "recent_unscored": [],
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
    detail = None
    for key in (
        "errorDescription",
        "error_description",
        "failureReason",
        "statusDescription",
        "statusMessage",
        "error",
    ):
        value = SubmissionOutcomeService._get_row_value_ci(row, key)
        if value and value.strip():
            detail = value.strip()
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
    if detail:
        entry["detail"] = detail
    if rank is not None:
        entry["rank"] = rank
    if total_teams is not None:
        entry["total_teams"] = total_teams
    if rank is not None and total_teams is not None and total_teams > 0:
        entry["rank_percentile"] = rank / total_teams
    return entry
