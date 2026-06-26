from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.scalar_utils import parse_finite_float, parse_int


@dataclass(frozen=True)
class SubmissionRankState:
    rank_payload: dict[str, object]
    rank: int | None
    total_teams: int | None
    rank_percentile: float | None
    rank_source: str | None
    estimated_rank: int | None
    estimated_total_teams: int | None
    estimated_rank_percentile: float | None
    rank_estimate_source: str | None
    force_major_overhaul: bool
    force_reason: str | None
    messages: tuple[str, ...]


def resolve_submission_rank_payload(
    *,
    slug: str,
    context_dir: Path,
    direction: str,
    outcome: dict[str, object],
    dry_run: bool,
    leaderboard_rank_for_score: Callable[..., dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    rank = parse_int(outcome.get("rank"), allow_float=True)
    total_teams = parse_int(outcome.get("total_teams"), allow_float=True)
    rank_percentile = parse_finite_float(outcome.get("rank_percentile"))
    rank_source = outcome.get("rank_source")

    if rank is not None:
        payload["rank"] = rank
    if total_teams is not None:
        payload["total_teams"] = total_teams
    if rank_percentile is not None:
        payload["rank_percentile"] = rank_percentile
    if isinstance(rank_source, str) and rank_source.strip():
        payload["rank_source"] = rank_source.strip()

    if rank is None or total_teams is None:
        score = parse_finite_float(outcome.get("score"))
        if score is not None:
            try:
                estimate = leaderboard_rank_for_score(
                    slug=slug,
                    output_dir=context_dir,
                    score=score,
                    direction=direction,
                    dry_run=dry_run,
                )
            except Exception:  # noqa: BLE001
                estimate = {}
            est_rank = parse_int(estimate.get("rank"), allow_float=True)
            est_total = parse_int(estimate.get("total_teams"), allow_float=True)
            est_percentile = parse_finite_float(estimate.get("rank_percentile"))
            if est_rank is not None:
                payload["estimated_rank"] = est_rank
            if est_total is not None:
                payload["estimated_total_teams"] = est_total
            if est_percentile is not None:
                payload["estimated_rank_percentile"] = est_percentile
            if est_rank is not None and isinstance(estimate.get("source"), str):
                payload["rank_estimate_source"] = "leaderboard_score_estimate"

    resolved_rank = parse_int(payload.get("rank"), allow_float=True)
    resolved_total = parse_int(payload.get("total_teams"), allow_float=True)
    if resolved_rank is not None and resolved_total is not None and resolved_total > 0:
        payload.setdefault("rank_percentile", resolved_rank / resolved_total)
    return payload


def format_rank_force_reason(
    *,
    rank: int,
    total_teams: int,
    rank_percentile: float | None,
    max_percentile: float,
    min_teams: int,
    source: str | None,
) -> str:
    resolved_percentile = (rank / total_teams) if rank_percentile is None and total_teams > 0 else rank_percentile
    percentile_text = f"{(resolved_percentile or 0.0) * 100:.2f}%" if resolved_percentile is not None else "n/a"
    source_text = f" source={source}" if source else ""
    return (
        "Leaderboard rank indicates large headroom for improvement: "
        f"{rank}/{total_teams} (percentile={percentile_text}, threshold={max_percentile * 100:.2f}%, "
        f"min_teams={min_teams}).{source_text}"
    )


def format_submission_rank_message(
    *,
    rank: int,
    total_teams: int,
    rank_percentile: float | None,
    source: str | None,
    estimated: bool = False,
) -> str:
    resolved_percentile = (rank / total_teams) if rank_percentile is None and total_teams > 0 else rank_percentile
    percentile_text = f"{resolved_percentile * 100:.2f}%" if resolved_percentile is not None else "n/a"
    source_text = f" source={source}" if source else ""
    prefix = "[yellow]submission rank estimate[/yellow]" if estimated else "[cyan]submission rank[/cyan]"
    return f"{prefix}: {rank}/{total_teams} (percentile={percentile_text}){source_text}"


def resolve_submission_rank_state(
    *,
    rank_payload: dict[str, object],
    rank_force_major_max_percentile: float,
    rank_force_major_min_teams: int,
    should_force_major_overhaul_by_rank: Callable[..., bool],
) -> SubmissionRankState:
    submission_rank = parse_int(rank_payload.get("rank"), allow_float=True)
    submission_total_teams = parse_int(rank_payload.get("total_teams"), allow_float=True)
    submission_rank_percentile = parse_finite_float(rank_payload.get("rank_percentile"))
    submission_rank_estimate = parse_int(rank_payload.get("estimated_rank"), allow_float=True)
    submission_total_teams_estimate = parse_int(rank_payload.get("estimated_total_teams"), allow_float=True)
    submission_rank_percentile_estimate = parse_finite_float(rank_payload.get("estimated_rank_percentile"))

    estimate_source_raw = rank_payload.get("rank_estimate_source")
    submission_rank_estimate_source = (
        estimate_source_raw.strip() if isinstance(estimate_source_raw, str) and estimate_source_raw.strip() else None
    )
    source_raw = rank_payload.get("rank_source")
    submission_rank_source = str(source_raw) if source_raw is not None else None

    messages: list[str] = []
    rank_forced_major_overhaul = False
    rank_force_reason: str | None = None
    if submission_rank is not None and submission_total_teams is not None and submission_total_teams > 0:
        if submission_rank_percentile is None:
            submission_rank_percentile = submission_rank / submission_total_teams
        messages.append(
            format_submission_rank_message(
                rank=submission_rank,
                total_teams=submission_total_teams,
                rank_percentile=submission_rank_percentile,
                source=submission_rank_source,
            )
        )
        rank_forced_major_overhaul = should_force_major_overhaul_by_rank(
            rank=submission_rank,
            total_teams=submission_total_teams,
            max_percentile=rank_force_major_max_percentile,
            min_teams=rank_force_major_min_teams,
        )
        if rank_forced_major_overhaul:
            rank_force_reason = format_rank_force_reason(
                rank=submission_rank,
                total_teams=submission_total_teams,
                rank_percentile=submission_rank_percentile,
                max_percentile=rank_force_major_max_percentile,
                min_teams=rank_force_major_min_teams,
                source=submission_rank_source,
            )
            messages.append(f"[yellow]rank guard[/yellow]: {rank_force_reason}")
    elif (
        submission_rank_estimate is not None
        and submission_total_teams_estimate is not None
        and submission_total_teams_estimate > 0
    ):
        if submission_rank_percentile_estimate is None:
            submission_rank_percentile_estimate = submission_rank_estimate / submission_total_teams_estimate
        messages.append(
            format_submission_rank_message(
                rank=submission_rank_estimate,
                total_teams=submission_total_teams_estimate,
                rank_percentile=submission_rank_percentile_estimate,
                source=submission_rank_estimate_source,
                estimated=True,
            )
        )

    return SubmissionRankState(
        rank_payload=rank_payload,
        rank=submission_rank,
        total_teams=submission_total_teams,
        rank_percentile=submission_rank_percentile,
        rank_source=submission_rank_source,
        estimated_rank=submission_rank_estimate,
        estimated_total_teams=submission_total_teams_estimate,
        estimated_rank_percentile=submission_rank_percentile_estimate,
        rank_estimate_source=submission_rank_estimate_source,
        force_major_overhaul=rank_forced_major_overhaul,
        force_reason=rank_force_reason,
        messages=tuple(messages),
    )
