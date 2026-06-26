from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from kagglebot.datetime_utils import parse_datetime_utc

SPARE_SUBMIT_RELAXABLE_QUALITY_REASONS = frozenset(
    {
        "selected_worse_than_detected_baseline",
        "below_code_reference_baseline",
    }
)


@dataclass(frozen=True)
class QualitySubmitOverrideDecision:
    quality_allows_submit: bool
    forced_submit_reason: str | None = None
    override_reason: str | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class InitialSubmitProbeDecision:
    force_initial_submit: bool
    quality_allows_submit: bool
    allow_submit: bool
    forced_submit_reason: str | None = None
    soft_probe_override: bool = False
    probe_forced: bool = False
    skipped_reason: str | None = None


@dataclass(frozen=True)
class LimitedSubmissionHoldbackDecision:
    holdback: bool
    reason: str | None = None


@dataclass(frozen=True)
class PlanSubmissionPolicyDecision:
    submit_policy: str
    submission_gate: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class MajorOverhaulPolicyDecision:
    force_major_overhaul: bool
    forced_major_overhaul_reason: str | None
    fallback_submit_blocked_reason: str | None


def meets_target(value: float, target: float, direction: str) -> bool:
    if direction == "minimize":
        return value <= target
    return value >= target


def is_top1_tier(value: float, top1_score: float | None, direction: str) -> bool:
    if top1_score is None:
        return False
    if direction == "minimize":
        return value <= top1_score
    return value >= top1_score


def has_spare_daily_submission_slot(
    *,
    submission_limit_per_day: int | None,
    submissions_used_today: int,
    iteration: int,
    max_iterations: int,
) -> bool:
    if not isinstance(submission_limit_per_day, int) or submission_limit_per_day <= 0:
        return False
    remaining_slots = max(0, submission_limit_per_day - max(0, int(submissions_used_today)))
    remaining_iterations = max(1, int(max_iterations) - int(iteration) + 1)
    return remaining_slots >= remaining_iterations


def non_final_submission_checkpoints(*, max_iterations: int, non_final_slots: int) -> set[int]:
    """Spread non-final submit slots across the loop to avoid early budget burn."""
    if max_iterations <= 1 or non_final_slots <= 0:
        return set()
    last_non_final = max_iterations - 1
    if non_final_slots >= last_non_final:
        return set(range(1, max_iterations))

    checkpoints: set[int] = set()
    for idx in range(1, non_final_slots + 1):
        # Integer spacing over [1, max_iterations-1], leaving room for final slot.
        candidate = (idx * max_iterations) // (non_final_slots + 1)
        candidate = max(1, min(last_non_final, candidate))
        checkpoints.add(candidate)

    if len(checkpoints) < non_final_slots:
        for candidate in range(last_non_final, 0, -1):
            checkpoints.add(candidate)
            if len(checkpoints) >= non_final_slots:
                break
    return checkpoints


def should_attempt_submit_for_readiness(
    *,
    gate: str,
    readiness_score: float | None,
    readiness_target: float,
    direction: str,
    iteration: int,
    max_iterations: int,
    submission_limit_per_day: int | None = None,
    successful_submissions: int = 0,
    top1_score: float | None = None,
) -> bool:
    normalized = normalized_submission_gate(gate, default="always")
    is_final_iteration = iteration >= max_iterations
    met_target = readiness_score is not None and meets_target(readiness_score, readiness_target, direction)
    top1_tier = readiness_score is not None and is_top1_tier(readiness_score, top1_score, direction)

    if isinstance(submission_limit_per_day, int) and submission_limit_per_day > 0:
        if max(0, int(successful_submissions)) >= submission_limit_per_day:
            return False

    if normalized in {"final_only", "at_final"}:
        return is_final_iteration
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return met_target

    if isinstance(submission_limit_per_day, int) and submission_limit_per_day > 0:
        if is_final_iteration:
            return True

        non_final_slots = max(0, submission_limit_per_day - 1)
        if non_final_slots <= 0:
            return False

        if has_spare_daily_submission_slot(
            submission_limit_per_day=submission_limit_per_day,
            submissions_used_today=successful_submissions,
            iteration=iteration,
            max_iterations=max_iterations,
        ):
            return True

        if successful_submissions >= non_final_slots:
            return top1_tier or met_target

        if max_iterations > submission_limit_per_day:
            checkpoints = non_final_submission_checkpoints(
                max_iterations=max_iterations,
                non_final_slots=non_final_slots,
            )
            return (iteration in checkpoints) or top1_tier or met_target

        return True

    if normalized in {"always", "each_iteration"}:
        return True
    if normalized in {"readiness_or_final", "target_or_final"}:
        return met_target or is_final_iteration
    if readiness_score is None:
        return is_final_iteration
    return met_target or is_final_iteration


def submission_gate_for_policy(policy: str | None) -> str:
    normalized = normalized_submit_policy(policy)
    if normalized == "improved":
        return "always"
    if normalized in {"always", "each_iteration"}:
        return "always"
    if normalized in {"final_only", "at_final"}:
        return "final_only"
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return "readiness_only"
    if normalized in {"readiness_or_final", "target_or_final"}:
        return "readiness_or_final"
    return "always"


def normalized_submit_policy(policy: str | None) -> str:
    normalized = str(policy or "").strip().lower()
    if normalized in {"improved", "improvement_only", "improved_only", "on_improvement"}:
        return "improved"
    if normalized in {"always", "each_iteration"}:
        return "always"
    if normalized in {"final_only", "at_final"}:
        return "final_only"
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return "readiness_only"
    if normalized in {"readiness_or_final", "target_or_final"}:
        return "readiness_or_final"
    return "always"


def normalize_watch_submit_policy(policy: str | None) -> str:
    """Normalize the narrower submit policy accepted by watch commands."""
    normalized = str(policy or "").strip().lower()
    if normalized in {"improved", "improvement_only", "improved_only", "on_improvement"}:
        return "improved"
    if normalized in {"none", "off", "disabled", "no_submit", "no-submit"}:
        return "none"
    raise ValueError("submit_policy must be one of: improved, none")


def normalized_submission_gate(gate: str | None, *, default: str) -> str:
    normalized = str(gate or "").strip().lower()
    if normalized in {"always", "each_iteration"}:
        return "always"
    if normalized in {"final_only", "at_final"}:
        return "final_only"
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return "readiness_only"
    if normalized in {"readiness_or_final", "target_or_final"}:
        return "readiness_or_final"
    return default


def resolve_plan_submission_policy(
    *,
    config_submit_policy: str | None,
    requested_submit_policy: str | None,
    requested_submission_gate: str | None,
    submission_limit_detected: bool,
    default_limited_submission_gate: str,
) -> PlanSubmissionPolicyDecision:
    """Resolve plan/config submit policy into the runtime policy and gate."""

    forced_submit_policy = normalized_submit_policy(config_submit_policy) if config_submit_policy else None
    requested_policy = str(requested_submit_policy or "always")
    requested_gate = str(requested_submission_gate or "").strip().lower() or None

    if forced_submit_policy == "improved":
        return PlanSubmissionPolicyDecision(submit_policy="improved", submission_gate="always")

    if submission_limit_detected:
        submit_policy = normalized_submit_policy(requested_policy)
        default_gate = submission_gate_for_policy(submit_policy)
        submission_gate = (
            normalized_submission_gate(requested_gate, default=default_gate)
            if requested_gate is not None
            else default_gate
        )
        if submission_gate == "always" and requested_gate is None and submit_policy == "always":
            submission_gate = normalized_submission_gate(default_limited_submission_gate, default="readiness_or_final")
            submit_policy = "readiness_or_final"
            return PlanSubmissionPolicyDecision(
                submit_policy=submit_policy,
                submission_gate=submission_gate,
                messages=(
                    "[yellow]note[/yellow]: submission limit detected in rules; "
                    f"defaulting submission_gate={submission_gate}.",
                ),
            )
        return PlanSubmissionPolicyDecision(submit_policy=submit_policy, submission_gate=submission_gate)

    messages: list[str] = []
    if normalized_submit_policy(requested_policy) != "always":
        messages.append(
            f"[yellow]note[/yellow]: no submission limit detected; ignoring submit_policy='{requested_policy}'."
        )
    normalized_requested_gate = (
        normalized_submission_gate(requested_gate, default="always") if requested_gate else "always"
    )
    if requested_gate and normalized_requested_gate != "always":
        messages.append(
            f"[yellow]note[/yellow]: no submission limit detected; ignoring submission_gate='{requested_gate}'."
        )
    return PlanSubmissionPolicyDecision(
        submit_policy="always",
        submission_gate="always",
        messages=tuple(messages),
    )


def should_force_initial_submit(
    *,
    deliverable_mode: str,
    iteration: int,
    submit_enabled: bool,
    dry_run: bool,
    submit_policy: str | None = None,
    submission_limit_per_day: int | None = None,
) -> bool:
    _ = submit_policy, submission_limit_per_day
    return submit_enabled and (not dry_run) and deliverable_mode == "leaderboard" and iteration == 1


def quality_reasons_allow_spare_submit(reasons: list[str]) -> bool:
    if not reasons:
        return False
    return all(reason in SPARE_SUBMIT_RELAXABLE_QUALITY_REASONS for reason in reasons)


def quality_reasons_allow_initial_submit_probe(reasons: list[str]) -> bool:
    if not reasons:
        return False
    return all(reason == "selected_worse_than_detected_baseline" for reason in reasons)


def decide_quality_submit_override(
    *,
    submit_enabled: bool,
    quality_allows_submit: bool,
    force_submit: bool,
    force_initial_submit: bool,
    spare_daily_submission_slot: bool,
    quality_reasons: list[str],
    spare_reason: str = "spare_daily_submission_slot",
) -> QualitySubmitOverrideDecision:
    if not submit_enabled or quality_allows_submit or force_submit or force_initial_submit:
        return QualitySubmitOverrideDecision(quality_allows_submit=quality_allows_submit)

    if spare_daily_submission_slot and quality_reasons_allow_spare_submit(quality_reasons):
        return QualitySubmitOverrideDecision(
            quality_allows_submit=True,
            forced_submit_reason=spare_reason,
            override_reason=spare_reason,
        )

    reason_text = ", ".join(quality_reasons) if quality_reasons else "quality_guard_blocked_submit"
    return QualitySubmitOverrideDecision(
        quality_allows_submit=False,
        blocked_reason=reason_text,
    )


def latest_iteration_fallback_submit_blocked_reason(quality_reasons: list[str]) -> str | None:
    for blocked_reason in (
        "untrusted_score_source",
        "competition_metric_mismatch",
        "competition_split_mismatch",
        "competition_score_source_mismatch",
        "competition_evaluation_unfaithful",
        "missing_competitive_data",
        "external_test_label_transfer_detected",
    ):
        if blocked_reason in quality_reasons:
            return f"latest_iteration_{blocked_reason}"
    return None


def resolve_fallback_submit_blocked_reason(
    *,
    current_reason: str | None,
    best_high_potential_meta: dict[str, object] | None,
    best_high_potential_submission: object | None,
    best_submittable_submission: object | None,
) -> str | None:
    if current_reason is not None:
        return current_reason
    if not isinstance(best_high_potential_meta, dict):
        return None
    if best_high_potential_submission is None:
        return None
    if best_high_potential_submission == best_submittable_submission:
        return None
    if bool(best_high_potential_meta.get("faithful", False)) and bool(best_high_potential_meta.get("trusted", False)):
        return None
    return "higher_potential_unsubmitted_candidate_exists"


def decide_major_overhaul_policy(
    *,
    noise_forced_major_overhaul: bool,
    rank_forced_major_overhaul: bool,
    quality_forced_major_overhaul: bool,
    code_reference_forced_reproduction: bool,
    noise_limited_streak: int,
    rank_force_reason: str | None,
    quality_force_reason: str | None,
    code_reference_force_reason: str | None,
    quality_reasons: list[str],
) -> MajorOverhaulPolicyDecision:
    forced_major_overhaul_reasons: list[str] = []
    if noise_forced_major_overhaul:
        forced_major_overhaul_reasons.append(
            f"Two consecutive iterations were noise-limited: |ΔSRS| < 0.5*CV std (streak={noise_limited_streak})."
        )
    if rank_forced_major_overhaul:
        forced_major_overhaul_reasons.append(
            rank_force_reason or "Leaderboard rank indicates major improvement is still required."
        )
    if quality_forced_major_overhaul:
        forced_major_overhaul_reasons.append(
            quality_force_reason or "Quality guard requires major overhaul due to code-reference underperformance."
        )
    if code_reference_forced_reproduction:
        forced_major_overhaul_reasons.append(
            code_reference_force_reason or "Mandatory code-reference implementation is required in the next iteration."
        )

    return MajorOverhaulPolicyDecision(
        force_major_overhaul=(
            noise_forced_major_overhaul
            or rank_forced_major_overhaul
            or quality_forced_major_overhaul
            or code_reference_forced_reproduction
        ),
        forced_major_overhaul_reason=(
            " ".join(forced_major_overhaul_reasons) if forced_major_overhaul_reasons else None
        ),
        fallback_submit_blocked_reason=latest_iteration_fallback_submit_blocked_reason(quality_reasons),
    )


def decide_initial_submit_probe(
    *,
    force_initial_submit: bool,
    quality_allows_submit: bool,
    force_submit: bool,
    quality_reasons: list[str],
    allow_submit: bool,
    forced_submit_reason: str | None,
    probe_reason: str = "initial_submit_contract_probe",
) -> InitialSubmitProbeDecision:
    if not force_initial_submit:
        return InitialSubmitProbeDecision(
            force_initial_submit=False,
            quality_allows_submit=quality_allows_submit,
            allow_submit=allow_submit,
            forced_submit_reason=forced_submit_reason,
        )

    soft_probe_override = False
    if not quality_allows_submit and not force_submit:
        if quality_reasons_allow_initial_submit_probe(quality_reasons):
            quality_allows_submit = True
            soft_probe_override = True
        else:
            return InitialSubmitProbeDecision(
                force_initial_submit=False,
                quality_allows_submit=quality_allows_submit,
                allow_submit=False,
                forced_submit_reason=None,
                skipped_reason="quality_guard",
            )

    return InitialSubmitProbeDecision(
        force_initial_submit=True,
        quality_allows_submit=quality_allows_submit,
        allow_submit=True,
        forced_submit_reason=probe_reason,
        soft_probe_override=soft_probe_override,
        probe_forced=True,
    )


def decide_limited_submission_holdback(
    *,
    submit_enabled: bool,
    submission_limit_per_day: int | None,
    quality_allows_submit: bool,
    submit_improvement_allowed: bool,
    successful_submit_count: int,
    max_iterations: int,
    allow_submit: bool,
) -> LimitedSubmissionHoldbackDecision:
    if not (
        submit_enabled
        and isinstance(submission_limit_per_day, int)
        and submission_limit_per_day > 0
        and quality_allows_submit
        and submit_improvement_allowed
    ):
        return LimitedSubmissionHoldbackDecision(False)

    if allow_submit:
        return LimitedSubmissionHoldbackDecision(False)

    reserve_start = max(0, submission_limit_per_day - 1)
    if successful_submit_count >= reserve_start:
        return LimitedSubmissionHoldbackDecision(True, "reserved_final_slot")

    if max_iterations > submission_limit_per_day:
        return LimitedSubmissionHoldbackDecision(True, "strict_limited_cadence")

    return LimitedSubmissionHoldbackDecision(False)


def parse_kaggle_submission_timestamp(value: str | None) -> datetime | None:
    return parse_datetime_utc(value, formats=("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"))


def submission_row_timestamp(row: dict[str, str]) -> datetime | None:
    for key, value in row.items():
        normalized = str(key).strip().lower().replace("_", "").replace(" ", "")
        if normalized in {"date", "submissiondate", "submitted", "submittedat"}:
            return parse_kaggle_submission_timestamp(value)
    return None


def count_submission_rows_on_utc_day(
    rows: list[dict[str, str]],
    *,
    now: datetime | None = None,
) -> int:
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    count = 0
    for row in rows:
        ts = submission_row_timestamp(row)
        if ts is not None and day_start <= ts < day_end:
            count += 1
    return count


def count_submission_rows_in_recent_window(
    rows: list[dict[str, str]],
    *,
    now: datetime | None = None,
    window: timedelta = timedelta(days=1),
) -> int:
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    window_start = now_utc - window
    count = 0
    for row in rows:
        ts = submission_row_timestamp(row)
        if ts is not None and window_start <= ts <= now_utc:
            count += 1
    return count


def count_daily_competition_submissions(
    slug: str,
    *,
    dry_run: bool = False,
    fetch_submission_rows: Callable[[str, bool], list[dict[str, str]]],
    now: datetime | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> int | None:
    if dry_run:
        return 0
    try:
        rows = fetch_submission_rows(slug, dry_run)
    except Exception as exc:  # noqa: BLE001 - quota lookup must not fail a training iteration.
        if on_warning is not None:
            on_warning(f"[yellow]submit quota warning[/yellow]: could not fetch today's Kaggle submissions ({exc}).")
        return None
    now_utc = now or datetime.now(UTC)
    return max(
        count_submission_rows_on_utc_day(rows, now=now_utc),
        count_submission_rows_in_recent_window(rows, now=now_utc),
    )


def submission_count_for_daily_limit(
    *,
    slug: str,
    fallback_count: int,
    submission_limit_per_day: int | None,
    dry_run: bool = False,
    fetch_submission_rows: Callable[[str, bool], list[dict[str, str]]],
    now: datetime | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> int:
    if not isinstance(submission_limit_per_day, int) or submission_limit_per_day <= 0:
        return max(0, int(fallback_count))

    daily_count = count_daily_competition_submissions(
        slug,
        dry_run=dry_run,
        fetch_submission_rows=fetch_submission_rows,
        now=now,
        on_warning=on_warning,
    )
    if daily_count is None:
        return max(0, int(fallback_count))
    return max(0, int(daily_count))
