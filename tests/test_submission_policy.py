from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kagglebot.submission_policy import (
    count_daily_competition_submissions,
    count_submission_rows_in_recent_window,
    count_submission_rows_on_utc_day,
    decide_initial_submit_probe,
    decide_limited_submission_holdback,
    decide_major_overhaul_policy,
    decide_quality_submit_override,
    has_spare_daily_submission_slot,
    is_top1_tier,
    latest_iteration_fallback_submit_blocked_reason,
    meets_target,
    non_final_submission_checkpoints,
    normalize_watch_submit_policy,
    normalized_submission_gate,
    normalized_submit_policy,
    parse_kaggle_submission_timestamp,
    quality_reasons_allow_initial_submit_probe,
    quality_reasons_allow_spare_submit,
    resolve_fallback_submit_blocked_reason,
    resolve_plan_submission_policy,
    should_attempt_submit_for_readiness,
    should_force_initial_submit,
    submission_count_for_daily_limit,
    submission_gate_for_policy,
)


def test_target_and_top1_tier_direction_checks() -> None:
    assert meets_target(0.4, 0.5, "minimize") is True
    assert meets_target(0.6, 0.5, "minimize") is False
    assert meets_target(0.9, 0.8, "maximize") is True
    assert is_top1_tier(0.4, 0.5, "minimize") is True
    assert is_top1_tier(0.7, 0.8, "maximize") is False
    assert is_top1_tier(0.7, None, "maximize") is False


def test_submission_policy_normalization() -> None:
    assert normalized_submit_policy("improved_only") == "improved"
    assert normalized_submit_policy("target_or_final") == "readiness_or_final"
    assert normalized_submit_policy("unknown") == "always"
    assert normalized_submission_gate("on_target_only", default="always") == "readiness_only"
    assert normalized_submission_gate("unknown", default="final_only") == "final_only"
    assert submission_gate_for_policy("improved") == "always"
    assert submission_gate_for_policy("final_only") == "final_only"


def test_watch_submit_policy_normalization_is_strict() -> None:
    assert normalize_watch_submit_policy(" improved_only ") == "improved"
    assert normalize_watch_submit_policy("no-submit") == "none"
    assert normalize_watch_submit_policy("off") == "none"

    with pytest.raises(ValueError, match="improved, none"):
        normalize_watch_submit_policy("always")


def test_resolve_plan_submission_policy_honors_forced_improved_policy() -> None:
    decision = resolve_plan_submission_policy(
        config_submit_policy="improved_only",
        requested_submit_policy="always",
        requested_submission_gate="readiness_only",
        submission_limit_detected=True,
        default_limited_submission_gate="readiness_or_final",
    )

    assert decision.submit_policy == "improved"
    assert decision.submission_gate == "always"
    assert decision.messages == ()


def test_resolve_plan_submission_policy_defaults_limited_rules_to_readiness_or_final() -> None:
    decision = resolve_plan_submission_policy(
        config_submit_policy=None,
        requested_submit_policy="always",
        requested_submission_gate=None,
        submission_limit_detected=True,
        default_limited_submission_gate="readiness_or_final",
    )

    assert decision.submit_policy == "readiness_or_final"
    assert decision.submission_gate == "readiness_or_final"
    assert decision.messages == (
        "[yellow]note[/yellow]: submission limit detected in rules; defaulting submission_gate=readiness_or_final.",
    )


def test_resolve_plan_submission_policy_keeps_explicit_limited_gate() -> None:
    decision = resolve_plan_submission_policy(
        config_submit_policy=None,
        requested_submit_policy="readiness_only",
        requested_submission_gate="on_target_only",
        submission_limit_detected=True,
        default_limited_submission_gate="readiness_or_final",
    )

    assert decision.submit_policy == "readiness_only"
    assert decision.submission_gate == "readiness_only"
    assert decision.messages == ()


def test_resolve_plan_submission_policy_ignores_limited_options_without_limits() -> None:
    decision = resolve_plan_submission_policy(
        config_submit_policy=None,
        requested_submit_policy="readiness_only",
        requested_submission_gate="final_only",
        submission_limit_detected=False,
        default_limited_submission_gate="readiness_or_final",
    )

    assert decision.submit_policy == "always"
    assert decision.submission_gate == "always"
    assert decision.messages == (
        "[yellow]note[/yellow]: no submission limit detected; ignoring submit_policy='readiness_only'.",
        "[yellow]note[/yellow]: no submission limit detected; ignoring submission_gate='final_only'.",
    )


def test_spare_slot_and_checkpoint_policy() -> None:
    assert has_spare_daily_submission_slot(
        submission_limit_per_day=5,
        submissions_used_today=2,
        iteration=3,
        max_iterations=5,
    )
    assert not has_spare_daily_submission_slot(
        submission_limit_per_day=5,
        submissions_used_today=3,
        iteration=3,
        max_iterations=5,
    )
    assert non_final_submission_checkpoints(max_iterations=10, non_final_slots=4) == {2, 4, 6, 8}


def test_should_attempt_submit_for_readiness_reserves_final_slot() -> None:
    assert should_attempt_submit_for_readiness(
        gate="readiness_or_final",
        readiness_score=0.10,
        readiness_target=0.90,
        direction="maximize",
        iteration=1,
        max_iterations=3,
        submission_limit_per_day=3,
        successful_submissions=0,
        top1_score=0.80,
    )
    assert not should_attempt_submit_for_readiness(
        gate="readiness_or_final",
        readiness_score=0.20,
        readiness_target=0.90,
        direction="maximize",
        iteration=2,
        max_iterations=3,
        submission_limit_per_day=3,
        successful_submissions=2,
        top1_score=0.80,
    )
    assert should_attempt_submit_for_readiness(
        gate="readiness_or_final",
        readiness_score=0.85,
        readiness_target=0.90,
        direction="maximize",
        iteration=2,
        max_iterations=3,
        submission_limit_per_day=3,
        successful_submissions=2,
        top1_score=0.80,
    )


def test_should_force_initial_submit_only_for_real_leaderboard_runs() -> None:
    assert should_force_initial_submit(
        deliverable_mode="leaderboard",
        iteration=1,
        submit_enabled=True,
        dry_run=False,
        submit_policy="improved",
        submission_limit_per_day=1,
    )
    assert not should_force_initial_submit(
        deliverable_mode="writeup",
        iteration=1,
        submit_enabled=True,
        dry_run=False,
    )
    assert not should_force_initial_submit(
        deliverable_mode="leaderboard",
        iteration=1,
        submit_enabled=True,
        dry_run=True,
    )


def test_quality_reason_soft_overrides_are_narrow() -> None:
    assert quality_reasons_allow_spare_submit(["selected_worse_than_detected_baseline"])
    assert quality_reasons_allow_spare_submit(
        ["selected_worse_than_detected_baseline", "below_code_reference_baseline"]
    )
    assert not quality_reasons_allow_spare_submit(["external_test_label_transfer_detected"])
    assert not quality_reasons_allow_spare_submit(["selected_worse_than_detected_baseline", "untrusted_score_source"])

    assert quality_reasons_allow_initial_submit_probe(["selected_worse_than_detected_baseline"])
    assert not quality_reasons_allow_initial_submit_probe(["below_code_reference_baseline"])
    assert not quality_reasons_allow_initial_submit_probe(
        ["selected_worse_than_detected_baseline", "untrusted_score_source"]
    )


def test_decide_quality_submit_override_promotes_soft_blocks_with_spare_slot() -> None:
    decision = decide_quality_submit_override(
        submit_enabled=True,
        quality_allows_submit=False,
        force_submit=False,
        force_initial_submit=False,
        spare_daily_submission_slot=True,
        quality_reasons=["selected_worse_than_detected_baseline"],
    )

    assert decision.quality_allows_submit is True
    assert decision.forced_submit_reason == "spare_daily_submission_slot"
    assert decision.override_reason == "spare_daily_submission_slot"
    assert decision.blocked_reason is None


def test_decide_quality_submit_override_reports_hard_block_reason() -> None:
    decision = decide_quality_submit_override(
        submit_enabled=True,
        quality_allows_submit=False,
        force_submit=False,
        force_initial_submit=False,
        spare_daily_submission_slot=True,
        quality_reasons=["untrusted_score_source"],
    )

    assert decision.quality_allows_submit is False
    assert decision.forced_submit_reason is None
    assert decision.override_reason is None
    assert decision.blocked_reason == "untrusted_score_source"


def test_decide_quality_submit_override_leaves_forced_paths_unchanged() -> None:
    decision = decide_quality_submit_override(
        submit_enabled=True,
        quality_allows_submit=False,
        force_submit=True,
        force_initial_submit=False,
        spare_daily_submission_slot=True,
        quality_reasons=["untrusted_score_source"],
    )

    assert decision.quality_allows_submit is False
    assert decision.blocked_reason is None


def test_latest_iteration_fallback_submit_blocked_reason_prefers_hard_quality_blocks() -> None:
    assert (
        latest_iteration_fallback_submit_blocked_reason(
            ["selected_worse_than_detected_baseline", "external_test_label_transfer_detected"]
        )
        == "latest_iteration_external_test_label_transfer_detected"
    )
    assert latest_iteration_fallback_submit_blocked_reason(["selected_worse_than_detected_baseline"]) is None


def test_resolve_fallback_submit_blocked_reason_preserves_existing_reason() -> None:
    assert (
        resolve_fallback_submit_blocked_reason(
            current_reason="latest_iteration_competition_metric_mismatch",
            best_high_potential_meta={"faithful": False, "trusted": False},
            best_high_potential_submission="iter-2/submission.csv",
            best_submittable_submission="iter-1/submission.csv",
        )
        == "latest_iteration_competition_metric_mismatch"
    )


def test_resolve_fallback_submit_blocked_reason_blocks_untrusted_high_potential_candidate() -> None:
    assert (
        resolve_fallback_submit_blocked_reason(
            current_reason=None,
            best_high_potential_meta={"faithful": True, "trusted": False},
            best_high_potential_submission="iter-2/submission.csv",
            best_submittable_submission="iter-1/submission.csv",
        )
        == "higher_potential_unsubmitted_candidate_exists"
    )


def test_resolve_fallback_submit_blocked_reason_allows_trusted_or_same_candidate() -> None:
    assert (
        resolve_fallback_submit_blocked_reason(
            current_reason=None,
            best_high_potential_meta={"faithful": True, "trusted": True},
            best_high_potential_submission="iter-2/submission.csv",
            best_submittable_submission="iter-1/submission.csv",
        )
        is None
    )
    assert (
        resolve_fallback_submit_blocked_reason(
            current_reason=None,
            best_high_potential_meta={"faithful": False, "trusted": False},
            best_high_potential_submission="iter-1/submission.csv",
            best_submittable_submission="iter-1/submission.csv",
        )
        is None
    )


def test_decide_major_overhaul_policy_collects_reasons_and_fallback_blocker() -> None:
    decision = decide_major_overhaul_policy(
        noise_forced_major_overhaul=True,
        rank_forced_major_overhaul=True,
        quality_forced_major_overhaul=True,
        code_reference_forced_reproduction=True,
        noise_limited_streak=2,
        rank_force_reason="rank is poor",
        quality_force_reason=None,
        code_reference_force_reason="code reference required",
        quality_reasons=["competition_metric_mismatch"],
    )

    assert decision.force_major_overhaul is True
    assert decision.fallback_submit_blocked_reason == "latest_iteration_competition_metric_mismatch"
    assert decision.forced_major_overhaul_reason == (
        "Two consecutive iterations were noise-limited: |ΔSRS| < 0.5*CV std (streak=2). "
        "rank is poor "
        "Quality guard requires major overhaul due to code-reference underperformance. "
        "code reference required"
    )


def test_decide_major_overhaul_policy_noops_without_signals() -> None:
    decision = decide_major_overhaul_policy(
        noise_forced_major_overhaul=False,
        rank_forced_major_overhaul=False,
        quality_forced_major_overhaul=False,
        code_reference_forced_reproduction=False,
        noise_limited_streak=0,
        rank_force_reason=None,
        quality_force_reason=None,
        code_reference_force_reason=None,
        quality_reasons=[],
    )

    assert decision.force_major_overhaul is False
    assert decision.forced_major_overhaul_reason is None
    assert decision.fallback_submit_blocked_reason is None


def test_decide_initial_submit_probe_allows_soft_baseline_probe() -> None:
    decision = decide_initial_submit_probe(
        force_initial_submit=True,
        quality_allows_submit=False,
        force_submit=False,
        quality_reasons=["selected_worse_than_detected_baseline"],
        allow_submit=False,
        forced_submit_reason=None,
    )

    assert decision.force_initial_submit is True
    assert decision.quality_allows_submit is True
    assert decision.allow_submit is True
    assert decision.forced_submit_reason == "initial_submit_contract_probe"
    assert decision.soft_probe_override is True
    assert decision.probe_forced is True
    assert decision.skipped_reason is None


def test_decide_initial_submit_probe_skips_hard_quality_failures() -> None:
    decision = decide_initial_submit_probe(
        force_initial_submit=True,
        quality_allows_submit=False,
        force_submit=False,
        quality_reasons=["untrusted_score_source"],
        allow_submit=True,
        forced_submit_reason="spare_daily_submission_slot",
    )

    assert decision.force_initial_submit is False
    assert decision.quality_allows_submit is False
    assert decision.allow_submit is False
    assert decision.forced_submit_reason is None
    assert decision.skipped_reason == "quality_guard"


def test_decide_initial_submit_probe_preserves_non_initial_state() -> None:
    decision = decide_initial_submit_probe(
        force_initial_submit=False,
        quality_allows_submit=False,
        force_submit=False,
        quality_reasons=["untrusted_score_source"],
        allow_submit=False,
        forced_submit_reason="existing",
    )

    assert decision.force_initial_submit is False
    assert decision.quality_allows_submit is False
    assert decision.allow_submit is False
    assert decision.forced_submit_reason == "existing"
    assert decision.probe_forced is False


def test_decide_limited_submission_holdback_reserves_final_slot() -> None:
    decision = decide_limited_submission_holdback(
        submit_enabled=True,
        submission_limit_per_day=3,
        quality_allows_submit=True,
        submit_improvement_allowed=True,
        successful_submit_count=2,
        max_iterations=5,
        allow_submit=False,
    )

    assert decision.holdback is True
    assert decision.reason == "reserved_final_slot"


def test_decide_limited_submission_holdback_reports_strict_cadence() -> None:
    decision = decide_limited_submission_holdback(
        submit_enabled=True,
        submission_limit_per_day=5,
        quality_allows_submit=True,
        submit_improvement_allowed=True,
        successful_submit_count=1,
        max_iterations=10,
        allow_submit=False,
    )

    assert decision.holdback is True
    assert decision.reason == "strict_limited_cadence"


def test_decide_limited_submission_holdback_ignores_allowed_or_blocked_paths() -> None:
    assert not decide_limited_submission_holdback(
        submit_enabled=True,
        submission_limit_per_day=3,
        quality_allows_submit=True,
        submit_improvement_allowed=True,
        successful_submit_count=0,
        max_iterations=5,
        allow_submit=True,
    ).holdback
    assert not decide_limited_submission_holdback(
        submit_enabled=True,
        submission_limit_per_day=3,
        quality_allows_submit=False,
        submit_improvement_allowed=True,
        successful_submit_count=2,
        max_iterations=5,
        allow_submit=False,
    ).holdback


def test_submission_row_counts_use_kaggle_cli_dates() -> None:
    rows = [
        {"date": "2026-05-09 06:16:21.527000", "status": "COMPLETE"},
        {"date": "2026-05-09T23:59:59+00:00", "status": "ERROR"},
        {"date": "2026-05-08 22:44:27.263000", "status": "COMPLETE"},
        {"date": "not-a-date", "status": "COMPLETE"},
    ]

    assert count_submission_rows_on_utc_day(rows, now=datetime(2026, 5, 9, 16, tzinfo=UTC)) == 2
    assert count_submission_rows_in_recent_window(rows, now=datetime(2026, 5, 9, 16, tzinfo=UTC)) == 2


def test_parse_kaggle_submission_timestamp_uses_shared_iso_policy_and_cli_fallbacks() -> None:
    assert parse_kaggle_submission_timestamp("2026-05-09T23:59:59Z") == datetime(2026, 5, 9, 23, 59, 59, tzinfo=UTC)
    assert parse_kaggle_submission_timestamp("2026-05-10T08:59:59+09:00") == datetime(
        2026, 5, 9, 23, 59, 59, tzinfo=UTC
    )
    assert parse_kaggle_submission_timestamp("2026-05-09 06:16:21.527000") == datetime(
        2026, 5, 9, 6, 16, 21, 527000, tzinfo=UTC
    )
    assert parse_kaggle_submission_timestamp("2026-05-09 06:16:21 UTC") == datetime(2026, 5, 9, 6, 16, 21, tzinfo=UTC)


def test_count_daily_competition_submissions_uses_max_of_utc_day_and_recent_window() -> None:
    rows = [
        {"date": "2026-05-09 06:16:21.527000", "status": "COMPLETE"},
        {"date": "2026-05-08 22:44:27.263000", "status": "COMPLETE"},
        {"date": "not-a-date", "status": "COMPLETE"},
    ]

    count = count_daily_competition_submissions(
        "demo",
        fetch_submission_rows=lambda slug, dry_run: rows,
        now=datetime(2026, 5, 9, 16, tzinfo=UTC),
    )

    assert count == 2


def test_submission_count_for_daily_limit_falls_back_without_limit_or_fetch() -> None:
    assert (
        submission_count_for_daily_limit(
            slug="demo",
            fallback_count=3,
            submission_limit_per_day=None,
            fetch_submission_rows=lambda slug, dry_run: [],
        )
        == 3
    )

    warnings: list[str] = []

    def fail_fetch(slug: str, dry_run: bool) -> list[dict[str, str]]:
        raise RuntimeError(f"boom {slug} {dry_run}")

    assert (
        submission_count_for_daily_limit(
            slug="demo",
            fallback_count=2,
            submission_limit_per_day=5,
            fetch_submission_rows=fail_fetch,
            on_warning=warnings.append,
        )
        == 2
    )
    assert warnings
