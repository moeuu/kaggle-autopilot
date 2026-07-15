from __future__ import annotations

from dataclasses import dataclass

from kagglebot.scalar_utils import parse_finite_float, parse_int

BOTTOM_DECILE_PERCENTILE = 0.90
BOTTOM_TWO_PERCENT_PERCENTILE = 0.98
MIN_OBSERVED_TEAMS = 20
MIN_ESTIMATED_TEAMS = 50
SCORE_COLLAPSE_RATIO = 0.02


@dataclass(frozen=True)
class LeaderboardAnomalyAssessment:
    severity: str
    confidence: str
    signals: tuple[str, ...]
    evidence: dict[str, object]
    note: str

    def to_payload(self) -> dict[str, object]:
        return {
            "suspected": True,
            "implementation_audit_required": True,
            "severity": self.severity,
            "confidence": self.confidence,
            "signals": list(self.signals),
            "evidence": self.evidence,
            "note": self.note,
        }


def assess_leaderboard_anomaly(
    *,
    direction: str,
    online_score: object,
    offline_score: object = None,
    top1_score: object = None,
    rank: object = None,
    total_teams: object = None,
    rank_percentile: object = None,
    estimated_rank: object = None,
    estimated_total_teams: object = None,
    estimated_rank_percentile: object = None,
) -> LeaderboardAnomalyAssessment | None:
    """Detect outcomes so poor that implementation fidelity should be audited first.

    Observed rank is considered stronger evidence than a score-derived rank estimate.
    Estimated bottom rank therefore needs an independent score-collapse signal. This
    keeps an ordinary weak baseline from causing a repository repair loop while still
    treating last-place-like results and zero-score collapses as implementation bugs.
    """
    online = parse_finite_float(online_score, allow_commas=True)
    if online is None:
        return None
    offline = parse_finite_float(offline_score, allow_commas=True)
    top1 = parse_finite_float(top1_score, allow_commas=True)
    observed_rank = parse_int(rank, allow_float=True)
    observed_total = parse_int(total_teams, allow_float=True)
    observed_percentile = _resolve_percentile(
        percentile=rank_percentile,
        rank=observed_rank,
        total=observed_total,
    )
    estimate_rank = parse_int(estimated_rank, allow_float=True)
    estimate_total = parse_int(estimated_total_teams, allow_float=True)
    estimate_percentile = _resolve_percentile(
        percentile=estimated_rank_percentile,
        rank=estimate_rank,
        total=estimate_total,
    )

    signals: list[str] = []
    observed_bottom = bool(
        observed_percentile is not None
        and observed_total is not None
        and observed_total >= MIN_OBSERVED_TEAMS
        and observed_percentile >= BOTTOM_DECILE_PERCENTILE
    )
    observed_critical = bool(
        observed_bottom and observed_percentile is not None and observed_percentile >= BOTTOM_TWO_PERCENT_PERCENTILE
    )
    estimated_bottom = bool(
        estimate_percentile is not None
        and estimate_total is not None
        and estimate_total >= MIN_ESTIMATED_TEAMS
        and estimate_percentile >= BOTTOM_TWO_PERCENT_PERCENTILE
    )
    if observed_critical:
        signals.append("observed_bottom_two_percent")
    elif observed_bottom:
        signals.append("observed_bottom_decile")
    if estimated_bottom:
        signals.append("estimated_bottom_two_percent")

    direction_normalized = str(direction or "maximize").strip().lower()
    score_signals = _score_collapse_signals(
        direction=direction_normalized,
        online=online,
        offline=offline,
        top1=top1,
    )
    signals.extend(score_signals)

    extreme_score_collapse = "online_score_collapse_vs_top1" in score_signals
    suspected = observed_bottom or extreme_score_collapse or (estimated_bottom and bool(score_signals))
    if not suspected:
        return None

    severity = "critical" if observed_critical or (observed_bottom and extreme_score_collapse) else "high"
    confidence = "high" if observed_bottom or extreme_score_collapse else "medium"
    evidence: dict[str, object] = {
        "direction": direction_normalized,
        "online_score": online,
        "offline_score": offline,
        "top1_score": top1,
        "rank": observed_rank,
        "total_teams": observed_total,
        "rank_percentile": observed_percentile,
        "estimated_rank": estimate_rank,
        "estimated_total_teams": estimate_total,
        "estimated_rank_percentile": estimate_percentile,
        "thresholds": {
            "bottom_decile_percentile": BOTTOM_DECILE_PERCENTILE,
            "bottom_two_percent_percentile": BOTTOM_TWO_PERCENT_PERCENTILE,
            "score_collapse_ratio": SCORE_COLLAPSE_RATIO,
        },
    }
    signal_text = ", ".join(signals)
    note = (
        "Leaderboard outcome is in an implementation-anomaly band "
        f"(severity={severity}, signals={signal_text}). Stop model-only tuning and audit the executed submission path: "
        "hidden-test input discovery, model/reference asset loading, runtime fallback activation, prediction variance, "
        "ID/order alignment, expected filename/schema, metric scale/direction, and the exact notebook output selected. "
        "Do not resubmit the same artifact hash; require runtime-fidelity evidence before the next submission."
    )
    return LeaderboardAnomalyAssessment(
        severity=severity,
        confidence=confidence,
        signals=tuple(dict.fromkeys(signals)),
        evidence=evidence,
        note=note,
    )


def _resolve_percentile(*, percentile: object, rank: int | None, total: int | None) -> float | None:
    parsed = parse_finite_float(percentile)
    if parsed is not None and 0.0 <= parsed <= 1.0:
        return parsed
    if rank is None or total is None or rank <= 0 or total <= 0:
        return None
    return min(1.0, rank / total)


def _score_collapse_signals(
    *,
    direction: str,
    online: float,
    offline: float | None,
    top1: float | None,
) -> list[str]:
    signals: list[str] = []
    if direction == "minimize":
        if top1 is not None and top1 > 0 and online >= top1 / SCORE_COLLAPSE_RATIO:
            signals.append("online_score_collapse_vs_top1")
        if offline is not None and offline > 0 and online >= offline / SCORE_COLLAPSE_RATIO:
            signals.append("offline_online_scale_or_output_collapse")
        return signals

    if top1 is not None and top1 > 0 and online <= top1 * SCORE_COLLAPSE_RATIO:
        signals.append("online_score_collapse_vs_top1")
    if offline is not None and offline > 0 and online <= offline * SCORE_COLLAPSE_RATIO:
        signals.append("offline_online_scale_or_output_collapse")
    return signals
