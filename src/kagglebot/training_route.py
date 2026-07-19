from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

VERY_HEAVY_LOCAL_TRAINING_MIN = 24 * 60

NON_TRAINING_MODES = frozenset(
    {
        "optimization",
        "pretrained_inference",
        "reference_notebook",
        "rule_based",
        "search",
        "simulation",
        "solver",
    }
)
NON_TRAINING_VALIDATION_MODES = frozenset(
    {
        "contract",
        "offline",
        "reference_reproduction",
        "smoke",
    }
)
UNSAFE_EVIDENCE_MARKERS = (
    "dummy",
    "placeholder",
    "sample submission",
    "sample_submission",
)
NON_TRAINING_SOURCE_MARKERS = (
    "KAGGLEBOT_EXECUTION_MODE",
    "non_training_submission",
    "training_performed",
    "non_training_validation_passed",
    "non_training_validation_mode",
)
NON_TRAINING_VALIDATION_REQUIRED_ISSUE = "metrics.json must declare non_training_validation_passed=true"


@dataclass(frozen=True)
class TrainingRouteDecision:
    skip_local_training: bool
    direct_notebook: bool
    reason: str
    mode: str | None = None
    validation_mode: str | None = None
    evidence: str | None = None
    estimated_local_training_min: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def decide_training_route(
    plan: Mapping[str, object],
    *,
    compute: str | None = None,
    deliverable_mode: str | None = None,
    submit_mode: str | None = None,
    code_competition: bool = False,
) -> TrainingRouteDecision:
    runtime_budget = _mapping(plan.get("runtime_budget"))
    proposal = _mapping(plan.get("non_training_submission")) or _mapping(runtime_budget.get("non_training_submission"))
    estimated_minutes = _positive_int(
        plan.get("estimated_local_training_min") or runtime_budget.get("estimated_local_training_min")
    )
    local_training_required = _explicit_bool(
        plan.get("local_training_required"),
        runtime_budget.get("local_training_required"),
    )

    if local_training_required is not False:
        return TrainingRouteDecision(
            skip_local_training=False,
            direct_notebook=False,
            reason="local_training_not_explicitly_optional",
            estimated_local_training_min=estimated_minutes,
        )
    very_heavy = bool(estimated_minutes is not None and estimated_minutes >= VERY_HEAVY_LOCAL_TRAINING_MIN)
    if not very_heavy:
        return TrainingRouteDecision(
            skip_local_training=False,
            direct_notebook=False,
            reason="local_training_not_proven_very_heavy",
            estimated_local_training_min=estimated_minutes,
        )

    mode = _token(proposal.get("mode"))
    if mode not in NON_TRAINING_MODES:
        return TrainingRouteDecision(
            skip_local_training=False,
            direct_notebook=False,
            reason="non_training_mode_missing_or_unsupported",
            mode=mode or None,
            estimated_local_training_min=estimated_minutes,
        )
    if proposal.get("implementation_ready") is not True:
        return TrainingRouteDecision(
            skip_local_training=False,
            direct_notebook=False,
            reason="non_training_implementation_not_ready",
            mode=mode,
            estimated_local_training_min=estimated_minutes,
        )

    validation_mode = _token(proposal.get("validation_mode"))
    if validation_mode not in NON_TRAINING_VALIDATION_MODES:
        return TrainingRouteDecision(
            skip_local_training=False,
            direct_notebook=False,
            reason="non_training_validation_missing_or_unsupported",
            mode=mode,
            validation_mode=validation_mode or None,
            estimated_local_training_min=estimated_minutes,
        )

    evidence = _evidence_text(proposal)
    if not evidence:
        return TrainingRouteDecision(
            skip_local_training=False,
            direct_notebook=False,
            reason="non_training_evidence_missing",
            mode=mode,
            validation_mode=validation_mode,
            estimated_local_training_min=estimated_minutes,
        )
    lowered_evidence = evidence.lower()
    if any(marker in lowered_evidence for marker in UNSAFE_EVIDENCE_MARKERS):
        return TrainingRouteDecision(
            skip_local_training=False,
            direct_notebook=False,
            reason="unsafe_sample_or_placeholder_evidence",
            mode=mode,
            validation_mode=validation_mode,
            evidence=evidence,
            estimated_local_training_min=estimated_minutes,
        )

    direct_notebook = bool(
        str(compute or "").strip().lower() == "local_gpu"
        and str(deliverable_mode or "").strip().lower() == "leaderboard"
        and str(submit_mode or "").strip().lower() == "notebook"
        and code_competition
    )
    return TrainingRouteDecision(
        skip_local_training=True,
        direct_notebook=direct_notebook,
        reason="validated_non_training_path_preferred_over_very_heavy_local_training",
        mode=mode,
        validation_mode=validation_mode,
        evidence=evidence,
        estimated_local_training_min=estimated_minutes,
    )


def plan_requests_non_training(plan: Mapping[str, object]) -> bool:
    route = _mapping(plan.get("execution_route"))
    if route.get("approved") is not True or route.get("mode") != "non_training_submission":
        return False
    return decide_training_route(plan).skip_local_training


def approved_execution_route(decision: TrainingRouteDecision) -> dict[str, object]:
    payload = decision.to_dict()
    payload.update(
        {
            "mode": "non_training_submission" if decision.skip_local_training else "train_and_validate",
            "approved": decision.skip_local_training,
        }
    )
    if decision.mode is not None:
        payload["non_training_mode"] = decision.mode
    return payload


def validate_non_training_metrics(
    metrics: Mapping[str, object],
    decision: TrainingRouteDecision,
) -> tuple[str, ...]:
    if not decision.skip_local_training:
        return ()
    issues: list[str] = []
    if metrics.get("execution_mode") != "non_training_submission":
        issues.append("metrics.json must declare execution_mode=non_training_submission")
    if metrics.get("training_performed") is not False:
        issues.append("metrics.json must declare training_performed=false")
    if metrics.get("non_training_validation_passed") is not True:
        issues.append(NON_TRAINING_VALIDATION_REQUIRED_ISSUE)
    actual_validation_mode = _token(metrics.get("non_training_validation_mode"))
    if decision.validation_mode and actual_validation_mode != decision.validation_mode:
        issues.append(
            f"metrics.json non_training_validation_mode must match the approved plan ({decision.validation_mode})"
        )
    return tuple(issues)


def resolve_non_training_validation_blockers(
    *,
    metrics: Mapping[str, object],
    route_result: Mapping[str, object],
    submission_contract: Mapping[str, object],
    decision: TrainingRouteDecision,
    canonical_submission_emitted: bool,
) -> tuple[str, ...] | None:
    """Recognize an honest, non-submittable validation-unavailable result."""
    if validate_non_training_metrics(metrics, decision) != (NON_TRAINING_VALIDATION_REQUIRED_ISSUE,):
        return None

    manifest = _mapping(route_result.get("manifest"))
    if (
        metrics.get("non_training_validation_passed") is not False
        or metrics.get("execution_status") != "validation_unavailable"
        or "primary_score" not in metrics
        or metrics.get("primary_score") is not None
        or any(metrics.get(key) is not None for key in ("offline_value", "value"))
        or manifest.get("submission_ready") is not False
        or submission_contract.get("submission_ready") is not False
        or submission_contract.get("canonical_submission_emitted") is not False
        or canonical_submission_emitted
    ):
        return None

    artifact_kind = _token(metrics.get("artifact_kind"))
    if artifact_kind and artifact_kind != "diagnostic":
        return None

    blockers = _nonempty_text_items(metrics.get("validation_blockers"))
    blockers.extend(_nonempty_text_items(metrics.get("remediation")))
    blockers.extend(_nonempty_text_items(submission_contract.get("remediation")))
    reason = metrics.get("reason")
    if isinstance(reason, str) and reason.strip():
        blockers.append(reason.strip())
    return tuple(dict.fromkeys(blockers)) or None


def is_unscored_non_training_diagnostic(
    *,
    metrics: Mapping[str, object],
    route_result: Mapping[str, object],
    submission_contract: Mapping[str, object],
    decision: TrainingRouteDecision,
    canonical_submission_emitted: bool,
) -> bool:
    """Return whether a validated skill artifact has no honest offline score."""
    if not decision.skip_local_training or validate_non_training_metrics(metrics, decision):
        return False

    manifest = _mapping(route_result.get("manifest"))
    artifact_paths = _nonempty_text_items(route_result.get("artifact_paths"))
    archive_name = manifest.get("archive")
    archive_basename = (
        archive_name.strip().replace("\\", "/").rsplit("/", 1)[-1]
        if isinstance(archive_name, str) and archive_name.strip()
        else ""
    )
    if (
        route_result.get("mode") != "skill_artifact"
        or route_result.get("validation_status") != "success"
        or not archive_basename
        or archive_basename != archive_name
        or archive_basename.lower() == "submission.zip"
        or manifest.get("submission_ready") is not False
        or not artifact_paths
        or any(path.replace("\\", "/").rsplit("/", 1)[-1].lower() == "submission.zip" for path in artifact_paths)
        or canonical_submission_emitted
    ):
        return False

    if submission_contract and (
        submission_contract.get("archive_name") != archive_basename
        or submission_contract.get("submission_ready") is not False
        or submission_contract.get("canonical_submission_emitted") is not False
        or submission_contract.get("official_paired_validation_passed") is not False
    ):
        return False

    return bool(
        metrics.get("execution_mode") == "non_training_submission"
        and metrics.get("training_performed") is False
        and metrics.get("execution_status") == "success"
        and metrics.get("non_training_validation_passed") is True
        and metrics.get("offline_artifact_validation_passed") is True
        and metrics.get("official_paired_validation_passed") is False
        and metrics.get("submission_ready") is False
        and "primary_score" in metrics
        and metrics.get("primary_score") is None
        and all(metrics.get(key) is None for key in ("offline_value", "value"))
    )


def validate_non_training_source(source: str) -> tuple[str, ...]:
    missing = tuple(marker for marker in NON_TRAINING_SOURCE_MARKERS if marker not in source)
    if not missing:
        return ()
    return (
        "kernel.py does not implement the approved non-training runtime contract; "
        f"missing markers: {', '.join(missing)}",
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _explicit_bool(*values: object) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", " ").replace(" ", "_")


def _evidence_text(proposal: Mapping[str, object]) -> str:
    for key in ("source", "evidence", "implementation", "model_source", "reference"):
        value = proposal.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _nonempty_text_items(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
