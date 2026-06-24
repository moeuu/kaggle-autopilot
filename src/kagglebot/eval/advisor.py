from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.agents.strategy_runner import StrategyResult, run_strategy
from kagglebot.competition_policy import load_competition_policy
from kagglebot.eval.core import MetricRegistry
from kagglebot.exec_utils import run_command
from kagglebot.json_utils import load_json_object_or_empty, write_json_object
from kagglebot.medals import (
    MEDAL_TARGET_PERCENTILES,
    TARGET_MEDAL_ERROR,
    TARGET_MEDAL_SCHEMA,
    normalize_target_medal,
    validate_target_rank_percentile,
)
from kagglebot.paths import CompetitionPaths
from kagglebot.solver.metrics import infer_direction
from kagglebot.writeup import (
    infer_deliverable_mode,
    infer_submit_mode,
    normalize_deliverable_mode,
    normalize_submit_mode,
)

SUPPORTED_SPLIT_STRATEGIES = {
    "kfold",
    "stratified_kfold",
    "group_kfold",
    "timeseries_split",
}
SUPPORTED_CI_METHODS = {"normal", "bootstrap"}
SUPPORTED_READINESS_METHODS = {"ci_bound", "mean_std"}
SUPPORTED_SUBMISSION_GATES = {
    "always",
    "each_iteration",
    "final_only",
    "at_final",
    "readiness_only",
    "readiness_target",
    "on_target_only",
    "readiness_or_final",
    "target_or_final",
}


class EvaluationAdvisor:
    """Optional GPT-5.5 + web-search advisor for evaluation spec selection."""

    def __init__(
        self,
        *,
        paths: CompetitionPaths,
        slug: str,
        dry_run: bool = False,
        force: bool = False,
        max_retries: int = 2,
        strategy_runner: Callable[[Path, Path], StrategyResult] | None = None,
        search_capability_check: Callable[[], bool] | None = None,
    ) -> None:
        self.paths = paths
        self.slug = slug
        self.dry_run = dry_run
        self.force = force
        self.max_retries = max(0, int(max_retries))
        self._strategy_runner = strategy_runner or _run_strategy_default
        self._search_capability_check = search_capability_check or _supports_live_search

    @property
    def spec_path(self) -> Path:
        return self.paths.context_dir / "evaluation_spec.json"

    @property
    def log_dir(self) -> Path:
        return self.paths.context_dir / "eval_advisor"

    def ensure_spec(self) -> tuple[dict[str, object], str]:
        self.paths.context_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        errors: list[str] = []
        if self.spec_path.exists() and not self.force:
            frozen = load_json_object_or_empty(self.spec_path)
            stale_reason = _stale_frozen_spec_reason(frozen=frozen, paths=self.paths)
            if stale_reason is None:
                self._write_status(source="frozen", attempts=0, errors=[], notes="existing frozen spec reused")
                return frozen, "frozen"
            errors.append(f"stale frozen spec detected: {stale_reason}")

        if self.dry_run or not self._search_capability_check():
            fallback = self._fallback_spec(reason="advisor_unavailable")
            fallback_errors = [*errors, "advisor unavailable"]
            self._persist_final_spec(fallback, source="fallback", attempts=0, errors=fallback_errors)
            return fallback, "fallback"

        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            prompt_text = self._build_prompt(previous_errors=errors)
            prompt_path = self.log_dir / f"prompt_attempt_{attempt}.md"
            prompt_path.write_text(prompt_text, encoding="utf-8")

            attempt_dir = self.log_dir / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = self._strategy_runner(prompt_path, attempt_dir)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"attempt {attempt}: strategy runner failed ({exc})")
                continue

            response_text = result.stdout.strip()
            response_path = attempt_dir / "advisor_response.txt"
            response_payload = response_text + ("\n" if response_text else "")
            response_path.write_text(response_payload, encoding="utf-8")
            payload = _parse_json_response(response_text)
            if payload is None:
                errors.append(f"attempt {attempt}: response is not valid JSON")
                continue

            spec, sources_summary, queries, payload_errors = validate_advisor_payload(payload)
            if payload_errors:
                compact = "; ".join(payload_errors)
                errors.append(f"attempt {attempt}: {compact}")
                continue

            assert spec is not None
            assert sources_summary is not None
            assert queries is not None
            write_json_object(attempt_dir / "validated_payload.json", payload)
            self._persist_sources_summary(sources_summary=sources_summary, queries=queries)
            self._persist_final_spec(spec, source="advisor", attempts=attempt, errors=errors)
            return spec, "advisor"

        fallback = self._fallback_spec(reason="advisor_invalid_output")
        self._persist_final_spec(fallback, source="fallback", attempts=total_attempts, errors=errors)
        return fallback, "fallback"

    def _build_prompt(self, *, previous_errors: list[str]) -> str:
        rules = _read_trimmed(self.paths.rules_md_path, limit_chars=6000)
        overview = _read_trimmed(self.paths.overview_md_path, limit_chars=6000)
        submission_format = _read_trimmed(self.paths.submission_format_md_path, limit_chars=4000)
        dataset_profile = _read_trimmed(self.paths.dataset_profile_path, limit_chars=6000)
        sample_head = _read_sample_head(self.paths, limit_lines=40)
        schema_text = _advisor_response_schema_text()
        error_block = ""
        if previous_errors:
            joined = "\n".join(f"- {item}" for item in previous_errors[-5:])
            error_block = f"\nPrevious output issues to fix:\n{joined}\n"
        return (
            "You are an Evaluation Advisor for Kaggle autopilot.\n"
            "Use LIVE web search aggressively and cite your findings in sources_summary_md.\n"
            "You MUST choose only from the supported metric/split options listed below.\n"
            "Return JSON only. No markdown fences. No extra keys.\n"
            f"{error_block}\n"
            f"Competition slug: {self.slug}\n\n"
            "Supported metrics:\n"
            "- auc\n"
            "- logloss\n"
            "- brier_score\n"
            "- accuracy\n"
            "- f1\n"
            "- rmse\n"
            "- mae\n"
            "- rmsle\n"
            "- mape\n"
            "- smape\n"
            "- pearson\n"
            "- spearman\n\n"
            "Supported split strategies:\n"
            "- kfold\n"
            "- stratified_kfold\n"
            "- group_kfold\n"
            "- timeseries_split\n\n"
            "Context (rules):\n<<<\n"
            f"{rules}\n"
            ">>>\n\nContext (overview):\n<<<\n"
            f"{overview}\n"
            ">>>\n\nContext (submission format):\n<<<\n"
            f"{submission_format}\n"
            ">>>\n\nContext (dataset profile json):\n<<<\n"
            f"{dataset_profile}\n"
            ">>>\n\nContext (sample submission head):\n<<<\n"
            f"{sample_head}\n"
            ">>>\n\n"
            "Task:\n"
            "1) Determine official metric and direction from competition material and web research.\n"
            "2) Determine whether the competition is a leaderboard competition or a judged/writeup competition.\n"
            "3) Determine whether submissions are uploaded as files or must be submitted through notebooks.\n"
            "4) Choose split strategy and evaluation parameters compatible with the supported options.\n"
            "5) Set submission_gate policy; use non-always gates only when rules mention submission-count limits.\n"
            "6) Include search queries used and concise source summary.\n\n"
            f"Output schema:\n{schema_text}\n"
        )

    def _fallback_spec(self, *, reason: str) -> dict[str, object]:
        profile = load_json_object_or_empty(self.paths.dataset_profile_path)
        competition_policy = load_competition_policy(self.paths)
        context_text = "\n".join(
            [
                _read_trimmed(self.paths.rules_md_path, limit_chars=4000),
                _read_trimmed(self.paths.overview_md_path, limit_chars=4000),
                _read_trimmed(self.paths.submission_format_md_path, limit_chars=4000),
            ]
        ).lower()
        metric = _infer_metric_from_context(profile=profile, context_text=context_text)
        direction = infer_direction(metric, "auto")
        split_strategy = _infer_split_strategy(profile=profile, context_text=context_text)
        top1_score = _load_top1_score(self.paths)

        readiness_rule: dict[str, object] = {
            "method": "ci_bound",
            "k": 1.0,
            "target_score": top1_score,
            "submission_gate": "always",
        }
        deliverable_mode = infer_deliverable_mode(context_text)
        submit_mode = infer_submit_mode(context_text)
        target_medal = "winner" if deliverable_mode == "leaderboard" else None
        target_rank_percentile = MEDAL_TARGET_PERCENTILES.get(target_medal) if target_medal else None
        spec = {
            "deliverable_mode": deliverable_mode,
            "submit_mode": submit_mode,
            "target_medal": target_medal,
            "target_rank_percentile": target_rank_percentile,
            "metric_name": metric,
            "direction": direction,
            "split_strategy": split_strategy,
            "n_splits": 5,
            "seeds": [42],
            "repeats": 1,
            "ci_method": "normal",
            "ci_alpha": 0.05,
            "readiness_rule": readiness_rule,
            "drift_check": {"enabled": False, "drift_weight": 1.0},
            "stop_policy": {
                "min_delta": 0.0,
                "no_improve_patience": 2,
                "same_config_patience": 2,
            },
        }
        sources_summary = (
            "# Advisor Fallback Sources\n\n"
            f"- local://{self.paths.rules_md_path.name} (reason: {reason})\n"
            f"- local://{self.paths.overview_md_path.name} (reason: {reason})\n"
            f"- local://{self.paths.dataset_profile_path.name} (reason: {reason})\n"
        )
        spec = _apply_fallback_policy_overrides(spec, competition_policy)
        self._persist_sources_summary(sources_summary=sources_summary, queries=[])
        return spec

    def _persist_sources_summary(self, *, sources_summary: str, queries: list[str]) -> None:
        summary_path = self.log_dir / "sources_summary.md"
        summary_path.write_text((sources_summary.strip() + "\n") if sources_summary.strip() else "", encoding="utf-8")
        write_json_object(self.log_dir / "search_queries.json", {"queries": queries})

    def _persist_final_spec(self, spec: dict[str, object], *, source: str, attempts: int, errors: list[str]) -> None:
        write_json_object(self.spec_path, spec)
        write_json_object(self.log_dir / "chosen_evaluation_spec.json", spec)
        self._write_status(source=source, attempts=attempts, errors=errors, notes="spec written")

    def _write_status(self, *, source: str, attempts: int, errors: list[str], notes: str) -> None:
        status = {
            "source": source,
            "attempts": attempts,
            "errors": errors,
            "notes": notes,
            "updated_at": datetime.now(UTC).isoformat(),
            "spec_path": str(self.spec_path),
        }
        write_json_object(self.log_dir / "status.json", status)


def validate_advisor_payload(
    payload: object,
) -> tuple[dict[str, object] | None, str | None, list[str] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(payload, dict):
        return None, None, None, ["payload must be a JSON object"]

    expected_top = {"evaluation_spec", "sources_summary_md", "search_queries"}
    issues.extend(_validate_exact_keys(payload, expected_top, path="payload"))

    spec_raw = payload.get("evaluation_spec")
    sources_summary = payload.get("sources_summary_md")
    queries = payload.get("search_queries")

    if not isinstance(sources_summary, str) or not sources_summary.strip():
        issues.append("payload.sources_summary_md must be a non-empty string")
    if not isinstance(queries, list) or not all(isinstance(item, str) and item.strip() for item in queries):
        issues.append("payload.search_queries must be a list of non-empty strings")

    spec, spec_issues = validate_evaluation_spec(spec_raw)
    issues.extend(spec_issues)
    if issues:
        return None, None, None, issues
    return spec, sources_summary, [str(item).strip() for item in queries], []


def _apply_fallback_policy_overrides(
    spec: dict[str, object],
    competition_policy,
) -> dict[str, object]:
    if not competition_policy.active:
        return spec
    payload = dict(spec)
    overrides = dict(competition_policy.evaluation.fallback_overrides)
    if "seeds" in overrides and isinstance(overrides["seeds"], list):
        payload["seeds"] = [int(item) for item in overrides["seeds"] if isinstance(item, (int, float))]
    if "repeats" in overrides and isinstance(overrides["repeats"], (int, float)):
        payload["repeats"] = max(1, int(overrides["repeats"]))
    if "ci_method" in overrides and str(overrides["ci_method"]).strip().lower() in SUPPORTED_CI_METHODS:
        payload["ci_method"] = str(overrides["ci_method"]).strip().lower()
    if "ci_alpha" in overrides and isinstance(overrides["ci_alpha"], (int, float)):
        payload["ci_alpha"] = float(overrides["ci_alpha"])
    if "n_splits" in overrides and isinstance(overrides["n_splits"], (int, float)):
        payload["n_splits"] = max(2, int(overrides["n_splits"]))
    search_stop = competition_policy.evaluation.search_stop_rank_percentile
    if search_stop is not None:
        current = payload.get("target_rank_percentile")
        if not isinstance(current, (int, float)):
            payload["target_rank_percentile"] = float(search_stop)
        else:
            payload["target_rank_percentile"] = min(float(current), float(search_stop))
    return payload


def validate_evaluation_spec(spec_raw: object) -> tuple[dict[str, object] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(spec_raw, dict):
        return None, ["evaluation_spec must be a JSON object"]

    required = {
        "metric_name",
        "direction",
        "split_strategy",
        "n_splits",
        "seeds",
        "repeats",
        "ci_method",
        "ci_alpha",
        "readiness_rule",
        "drift_check",
        "stop_policy",
    }
    optional = {"faithfulness", "deliverable_mode", "submit_mode", "target_medal", "target_rank_percentile"}
    actual = set(spec_raw.keys())
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing:
        issues.append(f"evaluation_spec missing keys: {', '.join(missing)}")
    if extra:
        issues.append(f"evaluation_spec unexpected keys: {', '.join(extra)}")

    metric_name = str(spec_raw.get("metric_name", "")).strip().lower()
    deliverable_mode = normalize_deliverable_mode(spec_raw.get("deliverable_mode"), default="leaderboard")
    submit_mode = normalize_submit_mode(spec_raw.get("submit_mode"), default="file")
    target_medal = _normalize_target_medal(spec_raw.get("target_medal"))
    if spec_raw.get("target_medal") is not None and target_medal is None:
        issues.append(TARGET_MEDAL_ERROR)
    target_rank_percentile, target_rank_percentile_issue = _normalize_target_rank_percentile(
        spec_raw.get("target_rank_percentile"),
        medal=target_medal,
    )
    if target_rank_percentile_issue is not None:
        issues.append(target_rank_percentile_issue)
    try:
        metric_def = MetricRegistry.definition(metric_name)
    except ValueError:
        metric_def = None
        issues.append(f"evaluation_spec.metric_name unsupported: {metric_name!r}")

    direction = str(spec_raw.get("direction", "")).strip().lower()
    if direction not in {"maximize", "minimize"}:
        issues.append("evaluation_spec.direction must be 'maximize' or 'minimize'")
    elif metric_def is not None and metric_def.direction != direction:
        issues.append("evaluation_spec.direction must match metric direction in registry")

    split_strategy = str(spec_raw.get("split_strategy", "")).strip().lower()
    if split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
        issues.append(f"evaluation_spec.split_strategy unsupported: {split_strategy!r}")

    n_splits = spec_raw.get("n_splits")
    if not isinstance(n_splits, int) or n_splits < 2 or n_splits > 20:
        issues.append("evaluation_spec.n_splits must be int in [2, 20]")

    seeds = spec_raw.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        issues.append("evaluation_spec.seeds must be a non-empty integer list")
        normalized_seeds: list[int] = []
    else:
        normalized_seeds = []
        for idx, item in enumerate(seeds):
            if not isinstance(item, int):
                issues.append(f"evaluation_spec.seeds[{idx}] must be int")
                continue
            normalized_seeds.append(int(item))

    repeats = spec_raw.get("repeats")
    if not isinstance(repeats, int) or repeats < 1 or repeats > 10:
        issues.append("evaluation_spec.repeats must be int in [1, 10]")

    ci_method = str(spec_raw.get("ci_method", "")).strip().lower()
    if ci_method not in SUPPORTED_CI_METHODS:
        issues.append("evaluation_spec.ci_method must be 'normal' or 'bootstrap'")

    ci_alpha = spec_raw.get("ci_alpha")
    if not isinstance(ci_alpha, (int, float)) or not (0 < float(ci_alpha) < 0.5):
        issues.append("evaluation_spec.ci_alpha must be in (0, 0.5)")

    readiness_raw = spec_raw.get("readiness_rule")
    readiness, readiness_issues = _validate_readiness_rule(readiness_raw)
    issues.extend(readiness_issues)

    drift_raw = spec_raw.get("drift_check")
    drift, drift_issues = _validate_drift_config(drift_raw)
    issues.extend(drift_issues)

    stop_raw = spec_raw.get("stop_policy")
    stop_policy, stop_issues = _validate_stop_policy(stop_raw)
    issues.extend(stop_issues)

    faithfulness_raw = spec_raw.get("faithfulness")
    faithfulness: dict[str, object] | None = None
    if faithfulness_raw is not None:
        faithfulness, faithfulness_issues = _validate_faithfulness(faithfulness_raw)
        issues.extend(faithfulness_issues)

    if issues:
        return None, issues

    assert metric_def is not None
    assert isinstance(n_splits, int)
    assert isinstance(repeats, int)
    assert isinstance(ci_alpha, (int, float))
    assert readiness is not None
    assert drift is not None
    assert stop_policy is not None

    spec = {
        "deliverable_mode": deliverable_mode,
        "submit_mode": submit_mode,
        "target_medal": target_medal,
        "target_rank_percentile": target_rank_percentile,
        "metric_name": metric_def.canonical_name,
        "direction": direction,
        "split_strategy": split_strategy,
        "n_splits": n_splits,
        "seeds": normalized_seeds,
        "repeats": repeats,
        "ci_method": ci_method,
        "ci_alpha": float(ci_alpha),
        "readiness_rule": readiness,
        "drift_check": drift,
        "stop_policy": stop_policy,
    }
    if faithfulness is not None:
        spec["faithfulness"] = faithfulness
    return spec, []


def _normalize_target_medal(value: object) -> str | None:
    return normalize_target_medal(value)


def _normalize_target_rank_percentile(
    value: object,
    *,
    medal: str | None,
) -> tuple[float | None, str | None]:
    return validate_target_rank_percentile(value, medal=medal)


def _validate_readiness_rule(value: object) -> tuple[dict[str, object] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(value, dict):
        return None, ["evaluation_spec.readiness_rule must be an object"]
    expected = {"method", "k", "target_score", "submission_gate"}
    issues.extend(_validate_exact_keys(value, expected, path="evaluation_spec.readiness_rule"))

    method = str(value.get("method", "")).strip().lower()
    if method not in SUPPORTED_READINESS_METHODS:
        issues.append("evaluation_spec.readiness_rule.method must be 'ci_bound' or 'mean_std'")

    k = value.get("k")
    if not isinstance(k, (int, float)) or float(k) < 0 or float(k) > 10:
        issues.append("evaluation_spec.readiness_rule.k must be number in [0, 10]")

    target_score = value.get("target_score")
    if target_score is not None and not isinstance(target_score, (int, float)):
        issues.append("evaluation_spec.readiness_rule.target_score must be number or null")

    submission_gate = str(value.get("submission_gate", "")).strip().lower()
    if submission_gate not in SUPPORTED_SUBMISSION_GATES:
        issues.append("evaluation_spec.readiness_rule.submission_gate unsupported")

    if issues:
        return None, issues
    return {
        "method": method,
        "k": float(k),  # type: ignore[arg-type]
        "target_score": float(target_score) if isinstance(target_score, (int, float)) else None,
        "submission_gate": submission_gate,
    }, []


def _validate_drift_config(value: object) -> tuple[dict[str, object] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(value, dict):
        return None, ["evaluation_spec.drift_check must be an object"]
    expected = {"enabled", "drift_weight"}
    issues.extend(_validate_exact_keys(value, expected, path="evaluation_spec.drift_check"))

    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        issues.append("evaluation_spec.drift_check.enabled must be bool")

    drift_weight = value.get("drift_weight")
    if not isinstance(drift_weight, (int, float)) or float(drift_weight) < 0 or float(drift_weight) > 10:
        issues.append("evaluation_spec.drift_check.drift_weight must be number in [0, 10]")

    if issues:
        return None, issues
    return {"enabled": bool(enabled), "drift_weight": float(drift_weight)}, []


def _validate_stop_policy(value: object) -> tuple[dict[str, object] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(value, dict):
        return None, ["evaluation_spec.stop_policy must be an object"]
    expected = {"min_delta", "no_improve_patience", "same_config_patience"}
    issues.extend(_validate_exact_keys(value, expected, path="evaluation_spec.stop_policy"))

    min_delta = value.get("min_delta")
    if not isinstance(min_delta, (int, float)) or float(min_delta) < 0:
        issues.append("evaluation_spec.stop_policy.min_delta must be non-negative number")

    no_improve = value.get("no_improve_patience")
    if not isinstance(no_improve, int) or no_improve < 0:
        issues.append("evaluation_spec.stop_policy.no_improve_patience must be non-negative int")

    same_config = value.get("same_config_patience")
    if not isinstance(same_config, int) or same_config < 0:
        issues.append("evaluation_spec.stop_policy.same_config_patience must be non-negative int")

    if issues:
        return None, issues
    return {
        "min_delta": float(min_delta),  # type: ignore[arg-type]
        "no_improve_patience": int(no_improve),
        "same_config_patience": int(same_config),
    }, []


def _validate_faithfulness(value: object) -> tuple[dict[str, object] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(value, dict):
        return None, ["evaluation_spec.faithfulness must be an object"]
    expected = {
        "accepted_score_sources",
        "require_metric_match",
        "require_split_match",
        "require_trusted_score_source",
        "require_competition_faithful",
        "require_full_dataset",
    }
    issues.extend(_validate_exact_keys(value, expected, path="evaluation_spec.faithfulness"))

    score_sources = value.get("accepted_score_sources")
    normalized_sources: list[str] = []
    if not isinstance(score_sources, list) or not score_sources:
        issues.append("evaluation_spec.faithfulness.accepted_score_sources must be a non-empty string list")
    else:
        for idx, item in enumerate(score_sources):
            if not isinstance(item, str) or not item.strip():
                issues.append(f"evaluation_spec.faithfulness.accepted_score_sources[{idx}] must be non-empty str")
                continue
            normalized_sources.append(item.strip().lower())

    normalized: dict[str, object] = {"accepted_score_sources": normalized_sources}
    for key in (
        "require_metric_match",
        "require_split_match",
        "require_trusted_score_source",
        "require_competition_faithful",
        "require_full_dataset",
    ):
        raw = value.get(key)
        if not isinstance(raw, bool):
            issues.append(f"evaluation_spec.faithfulness.{key} must be bool")
            continue
        normalized[key] = bool(raw)

    if issues:
        return None, issues
    return normalized, []


def _validate_exact_keys(value: dict[str, object], expected: set[str], *, path: str) -> list[str]:
    issues: list[str] = []
    actual = set(value.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        issues.append(f"{path} missing keys: {', '.join(missing)}")
    if extra:
        issues.append(f"{path} has unexpected keys: {', '.join(extra)}")
    return issues


def _run_strategy_default(prompt_path: Path, output_dir: Path) -> StrategyResult:
    return run_strategy(prompt_path=prompt_path, output_dir=output_dir, dry_run=False)


def _supports_live_search() -> bool:
    try:
        result = run_command(["codex", "exec", "--help"])
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    # Older codex CLIs exposed explicit "--search"; newer CLIs can still perform
    # web-backed runs without that flag. If codex exec is available, allow the
    # advisor path and let runtime execution decide exact capabilities.
    return True


def _parse_json_response(text: str) -> dict[str, object] | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1 or first >= last:
        return None
    candidate = raw[first : last + 1]
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _advisor_response_schema_text() -> str:
    schema = {
        "evaluation_spec": {
            "deliverable_mode": "leaderboard|writeup",
            "submit_mode": "file|notebook",
            "target_medal": TARGET_MEDAL_SCHEMA,
            "target_rank_percentile": "0<float<=1|null",
            "metric_name": (
                "one of [auc, logloss, brier_score, accuracy, f1, rmse, mae, rmsle, mape, smape, pearson, spearman]"
            ),
            "direction": "maximize|minimize",
            "split_strategy": "kfold|stratified_kfold|group_kfold|timeseries_split",
            "n_splits": "int>=2",
            "seeds": ["int", "int"],
            "repeats": "int>=1",
            "ci_method": "normal|bootstrap",
            "ci_alpha": "0<float<0.5",
            "readiness_rule": {
                "method": "ci_bound|mean_std",
                "k": "number>=0",
                "target_score": "number|null",
                "submission_gate": "readiness_only|readiness_or_final|final_only|always",
            },
            "drift_check": {
                "enabled": "bool",
                "drift_weight": "number>=0",
            },
            "stop_policy": {
                "min_delta": "number>=0",
                "no_improve_patience": "int>=0",
                "same_config_patience": "int>=0",
            },
            "faithfulness": {
                "accepted_score_sources": ["cv", "holdout"],
                "require_metric_match": "bool",
                "require_split_match": "bool",
                "require_trusted_score_source": "bool",
                "require_competition_faithful": "bool",
                "require_full_dataset": "bool",
            },
        },
        "sources_summary_md": "markdown summary with source urls and dates",
        "search_queries": ["string"],
    }
    return json.dumps(schema, indent=2)


def _read_trimmed(path: Path, *, limit_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) <= limit_chars:
        return text
    return text[:limit_chars].rstrip() + "\n...[truncated]"


def _read_sample_head(paths: CompetitionPaths, *, limit_lines: int) -> str:
    if paths.sample_submission_head_path.exists():
        candidate = paths.sample_submission_head_path
    else:
        candidate = paths.sample_submission_path
    if not candidate.exists():
        return ""
    lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[:limit_lines])


def _load_top1_score(paths: CompetitionPaths) -> float | None:
    payload = load_json_object_or_empty(paths.top1_public_path)
    score = payload.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _infer_metric_from_context(*, profile: dict[str, object], context_text: str) -> str:
    context_candidates = _extract_metric_candidates_from_text(context_text)
    if context_candidates:
        return context_candidates[0]

    profile_metric = profile.get("metric")
    if isinstance(profile_metric, str):
        metric_name = profile_metric.strip().lower()
        try:
            return MetricRegistry.definition(metric_name).canonical_name
        except ValueError:
            pass

    task = str(profile.get("task", "")).strip().lower()
    if task == "regression":
        return "rmse"
    if task in {"classification", "binary", "multiclass"}:
        return "logloss"
    return "rmse"


def _infer_split_strategy(*, profile: dict[str, object], context_text: str) -> str:
    task = str(profile.get("task", "")).strip().lower()
    if re.search(r"\btime[-\s]?series\b|\bforecast\b|\bchronolog", context_text):
        return "timeseries_split"
    modality = str(profile.get("modality", "")).strip().lower()
    if modality == "timeseries" and _profile_has_temporal_signal(profile):
        return "timeseries_split"
    if re.search(r"\bgroupkfold\b|\bgroup fold\b|\bgroup leakage\b", context_text):
        return "group_kfold"
    if task in {"classification", "binary", "multiclass"}:
        return "stratified_kfold"
    return "kfold"


def _extract_metric_candidates_from_text(context_text: str) -> list[str]:
    patterns = [
        (r"\bauc\b|\broc[-\s_]?auc\b", "auc"),
        (r"\blog\s*loss\b|\bcross[-\s_]?entropy\b", "logloss"),
        (r"\bbrier(?:[-\s_]?score)?(?:[-\s_]?loss)?\b", "brier_score"),
        (r"\baccuracy\b|\bacc\b", "accuracy"),
        (r"\bf1\b", "f1"),
        (r"\brmsle\b", "rmsle"),
        (r"\brmse\b", "rmse"),
        (r"\bmae\b", "mae"),
        (r"\bsmape\b", "smape"),
        (r"\bmape\b", "mape"),
        (r"\bpearson\b", "pearson"),
        (r"\bspearman\b", "spearman"),
    ]
    hits: list[str] = []
    for pattern, metric in patterns:
        if re.search(pattern, context_text, flags=re.IGNORECASE) and metric not in hits:
            hits.append(metric)
    return hits


def _profile_has_temporal_signal(profile: dict[str, object]) -> bool:
    dtype_map_raw = profile.get("dtype_by_column")
    if not isinstance(dtype_map_raw, dict):
        return False
    temporal_name = re.compile(r"\b(date|datetime|timestamp|time)\b", flags=re.IGNORECASE)
    for name, dtype in dtype_map_raw.items():
        column_name = str(name)
        dtype_name = str(dtype).lower()
        if "datetime" in dtype_name or "timedelta" in dtype_name:
            return True
        if temporal_name.search(column_name):
            return True
    return False


def _normalize_metric_name(name: str) -> str:
    try:
        return MetricRegistry.definition(name).canonical_name
    except ValueError:
        return name.strip().lower()


def _stale_frozen_spec_reason(*, frozen: dict[str, object], paths: CompetitionPaths) -> str | None:
    metric_name = frozen.get("metric_name")
    if not isinstance(metric_name, str) or not metric_name.strip():
        return None

    context_text = "\n".join(
        [
            _read_trimmed(paths.rules_md_path, limit_chars=4000),
            _read_trimmed(paths.overview_md_path, limit_chars=4000),
            _read_trimmed(paths.submission_format_md_path, limit_chars=4000),
        ]
    )
    candidates = _extract_metric_candidates_from_text(context_text)
    if len(candidates) != 1:
        return None
    inferred = candidates[0]
    if _normalize_metric_name(metric_name) != _normalize_metric_name(inferred):
        return f"metric mismatch frozen={metric_name} context={inferred}"
    return None
