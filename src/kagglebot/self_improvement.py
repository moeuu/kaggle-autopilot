from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.identity import IMPLEMENTATION_AGENT
from kagglebot.exec_utils import run_command
from kagglebot.paths import CompetitionPaths, KnowledgePaths


@dataclass(frozen=True)
class SelfImprovementConfig:
    artifacts_dir: Path
    knowledge_paths: KnowledgePaths
    max_runs: int = 80
    min_interval_hours: float | None = 6.0
    invoke_codex: bool = True
    force: bool = False
    dry_run: bool = False

    @property
    def output_dir(self) -> Path:
        return self.artifacts_dir / "_self_improvement"

    @property
    def latest_json_path(self) -> Path:
        return self.output_dir / "latest.json"

    @property
    def latest_markdown_path(self) -> Path:
        return self.output_dir / "latest.md"

    @property
    def reports_jsonl_path(self) -> Path:
        return self.output_dir / "reports.jsonl"

    @property
    def codex_dir(self) -> Path:
        return self.output_dir / "codex"


def run_self_improvement_cycle(config: SelfImprovementConfig) -> dict[str, object]:
    """Analyze recent autopilot outcomes and persist a system-improvement report."""
    if config.dry_run:
        return {"status": "dry_run", "report_path": str(config.latest_json_path)}
    if not _self_improvement_due(config):
        return {"status": "skipped_not_due", "report_path": str(config.latest_json_path)}

    runs = _collect_recent_runs(config.artifacts_dir, limit=max(1, config.max_runs))
    report = _build_report(artifacts_dir=config.artifacts_dir, runs=runs)
    markdown = _render_markdown(report)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.latest_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    config.latest_markdown_path.write_text(markdown, encoding="utf-8")
    codex_result = _maybe_run_codex_improvement(config=config, report=report)
    if codex_result is not None:
        report["codex_improvement"] = codex_result
        config.latest_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        config.latest_markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    with config.reports_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True) + "\n")
    return {
        "status": "written",
        "report_path": str(config.latest_json_path),
        "markdown_path": str(config.latest_markdown_path),
        "runs_analyzed": len(runs),
        "codex_improvement": codex_result,
        "top_actions": report.get("recommended_actions", [])[:3],
    }


def _maybe_run_codex_improvement(
    *,
    config: SelfImprovementConfig,
    report: dict[str, object],
) -> dict[str, object] | None:
    if not config.invoke_codex:
        return {"status": "disabled"}
    dirty = _git_dirty(config.knowledge_paths.workdir)
    if dirty:
        return {
            "status": "skipped_dirty_worktree",
            "reason": "Codex self-improvement only runs from a clean git worktree.",
        }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.codex_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(_build_codex_prompt(config=config, report=report), encoding="utf-8")
    result = run_codex(
        prompt_path,
        output_dir,
        dry_run=config.dry_run,
        heartbeat_label="self-improvement",
        model=IMPLEMENTATION_AGENT.model,
        reasoning_effort=IMPLEMENTATION_AGENT.reasoning_effort,
    )
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "prompt_path": str(prompt_path),
        "transcript_path": str(result.transcript_path),
        "last_message_path": str(result.last_message_path),
    }


def _build_codex_prompt(*, config: SelfImprovementConfig, report: dict[str, object]) -> str:
    actions = report.get("recommended_actions")
    actions_text = json.dumps(actions if isinstance(actions, list) else [], indent=2, sort_keys=True)
    return f"""# Kagglebot Self-Improvement Task

You are the implementation agent for this repository.

Goal: improve Kagglebot's ability to win future Kaggle competitions by addressing the highest-signal
root cause from the latest self-improvement report.

Hard constraints:
- Do not submit to Kaggle, accept rules, join competitions, or call external side-effect APIs.
- Do not write secrets, credentials, datasets, or large artifacts.
- Make a small, testable repo change. Prefer orchestration, diagnostics, validation, strategy prompts, or
  reusable runtime improvements over competition-specific hacks.
- Preserve existing guardrails: validation, duplicate detection, rate limits, and human-readable submit messages.
- Run focused tests plus `uv run ruff check .` when feasible.
- Do not commit or push; the outer operator decides when to publish.

Latest report files:
- JSON: {config.latest_json_path}
- Markdown: {config.latest_markdown_path}

Recommended actions:
```json
{actions_text}
```

Read the report, inspect the relevant code/tests, implement the single best structural improvement, and leave a concise
summary in your final message.
"""


def _git_dirty(workdir: Path) -> bool:
    try:
        result = run_command(["git", "status", "--porcelain"], cwd=workdir)
    except (OSError, RuntimeError):
        return True
    return bool(result.stdout.strip() or result.stderr.strip() or result.returncode != 0)


def _self_improvement_due(config: SelfImprovementConfig) -> bool:
    if config.force:
        return True
    if config.min_interval_hours is None or config.min_interval_hours <= 0:
        return False
    path = config.latest_json_path
    if not path.exists():
        return True
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return True
    return mtime + timedelta(hours=config.min_interval_hours) <= datetime.now(UTC)


def _collect_recent_runs(artifacts_dir: Path, *, limit: int) -> list[dict[str, object]]:
    candidates: list[tuple[float, Path, str]] = []
    if not artifacts_dir.exists():
        return []
    for competition_dir in artifacts_dir.iterdir():
        if not competition_dir.is_dir() or competition_dir.name.startswith("_"):
            continue
        runs_dir = competition_dir / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                mtime = run_dir.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, run_dir, competition_dir.name))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [_load_run_summary(run_dir=run_dir, slug=slug) for _mtime, run_dir, slug in candidates[:limit]]


def _load_run_summary(*, run_dir: Path, slug: str) -> dict[str, object]:
    paths = CompetitionPaths(slug=slug, artifacts_dir=run_dir.parents[2])
    run_payload = _read_json_object(run_dir / "run.json")
    run_id = str(run_payload.get("run_id") or run_dir.name)
    config = run_payload.get("config") if isinstance(run_payload.get("config"), dict) else {}
    direction = str((config or {}).get("target_direction") or (config or {}).get("direction") or "").lower() or None
    metric = str((config or {}).get("target_metric") or (config or {}).get("goal_metric") or "").strip() or None
    status = str(run_payload.get("status") or "unknown")
    iterations = _load_iteration_summaries(run_dir)
    best_offline = _best_iteration_value(iterations=iterations, direction=direction)
    top1_score = _to_float(_read_json_object(paths.top1_public_path).get("score"))
    outcomes = _load_submission_outcomes(paths.submission_ledger_path, run_id=run_id)
    best_online = _best_online_score(outcomes=outcomes, direction=direction)
    top1_gap = _score_gap(best_score=best_online, top1_score=top1_score, direction=direction)
    submit_failures = _load_submit_failures(run_dir / "submit_attempts.jsonl")
    latest_diagnostics = _latest_text(run_dir, "diagnostics.md", max_chars=1800)
    failure_contexts = [_read_json_object(path) for path in sorted(run_dir.glob("iter-*/submit_failure_context.json"))]
    cause_tags = _infer_cause_tags(
        status=status,
        iterations=iterations,
        outcomes=outcomes,
        best_offline=best_offline,
        best_online=best_online,
        top1_score=top1_score,
        top1_gap=top1_gap,
        submit_failures=submit_failures,
        failure_contexts=failure_contexts,
        diagnostics=latest_diagnostics,
    )
    return {
        "slug": slug,
        "run_id": run_id,
        "status": status,
        "metric": metric,
        "direction": direction,
        "iteration_count": len(iterations),
        "best_offline": best_offline,
        "best_online": best_online,
        "top1_public_score": top1_score,
        "top1_gap": top1_gap,
        "submission_outcome_count": len(outcomes),
        "submit_failure_count": len(submit_failures),
        "cause_tags": cause_tags,
        "diagnostics_excerpt": latest_diagnostics,
    }


def _load_iteration_summaries(run_dir: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for iter_dir in sorted(run_dir.glob("iter-*"), key=_iteration_sort_key):
        metrics_path = iter_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path = iter_dir / "output" / "metrics.json"
        metrics = _read_json_object(metrics_path)
        value = _metric_value(metrics)
        summaries.append(
            {
                "iteration": _iteration_sort_key(iter_dir),
                "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                "value": value,
                "score_source": metrics.get("score_source"),
                "metric_status": metrics.get("metric_status"),
            }
        )
    return summaries


def _load_submission_outcomes(path: Path, *, run_id: str) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for record in _read_jsonl(path):
        if str(record.get("run_id") or "") != run_id:
            continue
        if str(record.get("event") or "") != "outcome":
            continue
        outcome = record.get("outcome")
        if isinstance(outcome, dict):
            outcomes.append(outcome)
    return outcomes


def _load_submit_failures(path: Path) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for record in _read_jsonl(path):
        action = str(record.get("action_taken") or record.get("event") or "").lower()
        reason = str(record.get("reason") or record.get("error") or "").lower()
        if "fail" in action or "abort" in action or "error" in reason or reason:
            failures.append(record)
    return failures


def _build_report(*, artifacts_dir: Path, runs: list[dict[str, object]]) -> dict[str, object]:
    cause_counter = Counter(tag for run in runs for tag in _string_list(run.get("cause_tags")))
    status_counter = Counter(str(run.get("status") or "unknown") for run in runs)
    top1_gap_runs = [
        run for run in runs if isinstance(run.get("top1_gap"), (int, float)) and math.isfinite(float(run["top1_gap"]))
    ]
    top1_gap_runs.sort(key=lambda item: float(item["top1_gap"]), reverse=True)
    actions = _recommended_actions(cause_counter)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifacts_dir": str(artifacts_dir),
        "run_count": len(runs),
        "status_counts": dict(status_counter.most_common()),
        "cause_counts": dict(cause_counter.most_common()),
        "largest_top1_gaps": [
            {
                "slug": run.get("slug"),
                "run_id": run.get("run_id"),
                "top1_gap": run.get("top1_gap"),
                "best_online": run.get("best_online"),
                "top1_public_score": run.get("top1_public_score"),
                "cause_tags": run.get("cause_tags"),
            }
            for run in top1_gap_runs[:10]
        ],
        "recent_problem_runs": [_compact_run(run) for run in runs if _is_problem_run(run)][:20],
        "recommended_actions": actions,
    }


def _recommended_actions(causes: Counter[str]) -> list[dict[str, object]]:
    action_map = {
        "no_successful_submission": "Prioritize submission-mode and artifact validation fixes before model search.",
        "submit_failed": "Improve submit fallback diagnostics, notebook/file mode inference, and retry classification.",
        "no_iteration_metrics": "Harden kernel runtime so every run emits metrics.json and diagnostics.md.",
        "online_far_from_top1": (
            "Force broader model-family search, ensembling, public-LB validation, and data-source review."
        ),
        "offline_online_mismatch": (
            "Investigate leakage, split mismatch, sample weighting, and public-LB proxy quality."
        ),
        "metric_or_validation_error": "Tighten metric contract validation and fail earlier when scoring is untrusted.",
        "resource_or_capacity": "Add cheaper smoke tests and resource-aware model schedules before expensive runs.",
    }
    actions: list[dict[str, object]] = []
    for cause, count in causes.most_common():
        action = action_map.get(cause)
        if action:
            actions.append({"cause": cause, "count": count, "action": action})
    if not actions:
        actions.append(
            {
                "cause": "insufficient_signal",
                "count": 0,
                "action": "Collect more submission outcomes and diagnostics before changing orchestration policy.",
            }
        )
    return actions


def _infer_cause_tags(
    *,
    status: str,
    iterations: list[dict[str, object]],
    outcomes: list[dict[str, object]],
    best_offline: float | None,
    best_online: float | None,
    top1_score: float | None,
    top1_gap: float | None,
    submit_failures: list[dict[str, object]],
    failure_contexts: list[dict[str, object]],
    diagnostics: str,
) -> list[str]:
    tags: list[str] = []
    status_l = status.lower()
    diagnostics_l = diagnostics.lower()
    if not iterations:
        tags.append("no_iteration_metrics")
    if submit_failures or "submit_failed" in status_l:
        tags.append("submit_failed")
    if not outcomes:
        tags.append("no_successful_submission")
    if top1_gap is not None and top1_gap > 0:
        tags.append("online_far_from_top1")
    if best_offline is not None and best_online is not None and top1_score is not None:
        if abs(best_offline) > 1e-12 and abs(best_online) <= 1e-12 and abs(top1_score) > 1e-12:
            tags.append("offline_online_mismatch")
    if any("metric" in str(item).lower() or "validation" in str(item).lower() for item in failure_contexts):
        tags.append("metric_or_validation_error")
    if any(token in diagnostics_l for token in ("metric mismatch", "validation", "nan", "schema", "row count")):
        tags.append("metric_or_validation_error")
    if any(token in diagnostics_l for token in ("out of memory", "oom", "capacity", "timeout", "killed")):
        tags.append("resource_or_capacity")
    if not tags:
        tags.append("near_top1_or_no_signal")
    return list(dict.fromkeys(tags))


def _render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Kagglebot Self-Improvement Report",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- run_count: {report.get('run_count')}",
        "",
        "## Cause Counts",
    ]
    cause_counts = report.get("cause_counts")
    if isinstance(cause_counts, dict) and cause_counts:
        for cause, count in cause_counts.items():
            lines.append(f"- {cause}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended Actions"])
    actions = report.get("recommended_actions")
    if isinstance(actions, list):
        for item in actions:
            if isinstance(item, dict):
                lines.append(f"- {item.get('cause')}: {item.get('action')} (count={item.get('count')})")
    lines.extend(["", "## Largest Top1 Gaps"])
    gaps = report.get("largest_top1_gaps")
    if isinstance(gaps, list) and gaps:
        for item in gaps:
            if isinstance(item, dict):
                lines.append(
                    "- "
                    f"{item.get('slug')} {item.get('run_id')}: "
                    f"gap={item.get('top1_gap')} best={item.get('best_online')} top1={item.get('top1_public_score')}"
                )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _compact_run(run: dict[str, object]) -> dict[str, object]:
    return {
        "slug": run.get("slug"),
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "best_offline": run.get("best_offline"),
        "best_online": run.get("best_online"),
        "top1_gap": run.get("top1_gap"),
        "cause_tags": run.get("cause_tags"),
    }


def _is_problem_run(run: dict[str, object]) -> bool:
    causes = _string_list(run.get("cause_tags"))
    return bool(causes) and "near_top1_or_no_signal" not in causes


def _best_iteration_value(*, iterations: list[dict[str, object]], direction: str | None) -> float | None:
    values = [_to_float(item.get("value")) for item in iterations]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return min(finite) if direction == "minimize" else max(finite)


def _best_online_score(*, outcomes: list[dict[str, object]], direction: str | None) -> float | None:
    values = [_to_float(item.get("score")) for item in outcomes]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return min(finite) if direction == "minimize" else max(finite)


def _score_gap(*, best_score: float | None, top1_score: float | None, direction: str | None) -> float | None:
    if best_score is None or top1_score is None:
        return None
    gap = best_score - top1_score if direction == "minimize" else top1_score - best_score
    return max(0.0, gap)


def _metric_value(metrics: dict[str, object]) -> float | None:
    for key in ("offline_value", "value", "score", "cv_score", "holdout_score"):
        parsed = _to_float(metrics.get(key))
        if parsed is not None and math.isfinite(parsed):
            return parsed
    selected = metrics.get("selected")
    if isinstance(selected, dict):
        parsed = _to_float(selected.get("offline_value") or selected.get("value") or selected.get("score"))
        if parsed is not None and math.isfinite(parsed):
            return parsed
    loop_decision = metrics.get("loop_decision")
    if isinstance(loop_decision, dict):
        parsed = _to_float(loop_decision.get("value"))
        if parsed is not None and math.isfinite(parsed):
            return parsed
    return None


def _latest_text(run_dir: Path, name: str, *, max_chars: int) -> str:
    paths = sorted(run_dir.glob(f"iter-*/{name}"), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    if not paths:
        return ""
    try:
        text = paths[-1].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _iteration_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0
