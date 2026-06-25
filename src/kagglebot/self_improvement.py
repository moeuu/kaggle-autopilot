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
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.scalar_utils import parse_finite_float as _to_float
from kagglebot.submit_attempts import load_submit_attempt_rows


@dataclass(frozen=True)
class SelfImprovementConfig:
    artifacts_dir: Path
    knowledge_paths: KnowledgePaths
    max_runs: int = 80
    min_interval_hours: float | None = 6.0
    invoke_codex: bool = True
    allow_architectural_changes: bool = True
    publish_codex_changes: bool = False
    publish_verify_commands: tuple[tuple[str, ...], ...] = (
        ("uv", "run", "ruff", "format", "."),
        ("uv", "run", "ruff", "check", "."),
        ("uv", "run", "pytest", "-q"),
    )
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
    def outcomes_jsonl_path(self) -> Path:
        return self.output_dir / "outcomes.jsonl"

    @property
    def strategy_context_path(self) -> Path:
        return self.output_dir / "strategy_context.md"

    @property
    def experiment_backlog_path(self) -> Path:
        return self.output_dir / "experiment_backlog.json"

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
    backlog = _build_experiment_backlog(report)
    outcomes = _normalized_outcomes(runs)
    report["strategy_context_path"] = str(config.strategy_context_path)
    report["experiment_backlog_path"] = str(config.experiment_backlog_path)
    report["playbook_paths"] = _write_playbooks(config.knowledge_paths, report)
    markdown = _render_markdown(report)
    strategy_context = _render_strategy_context(report, backlog=backlog)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_object(config.latest_json_path, report, sort_keys=True)
    config.latest_markdown_path.write_text(markdown, encoding="utf-8")
    config.strategy_context_path.write_text(strategy_context, encoding="utf-8")
    config.experiment_backlog_path.write_text(json.dumps(backlog, indent=2, sort_keys=True), encoding="utf-8")
    config.outcomes_jsonl_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in outcomes),
        encoding="utf-8",
    )
    codex_result = _maybe_run_codex_improvement(config=config, report=report)
    if codex_result is not None:
        report["codex_improvement"] = codex_result
        write_json_object(config.latest_json_path, report, sort_keys=True)
        config.latest_markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    with config.reports_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True) + "\n")
    return {
        "status": "written",
        "report_path": str(config.latest_json_path),
        "markdown_path": str(config.latest_markdown_path),
        "strategy_context_path": str(config.strategy_context_path),
        "experiment_backlog_path": str(config.experiment_backlog_path),
        "runs_analyzed": len(runs),
        "codex_improvement": codex_result,
        "top_actions": report.get("recommended_actions", [])[:3],
    }


def load_self_improvement_context(artifacts_dir: Path, *, max_chars: int = 6000) -> str:
    """Load the latest self-improvement directives for planner/improvement prompts."""
    for path in (
        artifacts_dir / "_self_improvement" / "strategy_context.md",
        artifacts_dir / "_self_improvement" / "latest.md",
    ):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text[-max_chars:]
    return ""


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
    publish_result = _maybe_publish_codex_changes(config=config, codex_returncode=result.returncode)
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "prompt_path": str(prompt_path),
        "transcript_path": str(result.transcript_path),
        "last_message_path": str(result.last_message_path),
        "publish": publish_result,
    }


def _build_codex_prompt(*, config: SelfImprovementConfig, report: dict[str, object]) -> str:
    actions = report.get("recommended_actions")
    actions_text = json.dumps(actions if isinstance(actions, list) else [], indent=2, sort_keys=True)
    architecture_policy = (
        "- Architectural changes are allowed when they are the best route to top1 performance. You may refactor "
        "or redesign planner, runner, evaluation, strategy, knowledge, model-search, or self-improvement flows if "
        "the report shows the current architecture is the bottleneck.\n"
        "- For any architectural change, include a migration/compatibility path, focused tests for the changed "
        "contract, and a short design note in docs or code comments where the new boundary is not obvious."
        if config.allow_architectural_changes
        else "- Keep changes narrowly scoped unless the operator explicitly enables architectural changes."
    )
    return f"""# Kagglebot Self-Improvement Task

You are the implementation agent for this repository.

Goal: improve Kagglebot's ability to reach first-place Kaggle leaderboard performance by addressing the
highest-signal root cause from the latest self-improvement report.

Hard constraints:
- Do not submit to Kaggle, accept rules, join competitions, or call external side-effect APIs.
- Do not write secrets, credentials, datasets, or large artifacts.
- Preserve existing guardrails: validation, duplicate detection, rate limits, and human-readable submit messages.
- Run focused tests plus `uv run ruff check .` when feasible.
- Do not commit or push from inside Codex; the outer self-improvement controller owns publish policy.

Change scope policy:
{architecture_policy}
- Prefer reusable top1-oriented capability improvements over competition-specific hacks.
- It is acceptable for the best change to span multiple modules when a local patch would only hide the failure mode.
- Keep the change reviewable: state the contract being changed, update tests/docs, and avoid unrelated churn.

Latest report files:
- JSON: {config.latest_json_path}
- Markdown: {config.latest_markdown_path}
- Strategy context: {config.strategy_context_path}
- Experiment backlog: {config.experiment_backlog_path}

Recommended actions:
```json
{actions_text}
```

Read the report and backlog, inspect the relevant code/tests, implement the best top1-oriented improvement at the
right architectural level, and leave a concise summary in your final message.
"""


def _maybe_publish_codex_changes(*, config: SelfImprovementConfig, codex_returncode: int) -> dict[str, object]:
    if codex_returncode != 0:
        return {"status": "skipped_codex_failed"}
    if not config.publish_codex_changes:
        return {"status": "disabled"}
    if config.dry_run:
        return {"status": "dry_run"}
    workdir = config.knowledge_paths.workdir
    if not _git_dirty(workdir):
        return {"status": "skipped_no_changes"}

    verification: list[dict[str, object]] = []
    for command in config.publish_verify_commands:
        result = run_command(list(command), cwd=workdir, stream_output=True)
        verification.append(
            {
                "args": list(command),
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
        if result.returncode != 0:
            return {"status": "verification_failed", "verification": verification}

    add_result = run_command(
        [
            "git",
            "add",
            "-A",
            "--",
            "src",
            "tests",
            "docs",
            "README.md",
            "STRATEGY.md",
            "AGENTS.md",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=workdir,
    )
    if add_result.returncode != 0:
        return {"status": "git_add_failed", "stderr": add_result.stderr, "verification": verification}
    if not _git_staged_changes(workdir):
        return {"status": "skipped_no_stageable_changes", "verification": verification}

    commit_result = run_command(
        ["git", "commit", "-m", "Self-improve autopilot from report"],
        cwd=workdir,
        stream_output=True,
    )
    if commit_result.returncode != 0:
        return {
            "status": "git_commit_failed",
            "stderr": commit_result.stderr,
            "stdout": commit_result.stdout,
            "verification": verification,
        }
    push_result = run_command(["git", "push", "origin", "HEAD"], cwd=workdir, stream_output=True)
    if push_result.returncode != 0:
        return {
            "status": "git_push_failed",
            "stderr": push_result.stderr,
            "stdout": push_result.stdout,
            "verification": verification,
        }
    head_result = run_command(["git", "rev-parse", "HEAD"], cwd=workdir)
    return {
        "status": "pushed",
        "commit": head_result.stdout.strip(),
        "verification": verification,
    }


def _git_dirty(workdir: Path) -> bool:
    try:
        result = run_command(["git", "status", "--porcelain"], cwd=workdir)
    except (OSError, RuntimeError):
        return True
    return bool(result.stdout.strip() or result.stderr.strip() or result.returncode != 0)


def _git_staged_changes(workdir: Path) -> bool:
    try:
        result = run_command(["git", "diff", "--cached", "--quiet"], cwd=workdir)
    except (OSError, RuntimeError):
        return False
    return result.returncode == 1


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
    campaign_outcomes = _load_campaign_outcomes(paths.context_dir / "campaign_outcomes.jsonl", run_id=run_id)
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
        "campaign_outcome_count": len(campaign_outcomes),
        "campaign_outcomes": campaign_outcomes[-20:],
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


def _load_campaign_outcomes(path: Path, *, run_id: str) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for record in _read_jsonl(path):
        if str(record.get("run_id") or "") != run_id:
            continue
        outcomes.append(record)
    return outcomes


def _load_submit_failures(path: Path) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for record in load_submit_attempt_rows(path.parent):
        action = str(record.get("action_taken") or record.get("event") or "").lower()
        reason = str(record.get("reason") or record.get("error") or "").lower()
        if "fail" in action or "abort" in action or "error" in reason or reason:
            failures.append(record)
    return failures


def _build_report(*, artifacts_dir: Path, runs: list[dict[str, object]]) -> dict[str, object]:
    cause_counter = Counter(tag for run in runs for tag in _string_list(run.get("cause_tags")))
    method_counter = Counter(
        str(outcome.get("method_id"))
        for run in runs
        for outcome in _dict_list(run.get("campaign_outcomes"))
        if outcome.get("method_id")
    )
    validation_counter = Counter(
        str(outcome.get("validation_profile_id"))
        for run in runs
        for outcome in _dict_list(run.get("campaign_outcomes"))
        if outcome.get("validation_profile_id")
    )
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
        "campaign_method_counts": dict(method_counter.most_common(20)),
        "campaign_validation_profile_counts": dict(validation_counter.most_common(20)),
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


def _normalized_outcomes(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for run in runs:
        outcomes.append(
            {
                "slug": run.get("slug"),
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "metric": run.get("metric"),
                "direction": run.get("direction"),
                "best_offline": run.get("best_offline"),
                "best_online": run.get("best_online"),
                "top1_public_score": run.get("top1_public_score"),
                "top1_gap": run.get("top1_gap"),
                "cause_tags": run.get("cause_tags"),
                "submission_outcome_count": run.get("submission_outcome_count"),
                "submit_failure_count": run.get("submit_failure_count"),
                "campaign_outcome_count": run.get("campaign_outcome_count"),
                "campaign_outcomes": run.get("campaign_outcomes"),
            }
        )
    return outcomes


def _build_experiment_backlog(report: dict[str, object]) -> list[dict[str, object]]:
    actions = report.get("recommended_actions")
    if not isinstance(actions, list):
        return []
    backlog: list[dict[str, object]] = []
    templates = {
        "no_successful_submission": (
            "Submission validation/submission-mode defects are blocking learning from the leaderboard.",
            "Add focused validation or recovery that turns one failed submission class into an actionable retry.",
            "A future run with this cause reaches at least one successful submission outcome.",
        ),
        "submit_failed": (
            "Submit failures contain recoverable mode, path, or API classifications.",
            "Improve failure classification and fallback selection from submit_attempts.jsonl and diagnostics.",
            "Submit-failed runs produce a classified retry or a non-retryable reason with artifact links.",
        ),
        "no_iteration_metrics": (
            "The runtime is losing metrics before the supervisor can make informed decisions.",
            "Harden kernel/runtime exit handling so metrics.json and diagnostics.md are emitted on every path.",
            "Every iter-* directory has metrics.json or an explicit failure_context artifact.",
        ),
        "online_far_from_top1": (
            "The current search space is too narrow for competitions with a visible public top score gap.",
            "Broaden the first-plan model family, ensemble, data-source, or public-LB proxy schedule.",
            "Median top1_gap decreases across the next comparable runs.",
        ),
        "offline_online_mismatch": (
            "Offline validation is not ranking submissions like the public leaderboard.",
            "Add split/leakage diagnostics and force alternate validation when mismatch signals appear.",
            "Future runs record lower offline-vs-online disagreement before late iterations.",
        ),
        "metric_or_validation_error": (
            "Metric/schema ambiguity is creating invalid confidence in candidate submissions.",
            "Strengthen metric contract parsing, sample-submission alignment, and early scoring checks.",
            "Invalid metric/schema runs fail before training expensive candidates.",
        ),
        "resource_or_capacity": (
            "Resource failures are consuming iterations before useful model evidence is generated.",
            "Schedule cheap smoke tests and capacity-aware model choices before expensive training.",
            "Runs with capacity signals emit a smaller retry plan instead of repeating the same failure.",
        ),
    }
    for index, item in enumerate(actions[:8], start=1):
        if not isinstance(item, dict):
            continue
        cause = str(item.get("cause") or "insufficient_signal")
        hypothesis, experiment, success_metric = templates.get(
            cause,
            (
                "The current evidence is not specific enough to justify a narrow fix.",
                "Improve diagnostic collection before changing model-selection behavior.",
                "Next reports include enough outcomes, metrics, and diagnostics to classify the failure.",
            ),
        )
        backlog.append(
            {
                "id": f"si-{index:03d}-{cause.replace('_', '-')}",
                "priority": index,
                "cause": cause,
                "count": item.get("count", 0),
                "hypothesis": hypothesis,
                "experiment": experiment,
                "success_metric": success_metric,
                "architecture_scope": (
                    "Architectural changes are allowed when they remove a repeated top1 blocker; keep a "
                    "compatibility path and tests for changed contracts."
                ),
                "guardrail": "Do not submit, accept rules, join competitions, write secrets, or commit artifacts.",
            }
        )
    return backlog


def _write_playbooks(knowledge_paths: KnowledgePaths, report: dict[str, object]) -> list[str]:
    playbooks_dir = knowledge_paths.knowledge_dir / "playbooks"
    playbooks_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    global_path = playbooks_dir / "global.md"
    global_path.write_text(_render_global_playbook(report), encoding="utf-8")
    paths.append(str(global_path))
    cause_counts = report.get("cause_counts")
    if isinstance(cause_counts, dict):
        for cause in cause_counts:
            cause_name = str(cause)
            path = playbooks_dir / f"{cause_name}.md"
            path.write_text(_render_cause_playbook(cause_name, report), encoding="utf-8")
            paths.append(str(path))
    return paths


def _render_global_playbook(report: dict[str, object]) -> str:
    lines = [
        "# Kagglebot Global Playbook",
        "",
        "Use this playbook before planning or improving a competition run.",
        "",
        "## Current Priorities",
    ]
    actions = report.get("recommended_actions")
    if isinstance(actions, list) and actions:
        for item in actions:
            if isinstance(item, dict):
                lines.append(f"- {item.get('cause')}: {item.get('action')}")
    else:
        lines.append("- Collect more outcomes before changing strategy.")
    lines.extend(
        [
            "",
            "## Guardrails",
            "- Keep submissions validated against sample_submission.csv.",
            "- Do not automate joining competitions or accepting rules.",
            "- Do not write secrets, datasets, or large artifacts to git.",
            "- Prefer structural or architectural improvements over one-off competition hacks.",
            "- Refactor core boundaries when repeated top1 blockers show the current architecture is the bottleneck.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_cause_playbook(cause: str, report: dict[str, object]) -> str:
    action = "Collect more diagnostic signal."
    actions = report.get("recommended_actions")
    if isinstance(actions, list):
        for item in actions:
            if isinstance(item, dict) and item.get("cause") == cause:
                action = str(item.get("action") or action)
                break
    examples = []
    problem_runs = report.get("recent_problem_runs")
    if isinstance(problem_runs, list):
        for run in problem_runs:
            if isinstance(run, dict) and cause in _string_list(run.get("cause_tags")):
                examples.append(f"- {run.get('slug')} {run.get('run_id')}: gap={run.get('top1_gap')}")
    lines = [
        f"# Playbook: {cause}",
        "",
        f"Recommended action: {action}",
        "",
        "## Signals",
    ]
    if examples:
        lines.extend(examples[:10])
    else:
        lines.append("- No concrete recent examples in the latest report.")
    lines.extend(
        [
            "",
            "## Next Experiment",
            "- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.",
            "- If a local fix would only mask the issue, change the responsible architecture boundary instead.",
            "- Add focused tests proving the behavior on synthetic artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_strategy_context(report: dict[str, object], *, backlog: list[dict[str, object]]) -> str:
    lines = [
        "# Kagglebot Self-Improvement Context",
        "",
        f"Generated at: {report.get('generated_at')}",
        f"Runs analyzed: {report.get('run_count')}",
        "",
        "## Highest-Priority Actions",
    ]
    actions = report.get("recommended_actions")
    if isinstance(actions, list) and actions:
        for item in actions[:5]:
            if isinstance(item, dict):
                lines.append(f"- {item.get('cause')} ({item.get('count')}): {item.get('action')}")
    else:
        lines.append("- No actions available.")
    lines.extend(["", "## Experiment Backlog"])
    if backlog:
        for item in backlog[:5]:
            lines.append(f"- {item['id']}: {item['experiment']} Success: {item['success_metric']}")
    else:
        lines.append("- No backlog items available.")
    method_counts = report.get("campaign_method_counts")
    if isinstance(method_counts, dict) and method_counts:
        lines.extend(["", "## Campaign Method Outcomes"])
        for method_id, count in list(method_counts.items())[:8]:
            lines.append(f"- {method_id}: {count}")
    validation_counts = report.get("campaign_validation_profile_counts")
    if isinstance(validation_counts, dict) and validation_counts:
        lines.extend(["", "## Validation Profile Outcomes"])
        for profile_id, count in list(validation_counts.items())[:8]:
            lines.append(f"- {profile_id}: {count}")
    lines.extend(["", "## Recent Problem Runs"])
    problem_runs = report.get("recent_problem_runs")
    if isinstance(problem_runs, list) and problem_runs:
        for run in problem_runs[:8]:
            if isinstance(run, dict):
                lines.append(
                    f"- {run.get('slug')} {run.get('run_id')}: "
                    f"status={run.get('status')} top1_gap={run.get('top1_gap')} causes={run.get('cause_tags')}"
                )
    else:
        lines.append("- No recent problem runs.")
    lines.extend(
        [
            "",
            "## How to Use This",
            "- Let these priorities influence the initial plan, model-search breadth, validation, and retry logic.",
            "- Prefer experiments that reduce repeated failure causes across competitions.",
            "- Architectural refactors are in scope when they are the cleanest path to remove a repeated top1 blocker.",
            "- Keep Kaggle side effects controlled by the normal autopilot submission guardrails.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    return load_json_object(path) or {}


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


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _iteration_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0
