from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.exceptions import KaggleBotError
from kagglebot.knowledge import (
    derive_problem_types,
    format_error_fix_insights,
    format_problem_type_insights,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
)
from kagglebot.logging_utils import truncate_lines
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.solution_guard import ensure_solution_path_allowed
from kagglebot.solver.metrics import infer_direction
from kagglebot.types import PlanConfig
from kagglebot.validators import scan_text_for_secrets

_PLANNING_CODEX_MODEL = "gpt-5.3-codex"
_PLANNING_REASONING_EFFORT = "extra_high"


@dataclass(frozen=True)
class AgentPipelineConfig:
    slug: str
    competition_url: str | None
    compute: str
    accelerator: str
    internet: str
    run_id: str
    dry_run: bool
    repo_root: Path


def run_agent_pipeline(*, paths: CompetitionPaths, config: AgentPipelineConfig) -> None:
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
    _ensure_context_materials(paths)

    brief_dir = paths.context_agent_dir / "brief"
    strategy_dir = paths.context_agent_dir / "strategy"
    implement_dir = paths.context_agent_dir / "implement"
    brief_dir.mkdir(parents=True, exist_ok=True)
    strategy_dir.mkdir(parents=True, exist_ok=True)
    implement_dir.mkdir(parents=True, exist_ok=True)

    brief_path = _run_codex_brief(paths, config, brief_dir)
    instructions_path = _run_strategy_plan(paths, config, strategy_dir, brief_path)
    _run_codex_kernel_implementation(paths, config, implement_dir, instructions_path)


def _run_codex_brief(paths: CompetitionPaths, config: AgentPipelineConfig, output_dir: Path) -> Path:
    template = _load_template("codex_brief.md")
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "competition_url": config.competition_url or "unknown",
            "rules_url": _read_text(paths.rules_url_path),
            "overview_path": str(paths.overview_md_path),
            "data_path": str(paths.data_md_path),
            "rules_path": str(paths.rules_md_path),
            "dataset_profile_path": str(paths.dataset_profile_path),
            "submission_format_path": str(paths.submission_format_md_path),
            "sample_submission_head_path": str(paths.sample_submission_head_path),
            "sample_submission_path": str(paths.sample_submission_path),
        },
    )
    prompt_text = _append_problem_type_knowledge(
        prompt_text,
        _load_problem_type_knowledge_text(paths=paths, repo_root=config.repo_root),
    )
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _print_block("codex brief prompt (sent)", prompt_text)

    brief_path = paths.context_agent_dir / "brief_for_strategy.md"
    brief_text, error_text = _run_codex_brief_with_retry(
        prompt_path=prompt_path,
        output_dir=output_dir,
        paths=paths,
        config=config,
    )
    if not brief_text:
        brief_text = _build_fallback_brief(paths, config, error_text)
    _print_block("codex brief (used for GPT)", brief_text)
    brief_path.write_text(brief_text + "\n", encoding="utf-8")
    return brief_path


def _run_codex_brief_with_retry(
    *,
    prompt_path: Path,
    output_dir: Path,
    paths: CompetitionPaths,
    config: AgentPipelineConfig,
    retries: int = 1,
) -> tuple[str, str]:
    attempt = 0
    last_error = ""
    while attempt <= retries:
        allowed_prefixes = [paths.context_agent_dir]
        guard_snapshot = _backup_guarded_files(config.repo_root, allowed_prefixes)
        before = _snapshot_tree(config.repo_root)
        result = run_codex(
            prompt_path,
            output_dir,
            dry_run=config.dry_run,
            model=_PLANNING_CODEX_MODEL,
            reasoning_effort=_PLANNING_REASONING_EFFORT,
        )
        after = _snapshot_tree(config.repo_root)
        _enforce_allowlist_changes(
            root=config.repo_root,
            before=before,
            after=after,
            allowed_prefixes=allowed_prefixes,
            stage="codex_brief",
            guard_snapshot=guard_snapshot,
            auto_repair=True,
        )
        brief_text = _read_text(result.last_message_path).strip()
        if result.returncode == 0 and brief_text:
            return brief_text, ""
        last_error = result.stderr or "Codex brief returned empty output."
        attempt += 1
    return "", last_error


def _run_strategy_plan(
    paths: CompetitionPaths,
    config: AgentPipelineConfig,
    output_dir: Path,
    brief_path: Path,
) -> Path:
    template = _load_template("strategy_plan.md")
    brief_content = _read_text(brief_path)
    knowledge_text = _load_problem_type_knowledge_text(paths=paths, repo_root=config.repo_root)
    compact_mode = False
    prompt_text = _build_strategy_prompt(
        template=template,
        config=config,
        paths=paths,
        brief_content=brief_content,
        compact=compact_mode,
    )
    prompt_text = _append_problem_type_knowledge(prompt_text, knowledge_text)
    if len(prompt_text) > _STRATEGY_PROMPT_MAX_CHARS:
        compact_mode = True
        print("[yellow]note[/yellow]: GPT prompt too large; using compact context to avoid length limits.")
        prompt_text = _build_strategy_prompt(
            template=template,
            config=config,
            paths=paths,
            brief_content=brief_content,
            compact=compact_mode,
        )
        prompt_text = _append_problem_type_knowledge(prompt_text, knowledge_text)

    attempt = 0
    plan_payload: dict[str, object] | None = None
    plan_issue: str | None = None
    while True:
        _assert_no_secrets(prompt_text)
        prompt_path = output_dir / "prompt.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        _print_block("gpt strategy prompt (sent)", prompt_text)

        result = run_strategy(prompt_path, output_dir, dry_run=config.dry_run)
        if result.returncode != 0:
            if "Prompt is too long" in result.stdout and not compact_mode:
                compact_mode = True
                print("[yellow]note[/yellow]: GPT prompt too long; retrying with compact context.")
                prompt_text = _build_strategy_prompt(
                    template=template,
                    config=config,
                    paths=paths,
                    brief_content=brief_content,
                    compact=compact_mode,
                )
                prompt_text = _append_problem_type_knowledge(prompt_text, knowledge_text)
                continue
            if _is_strategy_rate_limited(result.stdout, result.stderr):
                error_text = (result.stderr or result.stdout).strip()
                print("[yellow]note[/yellow]: GPT rate-limited; using fallback strategy.")
                strategy_text, instructions_text, plan_payload = _build_fallback_strategy(
                    paths=paths,
                    config=config,
                    brief_content=brief_content,
                    error_text=error_text,
                )
                transcript = _format_fallback_transcript(error_text)
                return _write_strategy_outputs(
                    paths=paths,
                    strategy_text=strategy_text,
                    instructions_text=instructions_text,
                    transcript_text=transcript,
                    plan_payload=plan_payload,
                )
            raise KaggleBotError(f"GPT strategy failed: {result.stderr or result.stdout}")

        strategy_text, instructions_text = _split_strategy_output(result.stdout)
        plan_payload, plan_issue = _extract_plan_json(result.stdout)
        issues = _validate_strategy_output(
            strategy_text,
            instructions_text,
            plan_payload,
            plan_issue,
            require_sources=config.internet != "off",
        )
        if issues:
            if attempt >= _QUALITY_RETRY_LIMIT:
                issue_text = "\n".join(f"- {issue}" for issue in issues)
                raise KaggleBotError(f"GPT strategy failed quality gate:\n{issue_text}")
            attempt += 1
            print("[yellow]note[/yellow]: GPT output failed quality gate; retrying with stricter instructions.")
            prompt_text = _apply_quality_gate(
                template=template,
                config=config,
                paths=paths,
                brief_content=brief_content,
                compact=compact_mode,
                issues=issues,
            )
            prompt_text = _append_problem_type_knowledge(prompt_text, knowledge_text)
            if len(prompt_text) > _STRATEGY_PROMPT_MAX_CHARS and not compact_mode:
                compact_mode = True
                prompt_text = _apply_quality_gate(
                    template=template,
                    config=config,
                    paths=paths,
                    brief_content=brief_content,
                    compact=compact_mode,
                    issues=issues,
                )
                prompt_text = _append_problem_type_knowledge(prompt_text, knowledge_text)
            continue
        break

    return _write_strategy_outputs(
        paths=paths,
        strategy_text=strategy_text,
        instructions_text=instructions_text,
        transcript_text=result.stdout,
        plan_payload=plan_payload,
    )


def _run_codex_kernel_implementation(
    paths: CompetitionPaths,
    config: AgentPipelineConfig,
    output_dir: Path,
    instructions_path: Path,
) -> None:
    kernel_dir = paths.kernel_source_dir
    ensure_solution_path_allowed(kernel_dir, artifacts_dir=paths.artifacts_dir, slug=paths.slug)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        kernel_path.write_text("# kagglebot kernel\n", encoding="utf-8")

    template = _load_template("codex_kernel_impl.md")
    strategy_path = paths.context_agent_dir / "strategy_plan.md"
    blocked_modules = _load_blocked_modules(paths.context_dir)
    blocked_text = "\n".join(f"- {name}" for name in blocked_modules) if blocked_modules else "None"
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "kernel_dir": str(kernel_dir),
            "kernel_path": str(kernel_path),
            "instructions": _read_text(instructions_path),
            "strategy": _read_text(strategy_path),
            "blocked_modules": blocked_text,
        },
    )
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    allowed_prefixes = [paths.kernel_source_dir, paths.runs_dir, paths.context_dir]
    guard_snapshot = _backup_guarded_files(config.repo_root, allowed_prefixes)
    before = _snapshot_tree(config.repo_root)
    result = run_codex(
        prompt_path,
        output_dir,
        dry_run=config.dry_run,
        model=_PLANNING_CODEX_MODEL,
        reasoning_effort=_PLANNING_REASONING_EFFORT,
    )
    after = _snapshot_tree(config.repo_root)
    _enforce_allowlist_changes(
        root=config.repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="codex_kernel_implementation",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )
    if result.returncode != 0:
        raise KaggleBotError(f"Codex kernel implementation failed: {result.stderr}")
    if not config.dry_run:
        _print_block("codex kernel implementation result", _read_text(result.last_message_path))


def _ensure_context_materials(paths: CompetitionPaths) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    rules_url = _read_text(paths.rules_url_path)
    if not rules_url:
        rules_url = f"https://www.kaggle.com/competitions/{paths.slug}/rules"
        paths.rules_url_path.write_text(rules_url + "\n", encoding="utf-8")

    if not paths.dataset_profile_path.exists():
        from kagglebot.knowledge import build_dataset_profile

        profile = build_dataset_profile(paths.data_dir)
        paths.dataset_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    if not paths.overview_md_path.exists():
        paths.overview_md_path.write_text(f"Overview not available. See {rules_url}\n", encoding="utf-8")
    if not paths.data_md_path.exists():
        paths.data_md_path.write_text(f"Data description not available. See {rules_url}\n", encoding="utf-8")
    if not paths.rules_md_path.exists():
        paths.rules_md_path.write_text(f"Rules not available. See {rules_url}\n", encoding="utf-8")
    if not paths.submission_format_md_path.exists():
        paths.submission_format_md_path.write_text(
            f"Submission format not available. See {rules_url}\n", encoding="utf-8"
        )

    if not paths.sample_submission_path.exists():
        from kagglebot.solver.io import find_competition_files

        try:
            _, _, sample_path = find_competition_files(paths.data_dir)
            paths.sample_submission_path.write_text(sample_path.read_text(encoding="utf-8"), encoding="utf-8")
        except FileNotFoundError:
            paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")
    if not paths.sample_submission_head_path.exists():
        head = _read_sample_submission_head(paths, max_lines=5)
        if head:
            paths.sample_submission_head_path.write_text(head + "\n", encoding="utf-8")


def _load_template(name: str) -> str:
    base_dir = Path(__file__).resolve().parents[1] / "prompts"
    return (base_dir / name).read_text(encoding="utf-8")


def _render_template(template: str, mapping: dict[str, str]) -> str:
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_blocked_modules(context_dir: Path) -> list[str]:
    path = context_dir / "blocked_modules.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload if item]
    return []


def _split_strategy_output(text: str) -> tuple[str, str]:
    strategy_marker = "===STRATEGY==="
    instructions_marker = "===CODEX_INSTRUCTIONS==="
    strategy_lines: list[str] = []
    instructions_lines: list[str] = []
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == strategy_marker:
            current = "strategy"
            continue
        if stripped == instructions_marker:
            current = "instructions"
            continue
        if current == "strategy":
            strategy_lines.append(line)
        elif current == "instructions":
            instructions_lines.append(line)
    strategy_text = "\n".join(strategy_lines).strip()
    instructions_text = "\n".join(instructions_lines).strip()
    if not strategy_text:
        strategy_text = text.strip()
    if not instructions_text:
        instructions_text = text.strip()
    return strategy_text, instructions_text


def _extract_plan_json(text: str) -> tuple[dict[str, object] | None, str | None]:
    marker = "===PLAN_JSON==="
    if marker not in text:
        return None, "PLAN_JSON section missing."
    lines = text.splitlines()
    capture: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == marker:
            in_block = True
            continue
        if in_block and stripped.startswith("==="):
            break
        if in_block:
            capture.append(line)
    raw = "\n".join(capture).strip()
    if not raw:
        return None, "PLAN_JSON section is empty."
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"PLAN_JSON is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "PLAN_JSON must be a JSON object."
    return payload, None


def _validate_plan_payload(payload: dict[str, object]) -> list[str]:
    issues: list[str] = []
    required = [
        "target_metric",
        "target_direction",
        "target_score",
        "score_source",
        "holdout_frac",
        "cv_folds",
        "seed",
        "max_iterations",
        "patience",
        "min_improvement",
    ]
    for key in required:
        if key not in payload:
            issues.append(f"PLAN_JSON missing key: {key}.")
    if payload.get("target_direction") not in ("minimize", "maximize"):
        issues.append("PLAN_JSON target_direction must be 'minimize' or 'maximize'.")
    if payload.get("score_source") not in ("holdout", "cv"):
        issues.append("PLAN_JSON score_source must be 'holdout' or 'cv'.")
    if not isinstance(payload.get("target_score"), (int, float)):
        issues.append("PLAN_JSON target_score must be a number.")
    return issues


def _write_plan_payload(paths: CompetitionPaths, payload: dict[str, object]) -> None:
    existing: dict[str, object] = {}
    if paths.plan_path.exists():
        try:
            existing = json.loads(paths.plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    merged = {**existing, **payload}
    plan = PlanConfig.from_dict(merged)
    paths.plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")


def _write_strategy_outputs(
    *,
    paths: CompetitionPaths,
    strategy_text: str,
    instructions_text: str,
    transcript_text: str,
    plan_payload: dict[str, object] | None,
) -> Path:
    if plan_payload is not None:
        _write_plan_payload(paths, plan_payload)

    transcript_path = paths.context_agent_dir / "strategy_transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")

    _print_block("gpt strategy (human-readable)", strategy_text)
    _print_block("gpt instructions for codex", instructions_text)
    strategy_path = paths.context_agent_dir / "strategy_plan.md"
    instructions_path = paths.context_agent_dir / "codex_instructions.md"
    strategy_path.write_text(strategy_text + "\n", encoding="utf-8")
    instructions_path.write_text(instructions_text + "\n", encoding="utf-8")
    return instructions_path


_STRATEGY_PROMPT_MAX_CHARS = 12000
_LOG_MAX_CHARS = 500
_MIN_STRATEGY_CHARS = 1200
_MIN_INSTRUCTIONS_CHARS = 200
_MIN_SOURCES = 3
_QUALITY_RETRY_LIMIT = 1
_MAX_GUARD_FILE_BYTES = 2_000_000
_PROTECTED_PATHS = (
    "src/",
    "tests/",
    "docs/",
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "AGENTS.md",
    "STRATEGY.md",
    "SECURITY.md",
)
_STRATEGY_RATE_LIMIT_MARKERS = (
    "you've hit your limit",
    "rate limit",
    "too many requests",
    "limit resets",
    "request limit",
    "error 429",
    "status 429",
)


def _build_strategy_prompt(
    *,
    template: str,
    config: AgentPipelineConfig,
    paths: CompetitionPaths,
    brief_content: str,
    compact: bool,
) -> str:
    if compact:
        dataset_profile = _read_text(paths.dataset_profile_path)
        submission_format = _read_text(paths.submission_format_md_path)
        sample_submission_head = _truncate(_read_sample_submission_head(paths), 800)
        brief_content = _truncate(brief_content, 2000)
    else:
        dataset_profile = _read_text(paths.dataset_profile_path)
        submission_format = _read_text(paths.submission_format_md_path)
        sample_submission_head = _read_sample_submission_head(paths)

    prompt = _render_template(
        template,
        {
            "slug": config.slug,
            "compute": config.compute,
            "accelerator": config.accelerator,
            "internet": config.internet,
            "rules_url": _read_text(paths.rules_url_path).strip() or config.competition_url or "",
            "interpretation": brief_content,
            "dataset_profile": dataset_profile,
            "submission_format": submission_format,
            "sample_submission_head": sample_submission_head,
        },
    )
    if compact:
        prompt += "\n\n[COMPACT]\n"
    return prompt


def _apply_quality_gate(
    *,
    template: str,
    config: AgentPipelineConfig,
    paths: CompetitionPaths,
    brief_content: str,
    compact: bool,
    issues: list[str],
) -> str:
    base_prompt = _build_strategy_prompt(
        template=template,
        config=config,
        paths=paths,
        brief_content=brief_content,
        compact=compact,
    )
    issue_text = "\n".join(f"- {issue}" for issue in issues)
    gate = (
        "\n\n[QUALITY_GATE]\n"
        "Your previous response did not meet requirements. Respond again and fix:\n"
        f"{issue_text}\n"
        "Include all required sections, deeper reasoning, and sources (if internet is on).\n"
    )
    return base_prompt + gate


def _append_problem_type_knowledge(prompt_text: str, knowledge_text: str) -> str:
    clean = knowledge_text.strip()
    if not clean:
        return prompt_text
    return f"{prompt_text}\n\n## Problem-Type Knowledge\n{clean}\n"


def _load_problem_type_knowledge_text(*, paths: CompetitionPaths, repo_root: Path, limit: int = 5) -> str:
    try:
        profile_text = _read_text(paths.dataset_profile_path)
        profile = json.loads(profile_text) if profile_text else {}
        if not isinstance(profile, dict):
            profile = {}
        problem_types = derive_problem_types(profile)
        knowledge_paths = KnowledgePaths(workdir=repo_root)
        insights = resolve_problem_type_insights(
            knowledge_paths,
            problem_types,
            limit=limit,
        )
        error_insights = resolve_error_fix_insights(
            knowledge_paths,
            problem_types,
            limit=limit,
        )
        sections = [
            format_problem_type_insights(insights, limit=limit),
            "",
            format_error_fix_insights(error_insights, limit=limit),
        ]
        return "\n".join(section for section in sections if section is not None)
    except Exception:  # noqa: BLE001
        return "No prior problem-type insights available."


def _is_strategy_rate_limited(stdout: str, stderr: str) -> bool:
    haystack = f"{stdout}\n{stderr}".lower()
    return any(marker in haystack for marker in _STRATEGY_RATE_LIMIT_MARKERS)


def _format_fallback_transcript(error_text: str) -> str:
    lines = [
        "GPT unavailable; using fallback strategy.",
    ]
    if error_text:
        lines.append("")
        lines.append("GPT error:")
        lines.append(error_text)
    return "\n".join(lines).strip() + "\n"


def _validate_strategy_output(
    strategy_text: str,
    instructions_text: str,
    plan_payload: dict[str, object] | None,
    plan_issue: str | None,
    *,
    require_sources: bool,
) -> list[str]:
    issues: list[str] = []
    if len(strategy_text.strip()) < _MIN_STRATEGY_CHARS:
        issues.append(f"Strategy is too short (<{_MIN_STRATEGY_CHARS} chars).")
    if len(instructions_text.strip()) < _MIN_INSTRUCTIONS_CHARS:
        issues.append(f"Instructions are too short (<{_MIN_INSTRUCTIONS_CHARS} chars).")
    if "kernel.py" not in instructions_text:
        issues.append("Instructions must mention kernel.py explicitly.")

    required_keywords = [
        "problem",
        "data",
        "candidate",
        "final",
        "train",
        "evaluation",
        "risk",
        "ablation",
        "source",
    ]
    lowered = strategy_text.lower()
    for keyword in required_keywords:
        if keyword not in lowered:
            issues.append(f"Strategy missing required section keyword: {keyword}.")

    if require_sources:
        source_count = _count_source_items(strategy_text)
        if source_count < _MIN_SOURCES:
            issues.append(f"Sources section must include at least {_MIN_SOURCES} items.")
    if plan_payload is None:
        issues.append(plan_issue or "PLAN_JSON section missing or invalid.")
    else:
        plan_errors = _validate_plan_payload(plan_payload)
        issues.extend(plan_errors)
    return issues


def _count_source_items(text: str) -> int:
    in_sources = False
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("## ") or lowered.startswith("### ") or lowered.startswith("#### "):
            title = lowered.lstrip("#").strip()
            in_sources = title.startswith("sources")
            continue
        if lowered.startswith("sources"):
            in_sources = True
            continue
        if in_sources:
            if not stripped:
                if count > 0:
                    break
                continue
            if stripped.startswith(("-", "*")):
                count += 1
                continue
            if stripped[0].isdigit() and stripped[1:2] in (".", ")"):
                count += 1
                continue
            if lowered.startswith("## "):
                break
    return count


def _build_fallback_strategy(
    *,
    paths: CompetitionPaths,
    config: AgentPipelineConfig,
    brief_content: str,
    error_text: str,
) -> tuple[str, str, dict[str, object]]:
    rules_url = _read_text(paths.rules_url_path).strip() or config.competition_url or "unknown"
    dataset_profile = _summarize_dataset_profile(_read_text(paths.dataset_profile_path))
    submission_format = _summarize_submission_format(_read_text(paths.submission_format_md_path))
    sample_submission_head = _truncate(_read_sample_submission_head(paths), 400)

    strategy_lines = [
        "# Fallback Strategy (GPT unavailable)",
        "",
        "## Problem",
        "Use the local context to implement a strong, competition-appropriate model and a valid submission.",
        "",
        "## Data",
        "Use dataset_profile.json, submission_format.md, and sample_submission to infer schema and target.",
        "Brief (Codex):",
        _truncate(brief_content, 800) or "No brief available.",
        "",
        "Dataset profile (trimmed):",
        dataset_profile or "Dataset profile unavailable.",
        "",
        "Submission format (trimmed):",
        submission_format or "Submission format unavailable.",
        "",
        "Sample submission head (trimmed):",
        sample_submission_head or "Sample submission head unavailable.",
        "",
        "## Candidate Approaches",
        "- Candidate 1: Strong tabular model (CatBoost/LightGBM/XGBoost) with tuned hyperparams.",
        "- Candidate 2: Neural model (MLP/transformer) with embeddings for high-cardinality features.",
        "- Candidate 3: Modality-specific approach (CNN for images, transformer for text, seq2seq for sequences).",
        "",
        "## Final",
        "Pick the strongest feasible approach given file formats and time budget.",
        "",
        "## Train",
        "Use a holdout/CV split with a fixed seed and log parameters. Prefer high-capacity models when compute allows.",
        "",
        "## Evaluation",
        "Compute an offline metric when possible; otherwise emit a numeric placeholder to keep the pipeline moving.",
        "",
        "## Risk",
        "Fallback may underperform; prioritize correctness of submission format and stable execution.",
        "",
        "## Ablation",
        "If time permits, compare baseline model vs. linear model; keep the best.",
        "",
        "## Sources",
        f"- Rules URL: {rules_url}",
        f"- Local dataset profile: {paths.dataset_profile_path}",
        f"- Sample submission: {paths.sample_submission_path}",
        f"- Error: {error_text or 'GPT rate limit'}",
    ]
    strategy_text = "\n".join(strategy_lines).strip()
    plan_payload = _build_fallback_plan_payload(paths)

    instructions_lines = [
        "# Fallback Codex Instructions",
        "",
        "GPT was rate-limited. Follow these steps to implement a strong model in kernel.py.",
        "",
        "1) Inspect `/kaggle/input/{slug}/` to locate train/test files and the sample submission.",
        "2) If train/test CSV files exist (tabular):",
        "   - Load train/test with pandas; identify target as column present in train but not test.",
        "   - Identify ID column from sample_submission (first column) if present.",
        "   - Split train into train/valid with a fixed seed.",
        "   - Preprocess: SimpleImputer for numerics, OneHotEncoder/TargetEncoder for categoricals.",
        "   - Model: CatBoost/LightGBM/XGBoost (GPU if available) with tuned hyperparams.",
        "   - Metric: use accuracy/F1/AUC for classification, RMSE/MAE for regression if unknown.",
        "3) If data is non-tabular or target is unclear:",
        "   - Use modality-appropriate models (CNN/transformer/sequence model).",
        "   - Only fall back to sample_submission if no valid training path exists.",
        "4) Always write:",
        "   - `/kaggle/working/submission.csv` matching sample_submission columns and row count.",
        "   - `/kaggle/working/metrics.json` with `offline_value` (numeric) and `metric` name.",
        "5) Keep changes confined to `kernel.py` and helper files under the kernel directory.",
    ]
    instructions_text = "\n".join(instructions_lines).replace("{slug}", config.slug).strip()
    return strategy_text, instructions_text, plan_payload


def _build_fallback_plan_payload(paths: CompetitionPaths) -> dict[str, object]:
    metric = _load_profile_metric(paths) or _infer_metric_from_context(paths) or "rmse"
    direction = infer_direction(metric, "auto")
    top1_score = _load_top1_score(paths)
    if top1_score is None:
        target_score = 0.9 if direction == "maximize" else 0.0
    else:
        target_score = float(top1_score)
    return {
        "target_metric": metric,
        "target_direction": direction,
        "target_score": target_score,
        "score_source": "holdout",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 1,
        "patience": 2,
        "min_improvement": 0.0,
    }


def _load_profile_metric(paths: CompetitionPaths) -> str | None:
    profile_path = paths.dataset_profile_path
    if not profile_path.exists():
        return None
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    metric = payload.get("metric")
    if isinstance(metric, str) and metric.strip():
        return metric.strip()
    return None


def _load_top1_score(paths: CompetitionPaths) -> float | None:
    if not paths.top1_public_path.exists():
        return None
    try:
        payload = json.loads(paths.top1_public_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    score = payload.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _infer_metric_from_context(paths: CompetitionPaths) -> str | None:
    context = "\n".join(
        [
            _read_text(paths.rules_md_path),
            _read_text(paths.overview_md_path),
            _read_text(paths.data_md_path),
            _read_text(paths.submission_format_md_path),
        ]
    ).lower()
    if not context.strip():
        return None
    recall_match = re.search(r"recall\\s*@\\s*(\\d+)", context)
    if recall_match:
        return f"recall_at_{recall_match.group(1)}"
    recall_at_match = re.search(r"recall\\s+at\\s+(\\d+)", context)
    if recall_at_match:
        return f"recall_at_{recall_at_match.group(1)}"
    patterns = [
        (r"mean[_\\s-]?ia[_\\s-]?weighted[_\\s-]?fmax", "mean_ia_weighted_fmax"),
        (r"mean[_\\s-]?weighted[_\\s-]?fmax", "mean_weighted_fmax"),
        (r"\\bfmax\\b|f-?max", "fmax"),
        (r"\\bf1\\b|f1[-\\s]?score", "f1"),
        (r"\\baccuracy\\b", "accuracy"),
        (r"\\brecall\\b", "recall"),
        (r"\\bauc\\b|roc\\s*auc|area\\s+under\\s+the\\s+curve", "auc"),
        (r"log\\s*loss|logloss|cross\\s*entropy", "logloss"),
        (r"rmsle", "rmsle"),
        (r"rmse|root\\s+mean\\s+squared\\s+error", "rmse"),
        (r"mae|mean\\s+absolute\\s+error", "mae"),
        (r"mse|mean\\s+squared\\s+error", "mse"),
        (r"mean\\s+average\\s+precision|\\bmap\\b", "map"),
        (r"r\^?2|r-?squared", "r2"),
    ]
    for pattern, metric in patterns:
        if re.search(pattern, context):
            return metric
    return None


def _print_block(title: str, content: str, *, max_chars: int = _LOG_MAX_CHARS) -> None:
    if content is None:
        content = ""
    trimmed = content.strip()
    if not trimmed:
        print(f"[yellow]{title}[/yellow]: (empty)")
        return
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rstrip()
    preview = truncate_lines(trimmed, max_lines=5)
    print(f"[cyan]{title}[/cyan]:\n{preview}")


def _summarize_dataset_profile(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _truncate(text, 1600)
    if not isinstance(payload, dict):
        return _truncate(text, 1600)
    lines: list[str] = []
    for key in (
        "status",
        "metric",
        "target_column",
        "id_column",
        "train_rows",
        "train_cols",
        "test_rows",
        "test_cols",
        "file_count",
    ):
        if key in payload:
            lines.append(f"- {key}: {payload.get(key)}")
    if "file_extension_counts" in payload:
        lines.append(f"- file_extension_counts: {payload.get('file_extension_counts')}")
    if "file_samples" in payload:
        samples = payload.get("file_samples") or []
        if isinstance(samples, list):
            lines.append(f"- file_samples: {samples[:8]}")
    if "missingness" in payload:
        lines.append(f"- missingness: {payload.get('missingness')}")
    if "categorical_columns" in payload:
        cols = payload.get("categorical_columns") or []
        if isinstance(cols, list):
            lines.append(f"- categorical_columns: {len(cols)}")
    if "high_cardinality_columns" in payload:
        cols = payload.get("high_cardinality_columns") or []
        if isinstance(cols, list):
            lines.append(f"- high_cardinality_columns: {len(cols)}")
    if "train_only_columns" in payload:
        cols = payload.get("train_only_columns") or []
        if isinstance(cols, list):
            lines.append(f"- train_only_columns: {_format_column_list(cols)}")
    if "test_only_columns" in payload:
        cols = payload.get("test_only_columns") or []
        if isinstance(cols, list):
            lines.append(f"- test_only_columns: {_format_column_list(cols)}")
    if "tags" in payload:
        lines.append(f"- tags: {payload.get('tags')}")
    if "error" in payload:
        lines.append(f"- error: {payload.get('error')}")
    return "\n".join(lines).strip() or _truncate(text, 1600)


def _format_column_list(values: list[object], *, max_items: int = 12) -> str:
    cleaned = [str(v) for v in values if str(v)]
    if not cleaned:
        return "[]"
    if len(cleaned) <= max_items:
        return str(cleaned)
    remainder = len(cleaned) - max_items
    return f"{cleaned[:max_items]} ... +{remainder} more"


def _summarize_submission_format(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= 80 and len(text) <= 2000:
        return text
    preview = "\n".join(lines[:80])
    return _truncate(preview, 2000)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... (truncated)"


def _read_sample_submission_head(paths: CompetitionPaths, max_lines: int = 5) -> str:
    head_path = paths.sample_submission_head_path
    if head_path.exists():
        return _read_text(head_path)
    sample_path = paths.sample_submission_path
    if not sample_path.exists():
        return ""
    lines: list[str] = []
    try:
        with sample_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(max_lines):
                line = handle.readline()
                if not line:
                    break
                lines.append(line)
    except OSError:
        return ""
    return "".join(lines).strip()


def _extract_headings(path: Path, max_items: int = 20) -> list[str]:
    text = _read_text(path)
    if not text:
        return []
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                headings.append(heading)
        if len(headings) >= max_items:
            break
    return headings


def _format_heading_list(headings: list[str]) -> str:
    if not headings:
        return "No headings found."
    return "\n".join(f"- {heading}" for heading in headings)


def _build_fallback_brief(paths: CompetitionPaths, config: AgentPipelineConfig, error_text: str) -> str:
    profile = _read_text(paths.dataset_profile_path).strip()
    submission_format = _read_text(paths.submission_format_md_path).strip()
    sample_submission_head = _read_sample_submission_head(paths)
    overview_headings = _extract_headings(paths.overview_md_path)
    data_headings = _extract_headings(paths.data_md_path)
    rules_headings = _extract_headings(paths.rules_md_path)

    lines = [
        "# Fallback Brief (Codex output missing)",
        "",
        f"- slug: {config.slug}",
        f"- url: {config.competition_url or 'unknown'}",
        f"- compute: {config.compute} ({config.accelerator})",
        f"- rules_url: {_read_text(paths.rules_url_path).strip() or 'unknown'}",
        "",
        "## Error",
        error_text or "Codex brief returned empty output.",
        "",
        "## Context paths",
        f"- overview: {paths.overview_md_path}",
        f"- data: {paths.data_md_path}",
        f"- rules: {paths.rules_md_path}",
        f"- dataset_profile: {paths.dataset_profile_path}",
        f"- submission_format: {paths.submission_format_md_path}",
        f"- sample_submission_head: {paths.sample_submission_head_path}",
        f"- sample_submission: {paths.sample_submission_path}",
        "",
        "## Rules headings",
        _format_heading_list(rules_headings),
        "",
        "## Overview headings",
        _format_heading_list(overview_headings),
        "",
        "## Data headings",
        _format_heading_list(data_headings),
        "",
        "## Sample Submission (head)",
        sample_submission_head or "Sample submission head unavailable.",
        "",
        "## Dataset Profile",
        profile if profile else "Dataset profile unavailable.",
        "",
        "## Submission Format",
        submission_format if submission_format else "Submission format unavailable.",
    ]
    return "\n".join(lines).rstrip()


@dataclass(frozen=True)
class GuardSnapshot:
    backup: dict[str, bytes]
    oversized: set[str]


_NOISE_PREFIXES = (
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".cache/",
)
_NOISE_SUFFIXES = (".pyc", ".pyo", ".DS_Store")


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        stat = path.stat()
        snapshot[rel] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _backup_guarded_files(root: Path, allowed_prefixes: list[Path]) -> GuardSnapshot:
    allowed = _allowed_prefixes(root, allowed_prefixes)
    backup: dict[str, bytes] = {}
    oversized: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        if _is_noise_path(rel):
            continue
        if not _is_protected_path(rel):
            continue
        if _is_allowed(rel, allowed):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_GUARD_FILE_BYTES:
            oversized.add(rel)
            continue
        try:
            backup[rel] = path.read_bytes()
        except OSError:
            continue
    return GuardSnapshot(backup=backup, oversized=oversized)


def _enforce_allowlist_changes(
    *,
    root: Path,
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
    allowed_prefixes: list[Path],
    stage: str,
    guard_snapshot: GuardSnapshot | None = None,
    auto_repair: bool = False,
) -> None:
    allowed = _allowed_prefixes(root, allowed_prefixes)
    changed = _diff_snapshots(before, after)
    unauthorized = [path for path in changed if not _is_allowed(path, allowed) and not _is_noise_path(path)]
    if not unauthorized:
        return
    if auto_repair and guard_snapshot is not None:
        errors = _repair_unauthorized_changes(root, unauthorized, guard_snapshot, before)
        after_repair = _snapshot_tree(root)
        changed = _diff_snapshots(before, after_repair)
        unauthorized = [path for path in changed if not _is_allowed(path, allowed) and not _is_noise_path(path)]
        unauthorized = _filter_restored_paths(root, unauthorized, guard_snapshot)
        if not unauthorized:
            return
        if errors:
            issue_text = "\n".join(f"- {error}" for error in errors)
            raise KaggleBotError(
                f"Agent write-guard failed in {stage} after repair:\n{issue_text}\nRemaining: {unauthorized}"
            )
    raise KaggleBotError(f"Agent write-guard failed in {stage}: {unauthorized}")


def _diff_snapshots(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    changed: list[str] = []
    for path, meta in after.items():
        if before.get(path) != meta:
            changed.append(path)
    for path in before:
        if path not in after:
            changed.append(path)
    return sorted(set(changed))


def _allowed_prefixes(root: Path, allowed_prefixes: list[Path]) -> list[str]:
    allowed: list[str] = []
    for prefix in allowed_prefixes:
        try:
            rel = prefix.relative_to(root).as_posix()
        except ValueError:
            continue
        allowed.append(rel.rstrip("/") + "/")
    return allowed


def _is_noise_path(path: str) -> bool:
    for prefix in _NOISE_PREFIXES:
        if path.startswith(prefix):
            return True
        if f"/{prefix.strip('/')}/" in path:
            return True
    if "/__pycache__/" in path or path.endswith("/__pycache__") or path == "__pycache__":
        return True
    for suffix in _NOISE_SUFFIXES:
        if path.endswith(suffix):
            return True
    return False


def _is_protected_path(path: str) -> bool:
    if path.startswith("artifacts/"):
        # Protect kernel artifacts so stray edits can be restored by the guard.
        return "/kernel/" in path
    for entry in _PROTECTED_PATHS:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
            continue
        if path == entry:
            return True
    return False


def _filter_restored_paths(root: Path, unauthorized: list[str], guard_snapshot: GuardSnapshot | None) -> list[str]:
    if guard_snapshot is None:
        return unauthorized
    filtered: list[str] = []
    for rel in unauthorized:
        path = root / rel
        if not path.exists():
            continue
        original = guard_snapshot.backup.get(rel)
        if original is None:
            filtered.append(rel)
            continue
        try:
            if path.read_bytes() == original:
                continue
        except OSError:
            filtered.append(rel)
            continue
        filtered.append(rel)
    return filtered


def _repair_unauthorized_changes(
    root: Path,
    unauthorized: list[str],
    guard_snapshot: GuardSnapshot,
    before: dict[str, tuple[int, int]],
) -> list[str]:
    errors: list[str] = []
    for rel in unauthorized:
        if _is_noise_path(rel):
            _remove_path(root / rel)
            continue
        if rel in guard_snapshot.oversized:
            errors.append(f"Cannot restore oversized file: {rel}")
            continue
        if rel in guard_snapshot.backup:
            try:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(guard_snapshot.backup[rel])
            except OSError as exc:
                errors.append(f"Failed to restore {rel}: {exc}")
            continue
        if rel not in before:
            _remove_path(root / rel)
            continue
        if _is_protected_path(rel):
            errors.append(f"Cannot restore protected file: {rel}")
            continue
        errors.append(f"Cannot auto-repair changed file: {rel}")
    return errors


def _remove_path(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _is_allowed(path: str, allowed_prefixes: list[str]) -> bool:
    for prefix in allowed_prefixes:
        if path.startswith(prefix):
            return True
    return False


def _assert_no_secrets(text: str) -> None:
    matches = scan_text_for_secrets(text)
    if matches:
        raise KaggleBotError(f"Secret pattern detected in prompt text: {matches}")
