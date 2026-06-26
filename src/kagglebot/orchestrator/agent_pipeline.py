from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.identity import (
    BRIEF_AGENT,
    IMPLEMENTATION_AGENT,
    STRATEGY_AGENT,
    prompt_identity_mapping,
    render_prompt_identity,
)
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.exceptions import KaggleBotError
from kagglebot.hardware import render_hardware_constraints, resolve_hardware_profile
from kagglebot.json_utils import load_json_array, load_json_object, parse_json_object_text, write_json_object
from kagglebot.knowledge import (
    derive_problem_types,
    record_research_artifacts,
    resolve_research_paths_for_slug,
)
from kagglebot.knowledge_context import load_problem_type_knowledge_text
from kagglebot.logging_utils import truncate_lines
from kagglebot.method_scout import render_method_registry_for_prompt, run_method_scout
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.plan_policy import (
    is_high_accuracy_tabular_blend_target,
    normalize_plan_payload,
    repair_plan_payload_for_profile,
    validate_plan_payload,
    write_plan_payload,
)
from kagglebot.solution_guard import ensure_solution_path_allowed
from kagglebot.solver.metrics import infer_direction
from kagglebot.write_guard import (
    WriteGuardPolicy,
    _assert_no_secrets,
    _backup_guarded_files,
    _default_external_guard_paths,
    _enforce_allowlist_changes,
    _repo_root_write_policy,
    _snapshot_tree,
)
from kagglebot.writeup import (
    infer_deliverable_mode_from_paths,
    infer_submit_mode_from_paths,
    summarize_writeup_requirements,
)

_PLANNING_CODEX_MODEL = IMPLEMENTATION_AGENT.model
_PLANNING_REASONING_EFFORT = IMPLEMENTATION_AGENT.reasoning_effort


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
    method_scout: str = "auto"
    method_scout_max_sources: int = 12
    hardware_profile: str | None = "auto"
    time_budget_min: int | None = None


@dataclass(frozen=True)
class CodexBriefStage:
    paths: CompetitionPaths
    config: AgentPipelineConfig
    output_dir: Path

    def run(self) -> Path:
        return _run_codex_brief(self.paths, self.config, self.output_dir)


@dataclass(frozen=True)
class StrategyStage:
    paths: CompetitionPaths
    config: AgentPipelineConfig
    output_dir: Path

    def run(self, *, brief_path: Path) -> Path:
        return _run_strategy_plan(self.paths, self.config, self.output_dir, brief_path)


@dataclass(frozen=True)
class CodexImplementationStage:
    paths: CompetitionPaths
    config: AgentPipelineConfig
    output_dir: Path

    def run(self, *, instructions_path: Path) -> None:
        _run_codex_kernel_implementation(self.paths, self.config, self.output_dir, instructions_path)


@dataclass(frozen=True)
class AgentPipeline:
    paths: CompetitionPaths
    config: AgentPipelineConfig

    def run(self) -> None:
        self.paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
        _ensure_context_materials(self.paths)

        brief_dir = self.paths.context_agent_dir / "brief"
        strategy_dir = self.paths.context_agent_dir / "strategy"
        implement_dir = self.paths.context_agent_dir / "implement"
        brief_dir.mkdir(parents=True, exist_ok=True)
        strategy_dir.mkdir(parents=True, exist_ok=True)
        implement_dir.mkdir(parents=True, exist_ok=True)

        brief_stage = CodexBriefStage(paths=self.paths, config=self.config, output_dir=brief_dir)
        strategy_stage = StrategyStage(paths=self.paths, config=self.config, output_dir=strategy_dir)
        implement_stage = CodexImplementationStage(paths=self.paths, config=self.config, output_dir=implement_dir)

        brief_path = brief_stage.run()
        instructions_path = strategy_stage.run(brief_path=brief_path)
        implement_stage.run(instructions_path=instructions_path)


def run_agent_pipeline(*, paths: CompetitionPaths, config: AgentPipelineConfig) -> None:
    AgentPipeline(paths=paths, config=config).run()


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
            "code_path": str(paths.code_md_path),
            "code_index_path": str(paths.code_notebooks_index_path),
            "models_path": str(paths.models_md_path),
            "discussion_path": str(paths.discussion_md_path),
            "discussion_threads_dir": str(paths.discussion_threads_dir),
            "discussion_index_path": str(paths.discussion_threads_index_path),
        },
    )
    prompt_text = _append_problem_type_knowledge(
        prompt_text,
        _load_problem_type_knowledge_text(paths=paths, repo_root=config.repo_root),
    )
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _print_block(f"{BRIEF_AGENT.log_alias} brief prompt (sent)", prompt_text)

    brief_path = paths.context_agent_dir / "brief_for_strategy.md"
    brief_text, error_text = _run_codex_brief_with_retry(
        prompt_path=prompt_path,
        output_dir=output_dir,
        paths=paths,
        config=config,
    )
    if not brief_text:
        brief_text = _build_fallback_brief(paths, config, error_text)
    _print_block(f"{BRIEF_AGENT.log_alias} brief (used for {STRATEGY_AGENT.log_alias})", brief_text)
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
        guard_root = paths.base_dir
        write_policy = WriteGuardPolicy(
            allowed_prefixes=(paths.context_agent_dir,),
            denied_prefixes=(paths.data_dir, paths.kernels_dir),
            external_guard_paths=_default_external_guard_paths(config.repo_root),
        )
        guard_snapshot = _backup_guarded_files(guard_root, write_policy)
        before = _snapshot_tree(guard_root)
        result = run_codex(
            prompt_path,
            output_dir,
            dry_run=config.dry_run,
            model=_PLANNING_CODEX_MODEL,
            reasoning_effort=_PLANNING_REASONING_EFFORT,
        )
        after = _snapshot_tree(guard_root)
        _enforce_allowlist_changes(
            root=guard_root,
            before=before,
            after=after,
            allowed_prefixes=write_policy,
            stage="codex_brief",
            guard_snapshot=guard_snapshot,
            auto_repair=True,
        )
        brief_text = _read_text(result.last_message_path).strip()
        if result.returncode == 0 and brief_text:
            return brief_text, ""
        last_error = result.stderr or f"{BRIEF_AGENT.display_name} brief returned empty output."
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
    profile = _load_dataset_profile_payload(paths)
    problem_types = derive_problem_types(profile)
    if config.method_scout != "off":
        run_method_scout(
            paths=paths,
            slug=config.slug,
            problem_types=problem_types,
            dataset_profile=profile,
            metric=str(profile.get("target_metric") or ""),
            mode=config.method_scout,
            max_sources=config.method_scout_max_sources,
        )
    compact_mode = False
    prompt_text = _build_strategy_prompt(
        template=template,
        config=config,
        paths=paths,
        brief_content=brief_content,
        compact=compact_mode,
    )
    prompt_text = _append_problem_type_knowledge_with_budget(
        prompt_text,
        knowledge_text,
        compact=compact_mode,
    )
    prompt_text = _append_method_registry_with_budget(prompt_text, paths=paths, compact=compact_mode)
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
        prompt_text = _append_problem_type_knowledge_with_budget(
            prompt_text,
            knowledge_text,
            compact=compact_mode,
        )
        prompt_text = _append_method_registry_with_budget(prompt_text, paths=paths, compact=compact_mode)

    attempt = 0
    plan_payload: dict[str, object] | None = None
    plan_issue: str | None = None
    research_sources_text: str = ""
    research_sources_issue: str | None = None
    research_summary_text: str = ""
    research_summary_issue: str | None = None
    while True:
        _assert_no_secrets(prompt_text)
        prompt_path = output_dir / "prompt.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        _print_block("gpt strategy prompt (sent)", prompt_text)

        result = run_strategy(prompt_path, output_dir, dry_run=config.dry_run)
        response_text = _select_strategy_response_text(result)
        if result.returncode != 0:
            if "Prompt is too long" in response_text and not compact_mode:
                compact_mode = True
                print("[yellow]note[/yellow]: GPT prompt too long; retrying with compact context.")
                prompt_text = _build_strategy_prompt(
                    template=template,
                    config=config,
                    paths=paths,
                    brief_content=brief_content,
                    compact=compact_mode,
                )
                prompt_text = _append_problem_type_knowledge_with_budget(
                    prompt_text,
                    knowledge_text,
                    compact=compact_mode,
                )
                prompt_text = _append_method_registry_with_budget(prompt_text, paths=paths, compact=compact_mode)
                continue
            if _should_use_fallback_strategy(response_text, result.stderr):
                error_text = (result.stderr or response_text).strip()
                print("[yellow]note[/yellow]: GPT strategy failed; using fallback strategy.")
                (
                    strategy_text,
                    instructions_text,
                    plan_payload,
                    research_sources_text,
                    research_summary_text,
                ) = _build_fallback_strategy(
                    paths=paths,
                    config=config,
                    brief_content=brief_content,
                    error_text=error_text,
                )
                transcript = _format_fallback_transcript(error_text)
                return _write_strategy_outputs(
                    paths=paths,
                    knowledge_paths=KnowledgePaths(workdir=config.repo_root),
                    method_scout_mode=config.method_scout,
                    method_scout_max_sources=config.method_scout_max_sources,
                    strategy_text=strategy_text,
                    instructions_text=instructions_text,
                    transcript_text=transcript,
                    plan_payload=plan_payload,
                    research_sources_text=research_sources_text,
                    research_summary_text=research_summary_text,
                )
            raise KaggleBotError(f"{STRATEGY_AGENT.display_name} strategy failed: {result.stderr or response_text}")

        strategy_text, instructions_text = _split_strategy_output(response_text)
        instructions_text = _ensure_kernel_instruction_reference(
            instructions_text=instructions_text,
            slug=config.slug,
        )
        research_sources_text, research_sources_issue = _extract_research_sources_jsonl(response_text)
        research_summary_text, research_summary_issue = _extract_research_summary(response_text)
        plan_payload, plan_issue = _extract_plan_json(response_text)
        profile = _load_dataset_profile_payload(paths)
        if plan_payload is not None:
            plan_payload = repair_plan_payload_for_profile(plan_payload, profile)
        issues = _validate_strategy_output(
            strategy_text,
            instructions_text,
            plan_payload,
            plan_issue,
            research_sources_text,
            research_sources_issue,
            research_summary_text,
            research_summary_issue,
            profile=profile,
            require_sources=config.internet != "off",
        )
        if issues:
            if attempt >= _QUALITY_RETRY_LIMIT:
                issue_text = "\n".join(f"- {issue}" for issue in issues)
                raise KaggleBotError(f"{STRATEGY_AGENT.display_name} strategy failed quality gate:\n{issue_text}")
            attempt += 1
            for issue in issues:
                print(f"[yellow]quality gate[/yellow]: {issue}")
            print("[yellow]note[/yellow]: GPT output failed quality gate; retrying with stricter instructions.")
            prompt_text = _apply_quality_gate(
                template=template,
                config=config,
                paths=paths,
                brief_content=brief_content,
                compact=compact_mode,
                issues=issues,
            )
            prompt_text = _append_problem_type_knowledge_with_budget(
                prompt_text,
                knowledge_text,
                compact=compact_mode,
            )
            prompt_text = _append_method_registry_with_budget(prompt_text, paths=paths, compact=compact_mode)
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
                prompt_text = _append_problem_type_knowledge_with_budget(
                    prompt_text,
                    knowledge_text,
                    compact=compact_mode,
                )
                prompt_text = _append_method_registry_with_budget(prompt_text, paths=paths, compact=compact_mode)
            continue
        break

    return _write_strategy_outputs(
        paths=paths,
        knowledge_paths=KnowledgePaths(workdir=config.repo_root),
        method_scout_mode=config.method_scout,
        method_scout_max_sources=config.method_scout_max_sources,
        strategy_text=strategy_text,
        instructions_text=instructions_text,
        transcript_text=response_text,
        plan_payload=plan_payload,
        research_sources_text=research_sources_text,
        research_summary_text=research_summary_text,
    )


def _select_strategy_response_text(result: object) -> str:
    primary = str(getattr(result, "stdout", "") or "")
    if _contains_required_strategy_sections(primary):
        return primary
    transcript_path = getattr(result, "transcript_path", None)
    if isinstance(transcript_path, Path) and transcript_path.exists():
        transcript = _read_text(transcript_path)
        if _contains_required_strategy_sections(transcript):
            print("[yellow]note[/yellow]: using strategy full transcript for quality parsing.")
            return transcript
    return primary


def _contains_required_strategy_sections(text: str) -> bool:
    if not text:
        return False
    required_markers = ("===STRATEGY===", "===PLAN_JSON===", "===CODEX_INSTRUCTIONS===")
    return all(marker in text for marker in required_markers)


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
    knowledge_paths = KnowledgePaths(workdir=config.repo_root)
    _, knowledge_summary_path = resolve_research_paths_for_slug(knowledge_paths=knowledge_paths, slug=paths.slug)
    research_summary_path = knowledge_summary_path or (paths.context_dir / "research_summary.md")
    blocked_modules = _resolve_blocked_modules_for_runtime(paths.context_dir, compute=config.compute)
    blocked_text = "\n".join(f"- {name}" for name in blocked_modules) if blocked_modules else "None"
    prompt_text = _render_template(
        template,
        {
            "slug": config.slug,
            "kernel_dir": str(kernel_dir),
            "kernel_path": str(kernel_path),
            "instructions": _read_text(instructions_path),
            "strategy": _read_text(strategy_path),
            "plan_path": str(paths.plan_path),
            "research_summary_path": str(research_summary_path),
            "blocked_modules": blocked_text,
        },
    )
    if paths.method_registry_path.exists():
        registry = load_json_object(paths.method_registry_path)
        if registry is not None:
            method_text = render_method_registry_for_prompt(registry, max_methods=8)
            if method_text.strip():
                prompt_text += (
                    "\n\n## Competition-Specific Method Scout\n"
                    f"{method_text}\n"
                    "Use these method candidates as the competition-specific shortlist. "
                    "Do not implement blocked methods; use fallbacks for unavailable optional dependencies.\n"
                )
    if infer_deliverable_mode_from_paths(paths) == "writeup":
        requirements = summarize_writeup_requirements(paths)
        prompt_text += (
            "\n\nWriteup mode is active for this competition.\n"
            "Do not treat submission.csv as the primary deliverable. Keep train/eval outputs useful as proxy "
            "evidence, and make sure the implementation can support a final judged writeup package.\n"
        )
        if requirements:
            prompt_text += f"Competition-specific writeup requirements:\n{requirements}\n"
    _assert_no_secrets(prompt_text)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    write_policy = _repo_root_write_policy(
        repo_root=config.repo_root,
        denied_prefixes=[paths.data_dir, paths.kernels_dir],
    )
    guard_snapshot = _backup_guarded_files(config.repo_root, write_policy)
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
        allowed_prefixes=write_policy,
        stage="codex_kernel_implementation",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )
    if result.returncode != 0:
        raise KaggleBotError(f"{IMPLEMENTATION_AGENT.display_name} kernel implementation failed: {result.stderr}")
    if not config.dry_run:
        _print_block(
            f"{IMPLEMENTATION_AGENT.log_alias} kernel implementation result",
            _read_text(result.last_message_path),
        )


def _ensure_context_materials(paths: CompetitionPaths) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    rules_url = _read_text(paths.rules_url_path)
    if not rules_url:
        rules_url = f"https://www.kaggle.com/competitions/{paths.slug}/rules"
        paths.rules_url_path.write_text(rules_url + "\n", encoding="utf-8")

    if not paths.dataset_profile_path.exists():
        from kagglebot.knowledge import build_dataset_profile

        profile = build_dataset_profile(paths.data_dir)
        write_json_object(paths.dataset_profile_path, profile)

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
    competition_base_url = f"https://www.kaggle.com/competitions/{paths.slug}"
    if not paths.code_md_path.exists():
        paths.code_md_path.write_text(
            f"Code tab snapshot not available. See {competition_base_url}/code\n",
            encoding="utf-8",
        )
    if not paths.models_md_path.exists():
        paths.models_md_path.write_text(
            f"Models tab snapshot not available. See {competition_base_url}/models\n",
            encoding="utf-8",
        )
    if not paths.discussion_md_path.exists():
        paths.discussion_md_path.write_text(
            f"Discussion tab snapshot not available. See {competition_base_url}/discussions\n",
            encoding="utf-8",
        )
    paths.code_notebooks_dir.mkdir(parents=True, exist_ok=True)
    if not paths.code_notebooks_index_path.exists():
        write_json_object(
            paths.code_notebooks_index_path,
            {"source_url": f"{competition_base_url}/code", "notebook_count": 0, "notebooks": []},
        )
    paths.discussion_threads_dir.mkdir(parents=True, exist_ok=True)
    if not paths.discussion_threads_index_path.exists():
        write_json_object(
            paths.discussion_threads_index_path,
            {"source_url": f"{competition_base_url}/discussions", "thread_count": 0, "threads": []},
        )

    sample_needs_refresh = (not paths.sample_submission_path.exists()) or (
        not _has_data_rows(paths.sample_submission_path)
    )
    if sample_needs_refresh:
        from kagglebot.solver.io import ensure_sample_submission

        resolved = ensure_sample_submission(paths.data_dir)
        if resolved is not None and resolved.exists() and _has_data_rows(resolved):
            paths.sample_submission_path.write_text(resolved.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            paths.sample_submission_path.write_text("id,prediction\n", encoding="utf-8")
    if sample_needs_refresh or (not paths.sample_submission_head_path.exists()):
        head = _read_sample_submission_head(paths, max_lines=5)
        if head:
            paths.sample_submission_head_path.write_text(head + "\n", encoding="utf-8")


def _load_template(name: str) -> str:
    base_dir = Path(__file__).resolve().parents[1] / "prompts"
    return render_prompt_identity((base_dir / name).read_text(encoding="utf-8"))


def _render_template(template: str, mapping: dict[str, str]) -> str:
    rendered = template
    merged = {**prompt_identity_mapping(), **mapping}
    for key, value in merged.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_dataset_profile_payload(paths: CompetitionPaths) -> dict[str, object]:
    raw = _read_text(paths.dataset_profile_path).strip()
    if not raw:
        return {}
    return parse_json_object_text(raw) or {}


def _has_data_rows(path: Path) -> bool:
    non_empty = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
                if non_empty >= 2:
                    return True
    except OSError:
        return False
    return False


def _load_blocked_modules(context_dir: Path) -> list[str]:
    payload = load_json_array(context_dir / "blocked_modules.json")
    return [str(item) for item in payload if item] if payload is not None else []


def _resolve_blocked_modules_for_runtime(context_dir: Path, *, compute: str) -> list[str]:
    blocked = set(_load_blocked_modules(context_dir))
    if compute.startswith("local"):
        for module_name in ("xgboost",):
            if importlib.util.find_spec(module_name) is None:
                blocked.add(module_name)
    return sorted(blocked)


def _split_strategy_output(text: str) -> tuple[str, str]:
    strategy_text, _ = _extract_marked_section(text, "===STRATEGY===")
    instructions_text, _ = _extract_marked_section(text, "===CODEX_INSTRUCTIONS===")
    if not strategy_text:
        strategy_text = text.strip()
    if not instructions_text:
        instructions_text = text.strip()
    return strategy_text, instructions_text


def _ensure_kernel_instruction_reference(*, instructions_text: str, slug: str) -> str:
    if "kernel.py" in instructions_text:
        return instructions_text
    injected = f"Primary entrypoint: `artifacts/{slug}/kernel/kernel.py`; update `kernel.py` directly.\n\n"
    print("[yellow]note[/yellow]: Injected missing kernel.py reference into strategy instructions.")
    return injected + instructions_text


def _extract_marked_section(text: str, marker: str) -> tuple[str, str | None]:
    if marker not in text:
        return "", f"{marker} section missing."
    capture: list[str] = []
    in_block = False
    for line in text.splitlines():
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
        return "", f"{marker} section is empty."
    return raw, None


def _extract_research_sources_jsonl(text: str) -> tuple[str, str | None]:
    return _extract_marked_section(text, "===RESEARCH_SOURCES_JSONL===")


def _extract_research_summary(text: str) -> tuple[str, str | None]:
    return _extract_marked_section(text, "===RESEARCH_SUMMARY_MD===")


def _extract_plan_json(text: str) -> tuple[dict[str, object] | None, str | None]:
    raw, issue = _extract_marked_section(text, "===PLAN_JSON===")
    if issue:
        return None, issue.replace("===PLAN_JSON===", "PLAN_JSON")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"PLAN_JSON is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "PLAN_JSON must be a JSON object."
    return normalize_plan_payload(payload), None


def _write_strategy_outputs(
    *,
    paths: CompetitionPaths,
    knowledge_paths: KnowledgePaths,
    method_scout_mode: str = "auto",
    method_scout_max_sources: int = 12,
    strategy_text: str,
    instructions_text: str,
    transcript_text: str,
    plan_payload: dict[str, object] | None,
    research_sources_text: str,
    research_summary_text: str,
) -> Path:
    if plan_payload is not None:
        write_plan_payload(paths, plan_payload)

    research_sources_path = paths.context_dir / "research_sources.jsonl"
    research_summary_path = paths.context_dir / "research_summary.md"
    if research_sources_text.strip():
        research_sources_path.write_text(research_sources_text.strip() + "\n", encoding="utf-8")
    if research_summary_text.strip():
        research_summary_path.write_text(research_summary_text.strip() + "\n", encoding="utf-8")
    if research_sources_text.strip() and research_summary_text.strip():
        persisted = _persist_research_to_knowledge(
            paths=paths,
            knowledge_paths=knowledge_paths,
            research_sources_text=research_sources_text,
            research_summary_text=research_summary_text,
        )
        info_path = paths.context_dir / "research_storage.json"
        write_json_object(info_path, persisted)
    profile = _load_dataset_profile_payload(paths)
    if method_scout_mode != "off" and research_sources_text.strip():
        method_registry = run_method_scout(
            paths=paths,
            slug=paths.slug,
            problem_types=derive_problem_types(profile),
            dataset_profile=profile,
            metric=str((plan_payload or {}).get("target_metric") or profile.get("target_metric") or ""),
            mode="refresh",
            max_sources=method_scout_max_sources,
        )
        _print_block("method scout registry", render_method_registry_for_prompt(method_registry, max_methods=8))

    transcript_path = paths.context_agent_dir / "strategy_transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")

    _print_block(f"{STRATEGY_AGENT.log_alias} strategy (human-readable)", strategy_text)
    _print_block(
        f"{STRATEGY_AGENT.log_alias} instructions for {IMPLEMENTATION_AGENT.log_alias}",
        instructions_text,
    )
    _print_block("research summary", research_summary_text)
    strategy_path = paths.context_agent_dir / "strategy_plan.md"
    instructions_path = paths.context_agent_dir / "codex_instructions.md"
    strategy_path.write_text(strategy_text + "\n", encoding="utf-8")
    instructions_path.write_text(instructions_text + "\n", encoding="utf-8")
    return instructions_path


def _persist_research_to_knowledge(
    *,
    paths: CompetitionPaths,
    knowledge_paths: KnowledgePaths,
    research_sources_text: str,
    research_summary_text: str,
) -> dict[str, object]:
    profile = _load_dataset_profile_payload(paths)
    problem_types = derive_problem_types(profile)
    persisted = record_research_artifacts(
        knowledge_paths=knowledge_paths,
        slug=paths.slug,
        problem_types=problem_types,
        research_sources_jsonl=research_sources_text,
        research_summary_md=research_summary_text,
    )
    return {
        "slug": paths.slug,
        "problem_types": problem_types,
        "primary_problem_type": persisted.get("primary_problem_type"),
        "knowledge_sources_path": persisted.get("sources_path"),
        "knowledge_summary_path": persisted.get("summary_path"),
        "artifacts_sources_path": str(paths.context_dir / "research_sources.jsonl"),
        "artifacts_summary_path": str(paths.context_dir / "research_summary.md"),
    }


_STRATEGY_PROMPT_MAX_CHARS = 18000
_LOG_MAX_CHARS = 500
_MIN_STRATEGY_CHARS = 1200
_MIN_INSTRUCTIONS_CHARS = 200
_MIN_SOURCES = 3
_MIN_RESEARCH_SUMMARY_CHARS = 300
_MIN_RESEARCH_ITEMS = 3
_QUALITY_RETRY_LIMIT = 1
_STRATEGY_RATE_LIMIT_MARKERS = (
    "you've hit your limit",
    "rate limit",
    "too many requests",
    "limit resets",
    "request limit",
    "error 429",
    "status 429",
)
_STRATEGY_TIMEOUT_MARKERS = (
    "strategy runner timed out",
    "timed out",
    "timeout",
    "deadline exceeded",
)


def _build_strategy_prompt(
    *,
    template: str,
    config: AgentPipelineConfig,
    paths: CompetitionPaths,
    brief_content: str,
    compact: bool,
) -> str:
    profile = _load_dataset_profile_payload(paths)
    if compact:
        dataset_profile = _truncate(_read_text(paths.dataset_profile_path), 2000)
        submission_format = _truncate(_read_text(paths.submission_format_md_path), 2000)
        sample_submission_head = _truncate(_read_sample_submission_head(paths), 800)
        code_snapshot = _truncate(_read_text(paths.code_md_path), 2500)
        models_snapshot = _truncate(_read_text(paths.models_md_path), 1200)
        discussion_snapshot = _truncate(_read_text(paths.discussion_md_path), 1200)
        brief_content = _truncate(brief_content, 2000)
    else:
        dataset_profile = _truncate(_read_text(paths.dataset_profile_path), 6000)
        submission_format = _truncate(_read_text(paths.submission_format_md_path), 4000)
        sample_submission_head = _truncate(_read_sample_submission_head(paths), 1200)
        code_snapshot = _truncate(_read_text(paths.code_md_path), 9000)
        models_snapshot = _truncate(_read_text(paths.models_md_path), 3500)
        discussion_snapshot = _truncate(_read_text(paths.discussion_md_path), 3500)
        brief_content = _truncate(brief_content, 5000)

    hardware_profile = resolve_hardware_profile(config.hardware_profile, compute=config.compute)
    hardware_constraints = render_hardware_constraints(
        hardware_profile,
        compute=config.compute,
        time_budget_min=config.time_budget_min,
    )
    prompt = _render_template(
        template,
        {
            "slug": config.slug,
            "compute": config.compute,
            "accelerator": config.accelerator,
            "internet": config.internet,
            "hardware_profile": hardware_profile.label,
            "hardware_constraints": hardware_constraints,
            "rules_url": _read_text(paths.rules_url_path).strip() or config.competition_url or "",
            "interpretation": brief_content,
            "dataset_profile": dataset_profile,
            "submission_format": submission_format,
            "sample_submission_head": sample_submission_head,
            "code_snapshot": code_snapshot,
            "models_snapshot": models_snapshot,
            "discussion_snapshot": discussion_snapshot,
        },
    )
    if infer_deliverable_mode_from_paths(paths) == "writeup":
        requirements = summarize_writeup_requirements(paths)
        prompt += (
            "\n\n[WRITEUP_MODE]\n"
            "This competition is judged/writeup-based. Treat offline metrics and any submission.csv artifact as "
            "proxy evidence only. The strategy must end in a competition-specific writeup/report package.\n"
        )
        if requirements:
            prompt += f"Requirements extracted from local context:\n{requirements}\n"
    elif is_high_accuracy_tabular_blend_target(profile):
        prompt += (
            "\n\n[HIGH_ACCURACY_TABULAR_POLICY]\n"
            "This is a large tabular binary problem with meaningful categorical structure.\n"
            "Set `target_medal` to `winner` and `target_rank_percentile` to 0.001 "
            "unless competition rules or runtime limits force a safer goal.\n"
            "The shortlist MUST include CatBoost raw categorical, XGBoost with leak-safe "
            "target/stat encodings, and LightGBM or a second CatBoost/XGBoost variant.\n"
            "The plan MUST also include at least one OOF blend candidate "
            "(weighted/rank/logit blend) so search does not collapse into same-family tuning.\n"
            "The PLAN_JSON MUST include suite-aware ablations for competition-only training, "
            "competition-plus-original training, and an orig-signal-only lightweight suite.\n"
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


def _append_problem_type_knowledge_with_budget(prompt_text: str, knowledge_text: str, *, compact: bool) -> str:
    combined = _append_problem_type_knowledge(prompt_text, knowledge_text)
    if len(combined) <= _STRATEGY_PROMPT_MAX_CHARS or not compact:
        return combined
    clean = knowledge_text.strip()
    if not clean:
        return prompt_text
    prefix = "\n\n## Problem-Type Knowledge\n"
    remaining = _STRATEGY_PROMPT_MAX_CHARS - len(prompt_text) - len(prefix) - 1
    if remaining <= 0:
        return prompt_text
    trimmed = _truncate_exact(clean, remaining)
    if not trimmed:
        return prompt_text
    return f"{prompt_text}{prefix}{trimmed}\n"


def _append_method_registry_with_budget(prompt_text: str, *, paths: CompetitionPaths, compact: bool) -> str:
    registry = load_json_object(paths.method_registry_path)
    if registry is None:
        return prompt_text
    clean = render_method_registry_for_prompt(registry, max_methods=5 if compact else 8).strip()
    if not clean:
        return prompt_text
    section = f"\n\n## Competition-Specific Method Scout\n{clean}\n"
    if len(prompt_text) + len(section) <= _STRATEGY_PROMPT_MAX_CHARS or not compact:
        return prompt_text + section
    remaining = _STRATEGY_PROMPT_MAX_CHARS - len(prompt_text) - len("\n\n## Competition-Specific Method Scout\n") - 1
    if remaining <= 0:
        return prompt_text
    trimmed = _truncate_exact(clean, remaining)
    if not trimmed:
        return prompt_text
    return f"{prompt_text}\n\n## Competition-Specific Method Scout\n{trimmed}\n"


def _load_problem_type_knowledge_text(*, paths: CompetitionPaths, repo_root: Path, limit: int = 5) -> str:
    return load_problem_type_knowledge_text(
        dataset_profile_path=paths.dataset_profile_path,
        knowledge_paths=KnowledgePaths(workdir=repo_root),
        limit=limit,
        include_research=True,
        unavailable_message="No prior problem-type insights available.",
    )


def _should_use_fallback_strategy(stdout: str, stderr: str) -> bool:
    haystack = f"{stdout}\n{stderr}".lower()
    return any(marker in haystack for marker in _STRATEGY_RATE_LIMIT_MARKERS + _STRATEGY_TIMEOUT_MARKERS)


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
    research_sources_text: str,
    research_sources_issue: str | None,
    research_summary_text: str,
    research_summary_issue: str | None,
    *,
    profile: dict[str, object] | None,
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
    if research_sources_issue:
        issues.append(research_sources_issue)
    elif research_sources_text.strip():
        issues.extend(_validate_research_sources_jsonl(research_sources_text, min_items=_MIN_RESEARCH_ITEMS))
    else:
        issues.append("RESEARCH_SOURCES_JSONL section is required.")
    if research_summary_issue:
        issues.append(research_summary_issue)
    elif len(research_summary_text.strip()) < _MIN_RESEARCH_SUMMARY_CHARS:
        issues.append(f"RESEARCH_SUMMARY_MD is too short (<{_MIN_RESEARCH_SUMMARY_CHARS} chars).")
    if plan_payload is None:
        issues.append(plan_issue or "PLAN_JSON section missing or invalid.")
    else:
        plan_errors = validate_plan_payload(plan_payload, profile=profile)
        issues.extend(plan_errors)
    return issues


def _validate_research_sources_jsonl(raw: str, *, min_items: int) -> list[str]:
    issues: list[str] = []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < min_items:
        issues.append(f"RESEARCH_SOURCES_JSONL must contain at least {min_items} items.")
        return issues
    required_keys = {
        "url",
        "title",
        "date",
        "why_relevant",
        "extracted_technique",
        "query",
        "top_urls",
        "publish_dates",
        "takeaway",
    }
    for index, line in enumerate(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"RESEARCH_SOURCES_JSONL line {index + 1} is not valid JSON: {exc}")
            continue
        if not isinstance(item, dict):
            issues.append(f"RESEARCH_SOURCES_JSONL line {index + 1} must be a JSON object.")
            continue
        missing = sorted(required_keys - set(item.keys()))
        if missing:
            issues.append(f"RESEARCH_SOURCES_JSONL line {index + 1} missing keys: {', '.join(missing)}.")
        top_urls = item.get("top_urls")
        publish_dates = item.get("publish_dates")
        if not isinstance(top_urls, list) or not top_urls:
            issues.append(f"RESEARCH_SOURCES_JSONL line {index + 1} top_urls must be a non-empty list.")
        if not isinstance(publish_dates, list) or not publish_dates:
            issues.append(f"RESEARCH_SOURCES_JSONL line {index + 1} publish_dates must be a non-empty list.")
    return issues


def _extract_section_heading_title(line: str) -> str | None:
    stripped = line.strip()
    atx_match = re.match(r"^#{2,4}\s+(.+?)\s*$", stripped)
    if atx_match:
        title = atx_match.group(1)
    else:
        bold_match = re.match(r"^\*\*(.+?)\*\*:?\s*$", stripped)
        if not bold_match:
            return None
        title = bold_match.group(1)
    title = title.strip().rstrip(":")
    title = re.sub(r"^\d+\s*(?:[\)\.]|[-:])?\s*", "", title).strip()
    return title.lower()


def _count_source_items(text: str) -> int:
    in_sources = False
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        heading = _extract_section_heading_title(stripped)
        if heading is not None:
            if in_sources and not heading.startswith("source"):
                break
            in_sources = heading.startswith("source")
            continue
        if lowered.startswith("sources"):
            in_sources = True
            continue
        if in_sources:
            if not stripped:
                continue
            if re.match(r"^[-*]\s+", stripped):
                count += 1
                continue
            if re.match(r"^\d+[\.\)]\s+", stripped):
                count += 1
                continue
    return count


def _build_fallback_strategy(
    *,
    paths: CompetitionPaths,
    config: AgentPipelineConfig,
    brief_content: str,
    error_text: str,
) -> tuple[str, str, dict[str, object], str, str]:
    deliverable_mode = infer_deliverable_mode_from_paths(paths)
    rules_url = _read_text(paths.rules_url_path).strip() or config.competition_url or "unknown"
    dataset_profile = _summarize_dataset_profile(_read_text(paths.dataset_profile_path))
    submission_format = _summarize_submission_format(_read_text(paths.submission_format_md_path))
    sample_submission_head = _truncate(_read_sample_submission_head(paths), 400)
    hardware_profile = resolve_hardware_profile(config.hardware_profile, compute=config.compute)
    hardware_constraints = render_hardware_constraints(
        hardware_profile,
        compute=config.compute,
        time_budget_min=config.time_budget_min,
    )

    strategy_lines = [
        "# Fallback Strategy (GPT unavailable)",
        "",
        "## Problem",
        (
            "Use the local context to implement a strong, competition-appropriate model and a valid submission."
            if deliverable_mode != "writeup"
            else (
                "Use the local context to build a judged-competition solution package "
                "with proxy evidence and a final writeup."
            )
        ),
        "",
        "## Data",
        "Use dataset_profile.json, submission_format.md, and sample_submission to infer schema and target.",
        f"Brief ({BRIEF_AGENT.display_name}):",
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
        "- Candidate 1: Tree-ensemble family tuned for the dataset type and metric.",
        "- Candidate 2: Neural family suitable for the modality (tabular/text/image/sequence).",
        "- Candidate 3: Simple/high-bias baseline or linear/kernel family for calibration and fallback.",
        "",
        "## Final",
        "Pick the strongest feasible approach given file formats and time budget.",
        "",
        "## Train",
        "Use a holdout/CV split with a fixed seed and log parameters. Prefer high-capacity models when compute allows.",
        "",
        "## Hardware",
        hardware_constraints,
        "",
        "## Evaluation",
        "Train at least one real candidate and compute competition-faithful validation from train data. "
        "Do not emit placeholder, proxy, packaging-only, or unscored metrics.",
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
        f"- Code snapshot: {paths.code_md_path}",
        f"- Models snapshot: {paths.models_md_path}",
        f"- Discussion snapshot: {paths.discussion_md_path}",
        f"- Error: {error_text or 'GPT rate limit'}",
    ]
    strategy_text = "\n".join(strategy_lines).strip()
    plan_payload = _build_fallback_plan_payload(paths)
    plan_payload["hardware_profile"] = hardware_profile.key
    runtime_budget = plan_payload.setdefault("runtime_budget", {})
    if isinstance(runtime_budget, dict):
        runtime_budget.setdefault("hardware_profile", hardware_profile.key)
        runtime_budget.setdefault("gpu_vram_gb", hardware_profile.vram_gb)
        runtime_budget.setdefault("gpu_count", hardware_profile.gpu_count)
        runtime_budget.setdefault(
            "max_runtime_min",
            config.time_budget_min if config.time_budget_min is not None else hardware_profile.time_budget_min,
        )
    research_sources = [
        {
            "url": rules_url,
            "title": "Competition rules page",
            "date": "unknown",
            "why_relevant": "Primary source for metric and constraints when live web search is unavailable.",
            "extracted_technique": "Constrain modeling and submission pipeline to competition rules and metric.",
            "query": f"{config.slug} kaggle rules metric",
            "top_urls": [rules_url, str(paths.submission_format_md_path), str(paths.dataset_profile_path)],
            "publish_dates": ["unknown", "unknown", "unknown"],
            "takeaway": "Fallback mode uses local context because strategy model was rate-limited.",
        },
        {
            "url": str(paths.submission_format_md_path),
            "title": "Local submission_format.md",
            "date": "unknown",
            "why_relevant": "Defines expected output schema for submission validation.",
            "extracted_technique": "Validate columns/rows against sample submission before writing final file.",
            "query": f"{config.slug} submission format csv columns",
            "top_urls": [str(paths.sample_submission_path), str(paths.submission_format_md_path)],
            "publish_dates": ["unknown", "unknown"],
            "takeaway": "Submission schema is enforced in fallback implementation instructions.",
        },
        {
            "url": str(paths.dataset_profile_path),
            "title": "Local dataset_profile.json",
            "date": "unknown",
            "why_relevant": "Provides train/test schema hints when online sources are unavailable.",
            "extracted_technique": "Use robust, modality-appropriate defaults with leak-safe validation.",
            "query": f"{config.slug} tabular baseline cv auc",
            "top_urls": [str(paths.dataset_profile_path)],
            "publish_dates": ["unknown"],
            "takeaway": "Fallback plan defaults to practical pipelines while keeping method choice flexible.",
        },
    ]
    research_sources_text = "\n".join(json.dumps(item, ensure_ascii=True) for item in research_sources)
    research_summary_text = "\n".join(
        [
            "# Research Summary (Fallback)",
            "",
            "1. Tree-ensemble family",
            "- Pros: strong baseline for many structured datasets.",
            "- Cons: can underperform on certain modalities without specialized features.",
            "- Runtime risk: medium.",
            "- Leakage risk: low when preprocessing is fold-safe.",
            "",
            "2. Neural family",
            "- Pros: flexible representation learning for complex patterns.",
            "- Cons: can be sensitive to training setup and data size.",
            "- Runtime risk: low to medium.",
            "- Leakage risk: low with strict train-fit/apply-to-val-test flow.",
            "",
            "3. Simple/high-bias fallback family",
            "- Pros: fast, stable, and useful for sanity checks.",
            "- Cons: lower ceiling on difficult tasks.",
            "- Runtime risk: medium.",
            "- Leakage risk: low.",
        ]
    )

    instructions_lines = [
        f"# Fallback {IMPLEMENTATION_AGENT.display_name} Instructions",
        "",
        f"{STRATEGY_AGENT.display_name} was rate-limited. Follow these steps to implement a strong model in kernel.py.",
        "",
        "1) Inspect `/kaggle/input/{slug}/` to locate train/test files and the sample submission.",
        "2) If train/test CSV files exist (tabular):",
        "   - Load train/test with pandas; identify target as column present in train but not test.",
        "   - Identify ID column from sample_submission (first column) if present.",
        "   - Split train into train/valid with a fixed seed.",
        "   - Preprocess with leak-safe train-fit/apply-to-val-test transforms.",
        "   - Train at least two model families and compare offline metrics.",
        "   - Metric: use accuracy/F1/AUC for classification, RMSE/MAE for regression if unknown.",
        "3) If data is non-tabular or target is unclear:",
        "   - For RNA sequence/structure tasks, treat each molecule/target as the split unit, not each residue row.",
        "   - Use sequence-aware or structure-aware models, preserve sample_submission anchor columns exactly, "
        "and write residue-level coordinates in the provided column order.",
        "   - Use modality-appropriate models (CNN/transformer/sequence model).",
        "   - Only fall back to sample_submission if no valid training path exists.",
        "4) Always write:",
        "   - `submission.csv` and `metrics.json` under a writable output directory.",
        "   - If `/kaggle/working` is writable, mirror artifacts there; if not, skip silently.",
        "5) Keep changes confined to `kernel.py` and helper files under the kernel directory.",
    ]
    if deliverable_mode == "writeup":
        instructions_lines.extend(
            [
                "6) This is a writeup/judged competition.",
                "   - Treat `submission.csv` as optional proxy evidence only, not the primary deliverable.",
                "   - Preserve experiment outputs that can support a final report/writeup package.",
            ]
        )
    instructions_text = "\n".join(instructions_lines).replace("{slug}", config.slug).strip()
    return strategy_text, instructions_text, plan_payload, research_sources_text, research_summary_text


def _build_fallback_plan_payload(paths: CompetitionPaths) -> dict[str, object]:
    metric = _load_profile_metric(paths) or _infer_metric_from_context(paths) or "rmse"
    direction = infer_direction(metric, "auto")
    deliverable_mode = infer_deliverable_mode_from_paths(paths)
    submit_mode = infer_submit_mode_from_paths(paths)
    top1_score = _load_top1_score(paths)
    if top1_score is None:
        target_score = 0.9 if direction == "maximize" else 0.0
    else:
        target_score = float(top1_score)
    return {
        "deliverable_mode": deliverable_mode,
        "submit_mode": submit_mode,
        "target_metric": metric,
        "target_direction": direction,
        "target_score": target_score,
        "score_source": "cv",
        "holdout_frac": 0.2,
        "cv_folds": 7,
        "seed": 42,
        "max_iterations": 5,
        "patience": 6,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "name": "tree_ensemble_family",
                "features": ["basic_impute", "safe_categorical_encoding", "optional_interactions"],
                "models": ["tree_ensemble"],
                "key_hyperparameters": {"training_budget": "medium"},
                "runtime_memory": "medium runtime, medium memory",
                "failure_modes": ["overfit on noisy folds", "slow convergence"],
                "fallbacks": ["simpler regularized model", "reduced complexity"],
            },
            {
                "name": "neural_family",
                "features": ["normalized_numeric", "encoded_categorical_or_embeddings"],
                "models": ["neural_model"],
                "key_hyperparameters": {"training_budget": "medium"},
                "runtime_memory": "medium runtime, medium-high memory",
                "failure_modes": ["unstable convergence on small data"],
                "fallbacks": ["smaller architecture", "fallback to non-neural model"],
            },
            {
                "name": "simple_fallback_family",
                "features": ["basic_impute", "low-variance-safe-transforms"],
                "models": ["linear_or_kernel_baseline"],
                "key_hyperparameters": {"training_budget": "low"},
                "runtime_memory": "low runtime, low memory",
                "failure_modes": ["underfitting on complex datasets"],
                "fallbacks": ["move to stronger family", "add informative features"],
            },
        ],
        "suites": [
            {
                "name": "competition_only",
                "train_mode": "competition_only",
                "feature_recipe": "full",
                "lightweight": False,
                "promotion_stage": "full_eval",
            },
            {
                "name": "competition_plus_original",
                "train_mode": "competition_plus_original",
                "feature_recipe": "full",
                "lightweight": False,
                "promotion_stage": "ablation_fast",
            },
            {
                "name": "orig_signal_only",
                "train_mode": "competition_only",
                "feature_recipe": "orig_signal_only",
                "lightweight": True,
                "promotion_stage": "ablation_fast",
            },
        ],
        "toggles": {
            "ENABLE_PIPELINE_1": True,
            "ENABLE_PIPELINE_2": True,
            "ENABLE_PIPELINE_3": True,
            "ENABLE_ENSEMBLE": True,
            "FAST_DEV": False,
        },
        "evaluation_protocol": {
            "cv_type": "Auto",
            "n_folds": 7,
            "seeds": [42, 2024, 777],
            "primary_metric": metric,
        },
        "stop_policy": {
            "max_iterations": 5,
            "error_fingerprint_abort": True,
        },
    }


def _load_profile_metric(paths: CompetitionPaths) -> str | None:
    payload = load_json_object(paths.dataset_profile_path)
    if payload is None:
        return None
    metric = payload.get("metric")
    if isinstance(metric, str) and metric.strip():
        return metric.strip()
    return None


def _load_top1_score(paths: CompetitionPaths) -> float | None:
    payload = load_json_object(paths.top1_public_path)
    if payload is None:
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
    payload = parse_json_object_text(text)
    if payload is None:
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


def _truncate_exact(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "\n... (truncated)"
    if max_chars <= len(suffix):
        return text[:max_chars].rstrip()
    return text[: max_chars - len(suffix)].rstrip() + suffix


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
    code_headings = _extract_headings(paths.code_md_path)
    models_headings = _extract_headings(paths.models_md_path)
    discussion_headings = _extract_headings(paths.discussion_md_path)

    lines = [
        f"# Fallback Brief ({BRIEF_AGENT.display_name} output missing)",
        "",
        f"- slug: {config.slug}",
        f"- url: {config.competition_url or 'unknown'}",
        f"- compute: {config.compute} ({config.accelerator})",
        f"- rules_url: {_read_text(paths.rules_url_path).strip() or 'unknown'}",
        "",
        "## Error",
        error_text or f"{BRIEF_AGENT.display_name} brief returned empty output.",
        "",
        "## Context paths",
        f"- overview: {paths.overview_md_path}",
        f"- data: {paths.data_md_path}",
        f"- rules: {paths.rules_md_path}",
        f"- dataset_profile: {paths.dataset_profile_path}",
        f"- submission_format: {paths.submission_format_md_path}",
        f"- sample_submission_head: {paths.sample_submission_head_path}",
        f"- sample_submission: {paths.sample_submission_path}",
        f"- code: {paths.code_md_path}",
        f"- code_index: {paths.code_notebooks_index_path}",
        f"- models: {paths.models_md_path}",
        f"- discussion: {paths.discussion_md_path}",
        f"- discussion_threads_dir: {paths.discussion_threads_dir}",
        f"- discussion_index: {paths.discussion_threads_index_path}",
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
        "## Code headings",
        _format_heading_list(code_headings),
        "",
        "## Models headings",
        _format_heading_list(models_headings),
        "",
        "## Discussion headings",
        _format_heading_list(discussion_headings),
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
