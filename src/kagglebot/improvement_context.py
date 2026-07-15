from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot import agent_prompts as _agent_prompts
from kagglebot import code_reference as _code_reference
from kagglebot import context_artifacts as _context_artifacts
from kagglebot import iteration_signals as _iteration_signals
from kagglebot import knowledge_context as _knowledge_context
from kagglebot import method_scout as _method_scout
from kagglebot import plan_policy as _plan_policy
from kagglebot import score_progress as _score_progress
from kagglebot import submission_history as _submission_history
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot.agents.identity import IMPLEMENTATION_AGENT, STRATEGY_AGENT, render_prompt_identity
from kagglebot.code_reference import CodeReferenceNotebook
from kagglebot.competition_policy import load_competition_policy
from kagglebot.hardware import render_hardware_constraints, resolve_hardware_profile
from kagglebot.writeup import infer_deliverable_mode_from_paths


class ImprovementContextPaths(Protocol):
    slug: str
    codex_improve_template: Path
    plan_path: Path
    knowledge_hints_path: Path
    rules_url_path: Path
    rules_md_path: Path
    rules_html_path: Path
    overview_md_path: Path
    data_md_path: Path
    submission_format_md_path: Path
    dataset_profile_path: Path
    sample_submission_path: Path
    code_md_path: Path
    code_notebooks_index_path: Path
    kaggle_discovery_md_path: Path
    kernel_source_dir: Path
    method_registry_path: Path

    def run_dir(self, run_id: str) -> Path: ...


class ImprovementContextConfig(Protocol):
    slug: str
    compute: str
    accelerator: str
    hardware_profile: str | None
    time_budget_min: int | None
    paths: ImprovementContextPaths
    knowledge_paths: object


class ImprovementEvaluation(Protocol):
    metric: str
    direction: str
    value: float


@dataclass(frozen=True)
class ImprovementModeNotice:
    kind: str
    previous_mode: str
    new_mode: str
    reason: str


@dataclass(frozen=True)
class ImprovementPromptPlan:
    prompt_path: Path
    strategy_dir: Path
    base_prompt_text: str
    strategy_prompt: str
    improvement_mode: str
    effective_current_score: float
    top1_score: object
    top1_gap: float | None
    code_reference_mandatory: bool
    required_reference_notebook: CodeReferenceNotebook | None
    mode_notices: tuple[ImprovementModeNotice, ...]


def build_improvement_prompt_plan(
    *,
    config: ImprovementContextConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    agent_dir: Path,
    evaluation: ImprovementEvaluation,
    top1_info: dict[str, object],
    target_score: float,
    delta_offline: float | None,
    current_score: float | None,
    current_score_source: str,
    minimum_improvement_mode: str | None,
    minimum_improvement_reason: str | None,
    target_medal: str | None,
    target_rank_percentile: float | None,
    forced_improvement_mode: str | None,
    forced_improvement_reason: str | None,
    extra_policy_notes: list[str] | None,
    enforce_code_reference_implementation: bool,
    code_reference_enforcement_reason: str | None,
    best_score_so_far: float | None,
    previous_submission_history: dict[str, object] | None,
    prompt_identity_args: dict[str, object],
    iteration_evidence_path: Path | None = None,
    iteration_evidence_sha256: str | None = None,
    iteration_evidence_summary: str | None = None,
) -> ImprovementPromptPlan:
    prompt_template = render_prompt_identity(config.paths.codex_improve_template.read_text(encoding="utf-8"))
    prompt_path = agent_dir / "prompt.md"
    run_dir = config.paths.run_dir(run_id)
    submit_failure_notes, submit_failure_force_reason = (
        _submit_failure_context.build_submit_failure_improvement_context_for_run(run_dir=run_dir)
    )
    top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
    effective_current_score = evaluation.value if current_score is None else current_score
    improvement_mode, top1_gap = _score_progress.classify_improvement_mode(
        effective_current_score,
        top1_score,
        evaluation.direction,
    )
    mode_notices: list[ImprovementModeNotice] = []
    upgraded_mode = _plan_policy.upgrade_improvement_mode(improvement_mode, minimum_improvement_mode)
    if upgraded_mode != improvement_mode:
        mode_notices.append(
            ImprovementModeNotice(
                kind="floor",
                previous_mode=improvement_mode,
                new_mode=upgraded_mode,
                reason=minimum_improvement_reason or "policy",
            )
        )
        improvement_mode = upgraded_mode
    if forced_improvement_mode:
        mode_notices.append(
            ImprovementModeNotice(
                kind="override",
                previous_mode=improvement_mode,
                new_mode=forced_improvement_mode,
                reason=forced_improvement_reason or "policy",
            )
        )
        improvement_mode = forced_improvement_mode

    kernel_main_path = config.paths.kernel_source_dir / "kernel.py"
    code_reference_score, code_reference_source = _code_reference.extract_code_reference_score(config.paths)
    code_reference_comparison_score = _score_progress.normalize_code_reference_score_for_comparison(
        current=effective_current_score,
        reference=code_reference_score,
        metric=evaluation.metric,
    )
    code_reference_delta = (
        _score_progress.score_delta_vs_reference(
            effective_current_score,
            code_reference_comparison_score,
            evaluation.direction,
        )
        if code_reference_comparison_score is not None
        else None
    )
    code_reference_underperforming = bool(
        code_reference_score is not None and code_reference_delta is not None and code_reference_delta < 0
    )
    if code_reference_score is None:
        code_reference_status = "code_reference_unavailable"
    elif code_reference_underperforming:
        code_reference_status = "underperforming_code_reference"
    else:
        code_reference_status = "at_or_above_code_reference"
    required_reference_notebook = _code_reference.load_required_reference_notebook(config.paths)
    ensemble_reference_notebook = _code_reference.load_ensemble_reference_notebook(config.paths)
    competition_policy = load_competition_policy(config.paths)
    base_prompt_text = prompt_template.format(
        **prompt_identity_args,
        slug=config.slug,
        iteration=iteration,
        plan_path=str(config.paths.plan_path),
        run_path=str(config.paths.run_dir(run_id) / "run.json"),
        metrics_path=str(iter_dir / "metrics.json"),
        diagnostics_path=str(iter_dir / "diagnostics.md"),
        logs_dir=str(iter_dir / "logs"),
        compute=config.compute,
        accelerator=config.accelerator,
        knowledge_hints=str(config.paths.knowledge_hints_path),
        metric=evaluation.metric,
        direction=evaluation.direction,
        current_score=f"{effective_current_score:.6f}",
        current_score_source=current_score_source,
        target_score=f"{target_score:.6f}",
        top1_score=str(top1_score or "unavailable"),
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap="unavailable" if top1_gap is None else f"{top1_gap:.6f}",
        delta_offline="unavailable" if delta_offline is None else f"{delta_offline:.6f}",
        improvement_mode=improvement_mode,
        next_iteration=str(iteration + 1),
        rules_url=str(config.paths.rules_url_path),
        rules_md=str(config.paths.rules_md_path),
        rules_html=str(config.paths.rules_html_path),
        overview_md=str(config.paths.overview_md_path),
        data_md=str(config.paths.data_md_path),
        submission_format=str(config.paths.submission_format_md_path),
        dataset_profile=str(config.paths.dataset_profile_path),
        sample_submission=str(config.paths.sample_submission_path),
        code_md=str(config.paths.code_md_path),
        code_index=str(config.paths.code_notebooks_index_path),
        code_reference_score=("unavailable" if code_reference_score is None else f"{code_reference_score:.6f}"),
        code_reference_source=code_reference_source,
        code_reference_delta=("unavailable" if code_reference_delta is None else f"{code_reference_delta:+.6f}"),
        code_reference_status=code_reference_status,
        kernel_main=str(kernel_main_path),
    )
    base_prompt_text = append_improvement_policy_context(
        base_prompt_text=base_prompt_text,
        config=config,
        run_id=run_id,
        evaluation=evaluation,
        improvement_mode=improvement_mode,
        forced_improvement_mode=forced_improvement_mode,
        forced_improvement_reason=forced_improvement_reason,
        minimum_improvement_reason=minimum_improvement_reason,
        target_medal=target_medal,
        target_rank_percentile=target_rank_percentile,
        competition_policy=competition_policy,
        ensemble_reference_notebook=ensemble_reference_notebook,
        best_score_so_far=best_score_so_far,
        previous_submission_history=previous_submission_history,
        extra_policy_notes=extra_policy_notes,
        submit_failure_notes=submit_failure_notes,
        submit_failure_force_reason=submit_failure_force_reason,
    )
    if iteration_evidence_path is not None:
        base_prompt_text += (
            "\n\n"
            + (iteration_evidence_summary or "## Iteration Evidence Contract")
            + "\n"
            + f"- Evidence path: {iteration_evidence_path}\n"
            + f"- Evidence SHA-256: {iteration_evidence_sha256 or 'unavailable'}\n"
            + "Treat this frozen bundle as the authoritative attribution record for the next-iteration decision.\n"
        )
    base_prompt_text += (
        "\n\n## Ranked Kaggle Ecosystem Discovery\n"
        f"- Snapshot: {config.paths.kaggle_discovery_md_path}\n"
        "Inspect high-relevance Datasets, Models, Code, Discussions, Game Arena, and Benchmarks records before "
        "choosing the next experiment. Ignore low-relevance trends and verify competition rules/licenses before use.\n"
    )

    code_reference_mandatory = bool(code_reference_underperforming or enforce_code_reference_implementation)
    base_prompt_text += (
        "\n\n"
        + "\n".join(
            build_code_reference_gate_lines(
                config=config,
                code_reference_score=code_reference_score,
                code_reference_comparison_score=code_reference_comparison_score,
                code_reference_delta=code_reference_delta,
                code_reference_source=code_reference_source,
                code_reference_status=code_reference_status,
                code_reference_mandatory=code_reference_mandatory,
                code_reference_underperforming=code_reference_underperforming,
                code_reference_enforcement_reason=code_reference_enforcement_reason,
                required_reference_notebook=required_reference_notebook,
                ensemble_reference_notebook=ensemble_reference_notebook,
                prefer_ensemble_reference=competition_policy.prompt.prefer_ensemble_reference,
            )
        )
        + "\n"
    )

    problem_type_knowledge = _knowledge_context.load_problem_type_knowledge_text(
        dataset_profile_path=config.paths.dataset_profile_path,
        knowledge_paths=config.knowledge_paths,
        include_research=False,
        unavailable_message="Problem-type knowledge unavailable: {error}",
    )
    hardware_profile = resolve_hardware_profile(config.hardware_profile, compute=config.compute)
    strategy_prompt = _agent_prompts.build_improvement_strategy_prompt(
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        metric=evaluation.metric,
        direction=evaluation.direction,
        current_score=effective_current_score,
        current_score_source=current_score_source,
        target_score=target_score,
        top1_score=top1_score,
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap=top1_gap,
        delta_offline=delta_offline,
        improvement_mode=improvement_mode,
        hardware_constraints=render_hardware_constraints(
            hardware_profile,
            compute=config.compute,
            time_budget_min=config.time_budget_min,
        ),
        codex_prompt=base_prompt_text,
        problem_type_knowledge=problem_type_knowledge,
    )
    return ImprovementPromptPlan(
        prompt_path=prompt_path,
        strategy_dir=agent_dir / f"improve_strategy-{iteration:02d}",
        base_prompt_text=base_prompt_text,
        strategy_prompt=strategy_prompt,
        improvement_mode=improvement_mode,
        effective_current_score=effective_current_score,
        top1_score=top1_score,
        top1_gap=top1_gap,
        code_reference_mandatory=code_reference_mandatory,
        required_reference_notebook=required_reference_notebook,
        mode_notices=tuple(mode_notices),
    )


def append_improvement_policy_context(
    *,
    base_prompt_text: str,
    config: ImprovementContextConfig,
    run_id: str,
    evaluation: ImprovementEvaluation,
    improvement_mode: str,
    forced_improvement_mode: str | None,
    forced_improvement_reason: str | None,
    minimum_improvement_reason: str | None,
    target_medal: str | None,
    target_rank_percentile: float | None,
    competition_policy: object,
    ensemble_reference_notebook: CodeReferenceNotebook | None,
    best_score_so_far: float | None,
    previous_submission_history: dict[str, object] | None,
    extra_policy_notes: list[str] | None,
    submit_failure_notes: list[str],
    submit_failure_force_reason: str | None,
) -> str:
    if infer_deliverable_mode_from_paths(config.paths) == "writeup":
        base_prompt_text += (
            "\n\nWriteup mode is active for this competition.\n"
            "Do not optimize only for submission artifact production. Treat offline metrics and any "
            "submission artifacts as "
            "proxy evidence supporting the final judged writeup package.\n"
        )
    if forced_improvement_reason:
        base_prompt_text += (
            "\n\nForced improvement mode policy is active.\n"
            f"Reason: {forced_improvement_reason}\n"
            "Do not propose minor_tuning; follow the forced improvement mode.\n"
        )
        if forced_improvement_mode == "validation_redesign":
            base_prompt_text += (
                "Mode is validation_redesign: first build and compare group/time/leak/proxy split candidates, "
                "calibrate against previous public outcomes, and only then rank new model-family changes.\n"
            )
        elif forced_improvement_mode == "implementation_audit":
            base_prompt_text += (
                "Mode is implementation_audit: assume the last-place-like leaderboard result is an execution or "
                "submission defect until disproved. Trace the exact hidden-test input, loaded model/assets, runtime "
                "fallbacks, prediction distribution, ID/order alignment, filename/schema, metric scale/direction, "
                "and selected notebook output. Reproduce the remote path with fidelity checks before changing model "
                "families, and do not resubmit an unchanged artifact.\n"
            )
    elif minimum_improvement_reason:
        base_prompt_text += (
            "\n\nMinimum improvement mode policy is active.\n"
            f"Reason: {minimum_improvement_reason}\n"
            "Do not propose minor_tuning while this policy remains active.\n"
        )
    if improvement_mode == "validation_redesign":
        base_prompt_text += (
            "\n\nValidation redesign campaign policy:\n"
            "- Treat online regression or low offline-online correlation as a split problem first.\n"
            "- Create validation_variant candidates for group, time, leak-safe, and proxy/adversarial splits.\n"
            "- Do not submit another model-only candidate until the active validation profile is justified.\n"
        )
    if improvement_mode == "implementation_audit":
        base_prompt_text += (
            "\n\nLeaderboard implementation-audit campaign policy:\n"
            "- Treat near-last rank, zero-score collapse, and extreme offline/online disagreement as implementation "
            "evidence before treating them as model-quality evidence.\n"
            "- Compare the packaged Notebook and executed logs against the locally evaluated candidate.\n"
            "- Verify non-constant predictions, hidden-test coverage, row/ID order, output filename and schema, and "
            "that diagnostic .npy/.json files cannot be selected as submissions.\n"
            "- Require a changed artifact hash plus runtime-fidelity evidence before another submission.\n"
        )
    if target_rank_percentile is not None:
        medal_label = target_medal or "rank"
        base_prompt_text += (
            "\n\nMedal-aware search policy:\n"
            f"- target_medal: {medal_label}\n"
            f"- target_rank_percentile: {target_rank_percentile * 100:.2f}%\n"
            "- Until this leaderboard percentile is reached, keep search breadth high and "
            "avoid same-family-only tweaks.\n"
        )
    if _iteration_signals.requires_tabular_multi_family_policy(
        _context_artifacts.load_dataset_profile(
            slug=config.paths.slug,
            dataset_profile_path=config.paths.dataset_profile_path,
        )
    ):
        base_prompt_text += (
            "\n\nHigh-accuracy tabular policy is active.\n"
            "- This dataset is tabular binary with meaningful categorical structure.\n"
            "- The next iteration must keep multi-family exploration active.\n"
            "- Require CatBoost raw categorical, XGBoost with leak-safe target/stat encodings, "
            "and LightGBM or a second CatBoost/XGBoost variant.\n"
            "- If two or more model pipelines exist, require at least one OOF-based blend "
            "candidate (weighted/rank/logit blend).\n"
        )
    if competition_policy.active:
        policy_lines = ["\n\nCompetition policy override is active."]
        if competition_policy.required_capabilities:
            policy_lines.append(
                "- Required capabilities: "
                + ", ".join(capability for capability in competition_policy.required_capabilities if capability)
            )
        if competition_policy.has_capability("recoverable_original_dataset"):
            policy_lines.append(
                "- If staged reference/original datasets are available, wire them into training or feature "
                "generation instead of leaving them unused."
            )
        if competition_policy.has_capability("heterogeneous_tabular_ensemble"):
            policy_lines.append(
                "- Keep orthogonal model families active; do not spend the next iteration on same-family-only tuning."
            )
        if competition_policy.has_capability("requires_oof_blend"):
            policy_lines.append(
                "- Persist OOF predictions for each candidate and emit at least one weighted or rank blend artifact."
            )
        if competition_policy.has_capability("text_translation_seq2seq"):
            policy_lines.append(
                "- For translation/text seq2seq tasks, prefer reusable helpers from "
                "`src/kagglebot/kernel_runtime/text_translation.py` for normalization, metrics, MBR, retrieval, "
                "and consistency logic; keep competition-specific joins and dictionaries in `kernel.py`."
            )
        if competition_policy.has_capability("requires_grouped_text_cv"):
            policy_lines.append(
                "- Use grouped text CV keyed by the plan/runtime group columns; "
                "do not rank candidates with plain row-level splits."
            )
        if competition_policy.has_capability("requires_candidate_rerank"):
            policy_lines.append(
                "- Treat retrieval as a candidate source or fallback only; "
                "keep seq2seq + candidate rerank/MBR as the primary path."
            )
        if competition_policy.has_capability("supports_metadata_supervision"):
            policy_lines.append(
                "- If metadata supervision is useful, declare required aux inputs in "
                "plan.json `text_runtime.required_aux_inputs` "
                "and keep the matching/join heuristics inside `kernel.py`."
            )
        if competition_policy.has_capability("supports_soft_constraint_rewrite"):
            policy_lines.append(
                "- Prefer soft constraint rewrites and rerank bonuses for "
                "entity/quantity/unit handling instead of hard-coded decode constraints."
            )
        if competition_policy.prompt.ablation_groups:
            policy_lines.append(
                "- Required ablations: "
                + ", ".join(group for group in competition_policy.prompt.ablation_groups if group)
            )
        if competition_policy.prompt.min_model_families_before_stop is not None:
            policy_lines.append(
                f"- Minimum model families before stop: {competition_policy.prompt.min_model_families_before_stop}"
            )
        if competition_policy.prompt.require_oof_blend_before_stop:
            policy_lines.append("- Do not stop until at least one OOF blend candidate is implemented.")
        if competition_policy.evaluation.search_stop_rank_percentile is not None:
            policy_lines.append(
                "- Internal search target rank percentile: "
                f"{competition_policy.evaluation.search_stop_rank_percentile * 100:.2f}%"
            )
        if competition_policy.prompt.prefer_ensemble_reference and ensemble_reference_notebook is not None:
            policy_lines.append(f"- ensemble_kernel_id: {ensemble_reference_notebook.kernel_id}")
        if competition_policy.execution_hints:
            policy_lines.append(
                "- execution_hints: "
                + json.dumps(competition_policy.execution_hints, sort_keys=True, ensure_ascii=True)
            )
        for note in competition_policy.prompt.extra_notes:
            policy_lines.append(f"- {note}")
        base_prompt_text += "\n".join(policy_lines) + "\n"
    if best_score_so_far is not None:
        base_prompt_text += (
            "\n\nRegression Guard Policy:\n"
            f"- Best known offline score so far: {float(best_score_so_far):.6f}\n"
            "- Do NOT introduce conservative fallback paths that intentionally reduce model capacity "
            "or collapse features (e.g., tiny robust subsets) when they materially degrade offline quality.\n"
            "- If suspiciously high CV is detected, keep leak fixes but preserve competitive model strength "
            "instead of defaulting to a weak baseline.\n"
        )
    history_prompt = _submission_history.format_previous_submission_history_for_prompt(previous_submission_history)
    if history_prompt:
        base_prompt_text += "\n\nPrevious Kaggle Submission Results:\n" + history_prompt + "\n"
    method_registry_payload = _method_scout.load_method_registry(config.paths.method_registry_path)
    method_prompt = _method_scout.render_method_registry_for_prompt(method_registry_payload, max_methods=8)
    if method_prompt:
        base_prompt_text += "\n\nCompetition-Specific Method Scout:\n" + method_prompt + "\n"
    if extra_policy_notes:
        note_lines = []
        for note in extra_policy_notes:
            clean = str(note).strip()
            if clean:
                note_lines.append(f"- {clean}")
        if note_lines:
            base_prompt_text += "\n\nAdditional repair targets:\n" + "\n".join(note_lines) + "\n"
    if submit_failure_notes:
        base_prompt_text += (
            "\n\nSubmit Contract Repair:\n" + "\n".join(f"- {note}" for note in submit_failure_notes) + "\n"
        )
        if submit_failure_force_reason:
            base_prompt_text += (
                "\nSubmit contract repair policy is active.\n"
                f"Reason: {submit_failure_force_reason}\n"
                "Repair the submission contract before spending iteration budget on further model tuning.\n"
            )
    return base_prompt_text


def build_code_reference_gate_lines(
    *,
    config: ImprovementContextConfig,
    code_reference_score: float | None,
    code_reference_comparison_score: float | None,
    code_reference_delta: float | None,
    code_reference_source: str,
    code_reference_status: str,
    code_reference_mandatory: bool,
    code_reference_underperforming: bool,
    code_reference_enforcement_reason: str | None,
    required_reference_notebook: CodeReferenceNotebook | None,
    ensemble_reference_notebook: CodeReferenceNotebook | None,
    prefer_ensemble_reference: bool,
) -> list[str]:
    gate_lines = [
        "## Code Reference Gate",
        f"- Code snapshot: {config.paths.code_md_path}",
        f"- Code notebook index: {config.paths.code_notebooks_index_path}",
        (
            "- Code reference score: unavailable"
            if code_reference_score is None
            else (
                f"- Code reference score: {code_reference_score:.6f} "
                f"(comparison_score={code_reference_comparison_score:.6f}, "
                f"source: {code_reference_source}, delta_vs_current={code_reference_delta:+.6f})"
            )
        ),
        f"- Code reference status: {code_reference_status}",
    ]
    if code_reference_mandatory:
        gate_lines.extend(
            [
                "",
                (
                    "Current score is below the code reference baseline."
                    if code_reference_underperforming
                    else "Code reference implementation is policy-mandatory for the next iteration."
                ),
                (
                    f"Enforcement reason: {code_reference_enforcement_reason}"
                    if code_reference_enforcement_reason
                    else "Enforcement reason: code-reference policy"
                ),
                "You MUST inspect code.md and code_notebooks_index.json and treat",
                "`Required Reference Notebook (Execution baseline)` as mandatory baseline context.",
            ]
        )
        if required_reference_notebook is not None:
            gate_lines.extend(
                [
                    f"- required_kernel_id: {required_reference_notebook.kernel_id}",
                    f"- required_title: {required_reference_notebook.title}",
                    (
                        f"- required_source_file: {required_reference_notebook.source_file}"
                        if required_reference_notebook.source_file
                        else "- required_source_file: unavailable"
                    ),
                    (
                        f"- required_local_dir: {required_reference_notebook.local_dir}"
                        if required_reference_notebook.local_dir
                        else "- required_local_dir: unavailable"
                    ),
                    f"- required_marker: {_code_reference.code_reference_marker(required_reference_notebook)}",
                    (
                        "- required_model_family: tabicl"
                        if _code_reference.reference_requires_tabicl(required_reference_notebook)
                        else "- required_model_family: follow required notebook strategy"
                    ),
                ]
            )
        if ensemble_reference_notebook is not None and prefer_ensemble_reference:
            gate_lines.extend(
                [
                    f"- ensemble_kernel_id: {ensemble_reference_notebook.kernel_id}",
                    f"- ensemble_title: {ensemble_reference_notebook.title}",
                    (
                        f"- ensemble_source_file: {ensemble_reference_notebook.source_file}"
                        if ensemble_reference_notebook.source_file
                        else "- ensemble_source_file: unavailable"
                    ),
                    "After reproducing the execution baseline, inspect the ensemble reference notebook "
                    "as the blend blueprint.",
                ]
            )
        gate_lines.extend(
            [
                "Either reproduce that baseline path first or justify concrete blockers and implement",
                "the closest leak-free fallback in kernel.py.",
                "When implementing the required notebook path, add the exact marker comment shown above.",
            ]
        )
    return gate_lines


def build_improvement_implementation_prompt(
    *,
    base_prompt_text: str,
    strategy_text: str,
) -> str:
    if not strategy_text:
        return base_prompt_text
    return (
        f"# {IMPLEMENTATION_AGENT.display_name} Improvement Implementation\n\n"
        f"Implement the {STRATEGY_AGENT.display_name}-authored improvement prompt below as the primary plan.\n\n"
        f"## {STRATEGY_AGENT.display_name} Extra-High Improvement Prompt\n"
        f"{strategy_text}\n\n"
        "## Local Context (for file paths and constraints)\n"
        f"{base_prompt_text}\n"
    )
