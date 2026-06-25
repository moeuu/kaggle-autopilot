from __future__ import annotations

from pathlib import Path

from kagglebot.json_utils import load_json_object
from kagglebot.knowledge import (
    derive_problem_types,
    ensure_taxonomy,
    format_error_fix_insights,
    format_problem_type_insights,
    format_research_artifacts,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
    resolve_research_artifacts,
    resolve_similar_improvements,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths


def resolve_problem_types_from_profile(*, dataset_profile_path: Path) -> list[str]:
    profile = load_json_object(dataset_profile_path) or {}
    if not isinstance(profile, dict):
        profile = {}
    return derive_problem_types(profile)


def refresh_knowledge_hints(*, paths: CompetitionPaths, knowledge_paths: KnowledgePaths) -> None:
    """Write competition knowledge hints from dataset tags and self-improvement context."""
    from kagglebot.self_improvement import load_self_improvement_context

    profile = load_json_object(paths.dataset_profile_path) or {}
    if not isinstance(profile, dict):
        profile = {}
    raw_tags = profile.get("tags", [])
    tags = [str(tag).strip() for tag in raw_tags if isinstance(tag, str) and str(tag).strip()]

    lines = ["# Knowledge Hints", ""]
    try:
        if not tags:
            lines.append("No dataset tags available yet; knowledge suggestions pending dataset profiling.")
        else:
            taxonomy = ensure_taxonomy(knowledge_paths)
            similar = resolve_similar_improvements(
                knowledge_paths=knowledge_paths,
                taxonomy=taxonomy,
                tags=tags,
            )
            if not similar:
                lines.append("No similar competitions found in knowledge base.")
            else:
                lines.append("Similar competitions and what improved score:")
                lines.append("")
                for item in similar:
                    slug = item.get("slug", "unknown")
                    overlap = item.get("overlap", 0)
                    summary = item.get("summary", "No summary recorded.")
                    lines.append(f"- {slug} ({overlap} tag overlap): {summary}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Knowledge lookup failed: {exc}")

    lines.extend(["", "## System Self-Improvement Context"])
    context = load_self_improvement_context(paths.artifacts_dir)
    if context:
        lines.append(context)
    else:
        lines.append("No self-improvement context available yet.")

    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.knowledge_hints_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_problem_type_knowledge_text(
    *,
    dataset_profile_path: Path,
    knowledge_paths: KnowledgePaths,
    limit: int = 5,
    include_research: bool = True,
    unavailable_message: str = "No prior problem-type insights available.",
) -> str:
    """Render reusable problem-type knowledge context for planning and improvement prompts."""
    try:
        problem_types = resolve_problem_types_from_profile(dataset_profile_path=dataset_profile_path)
        sections = [
            format_problem_type_insights(
                resolve_problem_type_insights(knowledge_paths, problem_types, limit=limit),
                limit=limit,
            ),
            "",
            format_error_fix_insights(
                resolve_error_fix_insights(knowledge_paths, problem_types, limit=limit),
                limit=limit,
            ),
        ]
        if include_research:
            sections.extend(
                [
                    "",
                    format_research_artifacts(
                        resolve_research_artifacts(
                            knowledge_paths=knowledge_paths,
                            problem_types=problem_types,
                            limit=limit,
                        ),
                        limit=limit,
                    ),
                ]
            )
        return "\n".join(section for section in sections if section is not None)
    except Exception as exc:  # noqa: BLE001
        return unavailable_message.format(error=exc)
