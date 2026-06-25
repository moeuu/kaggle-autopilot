from __future__ import annotations

from pathlib import Path

from kagglebot.json_utils import load_json_object
from kagglebot.knowledge import (
    derive_problem_types,
    format_error_fix_insights,
    format_problem_type_insights,
    format_research_artifacts,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
    resolve_research_artifacts,
)
from kagglebot.paths import KnowledgePaths


def resolve_problem_types_from_profile(*, dataset_profile_path: Path) -> list[str]:
    profile = load_json_object(dataset_profile_path) or {}
    if not isinstance(profile, dict):
        profile = {}
    return derive_problem_types(profile)


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
