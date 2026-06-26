from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kagglebot import context_artifacts as _context_artifacts
from kagglebot import knowledge_context as _knowledge_context


class KnowledgePhaseConfig(Protocol):
    paths: object
    knowledge_paths: object


@dataclass(frozen=True)
class KnowledgePhase:
    config: KnowledgePhaseConfig

    def refresh(self) -> None:
        _knowledge_context.refresh_knowledge_hints(paths=self.config.paths, knowledge_paths=self.config.knowledge_paths)

    def load_dataset_profile(self) -> dict[str, object]:
        return _context_artifacts.load_dataset_profile(
            slug=self.config.paths.slug,
            dataset_profile_path=self.config.paths.dataset_profile_path,
        )

    def derive_problem_types(self) -> list[str]:
        return _knowledge_context.resolve_problem_types_from_profile(
            dataset_profile_path=self.config.paths.dataset_profile_path
        )
