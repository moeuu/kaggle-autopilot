from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kagglebot.env_utils import parse_bool_value
from kagglebot.json_utils import load_json_object
from kagglebot.paths import CompetitionPaths
from kagglebot.scalar_utils import parse_finite_float, parse_int


@dataclass(frozen=True)
class NotebookSelectionPolicy:
    keyword_boosts: dict[str, float] = field(default_factory=dict)
    required_reference_keywords: tuple[str, ...] = ()
    ensemble_reference_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceInputPolicy:
    proactive: bool = False
    required_datasets: tuple[str, ...] = ()
    extra_dataset_refs: tuple[str, ...] = ()
    extra_kernel_refs: tuple[str, ...] = ()
    extra_competition_refs: tuple[str, ...] = ()
    block_on_missing_required: bool = False


@dataclass(frozen=True)
class PromptPolicy:
    ablation_groups: tuple[str, ...] = ()
    min_model_families_before_stop: int | None = None
    require_oof_blend_before_stop: bool = False
    prefer_ensemble_reference: bool = False
    extra_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairPolicy:
    missing_ensemble_signal: bool = False
    original_data_unused_signal: bool = False
    same_family_plateau_signal: bool = False


@dataclass(frozen=True)
class EvaluationPolicy:
    fallback_overrides: dict[str, Any] = field(default_factory=dict)
    search_stop_rank_percentile: float | None = None


@dataclass(frozen=True)
class CompetitionPolicy:
    slug: str = ""
    archetype_tags: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    execution_hints: dict[str, Any] = field(default_factory=dict)
    notebook_selection: NotebookSelectionPolicy = field(default_factory=NotebookSelectionPolicy)
    reference_inputs: ReferenceInputPolicy = field(default_factory=ReferenceInputPolicy)
    prompt: PromptPolicy = field(default_factory=PromptPolicy)
    repair: RepairPolicy = field(default_factory=RepairPolicy)
    evaluation: EvaluationPolicy = field(default_factory=EvaluationPolicy)

    @property
    def active(self) -> bool:
        return any(
            [
                bool(self.archetype_tags),
                bool(self.required_capabilities),
                bool(self.execution_hints),
                bool(self.notebook_selection.keyword_boosts),
                bool(self.notebook_selection.required_reference_keywords),
                bool(self.notebook_selection.ensemble_reference_keywords),
                self.reference_inputs.proactive,
                bool(self.reference_inputs.required_datasets),
                bool(self.reference_inputs.extra_dataset_refs),
                bool(self.reference_inputs.extra_kernel_refs),
                bool(self.reference_inputs.extra_competition_refs),
                bool(self.prompt.ablation_groups),
                self.prompt.min_model_families_before_stop is not None,
                self.prompt.require_oof_blend_before_stop,
                self.prompt.prefer_ensemble_reference,
                bool(self.prompt.extra_notes),
                self.repair.missing_ensemble_signal,
                self.repair.original_data_unused_signal,
                self.repair.same_family_plateau_signal,
                bool(self.evaluation.fallback_overrides),
                self.evaluation.search_stop_rank_percentile is not None,
            ]
        )

    def has_capability(self, capability: str) -> bool:
        target = str(capability).strip().lower()
        if not target:
            return False
        return any(item.strip().lower() == target for item in self.required_capabilities)

    def execution_hint(self, key: str, default: Any = None) -> Any:
        return self.execution_hints.get(key, default)


def load_competition_policy(paths: CompetitionPaths) -> CompetitionPolicy:
    path = paths.competition_policy_path
    payload = load_json_object(path)
    if payload is None:
        return CompetitionPolicy(slug=paths.slug)

    notebook_selection = payload.get("notebook_selection")
    reference_inputs = payload.get("reference_inputs")
    prompt = payload.get("prompt")
    repair = payload.get("repair")
    evaluation = payload.get("evaluation")

    return CompetitionPolicy(
        slug=str(payload.get("slug") or paths.slug),
        archetype_tags=_to_str_tuple(payload.get("archetype_tags")),
        required_capabilities=_to_str_tuple(payload.get("required_capabilities")),
        execution_hints=_to_mapping_dict(payload, key="execution_hints"),
        notebook_selection=NotebookSelectionPolicy(
            keyword_boosts=_to_float_dict(notebook_selection, key="keyword_boosts"),
            required_reference_keywords=_to_str_tuple_from_mapping(
                notebook_selection,
                key="required_reference_keywords",
            ),
            ensemble_reference_keywords=_to_str_tuple_from_mapping(
                notebook_selection,
                key="ensemble_reference_keywords",
            ),
        ),
        reference_inputs=ReferenceInputPolicy(
            proactive=_to_bool(reference_inputs, key="proactive"),
            required_datasets=_to_str_tuple_from_mapping(reference_inputs, key="required_datasets"),
            extra_dataset_refs=_to_str_tuple_from_mapping(reference_inputs, key="extra_dataset_refs"),
            extra_kernel_refs=_to_str_tuple_from_mapping(reference_inputs, key="extra_kernel_refs"),
            extra_competition_refs=_to_str_tuple_from_mapping(reference_inputs, key="extra_competition_refs"),
            block_on_missing_required=_to_bool(reference_inputs, key="block_on_missing_required"),
        ),
        prompt=PromptPolicy(
            ablation_groups=_to_str_tuple_from_mapping(prompt, key="ablation_groups"),
            min_model_families_before_stop=_to_int(prompt, key="min_model_families_before_stop"),
            require_oof_blend_before_stop=_to_bool(prompt, key="require_oof_blend_before_stop"),
            prefer_ensemble_reference=_to_bool(prompt, key="prefer_ensemble_reference"),
            extra_notes=_to_str_tuple_from_mapping(prompt, key="extra_notes"),
        ),
        repair=RepairPolicy(
            missing_ensemble_signal=_to_bool(repair, key="missing_ensemble_signal"),
            original_data_unused_signal=_to_bool(repair, key="original_data_unused_signal"),
            same_family_plateau_signal=_to_bool(repair, key="same_family_plateau_signal"),
        ),
        evaluation=EvaluationPolicy(
            fallback_overrides=_to_mapping_dict(evaluation, key="fallback_overrides"),
            search_stop_rank_percentile=_to_float(evaluation, key="search_stop_rank_percentile"),
        ),
    )


def _to_str_tuple(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    values = [str(item).strip() for item in raw if str(item).strip()]
    return tuple(values)


def _to_str_tuple_from_mapping(raw: object, *, key: str) -> tuple[str, ...]:
    if not isinstance(raw, dict):
        return ()
    return _to_str_tuple(raw.get(key))


def _to_float_dict(raw: object, *, key: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    nested = raw.get(key)
    if not isinstance(nested, dict):
        return {}
    converted: dict[str, float] = {}
    for item_key, value in nested.items():
        keyword = str(item_key).strip()
        if not keyword:
            continue
        parsed = parse_finite_float(value)
        if parsed is None:
            continue
        converted[keyword] = parsed
    return converted


def _to_mapping_dict(raw: object, *, key: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    nested = raw.get(key)
    return dict(nested) if isinstance(nested, dict) else {}


def _to_bool(raw: object, *, key: str) -> bool:
    if not isinstance(raw, dict):
        return False
    return parse_bool_value(raw.get(key), default=False)


def _to_int(raw: object, *, key: str) -> int | None:
    if not isinstance(raw, dict):
        return None
    return parse_int(raw.get(key), allow_float=True)


def _to_float(raw: object, *, key: str) -> float | None:
    if not isinstance(raw, dict):
        return None
    return parse_finite_float(raw.get(key))
