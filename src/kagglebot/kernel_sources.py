from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from kagglebot.json_utils import load_json_object


@dataclass(frozen=True)
class DomainAdaptationConfig:
    adapted_checkpoint_hints: tuple[str, ...] = ()
    allow_kernel_finetune: bool = False


@dataclass(frozen=True)
class TextRuntimeConfig:
    required_aux_inputs: tuple[str, ...] = ()
    metadata_supervision: str = ""
    constraint_rewrite_mode: str = ""
    group_key_columns: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(
            self.required_aux_inputs
            or self.metadata_supervision
            or self.constraint_rewrite_mode
            or self.group_key_columns
        )


@dataclass(frozen=True)
class KernelSourceConfig:
    dataset_sources: tuple[str, ...] = ()
    kernel_sources: tuple[str, ...] = ()
    model_sources: tuple[str, ...] = ()
    pipeline_model_hints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    required_local_seq2seq_pipelines: tuple[str, ...] = ()
    domain_adaptation: DomainAdaptationConfig = field(default_factory=DomainAdaptationConfig)
    text_runtime: TextRuntimeConfig = field(default_factory=TextRuntimeConfig)

    def has_explicit_kernel_sources(self) -> bool:
        return bool(self.kernel_sources)

    def has_text_runtime_features(self) -> bool:
        return self.text_runtime.active or bool(self.domain_adaptation.adapted_checkpoint_hints)


def pipeline_env_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).upper()


def load_kernel_source_config(plan_path: Path) -> KernelSourceConfig:
    payload = load_json_object(plan_path)
    if payload is None:
        return KernelSourceConfig()

    raw = payload.get("kaggle_kernel_sources")
    raw_domain_adaptation = payload.get("domain_adaptation")
    raw_text_runtime = payload.get("text_runtime")
    if not isinstance(raw, dict):
        raw = {}

    return KernelSourceConfig(
        dataset_sources=_normalize_source_list(raw.get("dataset_sources")),
        kernel_sources=_normalize_source_list(raw.get("kernel_sources")),
        model_sources=_normalize_source_list(raw.get("model_sources")),
        pipeline_model_hints=_normalize_pipeline_model_hints(raw.get("pipeline_model_hints")),
        required_local_seq2seq_pipelines=_normalize_source_list(raw.get("required_local_seq2seq_pipelines")),
        domain_adaptation=_normalize_domain_adaptation(raw_domain_adaptation),
        text_runtime=_normalize_text_runtime(raw_text_runtime),
    )


def _normalize_source_list(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = _normalize_source_item(item)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)


def _normalize_source_item(item: object) -> str:
    if isinstance(item, str):
        return item.strip().strip("/")
    if not isinstance(item, dict):
        return ""
    for key in ("ref", "source", "path", "dataset", "kernel", "model"):
        value = str(item.get(key) or "").strip().strip("/")
        if value:
            return value
    owner = str(item.get("ownerSlug") or item.get("owner_slug") or item.get("owner") or "").strip()
    slug = str(item.get("datasetSlug") or item.get("dataset_slug") or item.get("slug") or "").strip()
    if owner and slug:
        return f"{owner}/{slug}"
    return ""


def _normalize_pipeline_model_hints(raw: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        name = str(key or "").strip()
        if not name:
            continue
        hints = _normalize_source_list(value)
        if hints:
            normalized[name] = hints
    return normalized


def _normalize_domain_adaptation(raw: object) -> DomainAdaptationConfig:
    if not isinstance(raw, dict):
        return DomainAdaptationConfig()
    allow_kernel_finetune = bool(raw.get("allow_kernel_finetune", False))
    adapted_checkpoint_hints = _normalize_source_list(raw.get("adapted_checkpoint_hints"))
    if not adapted_checkpoint_hints:
        adapted_checkpoint_hints = _normalize_source_list(raw.get("checkpoint_hints"))
    return DomainAdaptationConfig(
        adapted_checkpoint_hints=adapted_checkpoint_hints,
        allow_kernel_finetune=allow_kernel_finetune,
    )


def _normalize_text_runtime(raw: object) -> TextRuntimeConfig:
    if not isinstance(raw, dict):
        return TextRuntimeConfig()
    metadata_supervision = str(raw.get("metadata_supervision") or "").strip()
    constraint_rewrite_mode = str(raw.get("constraint_rewrite_mode") or "").strip()
    return TextRuntimeConfig(
        required_aux_inputs=_normalize_aux_input_list(raw.get("required_aux_inputs")),
        metadata_supervision=metadata_supervision,
        constraint_rewrite_mode=constraint_rewrite_mode,
        group_key_columns=_normalize_aux_input_list(raw.get("group_key_columns")),
    )


def _normalize_aux_input_list(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            value = item.strip().strip("/")
        elif isinstance(item, dict):
            value = str(item.get("path") or item.get("ref") or item.get("name") or "").strip().strip("/")
        else:
            value = ""
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)
