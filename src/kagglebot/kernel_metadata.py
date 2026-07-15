from __future__ import annotations

import re
from pathlib import Path

from kagglebot.json_utils import load_json_object_or_empty, write_json_object
from kagglebot.kernel_sources import KernelSourceConfig


def sanitize_kernel_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return cleaned[:50]


def resolve_submit_kernel_slug(
    kernel_name: str | None,
    slug: str,
    run_id: str,
    iteration: int,
    *,
    machine_shape: str | None = None,
) -> str:
    shape_token = _machine_shape_slug_token(machine_shape)
    if kernel_name:
        if not shape_token:
            return sanitize_kernel_slug(kernel_name)
        return build_versioned_kernel_slug(
            prefix_parts=("submit", sanitize_kernel_slug(kernel_name)),
            run_id=run_id,
            iteration=iteration,
            fallback_prefix="submit",
            variant=shape_token,
        )
    return build_versioned_kernel_slug(
        prefix_parts=("kagglebot", "submit", slug),
        run_id=run_id,
        iteration=iteration,
        fallback_prefix="kagglebot-submit",
        variant=shape_token,
    )


def _machine_shape_slug_token(machine_shape: str | None) -> str:
    value = str(machine_shape or "").strip()
    if not value:
        return ""
    aliases = {
        "NvidiaTeslaP100": "p100",
        "NvidiaTeslaT4": "t4",
        "NvidiaTeslaT4Highmem": "t4-highmem",
        "NvidiaTeslaA100": "a100",
        "NvidiaL4": "l4",
        "NvidiaL4X1": "l4x1",
        "NvidiaH100": "h100",
        "NvidiaRtxPro6000": "rtx-pro-6000",
    }
    return aliases.get(value, sanitize_kernel_slug(value))


def resolve_kernel_slug(
    kernel_name: str | None,
    slug: str,
    run_id: str,
    iteration: int,
    *,
    machine_shape: str | None = None,
) -> str:
    shape_token = _machine_shape_slug_token(machine_shape)
    if kernel_name:
        return build_versioned_kernel_slug(
            prefix_parts=(sanitize_kernel_slug(kernel_name),),
            run_id=run_id,
            iteration=iteration,
            fallback_prefix="kagglebot",
            variant=shape_token,
        )
    return build_versioned_kernel_slug(
        prefix_parts=("kagglebot", slug),
        run_id=run_id,
        iteration=iteration,
        fallback_prefix="kagglebot",
        variant=shape_token,
    )


def build_versioned_kernel_slug(
    *,
    prefix_parts: tuple[str, ...],
    run_id: str,
    iteration: int,
    fallback_prefix: str,
    variant: str = "",
) -> str:
    suffix_parts = [part for part in (sanitize_kernel_slug(variant), run_id[-6:], f"i{iteration}") if part]
    suffix = "-".join(suffix_parts)
    prefix = "-".join(part for part in prefix_parts if part)
    max_len = 50
    allowed_prefix_len = max_len - len(suffix) - 1
    if allowed_prefix_len < 1:
        prefix = fallback_prefix
    else:
        prefix = prefix[:allowed_prefix_len].rstrip("-")
    return sanitize_kernel_slug(f"{prefix}-{suffix}")


def metadata_source_lists(
    *,
    existing_meta: dict[str, object],
    source_config: KernelSourceConfig | None,
) -> tuple[list[str], list[str], list[str]]:
    source_config = source_config or KernelSourceConfig()
    dataset_sources = list(source_config.dataset_sources)
    model_sources = list(dict.fromkeys((*source_config.model_sources, *source_config.required_model_sources)))
    if source_config.has_explicit_kernel_sources():
        kernel_sources = list(source_config.kernel_sources)
    else:
        raw_existing = existing_meta.get("kernel_sources")
        if isinstance(raw_existing, list):
            kernel_sources = [str(item).strip() for item in raw_existing if str(item).strip()]
        else:
            kernel_sources = []
    return dataset_sources, kernel_sources, model_sources


def write_kernel_metadata(
    *,
    kernel_dir: Path,
    kernel_id: str,
    title: str,
    code_file: str,
    kernel_type: str,
    accelerator: str,
    enable_internet: bool,
    competition_slug: str,
    source_config: KernelSourceConfig | None = None,
) -> None:
    meta_path = kernel_dir / "kernel-metadata.json"
    meta = load_json_object_or_empty(meta_path)
    dataset_sources, kernel_sources, model_sources = metadata_source_lists(
        existing_meta=meta,
        source_config=source_config,
    )
    meta.update(
        {
            "id": kernel_id,
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": kernel_type,
            "is_private": True,
            "enable_gpu": accelerator == "gpu",
            "enable_tpu": accelerator == "tpu",
            "enable_internet": bool(enable_internet),
            "competition_sources": [competition_slug],
            "dataset_sources": dataset_sources,
            "kernel_sources": kernel_sources,
            "model_sources": model_sources,
        }
    )
    if meta["enable_gpu"] and meta["enable_tpu"]:
        raise ValueError("kernel-metadata.json cannot enable both GPU and TPU.")
    write_json_object(meta_path, meta)
