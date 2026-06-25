from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from kagglebot import local_kernel_context as _local_kernel_context
from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_sources import load_kernel_source_config, pipeline_env_suffix

_LOCAL_MODEL_SCAN_MAX_DEPTH = 4
_LOADABLE_MODEL_CONFIG_FILENAMES = ("config.json", "tokenizer_config.json")
_LOADABLE_MODEL_WEIGHT_FILENAMES = ("pytorch_model.bin", "model.safetensors", "tf_model.h5", "flax_model.msgpack")
_GENERIC_MODEL_ALIAS_TOKENS = {"model", "models", "checkpoint", "checkpoints", "ckpt", "snapshot", "snapshots"}


def stage_local_kernel_models(
    *,
    base_dir: Path,
    slug: str,
    kernel_stage_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    source_config = load_kernel_source_config(base_dir / slug / "plan.json")
    if not source_config.pipeline_model_hints and not source_config.model_sources:
        return {}, []

    candidate_dirs = discover_local_model_dirs(base_dir=base_dir, slug=slug)
    staged_root = kernel_stage_dir / "models"
    staged_root.mkdir(parents=True, exist_ok=True)

    env_updates: dict[str, str] = {}
    notes: list[str] = []

    generic_paths = stage_resolved_model_hints(
        hints=source_config.model_sources,
        candidate_dirs=candidate_dirs,
        staged_root=staged_root,
    )
    if generic_paths:
        env_updates["KAGGLEBOT_MODEL_PATHS"] = ",".join(str(path) for path in generic_paths)
        notes.append(f"staged {len(generic_paths)} generic local model source(s)")

    unresolved_required: list[str] = []
    for pipeline_name, hints in source_config.pipeline_model_hints.items():
        staged_paths = stage_resolved_model_hints(
            hints=hints,
            candidate_dirs=candidate_dirs,
            staged_root=staged_root,
        )
        if staged_paths:
            env_updates[f"KAGGLEBOT_MODEL_PATHS_{pipeline_env_suffix(pipeline_name)}"] = ",".join(
                str(path) for path in staged_paths
            )
            notes.append(f"staged {len(staged_paths)} local model source(s) for pipeline={pipeline_name}")
            continue
        if pipeline_name in source_config.required_local_seq2seq_pipelines:
            unresolved_required.append(pipeline_name)

    if unresolved_required:
        required_text = ", ".join(sorted(unresolved_required))
        raise KernelFailedError(
            "Required local seq2seq model sources could not be resolved for "
            f"{required_text}. Checked prior kernel model caches, kernel/models, "
            "context/reference_inputs, and Hugging Face snapshot cache."
        )
    return env_updates, notes


def stage_resolved_model_hints(
    *,
    hints: Sequence[str],
    candidate_dirs: Sequence[Path],
    staged_root: Path,
) -> list[Path]:
    staged_paths: list[Path] = []
    seen_sources: set[Path] = set()
    for hint in hints:
        resolved = resolve_local_model_dir_for_hint(hint=hint, candidate_dirs=candidate_dirs)
        if resolved is None or resolved in seen_sources:
            continue
        seen_sources.add(resolved)
        target_dir = staged_root / sanitize_local_model_stage_name(hint)
        _local_kernel_context.stage_local_data_alias(source_dir=resolved, target_dir=target_dir)
        staged_paths.append(target_dir)
    return staged_paths


def discover_local_model_dirs(*, base_dir: Path, slug: str) -> list[Path]:
    competition_dir = base_dir / slug
    kernels_dir = competition_dir / "kernels"
    roots: list[Path] = [
        competition_dir / "kernel" / "models",
        competition_dir / "context" / "reference_inputs",
    ]
    if kernels_dir.exists():
        roots.extend(sorted(kernels_dir.glob("*/models")))
        roots.extend(sorted(kernels_dir.glob("*/local-iter-*/models")))

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        roots.extend(sorted(hf_cache.glob("models--*/snapshots/*")))

    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in iter_dirs_within_depth(root, _LOCAL_MODEL_SCAN_MAX_DEPTH):
            if not looks_like_local_model_dir(candidate):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(candidate)
    return discovered


def iter_dirs_within_depth(root: Path, max_depth: int) -> list[Path]:
    stack: list[tuple[Path, int]] = [(root, 0)]
    out: list[Path] = []
    while stack:
        current, depth = stack.pop()
        out.append(current)
        if depth >= max_depth:
            continue
        try:
            children = sorted((child for child in current.iterdir() if child.is_dir()), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in reversed(children):
            stack.append((child, depth + 1))
    return out


def looks_like_local_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = any((path / filename).exists() for filename in _LOADABLE_MODEL_CONFIG_FILENAMES)
    has_weights = any((path / filename).exists() for filename in _LOADABLE_MODEL_WEIGHT_FILENAMES)
    return has_config and has_weights


def resolve_local_model_dir_for_hint(*, hint: str, candidate_dirs: Sequence[Path]) -> Path | None:
    hint_text = str(hint).strip()
    if not hint_text:
        return None
    ranked_candidates = [
        path for path in candidate_dirs if local_model_candidate_matches_hint(path=path, hint=hint_text)
    ]
    ranked = sorted(
        ranked_candidates,
        key=lambda path: local_model_rank_key(path=path, hint=hint_text),
    )
    if not ranked:
        return None
    best = ranked[0]
    owner_slug = model_hint_owner_slug_tokens(hint_text)
    owner_slug_score, alias_score = _local_model_match_scores(path=best, hint=hint_text)
    if owner_slug is not None and owner_slug_score <= 0:
        return None
    if alias_score <= 0:
        return None
    return best


def compact_model_ref_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def model_hint_owner_slug_tokens(hint: str) -> tuple[str, str] | None:
    raw = str(hint).strip().strip("/").lower()
    if not raw:
        return None
    parts = [part for part in raw.split("/") if part]
    if len(parts) < 2:
        return None
    owner = compact_model_ref_text(parts[0])
    slug = compact_model_ref_text(parts[1])
    if not owner or not slug:
        return None
    return owner, slug


def local_model_owner_slug_match(path: Path, hint: str) -> int:
    owner_slug = model_hint_owner_slug_tokens(hint)
    if owner_slug is None:
        return 1
    owner_token, slug_token = owner_slug
    compact_path = compact_model_ref_text(path)
    try:
        compact_resolved = compact_model_ref_text(path.resolve())
    except OSError:
        compact_resolved = compact_path
    raw_match = owner_token in compact_path and slug_token in compact_path
    resolved_match = owner_token in compact_resolved and slug_token in compact_resolved
    if path.exists() and raw_match and not resolved_match:
        return -1
    if resolved_match:
        return 3
    if raw_match and not path.exists():
        return 2
    return 0


def local_model_candidate_matches_hint(*, path: Path, hint: str) -> bool:
    return local_model_owner_slug_match(path, hint) > 0


def local_model_rank_key(*, path: Path, hint: str) -> tuple[int, int, int, str]:
    owner_slug_score, score = _local_model_match_scores(path=path, hint=hint)
    text = str(path).lower()
    depth = len(path.parts)
    return (-owner_slug_score, -score, depth, len(text), text)


def _local_model_match_scores(*, path: Path, hint: str) -> tuple[int, int]:
    text = str(path).lower()
    name = path.name.lower()
    owner_slug_score = local_model_owner_slug_match(path, hint)
    score = 0
    for alias in model_ref_aliases(hint):
        if not alias:
            continue
        if name == alias:
            score += 120
        elif text.endswith(f"/{alias}") or text.endswith(f"\\{alias}"):
            score += 100
        elif f"/{alias}/" in text or f"\\{alias}\\" in text:
            score += 70
        elif alias in name:
            score += 55
        elif alias in text:
            score += 35
    lowered_hint = hint.lower()
    for token, weight in (
        ("byt5", 45),
        ("akkadian", 30),
        ("final-byt5", 25),
        ("dpc", 20),
        ("google", 10),
    ):
        if token in lowered_hint and token in text:
            score += weight
    return owner_slug_score, score


def model_ref_aliases(hint: str) -> tuple[str, ...]:
    raw = str(hint).strip().strip("/").lower()
    if not raw:
        return ()
    aliases: list[str] = [
        raw,
        raw.replace("/", "--"),
        raw.replace("/", "-"),
        raw.split("/")[-1],
    ]
    if raw.startswith("models--"):
        aliases.append(raw.removeprefix("models--"))
    tokens = [token for token in re.split(r"[^a-z0-9]+", raw) if token and token not in _GENERIC_MODEL_ALIAS_TOKENS]
    aliases.extend(tokens)
    seen: set[str] = set()
    ordered: list[str] = []
    for alias in aliases:
        cleaned = alias.strip("-_/")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return tuple(ordered)


def sanitize_local_model_stage_name(hint: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(hint)).strip("_").lower()
    return slug or "model"
