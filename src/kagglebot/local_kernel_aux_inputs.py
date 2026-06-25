from __future__ import annotations

import shutil
from pathlib import Path

from kagglebot import local_kernel_context as _local_kernel_context
from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_sources import load_kernel_source_config


def stage_local_kernel_aux_inputs(
    *,
    base_dir: Path,
    slug: str,
    kernel_stage_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    source_config = load_kernel_source_config(base_dir / slug / "plan.json")
    text_runtime = source_config.text_runtime
    if not text_runtime.active and not source_config.domain_adaptation.allow_kernel_finetune:
        return {}, []

    env_updates: dict[str, str] = {}
    notes: list[str] = []
    if source_config.domain_adaptation.allow_kernel_finetune:
        env_updates["KAGGLEBOT_ALLOW_KERNEL_FINETUNE"] = "1"
    if text_runtime.metadata_supervision:
        env_updates["KAGGLEBOT_TEXT_METADATA_SUPERVISION"] = text_runtime.metadata_supervision
    if text_runtime.constraint_rewrite_mode:
        env_updates["KAGGLEBOT_TEXT_CONSTRAINT_REWRITE_MODE"] = text_runtime.constraint_rewrite_mode
    if text_runtime.group_key_columns:
        env_updates["KAGGLEBOT_TEXT_GROUP_KEYS"] = ",".join(text_runtime.group_key_columns)

    if not text_runtime.required_aux_inputs:
        return env_updates, notes

    competition_dir = base_dir / slug
    aux_root = kernel_stage_dir / "aux_inputs"
    staged_relpaths: list[str] = []
    missing: list[str] = []
    for spec in text_runtime.required_aux_inputs:
        resolved = resolve_required_aux_input(competition_dir=competition_dir, spec=spec)
        if resolved is None:
            missing.append(spec)
            continue
        relpath = relative_aux_stage_path(competition_dir=competition_dir, source_path=resolved, spec=spec)
        target_path = aux_root / relpath
        stage_local_path_alias(source_path=resolved, target_path=target_path)
        staged_relpaths.append(relpath.as_posix())

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KernelFailedError(
            "Required text runtime aux inputs could not be resolved: "
            f"{missing_text}. Checked competition root, data/, and context/."
        )

    env_updates["KAGGLEBOT_AUX_INPUT_ROOT"] = str(aux_root)
    env_updates["KAGGLEBOT_REQUIRED_AUX_INPUTS"] = ",".join(staged_relpaths)
    notes.append(f"staged {len(staged_relpaths)} text aux input(s)")
    return env_updates, notes


def resolve_required_aux_input(*, competition_dir: Path, spec: str) -> Path | None:
    raw = str(spec).strip().strip("/")
    if not raw:
        return None
    candidates: list[Path] = []
    raw_path = competition_dir / raw
    if "/" in raw or "\\" in raw:
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                competition_dir / "data" / raw,
                competition_dir / "context" / raw,
                competition_dir / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def relative_aux_stage_path(*, competition_dir: Path, source_path: Path, spec: str) -> Path:
    try:
        relative = source_path.resolve().relative_to(competition_dir.resolve())
    except ValueError:
        relative = Path(str(spec).strip().strip("/")).name
    return Path(relative)


def stage_local_path_alias(*, source_path: Path, target_path: Path) -> None:
    if source_path.is_dir():
        _local_kernel_context.stage_local_data_alias(source_dir=source_path, target_dir=target_path)
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        if target_path.is_dir() and not target_path.is_symlink():
            shutil.rmtree(target_path, ignore_errors=True)
        else:
            target_path.unlink(missing_ok=True)
    try:
        target_path.symlink_to(source_path)
    except Exception:
        shutil.copy2(source_path, target_path)
