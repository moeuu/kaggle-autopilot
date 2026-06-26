from __future__ import annotations

import shutil
from pathlib import Path

from kagglebot.json_utils import load_json_object
from kagglebot.kernel_outputs import copy_artifact_if_needed


def stage_local_kernel_data_dir(*, base_dir: Path, slug: str, run_dir: Path) -> None:
    """Stage canonical and compatibility local data directories for generated kernels."""
    competition_dir = base_dir / slug
    source_dir = (competition_dir / "data").resolve()
    if not source_dir.exists():
        return

    stage_local_data_alias(source_dir=source_dir, target_dir=run_dir / "data")
    # Some generated kernels incorrectly resolve local data as
    # <competition_dir>/artifacts/<slug>/data. Keep a compatibility alias
    # to prevent unnecessary runtime autofix loops.
    stage_local_data_alias(
        source_dir=source_dir,
        target_dir=competition_dir / "artifacts" / slug / "data",
    )


def stage_local_kernel_context_profile(*, base_dir: Path, slug: str, run_dir: Path) -> None:
    """Stage dataset profile metadata for kernels that resolve context relative to run_dir."""
    source_path = base_dir / slug / "context" / "dataset_profile.json"
    if not source_path.exists():
        return

    context_dir = run_dir / "context"
    if context_dir.exists() and not context_dir.is_dir():
        if context_dir.is_symlink() or context_dir.is_file():
            context_dir.unlink(missing_ok=True)
        else:
            shutil.rmtree(context_dir, ignore_errors=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    target_path = context_dir / "dataset_profile.json"
    if source_path.resolve() == target_path.resolve():
        return
    if target_path.exists() or target_path.is_symlink():
        if target_path.is_dir() and not target_path.is_symlink():
            shutil.rmtree(target_path, ignore_errors=True)
        else:
            target_path.unlink(missing_ok=True)
    copy_artifact_if_needed(source=source_path, destination=target_path)


def stage_local_data_alias(*, source_dir: Path, target_dir: Path) -> None:
    """Create a symlink/copy alias from target_dir to source_dir."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.is_symlink():
        try:
            if target_dir.resolve() == source_dir:
                return
        except Exception:
            pass
        try:
            target_dir.unlink()
        except OSError:
            pass
    elif target_dir.exists():
        if target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
        else:
            try:
                target_dir.unlink()
            except OSError:
                return

    try:
        target_dir.symlink_to(source_dir, target_is_directory=True)
        return
    except Exception:
        pass

    # Fallback for filesystems where directory symlink is unavailable.
    shutil.copytree(source_dir, target_dir, symlinks=True, dirs_exist_ok=True)


def load_dataset_profile_identity(*, context_dir: Path) -> tuple[str | None, str | None]:
    profile_path = context_dir / "dataset_profile.json"
    payload = load_json_object(profile_path)
    if payload is None:
        return None, None
    target_col = payload.get("target_column")
    id_col = payload.get("id_column")
    target = str(target_col).strip() if isinstance(target_col, str) and str(target_col).strip() else None
    id_value = str(id_col).strip() if isinstance(id_col, str) and str(id_col).strip() else None
    return target, id_value
