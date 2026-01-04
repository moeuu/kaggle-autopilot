from __future__ import annotations

from pathlib import Path


def ensure_solution_path_allowed(path: Path, *, artifacts_dir: Path, slug: str) -> None:
    """Ensure solution code lives only under artifacts/<slug> allowed roots."""
    resolved = path.resolve()
    artifacts_root = artifacts_dir.resolve()

    repo_root = artifacts_root.parent
    forbidden = repo_root / "src" / "kagglebot"
    if _is_relative_to(resolved, forbidden):
        raise ValueError(f"Competition code must not live under {forbidden}.")

    allowed_roots = [
        artifacts_root / slug / "kernel",
        artifacts_root / slug / "solution",
        artifacts_root / slug / "runs",
    ]
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        allowed_str = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"Solution code must live under one of: {allowed_str}.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True
