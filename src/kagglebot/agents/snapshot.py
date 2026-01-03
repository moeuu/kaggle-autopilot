"""
File snapshot and change detection for agent operations.

Captures filesystem state before/after agent execution to detect
unauthorized modifications.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from kagglebot.agents.allowlist import WriteAllowlist


@dataclass(frozen=True)
class FileState:
    """Immutable file state snapshot."""

    path: Path
    mtime: float
    size: int
    sha256: str


class FileSnapshot:
    """
    Capture and compare filesystem state to detect changes.

    Usage:
        # Before agent execution
        snapshot_pre = FileSnapshot.create(artifacts_dir)

        # Run agent
        run_agent()

        # After agent execution
        snapshot_post = FileSnapshot.create(artifacts_dir)

        # Find violations (files changed outside allowlist)
        allowlist = WriteAllowlist(artifacts_dir)
        allowlist.allow("kernel/**")
        violations = snapshot_pre.diff(snapshot_post, allowlist)

        if violations:
            raise AllowlistViolationError(f"Forbidden changes: {violations}")
    """

    def __init__(self, file_states: dict[Path, FileState]):
        """
        Initialize snapshot with file states.

        Args:
            file_states: Map of absolute paths to their FileState.
        """
        self.file_states = file_states

    @classmethod
    def create(cls, root_dir: Path, *, follow_symlinks: bool = False) -> FileSnapshot:
        """
        Create a snapshot of all files under root_dir.

        Args:
            root_dir: Directory to snapshot recursively.
            follow_symlinks: Whether to follow symbolic links (default: False).

        Returns:
            FileSnapshot capturing current filesystem state.
        """
        root_dir = Path(root_dir).resolve()
        file_states: dict[Path, FileState] = {}

        if not root_dir.exists():
            return cls(file_states)

        for file_path in root_dir.rglob("*"):
            # Skip symlinks unless explicitly following them
            if file_path.is_symlink() and not follow_symlinks:
                continue

            # Only snapshot regular files
            if not file_path.is_file():
                continue

            try:
                stat = os.stat(file_path, follow_symlinks=follow_symlinks)
                mtime = stat.st_mtime
                size = stat.st_size

                # Compute SHA256 hash
                hasher = hashlib.sha256()
                with open(file_path, "rb") as f:
                    # Read in chunks to handle large files
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                sha256 = hasher.hexdigest()

                file_states[file_path] = FileState(
                    path=file_path,
                    mtime=mtime,
                    size=size,
                    sha256=sha256,
                )
            except (OSError, PermissionError):
                # Skip files we can't read
                continue

        return cls(file_states)

    def diff(
        self,
        other: FileSnapshot,
        allowlist: WriteAllowlist | None = None,
    ) -> set[Path]:
        """
        Find files that changed between this snapshot and another.

        Args:
            other: Snapshot to compare against (typically created after this one).
            allowlist: If provided, only return changes that violate the allowlist.
                      If None, return all changes.

        Returns:
            Set of absolute paths that were added, modified, or deleted.
            If allowlist is provided, only returns paths NOT allowed by it.
        """
        changed: set[Path] = set()

        # Find new or modified files
        for path, other_state in other.file_states.items():
            if path not in self.file_states:
                # File was created
                changed.add(path)
            elif self.file_states[path].sha256 != other_state.sha256:
                # File was modified (hash changed)
                changed.add(path)

        # Find deleted files
        for path in self.file_states:
            if path not in other.file_states:
                # File was deleted
                changed.add(path)

        # Filter by allowlist if provided
        if allowlist is not None:
            violations: set[Path] = set()
            for path in changed:
                try:
                    if not allowlist.is_allowed(path):
                        violations.add(path)
                except ValueError:
                    # Path not under allowlist base_dir - always a violation
                    violations.add(path)
            return violations

        return changed

    def __len__(self) -> int:
        """Return number of files in snapshot."""
        return len(self.file_states)

    def __repr__(self) -> str:
        return f"FileSnapshot(files={len(self.file_states)})"
