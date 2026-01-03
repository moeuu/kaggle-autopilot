"""
Write allowlist enforcement for agent file operations.

Provides glob-pattern-based file write restrictions to prevent agents
from modifying files outside their designated workspace.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath


class WriteAllowlist:
    """
    Enforce write restrictions on file paths using glob patterns.

    Usage:
        allowlist = WriteAllowlist(base_dir=Path("artifacts/my-competition"))
        allowlist.allow("kernel/**")  # Allow all files under kernel/
        allowlist.allow("*.md")        # Allow markdown files in base dir

        if allowlist.is_allowed(Path("artifacts/my-competition/kernel/train.py")):
            # File write is permitted
            ...
    """

    def __init__(self, base_dir: Path):
        """
        Initialize allowlist with a base directory.

        Args:
            base_dir: Root directory for relative path matching.
                     All patterns are matched relative to this directory.
        """
        self.base_dir = Path(base_dir).resolve()
        self.allowed_patterns: set[str] = set()

    def allow(self, pattern: str) -> None:
        """
        Add an allowed path pattern (glob-style).

        Patterns are matched against paths relative to base_dir using fnmatch.

        Examples:
            - "kernel/**" - All files under kernel/ directory
            - "*.py" - Python files in base directory
            - "data/*.csv" - CSV files in data/ directory
            - "**/*.json" - All JSON files recursively

        Args:
            pattern: Glob-style pattern to allow (fnmatch syntax).
        """
        self.allowed_patterns.add(pattern)

    def is_allowed(self, file_path: Path) -> bool:
        """
        Check if a file path matches any allowed pattern.

        Args:
            file_path: Absolute or relative path to check.
                      If relative, resolved against base_dir.

        Returns:
            True if path matches at least one allowed pattern, False otherwise.

        Raises:
            ValueError: If file_path is not under base_dir.
        """
        file_path = Path(file_path).resolve()

        # Ensure file is under base_dir
        try:
            rel_path = file_path.relative_to(self.base_dir)
        except ValueError as e:
            raise ValueError(
                f"Path {file_path} is not under base directory {self.base_dir}"
            ) from e

        # Use PurePosixPath for consistent Unix-style matching
        posix_path = PurePosixPath(rel_path)
        posix_str = posix_path.as_posix()

        # Check against all patterns
        for pattern in self.allowed_patterns:
            if "/" not in pattern:
                if len(posix_path.parts) == 1 and fnmatchcase(posix_path.name, pattern):
                    return True
                continue
            if pattern.endswith("/**"):
                prefix = pattern[:-3].rstrip("/")
                if posix_str == prefix or posix_str.startswith(prefix + "/"):
                    return True
                continue
            if posix_path.match(pattern):
                return True

        return False

    def __repr__(self) -> str:
        return f"WriteAllowlist(base_dir={self.base_dir}, patterns={self.allowed_patterns})"
