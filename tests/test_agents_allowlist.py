"""
Tests for WriteAllowlist file restriction enforcement.

Uses tmp_path fixtures to avoid touching real artifacts.
"""

from pathlib import Path

import pytest

from kagglebot.agents import WriteAllowlist


def test_allowlist_basic_pattern(tmp_path: Path):
    """Test basic glob pattern matching."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("kernel/**")

    # Allowed paths
    assert allowlist.is_allowed(tmp_path / "kernel/train.py")
    assert allowlist.is_allowed(tmp_path / "kernel/models/xgb.py")
    assert allowlist.is_allowed(tmp_path / "kernel/utils/preprocess.py")

    # Forbidden paths
    assert not allowlist.is_allowed(tmp_path / "context/data.md")
    assert not allowlist.is_allowed(tmp_path / "plan.json")


def test_allowlist_multiple_patterns(tmp_path: Path):
    """Test multiple allowed patterns."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("kernel/**")
    allowlist.allow("*.md")
    allowlist.allow("runs/*.json")

    # All patterns should work
    assert allowlist.is_allowed(tmp_path / "kernel/train.py")
    assert allowlist.is_allowed(tmp_path / "README.md")
    assert allowlist.is_allowed(tmp_path / "runs/metrics.json")

    # Still forbidden
    assert not allowlist.is_allowed(tmp_path / "context/overview.md")
    assert not allowlist.is_allowed(tmp_path / "plan.json")


def test_allowlist_specific_files(tmp_path: Path):
    """Test allowing specific files."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("context/agent/brief.md")
    allowlist.allow("context/agent/brief.json")

    # Only these specific files allowed
    assert allowlist.is_allowed(tmp_path / "context/agent/brief.md")
    assert allowlist.is_allowed(tmp_path / "context/agent/brief.json")

    # Other files in same directory forbidden
    assert not allowlist.is_allowed(tmp_path / "context/agent/strategy.md")
    assert not allowlist.is_allowed(tmp_path / "context/data.md")


def test_allowlist_path_outside_base_dir(tmp_path: Path):
    """Test that paths outside base_dir raise ValueError."""
    allowlist = WriteAllowlist(base_dir=tmp_path / "competition")

    with pytest.raises(ValueError, match="not under base directory"):
        allowlist.is_allowed(tmp_path / "other/file.txt")


def test_allowlist_relative_paths(tmp_path: Path):
    """Test that relative paths are resolved correctly."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("kernel/**")

    # Relative path should work
    (tmp_path / "kernel").mkdir()
    kernel_file = tmp_path / "kernel" / "train.py"
    kernel_file.touch()

    assert allowlist.is_allowed(kernel_file)


def test_allowlist_empty(tmp_path: Path):
    """Test allowlist with no patterns allows nothing."""
    allowlist = WriteAllowlist(base_dir=tmp_path)

    assert not allowlist.is_allowed(tmp_path / "anything.txt")
    assert not allowlist.is_allowed(tmp_path / "kernel/train.py")


def test_allowlist_repr(tmp_path: Path):
    """Test string representation."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("kernel/**")

    repr_str = repr(allowlist)
    assert "WriteAllowlist" in repr_str
    assert "kernel/**" in repr_str
