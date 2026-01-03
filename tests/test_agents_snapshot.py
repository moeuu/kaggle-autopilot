"""
Tests for FileSnapshot change detection.

Uses tmp_path fixtures to avoid touching real artifacts.
"""

from pathlib import Path

import pytest

from kagglebot.agents import FileSnapshot, WriteAllowlist


def test_snapshot_empty_directory(tmp_path: Path):
    """Test snapshot of empty directory."""
    snapshot = FileSnapshot.create(tmp_path)
    assert len(snapshot) == 0


def test_snapshot_capture_files(tmp_path: Path):
    """Test snapshot captures all files."""
    # Create some files
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.txt").write_text("content2")

    snapshot = FileSnapshot.create(tmp_path)
    assert len(snapshot) == 2
    assert tmp_path / "file1.txt" in snapshot.file_states
    assert tmp_path / "dir1" / "file2.txt" in snapshot.file_states


def test_snapshot_diff_no_changes(tmp_path: Path):
    """Test diff with no changes returns empty set."""
    (tmp_path / "file.txt").write_text("content")

    snapshot1 = FileSnapshot.create(tmp_path)
    snapshot2 = FileSnapshot.create(tmp_path)

    changes = snapshot1.diff(snapshot2)
    assert len(changes) == 0


def test_snapshot_diff_new_file(tmp_path: Path):
    """Test diff detects new files."""
    snapshot1 = FileSnapshot.create(tmp_path)

    (tmp_path / "new_file.txt").write_text("new content")

    snapshot2 = FileSnapshot.create(tmp_path)

    changes = snapshot1.diff(snapshot2)
    assert len(changes) == 1
    assert tmp_path / "new_file.txt" in changes


def test_snapshot_diff_modified_file(tmp_path: Path):
    """Test diff detects modified files."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("original content")

    snapshot1 = FileSnapshot.create(tmp_path)

    # Modify file
    file_path.write_text("modified content")

    snapshot2 = FileSnapshot.create(tmp_path)

    changes = snapshot1.diff(snapshot2)
    assert len(changes) == 1
    assert file_path in changes


def test_snapshot_diff_deleted_file(tmp_path: Path):
    """Test diff detects deleted files."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")

    snapshot1 = FileSnapshot.create(tmp_path)

    # Delete file
    file_path.unlink()

    snapshot2 = FileSnapshot.create(tmp_path)

    changes = snapshot1.diff(snapshot2)
    assert len(changes) == 1
    assert file_path in changes


def test_snapshot_diff_with_allowlist_allowed(tmp_path: Path):
    """Test diff with allowlist returns no violations for allowed changes."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("kernel/**")

    snapshot1 = FileSnapshot.create(tmp_path)

    # Create file in allowed directory
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "train.py").write_text("code")

    snapshot2 = FileSnapshot.create(tmp_path)

    violations = snapshot1.diff(snapshot2, allowlist)
    assert len(violations) == 0


def test_snapshot_diff_with_allowlist_violations(tmp_path: Path):
    """Test diff with allowlist detects violations."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("kernel/**")

    snapshot1 = FileSnapshot.create(tmp_path)

    # Create file in forbidden directory
    (tmp_path / "context").mkdir()
    (tmp_path / "context" / "data.md").write_text("data")

    snapshot2 = FileSnapshot.create(tmp_path)

    violations = snapshot1.diff(snapshot2, allowlist)
    assert len(violations) == 1
    assert tmp_path / "context" / "data.md" in violations


def test_snapshot_diff_mixed_changes(tmp_path: Path):
    """Test diff with both allowed and forbidden changes."""
    allowlist = WriteAllowlist(base_dir=tmp_path)
    allowlist.allow("kernel/**")

    (tmp_path / "kernel").mkdir()
    (tmp_path / "context").mkdir()

    snapshot1 = FileSnapshot.create(tmp_path)

    # Allowed change
    (tmp_path / "kernel" / "train.py").write_text("code")

    # Forbidden change
    (tmp_path / "context" / "data.md").write_text("data")

    snapshot2 = FileSnapshot.create(tmp_path)

    # Without allowlist, both detected
    all_changes = snapshot1.diff(snapshot2)
    assert len(all_changes) == 2

    # With allowlist, only forbidden detected
    violations = snapshot1.diff(snapshot2, allowlist)
    assert len(violations) == 1
    assert tmp_path / "context" / "data.md" in violations
    assert tmp_path / "kernel" / "train.py" not in violations


def test_snapshot_repr(tmp_path: Path):
    """Test string representation."""
    (tmp_path / "file1.txt").write_text("content")
    (tmp_path / "file2.txt").write_text("content")

    snapshot = FileSnapshot.create(tmp_path)
    repr_str = repr(snapshot)

    assert "FileSnapshot" in repr_str
    assert "files=2" in repr_str


def test_snapshot_nonexistent_directory(tmp_path: Path):
    """Test snapshot of nonexistent directory."""
    nonexistent = tmp_path / "does_not_exist"
    snapshot = FileSnapshot.create(nonexistent)

    assert len(snapshot) == 0
