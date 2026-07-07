"""Tests for hashing helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kagglebot.hashing import sha256_file, sha256_file_or_none, sha256_path


def test_sha256_file():
    """Test file hashing produces expected SHA256 digest."""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"abc")
        temp_path = f.name

    try:
        result = sha256_file(temp_path)
        # SHA256 of "abc"
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert result == expected
    finally:
        Path(temp_path).unlink()


def test_sha256_file_large():
    """Test file hashing works with files larger than chunk size."""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        # Write 2MB of data (larger than 1MB chunk size)
        f.write(b"x" * (2 * 1024 * 1024))
        temp_path = f.name

    try:
        result = sha256_file(temp_path)
        # Should produce consistent hash
        assert len(result) == 64  # SHA256 produces 64 hex characters
        assert result == sha256_file(temp_path)  # Consistent
    finally:
        Path(temp_path).unlink()


def test_sha256_file_or_none_handles_missing_and_existing_paths(tmp_path: Path):
    path = tmp_path / "payload.txt"
    path.write_text("abc", encoding="utf-8")

    assert sha256_file_or_none(path) == sha256_file(str(path))
    assert sha256_file_or_none(tmp_path / "missing.txt") is None
    assert sha256_file_or_none(None) is None


def test_sha256_path_hashes_directory_contents_deterministically(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "nested").mkdir(parents=True)
    (second / "nested").mkdir(parents=True)
    (first / "nested" / "chunk").write_bytes(b"payload")
    (first / ".zgroup").write_text("{}", encoding="utf-8")
    (second / ".zgroup").write_text("{}", encoding="utf-8")
    (second / "nested" / "chunk").write_bytes(b"payload")

    digest = sha256_path(first)

    assert digest == sha256_path(second)
    assert digest == sha256_file_or_none(first)
    (second / "nested" / "chunk").write_bytes(b"changed")
    assert digest != sha256_path(second)


def test_sha256_path_includes_directory_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "a").mkdir(parents=True)
    (second / "b").mkdir(parents=True)
    (first / "a" / "chunk").write_bytes(b"same")
    (second / "b" / "chunk").write_bytes(b"same")

    assert sha256_path(first) != sha256_path(second)


def test_sha256_path_includes_empty_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "empty_group").mkdir()

    assert sha256_path(first) != sha256_path(second)


def test_sha256_path_rejects_symlink_file(tmp_path: Path) -> None:
    target = tmp_path / "payload.txt"
    target.write_text("payload", encoding="utf-8")
    link = tmp_path / "submission.txt"
    try:
        link.symlink_to(target.name)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    with pytest.raises(OSError, match="unsupported symlink path for hashing"):
        sha256_path(link)
    assert sha256_file_or_none(link) is None


def test_sha256_path_rejects_symlink_inside_directory(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    target = submission / "payload.txt"
    target.write_text("payload", encoding="utf-8")
    link = submission / "latest.txt"
    try:
        link.symlink_to(target.name)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    with pytest.raises(OSError, match="unsupported symlink path for hashing"):
        sha256_path(submission)
    assert sha256_file_or_none(submission) is None
