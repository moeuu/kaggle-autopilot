"""Tests for hashing helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from kagglebot.hashing import sha256_file, sha256_file_or_none


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
