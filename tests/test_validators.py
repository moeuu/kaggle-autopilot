"""Tests for validator helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from kagglebot.validators import safe_extract_zip, validate_kernel_package, validate_slug


def test_validate_slug() -> None:
    assert validate_slug("titanic") == "titanic"
    with pytest.raises(ValueError, match="Invalid competition slug"):
        validate_slug("../etc/passwd")


def test_safe_extract_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("train.csv", "id,target\n1,0.1\n")

    extracted = safe_extract_zip(zip_path, tmp_path)
    assert any(path.name == "train.csv" for path in extracted)


def test_safe_extract_zip_blocks_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "nope")

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_zip(zip_path, tmp_path)


def test_safe_extract_zip_blocks_prefix_sibling_escape(tmp_path: Path) -> None:
    dest = tmp_path / "data"
    dest.mkdir()
    zip_path = tmp_path / "evil-prefix.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../data_evil.txt", "nope")

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_zip(zip_path, dest)


def test_validate_kernel_package_secret_scan(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("KAGGLE_KEY = 'abc'\n", encoding="utf-8")
    (tmp_path / "kernel-metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Secret pattern detected"):
        validate_kernel_package(tmp_path)


def test_validate_kernel_package_scans_kernel(tmp_path: Path) -> None:
    (tmp_path / "kernel.py").write_text("KAGGLE_KEY = 'abc'\n", encoding="utf-8")
    (tmp_path / "kernel-metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Secret pattern detected"):
        validate_kernel_package(tmp_path)


def test_validate_kernel_package_ignores_non_string_code_file(tmp_path: Path) -> None:
    (tmp_path / "kernel-metadata.json").write_text('{"code_file": ["kernel.py"]}', encoding="utf-8")

    validate_kernel_package(tmp_path)


def test_validate_kernel_package_scans_auxiliary_sources(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "helpers").mkdir()
    (tmp_path / "helpers" / "creds.py").write_text("api_key = '1234567890abcdef'\n", encoding="utf-8")
    (tmp_path / "kernel-metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Secret pattern detected"):
        validate_kernel_package(tmp_path)
