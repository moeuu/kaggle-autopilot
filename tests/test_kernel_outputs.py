from __future__ import annotations

import gzip
import io
import json
import os
import sqlite3
import stat
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import py7zr
import pytest
import rarfile
import zstandard as zstd

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.kernel_outputs import (
    copy_local_kernel_primary_artifacts,
    copy_optional_local_kernel_artifacts,
    find_intermediate_submission_file,
    find_newest_existing_path,
    find_output_file,
    find_submission_file,
    pick_latest_artifact,
    resolve_local_kernel_artifact_file,
    resolve_local_kernel_artifacts,
)

_ARCHIVE_SUBMISSION_NAMES = (
    "submission.zip",
    "submission.tar",
    "submission.tar.zst",
    "submission.7z",
    "submission.rar",
)


class _FakeZipMember:
    def __init__(
        self,
        filename: str,
        *,
        flag_bits: int = 0,
        is_dir: bool = False,
        external_attr: int = 0,
        create_system: int = 0,
    ) -> None:
        self.filename = filename
        self.flag_bits = flag_bits
        self._is_dir = is_dir
        self.external_attr = external_attr
        self.create_system = create_system

    def is_dir(self) -> bool:
        return self._is_dir


class _FakeZipFile:
    members_by_name: dict[str, list[_FakeZipMember]] = {}

    def __init__(self, path: Path, mode: str = "r") -> None:
        del mode
        self.path = Path(path)

    def __enter__(self) -> _FakeZipFile:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def infolist(self) -> list[_FakeZipMember]:
        return self.members_by_name[self.path.name]


class _Fake7zMember:
    def __init__(
        self,
        filename: str,
        *,
        is_directory: bool = False,
        is_file: bool = True,
        is_symlink: bool = False,
    ) -> None:
        self.filename = filename
        self.is_directory = is_directory
        self.is_file = is_file
        self.is_symlink = is_symlink


class _Fake7zFile:
    members_by_name: dict[str, list[_Fake7zMember]] = {}
    password_required_by_name: dict[str, bool] = {}

    def __init__(self, path: Path, mode: str = "r") -> None:
        del mode
        self.path = Path(path)

    def __enter__(self) -> _Fake7zFile:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def list(self) -> list[_Fake7zMember]:
        return self.members_by_name[self.path.name]

    def needs_password(self) -> bool:
        return self.password_required_by_name.get(self.path.name, False)


class _Failing7zFile:
    error: Exception = py7zr.Bad7zFile("bad 7z")

    def __init__(self, path: Path, mode: str = "r") -> None:
        del path, mode

    def __enter__(self) -> _Failing7zFile:
        raise self.error

    def __exit__(self, *args: object) -> None:
        return None


class _FakeRarMember:
    def __init__(
        self,
        filename: str,
        *,
        is_dir: bool = False,
        is_file: bool = True,
        is_symlink: bool = False,
        needs_password: bool = False,
    ) -> None:
        self.filename = filename
        self._is_dir = is_dir
        self._is_file = is_file
        self._is_symlink = is_symlink
        self._needs_password = needs_password

    def is_dir(self) -> bool:
        return self._is_dir

    def is_file(self) -> bool:
        return self._is_file

    def is_symlink(self) -> bool:
        return self._is_symlink

    def needs_password(self) -> bool:
        return self._needs_password


class _FakeRarFile:
    members_by_name: dict[str, list[_FakeRarMember]] = {}

    def __init__(self, path: Path) -> None:
        self.members = self.members_by_name[Path(path).name]

    def __enter__(self) -> _FakeRarFile:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def infolist(self) -> list[_FakeRarMember]:
        return self.members


class _FailingRarFile:
    error: Exception = rarfile.BadRarFile("bad rar")

    def __init__(self, path: Path) -> None:
        del path

    def __enter__(self) -> _FailingRarFile:
        raise self.error

    def __exit__(self, *args: object) -> None:
        return None


def _write_valid_archive(path: Path) -> None:
    payload = b"prediction\n1\n"
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("predictions.txt", payload)
        return
    if name.endswith(".7z"):
        source = path.with_name(f"{path.stem}_payload.txt")
        source.write_bytes(payload)
        with py7zr.SevenZipFile(path, "w") as archive:
            archive.write(source, "predictions.txt")
        source.unlink()
        return
    mode = {
        ".tar": "w",
        ".tar.gz": "w:gz",
        ".tgz": "w:gz",
        ".tar.bz2": "w:bz2",
        ".tbz2": "w:bz2",
        ".tar.xz": "w:xz",
        ".txz": "w:xz",
    }.get(_archive_suffix_for_test(path))
    if mode is not None:
        with tarfile.open(path, mode) as archive:
            info = tarfile.TarInfo("predictions.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        info = tarfile.TarInfo("predictions.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))


def _write_archive_with_member(path: Path, member_name: str) -> None:
    payload = b"prediction\n1\n"
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member_name, payload)
        return
    mode = {
        ".tar": "w",
        ".tar.gz": "w:gz",
        ".tgz": "w:gz",
        ".tar.bz2": "w:bz2",
        ".tbz2": "w:bz2",
        ".tar.xz": "w:xz",
        ".txz": "w:xz",
    }.get(_archive_suffix_for_test(path))
    if mode is not None:
        with tarfile.open(path, mode) as archive:
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))


def _write_archive_with_symlink(path: Path, member_name: str) -> None:
    name = path.name.lower()
    if name.endswith(".zip"):
        info = zipfile.ZipInfo(member_name)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(info, "target.txt")
        return
    mode = {
        ".tar": "w",
        ".tar.gz": "w:gz",
        ".tgz": "w:gz",
        ".tar.bz2": "w:bz2",
        ".tbz2": "w:bz2",
        ".tar.xz": "w:xz",
        ".txz": "w:xz",
    }.get(_archive_suffix_for_test(path))
    if mode is not None:
        with tarfile.open(path, mode) as archive:
            info = tarfile.TarInfo(member_name)
            info.type = tarfile.SYMTYPE
            info.linkname = "target.txt"
            archive.addfile(info)
        return
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        info = tarfile.TarInfo(member_name)
        info.type = tarfile.SYMTYPE
        info.linkname = "target.txt"
        archive.addfile(info)
    path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))


def _write_archive_with_directory(path: Path, member_name: str) -> None:
    name = path.name.lower()
    directory_name = f"{member_name.rstrip('/')}/"
    if name.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(directory_name, b"")
        return
    mode = {
        ".tar": "w",
        ".tar.gz": "w:gz",
        ".tgz": "w:gz",
        ".tar.bz2": "w:bz2",
        ".tbz2": "w:bz2",
        ".tar.xz": "w:xz",
        ".txz": "w:xz",
    }.get(_archive_suffix_for_test(path))
    if mode is not None:
        with tarfile.open(path, mode) as archive:
            info = tarfile.TarInfo(directory_name)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        return
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        info = tarfile.TarInfo(directory_name)
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
    path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))


def _write_archive_with_duplicate_members(
    path: Path,
    member_name: str,
    *,
    duplicate_member_name: str | None = None,
) -> None:
    payloads = [b"first\n", b"second\n"]
    member_names = [member_name, duplicate_member_name or member_name]
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member_names[0], payloads[0])
            if member_names[1] == member_names[0]:
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(member_names[1], payloads[1])
                return
            else:
                archive.writestr(member_names[1], payloads[1])
        return
    mode = {
        ".tar": "w",
        ".tar.gz": "w:gz",
        ".tgz": "w:gz",
        ".tar.bz2": "w:bz2",
        ".tbz2": "w:bz2",
        ".tar.xz": "w:xz",
        ".txz": "w:xz",
    }.get(_archive_suffix_for_test(path))
    if mode is not None:
        with tarfile.open(path, mode) as archive:
            for member, payload in zip(member_names, payloads, strict=True):
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        for member, payload in zip(member_names, payloads, strict=True):
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))


def _archive_suffix_for_test(path: Path) -> str:
    name = path.name.lower()
    for suffix in (
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
        ".tar.zst",
        ".tgz",
        ".tbz2",
        ".txz",
        ".tzst",
        ".tar",
        ".zip",
        ".7z",
        ".rar",
    ):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def test_find_submission_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    nested = output_dir / "nested"
    nested.mkdir()
    submission = nested / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_zip_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "submission.zip"
    _write_valid_archive(submission)
    assert find_submission_file(output_dir) == submission


def test_find_submission_file_prefers_shapefile_primary_over_newer_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "nested" / "submission.shp"
    submission.parent.mkdir()
    submission.write_bytes(b"shape")
    sidecars = [
        output_dir / "nested" / "submission.dbf",
        output_dir / "nested" / "submission.shx",
        output_dir / "nested" / "submission.prj",
        output_dir / "nested" / "submission.qix",
        output_dir / "nested" / "submission.shp.aux.xml",
    ]
    for index, sidecar in enumerate(sidecars, start=1):
        sidecar.write_bytes(f"sidecar-{index}".encode("ascii"))
        os.utime(sidecar, (2000 + index, 2000 + index))
    os.utime(submission, (1000, 1000))

    assert find_submission_file(output_dir) == submission


def test_find_submission_file_ignores_empty_zip_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    empty = output_dir / "submission.zip"
    valid = output_dir / "submission.csv"
    empty.write_bytes(b"")
    valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(empty, (2000, 2000))
    os.utime(valid, (1000, 1000))

    assert find_submission_file(output_dir) == valid


def test_find_submission_file_ignores_archive_without_file_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ARCHIVE_SUBMISSION_NAMES:
        output_dir = tmp_path / f"output_{name.replace('.', '_')}"
        output_dir.mkdir()
        archive = output_dir / name
        if name.endswith(".7z"):
            archive.write_bytes(b"fake-7z")
            _Fake7zFile.members_by_name = {name: [_Fake7zMember("outputs", is_directory=True, is_file=False)]}
            _Fake7zFile.password_required_by_name = {}
            monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Fake7zFile)
        elif name.endswith(".rar"):
            archive.write_bytes(b"fake-rar")
            _FakeRarFile.members_by_name = {name: [_FakeRarMember("outputs", is_dir=True, is_file=False)]}
            monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)
        else:
            _write_archive_with_directory(archive, "outputs")
        valid = output_dir / "submission.csv"
        valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

        os.utime(archive, (2000, 2000))
        os.utime(valid, (1000, 1000))

        assert find_submission_file(output_dir) == valid


def test_find_submission_file_ignores_invalid_archive_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ARCHIVE_SUBMISSION_NAMES:
        output_dir = tmp_path / f"output_{name.replace('.', '_')}"
        output_dir.mkdir()
        invalid = output_dir / name
        valid = output_dir / "submission.csv"
        invalid.write_bytes(b"not-an-archive")
        if name.endswith(".7z"):
            _Failing7zFile.error = py7zr.Bad7zFile("bad 7z")
            monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Failing7zFile)
        elif name.endswith(".rar"):
            _FailingRarFile.error = rarfile.BadRarFile("bad rar")
            monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FailingRarFile)
        valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

        os.utime(invalid, (2000, 2000))
        os.utime(valid, (1000, 1000))

        assert find_submission_file(output_dir) == valid


def test_find_submission_file_ignores_zip_with_encrypted_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    encrypted = output_dir / "submission.zip"
    encrypted.write_bytes(b"fake-zip")
    valid = output_dir / "submission.csv"
    valid.write_text("id,target\n1,0.1\n", encoding="utf-8")
    _FakeZipFile.members_by_name = {"submission.zip": [_FakeZipMember("predictions.txt", flag_bits=0x1)]}
    monkeypatch.setattr("kagglebot.kernel_outputs.zipfile.ZipFile", _FakeZipFile)

    os.utime(encrypted, (2000, 2000))
    os.utime(valid, (1000, 1000))

    assert find_submission_file(output_dir) == valid


def test_find_submission_file_ignores_archive_with_empty_member_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ARCHIVE_SUBMISSION_NAMES:
        output_dir = tmp_path / f"output_{name.replace('.', '_')}"
        output_dir.mkdir()
        archive = output_dir / name
        if name.endswith(".7z"):
            archive.write_bytes(b"fake-7z")
            _Fake7zFile.members_by_name = {name: [_Fake7zMember("")]}
            _Fake7zFile.password_required_by_name = {}
            monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Fake7zFile)
        elif name.endswith(".rar"):
            archive.write_bytes(b"fake-rar")
            _FakeRarFile.members_by_name = {name: [_FakeRarMember("")]}
            monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)
        else:
            _write_archive_with_member(archive, "")
        valid = output_dir / "submission.csv"
        valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

        os.utime(archive, (2000, 2000))
        os.utime(valid, (1000, 1000))

        assert find_submission_file(output_dir) == valid


def test_find_submission_file_ignores_archive_with_duplicate_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ARCHIVE_SUBMISSION_NAMES:
        output_dir = tmp_path / f"output_{name.replace('.', '_')}"
        output_dir.mkdir()
        duplicate = output_dir / name
        if name.endswith(".7z"):
            duplicate.write_bytes(b"fake-7z")
            _Fake7zFile.members_by_name = {name: [_Fake7zMember("predictions.txt"), _Fake7zMember("predictions.txt")]}
            _Fake7zFile.password_required_by_name = {}
            monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Fake7zFile)
        elif name.endswith(".rar"):
            duplicate.write_bytes(b"fake-rar")
            _FakeRarFile.members_by_name = {
                name: [_FakeRarMember("predictions.txt"), _FakeRarMember("predictions.txt")]
            }
            monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)
        else:
            _write_archive_with_duplicate_members(duplicate, "predictions.txt")
        valid = output_dir / "submission.csv"
        valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

        os.utime(duplicate, (2000, 2000))
        os.utime(valid, (1000, 1000))

        assert find_submission_file(output_dir) == valid


def test_find_submission_file_ignores_archive_with_normalized_duplicate_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ARCHIVE_SUBMISSION_NAMES:
        output_dir = tmp_path / f"output_{name.replace('.', '_')}"
        output_dir.mkdir()
        duplicate = output_dir / name
        if name.endswith(".7z"):
            duplicate.write_bytes(b"fake-7z")
            _Fake7zFile.members_by_name = {name: [_Fake7zMember("predictions.txt"), _Fake7zMember("./predictions.txt")]}
            _Fake7zFile.password_required_by_name = {}
            monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Fake7zFile)
        elif name.endswith(".rar"):
            duplicate.write_bytes(b"fake-rar")
            _FakeRarFile.members_by_name = {
                name: [_FakeRarMember("predictions.txt"), _FakeRarMember("./predictions.txt")]
            }
            monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)
        else:
            _write_archive_with_duplicate_members(
                duplicate,
                "predictions.txt",
                duplicate_member_name="./predictions.txt",
            )
        valid = output_dir / "submission.csv"
        valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

        os.utime(duplicate, (2000, 2000))
        os.utime(valid, (1000, 1000))

        assert find_submission_file(output_dir) == valid


@pytest.mark.parametrize(
    ("name", "member_name"),
    [
        ("submission.zip", "../evil.txt"),
        ("submission.tar", "..\\evil.txt"),
        ("submission.tar.zst", "/tmp/evil.txt"),
        ("submission.7z", "C:/tmp/evil.txt"),
        ("submission.rar", "C:\\tmp\\evil.txt"),
    ],
)
def test_find_submission_file_ignores_archive_with_unsafe_member_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    member_name: str,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    unsafe = output_dir / name
    if name.endswith(".7z"):
        unsafe.write_bytes(b"fake-7z")
        _Fake7zFile.members_by_name = {name: [_Fake7zMember(member_name)]}
        _Fake7zFile.password_required_by_name = {}
        monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Fake7zFile)
    elif name.endswith(".rar"):
        unsafe.write_bytes(b"fake-rar")
        _FakeRarFile.members_by_name = {name: [_FakeRarMember(member_name)]}
        monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)
    else:
        _write_archive_with_member(unsafe, member_name)
    valid = output_dir / "submission.csv"
    valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(unsafe, (2000, 2000))
    os.utime(valid, (1000, 1000))

    assert find_submission_file(output_dir) == valid


def test_find_submission_file_ignores_archive_with_symlink_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ARCHIVE_SUBMISSION_NAMES:
        output_dir = tmp_path / f"output_{name.replace('.', '_')}"
        output_dir.mkdir()
        archive = output_dir / name
        if name.endswith(".7z"):
            archive.write_bytes(b"fake-7z")
            _Fake7zFile.members_by_name = {name: [_Fake7zMember("latest.txt", is_symlink=True)]}
            _Fake7zFile.password_required_by_name = {}
            monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Fake7zFile)
        elif name.endswith(".rar"):
            archive.write_bytes(b"fake-rar")
            _FakeRarFile.members_by_name = {name: [_FakeRarMember("latest.txt", is_symlink=True)]}
            monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)
        else:
            _write_archive_with_symlink(archive, "latest.txt")
        valid = output_dir / "submission.csv"
        valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

        os.utime(archive, (2000, 2000))
        os.utime(valid, (1000, 1000))

        assert find_submission_file(output_dir) == valid


@pytest.mark.parametrize(
    "name",
    [
        "submission.tsv",
        "submission.jsonl",
        "submission.jsonlines",
        "submission.ndjson",
        "submission.pkl",
        "submission.dta",
        "submission.xml",
        "submission.orc",
        "submission.hdf5",
    ],
)
def test_find_submission_file_supports_non_csv_tabular_submission(tmp_path: Path, name: str) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "nested" / name
    submission.parent.mkdir()
    if name.endswith(".pkl"):
        pd.DataFrame({"id": [1], "target": [0.1]}).to_pickle(submission)
    elif name.endswith(".dta"):
        pd.DataFrame({"id": [1], "target": [0.1]}).to_stata(submission, write_index=False)
    elif name.endswith(".xml"):
        pd.DataFrame({"id": [1], "target": [0.1]}).to_xml(submission, index=False, parser="etree")
    elif name.endswith(".orc"):
        pd.DataFrame({"id": [1], "target": [0.1]}).to_orc(submission, index=False)
    elif name.endswith(".hdf5"):
        pd.DataFrame({"id": [1], "target": [0.1]}).to_hdf(
            submission,
            key="submission",
            mode="w",
            format="table",
            index=False,
        )
    else:
        payload = "id\ttarget\n1\t0.1\n" if name.endswith(".tsv") else '{"id":1,"target":0.1}\n'
        submission.write_text(payload, encoding="utf-8")

    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_compressed_tabular_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "nested" / "submission.csv.gz"
    submission.parent.mkdir()
    with gzip.open(submission, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.1\n")

    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_zstd_compressed_tabular_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "nested" / "submission.csv.zst"
    submission.parent.mkdir()
    submission.write_bytes(zstd.ZstdCompressor().compress(b"id,target\n1,0.1\n"))

    assert find_submission_file(output_dir) == submission


@pytest.mark.parametrize(
    "name",
    [
        "submission.jsonl.gz",
        "submission.jsonl.zst",
        "submission.jsonlines.gz",
        "submission.jsonlines.zst",
        "submission.ndjson.gz",
        "submission.ndjson.zst",
    ],
)
def test_find_submission_file_supports_compressed_json_lines_submission(
    tmp_path: Path,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "nested" / name
    submission.parent.mkdir()
    payload = b'{"id":1,"target":0.1}\n'
    if name.endswith(".gz"):
        with gzip.open(submission, "wb") as handle:
            handle.write(payload)
    else:
        submission.write_bytes(zstd.ZstdCompressor().compress(payload))

    assert find_submission_file(output_dir) == submission


@pytest.mark.parametrize("name", ["submission.csv.gz", "submission.csv.zst"])
def test_find_submission_file_ignores_corrupt_compressed_tabular_submission(
    tmp_path: Path,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    corrupt = output_dir / name
    corrupt.write_bytes(b"not-compressed")
    valid = output_dir / "submission.csv"
    valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(corrupt, (2000, 2000))
    os.utime(valid, (1000, 1000))

    assert find_submission_file(output_dir) == valid


@pytest.mark.parametrize(
    "name",
    [
        "submission.jsonl",
        "submission.jsonl.zst",
        "submission.jsonlines",
        "submission.jsonlines.zst",
        "submission.ndjson",
        "submission.ndjson.zst",
    ],
)
def test_find_submission_file_ignores_corrupt_json_lines_submission(
    tmp_path: Path,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    corrupt = output_dir / name
    if name.endswith(".zst"):
        corrupt.write_bytes(zstd.ZstdCompressor().compress(b"not-json\n"))
    else:
        corrupt.write_text("not-json\n", encoding="utf-8")
    valid = output_dir / "submission.csv"
    valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(corrupt, (2000, 2000))
    os.utime(valid, (1000, 1000))

    assert find_submission_file(output_dir) == valid


def test_find_submission_file_supports_excel_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "nested" / "submission.xlsx"
    submission.parent.mkdir()
    pd.DataFrame({"id": [1], "target": [0.1]}).to_excel(submission, index=False)

    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_feather_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "submission.feather"
    pd.DataFrame({"id": [1], "target": [0.1]}).to_feather(submission)

    assert find_submission_file(output_dir) == submission


@pytest.mark.parametrize(
    ("name", "nested"),
    [
        ("submission.tar", False),
        ("submission.tar.gz", False),
        ("submission.tgz", False),
        ("submission.tar.bz2", False),
        ("submission.tbz2", False),
        ("submission.tar.xz", False),
        ("submission.txz", False),
        ("submission.tar.zst", False),
        ("submission.tzst", False),
        ("submission.7z", False),
        ("submission.tar.zst", True),
    ],
)
def test_find_submission_file_supports_compound_code_submission_archives(
    tmp_path: Path,
    name: str,
    nested: bool,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact_dir = output_dir / "nested" if nested else output_dir
    artifact_dir.mkdir(exist_ok=True)
    submission = artifact_dir / name
    _write_valid_archive(submission)
    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_standard_rar_submission_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "submission.rar"
    submission.write_bytes(b"rar")
    _FakeRarFile.members_by_name = {"submission.rar": [_FakeRarMember("predictions.txt")]}
    monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)

    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_submission_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "mask.tif").write_bytes(b"mask")
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "staging_dir": "bundle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert find_submission_file(output_dir) == manifest


def test_find_submission_file_prefers_manifest_single_file_submission_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    csv_submission = output_dir / "submission.csv"
    csv_submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    archive_submission = output_dir / "submission.tar.gz"
    _write_valid_archive(archive_submission)
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "submission.tar.gz",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert find_submission_file(output_dir) == archive_submission


def test_find_submission_file_uses_manifest_output_file_alias(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "predictions.jsonl"
    submission.write_text('{"id":1,"target":0.1}\n', encoding="utf-8")
    default = output_dir / "submission.csv"
    default.write_text("id,target\n1,0.0\n", encoding="utf-8")
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "output_file": "predictions.jsonl",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert find_submission_file(output_dir) == submission


def test_find_submission_file_uses_manifest_single_file_directory_submission_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "predictions.zarr"
    submission.mkdir()
    (submission / ".zarray").write_text("{}", encoding="utf-8")
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "output_path": "predictions.zarr",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert find_submission_file(output_dir) == submission


def test_find_submission_file_ignores_manifest_single_file_directory_with_only_symlinks(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "predictions.zarr"
    submission.mkdir()
    target = tmp_path / ".zarray"
    target.write_text("{}", encoding="utf-8")
    link = submission / ".zarray"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "output_path": "predictions.zarr",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert find_submission_file(output_dir) is None


def test_find_submission_file_keeps_bundle_directory_manifest_authoritative(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    bundle = output_dir / "bundle"
    bundle.mkdir()
    (bundle / "mask.tif").write_bytes(b"mask")
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "bundle",
                "submission_path": "bundle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert find_submission_file(output_dir) == manifest


def test_find_submission_file_prefers_archive_over_header_only_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    archive_submission = output_dir / "nested" / "submission.tar.gz"
    archive_submission.parent.mkdir()
    _write_valid_archive(archive_submission)
    csv_submission = output_dir / "submission.csv"
    csv_submission.write_text("id,target\n", encoding="utf-8")

    os.utime(archive_submission, (1000, 1000))
    os.utime(csv_submission, (2000, 2000))

    assert find_submission_file(output_dir) == archive_submission


def test_find_output_file_picks_newest_match(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    direct = output_dir / "metrics.json"
    direct.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    nested = output_dir / "nested"
    nested.mkdir()
    newest = nested / "metrics.json"
    newest.write_text('{"metric":"rmse","offline_value":0.1}\n', encoding="utf-8")

    os.utime(direct, (1000, 1000))
    os.utime(newest, (2000, 2000))

    assert find_output_file(output_dir, "metrics.json") == newest


def test_find_output_file_prefers_newest_under_run_tree(tmp_path: Path) -> None:
    root = tmp_path / "kernel-run"
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "run_2").mkdir(parents=True, exist_ok=True)
    older = root / "outputs" / "metrics.json"
    newer = root / "runs" / "run_2" / "metrics.json"
    older.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    newer.write_text('{"metric":"rmse","offline_value":0.1}\n', encoding="utf-8")

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    assert find_output_file(root, "metrics.json") == newer


def test_find_output_file_ignores_generated_runtime_site(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    runtime_dir = output_dir / ".model_runtime_site" / "package"
    runtime_dir.mkdir(parents=True)
    expected = output_dir / "metrics.json"
    decoy = runtime_dir / "metrics.json"
    expected.write_text('{"metric":"rmse","offline_value":0.1}\n', encoding="utf-8")
    decoy.write_text('{"package_metadata":true}\n', encoding="utf-8")
    os.utime(expected, (1000, 1000))
    os.utime(decoy, (2000, 2000))

    assert find_output_file(output_dir, "metrics.json") == expected


def test_find_submission_file_uses_newest_fold_intermediate_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fold1 = output_dir / "submission_qwen_fold1.csv"
    fold2 = output_dir / "nested" / "submission_qwen_fold2.csv"
    fold2.parent.mkdir()
    fold1.write_text("id,target\n1,0.1\n", encoding="utf-8")
    fold2.write_text("id,target\n1,0.2\n", encoding="utf-8")

    os.utime(fold1, (1000, 1000))
    os.utime(fold2, (2000, 2000))

    assert find_intermediate_submission_file(output_dir) == fold2
    assert find_submission_file(output_dir) == fold2


def test_find_submission_file_uses_non_csv_fold_intermediate_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fold1 = output_dir / "submission_qwen_fold1.tsv"
    fold2 = output_dir / "nested" / "submission_qwen_fold2.tsv"
    fold2.parent.mkdir()
    fold1.write_text("id\ttarget\n1\t0.1\n", encoding="utf-8")
    fold2.write_text("id\ttarget\n1\t0.2\n", encoding="utf-8")

    os.utime(fold1, (1000, 1000))
    os.utime(fold2, (2000, 2000))

    assert find_intermediate_submission_file(output_dir) == fold2
    assert find_submission_file(output_dir) == fold2


def test_find_submission_file_uses_compressed_fold_intermediate_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fold1 = output_dir / "submission_qwen_fold1.csv.gz"
    fold2 = output_dir / "nested" / "submission_qwen_fold2.csv.gz"
    fold2.parent.mkdir()
    for path, target in ((fold1, "0.1"), (fold2, "0.2")):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(f"id,target\n1,{target}\n")

    os.utime(fold1, (1000, 1000))
    os.utime(fold2, (2000, 2000))

    assert find_intermediate_submission_file(output_dir) == fold2
    assert find_submission_file(output_dir) == fold2


def test_find_submission_file_prefers_final_submission_over_fold_intermediate(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    final = output_dir / "submission.csv"
    fold = output_dir / "submission_qwen_fold1.csv"
    final.write_text("id,target\n1,0.3\n", encoding="utf-8")
    fold.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(final, (1000, 1000))
    os.utime(fold, (2000, 2000))

    assert find_submission_file(output_dir) == final


def test_find_submission_file_prefers_non_csv_final_submission_over_fold_intermediate(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    final = output_dir / "submission.tsv"
    fold = output_dir / "submission_qwen_fold1.tsv"
    final.write_text("id\ttarget\n1\t0.3\n", encoding="utf-8")
    fold.write_text("id\ttarget\n1\t0.1\n", encoding="utf-8")

    os.utime(final, (1000, 1000))
    os.utime(fold, (2000, 2000))

    assert find_submission_file(output_dir) == final


def test_find_submission_file_prefers_excel_final_submission_over_fold_intermediate(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    final = output_dir / "submission.xlsx"
    fold = output_dir / "submission_qwen_fold1.csv"
    pd.DataFrame({"id": [1], "target": [0.3]}).to_excel(final, index=False)
    fold.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(final, (1000, 1000))
    os.utime(fold, (2000, 2000))

    assert find_submission_file(output_dir) == final


def test_find_submission_file_prefers_configured_submission_filename(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    configured = output_dir / "submission.jsonl"
    default = output_dir / "submission.csv"
    configured.write_text('{"id":1,"target":0.3}\n', encoding="utf-8")
    default.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(configured, (1000, 1000))
    os.utime(default, (2000, 2000))
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.jsonl")

    assert find_submission_file(output_dir) == configured


def test_find_submission_file_prefers_configured_compressed_tabular_over_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    configured = output_dir / "submission.ndjson.zst"
    archive = output_dir / "submission.tar.gz"
    configured.write_bytes(zstd.ZstdCompressor().compress(b'{"id":1,"target":0.3}\n'))
    _write_valid_archive(archive)

    os.utime(configured, (1000, 1000))
    os.utime(archive, (2000, 2000))
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "submission.ndjson.zst")

    assert find_submission_file(output_dir) == configured


def test_find_submission_file_finds_configured_non_submission_filename_in_nested_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    configured = nested / "predictions.csv.gz"
    default = output_dir / "submission.csv"
    with gzip.open(configured, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.3\n")
    default.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(configured, (1000, 1000))
    os.utime(default, (2000, 2000))
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "predictions.csv.gz")

    assert find_submission_file(output_dir) == configured


def test_find_submission_file_honors_configured_non_tabular_single_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    configured = nested / "predictions.bin"
    configured.write_bytes(b"opaque-single-file-submission")

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "predictions.bin")

    assert find_submission_file(output_dir) == configured


def test_find_submission_file_ignores_empty_configured_non_tabular_single_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    configured = output_dir / "predictions.bin"
    valid = output_dir / "submission.csv"
    configured.write_bytes(b"")
    valid.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(configured, (2000, 2000))
    os.utime(valid, (1000, 1000))
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "predictions.bin")

    assert find_submission_file(output_dir) == valid


@pytest.mark.parametrize("name", ["metrics.json", "plan.json", "cv_results.json", "CV_RESULTS.JSON"])
def test_find_submission_file_ignores_configured_non_submission_artifact_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact = output_dir / name
    artifact.write_text('{"value": 1}\n', encoding="utf-8")

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", name)

    assert find_submission_file(output_dir) is None


@pytest.mark.parametrize("name", ["oof_predictions.parquet", "feature_suspects.tsv.gz"])
def test_find_submission_file_ignores_configured_optional_artifact_name_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact = output_dir / name
    if name.endswith(".parquet"):
        pd.DataFrame({"id": [1], "target": [0.3]}).to_parquet(artifact, index=False)
    else:
        artifact.write_text("id,target\n1,0.3\n", encoding="utf-8")

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", name)

    assert find_submission_file(output_dir) is None


@pytest.mark.parametrize("name", ["sample_submission.csv", "sample-submission.csv.gz", "submission_template.csv"])
def test_find_submission_file_ignores_configured_template_submission_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    template = output_dir / name
    valid = output_dir / "submission.csv"
    if name.endswith(".gz"):
        with gzip.open(template, "wt", encoding="utf-8") as handle:
            handle.write("id,target\n1,0\n")
    else:
        template.write_text("id,target\n1,0\n", encoding="utf-8")
    valid.write_text("id,target\n1,0.3\n", encoding="utf-8")
    os.utime(template, (2000, 2000))
    os.utime(valid, (1000, 1000))

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", name)

    assert find_submission_file(output_dir) == valid


@pytest.mark.parametrize(
    "name, marker_name",
    [
        ("answers.nii.gz", None),
        ("predictions.zarr", ".zarray"),
        ("labels.ome.zarr", "zarr.json"),
        ("volumes.n5", "attributes.json"),
        ("model_bundle.tar.zst", None),
    ],
)
def test_find_submission_file_prefers_configured_non_tabular_submission_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    marker_name: str | None,
) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    configured = nested / name
    default = output_dir / "submission.csv"
    if name.endswith(".tar.zst"):
        compressor = zstd.ZstdCompressor()
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w") as archive:
            payload = b"answer"
            info = tarfile.TarInfo("predictions.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        configured.write_bytes(compressor.compress(tar_bytes.getvalue()))
    elif marker_name is not None:
        configured.mkdir()
        (configured / marker_name).write_text("{}", encoding="utf-8")
    else:
        configured.write_bytes(b"single-file-submission")
    default.write_text("id,target\n1,0.1\n", encoding="utf-8")
    os.utime(configured, (1000, 1000))
    os.utime(default, (2000, 2000))

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", name)

    assert find_submission_file(output_dir) == configured


def test_find_submission_file_prefers_configured_non_tabular_single_file_over_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    configured = nested / "answers.nii.gz"
    archive = output_dir / "submission.tar.gz"
    configured.write_bytes(b"single-file-submission")
    _write_valid_archive(archive)

    os.utime(configured, (1000, 1000))
    os.utime(archive, (2000, 2000))
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "answers.nii.gz")

    assert find_submission_file(output_dir) == configured


def test_find_submission_file_finds_generic_prediction_filename_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    predictions = output_dir / "predictions.csv"
    predictions.write_text("id,target\n1,0.3\n", encoding="utf-8")

    assert find_submission_file(output_dir) == predictions


@pytest.mark.parametrize(
    "name",
    [
        "preds.csv",
        "sub.csv",
        "solution.csv",
        "outputs.csv",
        "submission_final.csv",
        "final_submission.csv",
    ],
)
def test_find_submission_file_finds_common_generic_final_aliases(tmp_path: Path, name: str) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    alias = output_dir / name
    alias.write_text("id,target\n1,0.3\n", encoding="utf-8")

    assert find_submission_file(output_dir) == alias


def test_find_submission_file_finds_plain_submission_directory_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    submission_dir = output_dir / "nested" / "submission"
    submission_dir.mkdir(parents=True)
    (submission_dir / "case_001.png").write_bytes(b"mask")

    assert find_submission_file(output_dir) == submission_dir


def test_find_submission_file_finds_generic_prediction_directory_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    prediction_dir = output_dir / "nested" / "predictions"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "case_001.png").write_bytes(b"mask")

    assert find_submission_file(output_dir) == prediction_dir


def test_find_submission_file_ignores_generic_prediction_directory_with_only_symlinks(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    prediction_dir = output_dir / "nested" / "predictions"
    prediction_dir.mkdir(parents=True)
    target = tmp_path / "case_001.png"
    target.write_bytes(b"mask")
    link = prediction_dir / "case_001.png"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_ignores_generic_prediction_directory_with_only_optional_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    prediction_dir = output_dir / "nested" / "predictions"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "metrics.json").write_text('{"score": 0.0}\n', encoding="utf-8")
    (prediction_dir / "cv_results.json").write_text('{"folds": []}\n', encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_ignores_generic_prediction_directory_with_only_empty_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    prediction_dir = output_dir / "nested" / "predictions"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "case_001.png").write_bytes(b"")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_ignores_validation_prediction_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    prediction_dir = output_dir / "validation_predictions"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "case_001.png").write_bytes(b"mask")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_prefers_generic_prediction_file_over_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "case_001.png").write_bytes(b"mask")
    prediction_file = output_dir / "predictions.csv"
    prediction_file.write_text("id,target\n1,0\n", encoding="utf-8")

    assert find_submission_file(output_dir) == prediction_file


def test_find_submission_file_ignores_plain_submission_directory_with_only_optional_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True)
    (submission_dir / "metrics.json").write_text('{"score": 0.0}\n', encoding="utf-8")

    assert find_submission_file(output_dir) is None


@pytest.mark.parametrize("name", ["oof_predictions.parquet", "feature_suspects.tsv.gz", "cv_results.csv"])
def test_find_submission_file_ignores_plain_submission_directory_with_only_optional_artifact_variants(
    tmp_path: Path,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True)
    artifact = submission_dir / name
    if name.endswith(".parquet"):
        pd.DataFrame({"id": [1], "target": [0.1]}).to_parquet(artifact, index=False)
    else:
        artifact.write_text("id,target\n1,0.1\n", encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_accepts_plain_submission_directory_with_output_and_optional_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True)
    (submission_dir / "oof_predictions.parquet").write_bytes(b"optional")
    (submission_dir / "predictions.tsv").write_text("id\ttarget\n1\t0.1\n", encoding="utf-8")

    assert find_submission_file(output_dir) == submission_dir


def test_find_submission_file_ignores_plain_submission_directory_with_only_symlinks(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True)
    target = tmp_path / "mask.png"
    target.write_bytes(b"mask")
    link = submission_dir / "case_001.png"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_prefers_submission_file_over_plain_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission_dir = output_dir / "submission"
    submission_dir.mkdir()
    (submission_dir / "case_001.png").write_bytes(b"mask")
    submission_file = output_dir / "submission.csv"
    submission_file.write_text("id,target\n1,0\n", encoding="utf-8")

    assert find_submission_file(output_dir) == submission_file


def test_find_submission_file_ignores_sample_submission_generic_alias(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sample = output_dir / "sample_submission.csv"
    sample.write_text("id,target\n1,0\n", encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_finds_generic_answer_parquet_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    answer = nested / "answer.parquet"
    pd.DataFrame({"id": [1], "target": [0.3]}).to_parquet(answer, index=False)

    assert find_submission_file(output_dir) == answer


@pytest.mark.parametrize("name", ["predictions.csv.gz", "results.jsonl.zst"])
def test_find_submission_file_finds_generic_compressed_tabular_alias_when_final_missing(
    tmp_path: Path,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    artifact = nested / name
    if name.endswith(".gz"):
        with gzip.open(artifact, "wt", encoding="utf-8") as handle:
            handle.write("id,target\n1,0.3\n")
    else:
        artifact.write_bytes(zstd.ZstdCompressor().compress(b'{"id":1,"target":0.3}\n'))

    assert find_submission_file(output_dir) == artifact


@pytest.mark.parametrize(
    "name",
    [
        "predictions.zip",
        "answers.tar.gz",
        "results.tgz",
        "masks.tar.bz2",
        "preds.tbz2",
        "answers.txz",
        "answers.7z",
        "results.rar",
    ],
)
def test_find_submission_file_finds_generic_archive_alias_when_final_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    artifact = nested / name
    if name.endswith(".rar"):
        artifact.write_bytes(b"rar")
        _FakeRarFile.members_by_name = {name: [_FakeRarMember("predictions.txt")]}
        monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)
    else:
        _write_valid_archive(artifact)

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_ignores_password_protected_7z_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    protected = output_dir / "submission.7z"
    protected.write_bytes(b"fake-7z")
    fallback = output_dir / "submission.csv"
    fallback.write_text("id,target\n1,0.3\n", encoding="utf-8")
    _Fake7zFile.members_by_name = {"submission.7z": [_Fake7zMember("predictions.txt")]}
    _Fake7zFile.password_required_by_name = {"submission.7z": True}
    monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Fake7zFile)

    assert find_submission_file(output_dir) == fallback


def test_find_submission_file_ignores_password_protected_rar_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    protected = output_dir / "submission.rar"
    protected.write_bytes(b"fake-rar")
    fallback = output_dir / "submission.csv"
    fallback.write_text("id,target\n1,0.3\n", encoding="utf-8")
    _FakeRarFile.members_by_name = {"submission.rar": [_FakeRarMember("predictions.txt", needs_password=True)]}
    monkeypatch.setattr("kagglebot.kernel_outputs.rarfile.RarFile", _FakeRarFile)

    assert find_submission_file(output_dir) == fallback


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(py7zr.DecompressionError("cannot decompress"), id="decompression"),
        pytest.param(
            py7zr.UnsupportedCompressionMethodError("bcj2", "unsupported compression"),
            id="unsupported-compression",
        ),
    ],
)
def test_find_submission_file_ignores_unreadable_7z_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    unreadable = output_dir / "submission.7z"
    unreadable.write_bytes(b"fake-7z")
    fallback = output_dir / "submission.csv"
    fallback.write_text("id,target\n1,0.3\n", encoding="utf-8")
    _Failing7zFile.error = exception
    monkeypatch.setattr("kagglebot.kernel_outputs.py7zr.SevenZipFile", _Failing7zFile)

    assert find_submission_file(output_dir) == fallback


def test_find_submission_file_finds_generic_non_tabular_single_file_when_final_missing(tmp_path: Path) -> None:
    names = [
        "predictions.npy",
        "results.npz",
        "answers.nii.gz",
        "mask.svs",
        "mask.ome.tif",
        "submission.pdf",
        "submission.docx",
        "submission.geojson",
        "submission.gpkg",
        "submission.pdb",
        "submission.sdf",
        "submission.fasta",
        "submission.graphml",
        "submission.gexf",
        "submission.edgelist",
        "submission.nc",
        "submission.grib2",
        "submission.fits",
        "submission.h5ad",
        "submission.loom",
        "submission.mat",
        "submission.zarr",
        "submission.onnx",
        "submission.safetensors",
        "submission.safetensors.index.json",
        "submission.gguf",
        "submission.msgpack",
        "submission.tflite",
        "submission.pb",
        "submission.joblib",
    ]
    for index, name in enumerate(names):
        output_dir = tmp_path / f"output-{index}"
        nested = output_dir / "nested"
        nested.mkdir(parents=True)
        artifact = nested / name
        artifact.write_bytes(b"single-file-submission")

        assert find_submission_file(output_dir) == artifact, name


def test_find_submission_file_finds_generic_sqlite_single_file_when_final_missing(tmp_path: Path) -> None:
    for index, name in enumerate(["predictions.sqlite", "answers.sqlite3", "results.db"]):
        output_dir = tmp_path / f"output-{index}"
        nested = output_dir / "nested"
        nested.mkdir(parents=True)
        artifact = nested / name
        with sqlite3.connect(artifact) as conn:
            conn.execute("CREATE TABLE predictions (id INTEGER, target REAL)")
            conn.execute("INSERT INTO predictions VALUES (?, ?)", (1, 0.5))

        assert find_submission_file(output_dir) == artifact, name


def test_find_submission_file_ignores_invalid_generic_sqlite_single_file(tmp_path: Path) -> None:
    for index, name in enumerate(["predictions.sqlite", "answers.sqlite3", "results.db"]):
        output_dir = tmp_path / f"output-{index}"
        output_dir.mkdir(parents=True)
        artifact = output_dir / name
        artifact.write_bytes(b"not a sqlite database")

        assert find_submission_file(output_dir) is None, name


def test_find_submission_file_ignores_sqlite_single_file_without_data_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    artifact = output_dir / "predictions.sqlite"
    with sqlite3.connect(artifact) as conn:
        conn.execute("CREATE TABLE predictions (id INTEGER, target REAL)")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_finds_generic_model_artifact_when_final_missing(tmp_path: Path) -> None:
    names = [
        "adapter_model.safetensors",
        "booster.bst",
        "catboost.cbm",
        "checkpoint.safetensors",
        "coreml.mlmodel",
        "model.ckpt.index",
        "model.pmml",
        "model.skops",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "weights.gguf",
        "xgboost.ubj",
        "xgboost.xgb",
    ]
    for index, name in enumerate(names):
        output_dir = tmp_path / f"output-{index}"
        nested = output_dir / "nested"
        nested.mkdir(parents=True)
        artifact = nested / name
        artifact.write_bytes(b"model-artifact-submission")
        if name.endswith(".ckpt.index"):
            (nested / "model.ckpt.data-00000-of-00001").write_bytes(b"checkpoint-shard")

        assert find_submission_file(output_dir) == artifact, name


def test_find_submission_file_ignores_durable_kernel_state_model_cache(tmp_path: Path) -> None:
    output_dir = tmp_path / "iter-1"
    launch_manifest = output_dir / "output" / "launch-signature" / "local_launch_manifest.json"
    launch_manifest.parent.mkdir(parents=True)
    launch_manifest.write_text('{"signature": "abc"}\n', encoding="utf-8")
    cached_model = (
        output_dir
        / "durable_kernel_state"
        / "shared"
        / "models"
        / "models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots"
        / "revision"
        / "model-00004-of-00005.safetensors"
    )
    cached_model.parent.mkdir(parents=True)
    cached_model.write_bytes(b"cached-model-shard")

    assert find_submission_file(output_dir) is None

    submission = output_dir / "output" / "adapter_model.safetensors"
    submission.write_bytes(b"actual-submission")
    assert find_submission_file(output_dir) == submission


def test_find_output_file_ignores_durable_kernel_state_cache(tmp_path: Path) -> None:
    output_dir = tmp_path / "iter-1"
    cached = output_dir / "durable_kernel_state" / "shared" / "kernel_output" / "cache" / "metrics.json"
    cached.parent.mkdir(parents=True)
    cached.write_text('{"score": 0.1}\n', encoding="utf-8")

    assert find_output_file(output_dir, "metrics.json") is None


def test_find_submission_file_ignores_lonely_tensorflow_checkpoint_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "submission.ckpt.index"
    artifact.write_bytes(b"index-without-data")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_prefers_tensorflow_checkpoint_directory_over_member(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "checkpoint"
    artifact.mkdir(parents=True)
    index_file = artifact / "model.ckpt.index"
    index_file.write_bytes(b"index")
    (artifact / "model.ckpt.data-00000-of-00001").write_bytes(b"weights")
    (artifact / "checkpoint").write_text('model_checkpoint_path: "model.ckpt"\n', encoding="utf-8")
    os.utime(index_file, (2000, 2000))
    os.utime(artifact, (1000, 1000))

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_ignores_incomplete_tensorflow_checkpoint_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "checkpoint"
    artifact.mkdir(parents=True)
    (artifact / "model.ckpt.index").write_bytes(b"index")
    (artifact / "checkpoint").write_text('model_checkpoint_path: "model.ckpt"\n', encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_prefers_huggingface_model_directory_over_weight_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "model"
    artifact.mkdir(parents=True)
    weight_file = artifact / "model.safetensors"
    weight_file.write_bytes(b"weights")
    (artifact / "config.json").write_text('{"architectures": ["DemoModel"]}\n', encoding="utf-8")
    (artifact / "tokenizer_config.json").write_text('{"model_max_length": 512}\n', encoding="utf-8")
    os.utime(weight_file, (2000, 2000))
    os.utime(artifact, (1000, 1000))

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_ignores_incomplete_huggingface_model_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "model"
    artifact.mkdir(parents=True)
    (artifact / "config.json").write_text('{"architectures": ["DemoModel"]}\n', encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_prefers_mlflow_model_directory_over_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "mlflow_model"
    data_dir = artifact / "data"
    data_dir.mkdir(parents=True)
    payload = data_dir / "model.pmml"
    payload.write_bytes(b"pmml-model")
    (artifact / "MLmodel").write_text("flavors:\n  python_function:\n    data: data/model.pmml\n", encoding="utf-8")
    os.utime(payload, (2000, 2000))
    os.utime(artifact, (1000, 1000))

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_ignores_incomplete_mlflow_model_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "mlflow_model"
    artifact.mkdir(parents=True)
    (artifact / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_prefers_coreml_package_directory_over_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "model.mlpackage"
    data_dir = artifact / "Data" / "com.apple.CoreML"
    data_dir.mkdir(parents=True)
    payload = data_dir / "model.mlmodel"
    payload.write_bytes(b"coreml-model")
    (artifact / "Manifest.json").write_text('{"fileFormatVersion": "1.0.0"}\n', encoding="utf-8")
    os.utime(payload, (2000, 2000))
    os.utime(artifact, (1000, 1000))

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_ignores_incomplete_coreml_package_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "model.mlpackage"
    artifact.mkdir(parents=True)
    (artifact / "Manifest.json").write_text('{"fileFormatVersion": "1.0.0"}\n', encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_finds_coreml_compiled_package_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "model.mlmodelc"
    payload_dir = artifact / "com.apple.CoreML"
    payload_dir.mkdir(parents=True)
    (payload_dir / "model.mil").write_bytes(b"compiled-coreml")

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_ignores_empty_coreml_compiled_package_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    artifact = output_dir / "nested" / "model.mlmodelc"
    artifact.mkdir(parents=True)

    assert find_submission_file(output_dir) is None


def test_find_submission_file_finds_saved_model_directory_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "saved_model"
    artifact.mkdir(parents=True)
    (artifact / "saved_model.pb").write_bytes(b"saved-model")

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_does_not_treat_model_named_tabular_artifact_as_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "model.csv"
    artifact.write_text("id,target\n1,0\n", encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_finds_directory_asset_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "submission.zarr"
    artifact.mkdir(parents=True)
    (artifact / ".zarray").write_text("{}", encoding="utf-8")

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_finds_saved_model_submission_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    artifact = nested / "submission.savedmodel"
    artifact.mkdir(parents=True)
    (artifact / "saved_model.pbtxt").write_text("saved_model_schema", encoding="utf-8")

    assert find_submission_file(output_dir) == artifact


def test_find_submission_file_ignores_generic_intermediate_prediction_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    oof_predictions = output_dir / "oof_predictions.csv"
    train_predictions = output_dir / "train_predictions.csv"
    cv_results = output_dir / "cv_results.csv"
    train_predictions_archive = output_dir / "train_predictions.zip"
    train_predictions_array = output_dir / "train_predictions.npy"
    for path in (oof_predictions, train_predictions, cv_results, train_predictions_archive, train_predictions_array):
        if path.suffix == ".zip":
            path.write_bytes(b"archive")
        elif path.suffix == ".npy":
            path.write_bytes(b"array")
        else:
            path.write_text("id,target\n1,0.3\n", encoding="utf-8")

    assert find_submission_file(output_dir) is None


@pytest.mark.parametrize(
    "name",
    [
        "oof_predictions.parquet",
        "train_predictions.jsonl",
        "cv_results.tsv.gz",
        "feature_suspects.xlsx",
    ],
)
def test_find_submission_file_ignores_generic_intermediate_artifact_variants(tmp_path: Path, name: str) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact = output_dir / name
    if name.endswith(".parquet"):
        pd.DataFrame({"id": [1], "target": [0.3]}).to_parquet(artifact, index=False)
    elif name.endswith(".xlsx"):
        pd.DataFrame({"id": [1], "target": [0.3]}).to_excel(artifact, index=False)
    else:
        artifact.write_text("id,target\n1,0.3\n", encoding="utf-8")

    assert find_submission_file(output_dir) is None


def test_find_submission_file_prefers_generic_final_over_fold_intermediate(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fold = output_dir / "submission_model_fold1.csv"
    predictions = output_dir / "predictions.csv"
    fold.write_text("id,target\n1,0.1\n", encoding="utf-8")
    predictions.write_text("id,target\n1,0.3\n", encoding="utf-8")

    os.utime(fold, (2000, 2000))
    os.utime(predictions, (1000, 1000))

    assert find_submission_file(output_dir) == predictions


def test_pick_latest_artifact_filters_stale_files(tmp_path: Path) -> None:
    stale = tmp_path / "stale.csv"
    fresh = tmp_path / "fresh.csv"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")
    os.utime(stale, (1000, 1000))
    os.utime(fresh, (2000, 2000))

    assert pick_latest_artifact([stale, fresh], min_mtime=1500) == fresh
    assert pick_latest_artifact([stale], min_mtime=1500) is None


def test_find_output_file_finds_same_stem_tabular_suffix(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    stale = output_dir / "oof_predictions.csv"
    fresh = nested / "oof_predictions.parquet"
    stale.write_text("y,oof_pred\n0,0.1\n", encoding="utf-8")
    pd.DataFrame({"y": [1], "oof_pred": [0.9]}).to_parquet(fresh, index=False)
    os.utime(stale, (1000, 1000))
    os.utime(fresh, (2000, 2000))

    assert find_output_file(output_dir, "oof_predictions.csv") == fresh


def test_find_output_file_finds_compressed_same_stem_tabular_suffix(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    compressed = output_dir / "feature_suspects.tsv.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write("feature\treason\nx\tdrift\n")

    assert find_output_file(output_dir, "feature_suspects.csv") == compressed


@pytest.mark.parametrize(
    "name, marker_name",
    [
        ("predictions.zarr", ".zarray"),
        ("labels.ome.zarr", "zarr.json"),
        ("volumes.n5", "attributes.json"),
    ],
)
def test_find_output_file_finds_directory_array_output(
    tmp_path: Path,
    name: str,
    marker_name: str,
) -> None:
    output_dir = tmp_path / "output"
    artifact = output_dir / name
    artifact.mkdir(parents=True)
    (artifact / marker_name).write_text("{}", encoding="utf-8")

    assert find_output_file(output_dir, name) == artifact


def test_find_output_file_finds_model_directory_by_virtual_suffix(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    stale = output_dir / "stale_model"
    fresh = output_dir / "nested" / "model"
    stale.mkdir(parents=True)
    fresh.mkdir(parents=True)
    (stale / "config.json").write_text('{"architectures": ["Old"]}\n', encoding="utf-8")
    (stale / "model.safetensors").write_bytes(b"old")
    (fresh / "config.json").write_text('{"architectures": ["New"]}\n', encoding="utf-8")
    (fresh / "model.safetensors").write_bytes(b"fresh")
    os.utime(stale, (1000, 1000))
    os.utime(fresh, (2000, 2000))

    assert find_output_file(output_dir, "submission.hfmodel") == fresh


def test_resolve_local_kernel_artifact_file_finds_model_directory_by_virtual_suffix(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "local-iter-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    artifact = kernel_dir / "outputs" / "saved_model"
    artifact.mkdir(parents=True)
    (artifact / "saved_model.pb").write_bytes(b"saved-model")
    os.utime(artifact, (2000, 2000))

    resolved = resolve_local_kernel_artifact_file(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        started_at=1500,
        filename="submission.savedmodel",
    )

    assert resolved == artifact


def test_find_newest_existing_path_uses_size_and_path_tiebreakers(tmp_path: Path) -> None:
    smaller = tmp_path / "a.json"
    larger = tmp_path / "b.json"
    smaller.write_text("{}", encoding="utf-8")
    larger.write_text('{"value": 1}', encoding="utf-8")
    os.utime(smaller, (2000, 2000))
    os.utime(larger, (2000, 2000))

    assert find_newest_existing_path([smaller, larger]) == larger


def test_resolve_local_kernel_artifacts_finds_fresh_nested_outputs(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "local-iter-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    nested_outputs = kernel_dir.parent / "outputs"
    nested_outputs.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    submission = nested_outputs / "submission.csv"
    metrics = nested_outputs / "metrics.json"
    submission.write_text("id,target\n1,0.2\n", encoding="utf-8")
    metrics.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    os.utime(submission, (2000, 2000))
    os.utime(metrics, (2000, 2000))

    resolved_submission, resolved_metrics = resolve_local_kernel_artifacts(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        started_at=1500,
    )

    assert resolved_submission == submission
    assert resolved_metrics == metrics


def test_resolve_local_kernel_artifact_file_and_copy(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "local-iter-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    artifact_dir = kernel_dir / "outputs"
    artifact_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source = artifact_dir / "cv_results.json"
    source.write_text("{}", encoding="utf-8")
    os.utime(source, (2000, 2000))

    resolved = resolve_local_kernel_artifact_file(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        started_at=1500,
        filename="cv_results.json",
    )
    destination = output_dir / "cv_results.json"

    assert resolved == source
    assert copy_artifact_if_needed(source=source, destination=destination) == destination
    assert destination.read_text(encoding="utf-8") == "{}"


def test_copy_local_kernel_primary_artifacts_copies_submission_and_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission_model_fold1.csv"
    metrics = source_dir / "metrics_nested.json"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    metrics.write_text('{"score": 0.1}', encoding="utf-8")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=metrics,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission_model_fold1.csv"
    assert metrics_dst == output_dir / "metrics.json"
    assert submission_dst.read_text(encoding="utf-8") == "id,target\n1,0.1\n"
    assert metrics_dst.read_text(encoding="utf-8") == '{"score": 0.1}'


@pytest.mark.parametrize(
    "name, marker_name",
    [
        ("submission.zarr", ".zarray"),
        ("submission.ome.zarr", "zarr.json"),
        ("submission.n5", "attributes.json"),
    ],
)
def test_copy_local_kernel_primary_artifacts_copies_directory_submission(
    tmp_path: Path,
    name: str,
    marker_name: str,
) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    submission = source_dir / name
    submission.mkdir(parents=True)
    (submission / marker_name).write_text("{}", encoding="utf-8")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / name
    assert metrics_dst is None
    assert (submission_dst / marker_name).read_text(encoding="utf-8") == "{}"


def test_copy_local_kernel_primary_artifacts_copies_shapefile_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission.shp"
    submission.write_bytes(b"shape")
    (source_dir / "submission.dbf").write_bytes(b"attributes")
    (source_dir / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (source_dir / "submission.qix").write_bytes(b"qix")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.shp"
    assert metrics_dst is None
    assert submission_dst.read_bytes() == b"shape"
    assert (output_dir / "submission.dbf").read_bytes() == b"attributes"
    assert (output_dir / "submission.prj").read_text(encoding="utf-8") == "EPSG:4326\n"
    assert (output_dir / "submission.qix").read_bytes() == b"qix"


def test_copy_local_kernel_primary_artifacts_copies_mapinfo_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission.tab"
    submission.write_text("!table\n!version 300\n", encoding="utf-8")
    (source_dir / "submission.dat").write_bytes(b"data")
    (source_dir / "submission.id").write_bytes(b"ids")
    (source_dir / "submission.map").write_bytes(b"map")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.tab"
    assert metrics_dst is None
    assert submission_dst.read_text(encoding="utf-8") == "!table\n!version 300\n"
    assert (output_dir / "submission.dat").read_bytes() == b"data"
    assert (output_dir / "submission.id").read_bytes() == b"ids"
    assert (output_dir / "submission.map").read_bytes() == b"map"


def test_copy_local_kernel_primary_artifacts_copies_mapinfo_interchange_sidecar(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission.mif"
    submission.write_text("Version 300\nColumns 1\n  Name Char(20)\nData\n", encoding="utf-8")
    (source_dir / "submission.mid").write_text('"parcel-a"\n', encoding="utf-8")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.mif"
    assert metrics_dst is None
    assert submission_dst.read_text(encoding="utf-8").startswith("Version 300")
    assert (output_dir / "submission.mid").read_text(encoding="utf-8") == '"parcel-a"\n'


def test_copy_local_kernel_primary_artifacts_copies_georeferenced_raster_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission.tif"
    submission.write_bytes(b"raster")
    (source_dir / "submission.tfw").write_text("1\n0\n0\n-1\n100\n200\n", encoding="ascii")
    (source_dir / "submission.tif.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.tif"
    assert metrics_dst is None
    assert submission_dst.read_bytes() == b"raster"
    assert (output_dir / "submission.tfw").read_text(encoding="ascii") == "1\n0\n0\n-1\n100\n200\n"
    assert (output_dir / "submission.tif.aux.xml").read_text(encoding="utf-8") == "<PAMDataset />\n"


def test_copy_local_kernel_primary_artifacts_copies_vrt_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "rasters").mkdir(parents=True)
    submission = source_dir / "submission.vrt"
    submission.write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource><SourceFilename relativeToVRT="1">rasters/source.tif</SourceFilename></SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )
    (source_dir / "rasters" / "source.tif").write_bytes(b"raster")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.vrt"
    assert metrics_dst is None
    assert "rasters/source.tif" in submission_dst.read_text(encoding="utf-8")
    assert (output_dir / "rasters" / "source.tif").read_bytes() == b"raster"


def test_copy_local_kernel_primary_artifacts_copies_las_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission.laz"
    submission.write_bytes(b"laz")
    (source_dir / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (source_dir / "submission.lasx").write_bytes(b"index")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.laz"
    assert metrics_dst is None
    assert submission_dst.read_bytes() == b"laz"
    assert (output_dir / "submission.prj").read_text(encoding="utf-8") == "EPSG:4326\n"
    assert (output_dir / "submission.lasx").read_bytes() == b"index"


def test_copy_local_kernel_primary_artifacts_copies_kml_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "icons").mkdir(parents=True)
    submission = source_dir / "submission.kml"
    submission.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document>
            <Style id="pin">
              <IconStyle><Icon><href>icons/pin.png</href></Icon></IconStyle>
            </Style>
            <Placemark><styleUrl>#pin</styleUrl><name>A</name></Placemark>
          </Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (source_dir / "icons" / "pin.png").write_bytes(b"pin")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.kml"
    assert metrics_dst is None
    assert (output_dir / "icons" / "pin.png").read_bytes() == b"pin"


def test_copy_local_kernel_primary_artifacts_preserves_nested_kml_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "layers").mkdir(parents=True)
    (source_dir / "icons").mkdir()
    submission = source_dir / "layers" / "submission.kml"
    submission.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>../icons/pin.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (source_dir / "icons" / "pin.png").write_bytes(b"pin")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "layers" / "submission.kml"
    assert metrics_dst is None
    assert (output_dir / "icons" / "pin.png").read_bytes() == b"pin"


def test_copy_local_kernel_primary_artifacts_copies_envi_header_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission.hdr"
    submission.write_text("ENVI\nsamples = 2\nlines = 2\nbands = 1\n", encoding="utf-8")
    (source_dir / "submission.dat").write_bytes(b"raster")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.hdr"
    assert metrics_dst is None
    assert submission_dst.read_text(encoding="utf-8").startswith("ENVI")
    assert (output_dir / "submission.dat").read_bytes() == b"raster"


def test_copy_local_kernel_primary_artifacts_copies_metaimage_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "raw").mkdir(parents=True)
    submission = source_dir / "submission.mhd"
    submission.write_text("ObjectType = Image\nElementDataFile = raw/volume.raw\n", encoding="utf-8")
    (source_dir / "raw" / "volume.raw").write_bytes(b"voxels")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.mhd"
    assert metrics_dst is None
    assert submission_dst.read_text(encoding="utf-8") == "ObjectType = Image\nElementDataFile = raw/volume.raw\n"
    assert (output_dir / "raw" / "volume.raw").read_bytes() == b"voxels"


def test_copy_local_kernel_primary_artifacts_copies_detached_nrrd_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "raw").mkdir(parents=True)
    submission = source_dir / "submission.nhdr"
    submission.write_text("NRRD0005\nsizes: 4 5 6\ndata file: raw/volume.raw\n", encoding="utf-8")
    (source_dir / "raw" / "volume.raw").write_bytes(b"voxels")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.nhdr"
    assert metrics_dst is None
    assert submission_dst.read_text(encoding="utf-8") == "NRRD0005\nsizes: 4 5 6\ndata file: raw/volume.raw\n"
    assert (output_dir / "raw" / "volume.raw").read_bytes() == b"voxels"


def test_copy_local_kernel_primary_artifacts_copies_analyze_pair_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "submission.hdr"
    submission.write_bytes(b"header")
    (source_dir / "submission.img").write_bytes(b"volume")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.hdr"
    assert metrics_dst is None
    assert submission_dst.read_bytes() == b"header"
    assert (output_dir / "submission.img").read_bytes() == b"volume"


def test_copy_local_kernel_primary_artifacts_copies_obj_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "materials" / "textures").mkdir(parents=True)
    submission = source_dir / "submission.obj"
    submission.write_text("mtllib materials/model.mtl\nv 0 0 0\n", encoding="utf-8")
    (source_dir / "materials" / "model.mtl").write_text(
        "newmtl surface\nmap_Kd textures/diffuse.png\n",
        encoding="utf-8",
    )
    (source_dir / "materials" / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.obj"
    assert metrics_dst is None
    assert (output_dir / "materials" / "model.mtl").read_text(encoding="utf-8") == (
        "newmtl surface\nmap_Kd textures/diffuse.png\n"
    )
    assert (output_dir / "materials" / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_copies_ply_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "textures").mkdir(parents=True)
    submission = source_dir / "submission.ply"
    submission.write_text(
        "ply\nformat ascii 1.0\ncomment TextureFile textures/diffuse.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.ply"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_copies_dae_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "textures").mkdir(parents=True)
    submission = source_dir / "submission.dae"
    submission.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>textures/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.dae"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_preserves_nested_dae_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "meshes").mkdir(parents=True)
    (source_dir / "textures").mkdir()
    submission = source_dir / "meshes" / "submission.dae"
    submission.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>../textures/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "meshes" / "submission.dae"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_copies_x3d_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "textures").mkdir(parents=True)
    submission = source_dir / "submission.x3d"
    submission.write_text(
        """
        <X3D>
          <Scene>
            <Shape>
              <Appearance><ImageTexture url='"textures/diffuse.png"'/></Appearance>
            </Shape>
          </Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.x3d"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_preserves_nested_x3d_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "scenes").mkdir(parents=True)
    (source_dir / "textures").mkdir()
    submission = source_dir / "scenes" / "submission.x3d"
    submission.write_text(
        """
        <X3D>
          <Scene>
            <Shape>
              <Appearance><ImageTexture url='"../textures/diffuse.png"'/></Appearance>
            </Shape>
          </Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "scenes" / "submission.x3d"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_copies_gltf_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "buffers").mkdir(parents=True)
    (source_dir / "textures").mkdir()
    submission = source_dir / "submission.gltf"
    submission.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "buffers/scene.bin"}],
                "images": [
                    {"uri": "textures/diffuse.png"},
                    {"uri": "data:image/png;base64,AAAA"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "buffers" / "scene.bin").write_bytes(b"buffer")
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.gltf"
    assert metrics_dst is None
    assert (output_dir / "buffers" / "scene.bin").read_bytes() == b"buffer"
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_preserves_nested_gltf_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "scenes").mkdir(parents=True)
    (source_dir / "textures").mkdir()
    submission = source_dir / "scenes" / "submission.gltf"
    submission.write_text(
        json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "../textures/diffuse.png"}]}),
        encoding="utf-8",
    )
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "scenes" / "submission.gltf"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_copies_usd_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "textures").mkdir(parents=True)
    submission = source_dir / "submission.usda"
    submission.write_text("#usda 1.0\nasset inputs:file = @textures/diffuse.png@\n", encoding="utf-8")
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "submission.usda"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_preserves_nested_usd_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    (source_dir / "scenes").mkdir(parents=True)
    (source_dir / "textures").mkdir()
    submission = source_dir / "scenes" / "submission.usda"
    submission.write_text("#usda 1.0\nasset inputs:file = @../textures/diffuse.png@\n", encoding="utf-8")
    (source_dir / "textures" / "diffuse.png").write_bytes(b"texture")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "scenes" / "submission.usda"
    assert metrics_dst is None
    assert (output_dir / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_copy_local_kernel_primary_artifacts_copies_model_index_shards(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "model.safetensors.index.json"
    submission.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (source_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "model.safetensors.index.json"
    assert metrics_dst is None
    assert (output_dir / "model-00001-of-00002.safetensors").read_bytes() == b"shard-1"
    assert (output_dir / "model-00002-of-00002.safetensors").read_bytes() == b"shard-2"


def test_copy_local_kernel_primary_artifacts_copies_tensorflow_checkpoint_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "model.ckpt.index"
    submission.write_bytes(b"index")
    (source_dir / "model.ckpt.data-00000-of-00001").write_bytes(b"weights")
    (source_dir / "checkpoint").write_text('model_checkpoint_path: "model.ckpt"\n', encoding="utf-8")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "model.ckpt.index"
    assert metrics_dst is None
    assert (output_dir / "model.ckpt.data-00000-of-00001").read_bytes() == b"weights"
    assert (output_dir / "checkpoint").read_text(encoding="utf-8") == 'model_checkpoint_path: "model.ckpt"\n'


def test_copy_local_kernel_primary_artifacts_copies_model_artifact_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    output_dir.mkdir()
    source_dir.mkdir()
    submission = source_dir / "adapter_model.safetensors"
    submission.write_bytes(b"weights")
    (source_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}\n', encoding="utf-8")
    (source_dir / "tokenizer_config.json").write_text('{"model_max_length": 512}\n', encoding="utf-8")

    submission_dst, metrics_dst = copy_local_kernel_primary_artifacts(
        submission_path=submission,
        metrics_path=None,
        output_dir=output_dir,
    )

    assert submission_dst == output_dir / "adapter_model.safetensors"
    assert metrics_dst is None
    assert submission_dst.read_bytes() == b"weights"
    assert (output_dir / "adapter_config.json").read_text(encoding="utf-8") == '{"peft_type": "LORA"}\n'
    assert (output_dir / "tokenizer_config.json").read_text(encoding="utf-8") == '{"model_max_length": 512}\n'


def test_copy_optional_local_kernel_artifacts_copies_configured_outputs(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "local-iter-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    artifact_dir = kernel_dir.parent / "outputs"
    artifact_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    fresh = artifact_dir / "cv_results.json"
    stale = artifact_dir / "metrics_summary.json"
    fresh.write_text('{"fresh": true}', encoding="utf-8")
    stale.write_text('{"stale": true}', encoding="utf-8")
    os.utime(fresh, (2000, 2000))
    os.utime(stale, (1000, 1000))

    copied = copy_optional_local_kernel_artifacts(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        started_at=1500,
        filenames=("cv_results.json", "metrics_summary.json", "missing.json"),
    )

    assert copied == [output_dir / "cv_results.json"]
    assert (output_dir / "cv_results.json").read_text(encoding="utf-8") == '{"fresh": true}'
