from __future__ import annotations

import gzip
import io
import json
import os
import stat
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import py7zr
import pytest
import zstandard as zstd

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission_artifacts import store_submission_artifact
from kagglebot.submission_format import SubmissionFormatHint
from kagglebot.submission_service import SubmissionConfig, SubmissionService


def _build_service(tmp_path: Path) -> SubmissionService:
    config = SubmissionConfig(
        slug="demo",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "missing_sample_submission.csv",
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=False,
    )
    return SubmissionService(config)


@pytest.mark.parametrize(
    ("name", "archive_kind"),
    [
        ("submission.zip", "zip"),
        ("submission.tar", "tar"),
        ("submission.tar.gz", "tar"),
        ("submission.tgz", "tar"),
        ("submission.tar.xz", "tar"),
        ("submission.tar.zst", "tar"),
        ("submission.tzst", "tar"),
        ("submission.7z", "external"),
        ("submission.rar", "external"),
    ],
)
def test_submission_service_classifies_archive_suffixes_by_shared_artifact_suffix(
    name: str,
    archive_kind: str,
) -> None:
    path = Path(name)

    assert SubmissionService._is_zip_submission(path) is (archive_kind == "zip")
    assert SubmissionService._is_tar_submission(path) is (archive_kind == "tar")
    assert SubmissionService._is_external_archive_submission(path) is (archive_kind == "external")


class _Fake7zMember:
    def __init__(
        self,
        filename: str,
        *,
        is_directory: bool = False,
        is_file: bool = True,
        is_symlink: bool = False,
        uncompressed: int = 1,
    ) -> None:
        self.filename = filename
        self.is_directory = is_directory
        self.is_file = is_file
        self.is_symlink = is_symlink
        self.uncompressed = uncompressed


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


class _FakeRarMember:
    def __init__(
        self,
        filename: str,
        *,
        is_dir: bool = False,
        is_file: bool = True,
        is_symlink: bool = False,
        needs_password: bool = False,
        file_size: int = 1,
    ) -> None:
        self.filename = filename
        self._is_dir = is_dir
        self._is_file = is_file
        self._is_symlink = is_symlink
        self._needs_password = needs_password
        self.file_size = file_size

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


def test_validate_and_prepare_submission_accepts_zip_without_sample_csv(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr("123.tif", b"dummy")

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared == submission_path


def test_validate_and_prepare_submission_rejects_invalid_zip(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    submission_path.write_bytes(b"not-a-zip")

    with pytest.raises(SubmissionValidationError, match="submission zip is invalid"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_zip_with_only_empty_files(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr("mask-a.tif", b"")
        archive.writestr("nested/mask-b.tif", b"")

    with pytest.raises(SubmissionValidationError, match="submission zip has no non-empty files"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_zip_path_traversal(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr("../evil.txt", b"nope")

    with pytest.raises(SubmissionValidationError, match="unsafe path traversal"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_existing_zip_duplicate_member_names(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr("mask.tif", b"mask-a")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("mask.tif", b"mask-b")

    with pytest.raises(SubmissionValidationError, match="duplicate archive member name: mask\\.tif"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_zip_symlink_member(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    link_info = zipfile.ZipInfo("latest-mask.tif")
    link_info.create_system = 3
    link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr(link_info, "mask.tif")

    with pytest.raises(SubmissionValidationError, match="unsupported symlink member: latest-mask\\.tif"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_accepts_tar_gzip_code_archive(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "cg").mkdir(parents=True)
    (bundle_dir / "main.py").write_text("def agent(obs):\n    return [0]\n", encoding="utf-8")
    (bundle_dir / "deck.csv").write_text("card_id\n1\n", encoding="utf-8")
    (bundle_dir / "cg" / "api.py").write_text("", encoding="utf-8")
    submission_path = tmp_path / "submission.tar.gz"
    with tarfile.open(submission_path, "w:gz") as archive:
        archive.add(bundle_dir / "main.py", arcname="main.py")
        archive.add(bundle_dir / "deck.csv", arcname="deck.csv")
        archive.add(bundle_dir / "cg" / "api.py", arcname="cg/api.py")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_validate_and_prepare_submission_accepts_generic_tar_gzip_without_code_contract_hint(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    payload = tmp_path / "payload.txt"
    payload.write_text("not a code submission\n", encoding="utf-8")
    submission_path = tmp_path / "submission.tgz"
    with tarfile.open(submission_path, "w:gz") as archive:
        archive.add(payload, arcname="payload.txt")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_validate_and_prepare_submission_rejects_tar_with_only_empty_files(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.tar` containing prediction files.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.tar"
    with tarfile.open(submission_path, "w") as archive:
        for member_name in ("mask-a.tif", "nested/mask-b.tif"):
            info = tarfile.TarInfo(member_name)
            info.size = 0
            archive.addfile(info, io.BytesIO(b""))

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    with pytest.raises(SubmissionValidationError, match="submission tar archive has no non-empty files"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_tar_gzip_when_context_requires_code_contract(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a tar.gz archive with top-level deck.csv, main.py, and cg/ support files.\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("not a code submission\n", encoding="utf-8")
    submission_path = tmp_path / "submission.tgz"
    with tarfile.open(submission_path, "w:gz") as archive:
        archive.add(payload, arcname="payload.txt")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    with pytest.raises(SubmissionValidationError, match="missing required top-level files"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_tar_path_traversal(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.tar.gz"
    payload = b"nope"
    with tarfile.open(submission_path, "w:gz") as archive:
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(SubmissionValidationError, match="unsafe path traversal"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_existing_tar_duplicate_member_names(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.tar"
    with tarfile.open(submission_path, "w") as archive:
        for payload in (b"mask-a", b"mask-b"):
            info = tarfile.TarInfo("mask.tif")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(SubmissionValidationError, match="duplicate archive member name: mask\\.tif"):
        service.validate_and_prepare_submission(submission_path)


@pytest.mark.parametrize(
    ("link_type", "member_name"),
    [
        (tarfile.SYMTYPE, "latest-mask.tif"),
        (tarfile.LNKTYPE, "copied-mask.tif"),
    ],
)
def test_validate_and_prepare_submission_rejects_existing_tar_link_members(
    tmp_path: Path,
    link_type: bytes,
    member_name: str,
) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.tar"
    with tarfile.open(submission_path, "w") as archive:
        payload = b"mask"
        target = tarfile.TarInfo("mask.tif")
        target.size = len(payload)
        archive.addfile(target, io.BytesIO(payload))
        link = tarfile.TarInfo(member_name)
        link.type = link_type
        link.linkname = "mask.tif"
        archive.addfile(link)

    with pytest.raises(SubmissionValidationError, match=f"unsupported link member: {member_name}"):
        service.validate_and_prepare_submission(submission_path)


@pytest.mark.parametrize(
    ("suffix", "mode"),
    [
        (".tar", "w"),
        (".tar.gz", "w:gz"),
        (".tgz", "w:gz"),
        (".tar.bz2", "w:bz2"),
        (".tbz2", "w:bz2"),
        (".tar.xz", "w:xz"),
        (".txz", "w:xz"),
        (".tar.zst", "zst"),
        (".tzst", "zst"),
    ],
)
def test_validate_and_prepare_submission_accepts_generic_tar_when_required(
    tmp_path: Path,
    suffix: str,
    mode: str,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` containing prediction files.\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("predictions\n", encoding="utf-8")
    submission_path = tmp_path / f"submission{suffix}"
    if mode == "zst":
        with submission_path.open("wb") as raw:
            with zstd.ZstdCompressor(level=9).stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    archive.add(payload, arcname="payload.txt")
    else:
        with tarfile.open(submission_path, mode) as archive:
            archive.add(payload, arcname="payload.txt")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_validate_and_prepare_submission_accepts_expected_7z_archive(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a single `submission.7z` archive.\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("predictions\n", encoding="utf-8")
    submission_path = tmp_path / "submission.7z"
    with py7zr.SevenZipFile(submission_path, "w") as archive:
        archive.write(payload, "payload.txt")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_validate_and_prepare_submission_rejects_invalid_7z_archive(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.7z"
    submission_path.write_bytes(b"not-a-7z")

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="submission 7z archive is invalid"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_7z_path_traversal(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("nope\n", encoding="utf-8")
    submission_path = tmp_path / "submission.7z"
    with py7zr.SevenZipFile(submission_path, "w") as archive:
        archive.write(payload, "../evil.txt")

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="unsafe path traversal"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_7z_duplicate_member_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission.7z"
    submission_path.write_bytes(b"7z payload")
    _Fake7zFile.members_by_name = {"submission.7z": [_Fake7zMember("mask.tif"), _Fake7zMember("mask.tif")]}
    _Fake7zFile.password_required_by_name = {}
    monkeypatch.setattr("kagglebot.submission_service.py7zr.SevenZipFile", _Fake7zFile)

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="duplicate archive member name: mask\\.tif"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_7z_with_only_empty_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission.7z"
    submission_path.write_bytes(b"7z payload")
    _Fake7zFile.members_by_name = {
        "submission.7z": [
            _Fake7zMember("mask-a.tif", uncompressed=0),
            _Fake7zMember("nested/mask-b.tif", uncompressed=0),
        ]
    }
    _Fake7zFile.password_required_by_name = {}
    monkeypatch.setattr("kagglebot.submission_service.py7zr.SevenZipFile", _Fake7zFile)

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="submission 7z archive has no non-empty files"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_7z_symlink_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission.7z"
    submission_path.write_bytes(b"7z payload")
    _Fake7zFile.members_by_name = {"submission.7z": [_Fake7zMember("latest-mask.tif", is_file=True, is_symlink=True)]}
    _Fake7zFile.password_required_by_name = {}
    monkeypatch.setattr("kagglebot.submission_service.py7zr.SevenZipFile", _Fake7zFile)

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="unsupported member type: latest-mask\\.tif"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_password_protected_7z_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission.7z"
    submission_path.write_bytes(b"7z payload")
    _Fake7zFile.members_by_name = {"submission.7z": [_Fake7zMember("mask.tif")]}
    _Fake7zFile.password_required_by_name = {"submission.7z": True}
    monkeypatch.setattr("kagglebot.submission_service.py7zr.SevenZipFile", _Fake7zFile)

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="7z archive requires a password"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_accepts_expected_rar_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a single `submission.rar` archive.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.rar"
    submission_path.write_bytes(b"rar payload")
    _FakeRarFile.members_by_name = {"submission.rar": [_FakeRarMember("payload.txt")]}
    monkeypatch.setattr("kagglebot.submission_service.rarfile.RarFile", _FakeRarFile)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_validate_and_prepare_submission_rejects_invalid_rar_archive(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.rar"
    submission_path.write_bytes(b"not-a-rar")

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="submission rar archive is invalid"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_rar_duplicate_member_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission.rar"
    submission_path.write_bytes(b"rar payload")
    _FakeRarFile.members_by_name = {"submission.rar": [_FakeRarMember("mask.tif"), _FakeRarMember("mask.tif")]}
    monkeypatch.setattr("kagglebot.submission_service.rarfile.RarFile", _FakeRarFile)

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="duplicate archive member name: mask\\.tif"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_rar_with_only_empty_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_path = tmp_path / "submission.rar"
    submission_path.write_bytes(b"rar payload")
    _FakeRarFile.members_by_name = {
        "submission.rar": [
            _FakeRarMember("mask-a.tif", file_size=0),
            _FakeRarMember("nested/mask-b.tif", file_size=0),
        ]
    }
    monkeypatch.setattr("kagglebot.submission_service.rarfile.RarFile", _FakeRarFile)

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="submission rar archive has no non-empty files"):
        service.validate_and_prepare_submission(submission_path)


@pytest.mark.parametrize(
    "member",
    [
        _FakeRarMember("secret.txt", needs_password=True),
        _FakeRarMember("latest-mask.tif", is_symlink=True),
    ],
)
def test_validate_and_prepare_submission_rejects_rar_unsupported_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member: _FakeRarMember,
) -> None:
    submission_path = tmp_path / "submission.rar"
    submission_path.write_bytes(b"rar payload")
    _FakeRarFile.members_by_name = {"submission.rar": [member]}
    monkeypatch.setattr("kagglebot.submission_service.rarfile.RarFile", _FakeRarFile)

    service = _build_service(tmp_path)

    with pytest.raises(SubmissionValidationError, match="unsupported member type"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_converts_tabular_to_zip_when_required(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a ZIP file.\n",
        encoding="utf-8",
    )

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["submission.csv"]


def test_validate_and_prepare_submission_converts_compressed_tabular_to_clean_zip_name(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a ZIP file.\n",
        encoding="utf-8",
    )

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.2\n2,0.8\n")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["submission.csv.gz"]


def test_validate_and_prepare_submission_builds_multi_file_zip_from_manifest(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a ZIP file containing one .tif mask per sample.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    nested = bundle_dir / "nested"
    nested.mkdir()
    (nested / "b.tif").write_bytes(b"mask-b")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "staging_dir": str(bundle_dir.relative_to(tmp_path)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(manifest_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif", "nested/b.tif"]


def test_validate_and_prepare_submission_accepts_run_specific_manifest_artifact(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a ZIP file containing one .tif mask per sample.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    manifest_path = tmp_path / "run-123_submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "staging_dir": str(bundle_dir.relative_to(tmp_path)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(manifest_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif"]


def test_validate_and_prepare_submission_rejects_manifest_without_submission_references(tmp_path: Path) -> None:
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "notes": "metadata only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    with pytest.raises(
        SubmissionValidationError,
        match="manifest submission artifact has no usable submission reference",
    ):
        service.validate_and_prepare_submission(manifest_path)


def test_validate_and_prepare_submission_uses_stored_manifest_primary_artifact_bundle(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a ZIP file containing one .tif mask per sample.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    source_manifest = output_dir / "submission_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "staging_dir": "bundle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stored_manifest = store_submission_artifact(
        source=source_manifest,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(stored_manifest)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif"]


def test_validate_and_prepare_submission_keeps_stored_manifest_member_layout(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a ZIP file containing mask files.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    left = output_dir / "fold1" / "mask.tif"
    right = output_dir / "fold2" / "mask.tif"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_bytes(b"mask-left")
    right.write_bytes(b"mask-right")
    source_manifest = output_dir / "submission_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": ["fold1/mask.tif", "fold2/mask.tif"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stored_manifest = store_submission_artifact(
        source=source_manifest,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(stored_manifest)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        payloads = {name: archive.read(name) for name in members}
    assert members == ["fold1/mask.tif", "fold2/mask.tif"]
    assert payloads == {
        "fold1/mask.tif": b"mask-left",
        "fold2/mask.tif": b"mask-right",
    }


def test_validate_and_prepare_submission_infers_bundle_zip_from_manifest_staging(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif"]


def test_validate_and_prepare_submission_rejects_manifest_staging_path_traversal(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside_bundle"
    outside.mkdir()
    (outside / "secret.tif").write_bytes(b"secret")
    manifest_path = output_dir / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifactClass": "bundle",
                "stagingDir": "../outside_bundle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    with pytest.raises(SubmissionValidationError, match="unsafe path traversal in manifest staging path"):
        service.validate_and_prepare_submission(manifest_path)


def test_validate_and_prepare_submission_builds_bundle_zip_from_manifest_members_only(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (bundle_dir / "b.tif").write_bytes(b"mask-b")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": {
                    "a.tif": "bundle/a.tif",
                    "b.tif": {"sourcePath": "bundle/b.tif"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif", "b.tif"]


def test_validate_and_prepare_submission_builds_bundle_zip_from_source_to_archive_mapping(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": {
                    "bundle/a.tif": "masks/a.tif",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        payload = archive.read("masks/a.tif")
    assert members == ["masks/a.tif"]
    assert payload == b"mask-a"


def test_validate_and_prepare_submission_builds_bundle_zip_from_single_member_object(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": {
                    "sourcePath": "bundle/a.tif",
                    "targetPath": "masks/a.tif",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        payload = archive.read("masks/a.tif")
    assert members == ["masks/a.tif"]
    assert payload == b"mask-a"


def test_validate_and_prepare_submission_builds_bundle_zip_from_member_path_objects(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": [
                    {
                        "sourcePath": {"path": "bundle/a.tif"},
                        "targetPath": {"path": "masks/a.tif"},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        payload = archive.read("masks/a.tif")
    assert members == ["masks/a.tif"]
    assert payload == b"mask-a"


def test_validate_and_prepare_submission_builds_bundle_zip_from_manifest_glob_members(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (bundle_dir / "b.tif").write_bytes(b"mask-b")
    (bundle_dir / "notes.txt").write_text("ignore\n", encoding="utf-8")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": ["bundle/*.tif"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif", "b.tif"]


def test_validate_and_prepare_submission_builds_bundle_zip_from_single_string_glob(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (bundle_dir / "b.tif").write_bytes(b"mask-b")
    (bundle_dir / "notes.txt").write_text("ignore\n", encoding="utf-8")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": "bundle/*.tif",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif", "b.tif"]


def test_validate_and_prepare_submission_builds_bundle_zip_from_manifest_directory_member(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    nested = bundle_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "empty_group").mkdir()
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (nested / "b.tif").write_bytes(b"mask-b")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": ["bundle"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        infos = archive.infolist()
        members = sorted(info.filename for info in infos if not info.is_dir())
        dirs = sorted(info.filename for info in infos if info.is_dir())
    assert members == ["a.tif", "nested/b.tif"]
    assert "nested/empty_group/" in dirs


def test_validate_and_prepare_submission_deduplicates_manifest_members(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (bundle_dir / "b.tif").write_bytes(b"mask-b")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": ["bundle/*.tif", "bundle/a.tif"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["a.tif", "b.tif"]


def test_validate_and_prepare_submission_uses_manifest_archive_paths(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (bundle_dir / "b.tif").write_bytes(b"mask-b")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": {
                    "masks/a.tif": "bundle/a.tif",
                    "nested/b.tif": {"sourcePath": "bundle/b.tif"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["masks/a.tif", "nested/b.tif"]


def test_validate_and_prepare_submission_allows_same_source_with_distinct_archive_paths(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": [
                    {"sourcePath": "bundle/a.tif", "targetPath": "masks/a.tif"},
                    {"sourcePath": "bundle/a.tif", "targetPath": "backup/a.tif"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["backup/a.tif", "masks/a.tif"]


def test_validate_and_prepare_submission_uses_manifest_archive_directory_for_globs(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (bundle_dir / "b.tif").write_bytes(b"mask-b")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": [
                    {
                        "sourcePath": "bundle/*.tif",
                        "targetPath": "masks/",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["masks/a.tif", "masks/b.tif"]


def test_validate_and_prepare_submission_preserves_recursive_glob_archive_layout(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    left = bundle_dir / "fold1" / "mask.tif"
    right = bundle_dir / "fold2" / "mask.tif"
    left.parent.mkdir(parents=True, exist_ok=True)
    right.parent.mkdir(parents=True, exist_ok=True)
    left.write_bytes(b"mask-left")
    right.write_bytes(b"mask-right")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": [
                    {
                        "sourcePath": "bundle/**/*.tif",
                        "targetPath": "masks/",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    prepared = service.validate_and_prepare_submission(manifest_path)

    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        payloads = {name: archive.read(name) for name in members}
    assert members == ["masks/fold1/mask.tif", "masks/fold2/mask.tif"]
    assert payloads == {
        "masks/fold1/mask.tif": b"mask-left",
        "masks/fold2/mask.tif": b"mask-right",
    }


def test_validate_and_prepare_submission_rejects_duplicate_archive_member_names(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    (bundle_dir / "b.tif").write_bytes(b"mask-b")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": [
                    {"sourcePath": "bundle/a.tif", "targetPath": "mask.tif"},
                    {"sourcePath": "bundle/b.tif", "targetPath": "mask.tif"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    with pytest.raises(SubmissionValidationError, match="duplicate archive member name: mask\\.tif"):
        service.validate_and_prepare_submission(manifest_path)


def test_validate_and_prepare_submission_rejects_manifest_archive_path_traversal(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": [
                    {"sourcePath": "bundle/a.tif", "targetPath": "../evil.tif"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    with pytest.raises(SubmissionValidationError, match="unsafe path traversal"):
        service.validate_and_prepare_submission(manifest_path)


def test_validate_and_prepare_submission_rejects_manifest_source_path_traversal(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "secret.tif").write_bytes(b"secret")
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stagingDir": str(bundle_dir.relative_to(tmp_path)),
                "files": [
                    {"sourcePath": "../secret.tif", "targetPath": "secret.tif"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = _build_service(tmp_path)
    with pytest.raises(SubmissionValidationError, match="unsafe path traversal in manifest source path"):
        service.validate_and_prepare_submission(manifest_path)


def test_validate_and_prepare_submission_builds_bundle_zip_from_directory_input(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a ZIP archive containing model weights (.pt) and the inference script.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "empty_group").mkdir()
    (bundle_dir / "model.pt").write_bytes(b"weights")
    (bundle_dir / "infer.py").write_text("print('ok')\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(bundle_dir)

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        infos = archive.infolist()
        members = sorted(info.filename for info in infos if not info.is_dir())
        dirs = sorted(info.filename for info in infos if info.is_dir())
    assert members == ["infer.py", "model.pt"]
    assert "empty_group/" in dirs


def test_validate_and_prepare_submission_rejects_bundle_directory_with_only_empty_files(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a ZIP archive containing prediction files.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "submission_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "empty_group").mkdir()
    (bundle_dir / "mask-a.tif").write_bytes(b"")
    (bundle_dir / "nested").mkdir()
    (bundle_dir / "nested" / "mask-b.tif").write_bytes(b"")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    with pytest.raises(SubmissionValidationError, match="submission zip has no non-empty files"):
        service.validate_and_prepare_submission(bundle_dir)


def test_validate_and_prepare_submission_builds_multi_file_zip_from_plain_submission_directory(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a ZIP file containing one file per image prediction.\n",
        encoding="utf-8",
    )

    submission_dir = tmp_path / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "case_001.png").write_bytes(b"mask-1")
    nested = submission_dir / "nested"
    nested.mkdir()
    (nested / "case_002.png").write_bytes(b"mask-2")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_dir)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        assert archive.read("case_001.png") == b"mask-1"
        assert archive.read("nested/case_002.png") == b"mask-2"
    assert members == ["case_001.png", "nested/case_002.png"]


def test_validate_and_prepare_submission_rejects_symlink_in_directory_zip_input(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a ZIP file containing one file per image prediction.\n",
        encoding="utf-8",
    )

    submission_dir = tmp_path / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    target = submission_dir / "case_001.png"
    target.write_bytes(b"mask")
    link = submission_dir / "latest-mask.png"
    try:
        link.symlink_to(target.name)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    with pytest.raises(SubmissionValidationError, match="unsupported symlink member: .*latest-mask\\.png"):
        service.validate_and_prepare_submission(submission_dir)


def test_validate_and_prepare_submission_builds_bundle_tar_xz_from_directory_input(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n"
        "Submit a submission.tar.xz archive containing model weights (.pt) and the inference script.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "empty_group").mkdir()
    (bundle_dir / "model.pt").write_bytes(b"weights")
    (bundle_dir / "infer.py").write_text("print('ok')\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(bundle_dir)

    assert prepared.name == "model_bundle.tar.xz"
    with tarfile.open(prepared, "r:xz") as archive:
        infos = archive.getmembers()
        members = sorted(info.name for info in infos if info.isfile())
        dirs = sorted(info.name for info in infos if info.isdir())
    assert members == ["infer.py", "model.pt"]
    assert "empty_group" in dirs


@pytest.mark.parametrize(
    ("suffix", "mode"),
    [
        (".tar.bz2", "r:bz2"),
        (".tbz2", "r:bz2"),
        (".txz", "r:xz"),
    ],
)
def test_validate_and_prepare_submission_builds_compressed_tar_from_directory_input(
    tmp_path: Path,
    suffix: str,
    mode: str,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n"
        f"Submit a submission{suffix} archive containing model weights (.pt) and the inference script.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "empty_group").mkdir()
    (bundle_dir / "model.pt").write_bytes(b"weights")
    (bundle_dir / "infer.py").write_text("print('ok')\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(bundle_dir)

    assert prepared.name == f"model_bundle{suffix}"
    with tarfile.open(prepared, mode) as archive:
        infos = archive.getmembers()
        members = sorted(info.name for info in infos if info.isfile())
        dirs = sorted(info.name for info in infos if info.isdir())
    assert members == ["infer.py", "model.pt"]
    assert "empty_group" in dirs


def test_validate_and_prepare_submission_rejects_symlink_in_directory_tar_input(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a submission.tar.xz archive containing one file per image prediction.\n",
        encoding="utf-8",
    )

    submission_dir = tmp_path / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    target = submission_dir / "case_001.png"
    target.write_bytes(b"mask")
    link = submission_dir / "latest-mask.png"
    try:
        link.symlink_to(target.name)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    with pytest.raises(SubmissionValidationError, match="unsupported symlink member: .*latest-mask\\.png"):
        service.validate_and_prepare_submission(submission_dir)


def test_validate_and_prepare_submission_builds_bundle_tar_zst_from_directory_input(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n"
        "Submit a submission.tar.zst archive containing model weights (.pt) and the inference script.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "empty_group").mkdir()
    (bundle_dir / "model.pt").write_bytes(b"weights")
    (bundle_dir / "infer.py").write_text("print('ok')\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(bundle_dir)

    assert prepared.name == "model_bundle.tar.zst"
    with prepared.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                infos = list(archive)
    members = sorted(info.name for info in infos if info.isfile())
    dirs = sorted(info.name for info in infos if info.isdir())
    assert members == ["infer.py", "model.pt"]
    assert "empty_group" in dirs


def test_validate_and_prepare_submission_rejects_local_rar_build_from_directory_input(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a submission.rar archive containing masks.\n",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "mask_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "a.tif").write_bytes(b"mask-a")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=context_dir / "missing_sample_submission.csv",
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    with pytest.raises(SubmissionValidationError, match=r"cannot build \.rar submission archives locally"):
        service.validate_and_prepare_submission(bundle_dir)


@pytest.mark.parametrize("suffix", [".tar.zst", ".7z", ".rar"])
def test_preferred_archive_suffix_uses_supported_archive_suffixes(suffix: str) -> None:
    hint = SubmissionFormatHint(
        columns=None,
        delimiter=None,
        expected_suffixes=[suffix],
        artifact_class="bundle",
        artifact_container=suffix.lstrip("."),
    )

    assert SubmissionService._preferred_archive_suffix(hint) == suffix


def test_build_submission_zip_is_deterministic_for_directory_members(tmp_path: Path) -> None:
    left = tmp_path / "left_bundle"
    right = tmp_path / "right_bundle"
    _write_same_bundle_with_mtime(left, mtime=1_000)
    _write_same_bundle_with_mtime(right, mtime=2_000)

    first = SubmissionService._build_submission_zip(left)
    second = SubmissionService._build_submission_zip(right)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first, "r") as archive:
        infos = {info.filename: info for info in archive.infolist()}
    assert infos["infer.py"].date_time == (1980, 1, 1, 0, 0, 0)
    assert infos["empty_group/"].date_time == (1980, 1, 1, 0, 0, 0)


def test_build_submission_tar_gz_is_deterministic_for_directory_members(tmp_path: Path) -> None:
    left = tmp_path / "left_bundle"
    right = tmp_path / "right_bundle"
    _write_same_bundle_with_mtime(left, mtime=1_000)
    _write_same_bundle_with_mtime(right, mtime=2_000)

    first = SubmissionService._build_submission_tar(left, target_suffix=".tar.gz")
    second = SubmissionService._build_submission_tar(right, target_suffix=".tar.gz")

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        infos = {info.name: info for info in archive.getmembers()}
    assert infos["infer.py"].mtime == 0
    assert infos["infer.py"].uid == 0
    assert infos["empty_group"].mtime == 0


def _write_same_bundle_with_mtime(path: Path, *, mtime: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    empty_group = path / "empty_group"
    empty_group.mkdir()
    model = path / "model.pt"
    infer = path / "infer.py"
    model.write_bytes(b"weights")
    infer.write_text("print('ok')\n", encoding="utf-8")
    for member in (empty_group, model, infer, path):
        os.utime(member, (mtime, mtime))
