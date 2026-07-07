"""Tests for validator helpers."""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import py7zr
import pytest
import zstandard as zstd

from kagglebot.validators import (
    extract_data_archives,
    extract_zip_archives,
    safe_extract_7z,
    safe_extract_rar,
    safe_extract_tar,
    safe_extract_tar_zst,
    safe_extract_zip,
    validate_kernel_package,
    validate_kernel_sources,
    validate_slug,
)


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


def test_safe_extract_zip_rejects_duplicate_targets(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("train.csv", "first\n")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("train.csv", "second\n")

    with pytest.raises(ValueError, match="Duplicate archive member target"):
        safe_extract_zip(zip_path, tmp_path)


def test_safe_extract_zip_rejects_symlink_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    link_info = zipfile.ZipInfo("train.csv")
    link_info.create_system = 3
    link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(link_info, "real_train.csv")

    with pytest.raises(ValueError, match="Unsupported zip symlink member"):
        safe_extract_zip(zip_path, tmp_path)


def test_safe_extract_tar_blocks_traversal(tmp_path: Path) -> None:
    tar_path = tmp_path / "evil.tar.gz"
    payload = b"nope"
    with tarfile.open(tar_path, "w:gz") as archive:
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_tar(tar_path, tmp_path)


def test_safe_extract_tar_rejects_links(tmp_path: Path) -> None:
    tar_path = tmp_path / "links.tar"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("link.csv")
        info.type = tarfile.SYMTYPE
        info.linkname = "train.csv"
        archive.addfile(info)

    with pytest.raises(ValueError, match="Unsupported tar member type"):
        safe_extract_tar(tar_path, tmp_path)


def test_safe_extract_tar_rejects_duplicate_targets(tmp_path: Path) -> None:
    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as archive:
        for payload in (b"first\n", b"second\n"):
            info = tarfile.TarInfo("train.csv")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Duplicate archive member target"):
        safe_extract_tar(tar_path, tmp_path)


def test_safe_extract_tar_zst_blocks_traversal(tmp_path: Path) -> None:
    raw_tar = io.BytesIO()
    payload = b"nope"
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    tar_path = tmp_path / "evil.tar.zst"
    tar_path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_tar_zst(tar_path, tmp_path)


def test_safe_extract_7z_blocks_traversal(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("nope", encoding="utf-8")
    archive_path = tmp_path / "evil.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(payload, "../evil.txt")

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_7z(archive_path, tmp_path)


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

    def extract(self, path: Path, targets: list[str]) -> None:
        for target in targets:
            destination = path / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("payload\n", encoding="utf-8")


def test_safe_extract_7z_rejects_duplicate_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "data.7z"
    archive_path.write_bytes(b"fake-7z")
    _Fake7zFile.members_by_name = {"data.7z": [_Fake7zMember("train.csv"), _Fake7zMember("train.csv")]}
    _Fake7zFile.password_required_by_name = {}
    monkeypatch.setattr("kagglebot.validators.py7zr.SevenZipFile", _Fake7zFile)

    with pytest.raises(ValueError, match="Duplicate archive member target"):
        safe_extract_7z(archive_path, tmp_path)


def test_safe_extract_7z_rejects_symlink_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "data.7z"
    archive_path.write_bytes(b"fake-7z")
    _Fake7zFile.members_by_name = {"data.7z": [_Fake7zMember("latest_train.csv", is_file=False, is_symlink=True)]}
    _Fake7zFile.password_required_by_name = {}
    monkeypatch.setattr("kagglebot.validators.py7zr.SevenZipFile", _Fake7zFile)

    with pytest.raises(ValueError, match="Unsupported 7z member type"):
        safe_extract_7z(archive_path, tmp_path)


def test_safe_extract_7z_rejects_password_protected_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "data.7z"
    archive_path.write_bytes(b"fake-7z")
    _Fake7zFile.members_by_name = {"data.7z": [_Fake7zMember("train.csv")]}
    _Fake7zFile.password_required_by_name = {"data.7z": True}
    monkeypatch.setattr("kagglebot.validators.py7zr.SevenZipFile", _Fake7zFile)

    with pytest.raises(ValueError, match="Unsupported password-protected 7z archive"):
        safe_extract_7z(archive_path, tmp_path)


class _FakeRarMember:
    def __init__(
        self,
        filename: str,
        payload: bytes = b"",
        *,
        is_dir: bool = False,
        is_file: bool = True,
        is_symlink: bool = False,
        needs_password: bool = False,
    ) -> None:
        self.filename = filename
        self.payload = payload
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

    def open(self, member: _FakeRarMember, mode: str = "r") -> io.BytesIO:
        del mode
        return io.BytesIO(member.payload)


def test_safe_extract_rar_blocks_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "evil.rar"
    archive_path.write_bytes(b"fake-rar")
    _FakeRarFile.members_by_name = {"evil.rar": [_FakeRarMember("../evil.txt", b"nope")]}
    monkeypatch.setattr("kagglebot.validators.rarfile.RarFile", _FakeRarFile)

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_rar(archive_path, tmp_path)


def test_safe_extract_rar_rejects_duplicate_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "data.rar"
    archive_path.write_bytes(b"fake-rar")
    _FakeRarFile.members_by_name = {
        "data.rar": [_FakeRarMember("train.csv", b"first\n"), _FakeRarMember("train.csv", b"second\n")]
    }
    monkeypatch.setattr("kagglebot.validators.rarfile.RarFile", _FakeRarFile)

    with pytest.raises(ValueError, match="Duplicate archive member target"):
        safe_extract_rar(archive_path, tmp_path)


def test_extract_zip_archives_does_not_overwrite_existing_files_by_default(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text("existing\n", encoding="utf-8")
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("train.csv", "new\n")
        archive.writestr("test.csv", "id\n1\n")

    extracted = extract_zip_archives(tmp_path)

    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == "existing\n"
    assert (tmp_path / "test.csv").exists()
    assert tmp_path / "train.csv" not in extracted
    assert tmp_path / "test.csv" in extracted


def test_extract_data_archives_handles_tgz(tmp_path: Path) -> None:
    tar_path = tmp_path / "competition.tgz"
    payload = b"id,target\n1,0\n"
    with tarfile.open(tar_path, "w:gz") as archive:
        info = tarfile.TarInfo("train.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    extracted = extract_data_archives(tmp_path)

    assert tmp_path / "train.csv" in extracted
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == payload.decode()


def test_extract_data_archives_handles_tar_xz(tmp_path: Path) -> None:
    tar_path = tmp_path / "competition.tar.xz"
    payload = b"id,target\n1,0\n"
    with tarfile.open(tar_path, "w:xz") as archive:
        info = tarfile.TarInfo("train.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    extracted = extract_data_archives(tmp_path)

    assert tmp_path / "train.csv" in extracted
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == payload.decode()


@pytest.mark.parametrize("suffix", [".tar.zst", ".tzst"])
def test_extract_data_archives_handles_tar_zst(tmp_path: Path, suffix: str) -> None:
    raw_tar = io.BytesIO()
    payload = b"id,target\n1,0\n"
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        info = tarfile.TarInfo("train.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    tar_path = tmp_path / f"competition{suffix}"
    tar_path.write_bytes(zstd.ZstdCompressor().compress(raw_tar.getvalue()))

    extracted = extract_data_archives(tmp_path)

    assert tmp_path / "train.csv" in extracted
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == payload.decode()


def test_extract_data_archives_dispatches_tar_zst_from_compound_archive_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "competition.tar.zst"
    archive_path.write_bytes(b"fake-zstd-tar")
    extracted_path = tmp_path / "train.csv"

    def fake_extract_tar_zst(path: Path, dest_dir: Path, *, overwrite: bool) -> list[Path]:
        assert path == archive_path
        assert dest_dir == tmp_path
        assert overwrite is False
        extracted_path.write_text("id,target\n1,0\n", encoding="utf-8")
        return [extracted_path]

    monkeypatch.setattr("kagglebot.validators.safe_extract_tar_zst", fake_extract_tar_zst)
    monkeypatch.setattr(
        "kagglebot.validators.safe_extract_tar",
        lambda *args, **kwargs: pytest.fail("tar.zst must use the zstd tar extractor"),
    )

    extracted = extract_data_archives(tmp_path)

    assert extracted == [extracted_path]


def test_extract_data_archives_handles_7z(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("id,target\n1,0\n", encoding="utf-8")
    archive_path = tmp_path / "competition.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(source, "train.csv")
    source.unlink()

    extracted = extract_data_archives(tmp_path)

    assert tmp_path / "train.csv" in extracted
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == "id,target\n1,0\n"


def test_extract_data_archives_handles_rar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "competition.rar"
    archive_path.write_bytes(b"fake-rar")
    _FakeRarFile.members_by_name = {"competition.rar": [_FakeRarMember("train.csv", b"id,target\n1,0\n")]}
    monkeypatch.setattr("kagglebot.validators.rarfile.RarFile", _FakeRarFile)

    extracted = extract_data_archives(tmp_path)

    assert tmp_path / "train.csv" in extracted
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == "id,target\n1,0\n"


def test_extract_data_archives_extracts_nested_archives(tmp_path: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("train.csv", "id,target\n1,0\n")
    with zipfile.ZipFile(tmp_path / "competition.zip", "w") as archive:
        archive.writestr("train.zip", inner.getvalue())
        archive.writestr("sample_submission.csv", "id,target\n2,0\n")

    extracted = extract_data_archives(tmp_path)

    assert tmp_path / "train.zip" in extracted
    assert tmp_path / "train.csv" in extracted
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == "id,target\n1,0\n"


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


@pytest.mark.parametrize("suffix", [".tar.gz", ".tgz", ".tar.xz", ".tar.zst"])
def test_validate_kernel_sources_accepts_named_tar_submission_output(tmp_path: Path, suffix: str) -> None:
    (tmp_path / "kernel.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "data = Path('/kaggle/input/demo/test.csv')",
                f"Path('/kaggle/working/submission_model{suffix}').write_bytes(b'archive')",
                "Path('/kaggle/working/metrics.json').write_text('{}')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert validate_kernel_sources(tmp_path) == []


@pytest.mark.parametrize("suffix", [".7z", ".rar"])
def test_validate_kernel_sources_accepts_named_external_archive_submission_output(tmp_path: Path, suffix: str) -> None:
    (tmp_path / "kernel.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "data = Path('/kaggle/input/demo/test.csv')",
                f"Path('/kaggle/working/submission_model{suffix}').write_bytes(b'archive')",
                "Path('/kaggle/working/metrics.json').write_text('{}')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert validate_kernel_sources(tmp_path) == []


@pytest.mark.parametrize("suffix", [".tar.gz", ".tgz", ".tar.xz", ".tar.zst"])
def test_validate_kernel_sources_accepts_generic_tar_submission_alias(tmp_path: Path, suffix: str) -> None:
    (tmp_path / "kernel.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "data = Path('/kaggle/input/demo/test.csv')",
                f"Path('/kaggle/working/predictions{suffix}').write_bytes(b'archive')",
                "Path('/kaggle/working/metrics.json').write_text('{}')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert validate_kernel_sources(tmp_path) == []


@pytest.mark.parametrize("suffix", [".7z", ".rar"])
def test_validate_kernel_sources_accepts_generic_external_archive_submission_alias(tmp_path: Path, suffix: str) -> None:
    (tmp_path / "kernel.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "data = Path('/kaggle/input/demo/test.csv')",
                f"Path('/kaggle/working/predictions{suffix}').write_bytes(b'archive')",
                "Path('/kaggle/working/metrics.json').write_text('{}')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert validate_kernel_sources(tmp_path) == []


@pytest.mark.parametrize(
    "name",
    [
        "submission.npy",
        "predictions.png",
        "masks.ome.tif",
        "results.onnx",
        "checkpoint.safetensors",
        "adapter_model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "submission.py",
        "submission.ipynb",
        "submission.r",
        "submission.jl",
    ],
)
def test_validate_kernel_sources_accepts_non_tabular_single_file_submission_output(tmp_path: Path, name: str) -> None:
    (tmp_path / "kernel.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "data = Path('/kaggle/input/demo/test.csv')",
                f"Path('/kaggle/working/{name}').write_bytes(b'artifact')",
                "Path('/kaggle/working/metrics.json').write_text('{}')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert validate_kernel_sources(tmp_path) == []


def test_validate_kernel_sources_does_not_treat_model_named_tabular_helper_as_submission(tmp_path: Path) -> None:
    (tmp_path / "kernel.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "data = Path('/kaggle/input/demo/test.csv')",
                "Path('/kaggle/working/model.csv').write_text('id,target\\n1,0\\n')",
                "Path('/kaggle/working/metrics.json').write_text('{}')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert validate_kernel_sources(tmp_path) == [
        "Kernel sources do not reference a supported submission output artifact."
    ]
