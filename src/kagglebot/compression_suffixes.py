from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import zstandard as zstd

ASSET_COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz", ".zst")


def compression_suffix_for(suffix: str) -> str | None:
    value = str(suffix or "").strip().lower()
    for compression_suffix in ASSET_COMPRESSION_SUFFIXES:
        if value.endswith(compression_suffix):
            return compression_suffix
    return None


def strip_compression_suffix(suffix: str, *, include_zip: bool = False) -> str:
    """Remove one supported compression/container suffix from a normalized file suffix."""

    value = str(suffix or "").strip().lower()
    suffixes = (*ASSET_COMPRESSION_SUFFIXES, ".zip") if include_zip else ASSET_COMPRESSION_SUFFIXES
    for compression_suffix in suffixes:
        if value.endswith(compression_suffix):
            return value[: -len(compression_suffix)]
    return value


def open_compressed_text(
    path: Path,
    *,
    suffix: str | None = None,
    encoding: str = "utf-8",
    errors: str = "ignore",
    newline: str | None = None,
):
    compression_suffix = compression_suffix_for(suffix or path.name)
    if compression_suffix == ".gz":
        return gzip.open(path, "rt", encoding=encoding, errors=errors, newline=newline)
    if compression_suffix == ".bz2":
        return bz2.open(path, "rt", encoding=encoding, errors=errors, newline=newline)
    if compression_suffix == ".xz":
        return lzma.open(path, "rt", encoding=encoding, errors=errors, newline=newline)
    if compression_suffix == ".zst":
        return zstd.open(path, "rt", encoding=encoding, errors=errors, newline=newline)
    return path.open("r", encoding=encoding, errors=errors, newline=newline)


@contextmanager
def open_compressed_binary(path: Path, *, suffix: str | None = None):
    compression_suffix = compression_suffix_for(suffix or path.name)
    if compression_suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            yield handle
        return
    if compression_suffix == ".bz2":
        with bz2.open(path, "rb") as handle:
            yield handle
        return
    if compression_suffix == ".xz":
        with lzma.open(path, "rb") as handle:
            yield handle
        return
    if compression_suffix == ".zst":
        yield BytesIO(read_compressed_bytes(path, suffix=suffix))
        return
    with path.open("rb") as handle:
        yield handle


def read_compressed_bytes(path: Path, *, suffix: str | None = None) -> bytes:
    compression_suffix = compression_suffix_for(suffix or path.name)
    if compression_suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    if compression_suffix == ".bz2":
        with bz2.open(path, "rb") as handle:
            return handle.read()
    if compression_suffix == ".xz":
        with lzma.open(path, "rb") as handle:
            return handle.read()
    if compression_suffix == ".zst":
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                return reader.read()
    return path.read_bytes()


def write_compressed_bytes(path: Path, payload: bytes, *, suffix: str | None = None) -> None:
    compression_suffix = compression_suffix_for(suffix or path.name)
    if compression_suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=9) as handle:
            handle.write(payload)
        return
    if compression_suffix == ".bz2":
        with bz2.open(path, "wb") as handle:
            handle.write(payload)
        return
    if compression_suffix == ".xz":
        with lzma.open(path, "wb") as handle:
            handle.write(payload)
        return
    if compression_suffix == ".zst":
        path.write_bytes(zstd.ZstdCompressor(level=9).compress(payload))
        return
    path.write_bytes(payload)


@contextmanager
def open_zstd_tar(path: Path):
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                yield archive
