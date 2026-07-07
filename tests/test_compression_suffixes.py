from __future__ import annotations

from pathlib import Path

import pytest

from kagglebot.asset_modality import ASSET_COMPRESSION_SUFFIXES as ASSET_MODALITY_COMPRESSION_SUFFIXES
from kagglebot.compression_suffixes import (
    ASSET_COMPRESSION_SUFFIXES,
    open_compressed_binary,
    open_compressed_text,
    read_compressed_bytes,
    strip_compression_suffix,
    write_compressed_bytes,
)


def test_asset_modality_reexports_shared_compression_suffixes() -> None:
    assert ASSET_MODALITY_COMPRESSION_SUFFIXES == ASSET_COMPRESSION_SUFFIXES


def test_strip_compression_suffix_removes_one_supported_suffix() -> None:
    assert strip_compression_suffix(".csv.gz") == ".csv"
    assert strip_compression_suffix(".jsonl.zst") == ".jsonl"
    assert strip_compression_suffix(".tar.gz") == ".tar"
    assert strip_compression_suffix(".csv.zip") == ".csv.zip"
    assert strip_compression_suffix(".csv.zip", include_zip=True) == ".csv"


@pytest.mark.parametrize("suffix", ["", *ASSET_COMPRESSION_SUFFIXES])
def test_compressed_bytes_helpers_round_trip(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"payload.txt{suffix}"
    payload = b"alpha,beta\n1,2\n"

    write_compressed_bytes(path, payload, suffix=f".txt{suffix}")

    assert read_compressed_bytes(path, suffix=f".txt{suffix}") == payload
    with open_compressed_binary(path, suffix=f".txt{suffix}") as handle:
        assert handle.read() == payload
    with open_compressed_text(path, suffix=f".txt{suffix}") as handle:
        assert handle.read() == payload.decode("utf-8")
