from __future__ import annotations

import struct
from pathlib import Path

import pytest

from kagglebot.writeup_card import ensure_writeup_card


def test_ensure_writeup_card_creates_deterministic_exact_size_png(tmp_path: Path) -> None:
    first = ensure_writeup_card(tmp_path / "first.png")
    second = ensure_writeup_card(tmp_path / "second.png")

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", first.read_bytes()[16:24]) == (560, 280)


def test_ensure_writeup_card_rejects_wrong_existing_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "card.png"
    path.write_bytes(b"not a png")

    with pytest.raises(ValueError, match="560x280"):
        ensure_writeup_card(path)
