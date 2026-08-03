from __future__ import annotations

import struct
import zlib
from pathlib import Path

CARD_WIDTH = 560
CARD_HEIGHT = 280


def ensure_writeup_card(path: Path) -> Path:
    """Keep an existing exact-size PNG or create a deterministic neutral card."""
    if path.is_file():
        if _png_dimensions(path) != (CARD_WIDTH, CARD_HEIGHT):
            raise ValueError(f"Writeup card must be {CARD_WIDTH}x{CARD_HEIGHT}: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_background_row(y) for y in range(CARD_HEIGHT)]
    _draw_portfolio(rows)
    path.write_bytes(_encode_png(rows))
    return path


def _background_row(y: int) -> bytearray:
    row = bytearray()
    for x in range(CARD_WIDTH):
        glow = max(0, 54 - abs(x - CARD_WIDTH // 2) // 6 - abs(y - CARD_HEIGHT // 2) // 4)
        row.extend((5 + glow // 7, 12 + glow // 3, 27 + glow // 2))
    return row


def _draw_portfolio(rows: list[bytearray]) -> None:
    center = (CARD_WIDTH // 2, CARD_HEIGHT // 2)
    nodes = ((100, 65), (100, 140), (100, 215), (460, 65), (460, 140), (460, 215))
    active = {0, 2, 4}
    for index, node in enumerate(nodes):
        _line(rows, center, node, (28, 190, 235) if index in active else (48, 73, 105), width=3)
    _shield(rows, center)
    for index, (x, y) in enumerate(nodes):
        border = (40, 205, 244) if index in active else (72, 101, 137)
        _rounded_tile(rows, x - 36, y - 25, 72, 50, border)
        _circle(rows, x, y, 8, (245, 174, 52) if index in active else (62, 91, 126))


def _shield(rows: list[bytearray], center: tuple[int, int]) -> None:
    cx, cy = center
    points = (
        (cx, cy - 76),
        (cx + 68, cy - 45),
        (cx + 57, cy + 42),
        (cx, cy + 83),
        (cx - 57, cy + 42),
        (cx - 68, cy - 45),
    )
    for index, point in enumerate(points):
        _line(rows, point, points[(index + 1) % len(points)], (47, 200, 243), width=4)
    _circle(rows, cx, cy - 3, 41, (15, 64, 101))
    _circle(rows, cx, cy - 3, 33, (20, 139, 189))
    _circle(rows, cx - 13, cy - 6, 4, (218, 249, 255))
    _circle(rows, cx + 13, cy - 6, 4, (218, 249, 255))
    _line(rows, (cx - 13, cy + 16), (cx - 2, cy + 27), (246, 184, 66), width=5)
    _line(rows, (cx - 2, cy + 27), (cx + 19, cy + 3), (246, 184, 66), width=5)


def _rounded_tile(rows: list[bytearray], x: int, y: int, width: int, height: int, border: tuple[int, int, int]) -> None:
    for py in range(y, y + height):
        for px in range(x, x + width):
            edge = min(px - x, x + width - 1 - px, py - y, y + height - 1 - py)
            color = border if edge < 3 else (13, 34, 59)
            _pixel(rows, px, py, color)


def _circle(rows: list[bytearray], cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    radius_sq = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius_sq:
                _pixel(rows, x, y, color)


def _line(
    rows: list[bytearray],
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    radius = max(0, width // 2)
    for step in range(steps + 1):
        x = round(x0 + (x1 - x0) * step / steps)
        y = round(y0 + (y1 - y0) * step / steps)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                _pixel(rows, x + dx, y + dy, color)


def _pixel(rows: list[bytearray], x: int, y: int, color: tuple[int, int, int]) -> None:
    if not 0 <= y < len(rows) or not 0 <= x < CARD_WIDTH:
        return
    offset = x * 3
    rows[y][offset : offset + 3] = bytes(color)


def _encode_png(rows: list[bytearray]) -> bytes:
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", CARD_WIDTH, CARD_HEIGHT, 8, 2, 0, 0, 0)
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, level=9)) + _chunk(b"IEND", b"")


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", header[16:24])
