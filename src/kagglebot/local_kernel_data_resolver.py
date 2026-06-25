from __future__ import annotations

import re
from pathlib import Path

from kagglebot import kernel_bootstrap

KERNEL_DATA_RESOLVER_MARKER = "# kagglebot:data_resolver"
DATA_DIR_LOCATE_FALLBACK_MARKER = "# kagglebot:data-dir-fallback-scan"

_DATA_DIR_JOIN_RE = re.compile(r"(\bdata_dir\s*/\s*)(['\"])([^'\"]+)\2")
_DATA_DIR_REQUIRED_RE = re.compile(r"all\(\(cand\s*/\s*name\)\.exists\(\)\s*for\s*name\s*in\s*required\)")
_DATA_DIR_RAISE_RE = re.compile(
    r"^\s*raise FileNotFoundError\(f\"Could not find required csv files for slug='\{slug\}'\"\)\s*$",
    re.MULTILINE,
)


def inject_data_dir_resolver(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if not _DATA_DIR_JOIN_RE.search(text):
        return
    lines = text.splitlines()
    if KERNEL_DATA_RESOLVER_MARKER not in text:
        resolver_block = [
            KERNEL_DATA_RESOLVER_MARKER,
            "from pathlib import Path as _KBPath",
            "",
            "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
            "    candidate = base / name",
            "    if candidate.exists():",
            "        return candidate",
            "    try:",
            "        matches = list(base.rglob(name))",
            "    except Exception:",
            "        matches = []",
            "    if matches:",
            "        return matches[0]",
            "    return candidate",
            "",
        ]
        insert_at = kernel_bootstrap.find_bootstrap_block_end(lines)
        if insert_at is None:
            insert_at = kernel_bootstrap.find_bootstrap_insertion_index(lines)
        lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    updated = _DATA_DIR_JOIN_RE.sub(r"_kb_find_file(data_dir, '\3')", updated)
    updated = _DATA_DIR_REQUIRED_RE.sub(
        "all(_kb_find_file(cand, name).exists() for name in required)",
        updated,
    )
    if DATA_DIR_LOCATE_FALLBACK_MARKER not in updated:
        fallback_block = (
            "    input_root = _KBPath('/kaggle/input')\n"
            "    if input_root.exists() and input_root.is_dir():\n"
            f"        {DATA_DIR_LOCATE_FALLBACK_MARKER}\n"
            "        for cand in sorted(input_root.iterdir(), key=lambda p: p.name):\n"
            "            if not cand.is_dir():\n"
            "                continue\n"
            "            if all(_kb_find_file(cand, name).exists() for name in required):\n"
            "                return cand\n"
            "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")"
        )
        updated = _DATA_DIR_RAISE_RE.sub(fallback_block, updated, count=1)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")
