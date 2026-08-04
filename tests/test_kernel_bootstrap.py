from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from kagglebot.kernel_bootstrap import KERNEL_BOOTSTRAP_MARKER, ensure_kernel_import_path


def test_ensure_kernel_import_path_atomically_replaces_read_only_hardlink(tmp_path: Path) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("print('ok')\n", encoding="utf-8")
    source_path.chmod(0o444)
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_mode = stat.S_IMODE(source_path.stat().st_mode)

    kernel_dir = tmp_path / "stage"
    kernel_dir.mkdir()
    kernel_path = kernel_dir / "kernel.py"
    os.link(source_path, kernel_path)

    ensure_kernel_import_path(kernel_dir)

    first_bytes = kernel_path.read_bytes()
    first_inode = kernel_path.stat().st_ino
    compile(first_bytes, str(kernel_path), "exec")
    assert first_bytes.decode("utf-8").count(KERNEL_BOOTSTRAP_MARKER) == 1
    assert stat.S_IMODE(kernel_path.stat().st_mode) == 0o444
    assert kernel_path.stat().st_ino != source_path.stat().st_ino
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_digest
    assert stat.S_IMODE(source_path.stat().st_mode) == source_mode

    ensure_kernel_import_path(kernel_dir)

    assert kernel_path.read_bytes() == first_bytes
    assert kernel_path.stat().st_ino == first_inode
    assert stat.S_IMODE(kernel_path.stat().st_mode) == 0o444
