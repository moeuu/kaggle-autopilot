from __future__ import annotations

from pathlib import Path

from kagglebot.local_kernel_data_resolver import inject_data_dir_resolver


def test_inject_data_dir_resolver_rewrites_candidate_presence_check(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def locate_data_dir(slug: str) -> Path:",
                "    required = ('train.csv', 'test.csv', 'sample_submission.csv')",
                "    for cand in [Path(f'/kaggle/input/{slug}')]:",
                "        if all((cand / name).exists() for name in required):",
                "            return cand",
                "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert "# kagglebot:data_resolver" in updated
    assert "all(_kb_find_file(cand, name).exists() for name in required)" in updated
    assert "_kb_find_file(data_dir, 'train.csv')" in updated
    assert "_kb_find_file(data_dir, 'test.csv')" in updated
    assert "# kagglebot:data-dir-fallback-scan" in updated
    assert "for cand in sorted(input_root.iterdir(), key=lambda p: p.name):" in updated


def test_inject_data_dir_resolver_upgrades_existing_marker(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "# kagglebot:data_resolver",
                "from pathlib import Path as _KBPath",
                "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
                "    return base / name",
                "",
                "def locate_data_dir(slug: str):",
                "    required = ('train.csv', 'test.csv', 'sample_submission.csv')",
                "    for cand in [Path(f'/kaggle/input/{slug}')]:",
                "        if all((cand / name).exists() for name in required):",
                "            return cand",
                "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")",
                "",
                "def load_competition_frames(data_dir):",
                "    return data_dir / 'train.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert updated.count("# kagglebot:data_resolver") == 1
    assert "all(_kb_find_file(cand, name).exists() for name in required)" in updated
    assert "_kb_find_file(data_dir, 'train.csv')" in updated
    assert "# kagglebot:data-dir-fallback-scan" in updated
