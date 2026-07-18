from __future__ import annotations

from pathlib import Path

from kagglebot.autopilot import _diff_snapshots, _snapshot_tree


def test_kernel_fix_change_detection_includes_untracked_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_dir = repo_root / "src" / "kagglebot"
    source_dir.mkdir(parents=True)
    before = _snapshot_tree(repo_root)

    untracked_path = source_dir / "new_generator.py"
    untracked_path.write_text("def render():\n    return 'kernel'\n", encoding="utf-8")
    after = _snapshot_tree(repo_root)

    assert _diff_snapshots(before, after) == ["src/kagglebot/new_generator.py"]
