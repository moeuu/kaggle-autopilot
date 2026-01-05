from pathlib import Path

from kagglebot.orchestrator import agent_pipeline


def test_guard_restores_other_competition_kernel(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_kernel = artifacts_dir / "other" / "kernel"
    other_kernel.mkdir(parents=True, exist_ok=True)
    other_kernel_path = other_kernel / "kernel.py"
    original = "print('original')\n"
    other_kernel_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    # Simulate an unauthorized edit outside the allowlist.
    other_kernel_path.write_text("print('changed')\n", encoding="utf-8")
    after = agent_pipeline._snapshot_tree(repo_root)

    agent_pipeline._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert other_kernel_path.read_text(encoding="utf-8") == original
