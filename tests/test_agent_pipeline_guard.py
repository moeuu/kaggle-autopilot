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


def test_guard_ignores_kagglebot_cache_churn(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    cache_file = artifacts_dir / "demo" / "data" / ".kagglebot_cache" / "sample_submission_synth.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("id,target\n1,0\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    cache_file.write_text("id,target\n1,1\n", encoding="utf-8")
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

    assert cache_file.read_text(encoding="utf-8").strip().endswith(",1")


def test_guard_restores_competition_control_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    meta_path = artifacts_dir / "demo" / "meta.json"
    plan_path = artifacts_dir / "demo" / "plan.json"
    prompts_path = artifacts_dir / "demo" / "prompts" / "codex_kernel_fix.md"
    kb_path = repo_root / "knowledge" / "kb.sqlite"

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    (prompts_path.parent).mkdir(parents=True, exist_ok=True)
    kb_path.parent.mkdir(parents=True, exist_ok=True)

    meta_original = '{"slug":"demo"}\n'
    plan_original = '{"pipelines":[]}\n'
    prompts_original = "# original prompt\n"
    kb_original = b"sqlite-bytes"

    meta_path.write_text(meta_original, encoding="utf-8")
    plan_path.write_text(plan_original, encoding="utf-8")
    prompts_path.write_text(prompts_original, encoding="utf-8")
    kb_path.write_bytes(kb_original)

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    meta_path.write_text('{"slug":"changed"}\n', encoding="utf-8")
    plan_path.write_text('{"pipelines":["oops"]}\n', encoding="utf-8")
    prompts_path.write_text("# changed prompt\n", encoding="utf-8")
    kb_path.write_bytes(b"changed-bytes")

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

    assert meta_path.read_text(encoding="utf-8") == meta_original
    assert plan_path.read_text(encoding="utf-8") == plan_original
    assert prompts_path.read_text(encoding="utf-8") == prompts_original
    assert kb_path.read_bytes() == kb_original


def test_guard_ignores_generated_kernel_staging_tree(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    staged_kernel = artifacts_dir / "demo" / "kernels" / "run123" / "local-iter-1" / "kernel.py"
    staged_kernel.parent.mkdir(parents=True, exist_ok=True)
    staged_kernel.write_text("print('original')\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    staged_kernel.write_text("print('changed')\n", encoding="utf-8")
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

    assert staged_kernel.read_text(encoding="utf-8") == "print('changed')\n"
