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


def test_guard_restores_other_competition_submission_ledger(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_submissions = artifacts_dir / "other" / "submissions"
    other_submissions.mkdir(parents=True, exist_ok=True)
    ledger_path = other_submissions / "ledger.jsonl"
    original = '{"run_id":"r1","hash":"abc"}\n'
    ledger_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    ledger_path.write_text('{"run_id":"r2","hash":"def"}\n', encoding="utf-8")
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

    assert ledger_path.read_text(encoding="utf-8") == original


def test_guard_restores_other_competition_data_sample_submission(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_data = artifacts_dir / "other" / "data"
    other_data.mkdir(parents=True, exist_ok=True)
    sample_path = other_data / "sample_submission.csv"
    original = "Id,Category\nval_1.tif,Health\n"
    sample_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    sample_path.write_text("Id,Category\nval_1.tif,Rust\n", encoding="utf-8")
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

    assert sample_path.read_text(encoding="utf-8") == original


def test_guard_restores_oversized_data_sample_submission_from_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    slug = "beyond-visible-spectrum-ai-for-agriculture-2026"
    context_sample = artifacts_dir / slug / "context" / "sample_submission.csv"
    data_sample = artifacts_dir / slug / "data" / "sample_submission.csv"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)

    context_content = "Id,Category\nval_1.tif,Healthy\n"
    context_sample.write_text(context_content, encoding="utf-8")

    row = "val_1.tif,Healthy\n"
    repeat = (agent_pipeline._MAX_GUARD_FILE_BYTES // len(row)) + 1_000
    oversized_content = "Id,Category\n" + (row * repeat)
    data_sample.write_text(oversized_content, encoding="utf-8")
    assert data_sample.stat().st_size > agent_pipeline._MAX_GUARD_FILE_BYTES

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    sample_rel = data_sample.relative_to(repo_root).as_posix()
    assert sample_rel in guard_snapshot.oversized
    before = agent_pipeline._snapshot_tree(repo_root)

    data_sample.write_text("Id,Category\nval_1.tif,Rust\n", encoding="utf-8")
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

    assert data_sample.read_text(encoding="utf-8") == context_content


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


def test_guard_restores_knowledge_research_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    research_dir = repo_root / "knowledge" / "research" / "unknown" / "demo"
    research_dir.mkdir(parents=True, exist_ok=True)
    sources_path = research_dir / "research_sources.jsonl"
    summary_path = research_dir / "research_summary.md"

    sources_original = '{"url":"https://example.com","title":"Example"}\n'
    summary_original = "# Research\n\nOriginal summary.\n"
    sources_path.write_text(sources_original, encoding="utf-8")
    summary_path.write_text(summary_original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    sources_path.write_text('{"url":"https://bad.example","title":"Changed"}\n', encoding="utf-8")
    summary_path.write_text("# Research\n\nChanged summary.\n", encoding="utf-8")

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

    assert sources_path.read_text(encoding="utf-8") == sources_original
    assert summary_path.read_text(encoding="utf-8") == summary_original


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


def test_guard_ignores_historical_run_submission_compact_csv_churn(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    compact_submission = artifacts_dir / "other" / "runs" / "run123" / "iter-3" / "submission.compact.csv"
    compact_submission.parent.mkdir(parents=True, exist_ok=True)
    compact_submission.write_text("id,target\n1,0\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    compact_submission.write_text("id,target\n1,1\n", encoding="utf-8")
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

    assert compact_submission.read_text(encoding="utf-8") == "id,target\n1,1\n"


def test_guard_ignores_venv_churn_and_restores_uv_lock(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    uv_lock = repo_root / "uv.lock"
    uv_lock_original = "lock-version = 1\n"
    uv_lock.write_text(uv_lock_original, encoding="utf-8")

    venv_entrypoint = repo_root / ".venv" / "bin" / "kagglebot"
    venv_entrypoint.parent.mkdir(parents=True, exist_ok=True)
    venv_entrypoint.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    uv_lock.write_text("lock-version = 999\n", encoding="utf-8")
    venv_entrypoint.write_text("# changed\n", encoding="utf-8")
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

    assert uv_lock.read_text(encoding="utf-8") == uv_lock_original
    assert venv_entrypoint.read_text(encoding="utf-8") == "# changed\n"


def test_guard_allows_explicit_dependency_file_edits(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    pyproject_path = repo_root / "pyproject.toml"
    uv_lock_path = repo_root / "uv.lock"
    pyproject_path.write_text("[project]\nname='demo'\n", encoding="utf-8")
    uv_lock_path.write_text("lock-version = 1\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel, pyproject_path, uv_lock_path]
    guard_snapshot = agent_pipeline._backup_guarded_files(repo_root, allowed_prefixes)
    before = agent_pipeline._snapshot_tree(repo_root)

    pyproject_path.write_text("[project]\nname='demo'\ndependencies=['albumentations']\n", encoding="utf-8")
    uv_lock_path.write_text("lock-version = 2\n", encoding="utf-8")
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

    assert "albumentations" in pyproject_path.read_text(encoding="utf-8")
    assert uv_lock_path.read_text(encoding="utf-8") == "lock-version = 2\n"
