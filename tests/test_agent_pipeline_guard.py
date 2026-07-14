from pathlib import Path
from types import SimpleNamespace

import pytest

from kagglebot import write_guard
from kagglebot.exceptions import KaggleBotError
from kagglebot.orchestrator import agent_pipeline
from kagglebot.orchestrator.agent_pipeline import AgentPipelineConfig
from kagglebot.paths import CompetitionPaths


def test_fallback_strategy_instructions_cover_general_tabular_formats(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    strategy, instructions, _, research_sources_text, _ = agent_pipeline._build_fallback_strategy(
        paths=paths,
        config=config,
        brief_content="",
        error_text="strategy unavailable",
    )

    assert "point-cloud/3D" in strategy
    assert "annotation" in strategy
    assert "artifact outputs" in strategy
    assert "tabular/text/image/sequence" not in strategy
    assert "Excel" in instructions
    assert "Feather" in instructions
    assert "Stata" in instructions
    assert "XML" in instructions
    assert "SQLite" in instructions
    assert "compressed tabular variants" in instructions
    assert "non-tabular artifacts" in instructions
    assert "point-cloud/3D" in instructions
    assert "model files" in instructions
    assert "forced to CSV" in instructions
    assert "annotation" in instructions
    assert "model-artifact specific paths" in instructions
    assert "CNN/transformer/sequence models where useful" in instructions
    assert "repo `read_table` helper" in instructions
    assert "submission format file schema artifact" in research_sources_text
    assert "submission format csv columns" not in research_sources_text


def test_codex_brief_guard_ignores_sibling_competition_artifact_churn(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "kaggle-autopilot-artifacts")
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
    output_dir = paths.context_agent_dir / "brief"
    prompt_path = output_dir / "prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("brief prompt\n", encoding="utf-8")

    other_context = paths.artifacts_dir / "other-slug" / "context"
    other_context.mkdir(parents=True, exist_ok=True)
    other_guard = other_context / "zero_overlap_drift_guard.json"
    other_guard.write_text('{"status":"original"}\n', encoding="utf-8")

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        other_guard.write_text('{"status":"changed"}\n', encoding="utf-8")
        last_message_path = output_dir / "last_message.txt"
        last_message_path.write_text("Brief text\n", encoding="utf-8")
        return SimpleNamespace(last_message_path=last_message_path, returncode=0, stderr="")

    monkeypatch.setattr(agent_pipeline, "run_codex", fake_run_codex)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    brief_text, error_text = agent_pipeline._run_codex_brief_with_retry(
        prompt_path=prompt_path,
        output_dir=output_dir,
        paths=paths,
        config=config,
    )

    assert brief_text == "Brief text"
    assert error_text == ""
    assert other_guard.read_text(encoding="utf-8") == '{"status":"changed"}\n'


def test_codex_brief_guard_rejects_active_context_edits_outside_agent(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "kaggle-autopilot-artifacts")
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
    output_dir = paths.context_agent_dir / "brief"
    prompt_path = output_dir / "prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("brief prompt\n", encoding="utf-8")

    guard_path = paths.context_dir / "zero_overlap_drift_guard.json"
    guard_path.write_text('{"status":"original"}\n', encoding="utf-8")

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        guard_path.write_text('{"status":"changed"}\n', encoding="utf-8")
        last_message_path = output_dir / "last_message.txt"
        last_message_path.write_text("Brief text\n", encoding="utf-8")
        return SimpleNamespace(last_message_path=last_message_path, returncode=0, stderr="")

    monkeypatch.setattr(agent_pipeline, "run_codex", fake_run_codex)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    try:
        agent_pipeline._run_codex_brief_with_retry(
            prompt_path=prompt_path,
            output_dir=output_dir,
            paths=paths,
            config=config,
        )
    except KaggleBotError as exc:
        message = str(exc)
        assert "Agent write-guard failed in codex_brief" in message
        assert "context/zero_overlap_drift_guard.json" in message
    else:
        raise AssertionError("expected active competition context edit to be rejected")


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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    # Simulate an unauthorized edit outside the allowlist.
    other_kernel_path.write_text("print('changed')\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    ledger_path.write_text('{"run_id":"r2","hash":"def"}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    sample_path.write_text("Id,Category\nval_1.tif,Rust\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert sample_path.read_text(encoding="utf-8") == original


def test_guard_restores_other_competition_jsonl_sample_submission(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_data = artifacts_dir / "other" / "data"
    other_data.mkdir(parents=True, exist_ok=True)
    sample_path = other_data / "sample_submission.jsonl"
    original = '{"id":1,"target":0.1}\n'
    sample_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    sample_path.write_text('{"id":1,"target":0.9}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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
    repeat = (write_guard._MAX_GUARD_FILE_BYTES // len(row)) + 1_000
    oversized_content = "Id,Category\n" + (row * repeat)
    data_sample.write_text(oversized_content, encoding="utf-8")
    assert data_sample.stat().st_size > write_guard._MAX_GUARD_FILE_BYTES

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    sample_rel = data_sample.relative_to(repo_root).as_posix()
    assert sample_rel in guard_snapshot.oversized
    before = write_guard._snapshot_tree(repo_root)

    data_sample.write_text("Id,Category\nval_1.tif,Rust\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert data_sample.read_text(encoding="utf-8") == context_content


def test_guard_restores_oversized_jsonl_data_sample_submission_from_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    slug = "jsonl-demo"
    context_sample = artifacts_dir / slug / "context" / "sample_submission.jsonl"
    data_sample = artifacts_dir / slug / "data" / "sample_submission.jsonl"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)

    context_content = '{"id":1,"target":0.1}\n'
    context_sample.write_text(context_content, encoding="utf-8")

    row = '{"id":1,"target":0.1}\n'
    repeat = (write_guard._MAX_GUARD_FILE_BYTES // len(row)) + 1_000
    data_sample.write_text(row * repeat, encoding="utf-8")
    assert data_sample.stat().st_size > write_guard._MAX_GUARD_FILE_BYTES

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    sample_rel = data_sample.relative_to(repo_root).as_posix()
    assert sample_rel in guard_snapshot.oversized
    before = write_guard._snapshot_tree(repo_root)

    data_sample.write_text('{"id":1,"target":0.9}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert data_sample.read_text(encoding="utf-8") == context_content


def test_guard_recognizes_compressed_data_sample_submission() -> None:
    assert write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.csv.gz")
    assert write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.xlsx")
    assert write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.jsonl.xz")
    assert not write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.zip")


def test_guard_matches_compressed_data_sample_submission_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    context_sample = repo_root / "artifacts" / "demo" / "context" / "sample_submission.csv.gz"
    data_sample = repo_root / "artifacts" / "demo" / "data" / "sample_submission.csv.gz"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\x1f\x8bcompressed-sample-bytes"
    context_sample.write_bytes(payload)
    data_sample.write_bytes(payload)

    assert write_guard._matches_artifact_data_sample_submission_context(
        repo_root,
        "artifacts/demo/data/sample_submission.csv.gz",
    )


def test_guard_matches_excel_data_sample_submission_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    context_sample = repo_root / "artifacts" / "demo" / "context" / "sample_submission.xlsx"
    data_sample = repo_root / "artifacts" / "demo" / "data" / "sample_submission.xlsx"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)
    payload = b"excel-sample-bytes"
    context_sample.write_bytes(payload)
    data_sample.write_bytes(payload)

    assert write_guard._matches_artifact_data_sample_submission_context(
        repo_root,
        "artifacts/demo/data/sample_submission.xlsx",
    )


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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    cache_file.write_text("id,target\n1,1\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    meta_path.write_text('{"slug":"changed"}\n', encoding="utf-8")
    plan_path.write_text('{"pipelines":["oops"]}\n', encoding="utf-8")
    prompts_path.write_text("# changed prompt\n", encoding="utf-8")
    kb_path.write_bytes(b"changed-bytes")

    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    sources_path.write_text('{"url":"https://bad.example","title":"Changed"}\n', encoding="utf-8")
    summary_path.write_text("# Research\n\nChanged summary.\n", encoding="utf-8")

    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    staged_kernel.write_text("print('changed')\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert staged_kernel.read_text(encoding="utf-8") == "print('changed')\n"


@pytest.mark.parametrize(
    ("filename", "payload", "updated_payload"),
    [
        ("submission.compact.csv", b"id,target\n1,0\n", b"id,target\n1,1\n"),
        ("submission.compact.csv.gz", b"compressed-before", b"compressed-after"),
        ("submission.compact.tsv", b"id\ttarget\n1\t0\n", b"id\ttarget\n1\t1\n"),
        ("submission.compact.tsv.zst", b"tsv-zst-before", b"tsv-zst-after"),
        ("submission.compact.jsonl.zst", b"jsonl-zst-before", b"jsonl-zst-after"),
    ],
)
def test_guard_ignores_historical_run_submission_compact_churn(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    updated_payload: bytes,
) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    compact_submission = artifacts_dir / "other" / "runs" / "run123" / "iter-3" / filename
    compact_submission.parent.mkdir(parents=True, exist_ok=True)
    compact_submission.write_bytes(payload)

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    compact_submission.write_bytes(updated_payload)
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert compact_submission.read_bytes() == updated_payload


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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    uv_lock.write_text("lock-version = 999\n", encoding="utf-8")
    venv_entrypoint.write_text("# changed\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    pyproject_path.write_text("[project]\nname='demo'\ndependencies=['albumentations']\n", encoding="utf-8")
    uv_lock_path.write_text("lock-version = 2\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
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


def test_repo_root_write_policy_allows_src_but_restores_data_and_generated_kernels(tmp_path: Path) -> None:
    repo_root = tmp_path
    src_path = repo_root / "src" / "demo.py"
    data_path = repo_root / "artifacts" / "demo" / "data" / "train.csv"
    staged_kernel_path = repo_root / "artifacts" / "demo" / "kernels" / "run123" / "local-iter-1" / "kernel.py"
    src_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    staged_kernel_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text("VALUE = 1\n", encoding="utf-8")
    data_path.write_text("id,target\n1,0\n", encoding="utf-8")
    staged_kernel_path.write_text("print('original')\n", encoding="utf-8")

    policy = write_guard._repo_root_write_policy(
        repo_root=repo_root,
        denied_prefixes=[repo_root / "artifacts" / "demo" / "data", repo_root / "artifacts" / "demo" / "kernels"],
    )
    guard_snapshot = write_guard._backup_guarded_files(repo_root, policy)
    before = write_guard._snapshot_tree(repo_root)

    src_path.write_text("VALUE = 2\n", encoding="utf-8")
    data_path.write_text("id,target\n1,1\n", encoding="utf-8")
    staged_kernel_path.write_text("print('changed')\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=policy,
        stage="test_broad_policy",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert src_path.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert data_path.read_text(encoding="utf-8") == "id,target\n1,0\n"
    assert staged_kernel_path.read_text(encoding="utf-8") == "print('original')\n"


def test_build_repair_write_policy_allows_src_and_denies_data_and_kernels(tmp_path: Path) -> None:
    repo_root = tmp_path
    module_file = repo_root / "src" / "kagglebot" / "autopilot.py"
    agent_dir = repo_root / "artifacts" / "demo" / "runs" / "run-1" / "iter-1" / "agent"
    module_file.parent.mkdir(parents=True, exist_ok=True)

    policy = write_guard.build_repair_write_policy(
        repo_root=repo_root,
        data_dir=repo_root / "artifacts" / "demo" / "data",
        kernels_dir=repo_root / "artifacts" / "demo" / "kernels",
        module_file=module_file,
        extra_allowed_prefixes=[agent_dir],
    )

    assert repo_root in policy.allowed_prefixes
    assert repo_root / "src" in policy.allowed_prefixes
    assert agent_dir in policy.allowed_prefixes
    assert repo_root / "artifacts" / "demo" / "data" in policy.denied_prefixes
    assert repo_root / "artifacts" / "demo" / "kernels" in policy.denied_prefixes


def test_repo_root_write_policy_rejects_sensitive_repo_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    env_path = repo_root / ".env.local"
    env_path.write_text("TOKEN=old\n", encoding="utf-8")
    policy = write_guard._repo_root_write_policy(
        repo_root=repo_root,
        denied_prefixes=[repo_root / "artifacts" / "demo" / "data", repo_root / "artifacts" / "demo" / "kernels"],
    )
    guard_snapshot = write_guard._backup_guarded_files(repo_root, policy)
    before = write_guard._snapshot_tree(repo_root)

    env_path.write_text("TOKEN=new\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=policy,
        stage="test_sensitive_repo_path",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert env_path.read_text(encoding="utf-8") == "TOKEN=old\n"


def test_guard_rejects_external_sensitive_path_edits(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    tracked = repo_root / "src" / "demo.py"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    external_path = tmp_path / "outside" / "kaggle.json"
    external_path.parent.mkdir(parents=True, exist_ok=True)
    external_path.write_text('{"username":"demo"}\n', encoding="utf-8")

    policy = write_guard.WriteGuardPolicy(
        allowed_prefixes=(repo_root,),
        external_guard_paths=(external_path,),
    )
    guard_snapshot = write_guard._backup_guarded_files(repo_root, policy)
    before = write_guard._snapshot_tree(repo_root)

    external_path.write_text('{"username":"changed"}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    try:
        write_guard._enforce_allowlist_changes(
            root=repo_root,
            before=before,
            after=after,
            allowed_prefixes=policy,
            stage="test_external_guard",
            guard_snapshot=guard_snapshot,
            auto_repair=True,
        )
    except KaggleBotError as exc:
        assert "forbidden external path edited" in str(exc)
        assert str(external_path) in str(exc)
    else:
        raise AssertionError("expected forbidden external path edit to be rejected")


def test_auto_strategy_engine_requires_oracle() -> None:
    config = agent_pipeline.AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=Path("."),
        strategy_engine="auto",
    )

    assert agent_pipeline._strategy_engine_is_required(config, "oracle") is True  # noqa: SLF001
