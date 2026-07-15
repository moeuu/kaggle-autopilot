from __future__ import annotations

from pathlib import Path

from kagglebot.paths import CompetitionPaths, resolve_agent_repository_root


def test_agent_repository_root_falls_back_from_invalid_cli_cwd(monkeypatch, tmp_path: Path) -> None:
    artifact_workdir = tmp_path / "artifact-workdir"
    artifact_workdir.mkdir()
    monkeypatch.chdir(artifact_workdir)

    resolved = resolve_agent_repository_root(artifact_workdir)
    paths = CompetitionPaths(
        slug="demo",
        artifacts_dir=artifact_workdir / "kaggle-autopilot-artifacts",
        repo_root=artifact_workdir,
    )

    assert (resolved / "src" / "kagglebot").is_dir()
    assert paths.repo_root == resolved
    assert paths.artifacts_dir == artifact_workdir / "kaggle-autopilot-artifacts"


def test_agent_repository_root_preserves_explicit_embedding_root(tmp_path: Path) -> None:
    explicit_root = tmp_path / "embedding-root"

    assert resolve_agent_repository_root(explicit_root) == explicit_root.resolve()
