from __future__ import annotations

from types import SimpleNamespace

from kagglebot.knowledge_phase import KnowledgePhase


def test_knowledge_phase_delegates_refresh_and_profile_loading(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    paths = SimpleNamespace(
        slug="demo",
        dataset_profile_path=tmp_path / "dataset_profile.json",
    )
    knowledge_paths = SimpleNamespace(root=tmp_path / "knowledge")
    config = SimpleNamespace(paths=paths, knowledge_paths=knowledge_paths)

    def fake_refresh_knowledge_hints(*, paths, knowledge_paths):  # noqa: ANN001
        calls["refresh_paths"] = paths
        calls["refresh_knowledge_paths"] = knowledge_paths

    monkeypatch.setattr("kagglebot.knowledge_context.refresh_knowledge_hints", fake_refresh_knowledge_hints)
    monkeypatch.setattr(
        "kagglebot.context_artifacts.load_dataset_profile",
        lambda *, slug, dataset_profile_path: {"slug": slug, "path": str(dataset_profile_path)},
    )
    monkeypatch.setattr(
        "kagglebot.knowledge_context.resolve_problem_types_from_profile",
        lambda *, dataset_profile_path: [f"profile:{dataset_profile_path.name}"],
    )

    phase = KnowledgePhase(config=config)
    phase.refresh()

    assert calls == {"refresh_paths": paths, "refresh_knowledge_paths": knowledge_paths}
    assert phase.load_dataset_profile() == {"slug": "demo", "path": str(paths.dataset_profile_path)}
    assert phase.derive_problem_types() == ["profile:dataset_profile.json"]
