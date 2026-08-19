from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.competition_artifact


def _load_deep_past_kernel_module():
    kernel_path = (
        Path(__file__).resolve().parent.parent
        / "artifacts"
        / "deep-past-initiative-machine-translation"
        / "kernel"
        / "kernel.py"
    )
    spec = importlib.util.spec_from_file_location("deep_past_kernel_sources_test", kernel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (path / filename).write_text("x", encoding="utf-8")
    return path


def test_deep_past_kernel_reads_required_seq2seq_sources(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_MODEL_PATHS", raising=False)
    monkeypatch.delenv("KAGGLEBOT_MODEL_PATHS_POOLED_MULTI_BYT5_MBR", raising=False)

    module = _load_deep_past_kernel_module()

    assert "pooled_multi_byt5_mbr" in module._required_local_seq2seq_pipeline_names()
    hints = module._plan_pipeline_model_hints("pooled_multi_byt5_mbr")
    assert "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6" in hints
    assert "google/byt5-base" in hints


def test_deep_past_kernel_prefers_writable_kaggle_output_dir(monkeypatch, tmp_path: Path) -> None:
    module = _load_deep_past_kernel_module()
    created: list[Path] = []

    def fake_ensure_dir(path: Path) -> Path:
        created.append(path)
        if path in {Path("/kaggle/working/output"), tmp_path / "output"}:
            return path
        raise OSError(f"read-only: {path}")

    monkeypatch.setattr(module, "IS_KAGGLE", True)
    monkeypatch.setattr(module, "KERNEL_DIR", Path("/kaggle/src"))
    monkeypatch.setattr(module, "ARTIFACT_DIR", Path("/artifact-root"))
    monkeypatch.setattr(module, "ensure_dir", fake_ensure_dir)
    monkeypatch.setattr(module, "is_writable_dir", lambda path: path == Path("/kaggle/working"))
    monkeypatch.chdir(tmp_path)

    primary, output_dirs, kaggle_writable = module.resolve_output_dirs("demo", "run-1")

    assert primary == Path("/kaggle/working/output")
    assert Path("/kaggle/src/output") not in created
    assert output_dirs == [Path("/kaggle/working/output"), Path("/kaggle/working")]
    assert kaggle_writable is True
    assert module._metric_output_dirs(output_dirs) == output_dirs


def test_reference_model_candidates_treat_empty_exact_asset_as_blocker(monkeypatch, tmp_path: Path) -> None:
    module = _load_deep_past_kernel_module()
    exact_dir = tmp_path / "dataset__mattiaangeli__byt5-akkadian-mbr__PyTorch__default__6"
    exact_dir.mkdir(parents=True, exist_ok=True)
    alternate_dir = _write_fake_checkpoint(tmp_path / "alternate-mattia" / "checkpoint")
    hint = "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6"

    monkeypatch.setattr(module, "REFERENCE_EXACT_MODEL_ASSET_PATHS", {hint: (exact_dir,)})
    monkeypatch.setattr(module, "_resolve_model_sources", lambda _hint: [alternate_dir])

    candidates, blockers = module._reference_model_candidates(hint)

    assert candidates == []
    assert any("local asset exists but is empty" in message for message in blockers)


def test_reference_model_candidates_reject_owner_slug_symlink_alias(monkeypatch, tmp_path: Path) -> None:
    module = _load_deep_past_kernel_module()
    assiaben_dir = _write_fake_checkpoint(tmp_path / "dataset__assiaben__final-byt5" / "byt5-akkadian-optimized-34x")
    alias_dir = tmp_path / "mattiaangeli_byt5_akkadian_mbr_pytorch_default_6"
    alias_dir.symlink_to(assiaben_dir, target_is_directory=True)
    hint = "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6"

    monkeypatch.setattr(module, "REFERENCE_EXACT_MODEL_ASSET_PATHS", {})
    monkeypatch.setattr(module, "_resolve_model_sources", lambda _hint: [alias_dir])

    candidates, blockers = module._reference_model_candidates(hint)

    assert candidates == []
    assert blockers == []
    assert module._source_matches_model_hint(alias_dir, hint) is False


def test_prepare_reference_baseline_cfg_uses_strongest_cached_fallback_pair_after_empty_exact_asset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_deep_past_kernel_module()
    cfg = module.get_pipeline_cfg("dual_checkpoint_public_mbr")
    assiaben_dir = _write_fake_checkpoint(tmp_path / "dataset__assiaben__final-byt5" / "checkpoint")
    artem_dir = _write_fake_checkpoint(tmp_path / "dataset__artemgoncarov__dpc-byt5-large" / "checkpoint")

    def fake_candidates(hint: str) -> tuple[list[str], list[str]]:
        mapping = {
            "assiaben/final-byt5": ([str(assiaben_dir)], []),
            "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6": (
                [],
                ["mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6 local asset exists but is empty"],
            ),
            "artemgoncarov/dpc-byt5-large": ([str(artem_dir)], []),
        }
        return mapping[hint]

    monkeypatch.setattr(module, "_reference_model_candidates", fake_candidates)
    monkeypatch.setattr(
        module,
        "_reference_cached_checkpoint_candidates",
        lambda: [str(assiaben_dir), str(artem_dir)],
    )

    resolved = module._prepare_reference_baseline_cfg(cfg)

    assert resolved.reference_runtime_mode == "competition_faithful_fallback_pair"
    assert resolved.model_hints == [str(assiaben_dir), str(artem_dir)]
    assert resolved.reference_slot_meta is not None
    assert resolved.reference_slot_meta[0]["resolved_source_path"] == str(assiaben_dir)
    assert resolved.reference_slot_meta[1]["resolved_source_path"] == str(artem_dir)


def test_prepare_reference_baseline_cfg_rejects_duplicate_dual_pair_and_blocks_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_deep_past_kernel_module()
    cfg = module.get_pipeline_cfg("dual_checkpoint_public_mbr")
    assiaben_dir = _write_fake_checkpoint(tmp_path / "dataset__assiaben__final-byt5" / "checkpoint")

    def fake_candidates(hint: str) -> tuple[list[str], list[str]]:
        mapping = {
            "assiaben/final-byt5": ([str(assiaben_dir)], []),
            "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6": (
                [],
                ["mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6 local asset exists but is empty"],
            ),
            "artemgoncarov/dpc-byt5-large": ([], ["artem blocked"]),
        }
        return mapping[hint]

    monkeypatch.setattr(module, "_reference_model_candidates", fake_candidates)
    monkeypatch.setattr(module, "_reference_cached_checkpoint_candidates", lambda: [str(assiaben_dir)])

    resolved = module._prepare_reference_baseline_cfg(cfg)

    assert resolved.reference_runtime_mode == "blocked_reference_runtime"
    assert resolved.model_hints == []
    assert resolved.reference_slot_meta is None


def test_pipeline_model_hints_preserve_pinned_reference_paths_over_env_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_deep_past_kernel_module()
    cfg = module.get_pipeline_cfg("dual_checkpoint_public_mbr")
    model_a = _write_fake_checkpoint(tmp_path / "slot-a")
    model_b = _write_fake_checkpoint(tmp_path / "slot-b")
    env_override = _write_fake_checkpoint(tmp_path / "env-override")
    resolved = module._reference_runtime_cfg(
        cfg,
        original_hints=["assiaben/final-byt5", "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6"],
        resolved_sources=[str(model_a), str(model_b)],
        runtime_tokens=["exact_public_pair"],
        runtime_mode="exact_required_public_pair",
        blocker_messages=[],
        use_multi_model_pool=True,
        use_mbr=True,
    )
    monkeypatch.setenv("KAGGLEBOT_MODEL_PATHS_DUAL_CHECKPOINT_PUBLIC_MBR", str(env_override))
    monkeypatch.setenv("KAGGLEBOT_MODEL_PATHS", str(env_override))

    hints = module._pipeline_model_hints(resolved)

    assert hints == [str(model_a), str(model_b)]


def test_reference_model_candidates_allow_valid_google_byt5_large_checkpoint(monkeypatch, tmp_path: Path) -> None:
    module = _load_deep_past_kernel_module()
    large_dir = _write_fake_checkpoint(tmp_path / "local-iter-1" / "models" / "google--byt5-large")

    monkeypatch.setattr(module, "REFERENCE_EXACT_MODEL_ASSET_PATHS", {})
    monkeypatch.setattr(module, "_resolve_model_sources", lambda _hint: [large_dir])

    candidates, blockers = module._reference_model_candidates(str(large_dir))

    assert candidates == [str(large_dir.resolve())]
    assert blockers == []


def test_resolve_final_seq2seq_cfg_reuses_runtime_cfg_for_reference_ablation(tmp_path: Path) -> None:
    module = _load_deep_past_kernel_module()
    pinned_dir = _write_fake_checkpoint(tmp_path / "slot-a")
    runtime_cfg = module.PipelineConfig(
        **{
            **module.get_pipeline_cfg("dual_checkpoint_public_mbr").__dict__,
            "name": "dual_checkpoint_public_mbr__single_model_ablation_slot_a",
            "model_hints": [str(pinned_dir)],
            "use_multi_model_pool": False,
            "use_mbr": False,
            "reference_runtime_mode": "single_model_seq2seq_ablation:slot_a",
            "reference_slot_meta": [
                {
                    "original_model_hint": "assiaben/final-byt5",
                    "resolved_source_path": str(pinned_dir),
                    "canonical_source_id": str(pinned_dir.resolve()),
                }
            ],
        }
    )
    chosen = module.PipelineResult(
        name=runtime_cfg.name,
        cv_score=0.1,
        bleu=0.1,
        chrfpp=0.1,
        complexity_rank=runtime_cfg.complexity_rank,
        oof_predictions=np.zeros((1, 1), dtype=object),
        test_predictions=np.zeros((1, 1), dtype=object),
        best_seed=42,
        executed_checkpoints=[str(pinned_dir)],
    )

    resolved = module._resolve_final_seq2seq_cfg(
        chosen,
        {runtime_cfg.name: runtime_cfg},
        accepted_finetune_adapter_dir=None,
        local_budget_skip_reason="watchdog",
    )

    assert resolved.model_hints == [str(pinned_dir)]
    assert resolved.reference_runtime_mode == "single_model_seq2seq_ablation:slot_a"
