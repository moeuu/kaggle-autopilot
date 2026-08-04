from __future__ import annotations

from pathlib import Path

import pytest

from kagglebot import local_kernel_runtime_env


def test_env_truthy_uses_shared_bool_parser() -> None:
    assert local_kernel_runtime_env.env_truthy("y") is True
    assert local_kernel_runtime_env.env_truthy("n") is False
    assert local_kernel_runtime_env.env_truthy("maybe") is False
    assert local_kernel_runtime_env.env_truthy(None) is False


def test_apply_local_runtime_env_defaults_sets_optional_backend_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        local_kernel_runtime_env,
        "module_available",
        lambda name: False if name == "xgboost" else True,
    )
    monkeypatch.setattr(local_kernel_runtime_env, "local_lightgbm_gpu_probe_usable", lambda: False)
    monkeypatch.delenv("KAGGLEBOT_FORCE_LGBM_GPU", raising=False)

    env: dict[str, str] = {}
    notes = local_kernel_runtime_env.apply_local_runtime_env_defaults(
        env=env,
        accelerator="gpu",
        local_working_dir=tmp_path / "local-working",
    )

    assert env["KAGGLEBOT_DISABLE_KAGGLE_WORKING_WRITES"] == "1"
    assert env["KAGGLEBOT_LOCAL_WORKING_DIR"] == str(tmp_path / "local-working")
    assert env["KAGGLEBOT_NUM_WORKERS"] == "0"
    assert env["KAGGLEBOT_TORCH_SHARING_STRATEGY"] == "file_system"
    assert env["KAGGLEBOT_LOCAL_NOFILE"] == "4096"
    assert env["KAGGLEBOT_LOCAL_KERNEL_STALL_SEC"] == "900"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["KAGGLEBOT_DO_TRAIN"] == "1"
    assert env["KAGGLEBOT_FORCE_TRAIN"] == "1"
    assert env["KAGGLEBOT_ALLOW_MODEL_DOWNLOAD"] == "1"
    assert env["USE_XGB"] == "0"
    assert env["KAGGLEBOT_DISABLE_XGBOOST"] == "1"
    assert env["USE_LGBM_GPU"] == "0"
    assert env["KAGGLEBOT_DISABLE_LGBM_GPU"] == "1"
    assert any("xgboost unavailable" in note for note in notes)
    assert any("LightGBM GPU probe failed" in note for note in notes)
    assert any("KAGGLEBOT_ALLOW_MODEL_DOWNLOAD=1" in note for note in notes)
    assert any("KAGGLEBOT_NUM_WORKERS=0" in note for note in notes)
    assert any("KAGGLEBOT_TORCH_SHARING_STRATEGY=file_system" in note for note in notes)
    assert any("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in note for note in notes)


def test_detect_cuda_oom_matches_gpu_memory_errors() -> None:
    assert local_kernel_runtime_env.detect_cuda_oom("RuntimeError: CUDA out of memory") is True
    assert local_kernel_runtime_env.detect_cuda_oom("CUBLAS_STATUS_ALLOC_FAILED out of memory") is True
    assert local_kernel_runtime_env.detect_cuda_oom("plain CPU out of memory") is False


def test_llm_disable_fallback_requires_explicit_kernel_opt_in() -> None:
    assert local_kernel_runtime_env.kernel_source_supports_llm_disable_fallback(
        "KAGGLEBOT_SUPPORTS_LLM_DISABLE_FALLBACK = True\nenabled = os.getenv('ENABLE_LLM', '1')"
    )
    assert not local_kernel_runtime_env.kernel_source_supports_llm_disable_fallback(
        "enabled = os.getenv('ENABLE_LLM', '1')"
    )


def test_apply_local_kernel_oom_fallback_env_disables_llm() -> None:
    env: dict[str, str] = {"ENABLE_LLM": "1"}

    notes = local_kernel_runtime_env.apply_local_kernel_oom_fallback_env(env)

    assert env["ENABLE_LLM"] == "0"
    assert env["PIPELINE_NAME"] == "retrieval_only_baseline"
    assert env["ENABLE_SELF_CONSIST"] == "0"
    assert env["SAVE_INTERMEDIATE"] == "0"
    assert any("CUDA OOM detected" in note for note in notes)
