from __future__ import annotations

import os
from pathlib import Path

from kagglebot import local_kernel_limits
from kagglebot.env_utils import parse_bool_value

_LOCAL_LGBM_GPU_PROBE_OK: bool | None = None


def env_truthy(raw: str | None) -> bool:
    return parse_bool_value(raw, default=False)


def module_available(module_name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def local_lightgbm_gpu_probe_usable() -> bool:
    global _LOCAL_LGBM_GPU_PROBE_OK
    if _LOCAL_LGBM_GPU_PROBE_OK is not None:
        return _LOCAL_LGBM_GPU_PROBE_OK
    if env_truthy(os.environ.get("KAGGLEBOT_SKIP_LGBM_GPU_PROBE")):
        _LOCAL_LGBM_GPU_PROBE_OK = False
        return False
    try:
        import lightgbm as lgb
        import numpy as np
    except Exception:
        _LOCAL_LGBM_GPU_PROBE_OK = False
        return False

    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 4)).astype(np.float32)
    y = (0.4 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2]).astype(np.float32)
    try:
        model = lgb.LGBMRegressor(
            n_estimators=16,
            learning_rate=0.1,
            num_leaves=15,
            max_depth=5,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            device_type="gpu",
            verbosity=-1,
        )
        model.fit(x, y)
    except Exception:
        _LOCAL_LGBM_GPU_PROBE_OK = False
        return False
    _LOCAL_LGBM_GPU_PROBE_OK = True
    return True


def apply_local_runtime_env_defaults(
    *,
    env: dict[str, str],
    accelerator: str,
    local_working_dir: Path,
) -> list[str]:
    """Apply local execution defaults and force training to stay enabled."""
    notes: list[str] = []
    env.setdefault("KAGGLEBOT_LOCAL_WORKING_DIR", str(local_working_dir))
    env.setdefault("KAGGLEBOT_DISABLE_KAGGLE_WORKING_WRITES", "1")
    env.setdefault("KAGGLEBOT_NUM_WORKERS", "0")
    env.setdefault("KAGGLEBOT_TORCH_SHARING_STRATEGY", "file_system")
    env.setdefault("KAGGLEBOT_LOCAL_NOFILE", "4096")
    env.setdefault(local_kernel_limits.STALL_ENV, str(int(local_kernel_limits.DEFAULT_STALL_SEC)))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["KAGGLEBOT_DO_TRAIN"] = "1"
    env["KAGGLEBOT_FORCE_TRAIN"] = "1"
    env["KAGGLEBOT_ALLOW_MODEL_DOWNLOAD"] = "1"
    notes.append("forcing KAGGLEBOT_DO_TRAIN=1 and KAGGLEBOT_FORCE_TRAIN=1")
    notes.append("forcing KAGGLEBOT_ALLOW_MODEL_DOWNLOAD=1")
    notes.append(f"defaulting KAGGLEBOT_NUM_WORKERS={env['KAGGLEBOT_NUM_WORKERS']} for local kernels")
    notes.append(f"defaulting KAGGLEBOT_TORCH_SHARING_STRATEGY={env['KAGGLEBOT_TORCH_SHARING_STRATEGY']}")
    notes.append(f"defaulting KAGGLEBOT_LOCAL_NOFILE={env['KAGGLEBOT_LOCAL_NOFILE']}")
    notes.append(f"defaulting {local_kernel_limits.STALL_ENV}={env[local_kernel_limits.STALL_ENV]}")
    notes.append(f"defaulting PYTHONUNBUFFERED={env['PYTHONUNBUFFERED']}")

    if not module_available("xgboost"):
        env.setdefault("USE_XGB", "0")
        env.setdefault("KAGGLEBOT_DISABLE_XGBOOST", "1")
        notes.append("xgboost unavailable; forcing USE_XGB=0")

    force_lgbm_gpu = env_truthy(os.environ.get("KAGGLEBOT_FORCE_LGBM_GPU"))
    if accelerator == "gpu" and not force_lgbm_gpu and not local_lightgbm_gpu_probe_usable():
        env.setdefault("USE_LGBM_GPU", "0")
        env.setdefault("KAGGLEBOT_DISABLE_LGBM_GPU", "1")
        notes.append("LightGBM GPU probe failed; forcing CPU LightGBM")
    return notes


def detect_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    if "out of memory" not in lowered:
        return False
    if "cuda" in lowered:
        return True
    if "cublas_status_alloc_failed" in lowered:
        return True
    if "hiperroroutofmemory" in lowered:
        return True
    if "mps" in lowered and "out of memory" in lowered:
        return True
    return False


def apply_local_kernel_oom_fallback_env(env: dict[str, str]) -> list[str]:
    notes: list[str] = []
    env["ENABLE_LLM"] = "0"
    env["PIPELINE_NAME"] = "retrieval_only_baseline"
    env["ENABLE_SELF_CONSIST"] = "0"
    env["SAVE_INTERMEDIATE"] = "0"
    notes.append("CUDA OOM detected; retrying with ENABLE_LLM=0 and retrieval_only_baseline")
    return notes
