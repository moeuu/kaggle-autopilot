from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

from kagglebot.runtime_policy import DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    label: str
    gpu_name: str
    gpu_count: int
    vram_gb: int
    system_ram_gb: int | None
    time_budget_min: int | None
    tier: str
    notes: tuple[str, ...]


_PROFILES: dict[str, HardwareProfile] = {
    "rtx3060": HardwareProfile(
        key="rtx3060",
        label="NVIDIA GeForce RTX 3060 12GB",
        gpu_name="RTX 3060",
        gpu_count=1,
        vram_gb=12,
        system_ram_gb=None,
        time_budget_min=DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN,
        tier="local_12gb",
        notes=(
            "Primary local planning target. Accuracy is the default priority; long runs are acceptable.",
            "Keep strong pretrained/OCR/VLM paths alive and reduce batch size, resolution, chunks, "
            "or candidate ordering before dropping high-ceiling candidates.",
        ),
    ),
    "rtx3090": HardwareProfile(
        key="rtx3090",
        label="NVIDIA GeForce RTX 3090 24GB",
        gpu_name="RTX 3090",
        gpu_count=1,
        vram_gb=24,
        system_ram_gb=None,
        time_budget_min=DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN,
        tier="local_24gb",
        notes=(
            "Can roughly double batch sizes or candidate counts versus RTX 3060, "
            "but still avoid unbounded rerank grids.",
        ),
    ),
    "rtx4090": HardwareProfile(
        key="rtx4090",
        label="NVIDIA GeForce RTX 4090 24GB",
        gpu_name="RTX 4090",
        gpu_count=1,
        vram_gb=24,
        system_ram_gb=None,
        time_budget_min=DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN,
        tier="local_24gb_fast",
        notes=("Use larger batches than RTX 3060; keep the same algorithmic caps unless explicitly scaling up.",),
    ),
    "rtx5090": HardwareProfile(
        key="rtx5090",
        label="NVIDIA GeForce RTX 5090 32GB",
        gpu_name="RTX 5090",
        gpu_count=1,
        vram_gb=32,
        system_ram_gb=None,
        time_budget_min=DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN,
        tier="local_32gb_fast",
        notes=("High-end local target. Scaling should be via profile knobs, not hard-coded model rewrites.",),
    ),
    "kaggle_p100": HardwareProfile(
        key="kaggle_p100",
        label="Kaggle GPU P100 16GB",
        gpu_name="P100",
        gpu_count=1,
        vram_gb=16,
        system_ram_gb=None,
        time_budget_min=720,
        tier="kaggle_16gb",
        notes=("Kaggle GPU runs must respect notebook session and quota limits; do not rely on multi-day jobs.",),
    ),
    "kaggle_t4": HardwareProfile(
        key="kaggle_t4",
        label="Kaggle GPU T4 16GB",
        gpu_name="T4",
        gpu_count=1,
        vram_gb=16,
        system_ram_gb=None,
        time_budget_min=720,
        tier="kaggle_16gb",
        notes=("Single T4 target. Prefer inference-friendly batches and cached features.",),
    ),
    "kaggle_t4x2": HardwareProfile(
        key="kaggle_t4x2",
        label="Kaggle GPU T4 x2 16GB each",
        gpu_name="T4 x2",
        gpu_count=2,
        vram_gb=16,
        system_ram_gb=None,
        time_budget_min=720,
        tier="kaggle_dual_16gb",
        notes=(
            "Do not assume speedup unless kernel.py explicitly uses multi-GPU inference/training.",
            "Per-device memory is still 16GB.",
        ),
    ),
    "kaggle_rtx_pro_6000": HardwareProfile(
        key="kaggle_rtx_pro_6000",
        label="Kaggle RTX PRO 6000 Blackwell 96GB (g4-standard-48)",
        gpu_name="RTX PRO 6000 Blackwell",
        gpu_count=1,
        vram_gb=96,
        system_ram_gb=180,
        time_budget_min=720,
        tier="kaggle_96gb",
        notes=(
            "Competition-restricted accelerator: use only for ARC-AGI-3 notebooks attached to that competition.",
            "Internet must remain disabled for every RTX PRO 6000 session.",
        ),
    ),
    "generic_12gb": HardwareProfile(
        key="generic_12gb",
        label="Generic single GPU 12GB",
        gpu_name="generic",
        gpu_count=1,
        vram_gb=12,
        system_ram_gb=None,
        time_budget_min=DEFAULT_LOCAL_GPU_TIME_BUDGET_MIN,
        tier="local_12gb",
        notes=(
            "Fallback local GPU profile. Treat as 12GB accuracy-first: keep high-ceiling pretrained paths alive, "
            "then scale batch/chunk sizes down if needed.",
        ),
    ),
}


def known_hardware_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def resolve_hardware_profile(name: str | None, *, compute: str) -> HardwareProfile:
    requested = _normalize_key(name or os.environ.get("KAGGLEBOT_HARDWARE_PROFILE") or "auto")
    if requested != "auto":
        try:
            return _PROFILES[requested]
        except KeyError as exc:
            allowed = ", ".join(("auto", *known_hardware_profiles()))
            raise ValueError(f"Unknown hardware profile '{name}'. Allowed values: {allowed}.") from exc

    detected = _detect_local_nvidia_profile()
    if detected is not None:
        return detected
    if str(compute).strip().lower() == "kaggle_gpu":
        return _PROFILES["kaggle_p100"]
    if str(compute).strip().lower() == "local_gpu":
        return _PROFILES["generic_12gb"]
    return _PROFILES["generic_12gb"]


def render_hardware_constraints(profile: HardwareProfile, *, compute: str, time_budget_min: int | None = None) -> str:
    budget = time_budget_min if time_budget_min is not None else profile.time_budget_min
    budget_text = "unlimited" if budget is None else f"<= {int(budget)} minutes"
    lines = [
        f"- Hardware profile: {profile.label} (`{profile.key}`)",
        f"- Compute mode: {compute}",
        f"- GPU count: {profile.gpu_count}",
        f"- Per-GPU VRAM budget: {profile.vram_gb}GB",
        f"- Planning time budget: {budget_text} per kernel iteration",
        f"- Runtime tier: {profile.tier}",
        "- Required design: expose scale knobs in `plan.json`/environment so switching profiles does not require "
        "rewriting kernel.py.",
        "- Required fallback: if CUDA OOM or ETA exceeds budget, reduce batch size, top-k, folds/seeds, or rerank "
        "candidates before dropping correctness checks.",
    ]
    if profile.key in {"rtx3060", "generic_12gb"}:
        lines.extend(
            [
                "- RTX3060-class accuracy-first rule: do not disable the strongest OCR/VLM/transformer path solely "
                "because the GPU has 12GB VRAM.",
                "- RTX3060-class scaling rule: prefer smaller batches, lower-but-useful resolution, chunking, "
                "4-bit/quantized loading, cached embeddings, or sequential candidates before dropping model families.",
                "- RTX3060-class multimodal rule: never tokenizer-truncate expanded image/video tokens. Preflight one "
                "real training batch and make the first rung use a feasible pixel/view budget; reduce visual tokens "
                "before changing model families.",
                "- RTX3060-class OOM-ladder rule: before loading the next rung in one Python process, clear exception "
                "tracebacks that can retain partial CUDA models, release optimizer/batch references, run garbage "
                "collection, and empty the CUDA cache.",
                "- RTX3060-class runtime rule: long local runs are acceptable when they keep a materially stronger "
                "candidate alive; use watchdogs/checkpoints instead of replacing it with a weak geometry baseline. "
                "Benchmark an early real train/inference unit and extrapolate every fold, epoch, and test row. If the "
                "projected end-to-end ETA exceeds the hard iteration budget, stop that candidate and finalize the "
                "strongest already-completed competition-faithful learned baseline so the run still submits.",
            ]
        )
    lines.extend(f"- Note: {note}" for note in profile.notes)
    return "\n".join(lines)


def hardware_env(profile: HardwareProfile) -> dict[str, str]:
    return {
        "KAGGLEBOT_HARDWARE_PROFILE": profile.key,
        "KAGGLEBOT_GPU_PROFILE": profile.key,
        "KAGGLEBOT_GPU_NAME": profile.gpu_name,
        "KAGGLEBOT_GPU_COUNT": str(profile.gpu_count),
        "KAGGLEBOT_GPU_VRAM_GB": str(profile.vram_gb),
        "KAGGLEBOT_RUNTIME_TIER": profile.tier,
        "KAGGLEBOT_ACCURACY_FIRST": "1",
        "KAGGLEBOT_RELAX_RTX3060_CONSTRAINTS": "1",
    }


def _normalize_key(value: str) -> str:
    lowered = value.strip().lower()
    lowered = lowered.replace("nvidia", "")
    lowered = lowered.replace("geforce", "")
    lowered = lowered.replace("gpu", "")
    lowered = re.sub(r"[^a-z0-9]+", "", lowered)
    aliases = {
        "3060": "rtx3060",
        "rtx306012gb": "rtx3060",
        "3090": "rtx3090",
        "4090": "rtx4090",
        "5090": "rtx5090",
        "p100": "kaggle_p100",
        "kagglep100": "kaggle_p100",
        "teslap100": "kaggle_p100",
        "t4": "kaggle_t4",
        "kagglet4": "kaggle_t4",
        "t4x2": "kaggle_t4x2",
        "2xt4": "kaggle_t4x2",
        "kagglet4x2": "kaggle_t4x2",
        "rtxpro6000": "kaggle_rtx_pro_6000",
        "rtxpro6000blackwell": "kaggle_rtx_pro_6000",
        "kagglertxpro6000": "kaggle_rtx_pro_6000",
        "generic12gb": "generic_12gb",
        "auto": "auto",
    }
    return aliases.get(lowered, lowered)


def _detect_local_nvidia_profile() -> HardwareProfile | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None
    parts = [part.strip() for part in first_line.split(",")]
    gpu_name = parts[0]
    memory_mb = _parse_int(parts[1]) if len(parts) > 1 else None
    normalized_name = _normalize_key(gpu_name)
    if normalized_name in _PROFILES:
        return _PROFILES[normalized_name]
    if memory_mb is not None:
        memory_gb = max(1, int(round(memory_mb / 1024)))
        if memory_gb <= 13:
            return _PROFILES["generic_12gb"]
        if memory_gb <= 18:
            return _PROFILES["kaggle_p100"]
        if memory_gb <= 26:
            return _PROFILES["rtx4090"]
        return _PROFILES["rtx5090"]
    return None


def _parse_int(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except ValueError:
        return None
