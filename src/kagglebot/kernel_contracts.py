from __future__ import annotations

import json
import re
from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.json_utils import read_json_object

_BVS_KERNEL_CONTRACT_SLUG_PREFIX = "beyond-visible-spectrum-ai-for-agriculture-2026"
_BVS_TIMM_FAILURE_MARKERS = (
    "timm is unavailable",
    "timm.create_model is missing",
    "skipping tri_branch_timm_gated because timm is unavailable",
    "falling back to smallspectralencoder for rgb",
)
_LOCAL_KERNEL_LOG_NAMES = (
    "local_kernel_stdout.log",
    "local_kernel_stderr.log",
    "local_kernel_stdout_oom_retry.log",
    "local_kernel_stderr_oom_retry.log",
)


def requires_bvs_kernel_contract(slug: str) -> bool:
    return slug.strip().lower().startswith(_BVS_KERNEL_CONTRACT_SLUG_PREFIX)


def collect_local_kernel_log_text(logs_dir: Path) -> str:
    chunks: list[str] = []
    for name in _LOCAL_KERNEL_LOG_NAMES:
        path = logs_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def extract_kernel_size_markers(log_text: str) -> list[int]:
    pattern = re.compile(r"\b(?:load_size|img_size)\s*=\s*(\d+)\b")
    values: list[int] = []
    for match in pattern.finditer(log_text):
        try:
            values.append(int(match.group(1)))
        except ValueError:
            continue
    return values


def enforce_competition_kernel_contract(
    *,
    slug: str,
    logs_dir: Path,
    metrics_path: Path | None,
) -> None:
    """Enforce competition-specific quality contracts to prevent silent regressions."""
    if not requires_bvs_kernel_contract(slug):
        return

    errors: list[str] = []
    payload: dict[str, object] = {}
    if metrics_path is None or not metrics_path.exists():
        errors.append("metrics.json is missing; cannot validate BVS kernel contract.")
    else:
        try:
            payload = read_json_object(metrics_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"metrics.json is unreadable: {exc}")
        except ValueError:
            errors.append("metrics.json payload must be a JSON object.")

    log_text = collect_local_kernel_log_text(logs_dir)
    lowered_log = log_text.lower()
    for marker in _BVS_TIMM_FAILURE_MARKERS:
        if marker in lowered_log:
            errors.append(f"timm/ConvNeXt fallback marker detected in logs: {marker}")

    size_markers = extract_kernel_size_markers(log_text)
    if not size_markers:
        errors.append("No img_size/load_size markers found in local kernel logs.")
    else:
        undersized = sorted({value for value in size_markers if value < 128})
        if undersized:
            errors.append(f"Detected img_size/load_size below 128 in logs: {undersized}")

    if payload:
        model_name = str(payload.get("model_name") or "").strip().lower()
        if model_name in {"resnet50", "small_rgb_encoder", "none"}:
            errors.append(f"Weak fallback backbone detected in metrics model_name={model_name!r}.")

        pipelines = payload.get("pipelines")
        if not isinstance(pipelines, list) or len(pipelines) < 2:
            errors.append("metrics.json must report at least two pipeline candidates for ensemble selection.")

        chosen_pipeline = str(payload.get("chosen_pipeline") or "").strip().lower()
        if not chosen_pipeline:
            errors.append("metrics.json must include chosen_pipeline.")
        elif "ensemble" not in chosen_pipeline:
            errors.append(f"chosen_pipeline must be ensemble-based, got: {chosen_pipeline!r}.")

    if errors:
        issue_text = "\n".join(f"- {message}" for message in errors)
        raise KernelFailedError(f"BVS kernel contract failed (timm/size/ensemble guard):\n{issue_text}")
