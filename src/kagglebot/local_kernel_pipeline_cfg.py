from __future__ import annotations

import json
from pathlib import Path

KERNEL_PIPELINE_CFG_MARKER = "# kagglebot:pipeline_cfg_fallback"
STAGED_PLAN_PATH_MARKER = "# kagglebot:staged_plan_path_fallback"
STAGED_PLAN_PAYLOAD_MARKER = "# kagglebot:staged_plan_payload_fallback"


def inject_staged_plan_payload_fallback(kernel_dir: Path) -> None:
    """Embed the staged plan because Kaggle does not expose package sidecars beside the running script."""
    kernel_path = kernel_dir / "kernel.py"
    plan_path = kernel_dir / "plan.json"
    if not kernel_path.exists() or not plan_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if STAGED_PLAN_PAYLOAD_MARKER in text:
        return
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot embed invalid staged plan: {plan_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Cannot embed non-object staged plan: {plan_path}")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

    block = [
        STAGED_PLAN_PAYLOAD_MARKER,
        "from pathlib import Path as _KBPlanPath",
        '_KB_PLAN_PATH = _KBPlanPath("/kaggle/working/plan.json")',
        "if not _KB_PLAN_PATH.exists():",
        "    _KB_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)",
        f'    _KB_PLAN_PATH.write_text({serialized!r}, encoding="utf-8")',
        "del _KBPlanPath, _KB_PLAN_PATH",
        "",
    ]
    lines = text.splitlines()
    insert_at = 0
    while insert_at < len(lines) and (
        lines[insert_at].startswith("#!")
        or lines[insert_at].startswith("# -*-")
        or lines[insert_at].startswith("# coding")
        or lines[insert_at].startswith("from __future__ import ")
    ):
        insert_at += 1
    lines[insert_at:insert_at] = block
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def inject_staged_plan_path_fallback(kernel_dir: Path) -> None:
    """Let packaged kernels load the plan snapshot staged beside ``kernel.py``."""
    kernel_path = kernel_dir / "kernel.py"
    plan_path = kernel_dir / "plan.json"
    if not kernel_path.exists() or not plan_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if STAGED_PLAN_PATH_MARKER in text or "KERNEL_DIR" not in text:
        return

    lines = text.splitlines()
    changed = False
    kernel_dir_defined = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("KERNEL_DIR ="):
            kernel_dir_defined = True
        plan_path_name = next(
            (name for name in ("PLAN_PATH", "FROZEN_PLAN_PATH") if stripped.startswith(f"{name} =")),
            None,
        )
        if plan_path_name is None or "plan.json" not in stripped:
            continue
        if not kernel_dir_defined:
            return
        indent = line[: len(line) - len(line.lstrip())]
        fallback = [
            f"{indent}{STAGED_PLAN_PATH_MARKER}",
            f'{indent}if not {plan_path_name}.exists() and (KERNEL_DIR / "plan.json").exists():',
            f'{indent}    {plan_path_name} = KERNEL_DIR / "plan.json"',
            f'{indent}if not {plan_path_name}.exists() and (KERNEL_DIR / "/kaggle/working/plan.json").exists():',
            f'{indent}    {plan_path_name} = KERNEL_DIR / "/kaggle/working/plan.json"',
        ]
        lines[idx + 1 : idx + 1] = fallback
        changed = True
        break
    if not changed:
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("for path in ("):
                continue
            candidate_block = "\n".join(lines[idx : idx + 8])
            if "plan.json" not in candidate_block:
                continue
            if "/kaggle/working/plan.json" in candidate_block:
                continue
            indent = line[: len(line) - len(line.lstrip())]
            marker = f"{indent}{STAGED_PLAN_PATH_MARKER}"
            if stripped == "for path in (":
                lines[idx:idx] = [marker]
                lines[idx + 2 : idx + 2] = [f'{indent}    Path("/kaggle/working/plan.json"),']
            else:
                prefix = line[: line.index("(") + 1]
                suffix = line[line.index("(") + 1 :]
                lines[idx] = f'{prefix}Path("/kaggle/working/plan.json"), {suffix}'
                lines[idx:idx] = [marker]
            changed = True
            break
    if not changed:
        return

    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def inject_pipeline_cfg_fallback(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if KERNEL_PIPELINE_CFG_MARKER in text:
        return

    lines = text.splitlines()
    changed = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("raise KeyError("):
            continue
        if "Pipeline not found in plan" not in stripped:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        replacement = [
            f"{indent}{KERNEL_PIPELINE_CFG_MARKER}",
            f"{indent}return {{",
            f'{indent}    "name": str(name),',
            f'{indent}    "features": [],',
            f'{indent}    "models": [str(name)],',
            f'{indent}    "key_hyperparameters": {{}},',
            f'{indent}    "runtime_memory": "unknown",',
            f'{indent}    "failure_modes": ["missing_pipeline_in_plan"],',
            f'{indent}    "fallbacks": ["use_default_pipeline_behavior"],',
            f"{indent}}}",
        ]
        lines[idx : idx + 1] = replacement
        changed = True
        break
    if not changed:
        return

    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")
