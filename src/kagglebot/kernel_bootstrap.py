from __future__ import annotations

import json
import re
from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.hardware import hardware_env, resolve_hardware_profile

KERNEL_BOOTSTRAP_MARKER = "# kagglebot:kernel_sys_path"
KERNEL_BOOTSTRAP_END = "del _os, _sys, _KROOT, _KWORK"
KERNEL_COMPETITION_SLUG_MARKER = "# kagglebot:competition_slug"
KERNEL_HARDWARE_PROFILE_MARKER = "# kagglebot:hardware_profile"
KERNEL_FORCE_TRAIN_MARKER = "# kagglebot:force_train"
KERNEL_SUBMIT_INFERENCE_MARKER = "# kagglebot:submit_inference"


def strip_kernel_bootstrap(lines: list[str]) -> list[str]:
    stripped = lines
    while KERNEL_BOOTSTRAP_MARKER in stripped:
        start = stripped.index(KERNEL_BOOTSTRAP_MARKER)
        end = None
        search_end = min(start + 60, len(stripped))
        for idx in range(start + 1, search_end):
            if stripped[idx].strip() == KERNEL_BOOTSTRAP_END:
                end = idx + 1
                break
        if end is None:
            stripped = stripped[:start] + stripped[start + 1 :]
        else:
            stripped = stripped[:start] + stripped[end:]
    return stripped


def ensure_kernel_import_path(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    bootstrap = (
        f"{KERNEL_BOOTSTRAP_MARKER}\n"
        "import os as _os\n"
        "import sys as _sys\n"
        "try:\n"
        "    _KROOT = _os.path.dirname(_os.path.abspath(__file__))\n"
        "except NameError:\n"
        "    _KROOT = _os.getcwd()\n"
        "if _KROOT not in _sys.path:\n"
        "    _sys.path.insert(0, _KROOT)\n"
        "_KWORK = '/kaggle/working'\n"
        "if _KWORK not in _sys.path:\n"
        "    _sys.path.insert(0, _KWORK)\n"
        "try:\n"
        "    _KSC = _os.path.join(_KROOT, 'sitecustomize.py')\n"
        "    if _os.path.exists(_KSC):\n"
        "        with open(_KSC, 'rb') as _kb_f:\n"
        "            exec(\n"
        "                compile(_kb_f.read(), _KSC, 'exec'),\n"
        "                {'__file__': _KSC, '__name__': 'kagglebot_sitecustomize'},\n"
        "            )\n"
        "except Exception:\n"
        "    pass\n"
        "del _os, _sys, _KROOT, _KWORK\n"
    )
    lines = strip_kernel_bootstrap(text.splitlines())
    insert_at = find_bootstrap_insertion_index(lines)
    bootstrap_lines = bootstrap.splitlines()
    new_lines = lines[:insert_at] + bootstrap_lines + lines[insert_at:]
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    kernel_path.write_text(new_text, encoding="utf-8")


def inject_competition_slug_env(kernel_dir: Path, competition_slug: str) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if KERNEL_COMPETITION_SLUG_MARKER in text:
        return

    slug_literal = json.dumps(str(competition_slug))
    resolver_block = [
        KERNEL_COMPETITION_SLUG_MARKER,
        "import os as _kb_os",
        f"_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = {slug_literal}",
        f"_kb_os.environ['KAGGLEBOT_SLUG'] = {slug_literal}",
        "del _kb_os",
        "",
    ]
    lines = text.splitlines()
    insert_at = find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def inject_hardware_profile_env(kernel_dir: Path, hardware_profile: str | None, *, compute: str) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if KERNEL_HARDWARE_PROFILE_MARKER in text:
        return

    profile = resolve_hardware_profile(hardware_profile, compute=compute)
    env_payload = hardware_env(profile)
    resolver_block = [
        KERNEL_HARDWARE_PROFILE_MARKER,
        "import os as _kb_os",
    ]
    for key, value in sorted(env_payload.items()):
        resolver_block.append(f"_kb_os.environ.setdefault({json.dumps(key)}, {json.dumps(value)})")
    resolver_block.extend(["del _kb_os", ""])
    lines = text.splitlines()
    insert_at = find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def inject_force_train_env(kernel_dir: Path) -> None:
    """Inject environment bootstrap that keeps training enabled in staged kernels."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if KERNEL_FORCE_TRAIN_MARKER in text:
        return

    resolver_block = [
        KERNEL_FORCE_TRAIN_MARKER,
        "import os as _kb_os",
        "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '1'",
        "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '1'",
        "del _kb_os",
        "",
    ]
    lines = text.splitlines()
    insert_at = find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def strip_competition_slug_bootstrap(lines: list[str]) -> list[str]:
    return _strip_env_bootstrap_block(lines, marker=KERNEL_COMPETITION_SLUG_MARKER)


def strip_force_train_bootstrap(lines: list[str]) -> list[str]:
    """Remove injected force-train bootstrap blocks from kernel text lines."""
    return _strip_env_bootstrap_block(lines, marker=KERNEL_FORCE_TRAIN_MARKER)


def inject_submit_inference_env(kernel_dir: Path) -> None:
    """Inject environment bootstrap that disables training and forces inference-only submit notebooks."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if KERNEL_SUBMIT_INFERENCE_MARKER in text:
        return

    resolver_block = [
        KERNEL_SUBMIT_INFERENCE_MARKER,
        "import os as _kb_os",
        "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'",
        "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '0'",
        "_kb_os.environ['KAGGLEBOT_DO_INFER'] = '1'",
        "_kb_os.environ['KAGGLEBOT_SUBMIT_NOTEBOOK'] = '1'",
        "_kb_os.environ['KAGGLEBOT_SUBMIT_SKIP_CV'] = '1'",
        "del _kb_os",
        "",
    ]
    lines = strip_force_train_bootstrap(text.splitlines())
    insert_at = find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def strip_submit_inference_bootstrap(lines: list[str]) -> list[str]:
    return _strip_env_bootstrap_block(lines, marker=KERNEL_SUBMIT_INFERENCE_MARKER)


def ensure_kernel_competition_slug_env(kernel_dir: Path, competition_slug: str) -> None:
    """Ensure the kernel runtime can resolve the competition slug on Kaggle."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    slug_literal = json.dumps(str(competition_slug))
    expected_slug_line = f"_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = {slug_literal}"
    expected_alias_line = f"_kb_os.environ['KAGGLEBOT_SLUG'] = {slug_literal}"
    if KERNEL_COMPETITION_SLUG_MARKER in text and expected_slug_line in text and expected_alias_line in text:
        return
    if KERNEL_COMPETITION_SLUG_MARKER in text:
        stripped_lines = strip_competition_slug_bootstrap(text.splitlines())
        stripped_text = "\n".join(stripped_lines)
        if text.endswith("\n"):
            stripped_text += "\n"
        kernel_path.write_text(stripped_text, encoding="utf-8")
    inject_competition_slug_env(kernel_dir, competition_slug)
    updated = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if (
        KERNEL_COMPETITION_SLUG_MARKER not in updated
        or expected_slug_line not in updated
        or expected_alias_line not in updated
    ):
        raise KernelFailedError(
            "Failed to inject competition slug bootstrap into kernel.py. "
            "Refusing to push a kernel that may mis-resolve /kaggle/input paths."
        )


def ensure_kernel_force_train_env(kernel_dir: Path) -> None:
    """Ensure staged kernel runtime has force-train env injection."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    expected_train_line = "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '1'"
    expected_force_line = "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '1'"
    if KERNEL_FORCE_TRAIN_MARKER in text and expected_train_line in text and expected_force_line in text:
        return
    if KERNEL_FORCE_TRAIN_MARKER in text:
        stripped_lines = strip_force_train_bootstrap(text.splitlines())
        stripped_text = "\n".join(stripped_lines)
        if text.endswith("\n"):
            stripped_text += "\n"
        kernel_path.write_text(stripped_text, encoding="utf-8")
    inject_force_train_env(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if (
        KERNEL_FORCE_TRAIN_MARKER not in updated
        or expected_train_line not in updated
        or expected_force_line not in updated
    ):
        raise KernelFailedError(
            "Failed to inject force-train bootstrap into kernel.py. "
            "Refusing to push a kernel that may auto-disable training."
        )


def ensure_kernel_submit_inference_env(kernel_dir: Path) -> None:
    """Ensure staged notebook submit kernel disables training and keeps inference on."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    expected_train_line = "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'"
    expected_force_line = "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '0'"
    expected_infer_line = "_kb_os.environ['KAGGLEBOT_DO_INFER'] = '1'"
    expected_submit_line = "_kb_os.environ['KAGGLEBOT_SUBMIT_NOTEBOOK'] = '1'"
    expected_skip_cv_line = "_kb_os.environ['KAGGLEBOT_SUBMIT_SKIP_CV'] = '1'"
    if (
        KERNEL_SUBMIT_INFERENCE_MARKER in text
        and expected_train_line in text
        and expected_force_line in text
        and expected_infer_line in text
        and expected_submit_line in text
        and expected_skip_cv_line in text
    ):
        return
    stripped_lines = strip_submit_inference_bootstrap(strip_force_train_bootstrap(text.splitlines()))
    stripped_text = "\n".join(stripped_lines)
    if text.endswith("\n"):
        stripped_text += "\n"
    kernel_path.write_text(stripped_text, encoding="utf-8")
    inject_submit_inference_env(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if (
        KERNEL_SUBMIT_INFERENCE_MARKER not in updated
        or expected_train_line not in updated
        or expected_force_line not in updated
        or expected_infer_line not in updated
        or expected_submit_line not in updated
        or expected_skip_cv_line not in updated
    ):
        raise KernelFailedError(
            "Failed to inject submit-inference bootstrap into kernel.py. "
            "Refusing to push a notebook submit kernel that may still force training."
        )


def find_bootstrap_block_end(lines: list[str]) -> int | None:
    if KERNEL_BOOTSTRAP_MARKER not in lines:
        return None
    start = lines.index(KERNEL_BOOTSTRAP_MARKER)
    search_end = min(start + 30, len(lines))
    for idx in range(start + 1, search_end):
        if lines[idx].strip() == KERNEL_BOOTSTRAP_END:
            return idx + 1
    return None


def find_bootstrap_insertion_index(lines: list[str]) -> int:
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if idx < len(lines) and _is_coding_comment(lines[idx]):
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx < len(lines) and (
        (lines[idx].startswith('"""') or lines[idx].startswith("'''"))
        and not lines[idx].strip().endswith(('"""', "'''"))
    ):
        quote = lines[idx].strip()[:3]
        idx += 1
        while idx < len(lines):
            current = lines[idx].strip()
            idx += 1
            if current.endswith(quote):
                break
    elif idx < len(lines) and (lines[idx].startswith('"""') or lines[idx].startswith("'''")):
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    while idx < len(lines) and lines[idx].startswith("from __future__ import "):
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return idx


def _is_coding_comment(line: str) -> bool:
    return bool(re.match(r"^#.*coding[:=]\s*[-\w.]+", line))


def _strip_env_bootstrap_block(lines: list[str], *, marker: str) -> list[str]:
    stripped = lines
    while marker in stripped:
        start = stripped.index(marker)
        end = None
        search_end = min(start + 12, len(stripped))
        for idx in range(start + 1, search_end):
            if stripped[idx].strip() == "del _kb_os":
                end = idx + 1
                if end < len(stripped) and stripped[end].strip() == "":
                    end += 1
                break
        if end is None:
            stripped = stripped[:start] + stripped[start + 1 :]
        else:
            stripped = stripped[:start] + stripped[end:]
    return stripped
