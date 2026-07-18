from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.hardware import hardware_env, resolve_hardware_profile

KERNEL_BOOTSTRAP_MARKER = "# kagglebot:kernel_sys_path"
KERNEL_BOOTSTRAP_END = "del _os, _sys, _KROOT, _KWORK"
KERNEL_COMPETITION_SLUG_MARKER = "# kagglebot:competition_slug"
KERNEL_HARDWARE_PROFILE_MARKER = "# kagglebot:hardware_profile"
KERNEL_FORCE_TRAIN_MARKER = "# kagglebot:force_train"
KERNEL_NON_TRAINING_MARKER = "# kagglebot:non_training_submission"
KERNEL_SUBMIT_INFERENCE_MARKER = "# kagglebot:submit_inference"
KERNEL_SUBMIT_FIDELITY_MARKER = "# kagglebot:submit_runtime_fidelity"
KERNEL_BOOTSTRAP_SCAN_LINES = 512


def strip_kernel_bootstrap(lines: list[str]) -> list[str]:
    stripped = lines
    while KERNEL_BOOTSTRAP_MARKER in stripped:
        start = stripped.index(KERNEL_BOOTSTRAP_MARKER)
        end = None
        search_end = min(start + KERNEL_BOOTSTRAP_SCAN_LINES, len(stripped))
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
    lines = strip_kernel_bootstrap(text.splitlines())
    source_without_bootstrap = "\n".join(lines)
    if text.endswith("\n"):
        source_without_bootstrap += "\n"
    needs_unsloth_prelude = bool(
        re.search(r"(?m)^\s*(?:import\s+unsloth\b|from\s+unsloth\b)", source_without_bootstrap)
    )
    unsloth_prelude = ""
    if needs_unsloth_prelude:
        unsloth_prelude = (
            "_kb_reference_proxy = _os.path.join(_KRUNTIME_CACHE, 'reference_packages')\n"
            "_os.makedirs(_kb_reference_proxy, exist_ok=True)\n"
            "import shutil as _kb_shutil\n"
            "_kb_cache_tag = str(getattr(_sys.implementation, 'cache_tag', '') or '')\n"
            "_kb_proxy_ready = False\n"
            "for _kb_reference_path in _KDEFERRED_NOTEBOOK_PATHS:\n"
            "    _kb_reference_resolved = _os.path.abspath(_kb_reference_path or _os.getcwd())\n"
            "    if not _os.path.isfile(_os.path.join(_kb_reference_resolved, 'unsloth', '__init__.py')):\n"
            "        continue\n"
            "    for _kb_name in _os.listdir(_kb_reference_resolved):\n"
            "        if _kb_name not in {'unsloth', 'unsloth_zoo'}:\n"
            "            continue\n"
            "        _kb_source = _os.path.join(_kb_reference_resolved, _kb_name)\n"
            "        if _kb_name in _KCRITICAL_IMPORTS or _kb_name.endswith(('.dist-info', '.egg-info')):\n"
            "            continue\n"
            "        if _os.path.isdir(_kb_source):\n"
            "            if not _os.path.isfile(_os.path.join(_kb_source, '__init__.py')):\n"
            "                continue\n"
            "            _kb_incompatible = False\n"
            "            for _kb_walk_root, _kb_walk_dirs, _kb_walk_files in _os.walk(_kb_source):\n"
            "                del _kb_walk_root, _kb_walk_dirs\n"
            "                if any(\n"
            "                    _kb_file.endswith('.so')\n"
            "                    and '.cpython-' in _kb_file\n"
            "                    and _kb_cache_tag not in _kb_file\n"
            "                    for _kb_file in _kb_walk_files\n"
            "                ):\n"
            "                    _kb_incompatible = True\n"
            "                    break\n"
            "            if _kb_incompatible:\n"
            "                continue\n"
            "        elif not _kb_name.endswith('.py'):\n"
            "            continue\n"
            "        _kb_target = _os.path.join(_kb_reference_proxy, _kb_name)\n"
            "        if not _os.path.lexists(_kb_target):\n"
            "            if _os.path.isdir(_kb_source):\n"
            "                _kb_shutil.copytree(_kb_source, _kb_target, symlinks=True)\n"
            "            else:\n"
            "                _kb_shutil.copy2(_kb_source, _kb_target)\n"
            "        if _kb_name == 'unsloth':\n"
            "            _kb_utils = _os.path.join(_kb_target, 'models', '_utils.py')\n"
            "            if _os.path.isfile(_kb_utils):\n"
            "                with open(_kb_utils, encoding='utf-8') as _kb_f:\n"
            "                    _kb_utils_text = _kb_f.read()\n"
            "                _kb_alias_import = 'from transformers import PretrainedConfig\\n'\n"
            "                _kb_alias_line = 'PreTrainedConfig = PretrainedConfig\\n'\n"
            "                if _kb_alias_import in _kb_utils_text and _kb_alias_line not in _kb_utils_text:\n"
            "                    _kb_utils_text = _kb_utils_text.replace(\n"
            "                        _kb_alias_import, _kb_alias_import + _kb_alias_line, 1\n"
            "                    )\n"
            "                    with open(_kb_utils, 'w', encoding='utf-8') as _kb_f:\n"
            "                        _kb_f.write(_kb_utils_text)\n"
            "                _kb_exec_line = '    exec(config, globals())\\n'\n"
            "                _kb_globals_patch = (\n"
            '                    "    _kagglebot_config_globals = vars(__import__("\n'
            "                    \"config_filepath, fromlist=['*']))\\n\"\n"
            '                    "    globals().update({name: value for name, value in "\n'
            "                    \"_kagglebot_config_globals.items() if not name.startswith('__')})\\n\"\n"
            "                )\n"
            "                if _kb_exec_line in _kb_utils_text and _kb_globals_patch not in _kb_utils_text:\n"
            "                    _kb_utils_text = _kb_utils_text.replace(\n"
            "                        _kb_exec_line, _kb_globals_patch + _kb_exec_line, 1\n"
            "                    )\n"
            "                    with open(_kb_utils, 'w', encoding='utf-8') as _kb_f:\n"
            "                        _kb_f.write(_kb_utils_text)\n"
            "                del _kb_exec_line, _kb_globals_patch\n"
            "                del _kb_alias_import, _kb_alias_line, _kb_utils_text\n"
            "            del _kb_utils\n"
            "            _kb_vision = _os.path.join(_kb_target, 'models', 'vision.py')\n"
            "            if _os.path.isfile(_kb_vision):\n"
            "                with open(_kb_vision, encoding='utf-8') as _kb_f:\n"
            "                    _kb_vision_text = _kb_f.read()\n"
            "                if (\n"
            "                    'HybridCache' in _kb_vision_text\n"
            "                    and all(\n"
            "                        not line.strip() or line.lstrip().startswith('from transformers import ')\n"
            "                        for line in _kb_vision_text.splitlines()\n"
            "                        if 'HybridCache' in line\n"
            "                    )\n"
            "                ):\n"
            "                    _kb_vision_text = _kb_vision_text.replace(', HybridCache', '')\n"
            "                    with open(_kb_vision, 'w', encoding='utf-8') as _kb_f:\n"
            "                        _kb_f.write(_kb_vision_text)\n"
            "                del _kb_vision_text\n"
            "            del _kb_vision\n"
            "            for _kb_model_module in ('llama.py', 'gemma.py'):\n"
            "                _kb_model_path = _os.path.join(_kb_target, 'models', _kb_model_module)\n"
            "                if not _os.path.isfile(_kb_model_path):\n"
            "                    continue\n"
            "                with open(_kb_model_path, encoding='utf-8') as _kb_f:\n"
            "                    _kb_model_text = _kb_f.read()\n"
            "                _kb_legacy_rope = 'config.rope_theta'\n"
            "                _kb_v5_rope = (\n"
            "                    \"(config.rope_theta if hasattr(config, 'rope_theta') else \"\n"
            "                    \"(lambda _kb_rp: _kb_rp.get('rope_theta', base) if isinstance(_kb_rp, dict) \"\n"
            "                    \"else getattr(_kb_rp, 'rope_theta', base))\"\n"
            "                    \"(getattr(config, 'rope_parameters', None)))\"\n"
            "                )\n"
            "                if _kb_legacy_rope in _kb_model_text:\n"
            "                    _kb_model_text = _kb_model_text.replace(_kb_legacy_rope, _kb_v5_rope)\n"
            "                    with open(_kb_model_path, 'w', encoding='utf-8') as _kb_f:\n"
            "                        _kb_f.write(_kb_model_text)\n"
            "                del _kb_model_path, _kb_model_text, _kb_legacy_rope, _kb_v5_rope\n"
            "            del _kb_model_module\n"
            "            _kb_save = _os.path.join(_kb_target, 'save.py')\n"
            "            if _os.path.isfile(_kb_save):\n"
            "                with open(_kb_save, encoding='utf-8') as _kb_f:\n"
            "                    _kb_save_text = _kb_f.read()\n"
            "                _kb_legacy_model_walk = (\n"
            "                    'if hasattr(original_model, \"model\"): original_model = original_model.model'\n"
            "                )\n"
            "                _kb_safe_model_walk = (\n"
            "                    'if hasattr(original_model, \"model\") and '\n"
            "                    'hasattr(original_model.model, \"push_to_hub\"): '\n"
            "                    'original_model = original_model.model'\n"
            "                )\n"
            "                if _kb_legacy_model_walk in _kb_save_text:\n"
            "                    _kb_save_text = _kb_save_text.replace(\n"
            "                        _kb_legacy_model_walk, _kb_safe_model_walk\n"
            "                    )\n"
            "                    with open(_kb_save, 'w', encoding='utf-8') as _kb_f:\n"
            "                        _kb_f.write(_kb_save_text)\n"
            "                del _kb_save_text, _kb_legacy_model_walk, _kb_safe_model_walk\n"
            "            del _kb_save\n"
            "        _kb_proxy_ready = True\n"
            "if _kb_proxy_ready and _kb_reference_proxy not in _sys.path:\n"
            "    _sys.path.insert(0, _kb_reference_proxy)\n"
            "del _kb_reference_proxy, _kb_cache_tag, _kb_proxy_ready, _kb_shutil\n"
            "try:\n"
            "    import unsloth as _kb_early_unsloth\n"
            "except Exception:\n"
            "    import traceback as _kb_traceback\n"
            "    print('[kagglebot] early Unsloth import failed\\n' + _kb_traceback.format_exc(), file=_sys.stderr)\n"
            "    del _kb_traceback\n"
            "else:\n"
            "    del _kb_early_unsloth\n"
            "try:\n"
            "    import torch.nn.functional as _kb_torch_functional\n"
            "    import unsloth.models.qwen3 as _kb_qwen3_module\n"
            "    if not callable(getattr(_kb_qwen3_module, 'flash_attn_func', None)):\n"
            "        def _kb_qwen3_sdpa(q, k, v, *_kb_args, "
            "_kb_sdpa=_kb_torch_functional.scaled_dot_product_attention, **_kb_kwargs):\n"
            "            del _kb_args\n"
            "            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)\n"
            "            if q.shape[1] != k.shape[1]:\n"
            "                if q.shape[1] % k.shape[1] != 0:\n"
            "                    raise RuntimeError('Incompatible Qwen3 Q/KV attention head counts')\n"
            "                _kb_groups = q.shape[1] // k.shape[1]\n"
            "                k = k.repeat_interleave(_kb_groups, dim=1)\n"
            "                v = v.repeat_interleave(_kb_groups, dim=1)\n"
            "            _kb_causal = bool(_kb_kwargs.get('causal', _kb_kwargs.get('is_causal', False)))\n"
            "            _kb_causal = _kb_causal and q.shape[-2] == k.shape[-2]\n"
            "            return _kb_sdpa(\n"
            "                q.contiguous(), k.contiguous(), v.contiguous(),\n"
            "                dropout_p=float(_kb_kwargs.get('dropout_p', 0.0) or 0.0),\n"
            "                is_causal=_kb_causal,\n"
            "            ).transpose(1, 2).contiguous()\n"
            "        _kb_qwen3_module.flash_attn_func = _kb_qwen3_sdpa\n"
            "        del _kb_qwen3_sdpa\n"
            "    del _kb_qwen3_module, _kb_torch_functional\n"
            "except Exception:\n"
            "    pass\n"
        )
    bootstrap = (
        f"{KERNEL_BOOTSTRAP_MARKER}\n"
        "import os as _os\n"
        "import sys as _sys\n"
        "_KNOTEBOOK_ROOT = _os.path.abspath(\n"
        "    _os.environ.get('KAGGLEBOT_NOTEBOOK_SOURCE_ROOT', '/kaggle/usr/lib/notebooks')\n"
        ")\n"
        "_KCRITICAL_IMPORTS = ('numpy', 'pandas', 'scipy', 'sklearn', 'torch')\n"
        "_KDEFERRED_NOTEBOOK_PATHS = []\n"
        "_KPATH_ENTRY = None\n"
        "_KPATH_RESOLVED = ''\n"
        "for _KPATH_ENTRY in tuple(_sys.path):\n"
        "    try:\n"
        "        _KPATH_RESOLVED = _os.path.abspath(_KPATH_ENTRY or _os.getcwd())\n"
        "    except Exception:\n"
        "        continue\n"
        "    if not _KPATH_RESOLVED.startswith(_KNOTEBOOK_ROOT + _os.sep):\n"
        "        continue\n"
        "    if not any(\n"
        "        _os.path.isfile(_os.path.join(_KPATH_RESOLVED, _KNAME, '__init__.py'))\n"
        "        for _KNAME in _KCRITICAL_IMPORTS\n"
        "    ):\n"
        "        continue\n"
        "    while _KPATH_ENTRY in _sys.path:\n"
        "        _sys.path.remove(_KPATH_ENTRY)\n"
        "    _KDEFERRED_NOTEBOOK_PATHS.append(_KPATH_ENTRY)\n"
        "_sys.path.extend(_KDEFERRED_NOTEBOOK_PATHS)\n"
        "try:\n"
        "    _KROOT = _os.path.dirname(_os.path.abspath(__file__))\n"
        "except NameError:\n"
        "    _KROOT = _os.getcwd()\n"
        "if _KROOT not in _sys.path:\n"
        "    _sys.path.insert(0, _KROOT)\n"
        "_KWORK = _os.environ.get('KAGGLEBOT_WORKING_DIR', '/kaggle/working')\n"
        "if _KWORK not in _sys.path:\n"
        "    _sys.path.insert(0, _KWORK)\n"
        "_KRUNTIME_CACHE = _os.environ.get(\n"
        "    'KAGGLEBOT_RUNTIME_CACHE_DIR', f'/tmp/kagglebot-runtime-{_os.getpid()}'\n"
        ")\n"
        "_os.makedirs(_KRUNTIME_CACHE, exist_ok=True)\n"
        "_os.environ.setdefault(\n"
        "    'UNSLOTH_COMPILE_LOCATION', _os.path.join(_KRUNTIME_CACHE, 'unsloth_compiled_cache')\n"
        ")\n"
        "_os.environ.setdefault('TORCH_EXTENSIONS_DIR', _os.path.join(_KRUNTIME_CACHE, 'torch_extensions'))\n"
        f"{unsloth_prelude}"
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
        "del _KNOTEBOOK_ROOT, _KCRITICAL_IMPORTS, _KDEFERRED_NOTEBOOK_PATHS\n"
        "del _KPATH_ENTRY, _KPATH_RESOLVED\n"
        "del _KRUNTIME_CACHE\n"
        "del _os, _sys, _KROOT, _KWORK\n"
    )
    new_text = compose_kernel_source(source_without_bootstrap, bootstrap, filename=str(kernel_path))
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


def inject_non_training_env(kernel_dir: Path) -> None:
    """Disable training while retaining inference and validation for an approved route."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if KERNEL_NON_TRAINING_MARKER in text:
        return

    resolver_block = [
        KERNEL_NON_TRAINING_MARKER,
        "import os as _kb_os",
        "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'",
        "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '0'",
        "_kb_os.environ['KAGGLEBOT_DO_INFER'] = '1'",
        "_kb_os.environ['KAGGLEBOT_EXECUTION_MODE'] = 'non_training_submission'",
        "_kb_os.environ['KAGGLEBOT_NON_TRAINING_VALIDATION_REQUIRED'] = '1'",
        "del _kb_os",
        "",
    ]
    lines = strip_submit_inference_bootstrap(strip_force_train_bootstrap(text.splitlines()))
    insert_at = find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def strip_non_training_bootstrap(lines: list[str]) -> list[str]:
    return _strip_env_bootstrap_block(lines, marker=KERNEL_NON_TRAINING_MARKER)


def inject_submit_inference_env(
    kernel_dir: Path,
    *,
    runtime_env: Mapping[str, str] | None = None,
) -> None:
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
        "_kb_os.environ.setdefault('XDG_CACHE_HOME', '/tmp/kagglebot-cache/xdg')",
        "_kb_os.environ.setdefault('PIP_CACHE_DIR', '/tmp/kagglebot-cache/pip')",
        "_kb_os.environ.setdefault('HF_HOME', '/tmp/kagglebot-cache/huggingface')",
        "_kb_os.environ.setdefault('TORCH_HOME', '/tmp/kagglebot-cache/torch')",
        "_kb_os.environ.setdefault('MPLCONFIGDIR', '/tmp/kagglebot-cache/matplotlib')",
        "_kb_os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/kagglebot-cache/numba')",
    ]
    for name, value in sorted((runtime_env or {}).items()):
        if not re.fullmatch(r"KAGGLEBOT_[A-Z0-9_]+", name):
            raise KernelFailedError(f"Invalid submit inference environment name: {name!r}")
        resolver_block.append(f"_kb_os.environ[{name!r}] = {str(value)!r}")
    resolver_block.extend(["del _kb_os", ""])
    lines = strip_non_training_bootstrap(strip_force_train_bootstrap(text.splitlines()))
    insert_at = find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def inject_submit_runtime_fidelity(kernel_dir: Path) -> None:
    """Install the cheap, fail-closed recorder in fresh code-submit packages."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if KERNEL_SUBMIT_FIDELITY_MARKER in text:
        return
    recorder_block = [
        KERNEL_SUBMIT_FIDELITY_MARKER,
        "try:",
        "    from submit_runtime_fidelity import install as _kb_install_submit_fidelity",
        "    _kb_install_submit_fidelity()",
        "    del _kb_install_submit_fidelity",
        "except Exception:",
        "    pass",
        "",
    ]
    lines = text.splitlines()
    insert_at = find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + recorder_block + lines[insert_at:]
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
    if (
        KERNEL_FORCE_TRAIN_MARKER in text
        and KERNEL_NON_TRAINING_MARKER not in text
        and KERNEL_SUBMIT_INFERENCE_MARKER not in text
        and expected_train_line in text
        and expected_force_line in text
    ):
        return
    if any(
        marker in text
        for marker in (KERNEL_FORCE_TRAIN_MARKER, KERNEL_NON_TRAINING_MARKER, KERNEL_SUBMIT_INFERENCE_MARKER)
    ):
        stripped_lines = strip_force_train_bootstrap(
            strip_non_training_bootstrap(strip_submit_inference_bootstrap(text.splitlines()))
        )
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


def ensure_kernel_non_training_env(kernel_dir: Path) -> None:
    """Ensure an approved non-training kernel cannot be changed back to training by staging."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    expected_mode_line = "_kb_os.environ['KAGGLEBOT_EXECUTION_MODE'] = 'non_training_submission'"
    expected_train_line = "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'"
    expected_validation_line = "_kb_os.environ['KAGGLEBOT_NON_TRAINING_VALIDATION_REQUIRED'] = '1'"
    if (
        KERNEL_NON_TRAINING_MARKER in text
        and KERNEL_FORCE_TRAIN_MARKER not in text
        and KERNEL_SUBMIT_INFERENCE_MARKER not in text
        and expected_mode_line in text
        and expected_train_line in text
        and expected_validation_line in text
    ):
        return
    stripped_lines = strip_non_training_bootstrap(
        strip_submit_inference_bootstrap(strip_force_train_bootstrap(text.splitlines()))
    )
    stripped_text = "\n".join(stripped_lines)
    if text.endswith("\n"):
        stripped_text += "\n"
    kernel_path.write_text(stripped_text, encoding="utf-8")
    inject_non_training_env(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if (
        KERNEL_NON_TRAINING_MARKER not in updated
        or expected_mode_line not in updated
        or expected_train_line not in updated
        or expected_validation_line not in updated
    ):
        raise KernelFailedError(
            "Failed to inject the approved non-training bootstrap into kernel.py. "
            "Refusing to run a kernel that may start local training."
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
    stripped_lines = strip_submit_inference_bootstrap(
        strip_non_training_bootstrap(strip_force_train_bootstrap(text.splitlines()))
    )
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
    search_end = min(start + KERNEL_BOOTSTRAP_SCAN_LINES, len(lines))
    for idx in range(start + 1, search_end):
        if lines[idx].strip() == KERNEL_BOOTSTRAP_END:
            return idx + 1
    return None


def find_bootstrap_insertion_index(lines: list[str]) -> int:
    source = "\n".join(lines)
    try:
        module = ast.parse(source)
    except SyntaxError:
        return _fallback_bootstrap_insertion_index(lines)

    if not module.body:
        return len(lines)

    first = module.body[0]
    has_docstring = (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    )
    body_idx = 1 if has_docstring else 0
    insert_at = int(first.end_lineno or first.lineno) if has_docstring else _statement_start_line(first) - 1
    while body_idx < len(module.body):
        node = module.body[body_idx]
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        insert_at = int(node.end_lineno or node.lineno)
        body_idx += 1
    return insert_at


def compose_kernel_source(source: str, bootstrap: str, *, filename: str = "<kernel.py>") -> str:
    """Insert bootstrap code after the module prologue and verify the result."""
    try:
        bootstrap_module = ast.parse(bootstrap, filename=f"{filename} bootstrap")
    except SyntaxError as exc:
        raise KernelFailedError(f"Invalid kernel bootstrap for {filename}: {exc}") from exc
    if any(isinstance(node, ast.ImportFrom) and node.module == "__future__" for node in ast.walk(bootstrap_module)):
        raise KernelFailedError("Kernel bootstrap must not contain from __future__ imports.")

    lines = source.splitlines()
    insert_at = find_bootstrap_insertion_index(lines)
    new_lines = lines[:insert_at] + bootstrap.splitlines() + lines[insert_at:]
    composed = "\n".join(new_lines)
    if source.endswith("\n"):
        composed += "\n"
    try:
        compile(composed, filename, "exec")
    except SyntaxError as exc:
        raise KernelFailedError(f"Composed kernel source is invalid for {filename}: {exc}") from exc
    return composed


def _statement_start_line(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _fallback_bootstrap_insertion_index(lines: list[str]) -> int:
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if idx < len(lines) and _is_coding_comment(lines[idx]):
        idx += 1
    while idx < len(lines) and (not lines[idx].strip() or lines[idx].lstrip().startswith("#")):
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
