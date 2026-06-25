from __future__ import annotations

from pathlib import Path

from kagglebot.exceptions import KernelFailedError

KAGGLE_WORKING_REDIRECT_SHIM_MARKER = "# kagglebot: kaggle-working-redirect-shim"
LGBM_GPU_GUARD_SHIM_MARKER = "# kagglebot: lgbm-gpu-guard-shim"
TORCH_RUNTIME_GUARD_SHIM_MARKER = "# kagglebot: torch-runtime-guard-shim"
TRAIN_PROGRESS_SHIM_MARKER = "# kagglebot: train-progress-shim"
TRANSFORMERS_EVAL_STRATEGY_SHIM_MARKER = "# kagglebot: transformers-eval-strategy-shim"


def append_sitecustomize_shim(kernel_dir: Path, marker: str, shim: list[str]) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if marker in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def inject_kaggle_working_redirect_shim(kernel_dir: Path) -> None:
    shim = [
        KAGGLE_WORKING_REDIRECT_SHIM_MARKER,
        "import builtins",
        "import io",
        "import os",
        "from pathlib import Path",
        "",
        "def _kb_local_kernel_mode() -> bool:",
        "    value = str(os.environ.get('KAGGLEBOT_LOCAL_KERNEL', '0')).strip().lower()",
        "    return value in {'1', 'true', 'yes', 'on'}",
        "",
        "def _kb_redirect_root() -> Path | None:",
        "    root = str(os.environ.get('KAGGLEBOT_LOCAL_WORKING_DIR', '')).strip()",
        "    if not root:",
        "        return None",
        "    return Path(root)",
        "",
        "def _kb_remap_path(path_value):",
        "    try:",
        "        raw = os.fspath(path_value)",
        "    except Exception:",
        "        return path_value",
        "    if not isinstance(raw, str):",
        "        return path_value",
        "    if raw == '/kaggle/working':",
        "        root = _kb_redirect_root()",
        "        return str(root) if root is not None else path_value",
        "    if raw.startswith('/kaggle/working/'):",
        "        root = _kb_redirect_root()",
        "        if root is None:",
        "            return path_value",
        "        suffix = raw[len('/kaggle/working/'):].lstrip('/')",
        "        return str(root / suffix)",
        "    return path_value",
        "",
        "def _kb_prepare_parent(path_value, mode: str) -> None:",
        "    if not any(flag in mode for flag in ('w', 'a', 'x', '+')):",
        "        return",
        "    try:",
        "        parent = Path(os.fspath(path_value)).parent",
        "        parent.mkdir(parents=True, exist_ok=True)",
        "    except Exception:",
        "        return",
        "",
        "def _kb_patch_open_redirect() -> None:",
        "    if not _kb_local_kernel_mode():",
        "        return",
        "    _orig_builtin_open = builtins.open",
        "    _orig_io_open = io.open",
        "",
        "    def _open_builtin(file, mode='r', *args, **kwargs):",
        "        mapped = _kb_remap_path(file)",
        "        _kb_prepare_parent(mapped, mode)",
        "        return _orig_builtin_open(mapped, mode, *args, **kwargs)",
        "",
        "    def _open_io(file, mode='r', *args, **kwargs):",
        "        mapped = _kb_remap_path(file)",
        "        _kb_prepare_parent(mapped, mode)",
        "        return _orig_io_open(mapped, mode, *args, **kwargs)",
        "",
        "    builtins.open = _open_builtin",
        "    io.open = _open_io",
        "",
        "_kb_patch_open_redirect()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, KAGGLE_WORKING_REDIRECT_SHIM_MARKER, shim)


def inject_lgbm_gpu_guard_shim(kernel_dir: Path) -> None:
    shim = [
        LGBM_GPU_GUARD_SHIM_MARKER,
        "import os",
        "",
        "def _kb_disable_lgbm_gpu_enabled() -> bool:",
        "    value = str(os.environ.get('KAGGLEBOT_DISABLE_LGBM_GPU', '0')).strip().lower()",
        "    return value in {'1', 'true', 'yes', 'on'}",
        "",
        "def _kb_patch_lgbm_gpu_guard() -> None:",
        "    if not _kb_disable_lgbm_gpu_enabled():",
        "        return",
        "    try:",
        "        import lightgbm as _lgb",
        "    except Exception:",
        "        return",
        "",
        "    def _force_cpu(estimator) -> None:",
        "        for key in ('device', 'device_type'):",
        "            try:",
        "                estimator.set_params(**{key: 'cpu'})",
        "            except Exception:",
        "                continue",
        "",
        "    targets = ('LGBMModel', 'LGBMRegressor', 'LGBMClassifier', 'LGBMRanker')",
        "    for cls_name in targets:",
        "        cls = getattr(_lgb, cls_name, None)",
        "        if cls is None:",
        "            continue",
        "        fit = getattr(cls, 'fit', None)",
        "        if fit is None or not callable(fit) or getattr(fit, '__kb_lgbm_cpu_wrapped__', False):",
        "            continue",
        "        def _wrapped(self, *args, _fit=fit, **kwargs):",
        "            _force_cpu(self)",
        "            return _fit(self, *args, **kwargs)",
        "        _wrapped.__kb_lgbm_cpu_wrapped__ = True",
        "        setattr(cls, 'fit', _wrapped)",
        "",
        "    train_fn = getattr(_lgb, 'train', None)",
        "    if callable(train_fn) and not getattr(train_fn, '__kb_lgbm_cpu_wrapped__', False):",
        "        def _train(params, *args, _train=train_fn, **kwargs):",
        "            if isinstance(params, dict):",
        "                updated = dict(params)",
        "                updated['device'] = 'cpu'",
        "                updated['device_type'] = 'cpu'",
        "                params = updated",
        "            return _train(params, *args, **kwargs)",
        "        _train.__kb_lgbm_cpu_wrapped__ = True",
        "        _lgb.train = _train",
        "",
        "_kb_patch_lgbm_gpu_guard()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, LGBM_GPU_GUARD_SHIM_MARKER, shim)


def inject_torch_runtime_guard_shim(kernel_dir: Path) -> None:
    shim = [
        TORCH_RUNTIME_GUARD_SHIM_MARKER,
        "import os",
        "",
        "def _kb_local_kernel_mode() -> bool:",
        "    value = str(os.environ.get('KAGGLEBOT_LOCAL_KERNEL', '0')).strip().lower()",
        "    return value in {'1', 'true', 'yes', 'on'}",
        "",
        "def _kb_patch_torch_runtime_guard() -> None:",
        "    if not _kb_local_kernel_mode():",
        "        return",
        "    target_nofile = str(os.environ.get('KAGGLEBOT_LOCAL_NOFILE', '')).strip()",
        "    if target_nofile:",
        "        try:",
        "            import resource",
        "            desired = max(256, int(target_nofile))",
        "            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)",
        "            hard_cap = desired if hard is None or int(hard) < 0 else int(hard)",
        "            new_soft = min(max(int(soft), desired), hard_cap)",
        "            if new_soft > int(soft):",
        "                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))",
        "        except Exception:",
        "            pass",
        "    strategy = str(os.environ.get('KAGGLEBOT_TORCH_SHARING_STRATEGY', '')).strip()",
        "    if strategy:",
        "        try:",
        "            import torch.multiprocessing as _kb_tmp",
        "            getter = getattr(_kb_tmp, 'get_sharing_strategy', None)",
        "            current = getter() if callable(getter) else None",
        "            if current != strategy:",
        "                _kb_tmp.set_sharing_strategy(strategy)",
        "        except Exception:",
        "            pass",
        "",
        "_kb_patch_torch_runtime_guard()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, TORCH_RUNTIME_GUARD_SHIM_MARKER, shim)


def inject_transformers_eval_strategy_shim(kernel_dir: Path) -> None:
    """Patch transformers API drift for Seq2SeqTrainingArguments eval strategy naming."""
    shim = [
        TRANSFORMERS_EVAL_STRATEGY_SHIM_MARKER,
        "import inspect",
        "",
        "def _kb_patch_transformers_eval_strategy_alias() -> None:",
        "    try:",
        "        import transformers as _tf",
        "    except Exception:",
        "        return",
        "    args_cls = getattr(_tf, 'Seq2SeqTrainingArguments', None)",
        "    if args_cls is None:",
        "        return",
        "    try:",
        "        params = inspect.signature(args_cls.__init__).parameters",
        "    except Exception:",
        "        return",
        "    if 'evaluation_strategy' in params:",
        "        return",
        "    if 'eval_strategy' not in params:",
        "        return",
        "    _orig_init = args_cls.__init__",
        "    def _patched_init(self, *args, **kwargs):",
        "        if 'evaluation_strategy' in kwargs and 'eval_strategy' not in kwargs:",
        "            kwargs['eval_strategy'] = kwargs.pop('evaluation_strategy')",
        "        return _orig_init(self, *args, **kwargs)",
        "    args_cls.__init__ = _patched_init",
        "",
        "_kb_patch_transformers_eval_strategy_alias()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, TRANSFORMERS_EVAL_STRATEGY_SHIM_MARKER, shim)


def ensure_training_progress_shim(kernel_dir: Path) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    if not site_path.exists():
        raise KernelFailedError(
            f"Training progress shim missing: {site_path}. Refusing to run a kernel without mandatory progress logging."
        )
    text = site_path.read_text(encoding="utf-8", errors="ignore")
    if TRAIN_PROGRESS_SHIM_MARKER not in text:
        raise KernelFailedError(
            f"Training progress shim marker not found in {site_path}. "
            "Refusing to run a kernel without mandatory progress logging."
        )
