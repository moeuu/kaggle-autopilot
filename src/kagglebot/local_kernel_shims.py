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


def inject_training_progress_shim(kernel_dir: Path) -> None:
    shim = (
        (
            f"""
{TRAIN_PROGRESS_SHIM_MARKER}
import importlib
import os
import threading
import time

_KB_PROGRESS = {{
    "started_at": time.monotonic(),
    "last_event_at": time.monotonic(),
    "watchdog_started": False,
}}

def _kb_progress_enabled() -> bool:
    value = str(os.environ.get("KAGGLEBOT_TRAIN_PROGRESS", "1")).strip().lower()
    return value not in {{"0", "false", "off", "no"}}

def _kb_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, value)

def _kb_float_env(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except Exception:
        value = default
    return max(minimum, value)

def _kb_emit(msg: str) -> None:
    _KB_PROGRESS["last_event_at"] = time.monotonic()
    print(f"[kernel] {{msg}}", flush=True)

def _kb_get_shape(args):
    if not args:
        return None, None
    x = args[0]
    rows = None
    cols = None
    try:
        rows = int(len(x))
    except Exception:
        rows = None
    try:
        shape = getattr(x, "shape", None)
        if shape is not None and len(shape) >= 2:
            cols = int(shape[1])
    except Exception:
        cols = None
    return rows, cols

def _kb_estimator_iter_budget(estimator) -> int | None:
    params = {{}}
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        params = {{}}
    for key in ("iterations", "n_estimators", "max_iter", "num_iterations"):
        value = params.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None

def _kb_resolve_boosting_log_every(estimator) -> int:
    forced = _kb_int_env("KAGGLEBOT_BOOSTING_LOG_EVERY", 0, 0)
    if forced > 0:
        return forced
    budget = _kb_estimator_iter_budget(estimator)
    if budget is None:
        return 100
    # Target around 20-30 evaluation points across a full fit.
    period = max(1, budget // 25)
    return min(max(period, 10), 200)

def _kb_choose_fit_tick_interval(label: str, rows: int | None) -> float:
    base = _kb_float_env("KAGGLEBOT_MODEL_PROGRESS_INTERVAL_SEC", 12.0, 5.0)
    if label in {{"catboost", "lightgbm", "xgboost"}}:
        # Boosting models also emit iteration logs; keep timer sparse.
        return max(base, 30.0)
    if rows is None:
        return base
    if rows >= 200000:
        return max(base, 30.0)
    if rows >= 50000:
        return max(base, 20.0)
    if rows >= 10000:
        return max(base, 12.0)
    return base

def _kb_start_watchdog_thread() -> None:
    if not _kb_progress_enabled():
        return
    if bool(_KB_PROGRESS.get("watchdog_started", False)):
        return
    _KB_PROGRESS["watchdog_started"] = True
    silence_sec = _kb_float_env("KAGGLEBOT_PROGRESS_INTERVAL_SEC", 45.0, 10.0)
    poll_sec = max(1.0, min(5.0, silence_sec / 6.0))
    def _run():
        while True:
            time.sleep(poll_sec)
            now = time.monotonic()
            last = float(_KB_PROGRESS.get("last_event_at", now))
            if now - last < silence_sec:
                continue
            elapsed = int(max(0.0, now - float(_KB_PROGRESS.get("started_at", now))))
            quiet = int(max(0.0, now - last))
            _kb_emit(f"train watchdog: elapsed={{elapsed}}s no_new_logs_for={{quiet}}s")
    t = threading.Thread(target=_run, daemon=True, name="kb-train-watchdog")
    t.start()

def _kb_wrap_splitter(module_name: str, class_name: str) -> None:
    if not _kb_progress_enabled():
        return
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return
    cls = getattr(mod, class_name, None)
    if cls is None:
        return
    split = getattr(cls, "split", None)
    if split is None or not callable(split):
        return
    if getattr(split, "__kb_progress_wrapped__", False):
        return
    def _wrapped(self, *args, **kwargs):
        iterator = split(self, *args, **kwargs)
        total = getattr(self, "n_splits", None)
        idx = 0
        for item in iterator:
            idx += 1
            train_n = "?"
            valid_n = "?"
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    train_n = str(len(item[0]))
                except Exception:
                    pass
                try:
                    valid_n = str(len(item[1]))
                except Exception:
                    pass
            fold_part = f"{{idx}}/{{total}}" if isinstance(total, int) and total > 0 else str(idx)
            _kb_emit(
                f"cv fold start: splitter={{class_name}} fold={{fold_part}} train={{train_n}} valid={{valid_n}}"
            )
            yield item
        if idx > 0:
            _kb_emit(f"cv split done: splitter={{class_name}} folds={{idx}}")
    _wrapped.__kb_progress_wrapped__ = True
    setattr(cls, "split", _wrapped)

def _kb_wrap_fit(module_name: str, class_name: str, label: str) -> None:
    if not _kb_progress_enabled():
        return
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return
    cls = getattr(mod, class_name, None)
    if cls is None:
        return
    fit = getattr(cls, "fit", None)
    if fit is None or not callable(fit):
        return
    if getattr(fit, "__kb_progress_wrapped__", False):
        return
    def _wrapped(self, *args, **kwargs):
        model_name = self.__class__.__name__
        rows, cols = _kb_get_shape(args)
        iter_budget = _kb_estimator_iter_budget(self)
        log_every = None
        if label in {{"catboost", "lightgbm", "xgboost"}}:
            log_every = _kb_resolve_boosting_log_every(self)
        summary = [f"train start: model={{label}}.{{model_name}}"]
        if rows is not None:
            summary.append(f"rows={{rows}}")
        if cols is not None:
            summary.append(f"cols={{cols}}")
        if iter_budget is not None:
            summary.append(f"iter_budget={{iter_budget}}")
        if log_every is not None:
            summary.append(f"log_every={{log_every}}")
        _kb_emit(" ".join(summary))
        try:
            if label == "lightgbm":
                import lightgbm as _lgb
                callbacks = list(kwargs.get("callbacks") or [])
                callbacks.append(_lgb.log_evaluation(period=log_every))
                kwargs["callbacks"] = callbacks
            elif label == "xgboost":
                if kwargs.get("eval_set"):
                    kwargs["verbose"] = log_every
            elif label == "catboost":
                try:
                    self.set_params(verbose=log_every)
                except Exception:
                    pass
                kwargs.setdefault("verbose", log_every)
        except Exception:
            pass
        started = time.monotonic()
        interval = _kb_choose_fit_tick_interval(label, rows)
        stop = threading.Event()
        def _ticker():
            while not stop.wait(interval):
                elapsed = int(max(0.0, time.monotonic() - started))
                _kb_emit(f"train running: model={{label}}.{{model_name}} elapsed={{elapsed}}s")
        thread = threading.Thread(target=_ticker, daemon=True, name=f"kb-fit-{{label}}")
        thread.start()
        try:
            return fit(self, *args, **kwargs)
        finally:
            stop.set()
            thread.join(timeout=0.2)
            elapsed = int(max(0.0, time.monotonic() - started))
            _kb_emit(f"train done: model={{label}}.{{model_name}} elapsed={{elapsed}}s")
    _wrapped.__kb_progress_wrapped__ = True
    setattr(cls, "fit", _wrapped)

def _kb_patch_training_progress() -> None:
    if not _kb_progress_enabled():
        return
    _kb_start_watchdog_thread()
    splitters = [
        ("sklearn.model_selection", "KFold"),
        ("sklearn.model_selection", "StratifiedKFold"),
        ("sklearn.model_selection", "GroupKFold"),
        ("sklearn.model_selection", "TimeSeriesSplit"),
    ]
    for module_name, class_name in splitters:
        _kb_wrap_splitter(module_name, class_name)
    targets = [
        ("catboost", "CatBoostRegressor", "catboost"),
        ("lightgbm", "LGBMRegressor", "lightgbm"),
        ("xgboost", "XGBRegressor", "xgboost"),
        ("sklearn.ensemble", "HistGradientBoostingRegressor", "sklearn"),
        ("sklearn.linear_model", "ElasticNet", "sklearn"),
        ("sklearn.linear_model", "Ridge", "sklearn"),
        ("sklearn.linear_model", "SGDRegressor", "sklearn"),
        ("sklearn.kernel_ridge", "KernelRidge", "sklearn"),
    ]
    for module_name, class_name, label in targets:
        _kb_wrap_fit(module_name, class_name, label)

_kb_patch_training_progress()
"""
        )
        .strip("\n")
        .splitlines()
    )
    append_sitecustomize_shim(kernel_dir, TRAIN_PROGRESS_SHIM_MARKER, shim)


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
