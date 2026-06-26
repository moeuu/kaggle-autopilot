from __future__ import annotations

from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_outputs import copy_artifact_if_needed
from kagglebot.local_kernel_drift_guard import ZERO_OVERLAP_DRIFT_GUARD_FILENAME

COLUMN_MAP_FILENAME = "column_map.json"
COLUMN_MAP_SHIM_MARKER = "# kagglebot: column-map-shim"
COLUMN_FILL_FILENAME = "column_fill.json"
COLUMN_FILL_SHIM_MARKER = "# kagglebot: column-fill-shim"
OBJECT_COERCE_FILENAME = "object_coerce.json"
OBJECT_COERCE_SHIM_MARKER = "# kagglebot: object-coerce-shim"
DEVICE_COERCE_FILENAME = "device_coerce.json"
DEVICE_COERCE_SHIM_MARKER = "# kagglebot: device-coerce-shim"
ZERO_OVERLAP_DRIFT_SHIM_MARKER = "# kagglebot: zero-overlap-drift-shim"
KAGGLE_WORKING_REDIRECT_SHIM_MARKER = "# kagglebot: kaggle-working-redirect-shim"
LGBM_GPU_GUARD_SHIM_MARKER = "# kagglebot: lgbm-gpu-guard-shim"
TORCH_RUNTIME_GUARD_SHIM_MARKER = "# kagglebot: torch-runtime-guard-shim"
TRAIN_PROGRESS_SHIM_MARKER = "# kagglebot: train-progress-shim"
TRANSFORMERS_EVAL_STRATEGY_SHIM_MARKER = "# kagglebot: transformers-eval-strategy-shim"


def inject_context_io_shims(kernel_dir: Path, context_dir: Path) -> None:
    inject_column_map_shim(kernel_dir, context_dir)
    inject_column_fill_shim(kernel_dir, context_dir)
    inject_object_coerce_shim(kernel_dir, context_dir)
    inject_device_coerce_shim(kernel_dir, context_dir)


def inject_local_runtime_shims(kernel_dir: Path) -> None:
    inject_kaggle_working_redirect_shim(kernel_dir)
    inject_lgbm_gpu_guard_shim(kernel_dir)
    inject_torch_runtime_guard_shim(kernel_dir)


def inject_training_compat_shims(kernel_dir: Path) -> None:
    inject_training_progress_shim(kernel_dir)
    inject_transformers_eval_strategy_shim(kernel_dir)


def append_sitecustomize_shim(kernel_dir: Path, marker: str, shim: list[str]) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if marker in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def inject_column_map_shim(kernel_dir: Path, context_dir: Path) -> None:
    map_path = context_dir / COLUMN_MAP_FILENAME
    if not map_path.exists():
        return
    kernel_map_path = kernel_dir / COLUMN_MAP_FILENAME
    copy_artifact_if_needed(source=map_path, destination=kernel_map_path)
    shim = [
        COLUMN_MAP_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_load_map() -> dict:",
        "    candidates = [",
        f"        Path(__file__).with_name('{COLUMN_MAP_FILENAME}'),",
        f"        Path('/kaggle/working/{COLUMN_MAP_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                continue",
        "            mapping = payload.get('mapping') if isinstance(payload, dict) else None",
        "            if isinstance(mapping, dict) and mapping:",
        "                return mapping",
        "    return {}",
        "",
        "def _kb_patch_pandas() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    mapping = _kb_load_map()",
        "    if not mapping:",
        "        return",
        "    _orig = _pd.read_csv",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        try:",
        "            return df.rename(columns=mapping)",
        "        except Exception:",
        "            return df",
        "    _pd.read_csv = _patched",
        "",
        "_kb_patch_pandas()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, COLUMN_MAP_SHIM_MARKER, shim)


def inject_column_fill_shim(kernel_dir: Path, context_dir: Path) -> None:
    fill_path = context_dir / COLUMN_FILL_FILENAME
    if not fill_path.exists():
        return
    kernel_fill_path = kernel_dir / COLUMN_FILL_FILENAME
    copy_artifact_if_needed(source=fill_path, destination=kernel_fill_path)
    shim = [
        COLUMN_FILL_SHIM_MARKER,
        "import json",
        "import re",
        "from pathlib import Path",
        "",
        "def _kb_load_fill() -> dict:",
        "    candidates = [",
        f"        Path(__file__).with_name('{COLUMN_FILL_FILENAME}'),",
        f"        Path('/kaggle/working/{COLUMN_FILL_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                continue",
        "            if isinstance(payload, dict):",
        "                return payload",
        "    return {}",
        "",
        "def _kb_missing_columns_for(path_value) -> list[str]:",
        "    payload = _kb_load_fill()",
        "    if not payload:",
        "        return []",
        "    file_map = payload.get('files') if isinstance(payload, dict) else None",
        "    try:",
        "        name = Path(str(path_value)).name",
        "    except Exception:",
        "        name = ''",
        "    if isinstance(file_map, dict) and name in file_map:",
        "        cols = file_map.get(name)",
        "        if isinstance(cols, list):",
        "            return [str(c) for c in cols if str(c).strip()]",
        "    cols = payload.get('missing_columns') if isinstance(payload, dict) else None",
        "    if isinstance(cols, list):",
        "        return [str(c) for c in cols if str(c).strip()]",
        "    return []",
        "",
        "def _kb_global_missing_columns() -> set[str]:",
        "    payload = _kb_load_fill()",
        "    if not payload:",
        "        return set()",
        "    allowed: set[str] = set()",
        "    cols = payload.get('missing_columns') if isinstance(payload, dict) else None",
        "    if isinstance(cols, list):",
        "        for col in cols:",
        "            name = str(col).strip()",
        "            if name:",
        "                allowed.add(name)",
        "    file_map = payload.get('files') if isinstance(payload, dict) else None",
        "    if isinstance(file_map, dict):",
        "        for value in file_map.values():",
        "            if not isinstance(value, list):",
        "                continue",
        "            for col in value:",
        "                name = str(col).strip()",
        "                if name:",
        "                    allowed.add(name)",
        "    return allowed",
        "",
        "def _kb_add_missing_columns(df, columns: list[str]) -> bool:",
        "    added = False",
        "    for col in columns:",
        "        if col in df.columns:",
        "            continue",
        "        try:",
        "            df[col] = float('nan')",
        "            added = True",
        "        except Exception:",
        "            continue",
        "    return added",
        "",
        "def _kb_parse_missing_from_keyerror(exc: Exception) -> list[str]:",
        "    text = str(exc)",
        '    match = re.search(r"\\[([^\\]]+)\\]\\s*not in index", text, flags=re.IGNORECASE)',
        "    if not match:",
        "        return []",
        "    raw = match.group(1).strip()",
        "    if not raw:",
        "        return []",
        "    values = []",
        "    for token in raw.split(','):",
        '        name = token.strip().strip("\'\\"")',
        "        if name:",
        "            values.append(name)",
        "    return values",
        "",
        "def _kb_patch_pandas_fill() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    _orig = _pd.read_csv",
        "    _orig_getitem = _pd.DataFrame.__getitem__",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        try:",
        "            path_value = args[0] if args else kwargs.get('filepath_or_buffer')",
        "            missing_cols = _kb_missing_columns_for(path_value)",
        "            _kb_add_missing_columns(df, missing_cols)",
        "        except Exception:",
        "            return df",
        "        return df",
        "    def _patched_getitem(df, key):",
        "        try:",
        "            return _orig_getitem(df, key)",
        "        except KeyError as exc:",
        "            if not isinstance(key, (list, tuple)):",
        "                raise",
        "            requested = [str(item) for item in key if isinstance(item, str)]",
        "            if not requested:",
        "                raise",
        "            allowed = _kb_global_missing_columns()",
        "            missing = [col for col in requested if col not in df.columns and (not allowed or col in allowed)]",
        "            if not missing:",
        "                parsed = _kb_parse_missing_from_keyerror(exc)",
        "                missing = [col for col in parsed if col in requested and (not allowed or col in allowed)]",
        "            if not missing:",
        "                raise",
        "            if not _kb_add_missing_columns(df, missing):",
        "                raise",
        "            return _orig_getitem(df, key)",
        "    _pd.read_csv = _patched",
        "    _pd.DataFrame.__getitem__ = _patched_getitem",
        "",
        "_kb_patch_pandas_fill()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, COLUMN_FILL_SHIM_MARKER, shim)


def inject_object_coerce_shim(kernel_dir: Path, context_dir: Path) -> None:
    coerce_path = context_dir / OBJECT_COERCE_FILENAME
    if not coerce_path.exists():
        return
    kernel_coerce_path = kernel_dir / OBJECT_COERCE_FILENAME
    copy_artifact_if_needed(source=coerce_path, destination=kernel_coerce_path)
    shim = [
        OBJECT_COERCE_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_object_coerce_enabled() -> bool:",
        "    candidates = [",
        f"        Path(__file__).with_name('{OBJECT_COERCE_FILENAME}'),",
        f"        Path('/kaggle/working/{OBJECT_COERCE_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                return True",
        "            if isinstance(payload, dict):",
        "                return bool(payload.get('enabled', True))",
        "            return True",
        "    return False",
        "",
        "def _kb_coerce_ndarray(value):",
        "    try:",
        "        import numpy as _np",
        "    except Exception:",
        "        return value",
        "    if not isinstance(value, _np.ndarray) or value.dtype != object:",
        "        return value",
        "    try:",
        "        return value.astype('float32')",
        "    except Exception:",
        "        try:",
        "            import pandas as _pd",
        "            flat = _pd.to_numeric(value.ravel(), errors='coerce').to_numpy()",
        "            flat = _np.nan_to_num(flat, nan=0.0)",
        "            return flat.reshape(value.shape).astype('float32')",
        "        except Exception:",
        "            try:",
        "                flat = _np.array([0.0 if v is None else v for v in value.ravel()], dtype='float32')",
        "                return flat.reshape(value.shape)",
        "            except Exception:",
        "                return value",
        "",
        "def _kb_patch_torch() -> None:",
        "    if not _kb_object_coerce_enabled():",
        "        return",
        "    try:",
        "        import torch as _torch",
        "    except Exception:",
        "        return",
        "    _orig_tensor = _torch.tensor",
        "    def _tensor(data, *args, **kwargs):",
        "        return _orig_tensor(_kb_coerce_ndarray(data), *args, **kwargs)",
        "    _torch.tensor = _tensor",
        "    try:",
        "        _orig_as_tensor = _torch.as_tensor",
        "    except Exception:",
        "        _orig_as_tensor = None",
        "    if _orig_as_tensor is not None:",
        "        def _as_tensor(data, *args, **kwargs):",
        "            return _orig_as_tensor(_kb_coerce_ndarray(data), *args, **kwargs)",
        "        _torch.as_tensor = _as_tensor",
        "    try:",
        "        _orig_from_numpy = _torch.from_numpy",
        "    except Exception:",
        "        _orig_from_numpy = None",
        "    if _orig_from_numpy is not None:",
        "        def _from_numpy(arr):",
        "            return _orig_from_numpy(_kb_coerce_ndarray(arr))",
        "        _torch.from_numpy = _from_numpy",
        "",
        "_kb_patch_torch()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, OBJECT_COERCE_SHIM_MARKER, shim)


def inject_device_coerce_shim(kernel_dir: Path, context_dir: Path) -> None:
    coerce_path = context_dir / DEVICE_COERCE_FILENAME
    if not coerce_path.exists():
        return
    kernel_coerce_path = kernel_dir / DEVICE_COERCE_FILENAME
    copy_artifact_if_needed(source=coerce_path, destination=kernel_coerce_path)
    shim = [
        DEVICE_COERCE_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_device_coerce_enabled() -> bool:",
        "    candidates = [",
        f"        Path(__file__).with_name('{DEVICE_COERCE_FILENAME}'),",
        f"        Path('/kaggle/working/{DEVICE_COERCE_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                return True",
        "            if isinstance(payload, dict):",
        "                return bool(payload.get('enabled', True))",
        "            return True",
        "    return False",
        "",
        "def _kb_default_device():",
        "    try:",
        "        import torch as _torch",
        "    except Exception:",
        "        return None",
        "    if _torch.cuda.is_available():",
        "        return _torch.device('cuda')",
        "    return None",
        "",
        "def _kb_patch_torch_device() -> None:",
        "    if not _kb_device_coerce_enabled():",
        "        return",
        "    try:",
        "        import torch as _torch",
        "    except Exception:",
        "        return",
        "    device = _kb_default_device()",
        "    if device is None:",
        "        return",
        "    def _wrap_factory(fn):",
        "        def _wrapped(*args, **kwargs):",
        "            if 'device' not in kwargs:",
        "                kwargs['device'] = device",
        "            return fn(*args, **kwargs)",
        "        return _wrapped",
        "    factories = (",
        "        'tensor', 'as_tensor', 'from_numpy', 'zeros', 'ones', 'full', 'rand',",
        "        'randn', 'arange', 'zeros_like', 'ones_like', 'full_like',",
        "    )",
        "    for name in factories:",
        "        fn = getattr(_torch, name, None)",
        "        if fn is None:",
        "            continue",
        "        if name == 'from_numpy':",
        "            def _from_numpy(arr, _fn=fn):",
        "                out = _fn(arr)",
        "                try:",
        "                    return out.to(device)",
        "                except Exception:",
        "                    return out",
        "            setattr(_torch, name, _from_numpy)",
        "        else:",
        "            setattr(_torch, name, _wrap_factory(fn))",
        "",
        "    _orig_setattr = _torch.nn.Module.__setattr__",
        "    def _module_setattr(self, name, value):",
        "        if isinstance(value, _torch.Tensor):",
        "            try:",
        "                if value.device.type == 'cpu':",
        "                    value = value.to(device)",
        "            except Exception:",
        "                pass",
        "        return _orig_setattr(self, name, value)",
        "    _torch.nn.Module.__setattr__ = _module_setattr",
        "",
        "_kb_patch_torch_device()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, DEVICE_COERCE_SHIM_MARKER, shim)


def inject_zero_overlap_drift_shim(kernel_dir: Path, context_dir: Path) -> None:
    guard_path = context_dir / ZERO_OVERLAP_DRIFT_GUARD_FILENAME
    if not guard_path.exists():
        return
    kernel_guard_path = kernel_dir / ZERO_OVERLAP_DRIFT_GUARD_FILENAME
    copy_artifact_if_needed(source=guard_path, destination=kernel_guard_path)
    shim = [
        ZERO_OVERLAP_DRIFT_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_load_zero_overlap_drift_guard() -> dict:",
        "    candidates = [",
        f"        Path(__file__).with_name('{ZERO_OVERLAP_DRIFT_GUARD_FILENAME}'),",
        f"        Path('/kaggle/working/{ZERO_OVERLAP_DRIFT_GUARD_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if not path.exists():",
        "            continue",
        "        try:",
        "            payload = json.loads(path.read_text(encoding='utf-8'))",
        "        except Exception:",
        "            continue",
        "        if isinstance(payload, dict):",
        "            return payload",
        "    return {}",
        "",
        "def _kb_is_train_or_test_csv(path_value: object) -> bool:",
        "    try:",
        "        name = Path(str(path_value)).name.lower()",
        "    except Exception:",
        "        return False",
        "    return name in {'train.csv', 'test.csv'}",
        "",
        "def _kb_patch_zero_overlap_drift_drop() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    guard = _kb_load_zero_overlap_drift_guard()",
        "    if not guard or not bool(guard.get('enabled')):",
        "        return",
        "    raw_cols = guard.get('drop_columns')",
        "    if not isinstance(raw_cols, list):",
        "        return",
        "    drop_columns = [str(col) for col in raw_cols if str(col).strip()]",
        "    if not drop_columns:",
        "        return",
        "    _orig = _pd.read_csv",
        "",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        path_value = args[0] if args else kwargs.get('filepath_or_buffer')",
        "        if not _kb_is_train_or_test_csv(path_value):",
        "            return df",
        "        try:",
        "            cols = [col for col in drop_columns if col in df.columns]",
        "        except Exception:",
        "            cols = []",
        "        if not cols:",
        "            return df",
        "        try:",
        "            return df.drop(columns=cols)",
        "        except Exception:",
        "            return df",
        "",
        "    _pd.read_csv = _patched",
        "",
        "_kb_patch_zero_overlap_drift_drop()",
        "",
    ]
    append_sitecustomize_shim(kernel_dir, ZERO_OVERLAP_DRIFT_SHIM_MARKER, shim)


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
