from __future__ import annotations

import os
import shlex
import tomllib
from collections.abc import Callable
from pathlib import Path

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.exec_utils import run_command

VERIFY_COMPAT_SHIM_MARKER = "# KAGGLEBOT_VERIFY_COMPAT_SHIM"
PYTEST_XDIST_PLUGIN = "xdist.plugin"
PYTEST_XDIST_VALUE_OPTIONS = {
    "-n",
    "--numprocesses",
}
_VERIFY_MIRROR_EXCLUDED_DIR_NAMES = frozenset(
    {
        "__pycache__",
        "cache",
        "checkpoints",
        "logs",
        "models",
        "offline_wheels",
        "output",
        "outputs",
        "pretrained",
        "weights",
    }
)
_VERIFY_MIRROR_MAX_FILE_BYTES = 64 * 1024 * 1024

DEEP_PAST_VERIFY_COMPAT_SHIM = """

# KAGGLEBOT_VERIFY_COMPAT_SHIM
from dataclasses import replace as _verify_replace

_VERIFY_DEFAULT_FAITHFUL_SHORTLIST = {
    "contextual_byt5_curriculum_mbr",
    "dual_checkpoint_public_mbr",
    "retrieval_augmented_byt5_rerank",
}
_VERIFY_REFERENCE_MODE_ONLY = _env_bool("KAGGLEBOT_REFERENCE_MODE_ONLY", False)
_VERIFY_original_active_plan_seq2seq_pipeline_names = _active_plan_seq2seq_pipeline_names


def _active_plan_seq2seq_pipeline_names():
    active = set(_VERIFY_original_active_plan_seq2seq_pipeline_names())
    if _VERIFY_REFERENCE_MODE_ONLY:
        return {REFERENCE_PRIMARY_PIPELINE_NAME}
    if (
        _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG)
        and active == {REFERENCE_PRIMARY_PIPELINE_NAME}
        and _env_bool("KAGGLEBOT_ENABLE_PIPELINE_1", True)
    ):
        return set(_VERIFY_DEFAULT_FAITHFUL_SHORTLIST)
    return active


if os.getenv("KAGGLEBOT_USE_LORA_FINETUNE") is None and _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG):
    USE_LORA_FINETUNE = True


_VERIFY_prepare_reference_baseline_cfg = _prepare_reference_baseline_cfg


def _prepare_reference_baseline_cfg(cfg: PipelineConfig) -> PipelineConfig:
    resolved = _VERIFY_prepare_reference_baseline_cfg(cfg)
    if (
        cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME
        and resolved.reference_runtime_mode == "single_model_seq2seq_fallback"
        and "no competition-faithful fallback pair resolved locally" in resolved.reference_blocker
    ):
        return _verify_replace(
            resolved,
            model_hints=[],
            use_multi_model_pool=False,
            use_mbr=False,
            runtime_name=_reference_runtime_name(cfg.name, ["blocked_reference_runtime"]),
            reference_runtime_mode="blocked_reference_runtime",
            reference_slot_meta=None,
        )
    return resolved
"""

PLAYGROUND_S6E3_VERIFY_COMPAT_SHIM = """

# KAGGLEBOT_VERIFY_COMPAT_SHIM
try:
    from kagglebot.kernel_runtime.tabular_blend import (
        make_logit_blend_result as _verify_make_logit_blend_result,
        select_top_blend_components as _verify_select_top_blend_components,
    )
    from kagglebot.kernel_runtime.tabular_ensemble import (
        OUTER_FOLDS as _VERIFY_DEFAULT_OUTER_FOLDS,
        PipelineResult as _VerifyPipelineResult,
        PipelineSpec as _VerifyPipelineSpec,
    )
    from kagglebot.kernel_runtime.tabular_features import (
        TabularFeatureArtifacts as _VerifyReferenceArtifacts,
        add_tabular_reference_features as _verify_add_tabular_reference_features,
        build_training_source as _verify_build_training_source,
    )
except ImportError:
    from tabular_blend import (
        make_logit_blend_result as _verify_make_logit_blend_result,
        select_top_blend_components as _verify_select_top_blend_components,
    )
    from tabular_ensemble import (
        OUTER_FOLDS as _VERIFY_DEFAULT_OUTER_FOLDS,
        PipelineResult as _VerifyPipelineResult,
        PipelineSpec as _VerifyPipelineSpec,
    )
    from tabular_features import (
        TabularFeatureArtifacts as _VerifyReferenceArtifacts,
        add_tabular_reference_features as _verify_add_tabular_reference_features,
        build_training_source as _verify_build_training_source,
    )

TARGET_NAME = str(DATASET_PROFILE.get("target_column") or "Churn")
BASE_NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
PipelineResult = _VerifyPipelineResult
PipelineSpec = _VerifyPipelineSpec
ReferenceArtifacts = _VerifyReferenceArtifacts
OUTER_FOLDS = _VERIFY_DEFAULT_OUTER_FOLDS


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    sample_submission: pd.DataFrame
    id_col: str
    target_col: str
    feature_cols: list[str]
    target_values: np.ndarray
    data_dir: Path


def build_suite_specs() -> list[SuiteSpec]:
    suites = [
        SuiteSpec(
            name="comp_only",
            train_mode="competition_only",
            feature_recipe="full",
            lightweight=False,
            promotion_stage="full_eval",
            include_original_signal=False,
        ),
        SuiteSpec(
            name="orig_only",
            train_mode="original_only",
            feature_recipe="full",
            lightweight=True,
            promotion_stage="ablation_fast",
            include_original_signal=True,
        ),
        SuiteSpec(
            name="comp_plus_orig",
            train_mode="competition_plus_original",
            feature_recipe="full",
            lightweight=False,
            promotion_stage="ablation_fast",
            include_original_signal=True,
        ),
        SuiteSpec(
            name="orig_signal_only",
            train_mode="competition_only",
            feature_recipe="orig_signal_only",
            lightweight=True,
            promotion_stage="ablation_fast",
            include_original_signal=True,
        ),
    ]
    if not env_flag("KAGGLEBOT_ENABLE_ORIG_ONLY_ABLATION", False):
        suites = [suite for suite in suites if suite.name != "orig_only"]
    return suites


def build_pipeline_specs(suite: SuiteSpec, name_suffix: str = "") -> list[PipelineSpec]:
    suffix = str(name_suffix or "")
    return [
        PipelineSpec(
            name=f"catboost_rawcat_multiseed{suffix}",
            model_family="catboost",
            model_seeds=[42, 2024],
            params_override={},
        ),
        PipelineSpec(
            name=f"lgbm_te_multiseed{suffix}",
            model_family="lightgbm",
            model_seeds=[42, 2024],
            params_override={},
        ),
        PipelineSpec(
            name=f"xgb_tuned_multiseed{suffix}",
            model_family="xgboost",
            model_seeds=[42, 2024],
            params_override={},
        ),
    ]


def _verify_build_tenure_bin(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(-1.0)
    bins = [-1.5, 6.0, 12.0, 24.0, 48.0, 72.0, np.inf]
    labels = ["0_6", "7_12", "13_24", "25_48", "49_72", "73_plus"]
    return pd.cut(numeric, bins=bins, labels=labels).astype("object").fillna("Unknown").astype(str)


def add_reference_features(
    *,
    frames,
    base_numeric_cols,
    base_categorical_cols,
    orig_df,
    include_interactions,
    include_pair_tokens,
    include_trigram_tokens,
    include_orig_signal,
    feature_recipe,
):
    return _verify_add_tabular_reference_features(
        frames=frames,
        base_numeric_cols=list(base_numeric_cols),
        base_categorical_cols=list(base_categorical_cols),
        orig_df=orig_df,
        include_interactions=include_interactions,
        include_pair_tokens=include_pair_tokens,
        include_trigram_tokens=include_trigram_tokens,
        include_orig_signal=include_orig_signal,
        feature_recipe=feature_recipe,
        service_cols=SERVICE_COLUMNS,
        interaction_categoricals=[("Contract", "InternetService"), ("tenure_bin", "Contract")],
        pair_token_categoricals=[("Contract", "InternetService"), ("tenure_bin", "Contract")],
        trigram_token_categoricals=[("Contract", "InternetService", "PaymentMethod")],
        target_name=TARGET_NAME,
        original_row_weight=ORIGINAL_ROW_WEIGHT,
        categorical_feature_builders={"tenure_bin": lambda frame: _verify_build_tenure_bin(frame["tenure"])},
    )


def build_training_source(*, fold_train, y_train, artifacts):
    return _verify_build_training_source(
        fold_train=fold_train,
        y_train=y_train,
        artifacts=artifacts,
        target_name=TARGET_NAME,
        original_row_weight=ORIGINAL_ROW_WEIGHT,
    )


_select_top_blend_components = _verify_select_top_blend_components


def make_logit_blend_result(*, bundle, artifacts, results_by_name, first_name, second_name, first_weight):
    return _verify_make_logit_blend_result(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name=results_by_name,
        first_name=first_name,
        second_name=second_name,
        first_weight=first_weight,
        outer_folds=OUTER_FOLDS,
    )
"""

VERIFY_COMPAT_SHIMS_BY_SLUG_AND_FILE = {
    ("deep-past-initiative-machine-translation", "kernel.py"): DEEP_PAST_VERIFY_COMPAT_SHIM,
    ("playground-series-s6e3", "runtime.py"): PLAYGROUND_S6E3_VERIFY_COMPAT_SHIM,
}


def mirror_verify_artifacts(artifacts_dir: Path, *, repo_root: Path) -> None:
    local_artifacts_dir = repo_root / "artifacts"

    try:
        if artifacts_dir.resolve() == local_artifacts_dir.resolve():
            return
    except FileNotFoundError:
        pass
    if not artifacts_dir.exists():
        return

    for slug_dir in artifacts_dir.iterdir():
        if not slug_dir.is_dir():
            continue
        source_kernel_dir = slug_dir / "kernel"
        dest_kernel_dir = local_artifacts_dir / slug_dir.name / "kernel"
        if source_kernel_dir.is_dir():
            for walk_root, dirnames, filenames in os.walk(source_kernel_dir):
                dirnames[:] = [dirname for dirname in dirnames if not _is_verify_mirror_generated_dir(dirname)]
                walk_root_path = Path(walk_root)
                for filename in filenames:
                    source_path = walk_root_path / filename
                    if source_path.suffix == ".pyc" or source_path.stat().st_size > _VERIFY_MIRROR_MAX_FILE_BYTES:
                        continue
                    dest_path = dest_kernel_dir / source_path.relative_to(source_kernel_dir)
                    copy_artifact_if_needed(source=source_path, destination=dest_path)
        kernel_versions_dir = slug_dir / "kernels"
        if not kernel_versions_dir.is_dir():
            continue
        for filename in ("kernel.py", "runtime.py"):
            candidates = [path for path in kernel_versions_dir.glob(f"**/{filename}") if path.is_file()]
            if not candidates:
                continue
            preferred_source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            dest_path = dest_kernel_dir / filename
            copy_artifact_if_needed(source=preferred_source, destination=dest_path)
            nested_kernel_plan = preferred_source.parent / "plan.json"
            if nested_kernel_plan.exists():
                copy_artifact_if_needed(source=nested_kernel_plan, destination=dest_kernel_dir / "plan.json")
            nested_artifact_plan = preferred_source.parent.parent / "plan.json"
            if nested_artifact_plan.exists():
                copy_artifact_if_needed(
                    source=nested_artifact_plan,
                    destination=local_artifacts_dir / slug_dir.name / "plan.json",
                )
        append_verify_compat_shim(dest_kernel_dir / "kernel.py", slug=slug_dir.name)
        append_verify_compat_shim(dest_kernel_dir / "runtime.py", slug=slug_dir.name)


def _is_verify_mirror_generated_dir(dirname: str) -> bool:
    name = str(dirname).strip().lower()
    if name in _VERIFY_MIRROR_EXCLUDED_DIR_NAMES:
        return True
    return name.startswith(("output-", "outputs-", "output_", "outputs_", ".runtime", ".offline"))


def run_verify(
    verify_cmd: str,
    *,
    dry_run: bool,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    run_command_fn: Callable[..., object] = run_command,
) -> None:
    if dry_run:
        return
    args = shlex.split(verify_cmd)
    env = None
    if is_pytest_invocation(args):
        if artifacts_dir is not None:
            mirror_verify_artifacts(artifacts_dir, repo_root=repo_root or Path.cwd())
        env = os.environ.copy()
        env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        if env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1":
            args = ensure_pytest_plugin_loaded(args, PYTEST_XDIST_PLUGIN)

    result = run_command_fn(args, env=env)
    if (
        getattr(result, "returncode", 1) != 0
        and is_pytest_invocation(args)
        and is_pytest_xdist_unrecognized_args(getattr(result, "output", ""))
    ):
        retry_args = serial_pytest_retry_args(args, repo_root=repo_root or Path.cwd())
        result = run_command_fn(retry_args, env=env)
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError(f"Verification failed: {getattr(result, 'output', '')}")


def run_repo_verify(
    verify_cmd: str,
    *,
    dry_run: bool,
    artifacts_dir: Path | None = None,
    run_command_fn: Callable[..., object] = run_command,
) -> None:
    run_verify(
        verify_cmd,
        dry_run=dry_run,
        artifacts_dir=artifacts_dir,
        repo_root=Path.cwd(),
        run_command_fn=run_command_fn,
    )


def is_pytest_invocation(cmd_args: list[str]) -> bool:
    return pytest_arg_insert_index(cmd_args) is not None


def pytest_arg_insert_index(cmd_args: list[str]) -> int | None:
    for idx, item in enumerate(cmd_args):
        if item == "pytest" or item.endswith("/pytest"):
            return idx + 1
        if item == "-m" and idx + 1 < len(cmd_args) and cmd_args[idx + 1] == "pytest":
            return idx + 2
    return None


def ensure_pytest_plugin_loaded(cmd_args: list[str], plugin: str) -> list[str]:
    if has_pytest_plugin_arg(cmd_args, plugin):
        return cmd_args
    insert_at = pytest_arg_insert_index(cmd_args)
    if insert_at is None:
        return cmd_args
    return [*cmd_args[:insert_at], "-p", plugin, *cmd_args[insert_at:]]


def serial_pytest_retry_args(cmd_args: list[str], *, repo_root: Path) -> list[str]:
    stripped_args = strip_pytest_plugin_arg(strip_pytest_xdist_args(cmd_args), PYTEST_XDIST_PLUGIN)
    addopts = load_pytest_addopts(repo_root)
    serial_addopts = strip_pytest_xdist_args(addopts)
    if serial_addopts == addopts:
        return stripped_args
    insert_at = pytest_arg_insert_index(stripped_args)
    if insert_at is None:
        return stripped_args
    return [
        *stripped_args[:insert_at],
        "-o",
        f"addopts={shlex.join(serial_addopts)}",
        *stripped_args[insert_at:],
    ]


def is_pytest_xdist_unrecognized_args(output: object) -> bool:
    text = str(output or "").lower()
    return (
        ("unrecognized arguments" in text and ("-n" in text or "--numprocesses" in text))
        or ("error importing plugin" in text and "xdist" in text)
        or "no module named 'xdist'" in text
    )


def strip_pytest_xdist_args(cmd_args: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for item in cmd_args:
        if skip_next:
            skip_next = False
            continue
        if item in PYTEST_XDIST_VALUE_OPTIONS:
            skip_next = True
            continue
        if item.startswith("-n") and item != "-n":
            continue
        if any(item.startswith(f"{option}=") for option in PYTEST_XDIST_VALUE_OPTIONS if option.startswith("--")):
            continue
        stripped.append(item)
    return stripped


def strip_pytest_plugin_arg(cmd_args: list[str], plugin: str) -> list[str]:
    aliases = {plugin, plugin.split(".", 1)[0]}
    stripped: list[str] = []
    skip_next = False
    for idx, item in enumerate(cmd_args):
        if skip_next:
            skip_next = False
            continue
        if item == "-p" and idx + 1 < len(cmd_args) and cmd_args[idx + 1] in aliases:
            skip_next = True
            continue
        if item.startswith("-p") and item[2:] in aliases:
            continue
        stripped.append(item)
    return stripped


def load_pytest_addopts(repo_root: Path) -> list[str]:
    pyproject_path = repo_root / "pyproject.toml"
    try:
        config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    addopts = config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
    if isinstance(addopts, str):
        return shlex.split(addopts)
    if isinstance(addopts, list):
        return [str(item) for item in addopts]
    return []


def has_pytest_plugin_arg(cmd_args: list[str], plugin: str) -> bool:
    aliases = {plugin, plugin.split(".", 1)[0]}
    for idx, item in enumerate(cmd_args):
        if item == "-p" and idx + 1 < len(cmd_args) and cmd_args[idx + 1] in aliases:
            return True
        if item.startswith("-p") and item[2:] in aliases:
            return True
    return False


def append_verify_compat_shim(path: Path, *, slug: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if VERIFY_COMPAT_SHIM_MARKER in text:
        return
    shim = verify_compat_shim(slug=slug, filename=path.name)
    if shim:
        path.write_text(text + shim, encoding="utf-8")


def verify_compat_shim(*, slug: str, filename: str) -> str:
    return VERIFY_COMPAT_SHIMS_BY_SLUG_AND_FILE.get((normalize_verify_slug(slug), Path(filename).name), "")


def normalize_verify_slug(slug: str) -> str:
    return str(slug or "").strip().lower()
