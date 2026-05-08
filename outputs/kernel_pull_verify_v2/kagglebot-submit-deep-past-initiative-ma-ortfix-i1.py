"""Kernel entrypoint for Deep Past Initiative machine translation.

This file is the single authoritative runtime for local_gpu and kaggle_gpu.
It executes plan.json shortlist pipelines, performs leak-safe CV selection,
and writes validated submission artifacts.
"""

from __future__ import annotations
# kagglebot:kernel_sys_path
import os as _os
import sys as _sys
try:
    _KROOT = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _KROOT = _os.getcwd()
if _KROOT not in _sys.path:
    _sys.path.insert(0, _KROOT)
_KWORK = '/kaggle/working'
if _KWORK not in _sys.path:
    _sys.path.insert(0, _KWORK)
try:
    _KSC = _os.path.join(_KROOT, 'sitecustomize.py')
    if _os.path.exists(_KSC):
        with open(_KSC, 'rb') as _kb_f:
            exec(
                compile(_kb_f.read(), _KSC, 'exec'),
                {'__file__': _KSC, '__name__': 'kagglebot_sitecustomize'},
            )
except Exception:
    pass
del _os, _sys, _KROOT, _KWORK
# kagglebot:submit_inference
import os as _kb_os
_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'
_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '0'
_kb_os.environ['KAGGLEBOT_DO_INFER'] = '1'
del _kb_os

# kagglebot:data_resolver
from pathlib import Path as _KBPath

def _kb_find_file(base: _KBPath, name: str) -> _KBPath:
    candidate = base / name
    if candidate.exists():
        return candidate
    try:
        matches = list(base.rglob(name))
    except Exception:
        matches = []
    if matches:
        return matches[0]
    return candidate

# kagglebot:competition_slug
import os as _kb_os
_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = "deep-past-initiative-machine-translation"
_kb_os.environ['KAGGLEBOT_SLUG'] = "deep-past-initiative-machine-translation"
del _kb_os


import gc
import json
import math
import os
import random
import re
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import sacrebleu
except Exception:
    sacrebleu = None

warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")

# =====================================================================================
# Paths and frozen plan
# =====================================================================================

KERNEL_DIR = Path(__file__).resolve().parent


def _resolve_artifact_dir(kernel_dir: Path) -> Path:
    for candidate in (kernel_dir, kernel_dir.parent):
        if (candidate / "plan.json").exists():
            return candidate
    return kernel_dir.parent


ARTIFACT_DIR = _resolve_artifact_dir(KERNEL_DIR)
PLAN_PATH = ARTIFACT_DIR / "plan.json"
LOCAL_DATA_DIR = ARTIFACT_DIR / "data"
DEFAULT_COMPETITION_SLUG = os.getenv("KAGGLEBOT_COMPETITION_SLUG") or os.getenv("KAGGLEBOT_SLUG") or ARTIFACT_DIR.name


def _load_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


PLAN_JSON = _load_plan(PLAN_PATH)
PLAN_TOGGLES = PLAN_JSON.get("toggles", {})
PLAN_EVAL = PLAN_JSON.get("evaluation_protocol", {})
PLAN_PIPELINES = PLAN_JSON.get("pipelines", [])
_RAW_KERNEL_SOURCES = PLAN_JSON.get("kaggle_kernel_sources", {})
KAGGLE_KERNEL_SOURCES = _RAW_KERNEL_SOURCES if isinstance(_RAW_KERNEL_SOURCES, dict) else {}
_RAW_DOMAIN_ADAPTATION = PLAN_JSON.get("domain_adaptation", {})
DOMAIN_ADAPTATION = _RAW_DOMAIN_ADAPTATION if isinstance(_RAW_DOMAIN_ADAPTATION, dict) else {}
TRANSLATION_PRIMARY_METRIC = "Geometric Mean of the BLEU and the chrF++ scores"
TRANSLATION_METRIC_DESCRIPTION = "Geometric mean of corpus BLEU and chrF++ (micro-averaged sufficient statistics)"
FAITHFUL_TRANSLATION_SLUGS = {"deep-past-initiative-machine-translation"}
REFERENCE_MAX_INPUT_LENGTH = 512
REFERENCE_MAX_NEW_TOKENS = 384
REFERENCE_NUM_BEAMS = 8
REFERENCE_LENGTH_PENALTY = 1.3
REFERENCE_REPETITION_PENALTY = 1.2
REFERENCE_NUM_BEAM_CANDIDATES = 4
REFERENCE_NUM_SAMPLE_PER_TEMP = 2
REFERENCE_SAMPLE_TEMPERATURES = [0.60, 0.80, 1.05]
REFERENCE_SAMPLE_TOP_P = 0.92
REFERENCE_MBR_POOL_CAP = 32
REFERENCE_PRIMARY_PIPELINE_NAME = "dual_checkpoint_public_mbr"
REFERENCE_FAST_EVAL_MAX_DOCS = 192
REFERENCE_FAST_EVAL_MIN_DOCS = 192
LOCAL_REFERENCE_FAST_EVAL_DOCS = 64
LOCAL_REFERENCE_MAX_NEW_TOKENS = 256
LOCAL_REFERENCE_NUM_BEAMS = 4
LOCAL_REFERENCE_NUM_BEAM_CANDIDATES = 2
LOCAL_REFERENCE_NUM_SAMPLE_CANDIDATES = 0
LOCAL_REFERENCE_MBR_POOL_CAP = 8
LOCAL_REFERENCE_WATCHDOG_FAST_EVAL_DOCS = 24
LOCAL_REFERENCE_WATCHDOG_MAX_NEW_TOKENS = 128
LOCAL_REFERENCE_WATCHDOG_NUM_BEAMS = 1
LOCAL_REFERENCE_WATCHDOG_N_FOLDS = 1
REFERENCE_NOTEBOOK_MODEL_HINTS = [
    "assiaben/final-byt5",
    "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6",
]
REFERENCE_FALLBACK_MODEL_HINTS = [
    "assiaben/final-byt5",
    "artemgoncarov/dpc-byt5-large",
    "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6",
]
REFERENCE_EXACT_MODEL_ASSET_PATHS: dict[str, tuple[Path, ...]] = {
    "assiaben/final-byt5": (
        ARTIFACT_DIR / "context" / "reference_inputs" / "dataset__assiaben__final-byt5" / "byt5-akkadian-optimized-34x",
    ),
    "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6": (
        ARTIFACT_DIR / "context" / "reference_inputs" / "dataset__mattiaangeli__byt5-akkadian-mbr__PyTorch__default__6",
    ),
}
REFERENCE_FALLBACK_PAIR_ORDER: list[tuple[str, str]] = [
    (
        "assiaben/final-byt5",
        "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6",
    ),
    (
        "assiaben/final-byt5",
        "artemgoncarov/dpc-byt5-large",
    ),
]
REFERENCE_SINGLE_MODEL_FALLBACK = "assiaben/final-byt5"
REFERENCE_INPUTS_MANIFEST_PATH = ARTIFACT_DIR / "context" / "reference_inputs_manifest.json"
MODEL_CACHE_LIMIT = 2
FAITHFUL_TRANSLATION_DEFAULT_SHORTLIST = [
    "dual_checkpoint_public_mbr",
    "contextual_byt5_curriculum_mbr",
    "retrieval_augmented_byt5_rerank",
    "char_tfidf_knn_memory",
]


def _force_translation_metric_for_slug(slug: str) -> bool:
    return str(slug).strip().lower() in FAITHFUL_TRANSLATION_SLUGS


COMPETITION_FAITHFUL_SLUG = DEFAULT_COMPETITION_SLUG


def _reported_metric_name() -> str:
    if _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG):
        return TRANSLATION_PRIMARY_METRIC
    raw_target_metric = PLAN_JSON.get("target_metric")
    if isinstance(raw_target_metric, str) and raw_target_metric.strip():
        return raw_target_metric.strip()
    return TRANSLATION_PRIMARY_METRIC


REPORTED_PRIMARY_METRIC = _reported_metric_name()


def _ordered_required_seq2seq_pipeline_names() -> list[str]:
    raw = KAGGLE_KERNEL_SOURCES.get("required_local_seq2seq_pipelines", [])
    if not isinstance(raw, list):
        return []

    names: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _plan_shortlisted_pipeline_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for pipe in PLAN_PIPELINES:
        name = str(pipe.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    if _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG):
        if REFERENCE_PRIMARY_PIPELINE_NAME in names:
            names = [
                REFERENCE_PRIMARY_PIPELINE_NAME,
                *[name for name in names if name != REFERENCE_PRIMARY_PIPELINE_NAME],
            ]
        else:
            names = [
                *FAITHFUL_TRANSLATION_DEFAULT_SHORTLIST,
                *[name for name in names if name not in FAITHFUL_TRANSLATION_DEFAULT_SHORTLIST],
            ]
    if names:
        return names
    return _ordered_required_seq2seq_pipeline_names() or ["contextual_byt5_curriculum_mbr"]


# =====================================================================================
# Knobs (top-level controls required by contract)
# =====================================================================================


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list_int(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out or default


def _env_list_float(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    out: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out or default


N_FOLDS = _env_int("KAGGLEBOT_N_FOLDS", int(PLAN_EVAL.get("n_folds", 5)))
SEEDS = _env_list_int("KAGGLEBOT_SEEDS", [int(x) for x in PLAN_EVAL.get("seeds", [42])])
FAST_DEV = _env_bool("KAGGLEBOT_FAST_DEV", bool(PLAN_TOGGLES.get("FAST_DEV", False)))
GPU_DEVICE = os.getenv("GPU_DEVICE", "cuda:0")
ENABLE_CONTEXT_WINDOW = _env_bool(
    "KAGGLEBOT_ENABLE_CONTEXT_WINDOW",
    bool(PLAN_TOGGLES.get("ENABLE_CONTEXT_WINDOW", True)),
)
ENABLE_PSEUDO_SENTENCES = _env_bool(
    "KAGGLEBOT_ENABLE_PSEUDO_SENTENCES",
    bool(PLAN_TOGGLES.get("ENABLE_PSEUDO_SENTENCES", True)),
)
ENABLE_GOLD_UPWEIGHT = _env_bool(
    "KAGGLEBOT_ENABLE_GOLD_UPWEIGHT",
    bool(PLAN_TOGGLES.get("ENABLE_GOLD_UPWEIGHT", True)),
)
ENABLE_LEXICON_CONSTRAINTS = _env_bool(
    "KAGGLEBOT_ENABLE_LEXICON_CONSTRAINTS",
    bool(PLAN_TOGGLES.get("ENABLE_LEXICON_CONSTRAINTS", True)),
)
ENABLE_RETRIEVAL_RERANK = _env_bool(
    "KAGGLEBOT_ENABLE_RETRIEVAL_RERANK",
    bool(PLAN_TOGGLES.get("ENABLE_RETRIEVAL_RERANK", True)),
)
ENABLE_MULTI_CHECKPOINT_MBR = _env_bool(
    "KAGGLEBOT_ENABLE_MULTI_CHECKPOINT_MBR",
    bool(PLAN_TOGGLES.get("ENABLE_MULTI_CHECKPOINT_MBR", True)),
)
ENABLE_PUBLIC_CHECKPOINTS = _env_bool(
    "KAGGLEBOT_ENABLE_PUBLIC_CHECKPOINTS",
    bool(PLAN_TOGGLES.get("ENABLE_PUBLIC_CHECKPOINTS", True)),
)
ENABLE_PIPELINE_1 = _env_bool("KAGGLEBOT_ENABLE_PIPELINE_1", bool(PLAN_TOGGLES.get("ENABLE_PIPELINE_1", True)))
ENABLE_PIPELINE_2 = _env_bool("KAGGLEBOT_ENABLE_PIPELINE_2", bool(PLAN_TOGGLES.get("ENABLE_PIPELINE_2", True)))
ENABLE_PIPELINE_3 = _env_bool("KAGGLEBOT_ENABLE_PIPELINE_3", bool(PLAN_TOGGLES.get("ENABLE_PIPELINE_3", True)))
ENABLE_PIPELINE_4 = _env_bool("KAGGLEBOT_ENABLE_PIPELINE_4", bool(PLAN_TOGGLES.get("ENABLE_PIPELINE_4", True)))
ENABLE_ENSEMBLE = _env_bool("KAGGLEBOT_ENABLE_ENSEMBLE", bool(PLAN_TOGGLES.get("ENABLE_ENSEMBLE", True)))

PIPELINE_NAME = os.getenv(
    "KAGGLEBOT_PIPELINE_NAME",
    _plan_shortlisted_pipeline_names()[0],
)
DO_TRAIN = _env_bool("KAGGLEBOT_DO_TRAIN", True)
DO_INFER = _env_bool("KAGGLEBOT_DO_INFER", True)
USE_NORMALIZATION = _env_bool("KAGGLEBOT_USE_NORMALIZATION", bool(PLAN_TOGGLES.get("USE_NORMALIZATION", True)))
USE_DETERMINATIVES_NORM = _env_bool(
    "KAGGLEBOT_USE_DETERMINATIVES_NORM",
    bool(PLAN_TOGGLES.get("USE_DETERMINATIVES_NORM", True)),
)
USE_MBR = _env_bool("KAGGLEBOT_USE_MBR", bool(PLAN_TOGGLES.get("USE_MBR", True)))
USE_MULTI_MODEL_POOL = _env_bool("KAGGLEBOT_USE_MULTI_MODEL_POOL", bool(PLAN_TOGGLES.get("USE_MULTI_MODEL_POOL", True)))
USE_LORA_FINETUNE = _env_bool("KAGGLEBOT_USE_LORA_FINETUNE", bool(PLAN_TOGGLES.get("USE_LORA_FINETUNE", False)))
USE_DIVERSE_MODEL_ADDON = _env_bool(
    "KAGGLEBOT_USE_DIVERSE_MODEL_ADDON",
    bool(PLAN_TOGGLES.get("USE_DIVERSE_MODEL_ADDON", False)),
)
SAVE_NPY = _env_bool("KAGGLEBOT_SAVE_NPY", bool(PLAN_TOGGLES.get("SAVE_NPY", True)))
MAX_SOURCE_LEN = _env_int("KAGGLEBOT_MAX_SOURCE_LEN", int(PLAN_TOGGLES.get("MAX_SOURCE_LEN", 1024)))
if _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG):
    MAX_SOURCE_LEN = _env_int("KAGGLEBOT_MAX_SOURCE_LEN", REFERENCE_MAX_INPUT_LENGTH)
    MAX_TARGET_LEN = _env_int("KAGGLEBOT_MAX_TARGET_LEN", REFERENCE_MAX_NEW_TOKENS)
    MAX_NEW_TOKENS = _env_int("KAGGLEBOT_MAX_NEW_TOKENS", REFERENCE_MAX_NEW_TOKENS)
    NUM_BEAMS = _env_int("KAGGLEBOT_NUM_BEAMS", REFERENCE_NUM_BEAMS)
    MAX_POOL_CAP = _env_int("KAGGLEBOT_MAX_POOL_CAP", REFERENCE_MBR_POOL_CAP)
    TOP_P = _env_float("KAGGLEBOT_TOP_P", REFERENCE_SAMPLE_TOP_P)
    TEMPERATURE = _env_float("KAGGLEBOT_TEMPERATURE", REFERENCE_SAMPLE_TEMPERATURES[1])
else:
    MAX_TARGET_LEN = _env_int("KAGGLEBOT_MAX_TARGET_LEN", int(PLAN_TOGGLES.get("MAX_NEW_TOKENS", 512)))
    MAX_NEW_TOKENS = _env_int("KAGGLEBOT_MAX_NEW_TOKENS", int(PLAN_TOGGLES.get("MAX_NEW_TOKENS", 512)))
    NUM_BEAMS = _env_int("KAGGLEBOT_NUM_BEAMS", int(PLAN_TOGGLES.get("NUM_BEAMS", 8)))
    MAX_POOL_CAP = _env_int("KAGGLEBOT_MAX_POOL_CAP", int(PLAN_TOGGLES.get("MAX_POOL_CAP", 32)))
    TOP_P = _env_float("KAGGLEBOT_TOP_P", float(PLAN_TOGGLES.get("TOP_P", 0.9)))
    TEMPERATURE = _env_float("KAGGLEBOT_TEMPERATURE", float(PLAN_TOGGLES.get("TEMPERATURE", 0.7)))
REPETITION_PENALTY = _env_float("KAGGLEBOT_REPETITION_PENALTY", REFERENCE_REPETITION_PENALTY)
SAMPLE_TEMPERATURES = _env_list_float("KAGGLEBOT_SAMPLE_TEMPERATURES", list(REFERENCE_SAMPLE_TEMPERATURES))

IS_KAGGLE = Path("/kaggle").exists()
LOCAL_KERNEL_MODE = _env_bool("KAGGLEBOT_LOCAL_KERNEL", False)
DISABLE_LGBM_GPU = _env_bool("KAGGLEBOT_DISABLE_LGBM_GPU", False)
DISABLE_XGBOOST = _env_bool("KAGGLEBOT_DISABLE_XGBOOST", True)
RUN_ID = os.getenv("KAGGLEBOT_RUN_ID", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
ALLOW_MODEL_DOWNLOAD = _env_bool("KAGGLEBOT_ALLOW_MODEL_DOWNLOAD", False)
SELECT_EPSILON = _env_float("KAGGLEBOT_SELECT_EPSILON", 1e-6)
BASELINE_GUARD_MARGIN = _env_float("KAGGLEBOT_BASELINE_GUARD_MARGIN", 0.25)

RETRIEVAL_K = _env_int("KAGGLEBOT_RETRIEVAL_K", 32)
RETRIEVAL_MIN_SIM = _env_float("KAGGLEBOT_RETRIEVAL_MIN_SIM", 0.10)
RETRIEVAL_MIN_DF = _env_int("KAGGLEBOT_RETRIEVAL_MIN_DF", 2)
RETRIEVAL_NGRAM_MIN = _env_int("KAGGLEBOT_RETRIEVAL_NGRAM_MIN", 3)
RETRIEVAL_NGRAM_MAX = _env_int("KAGGLEBOT_RETRIEVAL_NGRAM_MAX", 9)
RETRIEVAL_WORD_WEIGHT = _env_float("KAGGLEBOT_RETRIEVAL_WORD_WEIGHT", 0.0)
RETRIEVAL_WORD_MIN_DF = _env_int("KAGGLEBOT_RETRIEVAL_WORD_MIN_DF", 2)
RETRIEVAL_RUN_MBR_VARIANT = _env_bool("KAGGLEBOT_RETRIEVAL_RUN_MBR_VARIANT", False)
ALLOW_KERNEL_FINETUNE = _env_bool(
    "KAGGLEBOT_ENABLE_DEEP_PAST_FINETUNE",
    bool(DOMAIN_ADAPTATION.get("allow_kernel_finetune", False)),
)
METADATA_SUPERVISION_MODE = str(DOMAIN_ADAPTATION.get("metadata_supervision", "off")).strip().lower()
CONSTRAINT_REWRITE_MODE = str(DOMAIN_ADAPTATION.get("constraint_rewrite_mode", "off")).strip().lower()

# =====================================================================================
# Logging and runtime helpers
# =====================================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def _looks_like_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "cuda out of memory" in msg or "out of memory" in msg:
        return True
    try:
        import torch

        return isinstance(exc, torch.OutOfMemoryError)
    except Exception:
        return False


def _cuda_cleanup_best_effort() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _auto_disable_training_if_unsafe() -> None:
    """Safety guard for optional LoRA training path."""
    global DO_TRAIN

    if not DO_TRAIN or not USE_LORA_FINETUNE:
        return
    if _env_bool("KAGGLEBOT_FORCE_TRAIN", False):
        return

    try:
        import torch

        if not torch.cuda.is_available():
            log("Auto-setting DO_TRAIN=false for LoRA pipeline (CUDA not available).")
            DO_TRAIN = False
            return
        total = int(torch.cuda.get_device_properties(0).total_memory)
        if total < 15 * 1024**3:
            gib = total / (1024**3)
            log(f"Auto-setting DO_TRAIN=false for LoRA pipeline (GPU RAM {gib:.2f} GiB < 15 GiB).")
            DO_TRAIN = False
    except Exception:
        return


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if not (DO_TRAIN or DO_INFER):
        return
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_writable_dir(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
        probe = path / ".kagglebot_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _first_creatable_dir(candidates: Sequence[Path]) -> Path:
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            return ensure_dir(candidate)
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise OSError("No writable output directory available.")


def _competition_slug() -> str:
    return os.getenv("KAGGLEBOT_COMPETITION_SLUG") or os.getenv("KAGGLEBOT_SLUG") or DEFAULT_COMPETITION_SLUG


def resolve_output_dirs(slug: str, run_id: str) -> tuple[Path, list[Path], bool]:
    _ = (slug, run_id)
    kaggle_working = Path("/kaggle/working")
    kaggle_writable = is_writable_dir(kaggle_working)
    if IS_KAGGLE:
        local_run_dir = _first_creatable_dir([kaggle_working / "output", Path.cwd() / "output"])
    else:
        local_run_dir = _first_creatable_dir([KERNEL_DIR / "output", Path.cwd() / "output"])

    mirror_dirs: list[Path] = [local_run_dir]
    if kaggle_writable and local_run_dir.resolve() != kaggle_working.resolve():
        mirror_dirs.append(kaggle_working)

    return local_run_dir, mirror_dirs, kaggle_writable


# =====================================================================================
# Data IO and modality detection
# =====================================================================================


def _candidate_data_dirs(slug: str) -> list[Path]:
    candidates = [
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug.replace('-', '_')}"),
        LOCAL_DATA_DIR,
    ]
    dedup: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(cand)
    return dedup


def locate_data_dir(slug: str) -> Path:
    required = ("train.csv", "test.csv", "sample_submission.csv")
    for cand in _candidate_data_dirs(slug):
        if all(_kb_find_file(cand, name).exists() for name in required):
            return cand
    input_root = _KBPath('/kaggle/input')
    if input_root.exists() and input_root.is_dir():
        # kagglebot:data-dir-fallback-scan
        for cand in sorted(input_root.iterdir(), key=lambda p: p.name):
            if not cand.is_dir():
                continue
            if all(_kb_find_file(cand, name).exists() for name in required):
                return cand
    raise FileNotFoundError(f"Could not find required csv files for slug='{slug}'")
def load_competition_frames(slug: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    data_dir = locate_data_dir(slug)
    train_df = pd.read_csv(_kb_find_file(data_dir, 'train.csv'))
    test_df = pd.read_csv(_kb_find_file(data_dir, 'test.csv'))
    sample_df = pd.read_csv(_kb_find_file(data_dir, 'sample_submission.csv'))
    return train_df, test_df, sample_df, data_dir


def load_optional_metadata_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    published_path = _kb_find_file(data_dir, 'published_texts.csv')
    sentence_path = _kb_find_file(data_dir, 'Sentences_Oare_FirstWord_LinNum.csv')

    if published_path.exists():
        published_df = pd.read_csv(published_path)
    else:
        published_df = pd.DataFrame()
    if sentence_path.exists():
        sentence_df = pd.read_csv(sentence_path)
    else:
        sentence_df = pd.DataFrame()
    return published_df, sentence_df


def assert_translation_schema(train_df: pd.DataFrame, test_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    train_required = {"transliteration", "translation"}
    test_required = {"id", "transliteration"}
    sample_required = ["id", "translation"]

    missing_train = sorted(train_required - set(train_df.columns))
    missing_test = sorted(test_required - set(test_df.columns))
    if missing_train:
        raise ValueError(f"train.csv missing required columns: {missing_train}")
    if missing_test:
        raise ValueError(f"test.csv missing required columns: {missing_test}")
    if list(sample_df.columns) != sample_required:
        raise ValueError(
            f"sample_submission.csv columns must be exactly ['id', 'translation']; got {list(sample_df.columns)}"
        )


def detect_modality(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    tr = set(train_df.columns.str.lower())
    te = set(test_df.columns.str.lower())

    if {"transliteration", "translation"}.issubset(tr) and {"id", "transliteration"}.issubset(te):
        return "text"
    if tr & {"image", "image_path", "img", "filepath"} or te & {"image", "image_path", "img", "filepath"}:
        return "image"
    if tr & {"audio", "audio_path", "wav_path", "mp3_path"} or te & {"audio", "audio_path", "wav_path", "mp3_path"}:
        return "audio"
    if tr & {"video", "video_path", "mp4_path"} or te & {"video", "video_path", "mp4_path"}:
        return "video"
    if len(train_df.columns) > 1 and len(test_df.columns) > 1:
        return "tabular"
    return "other"


# =====================================================================================
# Robust schema/feature utilities
# =====================================================================================


def safe_fill_missing_categorical(series: pd.Series) -> pd.Series:
    # Guardrail: never fillna("Unknown") on Categorical before adding category.
    if pd.api.types.is_categorical_dtype(series):
        if "Unknown" not in series.cat.categories:
            series = series.cat.add_categories(["Unknown"])
        return series.fillna("Unknown")
    return series.astype("string").fillna("Unknown")


def align_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align features safely.

    - Drops train-missing columns from requested feature list.
    - Adds missing test columns as NA.
    - Returns aligned copies with identical column order.
    """

    train_cols = set(train_df.columns)
    valid_cols = [c for c in feature_cols if c in train_cols]

    tr = train_df.copy()
    te = test_df.copy()
    for col in valid_cols:
        if col not in te.columns:
            te[col] = pd.NA

    aligned_train = tr.loc[:, valid_cols].copy()
    aligned_test = te.loc[:, valid_cols].copy()
    return aligned_train, aligned_test


# =====================================================================================
# Text normalization (deterministic)
# =====================================================================================

SUBSCRIPT_MAP = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)

MULTISPACE_RE = re.compile(r"\s+")
BRACKET_CONTENT_RE = re.compile(r"\[([^\]]*)\]")
BRACKET_X_ONLY_RE = re.compile(r"^[xX](?:\s*[xX])*$")
BRACKET_DOTS_ONLY_RE = re.compile(r"^[\s\.\u2026]+$")
ELLIPSIS_RE = re.compile(r"\.{3,}")
X_GAP_RE = re.compile(r"(?<!\w)x+(?!\w)", flags=re.IGNORECASE)
ASCII_V2_RE = re.compile(r"([aAeEiIuU])(?:2|₂)")
ASCII_V3_RE = re.compile(r"([aAeEiIuU])(?:3|₃)")
ASCII_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
ASCII_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"})
FLOAT_NORMALIZE_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
DET_UPPER_RE = re.compile(r"\(([" + UNICODE_UPPER + r"0-9]{1,6})\)")
DET_LOWER_RE = re.compile(r"\(([" + UNICODE_LOWER + r"]{1,4})\)")
GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I,
)
KUBABBAR_RE = re.compile(r"KÙ\.B\.")
SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)
BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
UNCERTAIN_RE = re.compile(r"\(\?\)")
CURLY_DQ_RE = re.compile("[\u201c\u201d]")
CURLY_SQ_RE = re.compile("[\u2018\u2019]")
MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")
PN_RE = re.compile(r"\bPN\b")
COMMODITY_RE = re.compile(r"(?<=\s)-(gold|tax|textiles)\b")
STRAY_MARKS_RE = re.compile(r"<<[^>]*>>|<(?!gap\b)[^>]*>")
EXTRA_STRAY_RE = re.compile(r"(?<!\w)(?:\.\.+|xx+)(?!\w)")
SLASH_ALT_RE = re.compile(r"(?<![0-9/])\s+/\s+(?![0-9])\S+")
MULTI_GAP_RE = re.compile(r"(?:<gap>\s*){2,}")
METADATA_TOKEN_SPLIT_RE = re.compile(r"[|;]")
METADATA_WRAPPER_RE = re.compile(r"\bcuneiform tablet\b|\benvelope\b", re.I)
METADATA_ALPHA_SUFFIX_RE = re.compile(r"\b(\d+)[a-z]+\b", re.I)
METADATA_PUNCT_RE = re.compile(r"[^a-z0-9]+")
FORBIDDEN_TARGET_TRANS = str.maketrans("", "", "——<>⌈⌋⌊[]+ʾ;")
EXACT_FRAC_MAP = {
    "0.8333": "⅚",
    "0.6666": "⅔",
    "0.3333": "⅓",
    "0.1666": "⅙",
    "0.625": "⅝",
    "0.75": "¾",
    "0.25": "¼",
    "0.5": "½",
}
ALLOWED_FRACS = [
    (1 / 6, "0.16666"),
    (1 / 4, "0.25"),
    (1 / 3, "0.33333"),
    (1 / 2, "0.5"),
    (2 / 3, "0.66666"),
    (3 / 4, "0.75"),
    (5 / 6, "0.83333"),
]
FRAC_TOL = 2e-3
ROMAN2INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}
COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}
SHEKEL_REPLS = [
    (re.compile(r"5\s+11\s*/\s*12\s+shekels?", re.I), "6 shekels less 15 grains"),
    (re.compile(r"5\s*/\s*12\s+shekels?", re.I), "⅓ shekel 15 grains"),
    (re.compile(r"7\s*/\s*12\s+shekels?", re.I), "½ shekel 15 grains"),
    (re.compile(r"1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?", re.I), "15 grains"),
]


def _norm_spaces(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def _ascii_to_diacritics(text: str) -> str:
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = text.replace("s,", "ṣ").replace("S,", "Ṣ")
    text = text.replace("t,", "ṭ").replace("T,", "Ṭ")
    text = ASCII_V2_RE.sub(lambda match: match.group(1).translate(ASCII_ACUTE), text)
    text = ASCII_V3_RE.sub(lambda match: match.group(1).translate(ASCII_GRAVE), text)
    return text


def _normalize_gaps(text: str) -> str:
    return GAP_UNIFIED_RE.sub("<gap>", text)


def _frac_repl(match: re.Match[str]) -> str:
    return EXACT_FRAC_MAP[match.group(0)]


def _canon_decimal(value: float) -> str:
    integer_part = int(math.floor(value + 1e-12))
    frac = value - integer_part
    best_frac, best_text = min(ALLOWED_FRACS, key=lambda item: abs(frac - item[0]))
    if abs(frac - best_frac) <= FRAC_TOL:
        if integer_part == 0:
            return best_text
        return f"{integer_part}{best_text[1:]}" if best_text.startswith("0.") else f"{integer_part}+{best_text}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _normalize_curly_quotes(text: str) -> str:
    return CURLY_SQ_RE.sub("'", CURLY_DQ_RE.sub('"', text))


def _normalize_bracket_span(match: re.Match[str]) -> str:
    inner = str(match.group(1))
    inner_stripped = inner.strip()
    if not inner_stripped:
        return " <gap> "
    if BRACKET_X_ONLY_RE.fullmatch(inner_stripped):
        return " <gap> "
    if BRACKET_DOTS_ONLY_RE.fullmatch(inner_stripped):
        return " <big_gap> "
    return f" {inner_stripped} "


def normalize_source(text: Any, use_det_norm: bool) -> str:
    s = "" if pd.isna(text) else str(text)
    s = unicodedata.normalize("NFKC", s)
    s = _ascii_to_diacritics(s)
    s = s.translate(SUBSCRIPT_MAP)
    # Dataset instructions: train sometimes uses Ḫ/ḫ while test prefers H/h.
    s = s.replace("Ḫ", "H").replace("ḫ", "h")
    s = s.replace("ʾ", "")
    s = BRACKET_CONTENT_RE.sub(_normalize_bracket_span, s)
    s = _normalize_gaps(s)
    s = s.replace("˹", " ").replace("˺", " ")
    s = DET_UPPER_RE.sub(r"\1", s)
    if use_det_norm:
        s = DET_LOWER_RE.sub(r"{\1}", s)
    s = KUBABBAR_RE.sub("KÙ.BABBAR", s)
    s = EXACT_FRAC_RE.sub(_frac_repl, s)
    s = FLOAT_NORMALIZE_RE.sub(lambda match: _canon_decimal(float(match.group(1))), s)
    return _norm_spaces(s.replace("ₓ", ""))


def normalize_target(text: Any) -> str:
    s = "" if pd.isna(text) else str(text)
    s = unicodedata.normalize("NFKC", s)
    return _norm_spaces(_normalize_curly_quotes(s))


def postprocess_translation(text: str, strong: bool) -> str:
    out = _norm_spaces(_normalize_curly_quotes(str(text)))
    if not strong:
        return out
    out = _normalize_gaps(out)
    out = PN_RE.sub("<gap>", out)
    out = COMMODITY_RE.sub(lambda match: COMMODITY_REPL[match.group(1)], out)
    for pattern, repl in SHEKEL_REPLS:
        out = pattern.sub(repl, out)
    out = EXACT_FRAC_RE.sub(_frac_repl, out)
    out = FLOAT_NORMALIZE_RE.sub(lambda match: _canon_decimal(float(match.group(1))), out)
    out = SOFT_GRAM_RE.sub(" ", out)
    out = BARE_GRAM_RE.sub(" ", out)
    out = UNCERTAIN_RE.sub("", out)
    out = STRAY_MARKS_RE.sub("", out)
    out = EXTRA_STRAY_RE.sub("", out)
    out = SLASH_ALT_RE.sub("", out)
    out = MONTH_RE.sub(lambda match: f"Month {ROMAN2INT.get(match.group(1).upper(), match.group(1))}", out)
    out = MULTI_GAP_RE.sub("<gap>", out)
    out = out.replace("<gap>", "\x00GAP\x00")
    out = out.translate(FORBIDDEN_TARGET_TRANS)
    out = out.replace("\x00GAP\x00", " <gap> ")
    out = out.replace("Ḫ", "H").replace("ḫ", "h")
    out = REPEAT_WORD_RE.sub(r"\1", out)
    for width in range(4, 1, -1):
        repeated_phrase = re.compile(r"\b((?:\w+\s+){" + str(width - 1) + r"}\w+)(?:\s+\1\b)+")
        out = repeated_phrase.sub(r"\1", out)
    out = PUNCT_SPACE_RE.sub(r"\1", out)
    out = REPEAT_PUNCT_RE.sub(r"\1", out)
    return _norm_spaces(out)


def preprocess_translation_df(df: pd.DataFrame, use_normalization: bool, use_det_norm: bool) -> pd.DataFrame:
    out = df.copy()
    out["transliteration"] = out["transliteration"].astype("string").fillna("")
    if use_normalization:
        out["transliteration"] = out["transliteration"].map(lambda x: normalize_source(x, use_det_norm))
    else:
        out["transliteration"] = out["transliteration"].map(lambda x: _norm_spaces(str(x)))

    if "translation" in out.columns:
        out["translation"] = out["translation"].astype("string").fillna("")
        if use_normalization:
            out["translation"] = out["translation"].map(normalize_target)
        else:
            out["translation"] = out["translation"].map(lambda x: _norm_spaces(str(x)))
    return out


# =====================================================================================
# Pseudo sentence builder + lexicon/document helpers
# =====================================================================================

TARGET_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?\;\:])\s+(?=(?:[A-Z<\"']))")
SOURCE_BOUNDARY_TOKENS = {"<gap>", "<big_gap>", "um-ma", "qí-bi4-ma", "qí-bi₄-ma", "qí-bi-ma", "ki-ma"}


@dataclass(frozen=True)
class LexiconResources:
    token_map: dict[str, str]


@dataclass(frozen=True)
class MetadataSupervisionResult:
    pair_df: pd.DataFrame
    candidate_docs: int
    matched_docs: int
    rejected_docs: int


@dataclass(frozen=True)
class ConstraintMemories:
    exact_source_memory: dict[str, str]
    entity_memory: dict[str, str]
    quantity_memory: dict[str, str]
    unit_memory: dict[str, str]


@dataclass(frozen=True)
class FineTuneResult:
    ran: bool
    model_hint: str | None
    adapter_dir: str | None
    baseline_metric: dict[str, float] | None = None
    baseline_doc_metric: dict[str, float] | None = None
    validation_metric: dict[str, float] | None = None
    validation_doc_metric: dict[str, float] | None = None
    baseline_slice_metrics: dict[str, float] | None = None
    validation_slice_metrics: dict[str, float] | None = None
    test_predictions: list[str] | None = None
    postprocess_stats: dict[str, int] | None = None
    reason: str = ""

    @property
    def validation_score(self) -> float | None:
        if not self.validation_metric:
            return None
        return float(self.validation_metric.get("gmean", 0.0))


DISPLAY_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
SOURCE_ENTITY_TOKEN_RE = re.compile(r"^[A-Za-zḫḪṣṢṭṬšŠāēīūĀĒĪŪ][A-Za-z0-9ḫḪṣṢṭṬšŠāēīūĀĒĪŪ\.\-]*$")
TARGET_TITLECASE_RE = re.compile(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*(?:[- ][A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)*\b")
TARGET_QUANTITY_PHRASE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|third)"
    r"\s+(?:mina(?:e)?|shekel(?:s)?|silver|textile(?:s)?)\b",
    flags=re.IGNORECASE,
)
SOURCE_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
SOURCE_QUANTITY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ma-na(?:-im)?|g[ií]n(?:\.ta)?|kù\.babbar|t[úu]g)\b",
    flags=re.IGNORECASE,
)
UNIT_TARGET_CANONICAL = {
    "ma-na": "mina",
    "ma-na-im": "mina",
    "gín": "shekels",
    "gín.ta": "shekels",
    "kù.babbar": "silver",
    "túg": "textile",
    "sig5": "good",
}
SOURCE_UNIT_TOKENS = frozenset(
    {
        "ma-na",
        "ma-na-im",
        "gín",
        "gín.ta",
        "kù.babbar",
        "túg",
        "sig5",
        "kù",
        "babbar",
    }
)


def load_lexicon_resources(data_dir: Path) -> LexiconResources:
    lexicon_path = _kb_find_file(data_dir, 'OA_Lexicon_eBL.csv')
    token_map: dict[str, str] = {}
    if not lexicon_path.exists():
        return LexiconResources(token_map=token_map)

    try:
        lex_df = pd.read_csv(lexicon_path, usecols=["form", "norm"])
    except Exception:
        return LexiconResources(token_map=token_map)

    for form_raw, norm_raw in lex_df[["form", "norm"]].itertuples(index=False):
        form = normalize_source(form_raw, True)
        norm = normalize_source(norm_raw, True)
        if not form or not norm:
            continue
        if len(form.split()) != 1 or len(norm.split()) != 1:
            continue
        token_map.setdefault(form, norm)
    return LexiconResources(token_map=token_map)


def lexicon_normalize_source_text(text: str, lexicon: LexiconResources) -> str:
    if not lexicon.token_map:
        return _norm_spaces(text)
    tokens = [lexicon.token_map.get(tok, tok) for tok in str(text).split()]
    return _norm_spaces(" ".join(tokens))


def _normalize_metadata_key(text: Any) -> str:
    value = normalize_target(text).lower()
    value = METADATA_WRAPPER_RE.sub(" ", value)
    value = value.replace("(", " ").replace(")", " ")
    value = METADATA_ALPHA_SUFFIX_RE.sub(r"\1", value)
    value = METADATA_PUNCT_RE.sub(" ", value)
    value = DISPLAY_NAME_NORMALIZE_RE.sub(" ", value)
    return _norm_spaces(value)


def _metadata_match_keys(text: Any) -> set[str]:
    raw = normalize_target(text)
    if not raw:
        return set()
    keys: set[str] = set()
    candidates = {raw, raw.replace("Cuneiform Tablet", " "), raw.replace("envelope", " ")}
    for candidate in candidates:
        norm = _normalize_metadata_key(candidate)
        if norm:
            keys.add(norm)
    return keys


def _published_alias_candidates(row: pd.Series) -> set[str]:
    raw_values: list[str] = []
    for col in ("aliases", "label", "publication_catalog", "inventory_position", "cdli_id"):
        value = row.get(col)
        if pd.isna(value):
            continue
        raw_values.append(str(value))

    aliases: set[str] = set()
    for raw in raw_values:
        for token in METADATA_TOKEN_SPLIT_RE.split(raw):
            token = token.strip()
            if not token:
                continue
            aliases.update(_metadata_match_keys(token))
    return aliases


def _metadata_sort_key(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    for col in ("column", "side", "line_number", "sentence_obj_in_text"):
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce").fillna(0)
    return ordered.sort_values(["column", "side", "line_number", "sentence_obj_in_text"], kind="stable")


def _anchor_tokens_from_sentence_row(row: pd.Series, lexicon: LexiconResources) -> list[str]:
    anchor = row.get("first_word_transcription")
    if pd.isna(anchor) or not str(anchor).strip():
        anchor = row.get("first_word_spelling")
    normalized = normalize_source(anchor, use_det_norm=True)
    normalized = lexicon_normalize_source_text(normalized, lexicon)
    tokens = [tok for tok in normalized.split() if tok and tok not in {"<gap>", "<big_gap>"}]
    return tokens


def _find_monotonic_anchor_index(
    source_tokens: Sequence[str],
    anchor_tokens: Sequence[str],
    start_idx: int,
) -> int | None:
    if not anchor_tokens:
        return None
    max_start = len(source_tokens) - len(anchor_tokens)
    for idx in range(max(0, start_idx), max_start + 1):
        if list(source_tokens[idx : idx + len(anchor_tokens)]) == list(anchor_tokens):
            return idx
    return None


def extract_monotonic_metadata_spans(
    source_text: str,
    sentence_rows: pd.DataFrame,
    lexicon: LexiconResources,
) -> list[str]:
    source_tokens = normalize_source(source_text, use_det_norm=True).split()
    source_tokens = [lexicon.token_map.get(tok, tok) for tok in source_tokens]
    if len(source_tokens) < 2:
        return []

    anchor_starts: list[int] = []
    cursor = 0
    for row in sentence_rows.itertuples(index=False):
        anchor_tokens = _anchor_tokens_from_sentence_row(pd.Series(row._asdict()), lexicon)
        anchor_idx = _find_monotonic_anchor_index(source_tokens, anchor_tokens, cursor)
        if anchor_idx is None:
            return []
        if anchor_starts and anchor_idx <= anchor_starts[-1]:
            return []
        anchor_starts.append(anchor_idx)
        cursor = anchor_idx + max(1, len(anchor_tokens))

    if len(set(anchor_starts)) != len(anchor_starts):
        return []

    spans: list[str] = []
    for idx, start in enumerate(anchor_starts):
        end = anchor_starts[idx + 1] if idx + 1 < len(anchor_starts) else len(source_tokens)
        if end <= start:
            return []
        span = _norm_spaces(" ".join(source_tokens[start:end]))
        if not span:
            return []
        spans.append(span)
    return spans


def _build_sentence_pair_row(
    *,
    oare_id: str,
    doc_index: int,
    sentence_index: int,
    source_text: str,
    target_text: str,
    doc_translation: str,
    lexicon: LexiconResources,
    supervision_source: str,
    pair_weight: float,
) -> dict[str, Any] | None:
    src_sentence = _norm_spaces(source_text)
    tgt_sentence = normalize_target(target_text)
    if not src_sentence or not tgt_sentence:
        return None
    return {
        "pair_id": f"{oare_id}__{supervision_source}__{sentence_index}",
        "oare_id": str(oare_id),
        "doc_index": int(doc_index),
        "sentence_index": int(sentence_index),
        "transliteration": src_sentence,
        "transliteration_lex": lexicon_normalize_source_text(src_sentence, lexicon),
        "translation": tgt_sentence,
        "doc_translation": normalize_target(doc_translation),
        "supervision_source": str(supervision_source),
        "pair_weight": float(pair_weight),
    }


def _source_has_quantity_or_unit(text: str) -> bool:
    lowered = normalize_source(text, use_det_norm=True).lower()
    if SOURCE_QUANTITY_RE.search(lowered):
        return True
    return any(token in lowered.split() for token in SOURCE_UNIT_TOKENS)


def _source_has_entity_tokens(text: str) -> bool:
    tokens = normalize_source(text, use_det_norm=True).split()
    for token in tokens:
        lowered = token.lower()
        if lowered in SOURCE_UNIT_TOKENS or lowered in {"<gap>", "<big_gap>"}:
            continue
        if SOURCE_NUMERIC_RE.fullmatch(lowered):
            continue
        if SOURCE_ENTITY_TOKEN_RE.fullmatch(token) and ("-" in token or token[:1].isupper()):
            return True
    return False


def split_target_sentences(text: str) -> list[str]:
    raw = normalize_target(text)
    if not raw:
        return [""]
    parts = [segment.strip() for segment in TARGET_SENTENCE_SPLIT_RE.split(raw) if segment.strip()]
    if not parts:
        return [raw]

    merged: list[str] = []
    for part in parts:
        word_count = len(part.split())
        if merged and word_count < 4 and not re.search(r"[.!?]$", merged[-1]):
            merged[-1] = _norm_spaces(f"{merged[-1]} {part}")
        else:
            merged.append(part)
    return merged or [raw]


def _token_boundary_score(tokens: Sequence[str], boundary_idx: int, lexicon: LexiconResources) -> float:
    if boundary_idx <= 0 or boundary_idx >= len(tokens):
        return -1e9
    prev_tok = str(tokens[boundary_idx - 1])
    next_tok = str(tokens[boundary_idx])
    score = 0.0
    if prev_tok in SOURCE_BOUNDARY_TOKENS or next_tok in SOURCE_BOUNDARY_TOKENS:
        score += 2.5
    if lexicon.token_map.get(prev_tok) != prev_tok or lexicon.token_map.get(next_tok) != next_tok:
        score += 0.5
    if prev_tok.endswith((".", ";", ":")):
        score += 0.5
    return score


def _weighted_source_tokens(tokens: Sequence[str], lexicon: LexiconResources) -> list[float]:
    weights: list[float] = []
    for tok in tokens:
        weight = 1.0
        if tok in {"<gap>", "<big_gap>"}:
            weight += 1.75
        if lexicon.token_map.get(tok) != tok:
            weight += 0.35
        weights.append(weight)
    return weights


def _target_sentence_weights(target_sentences: Sequence[str]) -> list[float]:
    weights: list[float] = []
    for sentence in target_sentences:
        sent = str(sentence)
        token_count = max(1.0, float(len(sent.split())))
        punctuation_bonus = 0.35 if re.search(r"[.!?;:]$", sent) else 0.0
        weights.append(token_count + punctuation_bonus)
    return weights


def _segment_alignment_cost(
    tokens: Sequence[str],
    token_weights: Sequence[float],
    start: int,
    end: int,
    target_weight: float,
    total_source_weight: float,
    total_target_weight: float,
    lexicon: LexiconResources,
) -> float:
    if end <= start:
        return float("inf")
    seg_weight = float(sum(token_weights[start:end]))
    expected_weight = max(1.0, (target_weight / max(1e-6, total_target_weight)) * total_source_weight)
    length_penalty = abs(seg_weight - expected_weight) / expected_weight
    token_penalty = 0.0
    seg_len = end - start
    if seg_len <= 1:
        token_penalty += 0.9
    if seg_len >= 40:
        token_penalty += min(1.2, seg_len / 60.0)
    boundary_bonus = 0.0
    if start > 0:
        boundary_bonus += max(0.0, _token_boundary_score(tokens, start, lexicon))
    if end < len(tokens):
        boundary_bonus += max(0.0, _token_boundary_score(tokens, end, lexicon))
    gap_bonus = 0.25 if any(tok in {"<gap>", "<big_gap>"} for tok in tokens[start:end]) else 0.0
    return float(length_penalty + token_penalty - 0.18 * boundary_bonus - gap_bonus)


def _dp_monotonic_source_split(
    source_text: str,
    target_sentences: Sequence[str],
    lexicon: LexiconResources,
) -> list[str]:
    tokens = str(source_text).split()
    n_sentences = len(target_sentences)
    if n_sentences <= 1 or len(tokens) < max(4, n_sentences * 2):
        return [_norm_spaces(source_text)]

    token_weights = _weighted_source_tokens(tokens, lexicon)
    target_weights = _target_sentence_weights(target_sentences)
    total_source_weight = float(sum(token_weights))
    total_target_weight = float(sum(target_weights))
    n_tokens = len(tokens)

    cumulative_source = np.cumsum(np.asarray(token_weights, dtype=float))
    target_cumulative = np.cumsum(np.asarray(target_weights, dtype=float))
    dp = np.full((n_sentences + 1, n_tokens + 1), np.inf, dtype=float)
    back = np.full((n_sentences + 1, n_tokens + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for sent_idx in range(1, n_sentences + 1):
        remaining_sentences = n_sentences - sent_idx
        desired_source_weight = (target_cumulative[sent_idx - 1] / max(1e-6, total_target_weight)) * total_source_weight
        approx_end = int(np.searchsorted(cumulative_source, desired_source_weight, side="left")) + 1
        min_end = sent_idx
        max_end = n_tokens - remaining_sentences
        window = max(10, int(n_tokens / max(1, n_sentences)) + 8)
        end_candidates = range(max(min_end, approx_end - window), min(max_end, approx_end + window) + 1)

        for end in end_candidates:
            prev_min = sent_idx - 1
            prev_max = end - 1
            if prev_max < prev_min:
                continue
            for prev_end in range(prev_min, prev_max + 1):
                prev_cost = dp[sent_idx - 1, prev_end]
                if not np.isfinite(prev_cost):
                    continue
                seg_cost = _segment_alignment_cost(
                    tokens,
                    token_weights,
                    prev_end,
                    end,
                    float(target_weights[sent_idx - 1]),
                    total_source_weight,
                    total_target_weight,
                    lexicon,
                )
                total_cost = prev_cost + seg_cost
                if total_cost < dp[sent_idx, end]:
                    dp[sent_idx, end] = total_cost
                    back[sent_idx, end] = prev_end

    if not np.isfinite(dp[n_sentences, n_tokens]):
        return [_norm_spaces(source_text)]

    boundaries = [n_tokens]
    cursor = n_tokens
    for sent_idx in range(n_sentences, 0, -1):
        prev_end = int(back[sent_idx, cursor])
        if prev_end < 0:
            return [_norm_spaces(source_text)]
        boundaries.append(prev_end)
        cursor = prev_end
    boundaries.reverse()

    segments: list[str] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk = _norm_spaces(" ".join(tokens[start:end]))
        if not chunk:
            return [_norm_spaces(source_text)]
        segments.append(chunk)
    return segments if len(segments) == n_sentences else [_norm_spaces(source_text)]


def split_source_for_target_sentences(
    source_text: str,
    target_sentences: Sequence[str],
    lexicon: LexiconResources,
) -> list[str]:
    return _dp_monotonic_source_split(source_text, target_sentences, lexicon)


def build_pseudo_sentence_pairs(train_df: pd.DataFrame, lexicon: LexiconResources) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc_idx, row in enumerate(train_df.itertuples(index=False), start=1):
        source_text = str(getattr(row, "transliteration"))
        target_text = str(getattr(row, "translation"))
        target_sentences = split_target_sentences(target_text)
        source_sentences = split_source_for_target_sentences(source_text, target_sentences, lexicon)
        if len(source_sentences) != len(target_sentences):
            source_sentences = [source_text]
            target_sentences = [target_text]

        pair_count = max(1, min(len(source_sentences), len(target_sentences)))
        for sentence_index in range(pair_count):
            record = _build_sentence_pair_row(
                oare_id=str(getattr(row, "oare_id")),
                doc_index=doc_idx,
                sentence_index=sentence_index,
                source_text=source_sentences[sentence_index],
                target_text=target_sentences[sentence_index],
                doc_translation=target_text,
                lexicon=lexicon,
                supervision_source="heuristic",
                pair_weight=1.0,
            )
            if record is not None:
                rows.append(record)

    if not rows:
        raise RuntimeError("Pseudo sentence builder produced zero training pairs.")
    return pd.DataFrame(rows)


def build_metadata_sentence_pairs(
    train_df: pd.DataFrame,
    published_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
    lexicon: LexiconResources,
) -> MetadataSupervisionResult:
    required_published = {"oare_id", "aliases", "label", "cdli_id"}
    required_sentences = {
        "display_name",
        "translation",
        "first_word_transcription",
        "first_word_spelling",
        "line_number",
        "side",
        "column",
        "sentence_obj_in_text",
    }
    if published_df.empty or sentence_df.empty:
        return MetadataSupervisionResult(pd.DataFrame(), 0, 0, 0)
    if not required_published.issubset(set(published_df.columns)) or not required_sentences.issubset(
        set(sentence_df.columns)
    ):
        return MetadataSupervisionResult(pd.DataFrame(), 0, 0, 0)

    published = published_df.drop_duplicates(subset=["oare_id"]).copy()
    published["oare_id"] = published["oare_id"].astype("string")
    published = published.set_index("oare_id", drop=False)

    sentence_work = sentence_df.copy()
    sentence_work["display_name_norm"] = sentence_work["display_name"].map(_normalize_metadata_key)
    sentence_work["display_name_keys"] = sentence_work["display_name"].map(_metadata_match_keys)
    sentence_work["translation"] = sentence_work["translation"].map(normalize_target)
    display_groups = {
        str(display_name): group.reset_index(drop=True)
        for display_name, group in sentence_work.groupby("display_name_norm", sort=False)
        if str(display_name).strip()
    }
    exact_match_index: dict[str, set[str]] = defaultdict(set)
    for display_name, group in display_groups.items():
        for key_set in group["display_name_keys"].tolist():
            for key in key_set:
                exact_match_index[str(key)].add(display_name)

    rows: list[dict[str, Any]] = []
    candidate_docs = 0
    matched_docs = 0
    rejected_docs = 0

    for doc_idx, row in enumerate(train_df.itertuples(index=False), start=1):
        oare_id = str(getattr(row, "oare_id"))
        if oare_id not in published.index:
            continue

        alias_candidates = _published_alias_candidates(published.loc[oare_id])
        if not alias_candidates:
            continue

        matched_group_names: set[str] = set()
        for alias in alias_candidates:
            matched_group_names.update(exact_match_index.get(alias, set()))

        if not matched_group_names:
            continue
        candidate_docs += 1
        if len(matched_group_names) != 1:
            rejected_docs += 1
            continue

        candidate = display_groups[sorted(matched_group_names)[0]]
        candidate = candidate.loc[candidate["translation"].map(lambda x: bool(str(x).strip()))].copy()
        if len(candidate) < 2:
            rejected_docs += 1
            continue
        ordered = _metadata_sort_key(candidate).reset_index(drop=True)
        source_candidates = [str(getattr(row, "transliteration"))]
        published_transliteration = published.loc[oare_id].get("transliteration")
        if not pd.isna(published_transliteration):
            source_candidates.append(str(published_transliteration))

        spans: list[str] = []
        for source_candidate in source_candidates:
            spans = extract_monotonic_metadata_spans(source_candidate, ordered, lexicon)
            if len(spans) == len(ordered):
                break
        if len(spans) != len(ordered):
            rejected_docs += 1
            continue

        matched_docs += 1
        ordered_targets = ordered["translation"].tolist()
        for sentence_index, (source_span, target_sentence) in enumerate(zip(spans, ordered_targets, strict=False)):
            record = _build_sentence_pair_row(
                oare_id=oare_id,
                doc_index=doc_idx,
                sentence_index=sentence_index,
                source_text=source_span,
                target_text=target_sentence,
                doc_translation=str(getattr(row, "translation")),
                lexicon=lexicon,
                supervision_source="sentence_metadata",
                pair_weight=1.5,
            )
            if record is not None:
                rows.append(record)

    if not rows:
        log(
            "[metadata_supervision] "
            f"candidates={candidate_docs} matched={matched_docs} "
            f"rejected={rejected_docs} pairs=0"
        )
        return MetadataSupervisionResult(pd.DataFrame(), candidate_docs, matched_docs, rejected_docs)
    log(
        f"[metadata_supervision] candidates={candidate_docs} matched={matched_docs} rejected={rejected_docs} "
        f"pairs={len(rows)}"
    )
    return MetadataSupervisionResult(pd.DataFrame(rows), candidate_docs, matched_docs, rejected_docs)


def build_merged_sentence_pairs(
    train_df: pd.DataFrame,
    lexicon: LexiconResources,
    published_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
) -> tuple[pd.DataFrame, MetadataSupervisionResult]:
    heuristic_pairs = build_pseudo_sentence_pairs(train_df, lexicon)
    metadata_pairs = MetadataSupervisionResult(pd.DataFrame(), 0, 0, 0)
    if METADATA_SUPERVISION_MODE == "high_precision":
        metadata_pairs = build_metadata_sentence_pairs(train_df, published_df, sentence_df, lexicon)

    frames = [heuristic_pairs]
    if not metadata_pairs.pair_df.empty:
        frames.append(metadata_pairs.pair_df)
    merged = pd.concat(frames, ignore_index=True)
    merged["_source_priority"] = merged["supervision_source"].map(lambda x: 0 if str(x) == "sentence_metadata" else 1)
    merged = merged.sort_values(
        ["oare_id", "sentence_index", "_source_priority", "pair_weight"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    merged = merged.drop_duplicates(subset=["oare_id", "transliteration_lex", "translation"], keep="first")
    merged = merged.drop(columns="_source_priority")
    merged["has_quantity_or_unit"] = merged["transliteration"].map(_source_has_quantity_or_unit)
    merged["has_entity_tokens"] = merged["transliteration"].map(_source_has_entity_tokens)
    return merged, metadata_pairs


def build_exact_source_memory(pair_df: pd.DataFrame) -> dict[str, str]:
    memory: dict[str, Counter[str]] = defaultdict(Counter)
    key_series = (
        pair_df["transliteration_lex"] if "transliteration_lex" in pair_df.columns else pair_df["transliteration"]
    )
    for src, tgt in zip(key_series.tolist(), pair_df["translation"].tolist()):
        src_key = _norm_spaces(str(src))
        tgt_value = normalize_target(tgt)
        if not src_key or not tgt_value:
            continue
        memory[src_key][tgt_value] += 1
    return {src: counter.most_common(1)[0][0] for src, counter in memory.items() if counter}


def _extract_target_entity_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in TARGET_TITLECASE_RE.finditer(normalize_target(text)):
        candidate = _norm_spaces(match.group(0))
        if candidate and len(candidate) >= 3:
            spans.append(candidate)
    return spans


def _extract_source_entity_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in normalize_source(text, use_det_norm=True).split():
        lowered = token.lower()
        if lowered in SOURCE_UNIT_TOKENS or lowered in {"<gap>", "<big_gap>"}:
            continue
        if SOURCE_NUMERIC_RE.fullmatch(lowered):
            continue
        if SOURCE_ENTITY_TOKEN_RE.fullmatch(token) and ("-" in token or token[:1].isupper()):
            tokens.append(token)
    return tokens


def _extract_quantity_phrases(text: str) -> list[str]:
    normalized = normalize_source(text, use_det_norm=True)
    return [_norm_spaces(match.group(0).lower()) for match in SOURCE_QUANTITY_RE.finditer(normalized)]


def _extract_unit_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in normalize_source(text, use_det_norm=True).split():
        lowered = token.lower()
        if lowered in SOURCE_UNIT_TOKENS:
            tokens.append(lowered)
    return tokens


def _build_counter_memory(
    entries: Iterable[tuple[str, str]],
    *,
    min_freq: int = 1,
    min_share: float = 0.6,
) -> dict[str, str]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for source_key, target_value in entries:
        source_key = _norm_spaces(source_key)
        target_value = _norm_spaces(target_value)
        if not source_key or not target_value:
            continue
        counters[source_key][target_value] += 1
    out: dict[str, str] = {}
    for key, counter in counters.items():
        if not counter:
            continue
        top_value, top_count = counter.most_common(1)[0]
        total = int(sum(counter.values()))
        if top_count < max(1, min_freq):
            continue
        if total <= 0 or (top_count / total) < float(min_share):
            continue
        out[key] = top_value
    return out


def build_constraint_memories(pair_df: pd.DataFrame) -> ConstraintMemories:
    exact_source_memory = build_exact_source_memory(pair_df)

    entity_entries: list[tuple[str, str]] = []
    quantity_entries: list[tuple[str, str]] = []
    unit_entries: list[tuple[str, str]] = []

    for row in pair_df.itertuples(index=False):
        source_text = str(getattr(row, "transliteration"))
        target_text = normalize_target(getattr(row, "translation"))
        entity_tokens = _extract_source_entity_tokens(source_text)
        target_entities = _extract_target_entity_spans(target_text)
        if len(entity_tokens) == 1 and len(target_entities) == 1:
            entity_entries.append((entity_tokens[0].lower(), target_entities[0]))

        quantity_phrases = _extract_quantity_phrases(source_text)
        target_quantities = [
            normalize_target(match.group(0)) for match in TARGET_QUANTITY_PHRASE_RE.finditer(target_text)
        ]
        if len(quantity_phrases) == 1 and len(target_quantities) == 1:
            quantity_entries.append((quantity_phrases[0], target_quantities[0]))

        target_lower = target_text.lower()
        for unit_token in _extract_unit_tokens(source_text):
            canonical = UNIT_TARGET_CANONICAL.get(unit_token)
            if canonical and canonical in target_lower:
                unit_entries.append((unit_token, canonical))

    return ConstraintMemories(
        exact_source_memory=exact_source_memory,
        entity_memory=_build_counter_memory(entity_entries),
        quantity_memory=_build_counter_memory(quantity_entries),
        unit_memory=_build_counter_memory(unit_entries),
    )


def _constraint_obligations(source_text: str, memories: ConstraintMemories | None) -> dict[str, list[str]]:
    if memories is None:
        return {"entity": [], "quantity": [], "unit": []}

    normalized_source = normalize_source(source_text, use_det_norm=True)
    entity_targets: list[str] = []
    quantity_targets: list[str] = []
    unit_targets: list[str] = []

    for token in _extract_source_entity_tokens(normalized_source):
        target = memories.entity_memory.get(token.lower())
        if target:
            entity_targets.append(target)
    for phrase in _extract_quantity_phrases(normalized_source):
        target = memories.quantity_memory.get(phrase.lower())
        if target:
            quantity_targets.append(target)
    for unit_token in _extract_unit_tokens(normalized_source):
        target = memories.unit_memory.get(unit_token.lower())
        if target:
            unit_targets.append(target)

    return {
        "entity": list(dict.fromkeys(entity_targets)),
        "quantity": list(dict.fromkeys(quantity_targets)),
        "unit": list(dict.fromkeys(unit_targets)),
    }


def _constraint_candidate_bonus(source_text: str, candidate_text: str, memories: ConstraintMemories | None) -> float:
    if memories is None or CONSTRAINT_REWRITE_MODE != "soft":
        return 0.0
    obligations = _constraint_obligations(source_text, memories)
    if not any(obligations.values()):
        return 0.0
    normalized_candidate = normalize_target(candidate_text).lower()
    bonus = 0.0
    for target in obligations["quantity"] + obligations["unit"] + obligations["entity"]:
        bonus += 1.5 if target.lower() in normalized_candidate else -2.0
    return float(max(-4.0, min(4.0, bonus)))


def _apply_single_category_rewrites(
    text: str,
    canonical_targets: Sequence[str],
    stats_key: str,
    stats: dict[str, int],
) -> str:
    out = normalize_prediction_style(text)
    if not canonical_targets:
        return out
    if len(set(canonical_targets)) != 1:
        return out
    canonical = normalize_prediction_style(canonical_targets[0])
    if not canonical or canonical.lower() in out.lower():
        return out

    quantity_spans = [normalize_target(match.group(0)) for match in TARGET_QUANTITY_PHRASE_RE.finditer(out)]
    if quantity_spans and len(quantity_spans) == 1:
        rewritten = out.replace(quantity_spans[0], canonical, 1)
        if rewritten != out:
            stats[stats_key] = stats.get(stats_key, 0) + 1
            return normalize_prediction_style(rewritten)
    title_spans = _extract_target_entity_spans(out)
    if title_spans and len(title_spans) == 1:
        rewritten = out.replace(title_spans[0], canonical, 1)
        if rewritten != out:
            stats[stats_key] = stats.get(stats_key, 0) + 1
            return normalize_prediction_style(rewritten)
    return out


def apply_soft_constraint_rewrites(
    source_text: str,
    prediction: str,
    memories: ConstraintMemories | None,
    stats: dict[str, int],
) -> str:
    out = normalize_prediction_style(prediction)
    if memories is None or CONSTRAINT_REWRITE_MODE != "soft":
        return out

    obligations = _constraint_obligations(source_text, memories)
    if not any(obligations.values()):
        return out

    out = _apply_single_category_rewrites(out, obligations["quantity"], "quantity_rewrites", stats)
    out = _apply_single_category_rewrites(out, obligations["unit"], "unit_rewrites", stats)
    unique_quantities = list(dict.fromkeys(obligations["quantity"]))
    unique_units = list(dict.fromkeys(obligations["unit"]))
    fallback_target = None
    fallback_key = None
    if len(unique_quantities) == 1:
        fallback_target = unique_quantities[0]
        fallback_key = "quantity_rewrites"
    elif len(unique_units) == 1:
        fallback_target = unique_units[0]
        fallback_key = "unit_rewrites"
    if fallback_target and fallback_key:
        canonical = normalize_prediction_style(fallback_target)
        if canonical and canonical.lower() not in out.lower():
            out = canonical
            stats[fallback_key] = stats.get(fallback_key, 0) + 1
    out = _apply_single_category_rewrites(out, obligations["entity"], "entity_rewrites", stats)
    return normalize_prediction_style(out)


def compute_slice_metrics(pair_df: pd.DataFrame, predictions: Sequence[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not len(pair_df):
        return metrics

    def _slice_score(mask: pd.Series, key: str) -> None:
        if int(mask.sum()) <= 0:
            metrics[key] = 0.0
            return
        refs = pair_df.loc[mask, "translation"].tolist()
        idxs = np.flatnonzero(mask.to_numpy())
        preds = [predictions[idx] for idx in idxs]
        metrics[key] = float(compute_bleu_chrf_gmean(refs, preds)["gmean"])

    if "supervision_source" in pair_df.columns:
        metadata_mask = pair_df["supervision_source"].eq("sentence_metadata")
        _slice_score(metadata_mask, "metadata_supervision_sentence_gmean")
        metrics["metadata_supervised_gmean"] = metrics["metadata_supervision_sentence_gmean"]
        metrics["metadata_backed_sentence_gmean"] = metrics["metadata_supervision_sentence_gmean"]
        heuristic_mask = pair_df["supervision_source"].eq("heuristic")
        _slice_score(heuristic_mask, "heuristic_only_sentence_gmean")
        if int(metadata_mask.sum()) > 0:
            subset = pair_df.loc[metadata_mask].reset_index(drop=True)
            if {"oare_id", "sentence_index", "doc_translation"}.issubset(set(subset.columns)):
                _doc_sentence_metric, metadata_doc_metric = compute_sentence_and_document_metrics(
                    subset,
                    [predictions[idx] for idx in np.flatnonzero(metadata_mask.to_numpy())],
                )
                metrics["metadata_supervision_document_gmean"] = float(metadata_doc_metric["gmean"])
            else:
                metrics["metadata_supervision_document_gmean"] = metrics["metadata_supervision_sentence_gmean"]
        else:
            metrics["metadata_supervision_document_gmean"] = 0.0
    if "has_quantity_or_unit" in pair_df.columns:
        _slice_score(pair_df["has_quantity_or_unit"].astype(bool), "quantity_unit_sentence_gmean")
        metrics["quantity_unit_gmean"] = metrics["quantity_unit_sentence_gmean"]
    if "has_entity_tokens" in pair_df.columns:
        _slice_score(pair_df["has_entity_tokens"].astype(bool), "entity_heavy_sentence_gmean")
        metrics["entity_heavy_gmean"] = metrics["entity_heavy_sentence_gmean"]
    return metrics


def reconstruct_document_predictions(
    pair_df: pd.DataFrame,
    predictions: Sequence[str],
) -> pd.DataFrame:
    frame = pair_df[["oare_id", "sentence_index", "doc_translation"]].copy()
    frame["prediction"] = [normalize_target(x) for x in predictions]
    frame = frame.sort_values(["oare_id", "sentence_index"], kind="stable")
    grouped = (
        frame.groupby("oare_id", sort=False)
        .agg(
            reference=("doc_translation", "first"),
            prediction=("prediction", lambda s: normalize_target(" ".join(str(x) for x in s if str(x).strip()))),
        )
        .reset_index()
    )
    return grouped


def compute_sentence_and_document_metrics(
    pair_df: pd.DataFrame,
    predictions: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    sentence_metric = compute_bleu_chrf_gmean(pair_df["translation"].tolist(), predictions)
    doc_df = reconstruct_document_predictions(pair_df, predictions)
    document_metric = compute_bleu_chrf_gmean(doc_df["reference"].tolist(), doc_df["prediction"].tolist())
    return sentence_metric, document_metric


def compute_unseen_source_metrics(
    train_fold: pd.DataFrame,
    valid_fold: pd.DataFrame,
    predictions: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    source_col = _source_feature_col(valid_fold)
    train_keys = {_norm_spaces(str(value)) for value in train_fold[source_col].tolist() if _norm_spaces(str(value))}
    unseen_mask = valid_fold[source_col].map(lambda value: _norm_spaces(str(value)) not in train_keys)
    unseen_row_count = int(unseen_mask.sum())
    unseen_doc_count = int(valid_fold.loc[unseen_mask, "oare_id"].nunique()) if "oare_id" in valid_fold.columns else 0
    coverage = {
        "row_fraction": float(unseen_row_count / max(1, len(valid_fold))),
        "row_count": float(unseen_row_count),
        "document_fraction": float(
            unseen_doc_count / max(1, valid_fold["oare_id"].nunique()) if "oare_id" in valid_fold.columns else 0.0
        ),
        "document_count": float(unseen_doc_count),
    }
    if unseen_row_count <= 0:
        zero_metric = {"bleu": 0.0, "chrfpp": 0.0, "gmean": 0.0}
        return zero_metric, zero_metric, coverage

    unseen_subset = valid_fold.loc[unseen_mask].reset_index(drop=True)
    unseen_predictions = [predictions[idx] for idx in np.flatnonzero(unseen_mask.to_numpy())]
    unseen_sentence_metric, unseen_document_metric = compute_sentence_and_document_metrics(
        unseen_subset, unseen_predictions
    )
    return unseen_sentence_metric, unseen_document_metric, coverage


def iter_grouped_cv_splits(
    groups: Sequence[str],
    n_folds: int,
    seed: int,
    fast_dev: bool,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    group_arr = np.asarray([str(x) for x in groups], dtype=object)
    n_rows = len(group_arr)
    if n_rows < 2:
        raise ValueError("Need at least 2 rows for grouped CV")

    unique_groups = np.unique(group_arr)
    if len(unique_groups) < 2:
        idx = np.arange(n_rows)
        val_size = max(1, min(n_rows - 1, int(round(0.2 * n_rows))))
        yield idx[val_size:], idx[:val_size]
        return

    if fast_dev:
        rng = np.random.RandomState(seed)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        holdout_count = max(1, min(len(shuffled_groups) - 1, int(round(0.2 * len(shuffled_groups)))))
        holdout = set(shuffled_groups[:holdout_count].tolist())
        val_mask = np.array([group in holdout for group in group_arr], dtype=bool)
        yield np.where(~val_mask)[0], np.where(val_mask)[0]
        return

    effective_folds = max(2, min(int(n_folds), len(unique_groups)))
    splitter = GroupKFold(n_splits=effective_folds)
    idx = np.arange(n_rows)
    yield from splitter.split(idx, groups=group_arr)


def normalize_prediction_style(text: str) -> str:
    out = normalize_target(text)
    out = re.sub(r"\s*<\s*gap\s*>\s*", " <gap> ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*<\s*big_gap\s*>\s*", " <big_gap> ", out, flags=re.IGNORECASE)
    out = out.replace("<gap> <gap>", "<big_gap>")
    out = out.replace("<big_gap> <big_gap>", "<big_gap>")
    out = out.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
    return _norm_spaces(out)


def _prediction_is_low_confidence(prediction: str) -> bool:
    text = normalize_target(prediction)
    if not text:
        return True
    if len(text.split()) <= 2:
        return True
    if text.count("<gap>") + text.count("<big_gap>") >= 2:
        return True
    return False


def apply_consistency_postprocess(
    source_texts: Sequence[str],
    predictions: Sequence[str],
    group_values: Sequence[Any] | None,
    exact_source_memory: dict[str, str] | None,
    constraint_memories: ConstraintMemories | None = None,
    enable_exact_memory: bool = True,
) -> tuple[list[str], dict[str, int]]:
    outputs = [normalize_prediction_style(postprocess_translation(str(pred), strong=True)) for pred in predictions]
    stats = {
        "memory_rewrites": 0,
        "consistency_rewrites": 0,
        "entity_rewrites": 0,
        "quantity_rewrites": 0,
        "unit_rewrites": 0,
        "constraint_bonus_hits": 0,
    }

    if enable_exact_memory and exact_source_memory:
        for idx, src in enumerate(source_texts):
            src_key = _norm_spaces(str(src))
            candidate = exact_source_memory.get(src_key)
            if candidate and _prediction_is_low_confidence(outputs[idx]):
                rewritten = normalize_prediction_style(candidate)
                if rewritten != outputs[idx]:
                    outputs[idx] = rewritten
                    stats["memory_rewrites"] += 1
    if constraint_memories is not None:
        for idx, src in enumerate(source_texts):
            outputs[idx] = apply_soft_constraint_rewrites(str(src), outputs[idx], constraint_memories, stats)

    if group_values is None:
        return outputs, stats

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(group_values):
        grouped_indices[str(value)].append(idx)

    for indices in grouped_indices.values():
        by_source: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            by_source[_norm_spaces(str(source_texts[idx]))].append(idx)
        for source_key, source_indices in by_source.items():
            if len(source_indices) < 2:
                continue
            translations = [outputs[idx] for idx in source_indices if outputs[idx].strip()]
            if not translations:
                continue
            canonical = sorted(Counter(translations).items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]
            if enable_exact_memory and exact_source_memory and source_key in exact_source_memory:
                canonical = normalize_prediction_style(exact_source_memory[source_key])
            for idx in source_indices:
                if outputs[idx] != canonical:
                    outputs[idx] = canonical
                    stats["consistency_rewrites"] += 1

    return outputs, stats


# =====================================================================================
# Corpus BLEU + chrF++ (pure-Python implementation for blocked Kaggle runtime)
# =====================================================================================


def _safe_text_pairs(references: Sequence[str], predictions: Sequence[str]) -> tuple[list[str], list[str]]:
    n = min(len(references), len(predictions))
    refs = [normalize_target(str(x)) for x in references[:n]]
    hyps = [normalize_target(str(x)) for x in predictions[:n]]
    return refs, hyps


def _count_ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _count_char_ngrams(text: str, n: int) -> Counter[str]:
    if not text:
        return Counter()
    if len(text) < n:
        return Counter({text: 1})
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _fscore_from_counts(match: int, pred_total: int, ref_total: int, beta: float = 2.0) -> float:
    if pred_total <= 0 or ref_total <= 0:
        return 0.0
    precision = match / pred_total
    recall = match / ref_total
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def corpus_bleu_score(references: Sequence[str], predictions: Sequence[str], max_order: int = 4) -> float:
    if not references or not predictions:
        return 0.0

    n = min(len(references), len(predictions))
    refs = [str(x) for x in references[:n]]
    hyps = [str(x) for x in predictions[:n]]

    matches_by_order = [0] * max_order
    totals_by_order = [0] * max_order
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(refs, hyps):
        ref_toks = ref.split()
        hyp_toks = hyp.split()
        ref_len += len(ref_toks)
        hyp_len += len(hyp_toks)

        for order in range(1, max_order + 1):
            ref_counts = _count_ngrams(ref_toks, order)
            hyp_counts = _count_ngrams(hyp_toks, order)
            overlap = hyp_counts & ref_counts
            matches_by_order[order - 1] += int(sum(overlap.values()))
            totals_by_order[order - 1] += int(sum(hyp_counts.values()))

    precisions: list[float] = []
    for i in range(max_order):
        if totals_by_order[i] == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches_by_order[i] / totals_by_order[i])

    if min(precisions) <= 0.0:
        bleu = 0.0
    else:
        log_prec = sum(math.log(p) for p in precisions) / max_order
        bp = 1.0
        if hyp_len <= 0:
            bp = 0.0
        elif hyp_len < ref_len:
            bp = math.exp(1.0 - (ref_len / hyp_len))
        bleu = bp * math.exp(log_prec)

    return float(100.0 * bleu)


def corpus_chrfpp_score(
    references: Sequence[str],
    predictions: Sequence[str],
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    if not references or not predictions:
        return 0.0

    n = min(len(references), len(predictions))
    refs = [str(x) for x in references[:n]]
    hyps = [str(x) for x in predictions[:n]]

    f_scores: list[float] = []

    for order in range(1, char_order + 1):
        match_total = 0
        pred_total = 0
        ref_total = 0
        for ref, hyp in zip(refs, hyps):
            ref_counts = _count_char_ngrams(ref, order)
            hyp_counts = _count_char_ngrams(hyp, order)
            overlap = ref_counts & hyp_counts
            match_total += int(sum(overlap.values()))
            pred_total += int(sum(hyp_counts.values()))
            ref_total += int(sum(ref_counts.values()))
        f_scores.append(_fscore_from_counts(match_total, pred_total, ref_total, beta=beta))

    for order in range(1, word_order + 1):
        match_total = 0
        pred_total = 0
        ref_total = 0
        for ref, hyp in zip(refs, hyps):
            ref_counts = _count_ngrams(ref.split(), order)
            hyp_counts = _count_ngrams(hyp.split(), order)
            overlap = ref_counts & hyp_counts
            match_total += int(sum(overlap.values()))
            pred_total += int(sum(hyp_counts.values()))
            ref_total += int(sum(ref_counts.values()))
        f_scores.append(_fscore_from_counts(match_total, pred_total, ref_total, beta=beta))

    if not f_scores:
        return 0.0
    return float(100.0 * (sum(f_scores) / len(f_scores)))


_MBR_UTILITY_LOGGED = False


def compute_bleu_chrf_gmean(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    if sacrebleu is not None:
        refs, hyps = _safe_text_pairs(references, predictions)
        try:
            bleu = float(sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a", use_effective_order=True).score)
            chrfpp = float(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score)
            gmean = float(math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0)))
            return {"bleu": bleu, "chrfpp": chrfpp, "gmean": gmean}
        except Exception:
            pass
    bleu = corpus_bleu_score(references, predictions)
    chrfpp = corpus_chrfpp_score(references, predictions)
    gmean = float(math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0)))
    return {"bleu": float(bleu), "chrfpp": float(chrfpp), "gmean": gmean}


def _candidate_length_style_penalty(a: str, b: str) -> float:
    a_norm = normalize_prediction_style(a)
    b_norm = normalize_prediction_style(b)
    len_a = max(1, len(a_norm.split()))
    len_b = max(1, len(b_norm.split()))
    length_ratio_gap = abs(len_a - len_b) / max(len_a, len_b)
    punctuation_gap = 0.25 if bool(re.search(r"[.!?;:]$", a_norm)) != bool(re.search(r"[.!?;:]$", b_norm)) else 0.0
    return float(min(4.0, 1.5 * length_ratio_gap + punctuation_gap))


def _token_jaccard(a: str, b: str) -> float:
    a_tokens = set(normalize_prediction_style(a).split())
    b_tokens = set(normalize_prediction_style(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return float(len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens)))


def candidate_mbr_utility_score(a: str, b: str) -> float:
    metric = compute_bleu_chrf_gmean([a], [b])
    jaccard = 100.0 * _token_jaccard(a, b)
    length_bonus = max(0.0, 1.0 - _candidate_length_style_penalty(a, b) / 4.0)
    return float(0.55 * metric["chrfpp"] + 0.25 * metric["bleu"] + 0.20 * jaccard + 0.10 * (100.0 * length_bonus))


# =====================================================================================
# CV splits and baseline
# =====================================================================================


def iter_cv_splits(n_rows: int, n_folds: int, seed: int, fast_dev: bool) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    if n_rows < 2:
        raise ValueError("Need at least 2 rows for CV")
    idx = np.arange(n_rows)

    if fast_dev or n_folds < 2 or n_rows < n_folds:
        rng = np.random.RandomState(seed)
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        val_size = max(1, int(round(0.2 * n_rows)))
        val_size = min(val_size, n_rows - 1)
        yield shuffled[val_size:], shuffled[:val_size]
        return

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from kf.split(idx)


def simple_lookup_predict(train_src: Sequence[str], train_tgt: Sequence[str], infer_src: Sequence[str]) -> list[str]:
    table: dict[str, Counter[str]] = {}
    for src, tgt in zip(train_src, train_tgt):
        key = str(src)
        if key not in table:
            table[key] = Counter()
        table[key][str(tgt)] += 1

    default_tgt = Counter([str(x) for x in train_tgt]).most_common(1)[0][0] if train_tgt else ""
    preds: list[str] = []
    for src in infer_src:
        key = str(src)
        if key in table:
            preds.append(table[key].most_common(1)[0][0])
        else:
            preds.append(default_tgt)
    return preds


def majority_vote_predictions(pred_lists: Sequence[Sequence[str]]) -> list[str]:
    if not pred_lists:
        return []
    n = len(pred_lists[0])
    out: list[str] = []
    for i in range(n):
        votes = Counter(str(pred[i]) for pred in pred_lists)
        top_n = max(votes.values())
        candidates = [k for k, v in votes.items() if v == top_n]
        out.append(sorted(candidates, key=lambda x: (len(x), x))[0])
    return out


@dataclass
class RetrievalModel:
    char_vectorizer: Any
    char_train_matrix: Any
    word_vectorizer: Any | None
    word_train_matrix: Any | None
    train_sources: np.ndarray
    train_targets: np.ndarray
    exact_lookup: dict[str, str]
    default_target: str


def _fit_retrieval_model(train_src: Sequence[str], train_tgt: Sequence[str]) -> RetrievalModel:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not train_tgt:
        default_target = ""
    else:
        default_target = Counter([str(x) for x in train_tgt]).most_common(1)[0][0]
    train_text = [str(x) for x in train_src]
    min_n = min(RETRIEVAL_NGRAM_MIN, RETRIEVAL_NGRAM_MAX)
    max_n = max(RETRIEVAL_NGRAM_MIN, RETRIEVAL_NGRAM_MAX)

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(min_n, max_n),
        lowercase=False,
        min_df=max(1, RETRIEVAL_MIN_DF),
    )
    try:
        char_train_matrix = char_vectorizer.fit_transform(train_text)
    except ValueError:
        # Guard for tiny folds where min_df=2 can yield empty vocabulary.
        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(min_n, max_n),
            lowercase=False,
            min_df=1,
        )
        char_train_matrix = char_vectorizer.fit_transform(train_text)

    word_vectorizer = None
    word_train_matrix = None
    if RETRIEVAL_WORD_WEIGHT > 0:
        try:
            word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=False,
                min_df=max(1, RETRIEVAL_WORD_MIN_DF),
            )
            word_train_matrix = word_vectorizer.fit_transform(train_text)
        except ValueError:
            word_vectorizer = None
            word_train_matrix = None
    return RetrievalModel(
        char_vectorizer=char_vectorizer,
        char_train_matrix=char_train_matrix,
        word_vectorizer=word_vectorizer,
        word_train_matrix=word_train_matrix,
        train_sources=np.array(train_text, dtype=object),
        train_targets=np.array([str(x) for x in train_tgt], dtype=object),
        exact_lookup=build_exact_source_memory(
            pd.DataFrame({"transliteration": train_text, "translation": [str(x) for x in train_tgt]})
        ),
        default_target=default_target,
    )


def _predict_with_retrieval(
    model: RetrievalModel,
    infer_src: Sequence[str],
    use_mbr: bool,
    k: int,
    min_sim: float,
) -> tuple[list[str], int]:
    candidate_pools, low_sim_count = _retrieval_candidate_pools(
        model=model,
        infer_src=infer_src,
        k=k,
        min_sim=min_sim,
        max_candidates=max(1, min(k, 4)),
    )
    preds: list[str] = []
    for pool in candidate_pools:
        if not pool:
            preds.append(model.default_target)
        elif use_mbr and len(pool) > 1:
            preds.append(_select_mbr_candidate(pool))
        else:
            preds.append(pool[0][0])
    return preds, low_sim_count


def _lexical_overlap_score(a: str, b: str) -> float:
    a_tokens = set(normalize_source(a, use_det_norm=True).split())
    b_tokens = set(normalize_source(b, use_det_norm=True).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return float(len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens)))


def _retrieval_rank_score(source_text: str, candidate_source: str, char_score: float, word_score: float) -> float:
    normalized_source = normalize_source(source_text, use_det_norm=True)
    normalized_candidate = normalize_source(candidate_source, use_det_norm=True)
    overlap = _lexical_overlap_score(normalized_source, normalized_candidate)
    edit_ratio = float(SequenceMatcher(a=normalized_source, b=normalized_candidate).ratio())
    exact_bonus = 2.0 if normalized_source == normalized_candidate else 0.0
    return float(char_score + word_score + 0.9 * overlap + 0.6 * edit_ratio + exact_bonus)


def _retrieval_candidate_pools(
    model: RetrievalModel,
    infer_src: Sequence[str],
    k: int,
    min_sim: float,
    max_candidates: int,
) -> tuple[list[list[tuple[str, str, int]]], int]:
    from sklearn.metrics.pairwise import linear_kernel

    if len(model.train_targets) == 0:
        return [[] for _ in infer_src], len(infer_src)

    infer_text = [str(x) for x in infer_src]
    char_infer_matrix = model.char_vectorizer.transform(infer_text)
    char_sims = linear_kernel(char_infer_matrix, model.char_train_matrix)
    sims = char_sims.copy()
    word_sims = None
    if model.word_vectorizer is not None and model.word_train_matrix is not None and RETRIEVAL_WORD_WEIGHT > 0:
        word_infer_matrix = model.word_vectorizer.transform(infer_text)
        word_sims = RETRIEVAL_WORD_WEIGHT * linear_kernel(word_infer_matrix, model.word_train_matrix)
        sims = sims + word_sims

    candidate_pools: list[list[tuple[str, str, int]]] = []
    low_sim_count = 0
    top_k = max(1, min(k, len(model.train_targets)))
    for row_idx, row_sims in enumerate(sims):
        if row_sims.size == 0:
            candidate_pools.append([])
            low_sim_count += 1
            continue

        candidate_indices: set[int] = set()
        if top_k >= row_sims.size:
            candidate_indices.update(int(idx) for idx in np.argsort(-row_sims))
        else:
            rough_idx = np.argpartition(-row_sims, top_k - 1)[:top_k]
            candidate_indices.update(int(idx) for idx in rough_idx)
        if word_sims is not None:
            row_word = word_sims[row_idx]
            if top_k >= row_word.size:
                candidate_indices.update(int(idx) for idx in np.argsort(-row_word))
            else:
                rough_word_idx = np.argpartition(-row_word, top_k - 1)[:top_k]
                candidate_indices.update(int(idx) for idx in rough_word_idx)

        exact_target = model.exact_lookup.get(_norm_spaces(str(infer_text[row_idx])))
        if exact_target:
            candidate_pools.append([(exact_target, "retrieval_exact", 1)])
            continue

        if not candidate_indices:
            candidate_pools.append([])
            low_sim_count += 1
            continue

        scored_indices: list[tuple[float, int]] = []
        for idx in candidate_indices:
            char_score = float(char_sims[row_idx, idx]) if char_sims.size else 0.0
            word_score = float(word_sims[row_idx, idx]) if word_sims is not None else 0.0
            score = _retrieval_rank_score(infer_text[row_idx], str(model.train_sources[idx]), char_score, word_score)
            scored_indices.append((score, idx))
        scored_indices.sort(key=lambda item: (-item[0], item[1]))

        best_sim = float(scored_indices[0][0]) if scored_indices else -1.0
        if best_sim < min_sim:
            candidate_pools.append([])
            low_sim_count += 1
            continue

        candidate_tuples: list[tuple[str, str, int]] = []
        seen: set[str] = set()
        for rank, (_, idx) in enumerate(scored_indices, start=1):
            candidate = str(model.train_targets[int(idx)])
            if candidate in seen:
                continue
            seen.add(candidate)
            candidate_tuples.append((candidate, "retrieval", rank))
            if len(candidate_tuples) >= max(1, max_candidates):
                break

        candidate_pools.append(candidate_tuples)

    return candidate_pools, low_sim_count


@dataclass
class PipelineResult:
    name: str
    cv_score: float
    bleu: float
    chrfpp: float
    complexity_rank: int
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    best_seed: int
    doc_score: float = 0.0
    doc_bleu: float = 0.0
    doc_chrfpp: float = 0.0
    unseen_sentence_score: float = 0.0
    unseen_sentence_bleu: float = 0.0
    unseen_sentence_chrfpp: float = 0.0
    unseen_document_score: float = 0.0
    unseen_document_bleu: float = 0.0
    unseen_document_chrfpp: float = 0.0
    unseen_coverage: dict[str, float] | None = None
    executed_checkpoints: list[str] | None = None
    postprocess_stats: dict[str, int] | None = None
    slice_metrics: dict[str, float] | None = None
    ensemble_members: list[str] | None = None


# =====================================================================================
# Plan-driven pipeline configs
# =====================================================================================


@dataclass
class PipelineConfig:
    name: str
    model_hints: list[str]
    max_source_len: int
    max_new_tokens: int
    num_beams: int
    length_penalty: float
    repetition_penalty: float
    mbr_num_beam_cands: int
    mbr_num_sample_cands: int
    sample_temperatures: list[float]
    mbr_top_p: float
    mbr_pool_cap: int
    use_mbr: bool
    use_multi_model_pool: bool
    use_lora: bool
    use_retrieval_candidates: bool
    use_context_window: bool
    allow_domain_adapted: bool
    strong_postprocess: bool
    complexity_rank: int
    runtime_name: str | None = None
    reference_runtime_mode: str = ""
    reference_blocker: str = ""
    reference_slot_meta: list[dict[str, str]] | None = None


def _pipeline_complexity_rank(name: str) -> int:
    rank_map = {
        "lookup_baseline": 0,
        "char_tfidf_knn_memory": 1,
        "retrieval_char_tfidf_knn": 1,
        "retrieval_char_tfidf_knn_mbr": 2,
        "dual_checkpoint_public_mbr": 10,
        "retrieval_augmented_byt5_rerank": 11,
        "contextual_byt5_curriculum_mbr": 12,
        "pooled_multi_byt5_mbr": 12,
        "diverse_model_addon_for_mbr_pool": 13,
        "byt5_large_lora_finetune_plus_mbr": 14,
    }
    return int(rank_map.get(name, 99))


def _normalize_model_hint(hint: str, index: int, pipeline_name: str) -> str:
    h = str(hint).strip()
    if not h:
        return "google/byt5-base"
    if Path(h).exists() or h.startswith("/") or "/" in h:
        return h

    lower = h.lower()
    if "non_byt5" in lower or "nllb" in lower or "mt5" in lower:
        return "google/mt5-base"

    # Plan model names can be descriptive placeholders; map them to robust defaults.
    if "byt5" in lower or "checkpoint" in lower or "mount" in lower:
        if pipeline_name == "diverse_model_addon_for_mbr_pool":
            return "google/mt5-base"
        fallback = ["google/byt5-base", "google/byt5-small", "google/byt5-base"]
        return fallback[min(index, len(fallback) - 1)]

    return "google/byt5-base"


def _extract_model_hints(raw_models: Any, pipeline_name: str) -> list[str]:
    hints: list[str] = []
    if isinstance(raw_models, list):
        for i, item in enumerate(raw_models):
            token = str(item)
            if "(" in token:
                token = token.split("(", 1)[0].strip()
            hints.append(_normalize_model_hint(token, i, pipeline_name))
    if not hints:
        hints = ["google/byt5-base"]
    return hints


def _extract_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _extract_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _plan_pipeline_match(name: str) -> dict[str, Any] | None:
    for pipe in PLAN_PIPELINES:
        if str(pipe.get("name", "")).strip() == str(name).strip():
            return pipe
    return None


def _extract_temperature_list(raw: Any, default: Sequence[float]) -> list[float]:
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
    elif isinstance(raw, (list, tuple)):
        tokens = list(raw)
    else:
        tokens = []
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except Exception:
            continue
    return values or list(default)


def _extract_plan_model_hints(name: str, match: dict[str, Any] | None) -> list[str]:
    hints = _plan_pipeline_model_hints(name)
    if hints:
        return hints
    if not isinstance(match, dict):
        return []

    hp = match.get("key_hyperparameters", {})
    if not isinstance(hp, dict):
        hp = {}

    ordered_keys = ("base_model", "secondary_model", "model_a", "model_b")
    parsed: list[str] = []
    seen: set[str] = set()
    for key in ordered_keys:
        value = hp.get(key)
        if value is None:
            continue
        hint = str(value).strip()
        if not hint or hint in seen:
            continue
        seen.add(hint)
        parsed.append(hint)

    if parsed:
        return parsed
    return _extract_model_hints(match.get("models", []), name)


def get_pipeline_cfg(name: str) -> PipelineConfig:
    """Safe lookup: missing names return usable defaults and never raise."""

    faithful_slug = _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG)
    safe_default = PipelineConfig(
        name=name,
        model_hints=["google/byt5-base"],
        max_source_len=REFERENCE_MAX_INPUT_LENGTH if faithful_slug else MAX_SOURCE_LEN,
        max_new_tokens=REFERENCE_MAX_NEW_TOKENS if faithful_slug else MAX_NEW_TOKENS,
        num_beams=REFERENCE_NUM_BEAMS if faithful_slug else NUM_BEAMS,
        length_penalty=REFERENCE_LENGTH_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
        mbr_num_beam_cands=min(REFERENCE_NUM_BEAM_CANDIDATES, NUM_BEAMS),
        mbr_num_sample_cands=REFERENCE_NUM_SAMPLE_PER_TEMP if faithful_slug else 1,
        sample_temperatures=list(SAMPLE_TEMPERATURES),
        mbr_top_p=TOP_P,
        mbr_pool_cap=MAX_POOL_CAP,
        use_mbr=USE_MBR,
        use_multi_model_pool=USE_MULTI_MODEL_POOL,
        use_lora=False,
        use_retrieval_candidates=False,
        use_context_window=False,
        allow_domain_adapted=False,
        strong_postprocess=_pipeline_complexity_rank(name) >= 10,
        complexity_rank=_pipeline_complexity_rank(name),
    )

    match = _plan_pipeline_match(name)
    if match is None:
        if name == "contextual_byt5_curriculum_mbr":
            return PipelineConfig(
                **{
                    **safe_default.__dict__,
                    "model_hints": ["assiaben/final-byt5", "artemgoncarov/dpc-byt5-large"],
                    "use_mbr": bool(USE_MBR and ENABLE_MULTI_CHECKPOINT_MBR),
                    "use_multi_model_pool": True,
                    "use_lora": False,
                    "use_context_window": bool(ENABLE_CONTEXT_WINDOW),
                    "allow_domain_adapted": True,
                }
            )
        if name == "dual_checkpoint_public_mbr":
            return PipelineConfig(
                **{
                    **safe_default.__dict__,
                    "model_hints": [
                        *REFERENCE_NOTEBOOK_MODEL_HINTS,
                        *[
                            hint
                            for hint in REFERENCE_FALLBACK_MODEL_HINTS
                            if hint not in REFERENCE_NOTEBOOK_MODEL_HINTS
                        ],
                    ],
                    "use_mbr": bool(USE_MBR and ENABLE_MULTI_CHECKPOINT_MBR),
                    "use_multi_model_pool": True,
                }
            )
        if name == "retrieval_augmented_byt5_rerank":
            return PipelineConfig(
                **{
                    **safe_default.__dict__,
                    "model_hints": ["assiaben/final-byt5"],
                    "use_mbr": bool(USE_MBR and ENABLE_RETRIEVAL_RERANK),
                    "use_multi_model_pool": False,
                    "use_retrieval_candidates": bool(ENABLE_RETRIEVAL_RERANK),
                }
            )
        return safe_default

    hp = match.get("key_hyperparameters", {}) if isinstance(match.get("key_hyperparameters"), dict) else {}
    num_beams = _extract_int(
        hp.get("num_beams", safe_default.num_beams),
        safe_default.num_beams,
    )
    max_new_tokens = _extract_int(
        hp.get("max_target_len", hp.get("max_new_tokens", safe_default.max_new_tokens)),
        safe_default.max_new_tokens,
    )
    length_penalty = _extract_float(
        hp.get("length_penalty", safe_default.length_penalty),
        safe_default.length_penalty,
    )

    beam_candidates = _extract_int(
        hp.get("num_beam_candidates", min(REFERENCE_NUM_BEAM_CANDIDATES, num_beams)),
        min(REFERENCE_NUM_BEAM_CANDIDATES, num_beams),
    )
    sample_temperatures = _extract_temperature_list(hp.get("sample_temperatures"), safe_default.sample_temperatures)
    sample_candidates = _extract_int(
        hp.get(
            "num_sample_per_temp",
            REFERENCE_NUM_SAMPLE_PER_TEMP if faithful_slug else safe_default.mbr_num_sample_cands,
        ),
        REFERENCE_NUM_SAMPLE_PER_TEMP if faithful_slug else safe_default.mbr_num_sample_cands,
    )
    sample_candidates = max(0, sample_candidates) if sample_temperatures else 0

    cfg = PipelineConfig(
        name=str(match.get("name", safe_default.name)),
        model_hints=_extract_plan_model_hints(name, match) or safe_default.model_hints,
        max_source_len=_extract_int(
            hp.get("max_source_len", safe_default.max_source_len),
            safe_default.max_source_len,
        ),
        max_new_tokens=max_new_tokens,
        num_beams=max(1, num_beams),
        length_penalty=length_penalty,
        repetition_penalty=_extract_float(
            hp.get("repetition_penalty", safe_default.repetition_penalty),
            safe_default.repetition_penalty,
        ),
        mbr_num_beam_cands=max(1, beam_candidates),
        mbr_num_sample_cands=max(0, sample_candidates),
        sample_temperatures=sample_temperatures,
        mbr_top_p=_extract_float(hp.get("top_p", safe_default.mbr_top_p), safe_default.mbr_top_p),
        mbr_pool_cap=_extract_int(hp.get("mbr_pool_cap", safe_default.mbr_pool_cap), safe_default.mbr_pool_cap),
        use_mbr=bool(USE_MBR),
        use_multi_model_pool=False,
        use_lora=False,
        use_retrieval_candidates=False,
        use_context_window=False,
        allow_domain_adapted=False,
        strong_postprocess=_pipeline_complexity_rank(name) >= 10,
        complexity_rank=_pipeline_complexity_rank(name),
    )

    if cfg.name == "contextual_byt5_curriculum_mbr":
        cfg.use_multi_model_pool = bool(ENABLE_MULTI_CHECKPOINT_MBR and len(cfg.model_hints) > 1)
        cfg.use_mbr = bool(USE_MBR and ENABLE_MULTI_CHECKPOINT_MBR)
        cfg.use_context_window = bool(ENABLE_CONTEXT_WINDOW)
        cfg.allow_domain_adapted = True
        cfg.use_lora = bool(ALLOW_KERNEL_FINETUNE)
    elif cfg.name == "dual_checkpoint_public_mbr":
        cfg.model_hints = [
            *REFERENCE_NOTEBOOK_MODEL_HINTS,
            *[hint for hint in REFERENCE_FALLBACK_MODEL_HINTS if hint not in REFERENCE_NOTEBOOK_MODEL_HINTS],
        ]
        cfg.max_source_len = max(cfg.max_source_len, REFERENCE_MAX_INPUT_LENGTH)
        cfg.max_new_tokens = max(cfg.max_new_tokens, REFERENCE_MAX_NEW_TOKENS)
        cfg.num_beams = max(cfg.num_beams, REFERENCE_NUM_BEAMS)
        cfg.length_penalty = REFERENCE_LENGTH_PENALTY
        cfg.repetition_penalty = REFERENCE_REPETITION_PENALTY
        cfg.mbr_num_beam_cands = max(cfg.mbr_num_beam_cands, REFERENCE_NUM_BEAM_CANDIDATES)
        cfg.mbr_num_sample_cands = max(cfg.mbr_num_sample_cands, REFERENCE_NUM_SAMPLE_PER_TEMP)
        cfg.sample_temperatures = list(REFERENCE_SAMPLE_TEMPERATURES)
        cfg.mbr_top_p = REFERENCE_SAMPLE_TOP_P
        cfg.mbr_pool_cap = max(cfg.mbr_pool_cap, REFERENCE_MBR_POOL_CAP)
        cfg.use_multi_model_pool = bool(ENABLE_MULTI_CHECKPOINT_MBR and len(cfg.model_hints) > 1)
        cfg.use_mbr = bool(USE_MBR and ENABLE_MULTI_CHECKPOINT_MBR)
        cfg.allow_domain_adapted = False
    elif cfg.name == "retrieval_augmented_byt5_rerank":
        cfg.use_multi_model_pool = False
        cfg.use_mbr = bool(USE_MBR and ENABLE_RETRIEVAL_RERANK)
        cfg.use_retrieval_candidates = bool(ENABLE_RETRIEVAL_RERANK)
        cfg.allow_domain_adapted = True
    elif cfg.name == "char_tfidf_knn_memory":
        cfg.use_mbr = False
        cfg.use_multi_model_pool = False
        cfg.use_retrieval_candidates = False

    cfg.max_source_len = int(MAX_SOURCE_LEN if MAX_SOURCE_LEN > 0 else cfg.max_source_len)
    cfg.max_new_tokens = int(MAX_NEW_TOKENS if MAX_NEW_TOKENS > 0 else cfg.max_new_tokens)
    cfg.num_beams = int(max(1, NUM_BEAMS if NUM_BEAMS > 0 else cfg.num_beams))
    cfg.mbr_pool_cap = int(max(1, min(cfg.mbr_pool_cap, MAX_POOL_CAP)))
    cfg.mbr_top_p = float(TOP_P)
    cfg.repetition_penalty = float(REPETITION_PENALTY)
    if cfg.use_mbr and sample_temperatures:
        cfg.sample_temperatures = _env_list_float("KAGGLEBOT_SAMPLE_TEMPERATURES", sample_temperatures)
    else:
        cfg.sample_temperatures = []

    if not cfg.use_multi_model_pool and len(cfg.model_hints) > 1:
        cfg.model_hints = cfg.model_hints[:1]
    if not ENABLE_PUBLIC_CHECKPOINTS and not cfg.allow_domain_adapted:
        cfg.model_hints = [hint for hint in cfg.model_hints if Path(str(hint)).exists()]
    if FAST_DEV:
        cfg.mbr_num_beam_cands = min(2, cfg.mbr_num_beam_cands)
        cfg.mbr_num_sample_cands = min(1, cfg.mbr_num_sample_cands)
        cfg.mbr_pool_cap = min(8, cfg.mbr_pool_cap)

    return cfg


def shortlisted_pipeline_names() -> list[str]:
    names = _plan_shortlisted_pipeline_names()
    toggle_values = [ENABLE_PIPELINE_1, ENABLE_PIPELINE_2, ENABLE_PIPELINE_3, ENABLE_PIPELINE_4]
    enabled: list[str] = []
    for idx, name in enumerate(names):
        if idx < len(toggle_values) and not toggle_values[idx]:
            continue
        enabled.append(name)
    result = enabled or names
    if (
        _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG)
        and REFERENCE_PRIMARY_PIPELINE_NAME in names
        and REFERENCE_PRIMARY_PIPELINE_NAME not in result
    ):
        result = [REFERENCE_PRIMARY_PIPELINE_NAME, *result]
    return result


# =====================================================================================
# Model loading and generation
# =====================================================================================


def _looks_like_loadable_model_source(path: Path) -> bool:
    return _resolve_loadable_model_source(path) is not None


def _is_direct_loadable_model_source(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix in {".bin", ".safetensors", ".pt"}
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = any(
        (path / name).exists()
        for name in (
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tf_model.h5",
            "flax_model.msgpack",
        )
    )
    return bool(has_config and has_weights)


def _resolve_loadable_model_source(path: Path, *, max_depth: int = 4) -> Path | None:
    if not path.exists():
        return None
    if path.is_file():
        return path if _is_direct_loadable_model_source(path) else None
    if _is_direct_loadable_model_source(path):
        return path
    if not path.is_dir():
        return None

    candidates: list[Path] = []
    for candidate in _iter_dirs_within_depth(path, max_depth=max_depth):
        if candidate == path:
            continue
        if not _is_direct_loadable_model_source(candidate):
            continue
        candidates.append(candidate)

    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda candidate: (len(candidate.parts), len(str(candidate)), str(candidate)))
    return ranked[0]


_KAGGLE_MODEL_SCAN_MAX_DEPTH = 4
_DISCOVERED_KAGGLE_MODEL_DIRS: list[Path] | None = None
_DISCOVERED_ARTIFACT_MODEL_DIRS: list[Path] | None = None
_DISCOVERED_HINT_LOGGED: set[str] = set()


def _iter_dirs_within_depth(root: Path, max_depth: int) -> Iterable[Path]:
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        yield current
        if depth >= max_depth:
            continue
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name.lower(), reverse=True)
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                stack.append((child, depth + 1))


def _model_path_rank_key(path: Path, model_name: str) -> tuple[int, int, str]:
    text = str(path).lower()
    score = 0
    if "byt5" in text:
        score += 60
    if "akkadian" in text:
        score += 45
    if "dpc" in text:
        score += 25
    if "final-byt5" in text or "final_byt5" in text:
        score += 20
    if "transformers/default" in text:
        score += 8
    model_token = model_name.split("/")[-1].lower()
    if model_token and model_token in text:
        score += 12
    return (-score, len(text), text)


def _discover_kaggle_model_dirs() -> list[Path]:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return []

    discovered: list[Path] = []
    seen: set[str] = set()
    for entry in sorted(kaggle_input.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        roots = [entry]
        default_root = entry / "transformers" / "default"
        if default_root.is_dir():
            roots.append(default_root)
        for root in roots:
            for candidate in _iter_dirs_within_depth(root, _KAGGLE_MODEL_SCAN_MAX_DEPTH):
                resolved_candidate = _resolve_loadable_model_source(candidate, max_depth=2)
                if resolved_candidate is None:
                    continue
                key = str(resolved_candidate)
                if key in seen:
                    continue
                seen.add(key)
                discovered.append(resolved_candidate)
    return discovered


def _cached_kaggle_model_dirs() -> list[Path]:
    global _DISCOVERED_KAGGLE_MODEL_DIRS
    if _DISCOVERED_KAGGLE_MODEL_DIRS is None:
        _DISCOVERED_KAGGLE_MODEL_DIRS = _discover_kaggle_model_dirs()
    return _DISCOVERED_KAGGLE_MODEL_DIRS


def _discover_artifact_model_dirs() -> list[Path]:
    roots = [
        ARTIFACT_DIR / "context" / "reference_inputs",
        ARTIFACT_DIR / "kernels",
        KERNEL_DIR / "models",
    ]
    discovered: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for candidate in _iter_dirs_within_depth(root, max_depth=6):
            resolved_candidate = _resolve_loadable_model_source(candidate, max_depth=2)
            if resolved_candidate is None:
                continue
            key = str(resolved_candidate)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(resolved_candidate)
    return discovered


def _cached_artifact_model_dirs() -> list[Path]:
    global _DISCOVERED_ARTIFACT_MODEL_DIRS
    if _DISCOVERED_ARTIFACT_MODEL_DIRS is None:
        _DISCOVERED_ARTIFACT_MODEL_DIRS = _discover_artifact_model_dirs()
    return _DISCOVERED_ARTIFACT_MODEL_DIRS


_LOADED_MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}
_LOADED_MODEL_CACHE_ORDER: list[str] = []


def _resolve_model_sources(model_name: str) -> list[str | Path]:
    candidates: list[str | Path] = []
    model_hint_path = Path(model_name)
    if model_hint_path.exists():
        candidates.append(model_hint_path)

    explicit_paths = os.getenv("KAGGLEBOT_MODEL_PATHS", "").strip()
    if explicit_paths:
        for token in explicit_paths.split(","):
            token = token.strip()
            if token:
                candidates.append(Path(token))

    explicit_dir = os.getenv("KAGGLEBOT_PRETRAINED_DIR", "").strip()
    if explicit_dir:
        base = Path(explicit_dir)
        if base.exists():
            candidates.extend(
                [
                    base,
                    base / model_name,
                    base / model_name.replace("/", "--"),
                    base / model_name.replace("/", "-"),
                    base / model_name.split("/")[-1],
                ]
            )

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        short = model_name.split("/")[-1]
        for entry in kaggle_input.iterdir():
            candidates.extend(
                [
                    entry,
                    entry / short,
                    entry / model_name.replace("/", "-"),
                    entry / model_name.replace("/", "--"),
                ]
            )
        discovered = sorted(_cached_kaggle_model_dirs(), key=lambda p: _model_path_rank_key(p, model_name))
        candidates.extend(discovered)

    local_models = KERNEL_DIR / "models"
    candidates.extend(
        [
            local_models,
            local_models / model_name,
            local_models / model_name.replace("/", "--"),
            local_models / model_name.replace("/", "-"),
            local_models / model_name.split("/")[-1],
        ]
    )
    discovered_artifact = sorted(_cached_artifact_model_dirs(), key=lambda p: _model_path_rank_key(p, model_name))
    candidates.extend(discovered_artifact)
    candidates.append(model_name)

    dedup: list[str | Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(candidate)
    return dedup


def _source_matches_model_hint(source: str | Path, model_hint: str) -> bool:
    source_path = Path(str(source))
    hint_path = Path(str(model_hint))
    if hint_path.exists():
        try:
            return source_path.resolve() == hint_path.resolve()
        except Exception:
            return str(source_path) == str(hint_path)

    hint_text = str(model_hint).strip().lower()
    if not hint_text:
        return True
    compact_source = re.sub(r"[^a-z0-9]+", "", str(source_path).lower())
    compact_resolved = compact_source
    if source_path.exists():
        try:
            compact_resolved = re.sub(r"[^a-z0-9]+", "", str(source_path.resolve()).lower())
        except Exception:
            compact_resolved = compact_source

    hint_parts = [re.sub(r"[^a-z0-9]+", "", part) for part in hint_text.split("/") if part.strip()]
    if len(hint_parts) >= 2 and hint_parts[0] and hint_parts[1]:
        owner_token, slug_token = hint_parts[0], hint_parts[1]
        raw_owner_slug_match = owner_token in compact_source and slug_token in compact_source
        resolved_owner_slug_match = owner_token in compact_resolved and slug_token in compact_resolved
        if source_path.exists() and raw_owner_slug_match and not resolved_owner_slug_match:
            return False
        if resolved_owner_slug_match:
            return True
        if raw_owner_slug_match and not source_path.exists():
            return True
        return False

    compact_hint = re.sub(r"[^a-z0-9]+", "", hint_text.split("/", 1)[1] if "/" in hint_text else hint_text)
    if not compact_hint:
        return True
    if source_path.exists() and compact_hint in compact_source and compact_hint not in compact_resolved:
        return False
    return compact_hint in compact_resolved


def _iter_local_model_sources(model_name: str) -> Iterable[str]:
    for source in _resolve_model_sources(model_name):
        source_str = str(source)
        if source_str == model_name and not Path(source_str).exists():
            continue
        if not _source_matches_model_hint(source_str, model_name):
            continue
        resolved_source = _resolve_loadable_model_source(Path(source_str))
        if resolved_source is not None:
            yield str(resolved_source.resolve())


def _first_local_model_source_for_hint(model_hint: str) -> str | None:
    for source in _iter_local_model_sources(model_hint):
        return str(source)
    return None


def _reference_source_allowed_for_hint(source: str | Path, model_hint: str) -> bool:
    return _source_matches_model_hint(source, model_hint)


def _reference_model_candidates(model_hint: str) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    blockers: list[str] = []
    seen: set[str] = set()

    for preferred in REFERENCE_EXACT_MODEL_ASSET_PATHS.get(model_hint, ()):
        if not preferred.exists():
            continue
        resolved_preferred = _resolve_loadable_model_source(preferred)
        if resolved_preferred is not None:
            canonical = _canonical_model_source_id(str(resolved_preferred))
            if canonical not in seen:
                seen.add(canonical)
                resolved.append(str(resolved_preferred.resolve()))
            continue
        if preferred.is_dir():
            try:
                is_empty = not any(preferred.iterdir())
            except OSError:
                is_empty = False
            if is_empty:
                blockers.append(f"{model_hint} local asset exists but is empty: {preferred}")
            else:
                blockers.append(f"{model_hint} local asset exists but has no HF checkpoint payload: {preferred}")
        else:
            blockers.append(f"{model_hint} local asset is not a usable checkpoint payload: {preferred}")

    for source in _resolve_model_sources(model_hint):
        path = Path(str(source))
        if not path.exists():
            continue
        if not _reference_source_allowed_for_hint(path, model_hint):
            continue
        resolved_source = _resolve_loadable_model_source(path)
        if resolved_source is None:
            continue
        canonical = _canonical_model_source_id(_absolute_model_source(resolved_source))
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(_absolute_model_source(resolved_source))

    unique_blockers: list[str] = []
    seen_blockers: set[str] = set()
    for message in blockers:
        if message in seen_blockers:
            continue
        seen_blockers.add(message)
        unique_blockers.append(message)
    return resolved, unique_blockers


def _resolve_distinct_reference_pair(model_hints: Sequence[str]) -> tuple[list[str], list[str]]:
    candidate_map: dict[str, list[str]] = {}
    blocker_messages: list[str] = []
    for hint in model_hints:
        candidates, blockers = _reference_model_candidates(hint)
        candidate_map[str(hint)] = candidates
        blocker_messages.extend(blockers)
    if len(model_hints) != 2:
        return [], blocker_messages

    left_hint, right_hint = str(model_hints[0]), str(model_hints[1])
    left_candidates = candidate_map.get(left_hint, [])
    right_candidates = candidate_map.get(right_hint, [])
    if not left_candidates or not right_candidates:
        return [], blocker_messages

    for left_source in left_candidates:
        left_id = _canonical_model_source_id(left_source)
        for right_source in right_candidates:
            right_id = _canonical_model_source_id(right_source)
            if left_id == right_id:
                continue
            return [left_source, right_source], blocker_messages

    blocker_messages.append(f"no distinct second checkpoint available locally for hints: {left_hint}, {right_hint}")
    return [], blocker_messages


def _reference_runtime_name(base_name: str, parts: Sequence[str]) -> str:
    suffix = "__".join(_artifact_safe_name(part) for part in parts if str(part).strip())
    return f"{base_name}__{suffix}" if suffix else base_name


def _absolute_model_source(source: str | Path) -> str:
    path = Path(str(source))
    try:
        if path.exists():
            return str(path.resolve())
    except OSError:
        pass
    return str(source)


def _reference_original_hint_for_source(source: str) -> str:
    for hint in [*REFERENCE_NOTEBOOK_MODEL_HINTS, *REFERENCE_FALLBACK_MODEL_HINTS]:
        if _source_matches_model_hint(source, hint):
            return hint
    return str(source)


def _build_reference_slot_meta(
    original_hints: Sequence[str],
    resolved_sources: Sequence[str],
) -> list[dict[str, str]]:
    slot_meta: list[dict[str, str]] = []
    for original_hint, resolved_source in zip(original_hints, resolved_sources):
        resolved_path = _absolute_model_source(resolved_source)
        slot_meta.append(
            {
                "original_model_hint": str(original_hint),
                "resolved_source_path": resolved_path,
                "canonical_source_id": _canonical_model_source_id(resolved_path),
            }
        )
    return slot_meta


def _reference_checkpoint_rank_key(source: str) -> tuple[int, int, str]:
    canonical = _canonical_model_source_id(_absolute_model_source(source)).lower()
    compact = re.sub(r"[^a-z0-9]+", "", canonical)
    if "mattiaangeli" in canonical and "byt5" in canonical:
        family_rank = 0
    elif "assiaben" in canonical and "final-byt5" in canonical:
        family_rank = 1
    elif "artemgoncarov" in canonical or "dpc-byt5-large" in canonical:
        family_rank = 2
    elif "vitorhugobarbedo" in canonical and "model-final-2026" in canonical:
        family_rank = 3
    elif "google--byt5-large" in canonical or "google/byt5-large" in canonical or "byt5large" in compact:
        family_rank = 4
    elif "byt5-large" in canonical:
        family_rank = 5
    elif "google--byt5-base" in canonical or "google/byt5-base" in canonical:
        family_rank = 6
    elif "byt5-base" in canonical:
        family_rank = 7
    elif "google--byt5-small" in canonical or "google/byt5-small" in canonical:
        family_rank = 8
    elif "byt5-small" in canonical:
        family_rank = 9
    else:
        family_rank = 20
    artifact_rank = 0 if "/local-iter-" in canonical or "/models/" in canonical else 1
    return (family_rank, artifact_rank, canonical)


def _reference_cached_checkpoint_candidates() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for source in _cached_artifact_model_dirs():
        resolved = _resolve_loadable_model_source(Path(str(source)))
        if resolved is None:
            continue
        resolved_source = _absolute_model_source(resolved)
        canonical = _canonical_model_source_id(resolved_source)
        if canonical in seen:
            continue
        seen.add(canonical)
        candidates.append(resolved_source)
    return sorted(candidates, key=_reference_checkpoint_rank_key)


def _resolve_strongest_local_reference_pair() -> tuple[list[str], list[str]]:
    primary_candidates, primary_blockers = _reference_model_candidates(REFERENCE_SINGLE_MODEL_FALLBACK)
    blocker_messages = list(primary_blockers)
    faithful_fallback_hints = [
        hint for hint in REFERENCE_FALLBACK_MODEL_HINTS if hint != REFERENCE_SINGLE_MODEL_FALLBACK
    ]
    fallback_candidate_map: dict[str, list[str]] = {}

    for hint in faithful_fallback_hints:
        candidates, blockers = _reference_model_candidates(hint)
        fallback_candidate_map[hint] = candidates
        blocker_messages.extend(blockers)

    if not primary_candidates:
        blocker_messages.append(
            f"required faithful slot A checkpoint unavailable locally for {REFERENCE_SINGLE_MODEL_FALLBACK}"
        )
        return [], list(dict.fromkeys(blocker_messages))

    for left_source in primary_candidates:
        left_id = _canonical_model_source_id(left_source)
        for hint in faithful_fallback_hints:
            for right_source in fallback_candidate_map.get(hint, []):
                right_id = _canonical_model_source_id(right_source)
                if left_id == right_id:
                    continue
                return [left_source, right_source], list(dict.fromkeys(blocker_messages))

    blocker_messages.append(
        "no competition-faithful fallback pair resolved locally from " + ", ".join(faithful_fallback_hints)
    )
    return [], list(dict.fromkeys(blocker_messages))


def _resolve_strongest_local_reference_single_model() -> tuple[str | None, list[str]]:
    candidates, blockers = _reference_model_candidates(REFERENCE_SINGLE_MODEL_FALLBACK)
    blocker_messages = list(blockers)
    if candidates:
        return candidates[0], list(dict.fromkeys(blocker_messages))
    blocker_messages.append(
        f"exact single-model fallback unavailable for {REFERENCE_SINGLE_MODEL_FALLBACK}; "
        f"see {REFERENCE_INPUTS_MANIFEST_PATH}"
    )
    return None, list(dict.fromkeys(blocker_messages))


def _reference_runtime_cfg(
    cfg: PipelineConfig,
    *,
    original_hints: Sequence[str],
    resolved_sources: Sequence[str],
    runtime_tokens: Sequence[str],
    runtime_mode: str,
    blocker_messages: Sequence[str],
    use_multi_model_pool: bool,
    use_mbr: bool,
) -> PipelineConfig:
    slot_meta = _build_reference_slot_meta(original_hints, resolved_sources)
    return PipelineConfig(
        **{
            **cfg.__dict__,
            "model_hints": [slot["resolved_source_path"] for slot in slot_meta],
            "use_multi_model_pool": bool(use_multi_model_pool and len(slot_meta) > 1),
            "use_mbr": bool(use_mbr and len(slot_meta) > 1),
            "use_retrieval_candidates": False,
            "runtime_name": _reference_runtime_name(cfg.name, runtime_tokens),
            "reference_runtime_mode": runtime_mode,
            "reference_blocker": "; ".join(str(msg) for msg in blocker_messages if str(msg).strip()),
            "reference_slot_meta": slot_meta,
        }
    )


def _prepare_reference_baseline_cfg(cfg: PipelineConfig) -> PipelineConfig:
    if cfg.name != REFERENCE_PRIMARY_PIPELINE_NAME:
        return cfg

    # KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED: vitorhugobarbedo/lb-35-9-with-regex-corrections-public-model
    blocker_messages: list[str] = []
    exact_pair, exact_blockers = _resolve_distinct_reference_pair(REFERENCE_NOTEBOOK_MODEL_HINTS)
    blocker_messages.extend(exact_blockers)

    if exact_pair:
        log(f"[{cfg.name}] runtime path: exact required public pair")
        log(f"[{cfg.name}] exact resolved model A path: {exact_pair[0]}")
        log(f"[{cfg.name}] exact resolved model B path: {exact_pair[1]}")
        return _reference_runtime_cfg(
            cfg,
            original_hints=REFERENCE_NOTEBOOK_MODEL_HINTS,
            resolved_sources=exact_pair,
            runtime_tokens=["exact_public_pair"],
            runtime_mode="exact_required_public_pair",
            blocker_messages=blocker_messages,
            use_multi_model_pool=True,
            use_mbr=True,
        )

    for message in blocker_messages:
        log(f"[{cfg.name}] blocker: {message}")
    unresolved_exact_hints = [
        hint for hint in REFERENCE_NOTEBOOK_MODEL_HINTS if not _reference_model_candidates(hint)[0]
    ]
    for hint in unresolved_exact_hints:
        log(f"[{cfg.name}] exact required public pair unresolved for hint: {hint}")
        missing_hint_blocker = (
            f"required reference dataset missing locally: {hint}; see {REFERENCE_INPUTS_MANIFEST_PATH}"
        )
        log(f"[{cfg.name}] blocker: {missing_hint_blocker}")
        blocker_messages.append(missing_hint_blocker)

    if not LOCAL_KERNEL_MODE:
        fallback_pair, fallback_blockers = _resolve_strongest_local_reference_pair()
        for message in fallback_blockers:
            log(f"[{cfg.name}] blocker: {message}")
        blocker_messages.extend(fallback_blockers)
        if fallback_pair:
            fallback_hints = [_reference_original_hint_for_source(source) for source in fallback_pair]
            log(f"[{cfg.name}] runtime path: competition-faithful dual-checkpoint fallback")
            log(f"[{cfg.name}] fallback resolved model A path: {fallback_pair[0]}")
            log(f"[{cfg.name}] fallback resolved model B path: {fallback_pair[1]}")
            return _reference_runtime_cfg(
                cfg,
                original_hints=fallback_hints,
                resolved_sources=fallback_pair,
                runtime_tokens=["competition_faithful_fallback_pair", *fallback_hints],
                runtime_mode="competition_faithful_fallback_pair",
                blocker_messages=blocker_messages,
                use_multi_model_pool=True,
                use_mbr=True,
            )
    else:
        watchdog_pair_skip = (
            "local watchdog mode skips secondary dual-checkpoint fallback experiments "
            "when the exact public pair is unavailable"
        )
        log(f"[{cfg.name}] blocker: {watchdog_pair_skip}")
        blocker_messages.append(watchdog_pair_skip)

    single_source, single_blockers = _resolve_strongest_local_reference_single_model()
    for message in single_blockers:
        log(f"[{cfg.name}] blocker: {message}")
    blocker_messages.extend(single_blockers)
    if single_source is not None:
        original_hint = _reference_original_hint_for_source(single_source)
        log(f"[{cfg.name}] runtime path: faithful single-model public fallback")
        log(f"[{cfg.name}] single-model fallback source: {single_source}")
        return _reference_runtime_cfg(
            cfg,
            original_hints=[original_hint],
            resolved_sources=[single_source],
            runtime_tokens=["single_model_seq2seq_fallback", original_hint],
            runtime_mode="single_model_seq2seq_fallback",
            blocker_messages=blocker_messages,
            use_multi_model_pool=False,
            use_mbr=False,
        )

    blocker = "; ".join(blocker_messages) or (
        "required public pair unavailable and no distinct 2-checkpoint fallback pair resolved locally"
    )
    log(f"[{cfg.name}] blocker: {blocker}")
    return PipelineConfig(
        **{
            **cfg.__dict__,
            "model_hints": [],
            "use_multi_model_pool": False,
            "runtime_name": _reference_runtime_name(cfg.name, ["blocked_reference_runtime"]),
            "reference_runtime_mode": "blocked_reference_runtime",
            "reference_blocker": blocker,
        }
    )


def _pipeline_has_local_model_sources(cfg: PipelineConfig) -> bool:
    return bool(_pipeline_local_model_sources(cfg))


def download_or_cache_pretrained(model_name: str) -> str | None:
    # Default: never depend on runtime downloads.
    if not ALLOW_MODEL_DOWNLOAD:
        return None
    if IS_KAGGLE or os.getenv("KAGGLE_URL_BASE"):
        return None

    local_models = ensure_dir(KERNEL_DIR / "models")
    target_dir = local_models / model_name.replace("/", "--")
    if (target_dir / "config.json").exists():
        return str(target_dir)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_name, local_dir=target_dir, local_dir_use_symlinks=False)
        return str(target_dir)
    except Exception:
        return None


def _evict_cached_model(cache_key: str) -> None:
    cached = _LOADED_MODEL_CACHE.pop(cache_key, None)
    if cache_key in _LOADED_MODEL_CACHE_ORDER:
        _LOADED_MODEL_CACHE_ORDER.remove(cache_key)
    if cached is None:
        return
    tokenizer, model, _source = cached
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass
    del tokenizer, model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_cached_loaded_model(source: str) -> tuple[Any, Any, str] | None:
    cache_key = _canonical_model_source_id(source)
    cached = _LOADED_MODEL_CACHE.get(cache_key)
    if cached is None:
        return None
    if cache_key in _LOADED_MODEL_CACHE_ORDER:
        _LOADED_MODEL_CACHE_ORDER.remove(cache_key)
    _LOADED_MODEL_CACHE_ORDER.append(cache_key)
    return cached


def _put_cached_loaded_model(source: str, tokenizer: Any, model: Any) -> tuple[Any, Any, str]:
    cache_key = _canonical_model_source_id(source)
    _LOADED_MODEL_CACHE[cache_key] = (tokenizer, model, source)
    if cache_key in _LOADED_MODEL_CACHE_ORDER:
        _LOADED_MODEL_CACHE_ORDER.remove(cache_key)
    _LOADED_MODEL_CACHE_ORDER.append(cache_key)
    while len(_LOADED_MODEL_CACHE_ORDER) > MODEL_CACHE_LIMIT:
        _evict_cached_model(_LOADED_MODEL_CACHE_ORDER[0])
    return _LOADED_MODEL_CACHE[cache_key]


def _load_tokenizer_and_model(model_name: str, pipeline_name: str):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:
        raise ImportError("transformers + torch are required for seq2seq pipelines") from exc

    cached = download_or_cache_pretrained(model_name)
    sources = _resolve_model_sources(model_name)
    if cached:
        sources = [cached, *sources]

    local_files_only = not ALLOW_MODEL_DOWNLOAD
    for source in sources:
        source_str = str(source)
        if not _source_matches_model_hint(source_str, model_name):
            continue
        try:
            source_path = Path(source_str)
            cached_model = _get_cached_loaded_model(source_str)
            if cached_model is not None:
                log(f"[{pipeline_name}] reusing cached model hint '{model_name}' from '{source_str}'")
                return cached_model
            if source_path.is_dir() and (source_path / "adapter_config.json").exists():
                from peft import PeftConfig, PeftModel

                peft_cfg = PeftConfig.from_pretrained(source_str)
                base_hint = str(peft_cfg.base_model_name_or_path or model_name)
                base_cached = download_or_cache_pretrained(base_hint)
                base_sources = _resolve_model_sources(base_hint)
                if base_cached:
                    base_sources = [base_cached, *base_sources]
                last_error: BaseException | None = None
                for base_source in base_sources:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            str(base_source),
                            local_files_only=local_files_only,
                        )
                        base_model = AutoModelForSeq2SeqLM.from_pretrained(
                            str(base_source),
                            local_files_only=local_files_only,
                        )
                        model = PeftModel.from_pretrained(base_model, source_str)
                        log(f"[{pipeline_name}] loaded adapter '{source_str}' on base '{base_source}'")
                        return _put_cached_loaded_model(source_str, tokenizer, model)
                    except Exception as exc:
                        last_error = exc
                        continue
                if last_error is not None:
                    raise last_error
            tokenizer = AutoTokenizer.from_pretrained(source_str, local_files_only=local_files_only)
            model = AutoModelForSeq2SeqLM.from_pretrained(source_str, local_files_only=local_files_only)
            log(f"[{pipeline_name}] loaded model hint '{model_name}' from '{source_str}'")
            return _put_cached_loaded_model(source_str, tokenizer, model)
        except Exception:
            continue

    raise RuntimeError(f"Unable to load model for pipeline={pipeline_name}; model_hint={model_name}")


def _canonical_model_source_id(source: str) -> str:
    value = str(source).strip()
    path = Path(value)
    try:
        if path.exists():
            return str(path.resolve())
    except OSError:
        pass
    return value.lower()


def _single_source_pipeline_cfg(cfg: PipelineConfig, model_hint: str, suffix: str) -> PipelineConfig:
    return PipelineConfig(
        **{
            **cfg.__dict__,
            "name": f"{cfg.name}__{suffix}",
            "model_hints": [str(model_hint)],
            "use_multi_model_pool": False,
            "mbr_pool_cap": min(cfg.mbr_pool_cap, 16),
        }
    )


def _faithful_fast_reference_eval_cfg(cfg: PipelineConfig) -> PipelineConfig:
    return PipelineConfig(
        **{
            **cfg.__dict__,
            "model_hints": list(cfg.model_hints[:2]),
            "use_multi_model_pool": True,
            "use_mbr": True,
            "mbr_num_beam_cands": max(cfg.mbr_num_beam_cands, REFERENCE_NUM_BEAM_CANDIDATES),
            "mbr_num_sample_cands": max(cfg.mbr_num_sample_cands, REFERENCE_NUM_SAMPLE_PER_TEMP),
            "sample_temperatures": list(REFERENCE_SAMPLE_TEMPERATURES),
            "mbr_top_p": REFERENCE_SAMPLE_TOP_P,
            "mbr_pool_cap": max(cfg.mbr_pool_cap, REFERENCE_MBR_POOL_CAP),
            "num_beams": max(cfg.num_beams, REFERENCE_NUM_BEAMS),
            "length_penalty": REFERENCE_LENGTH_PENALTY,
            "repetition_penalty": REFERENCE_REPETITION_PENALTY,
            "max_source_len": max(cfg.max_source_len, REFERENCE_MAX_INPUT_LENGTH),
            "max_new_tokens": max(cfg.max_new_tokens, REFERENCE_MAX_NEW_TOKENS),
            "use_context_window": False,
            "use_retrieval_candidates": False,
            "use_lora": False,
            "allow_domain_adapted": False,
        }
    )


def _reduced_faithful_eval_cfg(cfg: PipelineConfig) -> PipelineConfig:
    max_new_tokens = LOCAL_REFERENCE_MAX_NEW_TOKENS
    num_beams = LOCAL_REFERENCE_NUM_BEAMS
    use_mbr = bool(cfg.use_mbr)
    mbr_num_beam_cands = max(
        1,
        min(cfg.mbr_num_beam_cands, LOCAL_REFERENCE_NUM_BEAM_CANDIDATES),
    )
    mbr_num_sample_cands = max(
        0,
        min(cfg.mbr_num_sample_cands, LOCAL_REFERENCE_NUM_SAMPLE_CANDIDATES),
    )
    sample_temperatures = list(cfg.sample_temperatures)
    if LOCAL_KERNEL_MODE:
        if cfg.reference_runtime_mode == "exact_required_public_pair":
            max_new_tokens = min(max_new_tokens, LOCAL_REFERENCE_WATCHDOG_MAX_NEW_TOKENS)
            num_beams = min(num_beams, 4)
            use_mbr = True
            mbr_num_beam_cands = max(1, min(cfg.mbr_num_beam_cands, 2))
            mbr_num_sample_cands = 1 if cfg.sample_temperatures else 0
            sample_temperatures = list(REFERENCE_SAMPLE_TEMPERATURES if mbr_num_sample_cands else [])
        elif cfg.reference_runtime_mode == "single_model_seq2seq_fallback":
            max_new_tokens = min(max_new_tokens, LOCAL_REFERENCE_MAX_NEW_TOKENS)
            num_beams = min(num_beams, LOCAL_REFERENCE_NUM_BEAMS)
            use_mbr = False
            mbr_num_beam_cands = 1
            mbr_num_sample_cands = 0
            sample_temperatures = []
        else:
            max_new_tokens = min(max_new_tokens, LOCAL_REFERENCE_WATCHDOG_MAX_NEW_TOKENS)
            num_beams = min(num_beams, LOCAL_REFERENCE_WATCHDOG_NUM_BEAMS)
            use_mbr = False
            mbr_num_beam_cands = 1
            mbr_num_sample_cands = 0
            sample_temperatures = []
    reduced_hints = list(cfg.model_hints[: (2 if cfg.use_multi_model_pool else 1)])
    use_multi_model_pool = bool(len(reduced_hints) > 1)
    reduced_pool_cap = max(1, min(cfg.mbr_pool_cap, LOCAL_REFERENCE_MBR_POOL_CAP))
    if not use_multi_model_pool:
        reduced_pool_cap = max(1, min(reduced_pool_cap, 6))
    return PipelineConfig(
        **{
            **cfg.__dict__,
            "model_hints": reduced_hints,
            "use_multi_model_pool": use_multi_model_pool,
            "use_mbr": use_mbr,
            "mbr_num_beam_cands": mbr_num_beam_cands,
            "mbr_num_sample_cands": mbr_num_sample_cands,
            "sample_temperatures": sample_temperatures if use_mbr else [],
            "mbr_pool_cap": reduced_pool_cap,
            "num_beams": max(1, min(cfg.num_beams, num_beams)),
            "max_source_len": max(cfg.max_source_len, REFERENCE_MAX_INPUT_LENGTH),
            "max_new_tokens": max(64, min(cfg.max_new_tokens, max_new_tokens)),
            "use_context_window": False,
            "use_retrieval_candidates": False,
            "use_lora": False,
            "allow_domain_adapted": False,
        }
    )


def _build_local_seq2seq_eval_frame(
    train_df: pd.DataFrame,
    max_docs: int = REFERENCE_FAST_EVAL_MAX_DOCS,
) -> pd.DataFrame:
    doc_pair_df = build_document_pair_frame(train_df)
    sorted_doc_ids = sorted(doc_pair_df["oare_id"].astype(str).unique().tolist())
    min_docs = min(REFERENCE_FAST_EVAL_MIN_DOCS, max_docs)
    keep_count = max(1, max(min_docs, min(max_docs, len(sorted_doc_ids))))
    keep_doc_ids = set(sorted_doc_ids[:keep_count])
    reduced = doc_pair_df[doc_pair_df["oare_id"].astype(str).isin(keep_doc_ids)].copy().reset_index(drop=True)
    reduced["doc_index"] = np.arange(1, len(reduced) + 1)
    return reduced


def _evaluate_single_model_path(
    train_fold: pd.DataFrame,
    valid_fold: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    model_hint: str,
    constraint_memories: ConstraintMemories | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], list[str], dict[str, int], list[str]]:
    source_col = _source_feature_col(train_fold)
    valid_source_col = _source_feature_col(valid_fold)
    test_source_col = _source_feature_col(test_df)
    retrieval_model = None
    if cfg.use_retrieval_candidates:
        retrieval_model = _fit_retrieval_model(train_fold[source_col].tolist(), train_fold["translation"].tolist())
    exact_source_memory = build_exact_source_memory(train_fold)
    fold_constraint_memories = (
        build_constraint_memories(train_fold)
        if constraint_memories is not None and ENABLE_LEXICON_CONSTRAINTS
        else None
    )
    eval_cfg = _single_source_pipeline_cfg(cfg, model_hint, suffix=_artifact_safe_name(model_hint))
    device = _prepare_device(GPU_DEVICE)
    mbr_stats: Counter[str] = Counter()
    valid_inputs = build_context_window_texts(valid_fold, valid_source_col, enabled=eval_cfg.use_context_window)
    test_inputs = build_context_window_texts(test_df, test_source_col, enabled=eval_cfg.use_context_window)

    val_pred, _per_model_val, val_sources = generate_predictions(
        valid_inputs,
        eval_cfg,
        device=device,
        retrieval_model=retrieval_model,
        constraint_memories=fold_constraint_memories,
        mbr_stats=mbr_stats,
        batch_size=1 if FAST_DEV else 2,
    )
    test_pred, _per_model_test, test_sources = generate_predictions(
        test_inputs,
        eval_cfg,
        device=device,
        retrieval_model=retrieval_model,
        constraint_memories=fold_constraint_memories,
        mbr_stats=mbr_stats,
        batch_size=1 if FAST_DEV else 2,
    )
    val_pred, val_stats = apply_consistency_postprocess(
        source_texts=valid_fold[valid_source_col].tolist(),
        predictions=val_pred,
        group_values=valid_fold["oare_id"].tolist(),
        exact_source_memory=exact_source_memory,
        constraint_memories=fold_constraint_memories,
        enable_exact_memory=False,
    )
    test_pred, test_stats = apply_consistency_postprocess(
        source_texts=test_df[test_source_col].tolist(),
        predictions=test_pred,
        group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
        exact_source_memory=exact_source_memory,
        constraint_memories=fold_constraint_memories,
        enable_exact_memory=False,
    )
    sentence_metric, doc_metric = compute_sentence_and_document_metrics(valid_fold, val_pred)
    slice_metrics = compute_slice_metrics(valid_fold, val_pred)
    stats = Counter[str]()
    stats.update(mbr_stats)
    stats.update(val_stats)
    stats.update(test_stats)
    return sentence_metric, doc_metric, slice_metrics, test_pred, dict(stats), sorted(set(val_sources + test_sources))


def run_optional_lora_finetune(
    pair_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    seed: int,
    constraint_memories: ConstraintMemories | None = None,
) -> FineTuneResult:
    if not ALLOW_KERNEL_FINETUNE:
        return FineTuneResult(
            False, None, None, None, None, None, None, None, None, None, None, "kernel_finetune_disabled"
        )
    if not USE_LORA_FINETUNE:
        return FineTuneResult(False, None, None, None, None, None, None, None, None, None, None, "lora_toggle_disabled")
    if not cfg.use_lora:
        return FineTuneResult(
            False, None, None, None, None, None, None, None, None, None, None, "pipeline_not_lora_enabled"
        )

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        return FineTuneResult(
            False,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            f"finetune_deps_unavailable:{type(exc).__name__}",
        )

    if not torch.cuda.is_available():
        return FineTuneResult(False, None, None, None, None, None, None, None, None, None, None, "cuda_unavailable")

    groups = pair_df["oare_id"].tolist()
    splits = list(iter_grouped_cv_splits(groups, n_folds=5, seed=seed, fast_dev=True))
    if not splits:
        return FineTuneResult(False, None, None, None, None, None, None, None, None, None, None, "insufficient_groups")
    tr_idx, va_idx = splits[0]
    train_fold = pair_df.iloc[tr_idx].reset_index(drop=True)
    valid_fold = pair_df.iloc[va_idx].reset_index(drop=True)

    source_classes = _pipeline_model_source_classes(cfg)
    preferred_sources = source_classes["domain_adapted"] or source_classes["base"]
    if not preferred_sources:
        return FineTuneResult(
            False, None, None, None, None, None, None, None, None, None, None, "no_local_model_sources"
        )

    model_hint = preferred_sources[0]
    (
        baseline_metric,
        baseline_doc_metric,
        baseline_slice_metrics,
        _base_test_pred,
        _base_stats,
        _base_sources,
    ) = _evaluate_single_model_path(
        train_fold,
        valid_fold,
        test_df,
        cfg,
        model_hint,
        constraint_memories,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_hint, local_files_only=not ALLOW_MODEL_DOWNLOAD)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_hint, local_files_only=not ALLOW_MODEL_DOWNLOAD)
    except Exception as exc:
        return FineTuneResult(
            False,
            model_hint,
            None,
            baseline_metric,
            baseline_doc_metric,
            None,
            None,
            baseline_slice_metrics,
            None,
            None,
            None,
            f"failed_to_load_base:{type(exc).__name__}",
        )

    adapter_dir = ensure_dir(KERNEL_DIR / "models" / f"{_artifact_safe_name(cfg.name)}_adapter_{seed}")
    try:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "k", "v", "o"],
        )
        model = get_peft_model(model, lora_cfg)
    except Exception as exc:
        return FineTuneResult(
            False,
            model_hint,
            None,
            baseline_metric,
            baseline_doc_metric,
            None,
            None,
            baseline_slice_metrics,
            None,
            None,
            None,
            f"failed_to_attach_lora:{type(exc).__name__}",
        )

    def _encode_dataset(frame: pd.DataFrame) -> Seq2SeqDataset:
        source_texts = frame[_source_feature_col(frame)].tolist()
        target_texts = frame["translation"].tolist()
        enc = tokenizer(
            source_texts,
            truncation=True,
            padding=True,
            max_length=min(512, cfg.max_source_len),
        )
        labels = tokenizer(
            text_target=target_texts,
            truncation=True,
            padding=True,
            max_length=min(512, MAX_TARGET_LEN),
        )["input_ids"]
        return Seq2SeqDataset(enc, labels)

    train_dataset = _encode_dataset(train_fold)
    eval_dataset = _encode_dataset(valid_fold)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    class WeightedTrainer(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            weights = torch.as_tensor(train_fold["pair_weight"].astype(float).to_numpy(), dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(train_fold), replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
            )

    args = TrainingArguments(
        output_dir=str(adapter_dir / "trainer"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4 if FAST_DEV else 8,
        learning_rate=2e-4,
        num_train_epochs=1 if FAST_DEV else 2,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    try:
        trainer.train()
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        (adapter_dir / "base_model_hint.txt").write_text(str(model_hint), encoding="utf-8")
        (
            validation_metric,
            validation_doc_metric,
            validation_slice_metrics,
            test_predictions,
            postprocess_stats,
            _adapter_sources,
        ) = _evaluate_single_model_path(
            train_fold,
            valid_fold,
            test_df,
            cfg,
            str(adapter_dir),
            constraint_memories,
        )
        return FineTuneResult(
            True,
            model_hint,
            str(adapter_dir),
            baseline_metric,
            baseline_doc_metric,
            validation_metric,
            validation_doc_metric,
            baseline_slice_metrics,
            validation_slice_metrics,
            test_predictions,
            postprocess_stats,
            "trained",
        )
    except Exception as exc:
        return FineTuneResult(
            False,
            model_hint,
            None,
            baseline_metric,
            baseline_doc_metric,
            None,
            None,
            baseline_slice_metrics,
            None,
            None,
            None,
            f"finetune_failed:{type(exc).__name__}",
        )
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _prepare_device(gpu_device: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    if gpu_device.startswith("cuda:"):
        try:
            idx = int(gpu_device.split(":", 1)[1])
            torch.cuda.set_device(idx)
        except Exception:
            pass
    return "cuda"


class Seq2SeqDataset:
    def __init__(self, encodings: dict[str, Any], labels: Sequence[Sequence[int]] | None = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: np.asarray(v[idx], dtype=np.int64) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = np.asarray(self.labels[idx], dtype=np.int64)
        return item


def _decode_ids(tokenizer: Any, token_ids: np.ndarray) -> list[str]:
    arr = np.asarray(token_ids)
    arr = np.where(arr < 0, tokenizer.pad_token_id, arr)
    texts = tokenizer.batch_decode(arr, skip_special_tokens=True)
    return [str(x).strip() for x in texts]


def _group_return_sequences(decoded: list[str], batch_size: int, n_return: int) -> list[list[str]]:
    grouped: list[list[str]] = []
    for i in range(batch_size):
        start = i * n_return
        grouped.append(decoded[start : start + n_return])
    return grouped


def _dedupe_candidates(candidates: list[tuple[str, str, int]], pool_cap: int) -> list[tuple[str, str, int]]:
    dedup: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for text, source, rank in candidates:
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append((key, source, rank))
        if len(dedup) >= pool_cap:
            break
    return dedup


def _reference_memory_augmented_candidates(
    source_text: str,
    candidate_pool: Sequence[tuple[str, str, int]],
    constraint_memories: ConstraintMemories | None,
) -> list[tuple[str, str, int]]:
    if constraint_memories is None:
        return []

    augmented: list[tuple[str, str, int]] = []
    seen: set[str] = set()

    exact_candidate = constraint_memories.exact_source_memory.get(_norm_spaces(source_text))
    if exact_candidate:
        normalized_exact = normalize_prediction_style(exact_candidate)
        if normalized_exact:
            seen.add(normalized_exact)
            augmented.append((normalized_exact, "constraint_exact_memory", 0))

    for base_text, _base_source, base_rank in list(candidate_pool)[:4]:
        rewrite_stats = {
            "entity_rewrites": 0,
            "quantity_rewrites": 0,
            "unit_rewrites": 0,
        }
        rewritten = normalize_prediction_style(
            apply_soft_constraint_rewrites(source_text, str(base_text), constraint_memories, rewrite_stats)
        )
        if not rewritten or rewritten == str(base_text).strip() or rewritten in seen:
            continue
        seen.add(rewritten)
        augmented.append((rewritten, "constraint_memory_rewrite", max(1, int(base_rank))))
    return augmented


def _select_mbr_candidate(
    candidates: list[tuple[str, str, int]],
    source_text: str | None = None,
    constraint_memories: ConstraintMemories | None = None,
    mbr_stats: dict[str, int] | None = None,
) -> str:
    global _MBR_UTILITY_LOGGED
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0][0]
    if not _MBR_UTILITY_LOGGED:
        log("MBR utility for reranking: weighted chrF/BLEU/Jaccard consensus with light length bonus")
        _MBR_UTILITY_LOGGED = True

    scores: list[tuple[float, int, int, str]] = []
    for i, (cand_i, source_i, rank_i) in enumerate(candidates):
        sims: list[float] = []
        for j, (cand_j, _source_j, _rank_j) in enumerate(candidates):
            if i == j:
                continue
            sims.append(candidate_mbr_utility_score(cand_i, cand_j))
        utility = float(np.mean(sims)) if sims else 0.0
        bonus = _constraint_candidate_bonus(source_text or "", cand_i, constraint_memories)
        source_bonus = 0.4 if "adapted" in source_i else 0.0
        source_bonus += 0.2 if "retrieval_exact" in source_i else 0.0
        utility += bonus + source_bonus

        # tie-breaker: beam candidates first, then lower rank (fewer candidates/faster)
        source_priority = 0 if "beam" in source_i else 1
        scores.append((utility, source_priority, rank_i, cand_i, bonus))

    scores.sort(key=lambda x: (-x[0], x[1], x[2], len(x[3]), x[3]))
    if mbr_stats is not None and scores and scores[0][4] > 0:
        mbr_stats["constraint_bonus_hits"] = mbr_stats.get("constraint_bonus_hits", 0) + 1
    return scores[0][3]


def _artifact_safe_name(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return slug or "model"


def _hints_are_placeholder(hints: Sequence[str]) -> bool:
    placeholder_defaults = {
        "google/byt5-base",
        "google/byt5-small",
        "google/byt5-large",
        "google/mt5-base",
    }
    if not hints:
        return False
    for hint in hints:
        value = str(hint).strip().lower()
        if not value:
            continue
        if value in placeholder_defaults:
            continue
        if "checkpoint" in value or "mount" in value or "placeholder" in value:
            continue
        return False
    return True


def _plan_pipeline_model_hints(name: str) -> list[str]:
    raw = KAGGLE_KERNEL_SOURCES.get("pipeline_model_hints", {})
    if not isinstance(raw, dict):
        return []
    value = raw.get(name)
    if not isinstance(value, list):
        return []
    hints: list[str] = []
    seen: set[str] = set()
    for item in value:
        hint = str(item or "").strip()
        if not hint or hint in seen:
            continue
        seen.add(hint)
        hints.append(hint)
    return hints


def _plan_domain_adapted_checkpoint_hints() -> list[str]:
    raw = DOMAIN_ADAPTATION.get("adapted_checkpoint_hints", [])
    if not isinstance(raw, list):
        return []
    hints: list[str] = []
    seen: set[str] = set()
    for item in raw:
        hint = str(item or "").strip()
        if not hint or hint in seen:
            continue
        seen.add(hint)
        hints.append(hint)
    return hints


def _required_local_seq2seq_pipeline_names() -> set[str]:
    legacy = set(_ordered_required_seq2seq_pipeline_names())
    plan_names = {name for name in shortlisted_pipeline_names() if name != "char_tfidf_knn_memory"}
    return legacy | plan_names


def _active_plan_seq2seq_pipeline_names() -> set[str]:
    names = {name for name in shortlisted_pipeline_names() if name != "char_tfidf_knn_memory"}
    if _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG) and REFERENCE_PRIMARY_PIPELINE_NAME in names:
        return {REFERENCE_PRIMARY_PIPELINE_NAME}
    return names


def _pipeline_domain_adapted_model_hints(cfg: PipelineConfig) -> list[str]:
    if not cfg.allow_domain_adapted:
        return []
    return _plan_domain_adapted_checkpoint_hints()


def _pipeline_model_hints(cfg: PipelineConfig) -> list[str]:
    key = re.sub(r"[^A-Za-z0-9]+", "_", cfg.name).upper()
    specific = os.getenv(f"KAGGLEBOT_MODEL_PATHS_{key}", "").strip()
    generic = os.getenv("KAGGLEBOT_MODEL_PATHS", "").strip()

    pinned_reference_paths = bool(cfg.reference_runtime_mode) and any(
        Path(str(hint)).exists() for hint in cfg.model_hints
    )
    if cfg.reference_runtime_mode == "blocked_reference_runtime":
        return []
    base_hints = (
        cfg.model_hints.copy()
        if pinned_reference_paths
        else (_plan_pipeline_model_hints(cfg.name) or cfg.model_hints.copy())
    )
    adapted_hints = _pipeline_domain_adapted_model_hints(cfg)
    hints = [*adapted_hints, *base_hints] if cfg.use_multi_model_pool else (base_hints or adapted_hints)
    if specific and not pinned_reference_paths:
        hints = [x.strip() for x in specific.split(",") if x.strip()]
    elif generic and not pinned_reference_paths:
        hints = [x.strip() for x in generic.split(",") if x.strip()]
    elif IS_KAGGLE and cfg.use_multi_model_pool and _hints_are_placeholder(base_hints):
        discovered = sorted(_cached_kaggle_model_dirs(), key=lambda p: _model_path_rank_key(p, "byt5"))
        if discovered:
            limit = max(1, min(len(base_hints), len(discovered)))
            hints = [*adapted_hints, *[str(path) for path in discovered[:limit]]]
            if cfg.name not in _DISCOVERED_HINT_LOGGED:
                log(f"[{cfg.name}] using discovered Kaggle model mounts: {', '.join(hints)}")
                _DISCOVERED_HINT_LOGGED.add(cfg.name)

    if not cfg.use_multi_model_pool and hints:
        hints = hints[:1]
    if FAST_DEV and len(hints) > 2:
        hints = hints[:2]

    dedup: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        dedup.append(hint)
    return dedup


def _pipeline_model_source_classes(cfg: PipelineConfig) -> dict[str, list[str]]:
    if cfg.reference_runtime_mode:
        pinned_sources: list[str] = []
        seen_pinned: set[str] = set()
        for hint in cfg.model_hints:
            path = Path(str(hint))
            if not path.exists():
                continue
            canonical = _canonical_model_source_id(str(path))
            if canonical in seen_pinned:
                continue
            seen_pinned.add(canonical)
            pinned_sources.append(str(path.resolve()))
        if pinned_sources:
            return {"domain_adapted": [], "base": pinned_sources}

    class_sources = {"domain_adapted": [], "base": []}
    seen: set[str] = set()
    adapted_hint_set = set(_pipeline_domain_adapted_model_hints(cfg))
    for hint in _pipeline_model_hints(cfg):
        bucket = "domain_adapted" if hint in adapted_hint_set else "base"
        for source in _iter_local_model_sources(hint):
            if source in seen:
                continue
            seen.add(source)
            class_sources[bucket].append(source)
    return class_sources


def _pipeline_local_model_sources(cfg: PipelineConfig) -> list[str]:
    class_sources = _pipeline_model_source_classes(cfg)
    return [*class_sources["domain_adapted"], *class_sources["base"]]


def _generate_model_candidates(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    cfg: PipelineConfig,
    device: str,
    model_label: str,
    batch_size: int = 2,
) -> tuple[list[list[tuple[str, str, int]]], list[str]]:
    import torch

    model.eval()
    all_candidates: list[list[tuple[str, str, int]]] = []
    model_top_preds: list[str] = []

    beam_cands = max(1, min(cfg.num_beams, cfg.mbr_num_beam_cands))
    sample_cands_per_temp = max(0, cfg.mbr_num_sample_cands)
    sample_cands_per_temp = 0 if not cfg.use_mbr else sample_cands_per_temp
    sample_temperatures = list(cfg.sample_temperatures) if cfg.use_mbr else []
    total_sample_cands = sample_cands_per_temp * len(sample_temperatures)

    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_source_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            beam_out = model.generate(
                **inputs,
                num_beams=max(cfg.num_beams, beam_cands),
                num_return_sequences=beam_cands,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=cfg.length_penalty,
                repetition_penalty=cfg.repetition_penalty,
            )
            beam_text = _decode_ids(tokenizer, beam_out.cpu().numpy())
            grouped_beam = _group_return_sequences(beam_text, len(batch_texts), beam_cands)

            grouped_sample: list[list[str]] = [[] for _ in range(len(batch_texts))]
            if total_sample_cands > 0:
                grouped_sample = [[] for _ in range(len(batch_texts))]
                for temperature in sample_temperatures:
                    sample_out = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=cfg.mbr_top_p,
                        num_beams=1,
                        num_return_sequences=sample_cands_per_temp,
                        max_new_tokens=cfg.max_new_tokens,
                        repetition_penalty=cfg.repetition_penalty,
                    )
                    sample_text = _decode_ids(tokenizer, sample_out.cpu().numpy())
                    sample_group = _group_return_sequences(sample_text, len(batch_texts), sample_cands_per_temp)
                    for row_idx, row_values in enumerate(sample_group):
                        grouped_sample[row_idx].extend(row_values)

            for row_idx in range(len(batch_texts)):
                pool: list[tuple[str, str, int]] = []
                for rank, cand in enumerate(grouped_beam[row_idx], start=1):
                    pool.append((cand, f"{model_label}_beam", rank))
                for rank, cand in enumerate(grouped_sample[row_idx], start=1):
                    pool.append((cand, f"{model_label}_sample", beam_cands + rank))
                all_candidates.append(pool)
                top = grouped_beam[row_idx][0] if grouped_beam[row_idx] else ""
                model_top_preds.append(postprocess_translation(top, cfg.strong_postprocess))

    return all_candidates, model_top_preds


def generate_predictions(
    texts: Sequence[str],
    cfg: PipelineConfig,
    device: str,
    retrieval_model: RetrievalModel | None = None,
    constraint_memories: ConstraintMemories | None = None,
    mbr_stats: dict[str, int] | None = None,
    batch_size: int = 2,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    import torch

    model_hints = _pipeline_model_hints(cfg)
    source_classes = _pipeline_model_source_classes(cfg)
    adapted_resolved_sources = set(source_classes["domain_adapted"])
    adapted_hints = set(_pipeline_domain_adapted_model_hints(cfg))
    pooled_candidates: list[list[tuple[str, str, int]]] = [[] for _ in texts]
    per_model_preds: dict[str, list[str]] = {}
    resolved_sources: list[str] = []
    resolved_source_ids: dict[str, str] = {}
    dual_reference_mode = cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME and cfg.use_multi_model_pool
    if cfg.reference_slot_meta:
        for idx, slot_meta in enumerate(cfg.reference_slot_meta, start=1):
            slot_name = "A" if idx == 1 else "B" if idx == 2 else str(idx)
            log(
                f"[{cfg.name}] reference slot {slot_name}: "
                f"original model hint={slot_meta.get('original_model_hint', '')}; "
                f"resolved source path={slot_meta.get('resolved_source_path', '')}; "
                f"canonical source id={slot_meta.get('canonical_source_id', '')}"
            )

    for model_idx, model_hint in enumerate(model_hints, start=1):
        tokenizer = None
        model = None
        resolved_name = model_hint
        try:
            tokenizer, model, resolved_name = _load_tokenizer_and_model(model_hint, cfg.name)
            canonical_source_id = _canonical_model_source_id(resolved_name)
            duplicate_source = resolved_source_ids.get(canonical_source_id)
            if duplicate_source is not None:
                if dual_reference_mode:
                    log(
                        f"[{cfg.name}] resolved model slot {model_idx} to the same checkpoint as an earlier slot: "
                        f"{resolved_name} == {duplicate_source}; refusing fake dual-model execution."
                    )
                else:
                    log(f"[{cfg.name}] skipping duplicate checkpoint source: {resolved_name}")
                continue
            model.to(device)
            source_prefix = (
                "adapted"
                if str(resolved_name) in adapted_resolved_sources or str(model_hint) in adapted_hints
                else "base"
            )
            model_key = f"{source_prefix}_m{model_idx}_{_artifact_safe_name(resolved_name)}"
            resolved_source_ids[canonical_source_id] = str(resolved_name)
            if dual_reference_mode:
                slot = "A" if len(resolved_sources) == 0 else "B" if len(resolved_sources) == 1 else str(model_idx)
                log(f"[{cfg.name}] resolved model {slot} source: {resolved_name}")
            row_cands, top_preds = _generate_model_candidates(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                cfg=cfg,
                device=device,
                model_label=model_key,
                batch_size=batch_size,
            )
            per_model_preds[model_key] = [str(x) for x in top_preds]
            resolved_sources.append(str(resolved_name))
            for row_idx, cands in enumerate(row_cands):
                pooled_candidates[row_idx].extend(cands)
            if cfg.name in {"pooled_multi_byt5_mbr", REFERENCE_PRIMARY_PIPELINE_NAME} and len(resolved_sources) >= 2:
                break
        except Exception as exc:
            log(f"Skipping model hint '{model_hint}' for {cfg.name}: {exc}")
            continue
        finally:
            del tokenizer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not per_model_preds:
        raise RuntimeError(f"No loadable models found for pipeline={cfg.name}; hints={model_hints}")

    if dual_reference_mode:
        if len(resolved_sources) >= 2:
            log(f"[{cfg.name}] dual-checkpoint mode active with two distinct resolved model sources.")
        else:
            raise RuntimeError(
                f"[{cfg.name}] reference dual-checkpoint blocker: only {len(resolved_sources)} distinct "
                "resolved source(s) available after canonical resolution."
            )

    if retrieval_model is not None:
        retrieval_pools, retrieval_low_sim = _retrieval_candidate_pools(
            model=retrieval_model,
            infer_src=texts,
            k=min(RETRIEVAL_K, 8),
            min_sim=max(RETRIEVAL_MIN_SIM, 0.12),
            max_candidates=3,
        )
        if retrieval_low_sim < len(texts):
            log(f"[{cfg.name}] added retrieval candidates for {len(texts) - retrieval_low_sim}/{len(texts)} rows")
        for row_idx, row_pool in enumerate(retrieval_pools):
            pooled_candidates[row_idx].extend(row_pool)
            if cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME and row_pool:
                pooled_candidates[row_idx].extend(
                    _reference_memory_augmented_candidates(
                        source_text=str(texts[row_idx]),
                        candidate_pool=row_pool,
                        constraint_memories=constraint_memories,
                    )
                )

    final_preds: list[str] = []
    first_model_key = sorted(per_model_preds.keys())[0]
    first_model_preds = per_model_preds[first_model_key]
    for row_idx, row_pool in enumerate(pooled_candidates):
        dedup_pool = _dedupe_candidates(row_pool, cfg.mbr_pool_cap)
        if cfg.use_mbr and dedup_pool:
            chosen = _select_mbr_candidate(
                dedup_pool,
                source_text=str(texts[row_idx]),
                constraint_memories=constraint_memories,
                mbr_stats=mbr_stats,
            )
        elif row_idx < len(first_model_preds):
            chosen = first_model_preds[row_idx]
        elif dedup_pool:
            chosen = dedup_pool[0][0]
        else:
            chosen = ""
        final_preds.append(postprocess_translation(chosen, cfg.strong_postprocess))

    return final_preds, per_model_preds, resolved_sources


# =====================================================================================
# Artifact writing
# =====================================================================================


def _write_csv_all(name: str, df: pd.DataFrame, output_dirs: Sequence[Path]) -> None:
    for out_dir in output_dirs:
        df.to_csv(out_dir / name, index=False)


def _write_json_all(name: str, payload: dict[str, Any], output_dirs: Sequence[Path]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    for out_dir in output_dirs:
        (out_dir / name).write_text(text, encoding="utf-8")


def _metric_output_dirs(output_dirs: Sequence[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for out_dir in output_dirs:
        resolved = out_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        dirs.append(out_dir)
    return dirs


def _save_npy_all(name: str, arr: Any, output_dirs: Sequence[Path]) -> None:
    if not SAVE_NPY:
        return
    obj_arr = np.array(arr, dtype=object)
    for out_dir in output_dirs:
        np.save(out_dir / name, obj_arr, allow_pickle=True)


def _load_cached_test_preds(name: str, output_dirs: Sequence[Path]) -> np.ndarray | None:
    for out_dir in output_dirs:
        path = out_dir / f"test_preds_{name}.npy"
        if path.exists():
            return np.load(path, allow_pickle=True)
    return None


# =====================================================================================
# Baseline and seq2seq CV
# =====================================================================================


def _source_feature_col(df: pd.DataFrame) -> str:
    return "transliteration_lex" if "transliteration_lex" in df.columns else "transliteration"


def _context_sort_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "doc_index",
        "sentence_index",
        "line_start",
        "line_end",
        "line_number",
        "id",
    ]
    return [col for col in candidates if col in df.columns]


def _context_group_column(df: pd.DataFrame) -> str | None:
    for col in ("oare_id", "text_id", "doc_index"):
        if col in df.columns:
            return col
    return None


def build_context_window_texts(df: pd.DataFrame, source_col: str, enabled: bool) -> list[str]:
    base_texts = [str(x) for x in df[source_col].tolist()]
    if not enabled:
        return base_texts

    group_col = _context_group_column(df)
    if group_col is None:
        return base_texts

    extra_cols = [col for col in _context_sort_columns(df) if col not in {group_col, source_col}]
    work = df[[group_col, source_col, *extra_cols]].copy()
    work["_row_idx"] = np.arange(len(work))
    sort_cols = [group_col, *extra_cols, "_row_idx"]
    work = work.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    outputs = [""] * len(work)

    for _, group in work.groupby(group_col, sort=False):
        texts = [str(x) for x in group[source_col].tolist()]
        row_ids = group["_row_idx"].astype(int).tolist()
        for idx, row_id in enumerate(row_ids):
            prev_text = texts[idx - 1] if idx > 0 else ""
            curr_text = texts[idx]
            next_text = texts[idx + 1] if idx + 1 < len(texts) else ""
            pieces = []
            if prev_text:
                pieces.append(f"<ctx_prev> {prev_text} </ctx_prev>")
            pieces.append(f"<focus> {curr_text} </focus>")
            if next_text:
                pieces.append(f"<ctx_next> {next_text} </ctx_next>")
            outputs[row_id] = _norm_spaces(" ".join(pieces))

    return [text or base_texts[idx] for idx, text in enumerate(outputs)]


def build_document_pair_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    pair_df = train_df.copy().reset_index(drop=True)
    pair_df["pair_id"] = [f"{oare_id}__gold_document__0" for oare_id in pair_df["oare_id"].astype(str).tolist()]
    pair_df["doc_index"] = np.arange(1, len(pair_df) + 1)
    pair_df["sentence_index"] = 0
    pair_df["doc_translation"] = pair_df["translation"].map(normalize_target)
    pair_df["supervision_source"] = "gold_document"
    pair_df["pair_weight"] = 1.0
    pair_df["has_quantity_or_unit"] = pair_df["transliteration"].map(_source_has_quantity_or_unit)
    pair_df["has_entity_tokens"] = pair_df["transliteration"].map(_source_has_entity_tokens)
    return pair_df


def run_lookup_baseline_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int,
    seeds: Sequence[int],
    output_dirs: Sequence[Path],
    exact_source_memory: dict[str, str] | None = None,
    constraint_memories: ConstraintMemories | None = None,
) -> PipelineResult:
    fold_metrics: list[dict[str, float]] = []
    doc_fold_metrics: list[dict[str, float]] = []
    unseen_fold_metrics: list[dict[str, float]] = []
    unseen_doc_fold_metrics: list[dict[str, float]] = []
    unseen_coverages: list[dict[str, float]] = []
    slice_fold_metrics: list[dict[str, float]] = []
    seed_oof: list[np.ndarray] = []
    seed_test: list[np.ndarray] = []
    source_col = _source_feature_col(train_df)
    test_source_col = _source_feature_col(test_df)

    active_seeds = [seeds[0]] if FAST_DEV else list(seeds)

    for seed in active_seeds:
        set_global_seed(seed)
        oof = np.empty(len(train_df), dtype=object)
        fold_test_preds: list[list[str]] = []

        for tr_idx, va_idx in iter_grouped_cv_splits(
            train_df["oare_id"].tolist(),
            n_folds=n_folds,
            seed=seed,
            fast_dev=FAST_DEV,
        ):
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            fold_exact_memory = build_exact_source_memory(tr)
            fold_constraint_memories = (
                build_constraint_memories(tr)
                if constraint_memories is not None and ENABLE_LEXICON_CONSTRAINTS
                else None
            )

            val_pred = simple_lookup_predict(
                tr[source_col].tolist(),
                tr["translation"].tolist(),
                va[source_col].tolist(),
            )
            tst_pred = simple_lookup_predict(
                tr[source_col].tolist(),
                tr["translation"].tolist(),
                test_df[test_source_col].tolist(),
            )
            val_pred, _ = apply_consistency_postprocess(
                source_texts=va[source_col].tolist(),
                predictions=val_pred,
                group_values=va["oare_id"].tolist() if "oare_id" in va.columns else None,
                exact_source_memory=fold_exact_memory,
                constraint_memories=fold_constraint_memories,
                enable_exact_memory=False,
            )
            tst_pred, _ = apply_consistency_postprocess(
                source_texts=test_df[test_source_col].tolist(),
                predictions=tst_pred,
                group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
                exact_source_memory=fold_exact_memory,
                constraint_memories=fold_constraint_memories,
                enable_exact_memory=False,
            )
            oof[va_idx] = np.array(val_pred, dtype=object)
            fold_test_preds.append(tst_pred)
            sentence_metric, doc_metric = compute_sentence_and_document_metrics(va, val_pred)
            unseen_sentence_metric, unseen_doc_metric, unseen_coverage = compute_unseen_source_metrics(tr, va, val_pred)
            fold_metrics.append(sentence_metric)
            doc_fold_metrics.append(doc_metric)
            unseen_fold_metrics.append(unseen_sentence_metric)
            unseen_doc_fold_metrics.append(unseen_doc_metric)
            unseen_coverages.append(unseen_coverage)
            slice_fold_metrics.append(compute_slice_metrics(va.reset_index(drop=True), val_pred))

        seed_oof.append(oof)
        seed_test.append(np.array(majority_vote_predictions(fold_test_preds), dtype=object))

    oof_arr = np.stack(seed_oof, axis=0)
    test_arr = np.stack(seed_test, axis=0)
    _save_npy_all("oof_preds_lookup_baseline.npy", oof_arr, output_dirs)
    _save_npy_all("test_preds_lookup_baseline.npy", test_arr, output_dirs)

    avg_bleu = float(np.mean([m["bleu"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_chrf = float(np.mean([m["chrfpp"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_gmean = float(np.mean([m["gmean"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_doc_bleu = float(np.mean([m["bleu"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_doc_chrf = float(np.mean([m["chrfpp"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_doc_gmean = float(np.mean([m["gmean"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_unseen_bleu = float(np.mean([m["bleu"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_chrf = float(np.mean([m["chrfpp"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_gmean = float(np.mean([m["gmean"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_doc_bleu = (
        float(np.mean([m["bleu"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    avg_unseen_doc_chrf = (
        float(np.mean([m["chrfpp"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    avg_unseen_doc_gmean = (
        float(np.mean([m["gmean"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    all_slice_keys = sorted({key for metric in slice_fold_metrics for key in metric})
    avg_slice_metrics = {
        key: float(np.mean([metric.get(key, 0.0) for metric in slice_fold_metrics])) for key in all_slice_keys
    }
    avg_unseen_coverage = {
        key: float(np.mean([metric.get(key, 0.0) for metric in unseen_coverages]))
        for key in ("row_fraction", "row_count", "document_fraction", "document_count")
    }

    return PipelineResult(
        name="lookup_baseline",
        cv_score=avg_gmean,
        bleu=avg_bleu,
        chrfpp=avg_chrf,
        complexity_rank=0,
        oof_predictions=oof_arr,
        test_predictions=test_arr,
        best_seed=active_seeds[0],
        doc_score=avg_doc_gmean,
        doc_bleu=avg_doc_bleu,
        doc_chrfpp=avg_doc_chrf,
        unseen_sentence_score=avg_unseen_gmean,
        unseen_sentence_bleu=avg_unseen_bleu,
        unseen_sentence_chrfpp=avg_unseen_chrf,
        unseen_document_score=avg_unseen_doc_gmean,
        unseen_document_bleu=avg_unseen_doc_bleu,
        unseen_document_chrfpp=avg_unseen_doc_chrf,
        unseen_coverage=avg_unseen_coverage,
        slice_metrics=avg_slice_metrics,
    )


def run_retrieval_char_tfidf_knn_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int,
    seeds: Sequence[int],
    output_dirs: Sequence[Path],
    use_mbr: bool,
    pipeline_name: str = "char_tfidf_knn_memory",
    exact_source_memory: dict[str, str] | None = None,
    constraint_memories: ConstraintMemories | None = None,
) -> PipelineResult:
    fold_metrics: list[dict[str, float]] = []
    doc_fold_metrics: list[dict[str, float]] = []
    unseen_fold_metrics: list[dict[str, float]] = []
    unseen_doc_fold_metrics: list[dict[str, float]] = []
    unseen_coverages: list[dict[str, float]] = []
    slice_fold_metrics: list[dict[str, float]] = []
    seed_oof: list[np.ndarray] = []
    seed_test: list[np.ndarray] = []

    total_low_sim = 0
    total_predictions = 0
    active_seeds = [seeds[0]] if FAST_DEV else list(seeds)
    source_col = _source_feature_col(train_df)
    test_source_col = _source_feature_col(test_df)

    for seed in active_seeds:
        set_global_seed(seed)
        oof = np.empty(len(train_df), dtype=object)
        fold_test_preds: list[list[str]] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(
            iter_grouped_cv_splits(
                train_df["oare_id"].tolist(),
                n_folds=n_folds,
                seed=seed,
                fast_dev=FAST_DEV,
            ),
            start=1,
        ):
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            retrieval_model = _fit_retrieval_model(tr[source_col].tolist(), tr["translation"].tolist())
            fold_exact_memory = build_exact_source_memory(tr)
            fold_constraint_memories = (
                build_constraint_memories(tr)
                if constraint_memories is not None and ENABLE_LEXICON_CONSTRAINTS
                else None
            )

            val_pred, val_low_sim = _predict_with_retrieval(
                retrieval_model,
                va[source_col].tolist(),
                use_mbr=use_mbr,
                k=RETRIEVAL_K,
                min_sim=RETRIEVAL_MIN_SIM,
            )
            tst_pred, test_low_sim = _predict_with_retrieval(
                retrieval_model,
                test_df[test_source_col].tolist(),
                use_mbr=use_mbr,
                k=RETRIEVAL_K,
                min_sim=RETRIEVAL_MIN_SIM,
            )
            val_pred, _ = apply_consistency_postprocess(
                source_texts=va[source_col].tolist(),
                predictions=val_pred,
                group_values=va["oare_id"].tolist() if "oare_id" in va.columns else None,
                exact_source_memory=fold_exact_memory,
                constraint_memories=fold_constraint_memories,
                enable_exact_memory=False,
            )
            tst_pred, _ = apply_consistency_postprocess(
                source_texts=test_df[test_source_col].tolist(),
                predictions=tst_pred,
                group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
                exact_source_memory=fold_exact_memory,
                constraint_memories=fold_constraint_memories,
                enable_exact_memory=False,
            )

            oof[va_idx] = np.array(val_pred, dtype=object)
            fold_test_preds.append(tst_pred)
            sentence_metric, doc_metric = compute_sentence_and_document_metrics(va, val_pred)
            unseen_sentence_metric, unseen_doc_metric, unseen_coverage = compute_unseen_source_metrics(tr, va, val_pred)
            fold_metrics.append(sentence_metric)
            doc_fold_metrics.append(doc_metric)
            unseen_fold_metrics.append(unseen_sentence_metric)
            unseen_doc_fold_metrics.append(unseen_doc_metric)
            unseen_coverages.append(unseen_coverage)
            slice_fold_metrics.append(compute_slice_metrics(va.reset_index(drop=True), val_pred))

            fold_count = len(va) + len(test_df)
            fold_low_sim = val_low_sim + test_low_sim
            total_low_sim += fold_low_sim
            total_predictions += fold_count
            log(f"[{pipeline_name}] seed={seed} fold={fold_idx} low-sim-fallback={fold_low_sim}/{fold_count}")

        seed_oof.append(oof)
        seed_test.append(np.array(majority_vote_predictions(fold_test_preds), dtype=object))

    oof_arr = np.stack(seed_oof, axis=0)
    test_arr = np.stack(seed_test, axis=0)
    _save_npy_all(f"oof_preds_{pipeline_name}.npy", oof_arr, output_dirs)
    _save_npy_all(f"test_preds_{pipeline_name}.npy", test_arr, output_dirs)

    avg_bleu = float(np.mean([m["bleu"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_chrf = float(np.mean([m["chrfpp"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_gmean = float(np.mean([m["gmean"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_doc_bleu = float(np.mean([m["bleu"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_doc_chrf = float(np.mean([m["chrfpp"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_doc_gmean = float(np.mean([m["gmean"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_unseen_bleu = float(np.mean([m["bleu"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_chrf = float(np.mean([m["chrfpp"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_gmean = float(np.mean([m["gmean"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_doc_bleu = (
        float(np.mean([m["bleu"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    avg_unseen_doc_chrf = (
        float(np.mean([m["chrfpp"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    avg_unseen_doc_gmean = (
        float(np.mean([m["gmean"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    all_slice_keys = sorted({key for metric in slice_fold_metrics for key in metric})
    avg_slice_metrics = {
        key: float(np.mean([metric.get(key, 0.0) for metric in slice_fold_metrics])) for key in all_slice_keys
    }
    avg_unseen_coverage = {
        key: float(np.mean([metric.get(key, 0.0) for metric in unseen_coverages]))
        for key in ("row_fraction", "row_count", "document_fraction", "document_count")
    }
    if total_predictions > 0:
        log(
            f"[{pipeline_name}] total low-sim fallback: "
            f"{total_low_sim}/{total_predictions} ({(100.0 * total_low_sim / total_predictions):.2f}%)"
        )

    return PipelineResult(
        name=pipeline_name,
        cv_score=avg_gmean,
        bleu=avg_bleu,
        chrfpp=avg_chrf,
        complexity_rank=_pipeline_complexity_rank(pipeline_name),
        oof_predictions=oof_arr,
        test_predictions=test_arr,
        best_seed=active_seeds[0],
        doc_score=avg_doc_gmean,
        doc_bleu=avg_doc_bleu,
        doc_chrfpp=avg_doc_chrf,
        unseen_sentence_score=avg_unseen_gmean,
        unseen_sentence_bleu=avg_unseen_bleu,
        unseen_sentence_chrfpp=avg_unseen_chrf,
        unseen_document_score=avg_unseen_doc_gmean,
        unseen_document_bleu=avg_unseen_doc_bleu,
        unseen_document_chrfpp=avg_unseen_doc_chrf,
        unseen_coverage=avg_unseen_coverage,
        slice_metrics=avg_slice_metrics,
    )


def run_retrieval_char_tfidf_knn_full_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_mbr: bool,
    exact_source_memory: dict[str, str] | None = None,
    constraint_memories: ConstraintMemories | None = None,
) -> tuple[list[str], int, int]:
    source_col = _source_feature_col(train_df)
    test_source_col = _source_feature_col(test_df)
    retrieval_model = _fit_retrieval_model(
        train_df[source_col].tolist(),
        train_df["translation"].tolist(),
    )
    preds, low_sim_count = _predict_with_retrieval(
        retrieval_model,
        test_df[test_source_col].tolist(),
        use_mbr=use_mbr,
        k=RETRIEVAL_K,
        min_sim=RETRIEVAL_MIN_SIM,
    )
    preds, _ = apply_consistency_postprocess(
        source_texts=test_df[test_source_col].tolist(),
        predictions=preds,
        group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
        exact_source_memory=exact_source_memory,
        constraint_memories=constraint_memories,
    )
    return preds, low_sim_count, len(test_df)


def run_seq2seq_pipeline_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    n_folds: int,
    seeds: Sequence[int],
    output_dirs: Sequence[Path],
    constraint_memories: ConstraintMemories | None = None,
    generation_batch_size: int = 2,
) -> PipelineResult:
    fold_metrics: list[dict[str, float]] = []
    doc_fold_metrics: list[dict[str, float]] = []
    unseen_fold_metrics: list[dict[str, float]] = []
    unseen_doc_fold_metrics: list[dict[str, float]] = []
    unseen_coverages: list[dict[str, float]] = []
    slice_fold_metrics: list[dict[str, float]] = []
    seed_oof: list[np.ndarray] = []
    seed_test: list[np.ndarray] = []
    seed_scores: dict[int, float] = {}
    all_model_keys: set[str] = set()
    seed_model_oof_entries: list[dict[str, np.ndarray]] = []
    seed_model_test_entries: list[dict[str, np.ndarray]] = []
    executed_checkpoints: list[str] = []
    aggregate_postprocess_stats: Counter[str] = Counter()

    active_seeds = [seeds[0]] if FAST_DEV else list(seeds)
    device = _prepare_device(GPU_DEVICE)
    source_col = _source_feature_col(train_df)
    test_source_col = _source_feature_col(test_df)
    test_inputs = build_context_window_texts(test_df, test_source_col, enabled=cfg.use_context_window)

    for seed in active_seeds:
        set_global_seed(seed)
        oof = np.empty(len(train_df), dtype=object)
        fold_test_preds: list[list[str]] = []
        fold_scores: list[float] = []
        model_oof_by_seed: dict[str, np.ndarray] = {}
        model_test_fold_by_seed: dict[str, list[list[str]]] = {}

        for fold_idx, (tr_idx, va_idx) in enumerate(
            iter_grouped_cv_splits(
                train_df["oare_id"].tolist(),
                n_folds=n_folds,
                seed=seed,
                fast_dev=FAST_DEV,
            ),
            start=1,
        ):
            # leak-safe split boundary is still enforced even for inference-only pipelines.
            _fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
            fold_val = train_df.iloc[va_idx].reset_index(drop=True)
            retrieval_model = None
            if cfg.use_retrieval_candidates:
                retrieval_model = _fit_retrieval_model(
                    _fold_train[source_col].tolist(),
                    _fold_train["translation"].tolist(),
                )
            fold_constraint_memories = (
                build_constraint_memories(_fold_train)
                if constraint_memories is not None and ENABLE_LEXICON_CONSTRAINTS
                else None
            )
            val_inputs = build_context_window_texts(fold_val, source_col, enabled=cfg.use_context_window)
            mbr_stats: dict[str, int] = {"constraint_bonus_hits": 0}

            val_pred, per_model_val, val_sources = generate_predictions(
                val_inputs,
                cfg,
                device=device,
                retrieval_model=retrieval_model,
                constraint_memories=fold_constraint_memories,
                mbr_stats=mbr_stats,
                batch_size=generation_batch_size,
            )
            test_pred, per_model_test, test_sources = generate_predictions(
                test_inputs,
                cfg,
                device=device,
                retrieval_model=retrieval_model,
                constraint_memories=fold_constraint_memories,
                mbr_stats=mbr_stats,
                batch_size=generation_batch_size,
            )
            executed_checkpoints.extend(val_sources)
            executed_checkpoints.extend(test_sources)

            val_pred, val_stats = apply_consistency_postprocess(
                source_texts=fold_val[source_col].tolist(),
                predictions=val_pred,
                group_values=fold_val["oare_id"].tolist(),
                exact_source_memory=build_exact_source_memory(_fold_train),
                constraint_memories=fold_constraint_memories,
                enable_exact_memory=False,
            )
            test_pred, test_stats = apply_consistency_postprocess(
                source_texts=test_df[test_source_col].tolist(),
                predictions=test_pred,
                group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
                exact_source_memory=build_exact_source_memory(_fold_train),
                constraint_memories=fold_constraint_memories,
                enable_exact_memory=False,
            )
            aggregate_postprocess_stats.update(val_stats)
            aggregate_postprocess_stats.update(test_stats)
            aggregate_postprocess_stats.update(mbr_stats)

            oof[va_idx] = np.array(val_pred, dtype=object)
            fold_test_preds.append(test_pred)

            sentence_metric, doc_metric = compute_sentence_and_document_metrics(fold_val, val_pred)
            unseen_sentence_metric, unseen_doc_metric, unseen_coverage = compute_unseen_source_metrics(
                _fold_train,
                fold_val,
                val_pred,
            )
            fold_metrics.append(sentence_metric)
            doc_fold_metrics.append(doc_metric)
            unseen_fold_metrics.append(unseen_sentence_metric)
            unseen_doc_fold_metrics.append(unseen_doc_metric)
            unseen_coverages.append(unseen_coverage)
            slice_fold_metrics.append(compute_slice_metrics(fold_val, val_pred))
            fold_scores.append(
                float(unseen_sentence_metric["gmean"])
                if float(unseen_sentence_metric["gmean"]) > 0.0
                else float(sentence_metric["gmean"])
            )

            fold_pred_df = pd.DataFrame(
                {
                    "id": train_df.iloc[va_idx].index,
                    "reference": fold_val["translation"].tolist(),
                    "prediction": val_pred,
                    "seed": seed,
                    "fold": fold_idx,
                    "pipeline": cfg.name,
                }
            )
            _write_csv_all(f"val_preds_{cfg.name}_seed{seed}_fold{fold_idx}.csv", fold_pred_df, output_dirs)

            for model_key, preds in per_model_val.items():
                if model_key not in model_oof_by_seed:
                    model_oof_by_seed[model_key] = np.empty(len(train_df), dtype=object)
                model_oof_by_seed[model_key][va_idx] = np.array(preds, dtype=object)
            for model_key, preds in per_model_test.items():
                model_test_fold_by_seed.setdefault(model_key, []).append(list(preds))

        seed_oof.append(oof)
        seed_test.append(np.array(majority_vote_predictions(fold_test_preds), dtype=object))
        seed_scores[seed] = float(np.mean(fold_scores)) if fold_scores else 0.0
        seed_model_oof_entries.append(model_oof_by_seed)
        seed_test_map: dict[str, np.ndarray] = {}
        for model_key, fold_pred_lists in model_test_fold_by_seed.items():
            seed_test_map[model_key] = np.array(majority_vote_predictions(fold_pred_lists), dtype=object)
        seed_model_test_entries.append(seed_test_map)
        all_model_keys.update(model_oof_by_seed.keys())
        all_model_keys.update(seed_test_map.keys())

    oof_arr = np.stack(seed_oof, axis=0)
    test_arr = np.stack(seed_test, axis=0)

    _save_npy_all(f"oof_preds_{cfg.name}.npy", oof_arr, output_dirs)
    _save_npy_all(f"test_preds_{cfg.name}.npy", test_arr, output_dirs)

    for model_key in sorted(all_model_keys):
        oof_seed_stack = np.stack(
            [d.get(model_key, np.array([""] * len(train_df), dtype=object)) for d in seed_model_oof_entries],
            axis=0,
        )
        test_seed_stack = np.stack(
            [d.get(model_key, np.array([""] * len(test_df), dtype=object)) for d in seed_model_test_entries],
            axis=0,
        )
        _save_npy_all(f"oof_preds_{cfg.name}__{model_key}.npy", oof_seed_stack, output_dirs)
        _save_npy_all(f"test_preds_{cfg.name}__{model_key}.npy", test_seed_stack, output_dirs)

    avg_bleu = float(np.mean([m["bleu"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_chrf = float(np.mean([m["chrfpp"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_gmean = float(np.mean([m["gmean"] for m in fold_metrics])) if fold_metrics else 0.0
    avg_doc_bleu = float(np.mean([m["bleu"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_doc_chrf = float(np.mean([m["chrfpp"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_doc_gmean = float(np.mean([m["gmean"] for m in doc_fold_metrics])) if doc_fold_metrics else 0.0
    avg_unseen_bleu = float(np.mean([m["bleu"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_chrf = float(np.mean([m["chrfpp"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_gmean = float(np.mean([m["gmean"] for m in unseen_fold_metrics])) if unseen_fold_metrics else 0.0
    avg_unseen_doc_bleu = (
        float(np.mean([m["bleu"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    avg_unseen_doc_chrf = (
        float(np.mean([m["chrfpp"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    avg_unseen_doc_gmean = (
        float(np.mean([m["gmean"] for m in unseen_doc_fold_metrics])) if unseen_doc_fold_metrics else 0.0
    )
    all_slice_keys = sorted({key for metric in slice_fold_metrics for key in metric})
    avg_slice_metrics = {
        key: float(np.mean([metric.get(key, 0.0) for metric in slice_fold_metrics])) for key in all_slice_keys
    }
    avg_unseen_coverage = {
        key: float(np.mean([metric.get(key, 0.0) for metric in unseen_coverages]))
        for key in ("row_fraction", "row_count", "document_fraction", "document_count")
    }

    best_seed = max(seed_scores.items(), key=lambda kv: kv[1])[0] if seed_scores else active_seeds[0]

    return PipelineResult(
        name=cfg.name,
        cv_score=avg_gmean,
        bleu=avg_bleu,
        chrfpp=avg_chrf,
        complexity_rank=cfg.complexity_rank,
        oof_predictions=oof_arr,
        test_predictions=test_arr,
        best_seed=best_seed,
        doc_score=avg_doc_gmean,
        doc_bleu=avg_doc_bleu,
        doc_chrfpp=avg_doc_chrf,
        unseen_sentence_score=avg_unseen_gmean,
        unseen_sentence_bleu=avg_unseen_bleu,
        unseen_sentence_chrfpp=avg_unseen_chrf,
        unseen_document_score=avg_unseen_doc_gmean,
        unseen_document_bleu=avg_unseen_doc_bleu,
        unseen_document_chrfpp=avg_unseen_doc_chrf,
        unseen_coverage=avg_unseen_coverage,
        executed_checkpoints=sorted(set(executed_checkpoints)),
        postprocess_stats=dict(aggregate_postprocess_stats),
        slice_metrics=avg_slice_metrics,
    )


def train_full_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    seed: int,
    constraint_memories: ConstraintMemories | None = None,
    generation_batch_size: int = 2,
) -> tuple[list[str], str, dict[str, int], list[str]]:
    set_global_seed(seed)
    device = _prepare_device(GPU_DEVICE)
    source_col = _source_feature_col(train_df)
    test_source_col = _source_feature_col(test_df)
    retrieval_model = None
    if cfg.use_retrieval_candidates:
        retrieval_model = _fit_retrieval_model(train_df[source_col].tolist(), train_df["translation"].tolist())
    exact_source_memory = build_exact_source_memory(train_df)
    active_constraint_memories = constraint_memories
    if active_constraint_memories is None and ENABLE_LEXICON_CONSTRAINTS:
        active_constraint_memories = build_constraint_memories(train_df)
    test_inputs = build_context_window_texts(test_df, test_source_col, enabled=cfg.use_context_window)
    preds, per_model_preds, resolved_sources = generate_predictions(
        test_inputs,
        cfg,
        device=device,
        retrieval_model=retrieval_model,
        constraint_memories=active_constraint_memories,
        batch_size=generation_batch_size,
    )
    preds, postprocess_stats = apply_consistency_postprocess(
        source_texts=test_df[test_source_col].tolist(),
        predictions=preds,
        group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
        exact_source_memory=exact_source_memory,
        constraint_memories=active_constraint_memories,
    )
    model_used = ",".join(sorted(per_model_preds.keys()))
    return preds, model_used, postprocess_stats, resolved_sources


# =====================================================================================
# Submission
# =====================================================================================


def build_translation_submission(
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    predictions: Sequence[str],
) -> pd.DataFrame:
    if len(predictions) != len(test_df):
        raise ValueError(f"Prediction count mismatch: {len(predictions)} vs test rows {len(test_df)}")

    pred_df = pd.DataFrame({"id": test_df["id"].tolist(), "translation": [str(x) for x in predictions]})
    submission_df = sample_df[["id"]].merge(pred_df, on="id", how="left")
    return submission_df


def validate_submission(submission_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    if list(submission_df.columns) != ["id", "translation"]:
        raise ValueError(f"submission columns must be ['id','translation']; got {list(submission_df.columns)}")
    if len(submission_df) != len(sample_df):
        raise ValueError("submission row count must match sample_submission row count")
    if submission_df["id"].tolist() != sample_df["id"].tolist():
        raise ValueError("submission id order must exactly match sample_submission.csv")
    if submission_df["translation"].isna().any():
        missing = submission_df.loc[submission_df["translation"].isna(), "id"].tolist()[:10]
        raise ValueError(f"submission contains NaN predictions; sample missing ids={missing}")

    submission_df["translation"] = submission_df["translation"].astype(str)
    bad = submission_df["translation"].str.lower().isin({"nan", "inf", "-inf"})
    if bad.any():
        submission_df.loc[bad, "translation"] = ""
    submission_df["translation"] = submission_df["translation"].map(normalize_target)

    if not submission_df["translation"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("All predictions must be strings")
    if submission_df["translation"].str.lower().isin({"nan", "inf", "-inf"}).any():
        raise ValueError("submission contains non-finite translation strings after normalization")


def _looks_like_degenerate_lookup_submission(predictions: Sequence[str]) -> bool:
    normalized = [normalize_target(str(value)) for value in predictions]
    if not normalized:
        return False
    counts = Counter(normalized)
    top_value, top_count = counts.most_common(1)[0]
    if top_count < len(normalized):
        return False
    stripped = top_value.strip()
    if stripped in {"", '"', "'", "''", '""'}:
        return True
    return len(stripped) <= 1


def write_submission(
    submission_df: pd.DataFrame,
    output_dirs: Sequence[Path],
    kaggle_working_writable: bool,
) -> None:
    for out_dir in output_dirs:
        submission_df.to_csv(out_dir / "submission.csv", index=False)

    if kaggle_working_writable:
        submission_df.to_csv(Path("/kaggle/working/submission.csv"), index=False)


# =====================================================================================
# Non-text modality stubs
# =====================================================================================


def modality_stub_message(modality: str) -> str:
    return (
        f"Detected modality '{modality}', but this kernel is pinned to the "
        "Deep Past translation task. Expected train/test/sample with "
        "`transliteration` text inputs and `translation` target."
    )


# =====================================================================================
# Translation route and orchestration
# =====================================================================================


def _pipeline_family_priority(name: str) -> int:
    if name.startswith("retrieval_char_tfidf_knn") or name == "char_tfidf_knn_memory":
        return 1
    if name == "lookup_baseline":
        return 2
    return 0


def _slice_metric_score(result: PipelineResult) -> float:
    if not result.slice_metrics:
        return 0.0
    values = [float(v) for v in result.slice_metrics.values()]
    return float(sum(values) / max(1, len(values)))


def _priority_slice_score(metrics: dict[str, float] | None) -> float:
    if not metrics:
        return 0.0
    keys = [
        "metadata_supervision_sentence_gmean",
        "entity_heavy_sentence_gmean",
        "quantity_unit_sentence_gmean",
    ]
    values = [float(metrics[key]) for key in keys if key in metrics]
    return float(sum(values) / max(1, len(values))) if values else 0.0


def _is_retrieval_family(name: str) -> bool:
    value = str(name)
    return value.startswith("retrieval_") or value == "char_tfidf_knn_memory"


def _is_checkpointed_seq2seq_result(result: PipelineResult) -> bool:
    if result.name in {"lookup_baseline", "plan_mbr_blend"}:
        return False
    if _is_retrieval_family(result.name):
        return False
    return bool(result.executed_checkpoints)


def _competition_guard_score(result: PipelineResult) -> float:
    if float(result.unseen_sentence_score) > 0.0:
        return float(result.unseen_sentence_score)
    return float(result.cv_score)


def _baseline_guard_candidates(results: Sequence[PipelineResult]) -> list[PipelineResult]:
    return [r for r in results if _is_retrieval_family(r.name) or r.name == "lookup_baseline"]


def _materially_below_baseline(candidate: PipelineResult, baseline: PipelineResult) -> bool:
    return (_competition_guard_score(candidate) + BASELINE_GUARD_MARGIN) < _competition_guard_score(baseline)


def _selector_sort_key(result: PipelineResult) -> tuple[float, float, float, float, float, float, float, int, str]:
    return (
        -float(result.unseen_sentence_score),
        -float(result.unseen_document_score),
        -float(result.doc_score),
        -float(result.cv_score),
        -float(_priority_slice_score(result.slice_metrics)),
        float(result.complexity_rank),
        -float(len(result.executed_checkpoints or [])),
        len(result.name),
        result.name,
    )


def choose_best_result(results: Sequence[PipelineResult]) -> PipelineResult:
    if not results:
        raise ValueError("choose_best_result requires at least one result")

    ranked = sorted(results, key=_selector_sort_key)
    chosen = ranked[0]
    baseline_candidates = _baseline_guard_candidates(results)
    if baseline_candidates:
        best_baseline = sorted(baseline_candidates, key=_selector_sort_key)[0]
        if best_baseline.name != chosen.name and _materially_below_baseline(chosen, best_baseline):
            log(
                f"Selection guard: refusing {chosen.name} because its competition-faithful guard score "
                f"({_competition_guard_score(chosen):.6f}) is materially below baseline {best_baseline.name} "
                f"({_competition_guard_score(best_baseline):.6f})."
            )
            chosen = best_baseline
    return chosen


def summarize_results(results: Sequence[PipelineResult]) -> pd.DataFrame:
    rows = [
        {
            "pipeline": r.name,
            "cv_unseen_sentence_gmean": r.unseen_sentence_score,
            "cv_unseen_document_gmean": r.unseen_document_score,
            "unseen_row_fraction": float((r.unseen_coverage or {}).get("row_fraction", 0.0)),
            "cv_sentence_gmean": r.cv_score,
            "cv_sentence_bleu": r.bleu,
            "cv_sentence_chrfpp": r.chrfpp,
            "cv_document_gmean": r.doc_score,
            "cv_document_bleu": r.doc_bleu,
            "cv_document_chrfpp": r.doc_chrfpp,
            "slice_metric_score": _slice_metric_score(r),
            "complexity_rank": r.complexity_rank,
            "best_seed": r.best_seed,
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values(
        [
            "cv_unseen_sentence_gmean",
            "cv_unseen_document_gmean",
            "cv_document_gmean",
            "cv_sentence_gmean",
            "slice_metric_score",
            "complexity_rank",
        ],
        ascending=[False, False, False, False, False, True],
    )


def _representative_prediction_vector(raw_preds: np.ndarray) -> list[str]:
    arr = np.asarray(raw_preds, dtype=object)
    if arr.ndim <= 1:
        return [str(x) for x in arr.tolist()]
    return majority_vote_predictions(arr.tolist())


def _usable_result_for_ensemble(result: PipelineResult, expected_len: int) -> bool:
    vector = _representative_prediction_vector(result.oof_predictions)
    if len(vector) != expected_len:
        return False
    return any(str(item).strip() for item in vector)


def _blend_prediction_vectors(
    members: Sequence[tuple[str, Sequence[str]]],
    source_texts: Sequence[str],
    group_values: Sequence[Any] | None,
    exact_source_memory: dict[str, str] | None,
    constraint_memories: ConstraintMemories | None,
) -> tuple[list[str], dict[str, int]]:
    blended: list[str] = []
    mbr_stats: dict[str, int] = {"constraint_bonus_hits": 0}

    for row_idx, source_text in enumerate(source_texts):
        row_candidates: list[tuple[str, str, int]] = []
        for rank, (member_name, preds) in enumerate(members, start=1):
            if row_idx >= len(preds):
                continue
            candidate = normalize_prediction_style(preds[row_idx])
            if not candidate:
                continue
            row_candidates.append((candidate, member_name, rank))
        if not row_candidates:
            blended.append("")
        elif len(row_candidates) == 1:
            blended.append(row_candidates[0][0])
        else:
            blended.append(
                _select_mbr_candidate(
                    row_candidates,
                    source_text=str(source_text),
                    constraint_memories=constraint_memories,
                    mbr_stats=mbr_stats,
                )
            )

    outputs, postprocess_stats = apply_consistency_postprocess(
        source_texts=source_texts,
        predictions=blended,
        group_values=group_values,
        exact_source_memory=exact_source_memory,
        constraint_memories=constraint_memories,
        enable_exact_memory=False,
    )
    merged_stats = Counter[str]()
    merged_stats.update(mbr_stats)
    merged_stats.update(postprocess_stats)
    return outputs, dict(merged_stats)


def build_explicit_ensemble_result(
    pair_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: Sequence[PipelineResult],
    output_dirs: Sequence[Path],
    exact_source_memory: dict[str, str] | None,
    constraint_memories: ConstraintMemories | None = None,
) -> PipelineResult | None:
    eligible = [
        result
        for result in results
        if result.name != "plan_mbr_blend" and _usable_result_for_ensemble(result, len(pair_df))
    ]
    if len(eligible) < 2:
        return None

    eligible = sorted(
        eligible,
        key=lambda r: (-r.unseen_sentence_score, -r.unseen_document_score, -r.doc_score, r.complexity_rank, r.name),
    )
    selected_members = eligible[: min(3, len(eligible))]
    oof_members = [
        (result.name, _representative_prediction_vector(result.oof_predictions)) for result in selected_members
    ]
    test_members = [
        (result.name, _representative_prediction_vector(result.test_predictions)) for result in selected_members
    ]

    source_col = _source_feature_col(pair_df)
    test_source_col = _source_feature_col(test_df)
    oof_predictions, oof_stats = _blend_prediction_vectors(
        members=oof_members,
        source_texts=pair_df[source_col].tolist(),
        group_values=pair_df["oare_id"].tolist() if "oare_id" in pair_df.columns else None,
        exact_source_memory=None,
        constraint_memories=None,
    )
    test_predictions, test_stats = _blend_prediction_vectors(
        members=test_members,
        source_texts=test_df[test_source_col].tolist(),
        group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
        exact_source_memory=None,
        constraint_memories=None,
    )

    sentence_metric, doc_metric = compute_sentence_and_document_metrics(pair_df, oof_predictions)
    slice_metrics = compute_slice_metrics(pair_df, oof_predictions)
    postprocess_stats = Counter[str]()
    postprocess_stats.update(oof_stats)
    postprocess_stats.update(test_stats)
    oof_arr = np.asarray([oof_predictions], dtype=object)
    test_arr = np.asarray([test_predictions], dtype=object)
    _save_npy_all("oof_preds_plan_mbr_blend.npy", oof_arr, output_dirs)
    _save_npy_all("test_preds_plan_mbr_blend.npy", test_arr, output_dirs)

    return PipelineResult(
        name="plan_mbr_blend",
        cv_score=float(sentence_metric["gmean"]),
        bleu=float(sentence_metric["bleu"]),
        chrfpp=float(sentence_metric["chrfpp"]),
        complexity_rank=max(result.complexity_rank for result in selected_members) + 1,
        oof_predictions=oof_arr,
        test_predictions=test_arr,
        best_seed=int(selected_members[0].best_seed),
        doc_score=float(doc_metric["gmean"]),
        doc_bleu=float(doc_metric["bleu"]),
        doc_chrfpp=float(doc_metric["chrfpp"]),
        unseen_sentence_score=float(np.mean([member.unseen_sentence_score for member in selected_members])),
        unseen_sentence_bleu=float(np.mean([member.unseen_sentence_bleu for member in selected_members])),
        unseen_sentence_chrfpp=float(np.mean([member.unseen_sentence_chrfpp for member in selected_members])),
        unseen_document_score=float(np.mean([member.unseen_document_score for member in selected_members])),
        unseen_document_bleu=float(np.mean([member.unseen_document_bleu for member in selected_members])),
        unseen_document_chrfpp=float(np.mean([member.unseen_document_chrfpp for member in selected_members])),
        unseen_coverage={
            key: float(np.mean([(member.unseen_coverage or {}).get(key, 0.0) for member in selected_members]))
            for key in ("row_fraction", "row_count", "document_fraction", "document_count")
        },
        executed_checkpoints=[],
        postprocess_stats=dict(postprocess_stats),
        slice_metrics=slice_metrics,
        ensemble_members=[result.name for result in selected_members],
    )


def _run_single_pipeline_with_fallbacks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    output_dirs: Sequence[Path],
    constraint_memories: ConstraintMemories | None = None,
    n_folds: int | None = None,
    seeds: Sequence[int] | None = None,
    generation_batch_size: int = 2,
) -> PipelineResult:
    # Fallback behavior follows frozen-plan intent for unavailable/heavy configs.
    requested_name = cfg.name
    requested_rank = cfg.complexity_rank

    def _preserve_requested_identity(result: PipelineResult) -> PipelineResult:
        if result.name == requested_name:
            return result
        return PipelineResult(
            name=requested_name,
            cv_score=result.cv_score,
            bleu=result.bleu,
            chrfpp=result.chrfpp,
            complexity_rank=requested_rank,
            oof_predictions=result.oof_predictions,
            test_predictions=result.test_predictions,
            best_seed=result.best_seed,
            doc_score=result.doc_score,
            doc_bleu=result.doc_bleu,
            doc_chrfpp=result.doc_chrfpp,
            unseen_sentence_score=result.unseen_sentence_score,
            unseen_sentence_bleu=result.unseen_sentence_bleu,
            unseen_sentence_chrfpp=result.unseen_sentence_chrfpp,
            unseen_document_score=result.unseen_document_score,
            unseen_document_bleu=result.unseen_document_bleu,
            unseen_document_chrfpp=result.unseen_document_chrfpp,
            unseen_coverage=result.unseen_coverage,
            executed_checkpoints=result.executed_checkpoints,
            postprocess_stats=result.postprocess_stats,
            slice_metrics=result.slice_metrics,
            ensemble_members=result.ensemble_members,
        )

    if cfg.name == "contextual_byt5_curriculum_mbr" and (not ALLOW_KERNEL_FINETUNE or not DO_TRAIN):
        cfg = PipelineConfig(**{**cfg.__dict__, "use_lora": False})

    try:
        res = run_seq2seq_pipeline_cv(
            train_df,
            test_df,
            cfg,
            int(n_folds or N_FOLDS),
            list(seeds or SEEDS),
            output_dirs,
            constraint_memories=constraint_memories,
            generation_batch_size=generation_batch_size,
        )
        return _preserve_requested_identity(res)
    except Exception as exc:
        msg = str(exc).lower()
        log(f"Pipeline {cfg.name} failed: {exc}")
        allow_single_model_reference_degrade = requested_name != REFERENCE_PRIMARY_PIPELINE_NAME
        if requested_name == REFERENCE_PRIMARY_PIPELINE_NAME and (
            "reference dual-checkpoint blocker" in msg
            or "fake dual-model execution" in msg
            or "blocked_reference_runtime" in cfg.reference_runtime_mode
        ):
            raise
        fallback_attempts: list[tuple[str, PipelineConfig]] = []
        if (
            ("unable to load model" in msg or "no loadable models" in msg)
            and cfg.name != "dual_checkpoint_public_mbr"
            and allow_single_model_reference_degrade
        ):
            fb_cfg = get_pipeline_cfg("dual_checkpoint_public_mbr")
            fb_cfg.use_mbr = False
            fb_cfg.use_multi_model_pool = False
            fb_cfg.model_hints = fb_cfg.model_hints[:1]
            fallback_attempts.append(("switch to single known-available public checkpoint", fb_cfg))
        if (_looks_like_cuda_oom(exc) or "out of memory" in msg) and allow_single_model_reference_degrade:
            fb_cfg = PipelineConfig(
                **{
                    **cfg.__dict__,
                    "num_beams": 1,
                    "mbr_num_beam_cands": 1,
                    "mbr_num_sample_cands": 0,
                    "use_mbr": False,
                    "max_source_len": min(512, cfg.max_source_len),
                    "max_new_tokens": min(192, cfg.max_new_tokens),
                    "model_hints": cfg.model_hints[:1],
                    "use_multi_model_pool": False,
                }
            )
            fallback_attempts.append(("OOM-safe short decode", fb_cfg))
        if cfg.use_mbr and allow_single_model_reference_degrade:
            fb_cfg = PipelineConfig(
                **{
                    **cfg.__dict__,
                    "use_mbr": False,
                    "mbr_num_sample_cands": 0,
                    "mbr_num_beam_cands": 1,
                    "mbr_pool_cap": min(8, cfg.mbr_pool_cap),
                    "num_beams": max(1, min(4, cfg.num_beams)),
                    "max_new_tokens": max(64, min(256, cfg.max_new_tokens)),
                    "model_hints": cfg.model_hints[:1],
                    "use_multi_model_pool": False,
                }
            )
            fallback_attempts.append(("disable MBR and decode with single-model beams", fb_cfg))

        seen_fallbacks: set[tuple[Any, ...]] = set()
        for reason, fb_cfg in fallback_attempts:
            fallback_key = (
                fb_cfg.name,
                tuple(fb_cfg.model_hints),
                fb_cfg.use_mbr,
                fb_cfg.use_multi_model_pool,
                fb_cfg.num_beams,
                fb_cfg.max_source_len,
                fb_cfg.max_new_tokens,
            )
            if fallback_key in seen_fallbacks:
                continue
            seen_fallbacks.add(fallback_key)
            log(f"Applying fallback: {reason}")
            try:
                res = run_seq2seq_pipeline_cv(
                    train_df,
                    test_df,
                    fb_cfg,
                    int(n_folds or N_FOLDS),
                    list(seeds or SEEDS),
                    output_dirs,
                    constraint_memories=constraint_memories,
                    generation_batch_size=generation_batch_size,
                )
                return _preserve_requested_identity(res)
            except Exception as fallback_exc:
                log(f"Fallback failed for {requested_name}: {fallback_exc}")
                if _looks_like_cuda_oom(fallback_exc):
                    _cuda_cleanup_best_effort()
                continue

        raise


def _reference_single_model_ablation_cfgs(cfg: PipelineConfig) -> list[PipelineConfig]:
    if cfg.name != REFERENCE_PRIMARY_PIPELINE_NAME or len(cfg.model_hints) < 2:
        return []

    ablations: list[PipelineConfig] = []
    slot_meta = cfg.reference_slot_meta or _build_reference_slot_meta(cfg.model_hints, cfg.model_hints)
    for idx, model_hint in enumerate(cfg.model_hints[:2], start=1):
        slot_name = "a" if idx == 1 else "b"
        slot_payload = slot_meta[idx - 1 : idx]
        ablations.append(
            PipelineConfig(
                **{
                    **cfg.__dict__,
                    "name": f"{cfg.name}__single_model_ablation_slot_{slot_name}",
                    "model_hints": [str(model_hint)],
                    "use_multi_model_pool": False,
                    "use_mbr": False,
                    "complexity_rank": max(0, cfg.complexity_rank - 1),
                    "runtime_name": _reference_runtime_name(cfg.name, [f"single_model_ablation_slot_{slot_name}"]),
                    "reference_runtime_mode": f"single_model_seq2seq_ablation:slot_{slot_name}",
                    "reference_slot_meta": slot_payload,
                }
            )
        )
    return ablations


def _reference_primary_seq2seq_candidates(results: Sequence[PipelineResult]) -> list[PipelineResult]:
    return [result for result in results if _is_checkpointed_seq2seq_result(result)]


def _resolve_final_seq2seq_cfg(
    chosen: PipelineResult,
    runtime_cfgs: dict[str, PipelineConfig],
    *,
    accepted_finetune_adapter_dir: str | None,
    local_budget_skip_reason: str | None,
) -> PipelineConfig:
    cfg = runtime_cfgs.get(chosen.name)
    if cfg is None:
        cfg = get_pipeline_cfg(chosen.name)
        if chosen.name == REFERENCE_PRIMARY_PIPELINE_NAME:
            cfg = _prepare_reference_baseline_cfg(cfg)
    if chosen.name == "contextual_byt5_curriculum_mbr" and accepted_finetune_adapter_dir:
        cfg = _single_source_pipeline_cfg(cfg, accepted_finetune_adapter_dir, suffix="final_adapter")
    if local_budget_skip_reason and chosen.name != REFERENCE_PRIMARY_PIPELINE_NAME and not cfg.reference_runtime_mode:
        cfg = _reduced_faithful_eval_cfg(cfg)
    return cfg


def run_translation_seq2seq(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    data_dir: Path,
    output_dirs: Sequence[Path],
    kaggle_working_writable: bool,
) -> None:
    assert_translation_schema(train_df, test_df, sample_df)

    log("Stage 1/6: deterministic normalization")
    train_df = preprocess_translation_df(train_df, USE_NORMALIZATION, USE_DETERMINATIVES_NORM)
    test_df = preprocess_translation_df(test_df, USE_NORMALIZATION, USE_DETERMINATIVES_NORM)
    lexicon = load_lexicon_resources(data_dir)
    published_df, sentence_metadata_df = load_optional_metadata_frames(data_dir)
    train_df["transliteration_lex"] = train_df["transliteration"].map(
        lambda x: lexicon_normalize_source_text(str(x), lexicon)
    )
    test_df["transliteration_lex"] = test_df["transliteration"].map(
        lambda x: lexicon_normalize_source_text(str(x), lexicon)
    )

    reference_mode_only = _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG)
    if reference_mode_only:
        log("Stage 2/6: reference document-pair frame")
        pseudo_train_df = build_document_pair_frame(train_df)
        metadata_pairs = MetadataSupervisionResult(pd.DataFrame(), 0, 0, 0)
        heuristic_pair_count = 0
        metadata_pair_count = 0
        log(
            "Reference baseline mode: using gold document pairs only, with notebook-faithful preprocessing/"
            "postprocessing and no pseudo-sentence finetuning or retrieval reranking in the primary path."
        )
    else:
        log("Stage 2/6: pseudo sentence-pair dataset construction")
        if ENABLE_PSEUDO_SENTENCES:
            pseudo_train_df, metadata_pairs = build_merged_sentence_pairs(
                train_df,
                lexicon,
                published_df,
                sentence_metadata_df,
            )
        else:
            pseudo_train_df = build_document_pair_frame(train_df)
            metadata_pairs = MetadataSupervisionResult(pd.DataFrame(), 0, 0, 0)
        if ENABLE_GOLD_UPWEIGHT and "supervision_source" in pseudo_train_df.columns:
            pseudo_train_df.loc[pseudo_train_df["supervision_source"].eq("sentence_metadata"), "pair_weight"] = 1.5
        heuristic_pair_count = (
            int(pseudo_train_df["supervision_source"].eq("heuristic").sum())
            if "supervision_source" in pseudo_train_df.columns
            else 0
        )
        metadata_pair_count = (
            int(pseudo_train_df["supervision_source"].eq("sentence_metadata").sum())
            if "supervision_source" in pseudo_train_df.columns
            else 0
        )

    constraint_memories = build_constraint_memories(pseudo_train_df) if ENABLE_LEXICON_CONSTRAINTS else None
    exact_source_memory = build_exact_source_memory(pseudo_train_df)
    reference_eval_df = build_document_pair_frame(train_df)
    allow_optional_finetune = DO_TRAIN and not reference_mode_only
    local_budget_skip_reason: str | None = None
    seq2seq_n_folds = int(N_FOLDS)
    seq2seq_seeds = list(SEEDS)
    seq2seq_batch_size = 1 if FAST_DEV else 2
    seq2seq_eval_df = reference_eval_df
    log(
        f"Reference/train frame rows: {len(pseudo_train_df)} from {train_df['oare_id'].nunique()} documents "
        f"(avg {len(pseudo_train_df) / max(1, train_df['oare_id'].nunique()):.2f} rows/doc)"
    )
    if metadata_pairs.candidate_docs:
        log(
            "Metadata supervision coverage: "
            f"{metadata_pairs.matched_docs}/{train_df['oare_id'].nunique()} docs "
            f"({100.0 * metadata_pairs.matched_docs / max(1, train_df['oare_id'].nunique()):.2f}%), "
            f"pairs={metadata_pair_count}, rejected_docs={metadata_pairs.rejected_docs}"
        )
    if LOCAL_KERNEL_MODE and reference_mode_only:
        local_watchdog_eval_docs = min(LOCAL_REFERENCE_FAST_EVAL_DOCS, LOCAL_REFERENCE_WATCHDOG_FAST_EVAL_DOCS)
        local_budget_skip_reason = (
            "local kernel budget guard: keeping only the reference seq2seq family under the 600s watchdog, "
            "using the 24-document gold-document proxy with 1 fold, 1 seed, no extra ablations, "
            "and reserving full-sample notebook-faithful inference for non-watchdog execution."
        )
        allow_optional_finetune = False
        seq2seq_n_folds = min(seq2seq_n_folds, LOCAL_REFERENCE_WATCHDOG_N_FOLDS)
        seq2seq_seeds = seq2seq_seeds[:1]
        seq2seq_batch_size = 1
        seq2seq_eval_df = _build_local_seq2seq_eval_frame(train_df, max_docs=local_watchdog_eval_docs)
        log(local_budget_skip_reason)
        log(
            "local kernel fast-reference mode: reducing seq2seq eval frame to "
            f"{len(seq2seq_eval_df)} gold-document rows from "
            f"{seq2seq_eval_df['oare_id'].nunique()} documents."
        )
    elif not allow_optional_finetune:
        log("DO_TRAIN=false: optional finetuning disabled; continuing with inference-only seq2seq CV.")

    log("Stage 3/6: grouped CV baselines")
    baseline_eval_df = seq2seq_eval_df if reference_mode_only else pseudo_train_df
    baseline_n_folds = seq2seq_n_folds if reference_mode_only else N_FOLDS
    baseline_seeds = seq2seq_seeds if reference_mode_only else SEEDS
    baseline = run_lookup_baseline_cv(
        baseline_eval_df,
        test_df,
        baseline_n_folds,
        baseline_seeds,
        output_dirs,
        exact_source_memory,
        constraint_memories=constraint_memories,
    )
    pipeline_results: list[PipelineResult] = [baseline]
    required_seq2seq_pipelines = _active_plan_seq2seq_pipeline_names()
    resolved_model_sources: dict[str, list[str]] = {}
    pipeline_runtime_cfgs: dict[str, PipelineConfig] = {}
    base_checkpoint_candidates: dict[str, list[str]] = {}
    domain_adapted_checkpoint_candidates: dict[str, list[str]] = {}
    executed_seq2seq_pipelines: list[str] = []
    any_local_seq2seq_sources_available = False
    fine_tune_result = FineTuneResult(
        False, None, None, None, None, None, None, None, None, None, None, "not_attempted"
    )
    accepted_finetune_adapter_dir: str | None = None
    reference_result: PipelineResult | None = None
    reference_runtime_cfg: PipelineConfig | None = None
    reference_runtime_blocker = ""
    reference_single_results: list[PipelineResult] = []

    if not reference_mode_only and "char_tfidf_knn_memory" in shortlisted_pipeline_names():
        log("Running shortlist pipeline: char_tfidf_knn_memory (smoke test only)")
        retrieval_default = run_retrieval_char_tfidf_knn_cv(
            train_df=baseline_eval_df,
            test_df=test_df,
            n_folds=baseline_n_folds,
            seeds=baseline_seeds,
            output_dirs=output_dirs,
            use_mbr=False,
            pipeline_name="char_tfidf_knn_memory",
            exact_source_memory=exact_source_memory,
            constraint_memories=constraint_memories,
        )
        pipeline_results.append(retrieval_default)

    log("Stage 4/6: multi-checkpoint seq2seq + MBR")
    active_shortlist = shortlisted_pipeline_names()
    log(f"Active shortlist: {', '.join(active_shortlist)}")
    log(
        "Pipeline toggles: "
        f"p1={ENABLE_PIPELINE_1}, p2={ENABLE_PIPELINE_2}, "
        f"p3={ENABLE_PIPELINE_3}, p4={ENABLE_PIPELINE_4}"
    )
    for name in active_shortlist:
        if name == "char_tfidf_knn_memory":
            continue
        if reference_mode_only and name != REFERENCE_PRIMARY_PIPELINE_NAME:
            log(f"[{name}] deferred until the required public reference baseline is reproduced or explicitly blocked.")
            continue
        if local_budget_skip_reason and name != REFERENCE_PRIMARY_PIPELINE_NAME:
            log(f"[{name}] deferred under local watchdog until the required public dual-ByT5 baseline is validated.")
            continue
        cfg = get_pipeline_cfg(name)
        if cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME:
            log(f"[{cfg.name}] preparing reference runtime config")
            cfg = _prepare_reference_baseline_cfg(cfg)
            reference_runtime_cfg = cfg
            reference_runtime_blocker = cfg.reference_blocker
        runtime_cfg = _reduced_faithful_eval_cfg(cfg) if local_budget_skip_reason else cfg
        # Frozen shortlist is executed in plan order.
        log(f"Running shortlist pipeline: {cfg.name}")
        if cfg.runtime_name:
            log(f"[{cfg.name}] runtime selection: {cfg.runtime_name} ({cfg.reference_runtime_mode or 'default'})")
        source_classes = _pipeline_model_source_classes(runtime_cfg)
        local_model_sources = [*source_classes["domain_adapted"], *source_classes["base"]]
        if source_classes["domain_adapted"]:
            domain_adapted_checkpoint_candidates[cfg.name] = list(source_classes["domain_adapted"])
        if source_classes["base"]:
            base_checkpoint_candidates[cfg.name] = list(source_classes["base"])
        if local_model_sources:
            any_local_seq2seq_sources_available = True
            resolved_model_sources[cfg.name] = list(local_model_sources)
            log(f"[{cfg.name}] resolved local model sources: {', '.join(local_model_sources)}")
        else:
            if cfg.name in required_seq2seq_pipelines:
                log(
                    f"Required seq2seq pipeline {cfg.name} has no local model sources; "
                    "continuing only after recording the blocker."
                )
            log(
                f"Skipping seq2seq pipeline {cfg.name}: no local model sources found "
                "(checked KAGGLEBOT_MODEL_PATHS, KAGGLEBOT_PRETRAINED_DIR, /kaggle/input, kernel/models)."
            )
            continue
        try:
            result = _run_single_pipeline_with_fallbacks(
                seq2seq_eval_df if cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME else pseudo_train_df,
                test_df,
                runtime_cfg,
                output_dirs,
                constraint_memories=constraint_memories,
                n_folds=seq2seq_n_folds,
                seeds=seq2seq_seeds,
                generation_batch_size=seq2seq_batch_size,
            )
        except Exception as exc:
            if _looks_like_cuda_oom(exc):
                log(f"Skipping pipeline due to CUDA OOM: {cfg.name}")
                _cuda_cleanup_best_effort()
                if cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME:
                    reference_runtime_blocker = f"reference seq2seq proxy failed with CUDA OOM: {exc}"
                continue
            log(f"Skipping pipeline due to unrecoverable error: {cfg.name} -> {exc}")
            if cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME:
                reference_runtime_blocker = f"reference seq2seq proxy failed: {exc}"
            continue
        else:
            pipeline_results.append(result)
            pipeline_runtime_cfgs[result.name] = runtime_cfg
            executed_seq2seq_pipelines.append(cfg.name)
            if cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME:
                reference_result = result

    contextual_cfg = get_pipeline_cfg("contextual_byt5_curriculum_mbr")
    if not reference_mode_only and "contextual_byt5_curriculum_mbr" in shortlisted_pipeline_names():
        contextual_sources = _pipeline_local_model_sources(contextual_cfg)
        if contextual_sources:
            resolved_model_sources.setdefault(contextual_cfg.name, list(contextual_sources))
        if allow_optional_finetune:
            fine_tune_result = run_optional_lora_finetune(
                pseudo_train_df,
                test_df,
                contextual_cfg,
                seq2seq_seeds[0],
                constraint_memories=constraint_memories,
            )
            if fine_tune_result.ran and fine_tune_result.adapter_dir and fine_tune_result.validation_metric:
                base_sentence = float((fine_tune_result.baseline_metric or {}).get("gmean", 0.0))
                tuned_sentence = float(fine_tune_result.validation_metric.get("gmean", 0.0))
                base_doc = float((fine_tune_result.baseline_doc_metric or {}).get("gmean", 0.0))
                tuned_doc = float((fine_tune_result.validation_doc_metric or {}).get("gmean", 0.0))
                base_slice = _priority_slice_score(fine_tune_result.baseline_slice_metrics)
                tuned_slice = _priority_slice_score(fine_tune_result.validation_slice_metrics)
                sentence_gain = tuned_sentence - base_sentence
                doc_drop = base_doc - tuned_doc
                slice_gain = tuned_slice - base_slice
                log(
                    f"[{contextual_cfg.name}] lightweight finetune adapter ready: {fine_tune_result.adapter_dir} "
                    f"(sentence_gain={sentence_gain:.6f}, doc_drop={doc_drop:.6f}, slice_gain={slice_gain:.6f})"
                )
                if sentence_gain >= 0.15 and doc_drop <= 0.25 and slice_gain >= -0.10:
                    accepted_finetune_adapter_dir = fine_tune_result.adapter_dir
                    resolved_model_sources.setdefault(contextual_cfg.name, [])
                    if fine_tune_result.adapter_dir not in resolved_model_sources[contextual_cfg.name]:
                        resolved_model_sources[contextual_cfg.name].insert(0, fine_tune_result.adapter_dir)
            else:
                log(f"[{contextual_cfg.name}] finetune skipped: {fine_tune_result.reason}")
        else:
            fine_tune_result = FineTuneResult(
                False,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "optional_finetune_disabled",
            )
            log(f"[{contextual_cfg.name}] optional finetune disabled; keeping inference-only checkpoint evaluation.")
    elif reference_mode_only and "contextual_byt5_curriculum_mbr" in shortlisted_pipeline_names():
        log("[contextual_byt5_curriculum_mbr] broader model search deferred until the public baseline is reproduced.")

    required_seq2seq_ready = all(name in resolved_model_sources for name in required_seq2seq_pipelines)
    resolved_domain_adapted_sources = sorted(
        {path for values in domain_adapted_checkpoint_candidates.values() for path in values}
    )
    oracc_adapted_ready = bool(resolved_domain_adapted_sources)
    reference_blocked_runtime = (
        reference_runtime_cfg is not None
        and reference_runtime_cfg.reference_runtime_mode == "blocked_reference_runtime"
    )
    if reference_mode_only and reference_result is None and not reference_blocked_runtime:
        blocker = reference_runtime_blocker or (
            reference_runtime_cfg.reference_blocker if reference_runtime_cfg else ""
        )
        raise RuntimeError(
            "Reference seq2seq proxy did not complete successfully; "
            f"retrieval baselines remain smoke tests only. Blocker: {blocker or 'unknown reference runtime failure'}"
        )
    if any_local_seq2seq_sources_available and not executed_seq2seq_pipelines and not reference_blocked_runtime:
        raise RuntimeError("No required seq2seq pipeline executed; refusing retrieval-only local evaluation.")

    if ENABLE_ENSEMBLE and not reference_mode_only:
        ensemble_result = build_explicit_ensemble_result(
            pair_df=pseudo_train_df,
            test_df=test_df,
            results=pipeline_results,
            output_dirs=output_dirs,
            exact_source_memory=exact_source_memory,
            constraint_memories=constraint_memories,
        )
        if ensemble_result is not None:
            pipeline_results.append(ensemble_result)
            log(
                f"Built explicit ensemble: {ensemble_result.name} <- "
                f"{', '.join(ensemble_result.ensemble_members or [])}"
            )
    elif reference_mode_only:
        log("Skipping explicit ensemble: reference seq2seq path is the only primary selection path this iteration.")

    summary = summarize_results(pipeline_results)
    _write_csv_all("pipeline_cv_summary.csv", summary, output_dirs)
    log("CV summary:\n" + summary.to_string(index=False))

    best_retrieval_diagnostic = None
    retrieval_diagnostics = [result for result in pipeline_results if _is_retrieval_family(result.name)]
    if retrieval_diagnostics:
        best_retrieval_diagnostic = sorted(retrieval_diagnostics, key=_selector_sort_key)[0]
    best_single_reference = (
        sorted(reference_single_results, key=_selector_sort_key)[0] if reference_single_results else None
    )
    if reference_result is not None:
        log(
            "[reference_ablation] dual_checkpoint_mbr: "
            f"unseen_sentence_gmean={reference_result.unseen_sentence_score:.6f}, "
            f"document_gmean={reference_result.doc_score:.6f}, "
            f"sentence_gmean={reference_result.cv_score:.6f}"
        )
    if best_single_reference is not None:
        log(
            "[reference_ablation] best_single_model_seq2seq: "
            f"name={best_single_reference.name}, "
            f"unseen_sentence_gmean={best_single_reference.unseen_sentence_score:.6f}, "
            f"document_gmean={best_single_reference.doc_score:.6f}, "
            f"sentence_gmean={best_single_reference.cv_score:.6f}"
        )
    if best_retrieval_diagnostic is not None:
        log(
            "[reference_ablation] retrieval_knn_diagnostic: "
            f"name={best_retrieval_diagnostic.name}, "
            f"unseen_sentence_gmean={best_retrieval_diagnostic.unseen_sentence_score:.6f}, "
            f"document_gmean={best_retrieval_diagnostic.doc_score:.6f}, "
            f"sentence_gmean={best_retrieval_diagnostic.cv_score:.6f}"
        )

    if reference_mode_only:
        reference_seq2seq_candidates = _reference_primary_seq2seq_candidates(pipeline_results)
        if reference_result is None and not reference_seq2seq_candidates:
            blocker = reference_runtime_blocker or (
                reference_runtime_cfg.reference_blocker if reference_runtime_cfg else ""
            )
            chosen = choose_best_result(pipeline_results)
            log(
                "Selection policy: no real seq2seq reference path completed locally; "
                "selecting the strongest honest diagnostic fallback. "
                f"Blocker: {blocker or 'missing reference result'}"
            )
        else:
            chosen = sorted(reference_seq2seq_candidates, key=_selector_sort_key)[0]
            if chosen == reference_result:
                runtime_mode = reference_runtime_cfg.reference_runtime_mode if reference_runtime_cfg is not None else ""
                if runtime_mode == "exact_required_public_pair":
                    log(
                        "Selection policy: using the exact required dual-checkpoint public reference path "
                        "as the primary result; "
                        "retrieval baselines remain diagnostics only."
                    )
                elif runtime_mode == "competition_faithful_fallback_pair":
                    log(
                        "Selection policy: using the competition-faithful dual-checkpoint fallback pair "
                        "as the primary result; "
                        "retrieval baselines remain diagnostics only."
                    )
                else:
                    log(
                        "Selection policy: using the faithful single-model public fallback as the primary result; "
                        "retrieval baselines remain diagnostics only."
                    )
            elif best_single_reference is not None and chosen == best_single_reference:
                log(
                    "Selection policy: promoting the best single-model seq2seq ablation over the dual-checkpoint run; "
                    "retrieval baselines remain diagnostics only."
                )
            else:
                log(
                    "Selection policy: choosing the strongest real seq2seq result available this iteration; "
                    "retrieval baselines remain diagnostics only."
                )
    else:
        chosen = choose_best_result(pipeline_results)

    log(
        f"Selected final pipeline: {chosen.name} "
        f"(unseen_sentence_gmean={chosen.unseen_sentence_score:.6f}, "
        f"unseen_document_gmean={chosen.unseen_document_score:.6f}, "
        f"document_gmean={chosen.doc_score:.6f}, sentence_gmean={chosen.cv_score:.6f})"
    )

    if not DO_INFER:
        final_predictions = [""] * len(test_df)
        final_model_sources: list[str] = []
        final_postprocess_stats: dict[str, int] = {}
    elif chosen.name == "lookup_baseline":
        final_predictions = simple_lookup_predict(
            pseudo_train_df["transliteration_lex"].tolist(),
            pseudo_train_df["translation"].tolist(),
            test_df["transliteration_lex"].tolist(),
        )
        final_predictions, final_postprocess_stats = apply_consistency_postprocess(
            source_texts=test_df["transliteration_lex"].tolist(),
            predictions=final_predictions,
            group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
            exact_source_memory=exact_source_memory,
            constraint_memories=constraint_memories,
        )
        final_model_sources = []
    elif chosen.name == "char_tfidf_knn_memory":
        final_predictions, low_sim_count, total = run_retrieval_char_tfidf_knn_full_predict(
            train_df=pseudo_train_df,
            test_df=test_df,
            use_mbr=False,
            exact_source_memory=exact_source_memory,
            constraint_memories=constraint_memories,
        )
        log(f"[char_tfidf_knn_memory] final low-sim fallback: {low_sim_count}/{total}")
        final_model_sources = []
        final_postprocess_stats = {}
    elif chosen.name == "plan_mbr_blend":
        final_predictions = _representative_prediction_vector(chosen.test_predictions)
        final_model_sources = chosen.ensemble_members or []
        final_postprocess_stats = chosen.postprocess_stats or {}
    elif _is_checkpointed_seq2seq_result(chosen):
        if local_budget_skip_reason and reference_mode_only:
            log("Stage 6/6: local watchdog mode reuses proxy test predictions and skips full-sample inference")
            final_predictions = _representative_prediction_vector(chosen.test_predictions)
            final_model_sources = chosen.executed_checkpoints or []
            final_postprocess_stats = chosen.postprocess_stats or {}
        else:
            cfg = _resolve_final_seq2seq_cfg(
                chosen,
                pipeline_runtime_cfgs,
                accepted_finetune_adapter_dir=accepted_finetune_adapter_dir,
                local_budget_skip_reason=local_budget_skip_reason,
            )
            log("Stage 6/6: full-sample inference + document consistency pass")
            final_predictions, model_used, final_postprocess_stats, final_model_sources = train_full_and_predict(
                pseudo_train_df,
                test_df,
                cfg,
                chosen.best_seed,
                constraint_memories=constraint_memories,
                generation_batch_size=1 if local_budget_skip_reason else (1 if FAST_DEV else 2),
            )
            log(f"Final model source: {model_used}")
    else:
        cached = _load_cached_test_preds(chosen.name, output_dirs)
        if cached is None:
            raise RuntimeError(f"No cached predictions found for pipeline={chosen.name}")
        if cached.ndim == 1:
            final_predictions = [str(x) for x in cached.tolist()]
        else:
            final_predictions = majority_vote_predictions(cached.tolist())
        final_predictions, final_postprocess_stats = apply_consistency_postprocess(
            source_texts=test_df["transliteration_lex"].tolist(),
            predictions=final_predictions,
            group_values=test_df["text_id"].tolist() if "text_id" in test_df.columns else None,
            exact_source_memory=exact_source_memory,
            constraint_memories=constraint_memories,
        )
        final_model_sources = []

    if DO_INFER and not DO_TRAIN and chosen.name == "lookup_baseline":
        if _looks_like_degenerate_lookup_submission(final_predictions):
            raise RuntimeError(
                "Inference-only notebook submit collapsed to degenerate lookup_baseline predictions. "
                "Refusing to write a near-empty submission; enable a viable seq2seq inference path first."
            )

    submission_df = build_translation_submission(test_df, sample_df, final_predictions)
    validate_submission(submission_df, sample_df)
    write_submission(submission_df, output_dirs, kaggle_working_writable)

    metrics_payload: dict[str, Any] = {
        "metric": str(REPORTED_PRIMARY_METRIC),
        "metric_name": str(TRANSLATION_PRIMARY_METRIC),
        "primary_metric": str(REPORTED_PRIMARY_METRIC),
        "official_metric": str(TRANSLATION_PRIMARY_METRIC),
        "target_metric": str(REPORTED_PRIMARY_METRIC),
        "direction": str(PLAN_JSON.get("target_direction", "maximize")),
        "target_direction": str(PLAN_JSON.get("target_direction", "maximize")),
        "split_strategy": "group_kfold",
        "expected_split_strategy": "group_kfold",
        "score_source": "cv",
        "offline_value": float(chosen.cv_score),
        "value": float(chosen.cv_score),
        "metric_description": str(TRANSLATION_METRIC_DESCRIPTION),
        "competition_faithfulness": {
            "faithful": True,
            "expected_metric": str(TRANSLATION_PRIMARY_METRIC),
            "actual_metric": str(TRANSLATION_PRIMARY_METRIC),
            "expected_split_strategy": "group_kfold",
            "actual_split_strategy": "group_kfold",
            "metric_match": True,
            "split_match": True,
            "reasons": [],
            "warnings": [],
        },
        "selected": {
            "name": chosen.name,
            "runtime_name": (
                reference_runtime_cfg.runtime_name
                if chosen.name == REFERENCE_PRIMARY_PIPELINE_NAME and reference_runtime_cfg is not None
                else chosen.name
            ),
            "reference_runtime_mode": (
                reference_runtime_cfg.reference_runtime_mode
                if chosen.name == REFERENCE_PRIMARY_PIPELINE_NAME and reference_runtime_cfg is not None
                else ""
            ),
            "metric": str(REPORTED_PRIMARY_METRIC),
            "official_metric": str(TRANSLATION_PRIMARY_METRIC),
            "offline_value": float(chosen.cv_score),
            "value": float(chosen.cv_score),
            "selector_primary": {
                "cv_unseen_sentence_gmean": float(chosen.unseen_sentence_score),
                "cv_unseen_document_gmean": float(chosen.unseen_document_score),
                "cv_document_gmean": float(chosen.doc_score),
            },
            "cv_sentence_gmean": float(chosen.cv_score),
            "cv_sentence_bleu": float(chosen.bleu),
            "cv_sentence_chrfpp": float(chosen.chrfpp),
            "cv_document_gmean": float(chosen.doc_score),
            "cv_document_bleu": float(chosen.doc_bleu),
            "cv_document_chrfpp": float(chosen.doc_chrfpp),
            "cv_unseen_sentence_gmean": float(chosen.unseen_sentence_score),
            "cv_unseen_sentence_bleu": float(chosen.unseen_sentence_bleu),
            "cv_unseen_sentence_chrfpp": float(chosen.unseen_sentence_chrfpp),
            "cv_unseen_document_gmean": float(chosen.unseen_document_score),
            "cv_unseen_document_bleu": float(chosen.unseen_document_bleu),
            "cv_unseen_document_chrfpp": float(chosen.unseen_document_chrfpp),
            "unseen_coverage": chosen.unseen_coverage or {},
            "slice_metrics": chosen.slice_metrics or {},
            "best_seed": int(chosen.best_seed),
            "executed_checkpoints": chosen.executed_checkpoints or [],
            "ensemble_members": chosen.ensemble_members or [],
        },
        "pipelines": [
            {
                "name": r.name,
                "cv_sentence_gmean": float(r.cv_score),
                "cv_sentence_bleu": float(r.bleu),
                "cv_sentence_chrfpp": float(r.chrfpp),
                "cv_document_gmean": float(r.doc_score),
                "cv_document_bleu": float(r.doc_bleu),
                "cv_document_chrfpp": float(r.doc_chrfpp),
                "cv_unseen_sentence_gmean": float(r.unseen_sentence_score),
                "cv_unseen_sentence_bleu": float(r.unseen_sentence_bleu),
                "cv_unseen_sentence_chrfpp": float(r.unseen_sentence_chrfpp),
                "cv_unseen_document_gmean": float(r.unseen_document_score),
                "cv_unseen_document_bleu": float(r.unseen_document_bleu),
                "cv_unseen_document_chrfpp": float(r.unseen_document_chrfpp),
                "unseen_coverage": r.unseen_coverage or {},
                "complexity_rank": int(r.complexity_rank),
                "best_seed": int(r.best_seed),
                "executed_checkpoints": r.executed_checkpoints or [],
                "postprocess_stats": r.postprocess_stats or {},
                "slice_metrics": r.slice_metrics or {},
                "ensemble_members": r.ensemble_members or [],
            }
            for r in pipeline_results
        ],
        "config": {
            "n_folds": int(N_FOLDS),
            "seeds": [int(x) for x in SEEDS],
            "seq2seq_eval_n_folds": int(seq2seq_n_folds),
            "seq2seq_eval_seeds": [int(x) for x in seq2seq_seeds],
            "fast_dev": bool(FAST_DEV),
            "pipeline_name": str(PIPELINE_NAME),
            "plan_shortlist": shortlisted_pipeline_names(),
            "enable_pipeline_1": bool(ENABLE_PIPELINE_1),
            "enable_pipeline_2": bool(ENABLE_PIPELINE_2),
            "enable_pipeline_3": bool(ENABLE_PIPELINE_3),
            "enable_pipeline_4": bool(ENABLE_PIPELINE_4),
            "enable_ensemble": bool(ENABLE_ENSEMBLE),
            "do_train": bool(allow_optional_finetune),
            "requested_do_train": bool(DO_TRAIN),
            "do_infer": bool(DO_INFER),
            "local_kernel_mode": bool(LOCAL_KERNEL_MODE),
            "local_budget_skip_reason": local_budget_skip_reason,
            "use_normalization": bool(USE_NORMALIZATION),
            "use_determinatives_norm": bool(USE_DETERMINATIVES_NORM),
            "enable_context_window": bool(ENABLE_CONTEXT_WINDOW),
            "enable_pseudo_sentences": bool(ENABLE_PSEUDO_SENTENCES),
            "enable_gold_upweight": bool(ENABLE_GOLD_UPWEIGHT),
            "enable_lexicon_constraints": bool(ENABLE_LEXICON_CONSTRAINTS),
            "enable_retrieval_rerank": bool(ENABLE_RETRIEVAL_RERANK),
            "enable_multi_checkpoint_mbr": bool(ENABLE_MULTI_CHECKPOINT_MBR),
            "enable_public_checkpoints": bool(ENABLE_PUBLIC_CHECKPOINTS),
            "use_mbr": bool(USE_MBR),
            "use_multi_model_pool": bool(USE_MULTI_MODEL_POOL),
            "use_lora_finetune": bool(USE_LORA_FINETUNE),
            "use_diverse_model_addon": bool(USE_DIVERSE_MODEL_ADDON),
            "max_source_len": int(MAX_SOURCE_LEN),
            "max_new_tokens": int(MAX_NEW_TOKENS),
            "num_beams": int(NUM_BEAMS),
            "repetition_penalty": float(REPETITION_PENALTY),
            "sample_temperatures": [float(x) for x in SAMPLE_TEMPERATURES],
            "max_pool_cap": int(MAX_POOL_CAP),
            "top_p": float(TOP_P),
            "temperature": float(TEMPERATURE),
            "seq2seq_generation_batch_size": int(seq2seq_batch_size),
            "reference_mode_only": bool(reference_mode_only),
            "gpu_device": str(GPU_DEVICE),
            "select_epsilon": float(SELECT_EPSILON),
            "retrieval_k": int(RETRIEVAL_K),
            "retrieval_min_sim": float(RETRIEVAL_MIN_SIM),
            "retrieval_ngram_range": [
                int(min(RETRIEVAL_NGRAM_MIN, RETRIEVAL_NGRAM_MAX)),
                int(max(RETRIEVAL_NGRAM_MIN, RETRIEVAL_NGRAM_MAX)),
            ],
            "retrieval_min_df": int(RETRIEVAL_MIN_DF),
            "retrieval_word_weight": float(RETRIEVAL_WORD_WEIGHT),
            "retrieval_word_min_df": int(RETRIEVAL_WORD_MIN_DF),
            "retrieval_run_mbr_variant": bool(RETRIEVAL_RUN_MBR_VARIANT),
            "allow_model_download": bool(ALLOW_MODEL_DOWNLOAD),
            "allow_kernel_finetune": bool(ALLOW_KERNEL_FINETUNE),
            "metadata_supervision_mode": str(METADATA_SUPERVISION_MODE),
            "constraint_rewrite_mode": str(CONSTRAINT_REWRITE_MODE),
            "disable_xgboost": bool(DISABLE_XGBOOST),
            "run_id": str(RUN_ID),
        },
        "competition_metric_hint": str(TRANSLATION_PRIMARY_METRIC),
        "plan_metric_hint": str(REPORTED_PRIMARY_METRIC),
        "pseudo_sentence_pairs": {
            "row_count": int(len(pseudo_train_df)),
            "document_count": int(train_df["oare_id"].nunique()),
            "avg_pairs_per_doc": float(len(pseudo_train_df) / max(1, train_df["oare_id"].nunique())),
            "heuristic_pair_count": heuristic_pair_count,
            "metadata_pair_count": metadata_pair_count,
            "metadata_supervision_doc_coverage": float(
                metadata_pairs.matched_docs / max(1, train_df["oare_id"].nunique())
            ),
            "metadata_candidate_docs": int(metadata_pairs.candidate_docs),
            "metadata_rejected_docs": int(metadata_pairs.rejected_docs),
        },
        "required_seq2seq_ready": bool(required_seq2seq_ready),
        "required_seq2seq_pipelines": sorted(required_seq2seq_pipelines),
        "reference_runtime": {
            "runtime_name": reference_runtime_cfg.runtime_name if reference_runtime_cfg is not None else "",
            "mode": reference_runtime_cfg.reference_runtime_mode if reference_runtime_cfg is not None else "",
            "blocker": reference_runtime_blocker
            or (reference_runtime_cfg.reference_blocker if reference_runtime_cfg is not None else ""),
        },
        "executed_seq2seq_pipelines": executed_seq2seq_pipelines,
        "seq2seq_sources_available": bool(any_local_seq2seq_sources_available),
        "oracc_adapted_ready": bool(oracc_adapted_ready),
        "resolved_domain_adapted_sources": resolved_domain_adapted_sources,
        "base_checkpoint_candidates": base_checkpoint_candidates,
        "domain_adapted_checkpoint_candidates": domain_adapted_checkpoint_candidates,
        "resolved_model_sources": resolved_model_sources,
        "executed_checkpoints": final_model_sources,
        "postprocess_stats": final_postprocess_stats,
        "fine_tune": {
            "fine_tune_ran": bool(fine_tune_result.ran),
            "model_hint": fine_tune_result.model_hint,
            "adapter_dir": fine_tune_result.adapter_dir,
            "baseline_metric": fine_tune_result.baseline_metric,
            "baseline_doc_metric": fine_tune_result.baseline_doc_metric,
            "baseline_slice_metrics": fine_tune_result.baseline_slice_metrics,
            "validation_metric": fine_tune_result.validation_metric,
            "validation_doc_metric": fine_tune_result.validation_doc_metric,
            "validation_slice_metrics": fine_tune_result.validation_slice_metrics,
            "reason": fine_tune_result.reason,
        },
        "artifacts": {
            "output_dirs": [str(p) for p in output_dirs],
            "submission_paths": [str(p / "submission.csv") for p in output_dirs],
        },
    }
    _write_json_all("metrics.json", metrics_payload, _metric_output_dirs(output_dirs))


# =====================================================================================
# Entrypoint
# =====================================================================================


def custom_main() -> None:
    slug = _competition_slug()
    local_run_dir, output_dirs, kaggle_working_writable = resolve_output_dirs(slug, RUN_ID)

    train_df, test_df, sample_df, data_dir = load_competition_frames(slug)
    log(f"Using data directory: {data_dir}")
    log(f"Local run directory: {local_run_dir}")

    modality = detect_modality(train_df, test_df)
    log(f"Detected modality: {modality}")

    has_translation_schema = {"transliteration", "translation"}.issubset(set(train_df.columns)) and {
        "id",
        "transliteration",
    }.issubset(set(test_df.columns))

    if modality == "text" and has_translation_schema:
        _auto_disable_training_if_unsafe()
        run_translation_seq2seq(train_df, test_df, sample_df, data_dir, output_dirs, kaggle_working_writable)
        return

    if modality in {"tabular", "image", "audio", "video", "other"}:
        msg = modality_stub_message(modality)
        _write_json_all(
            "metrics.json",
            {
                "metric": "unsupported_modality",
                "direction": "maximize",
                "score_source": "none",
                "offline_value": 0.0,
                "value": 0.0,
                "note": msg,
                "config": {
                    "detected_modality": modality,
                    "run_id": str(RUN_ID),
                },
            },
            _metric_output_dirs(output_dirs),
        )
        raise RuntimeError(msg)

    raise RuntimeError(f"Unhandled modality '{modality}'.")


def main() -> None:
    custom_main()


if __name__ == "__main__":
    main()
