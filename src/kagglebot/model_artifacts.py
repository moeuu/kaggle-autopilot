from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.asset_modality import (
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_FILENAMES,
    MODEL_ARTIFACT_SUFFIXES,
    asset_suffix,
)

MODEL_INDEX_SUFFIXES = frozenset({".bin.index.json", ".safetensors.index.json"} & MODEL_ARTIFACT_COMPOUND_SUFFIXES)
MODEL_DIRECTORY_ARTIFACT_SUFFIX = ".savedmodel"
COREML_PACKAGE_DIRECTORY_SUFFIX = ".mlpackage"
COREML_COMPILED_PACKAGE_DIRECTORY_SUFFIX = ".mlmodelc"
HUGGINGFACE_MODEL_DIRECTORY_SUFFIX = ".hfmodel"
MLFLOW_MODEL_DIRECTORY_SUFFIX = ".mlflowmodel"
TENSORFLOW_CHECKPOINT_INDEX_SUFFIX = ".ckpt.index"
TENSORFLOW_CHECKPOINT_DIRECTORY_SUFFIX = ".tfcheckpoint"
MODEL_DIRECTORY_ARTIFACT_SUFFIXES = frozenset(
    {
        MODEL_DIRECTORY_ARTIFACT_SUFFIX,
        COREML_PACKAGE_DIRECTORY_SUFFIX,
        COREML_COMPILED_PACKAGE_DIRECTORY_SUFFIX,
        HUGGINGFACE_MODEL_DIRECTORY_SUFFIX,
        MLFLOW_MODEL_DIRECTORY_SUFFIX,
        TENSORFLOW_CHECKPOINT_DIRECTORY_SUFFIX,
    }
)
SAVED_MODEL_MARKER_FILENAMES = frozenset({"saved_model.pb", "saved_model.pbtxt"})
TENSORFLOW_CHECKPOINT_STATE_FILENAME = "checkpoint"
HUGGINGFACE_MODEL_WEIGHT_FILENAMES = frozenset(
    {
        "adapter_model.bin",
        "adapter_model.safetensors",
        "flax_model.msgpack",
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
    }
)
HUGGINGFACE_MODEL_CONFIG_FILENAMES = frozenset(
    {
        "adapter_config.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
MLFLOW_MODEL_MARKER_FILENAME = "MLmodel"
COREML_PACKAGE_MARKER_FILENAME = "Manifest.json"
MLFLOW_MODEL_PAYLOAD_FILENAMES = frozenset(
    {
        "model.pkl",
        "model.joblib",
        "python_model.pkl",
        "model.cb",
        "model.cbm",
        "model.ubj",
        "model.skops",
    }
)
_TENSORFLOW_CHECKPOINT_DATA_RE = re.compile(r"^(?P<prefix>.+)\.data-\d+-of-\d+$", re.IGNORECASE)
MODEL_ARTIFACT_SIDECAR_FILENAMES = frozenset(
    name for name in MODEL_ARTIFACT_FILENAMES if not name.endswith(".index.json")
)


def is_saved_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / marker).is_file() for marker in SAVED_MODEL_MARKER_FILENAMES)


def is_saved_model_marker_file(path: Path) -> bool:
    return (
        path.is_file() and path.name.lower() in SAVED_MODEL_MARKER_FILENAMES and is_saved_model_directory(path.parent)
    )


def is_tensorflow_checkpoint_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / TENSORFLOW_CHECKPOINT_STATE_FILENAME).is_file():
        return False
    try:
        index_files = sorted(
            child for child in path.iterdir() if child.is_file() and child.name.lower().endswith(".index")
        )
    except OSError:
        return False
    return any(is_tensorflow_checkpoint_index_artifact(index_file) for index_file in index_files)


def is_huggingface_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        files = [child for child in path.iterdir() if child.is_file()]
    except OSError:
        return False
    names = {child.name.lower() for child in files}
    if not names & HUGGINGFACE_MODEL_CONFIG_FILENAMES:
        return False
    for child in files:
        if _is_huggingface_model_weight_file(child):
            return True
        if is_model_index_artifact(child) and not missing_model_index_shards(child):
            return True
    return False


def is_mlflow_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / MLFLOW_MODEL_MARKER_FILENAME).is_file():
        return False
    try:
        files = [child for child in path.rglob("*") if child.is_file()]
    except OSError:
        return False
    return any(_is_mlflow_model_payload_file(child) for child in files)


def is_coreml_package_directory(path: Path) -> bool:
    if not path.is_dir() or path.suffix.lower() != COREML_PACKAGE_DIRECTORY_SUFFIX:
        return False
    if not (path / COREML_PACKAGE_MARKER_FILENAME).is_file():
        return False
    try:
        return any(child.is_file() and child.suffix.lower() == ".mlmodel" for child in path.rglob("*"))
    except OSError:
        return False


def is_coreml_compiled_package_directory(path: Path) -> bool:
    if not path.is_dir() or path.suffix.lower() != COREML_COMPILED_PACKAGE_DIRECTORY_SUFFIX:
        return False
    try:
        return any(child.is_file() for child in path.rglob("*"))
    except OSError:
        return False


def is_model_directory_artifact(path: Path) -> bool:
    return (
        is_saved_model_directory(path)
        or is_tensorflow_checkpoint_directory(path)
        or is_huggingface_model_directory(path)
        or is_mlflow_model_directory(path)
        or is_coreml_package_directory(path)
        or is_coreml_compiled_package_directory(path)
    )


def model_directory_artifact_suffix(path: Path) -> str:
    if is_saved_model_directory(path):
        return MODEL_DIRECTORY_ARTIFACT_SUFFIX
    if is_tensorflow_checkpoint_directory(path):
        return TENSORFLOW_CHECKPOINT_DIRECTORY_SUFFIX
    if is_huggingface_model_directory(path):
        return HUGGINGFACE_MODEL_DIRECTORY_SUFFIX
    if is_mlflow_model_directory(path):
        return MLFLOW_MODEL_DIRECTORY_SUFFIX
    if is_coreml_package_directory(path):
        return COREML_PACKAGE_DIRECTORY_SUFFIX
    if is_coreml_compiled_package_directory(path):
        return COREML_COMPILED_PACKAGE_DIRECTORY_SUFFIX
    return ""


def is_tensorflow_checkpoint_index_artifact(path: Path) -> bool:
    prefix = _tensorflow_checkpoint_index_prefix(path)
    return bool(prefix and _tensorflow_checkpoint_data_shard_specs(path.parent, prefix))


def is_tensorflow_checkpoint_artifact(path: Path) -> bool:
    if is_tensorflow_checkpoint_index_artifact(path):
        return True
    return asset_suffix(path) == ".ckpt" and bool(tensorflow_checkpoint_sidecar_specs(path))


def tensorflow_checkpoint_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    prefix = _tensorflow_checkpoint_prefix(path)
    if not prefix:
        return []
    specs: list[tuple[Path, str]] = []
    index_path = path.parent / f"{prefix}.index"
    if index_path.is_file() and index_path != path:
        specs.append((index_path, index_path.name))
    specs.extend(_tensorflow_checkpoint_data_shard_specs(path.parent, prefix))
    meta_path = path.parent / f"{prefix}.meta"
    if meta_path.is_file() and meta_path != path:
        specs.append((meta_path, meta_path.name))
    state_path = path.parent / TENSORFLOW_CHECKPOINT_STATE_FILENAME
    if state_path.is_file() and state_path != path:
        specs.append((state_path, state_path.name))
    return specs


def copy_tensorflow_checkpoint_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in tensorflow_checkpoint_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / sidecar_name,
            )
        )
    return copied


def is_model_index_artifact(path: Path) -> bool:
    return asset_suffix(path) in MODEL_INDEX_SUFFIXES


def model_index_shard_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw_name in _model_index_weight_map_values(path):
        name = _safe_model_shard_name(raw_name)
        if name and name not in names:
            names.append(name)
    return names


def invalid_model_index_shard_names(path: Path) -> list[str]:
    invalid: list[str] = []
    for raw_name in _model_index_weight_map_values(path):
        if _safe_model_shard_name(raw_name) is None and raw_name not in invalid:
            invalid.append(raw_name)
    return invalid


def model_index_shard_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in model_index_shard_names(path):
        shard_path = path.parent / Path(name)
        if shard_path.is_file():
            specs.append((shard_path, name))
    return specs


def missing_model_index_shards(path: Path) -> list[str]:
    missing: list[str] = []
    for name in model_index_shard_names(path):
        if not (path.parent / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_model_index_shards_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for shard_path, shard_name in model_index_shard_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=shard_path,
                destination=destination.parent / Path(shard_name),
            )
        )
    return copied


def model_artifact_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    if asset_suffix(path) not in MODEL_ARTIFACT_SUFFIXES:
        return []
    if path.name.lower() in MODEL_ARTIFACT_SIDECAR_FILENAMES:
        return []
    specs: list[tuple[Path, str]] = []
    for filename in sorted(MODEL_ARTIFACT_SIDECAR_FILENAMES):
        sidecar = path.parent / filename
        if sidecar.is_file() and sidecar != path:
            specs.append((sidecar, filename))
    return specs


def copy_model_artifact_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in model_artifact_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / sidecar_name,
            )
        )
    return copied


def _model_index_weight_map_values(path: Path) -> list[str]:
    if not is_model_index_artifact(path):
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return []
    return [raw_name for raw_name in weight_map.values() if isinstance(raw_name, str)]


def _safe_model_shard_name(value: str) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _is_huggingface_model_weight_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower() in HUGGINGFACE_MODEL_WEIGHT_FILENAMES:
        return True
    suffix = asset_suffix(path)
    return suffix in {".bin", ".safetensors"} and any(
        token in path.stem.lower() for token in ("adapter", "checkpoint", "model", "pytorch", "weights")
    )


def _is_mlflow_model_payload_file(path: Path) -> bool:
    if not path.is_file() or path.name == MLFLOW_MODEL_MARKER_FILENAME:
        return False
    if path.name.lower() in MLFLOW_MODEL_PAYLOAD_FILENAMES:
        return True
    suffix = asset_suffix(path)
    if suffix in MODEL_ARTIFACT_SUFFIXES | MODEL_ARTIFACT_COMPOUND_SUFFIXES:
        return True
    return False


def _tensorflow_checkpoint_index_prefix(path: Path) -> str:
    if not path.is_file():
        return ""
    name = path.name
    if not name.lower().endswith(".index"):
        return ""
    prefix = name[: -len(".index")]
    return prefix if prefix else ""


def _tensorflow_checkpoint_prefix(path: Path) -> str:
    index_prefix = _tensorflow_checkpoint_index_prefix(path)
    if index_prefix:
        return index_prefix
    if path.is_file() and asset_suffix(path) == ".ckpt":
        return path.name
    return ""


def _tensorflow_checkpoint_data_shard_specs(parent: Path, prefix: str) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    try:
        candidates = sorted(parent.glob(f"{prefix}.data-*-of-*"), key=lambda candidate: candidate.name)
    except OSError:
        return []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        match = _TENSORFLOW_CHECKPOINT_DATA_RE.match(candidate.name)
        if match is None or match.group("prefix") != prefix:
            continue
        specs.append((candidate, candidate.name))
    return specs
