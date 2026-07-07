from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from kagglebot.asset_collections import ASSET_COLLECTION_DIR_NAMES as ASSET_COLLECTION_DIR_NAMES
from kagglebot.compression_suffixes import ASSET_COMPRESSION_SUFFIXES as ASSET_COMPRESSION_SUFFIXES
from kagglebot.submission_sample_discovery import TABULAR_GEOJSON_SUFFIXES, TABULAR_INPUT_SUFFIXES


def _compressed_asset_suffixes(base_suffixes: Iterable[str]) -> tuple[str, ...]:
    return tuple(f"{base}{compression}" for base in base_suffixes for compression in ASSET_COMPRESSION_SUFFIXES)


_VECTOR_IMAGE_BASE_SUFFIXES = (".svg",)
VECTOR_IMAGE_SUFFIXES = {
    *_VECTOR_IMAGE_BASE_SUFFIXES,
    ".svgz",
    *_compressed_asset_suffixes(_VECTOR_IMAGE_BASE_SUFFIXES),
}
IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jxl",
    ".jp2",
    ".jpg",
    ".jpeg",
    ".exr",
    ".png",
    ".pbm",
    ".pgm",
    ".pnm",
    ".ppm",
    ".tif",
    ".tiff",
    ".tga",
    ".webp",
    *VECTOR_IMAGE_SUFFIXES,
}


DICOM_IMAGE_BASE_SUFFIXES = (".dcm", ".dicom", ".ima")
NIFTI_IMAGE_BASE_SUFFIXES = (".nii",)
ANALYZE_IMAGE_PAIR_BASE_SUFFIXES = (".hdr", ".img")
MEDICAL_HEADER_IMAGE_BASE_SUFFIXES = (".mha", ".mhd", ".nhdr", ".nrrd")
CRYOEM_IMAGE_BASE_SUFFIXES = (".ccp4", ".mrc", ".mrcs")
MICROSCOPY_IMAGE_BASE_SUFFIXES = (".dm3", ".dm4", ".lif", ".lsm", ".nd2", ".oir", ".vsi", ".zvi")
_MEDICAL_IMAGE_COMPRESSIBLE_BASE_SUFFIXES = (
    *ANALYZE_IMAGE_PAIR_BASE_SUFFIXES,
    *CRYOEM_IMAGE_BASE_SUFFIXES,
    *DICOM_IMAGE_BASE_SUFFIXES,
    *MEDICAL_HEADER_IMAGE_BASE_SUFFIXES,
    *NIFTI_IMAGE_BASE_SUFFIXES,
)
ANALYZE_IMAGE_PAIR_SUFFIXES = {
    *ANALYZE_IMAGE_PAIR_BASE_SUFFIXES,
    *_compressed_asset_suffixes(ANALYZE_IMAGE_PAIR_BASE_SUFFIXES),
}
DICOM_IMAGE_SUFFIXES = {
    *DICOM_IMAGE_BASE_SUFFIXES,
    *_compressed_asset_suffixes(DICOM_IMAGE_BASE_SUFFIXES),
}
NIFTI_IMAGE_SUFFIXES = {
    *NIFTI_IMAGE_BASE_SUFFIXES,
    *_compressed_asset_suffixes(NIFTI_IMAGE_BASE_SUFFIXES),
}
MEDICAL_HEADER_IMAGE_SUFFIXES = {
    *MEDICAL_HEADER_IMAGE_BASE_SUFFIXES,
    *_compressed_asset_suffixes(MEDICAL_HEADER_IMAGE_BASE_SUFFIXES),
}
CRYOEM_IMAGE_SUFFIXES = {
    *CRYOEM_IMAGE_BASE_SUFFIXES,
    *_compressed_asset_suffixes(CRYOEM_IMAGE_BASE_SUFFIXES),
}
MEDICAL_IMAGE_SUFFIXES = {
    ".bif",
    ".czi",
    ".mrxs",
    ".ndpi",
    ".ome.tif",
    ".ome.tiff",
    ".qptiff",
    ".scn",
    ".svs",
    ".vms",
    ".vmu",
    *ANALYZE_IMAGE_PAIR_SUFFIXES,
    *CRYOEM_IMAGE_SUFFIXES,
    *DICOM_IMAGE_SUFFIXES,
    *MEDICAL_HEADER_IMAGE_SUFFIXES,
    *MICROSCOPY_IMAGE_BASE_SUFFIXES,
    *NIFTI_IMAGE_SUFFIXES,
}
AUDIO_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mid",
    ".midi",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
VIDEO_SUFFIXES = {".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
_SIGNAL_NEUROPHYS_BASE_SUFFIXES = (".cnt", ".eeg", ".fdt", ".fif", ".set", ".trc", ".vhdr", ".vmrk")
_SIGNAL_NEUROPHYS_COMPRESSIBLE_SUFFIXES = (".fif", ".set", ".vhdr", ".vmrk")
SIGNAL_NEUROPHYS_BASE_SUFFIXES = _SIGNAL_NEUROPHYS_BASE_SUFFIXES
SIGNAL_SUFFIXES = {
    ".abf",
    ".bdf",
    ".edf",
    ".hea",
    ".nwb",
    ".tdms",
    *_SIGNAL_NEUROPHYS_BASE_SUFFIXES,
    *_compressed_asset_suffixes((".hea",)),
    *_compressed_asset_suffixes(_SIGNAL_NEUROPHYS_COMPRESSIBLE_SUFFIXES),
}
ARRAY_SUFFIXES = {".npy", ".npz"}
_FITS_BASE_SUFFIXES = (".fits", ".fit", ".fts")
SCIENTIFIC_ARRAY_SUFFIXES = {
    *_FITS_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_FITS_BASE_SUFFIXES),
    ".nc",
    ".nc4",
    ".cdf",
    ".grib",
    ".grib2",
    ".grb",
    ".h5ad",
    ".loom",
    ".mat",
}
DIRECTORY_ARRAY_SUFFIXES = {".n5", ".ome.zarr", ".zarr"}
_POINT_CLOUD_BASE_SUFFIXES = (
    ".abc",
    ".3ds",
    ".blend",
    ".brep",
    ".dae",
    ".e57",
    ".fbx",
    ".glb",
    ".gltf",
    ".ifc",
    ".iges",
    ".igs",
    ".las",
    ".laz",
    ".mesh",
    ".msh",
    ".obj",
    ".off",
    ".pcd",
    ".ply",
    ".pts",
    ".ptx",
    ".stl",
    ".step",
    ".stp",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".vtk",
    ".vtp",
    ".vtu",
    ".x3d",
    ".xyz",
)
_POINT_CLOUD_TEXT_COMPRESSIBLE_SUFFIXES = (
    ".dae",
    ".brep",
    ".ifc",
    ".iges",
    ".igs",
    ".mesh",
    ".msh",
    ".ply",
    ".obj",
    ".off",
    ".pcd",
    ".pts",
    ".ptx",
    ".stl",
    ".step",
    ".stp",
    ".usd",
    ".usda",
    ".vtk",
    ".vtp",
    ".vtu",
    ".x3d",
    ".xyz",
)
POINT_CLOUD_TEXT_METADATA_SUFFIXES = (".xyz", ".pts", ".ptx")
POINT_CLOUD_SUFFIXES = {
    *_POINT_CLOUD_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_POINT_CLOUD_TEXT_COMPRESSIBLE_SUFFIXES),
}
_GEOSPATIAL_TEXT_BASE_SUFFIXES = (".geojsonl", ".geojsonseq", ".osm", ".topojson")
_GEOSPATIAL_TILE_SUFFIXES = (".mbtiles", ".mvt", ".pmtiles")
_GEOSPATIAL_RASTER_BASE_SUFFIXES = (".dem", ".dt0", ".dt1", ".dt2", ".ecw", ".hgt", ".sid")
GEOSPATIAL_RASTER_BASE_SUFFIXES = _GEOSPATIAL_RASTER_BASE_SUFFIXES
GEOSPATIAL_SUFFIXES = {
    *TABULAR_GEOJSON_SUFFIXES,
    *_GEOSPATIAL_TEXT_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_GEOSPATIAL_TEXT_BASE_SUFFIXES),
    *_GEOSPATIAL_RASTER_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_GEOSPATIAL_RASTER_BASE_SUFFIXES),
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".gpkg",
    ".geopackage",
    ".osm.pbf",
    *_GEOSPATIAL_TILE_SUFFIXES,
    ".bil",
    ".bip",
    ".bsq",
    ".kml",
    *_compressed_asset_suffixes((".kml",)),
    ".kmz",
    ".mif",
    ".vrt",
}
_TEXT_DOCUMENT_BASE_SUFFIXES = (".rtf", ".html", ".htm", ".md", ".tex", ".rst", ".adoc", ".srt", ".vtt")
DOCUMENT_HTML_BASE_SUFFIXES = (".html", ".htm")
DOCUMENT_TEXT_METADATA_SUFFIXES = (*_TEXT_DOCUMENT_BASE_SUFFIXES, ".txt")
DOCUMENT_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".epub",
    ".odp",
    ".odt",
    ".ppt",
    ".pptx",
    *_TEXT_DOCUMENT_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_TEXT_DOCUMENT_BASE_SUFFIXES),
}
_BIO_SEQUENCE_BASE_SUFFIXES = (".fasta", ".fa", ".fna", ".faa", ".ffn", ".frn", ".fastq", ".fq")
_BIO_GENOMIC_BASE_SUFFIXES = (
    ".bam",
    ".bcf",
    ".bed",
    ".bigbed",
    ".bigwig",
    ".cram",
    ".gff",
    ".gff3",
    ".gtf",
    ".pod5",
    ".sam",
    ".vcf",
)
_BIO_GENOMIC_TEXT_BASE_SUFFIXES = (".bed", ".gff", ".gff3", ".gtf", ".sam", ".vcf")
_BIO_STRUCTURE_BASE_SUFFIXES = (".pdb", ".cif", ".mmcif", ".sdf", ".mol", ".mol2")
_BIO_CHEM_TEXT_BASE_SUFFIXES = (".smi", ".smiles", ".inchi", ".selfies", ".rxn")
BIO_SEQUENCE_BASE_SUFFIXES = _BIO_SEQUENCE_BASE_SUFFIXES
BIO_GENOMIC_BASE_SUFFIXES = _BIO_GENOMIC_BASE_SUFFIXES
BIO_GENOMIC_TEXT_BASE_SUFFIXES = _BIO_GENOMIC_TEXT_BASE_SUFFIXES
BIO_CHEM_TEXT_BASE_SUFFIXES = _BIO_CHEM_TEXT_BASE_SUFFIXES
BIO_FASTQ_BASE_SUFFIXES = (".fastq", ".fq")
BIO_PDB_STRUCTURE_BASE_SUFFIXES = (".pdb", ".cif", ".mmcif")
BIO_MOL_STRUCTURE_BASE_SUFFIXES = (".sdf", ".mol")
BIO_STRUCTURE_SUFFIXES = {
    *_BIO_STRUCTURE_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_BIO_STRUCTURE_BASE_SUFFIXES),
    *_BIO_CHEM_TEXT_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_BIO_CHEM_TEXT_BASE_SUFFIXES),
    *_BIO_SEQUENCE_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_BIO_SEQUENCE_BASE_SUFFIXES),
    *_BIO_GENOMIC_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_BIO_GENOMIC_TEXT_BASE_SUFFIXES),
}
_GRAPH_BASE_SUFFIXES = (".graphml", ".gexf", ".gml", ".mtx", ".edgelist", ".edges")
_GRAPH_RDF_BASE_SUFFIXES = (".jsonld", ".nq", ".nt", ".owl", ".rdf", ".trig", ".ttl")
GRAPH_XML_BASE_SUFFIXES = (".graphml", ".gexf")
GRAPH_EDGE_LIST_BASE_SUFFIXES = (".edgelist", ".edges")
GRAPH_RDF_BASE_SUFFIXES = _GRAPH_RDF_BASE_SUFFIXES
GRAPH_SUFFIXES = {
    *_GRAPH_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_GRAPH_BASE_SUFFIXES),
    *_GRAPH_RDF_BASE_SUFFIXES,
    *_compressed_asset_suffixes(_GRAPH_RDF_BASE_SUFFIXES),
}
ANNOTATION_JSON_SUFFIXES = {
    ".json",
    *_compressed_asset_suffixes((".json",)),
}
ANNOTATION_TEXT_SUFFIXES = {
    ".txt",
    *_compressed_asset_suffixes((".txt",)),
}
ANNOTATION_XML_SUFFIXES = {
    ".xml",
    *_compressed_asset_suffixes((".xml",)),
}
ANNOTATION_FILE_SUFFIXES = ANNOTATION_JSON_SUFFIXES | ANNOTATION_TEXT_SUFFIXES | ANNOTATION_XML_SUFFIXES
ANNOTATION_DIR_NAME_TOKENS = {
    "annotation",
    "annotations",
    "bbox",
    "bboxes",
    "bounding_box",
    "bounding_boxes",
    "instance",
    "instances",
    "label",
    "labels",
    "mask",
    "masks",
    "segmentation",
    "segmentations",
}
ANNOTATION_FILE_NAME_TOKENS = {
    "annotation",
    "annotations",
    "bbox",
    "bboxes",
    "boundingbox",
    "boundingboxes",
    "coco",
    "cvat",
    "instance",
    "instances",
    "label",
    "labelstudio",
    "labelme",
    "labels",
    "mask",
    "masks",
    "pascal",
    "segmentation",
    "segmentations",
    "voc",
    "yolo",
}
TABULAR_DATA_SUFFIXES = set(TABULAR_INPUT_SUFFIXES) - {".h5"}
MODEL_ARTIFACT_SUFFIXES = {
    ".bin",
    ".blob",
    ".bst",
    ".cbm",
    ".dlc",
    ".engine",
    ".hef",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".keras",
    ".mlmodel",
    ".onnx",
    ".plan",
    ".pmml",
    ".rknn",
    ".safetensors",
    ".gguf",
    ".msgpack",
    ".tflite",
    ".pb",
    ".joblib",
    ".skops",
    ".spm",
    ".ubj",
    ".xgb",
}
MODEL_ARTIFACT_COMPOUND_SUFFIXES = {
    ".bin.index.json",
    ".ckpt.index",
    ".safetensors.index.json",
}
MODEL_ARTIFACT_FILENAMES = {
    "adapter_config.json",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "feature_extractor_config.json",
    "generation_config.json",
    "image_processor_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "model.safetensors.index.json",
}
MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES = {
    name for name in MODEL_ARTIFACT_FILENAMES if name.endswith(".json") and not name.endswith(".index.json")
}
MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES = {
    "merges.txt",
    "vocab.txt",
}
MODEL_ARTIFACT_NAME_TOKENS = {
    "adapter",
    "booster",
    "checkpoint",
    "checkpoints",
    "ckpt",
    "coreml",
    "model",
    "models",
    "pmml",
    "pytorch",
    "catboost",
    "lightgbm",
    "sklearn",
    "skops",
    "tensorflow",
    "tf",
    "weights",
    "xgboost",
}
CODE_SUFFIXES = {".py", ".ipynb", ".r", ".jl"}
ARCHIVE_SUFFIXES = {
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
    ".7z",
    ".rar",
}
GENERIC_ARCHIVE_SUFFIXES = ARCHIVE_SUFFIXES & {".7z", ".rar"}
TAR_ARCHIVE_SUFFIXES = ARCHIVE_SUFFIXES - {".zip", *GENERIC_ARCHIVE_SUFFIXES}
ZSTD_TAR_ARCHIVE_SUFFIXES = {".tar.zst", ".tzst"}
COMPOUND_ASSET_SUFFIXES = tuple(
    sorted(
        {
            ".nii.gz",
            *VECTOR_IMAGE_SUFFIXES,
            *MEDICAL_IMAGE_SUFFIXES,
            ".ome.tiff",
            ".ome.tif",
            ".tar.gz",
            ".tar.bz2",
            ".tar.xz",
            ".tar.zst",
            *SCIENTIFIC_ARRAY_SUFFIXES,
            *SIGNAL_SUFFIXES,
            *GEOSPATIAL_SUFFIXES,
            *DOCUMENT_SUFFIXES,
            *BIO_STRUCTURE_SUFFIXES,
            *GRAPH_SUFFIXES,
            *POINT_CLOUD_SUFFIXES,
            *DIRECTORY_ARRAY_SUFFIXES,
            *MODEL_ARTIFACT_COMPOUND_SUFFIXES,
            *TABULAR_DATA_SUFFIXES,
        },
        key=len,
        reverse=True,
    )
)
DATA_ASSET_SUFFIXES = (
    IMAGE_SUFFIXES
    | MEDICAL_IMAGE_SUFFIXES
    | AUDIO_SUFFIXES
    | VIDEO_SUFFIXES
    | SIGNAL_SUFFIXES
    | ARRAY_SUFFIXES
    | SCIENTIFIC_ARRAY_SUFFIXES
    | DIRECTORY_ARRAY_SUFFIXES
    | POINT_CLOUD_SUFFIXES
    | GEOSPATIAL_SUFFIXES
    | DOCUMENT_SUFFIXES
    | BIO_STRUCTURE_SUFFIXES
    | GRAPH_SUFFIXES
    | TABULAR_DATA_SUFFIXES
    | MODEL_ARTIFACT_SUFFIXES
    | MODEL_ARTIFACT_COMPOUND_SUFFIXES
)


def asset_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in COMPOUND_ASSET_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def is_model_artifact_path(path: Path) -> bool:
    name = path.name.lower()
    return name in MODEL_ARTIFACT_FILENAMES or asset_suffix(path) in (
        MODEL_ARTIFACT_SUFFIXES | MODEL_ARTIFACT_COMPOUND_SUFFIXES
    )


def artifact_suffix(path: Path) -> str:
    """Return the longest known suffix for a submission/data artifact path."""

    name = path.name.lower()
    for suffix in sorted(DATA_ASSET_SUFFIXES | ARCHIVE_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def artifact_stem(path: Path) -> str:
    """Return a filename stem after removing a known compound artifact suffix."""

    name = path.name
    suffix = artifact_suffix(path)
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)].strip(".") or "submission"
    return path.stem.strip(".") or "submission"


def archive_container(suffixes: Iterable[str], *, default: str | None = None) -> str | None:
    """Return the submission/archive container implied by known archive suffixes."""

    normalized = [str(suffix or "").strip().lower() for suffix in suffixes]
    if any(suffix in TAR_ARCHIVE_SUFFIXES for suffix in normalized):
        return "tar"
    for suffix in normalized:
        if suffix in GENERIC_ARCHIVE_SUFFIXES:
            return suffix.lstrip(".")
    if ".zip" in normalized:
        return "zip"
    return default


def is_data_asset_path(path: Path) -> bool:
    if is_model_artifact_path(path):
        return path.is_file()
    suffix = asset_suffix(path)
    if suffix not in DATA_ASSET_SUFFIXES:
        return False
    return path.is_file() or (path.is_dir() and suffix in DIRECTORY_ARRAY_SUFFIXES)


def infer_asset_modality(data_dir: Path, *, include_code_artifact: bool = False) -> str:
    try:
        paths = [path for path in data_dir.rglob("*") if is_data_asset_path(path)]
    except OSError:
        return "unknown"
    if _looks_like_detection_annotation_dataset(paths):
        return "image"
    if include_code_artifact and _looks_like_model_artifact_dataset(paths):
        return "artifact"
    file_exts = {asset_suffix(path) for path in paths}
    return infer_asset_modality_from_extensions(file_exts, include_code_artifact=include_code_artifact)


def is_detection_annotation_path(path: Path) -> bool:
    suffix = asset_suffix(path)
    if suffix not in ANNOTATION_FILE_SUFFIXES:
        return False
    parent_tokens = {
        token for parent in path.parents for token in _path_name_tokens(parent.name) if token and token not in {"", "."}
    }
    name_tokens = _path_name_tokens(path.name)
    if suffix in ANNOTATION_JSON_SUFFIXES and name_tokens & ANNOTATION_FILE_NAME_TOKENS:
        return True
    if parent_tokens & ANNOTATION_DIR_NAME_TOKENS:
        return True
    return bool(name_tokens & {"coco", "cvat", "labelme", "labelstudio", "pascal", "voc", "yolo"})


def _looks_like_detection_annotation_dataset(paths: Iterable[Path]) -> bool:
    files = [path for path in paths if path.is_file()]
    if not files:
        return False
    return any(is_detection_annotation_path(path) for path in files)


def _path_name_tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9]+", name.lower()) if token}


def _looks_like_model_artifact_dataset(paths: Iterable[Path]) -> bool:
    files = [path for path in paths if path.is_file()]
    if not files:
        return False
    if not any(is_model_artifact_path(path) for path in files):
        return False
    return all(is_model_artifact_path(path) or asset_suffix(path) in CODE_SUFFIXES for path in files)


def infer_asset_modality_from_extensions(
    file_exts: set[str],
    *,
    include_code_artifact: bool = False,
) -> str:
    if file_exts & MEDICAL_IMAGE_SUFFIXES:
        return "medical_imaging"
    if file_exts & IMAGE_SUFFIXES:
        return "image"
    if file_exts & AUDIO_SUFFIXES:
        return "audio"
    if file_exts & VIDEO_SUFFIXES:
        return "video"
    if file_exts & SIGNAL_SUFFIXES:
        return "signal"
    if file_exts & (ARRAY_SUFFIXES | SCIENTIFIC_ARRAY_SUFFIXES | DIRECTORY_ARRAY_SUFFIXES):
        return "array"
    if file_exts & POINT_CLOUD_SUFFIXES:
        return "point_cloud"
    if file_exts & GEOSPATIAL_SUFFIXES:
        return "geospatial"
    if file_exts & DOCUMENT_SUFFIXES:
        return "text"
    if file_exts & BIO_STRUCTURE_SUFFIXES:
        return "bio"
    if file_exts & GRAPH_SUFFIXES:
        return "graph"
    if file_exts & TABULAR_DATA_SUFFIXES:
        return "tabular"
    if include_code_artifact and file_exts & (MODEL_ARTIFACT_SUFFIXES | MODEL_ARTIFACT_COMPOUND_SUFFIXES):
        return "artifact"
    if include_code_artifact and file_exts & CODE_SUFFIXES:
        return "code"
    if include_code_artifact and file_exts & ARCHIVE_SUFFIXES:
        return "artifact"
    return "unknown"
