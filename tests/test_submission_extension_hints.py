from __future__ import annotations

import re

from kagglebot.asset_modality import (
    ARCHIVE_SUFFIXES,
    ARRAY_SUFFIXES,
    AUDIO_SUFFIXES,
    BIO_STRUCTURE_SUFFIXES,
    CODE_SUFFIXES,
    DIRECTORY_ARRAY_SUFFIXES,
    DOCUMENT_SUFFIXES,
    GEOSPATIAL_SUFFIXES,
    GRAPH_SUFFIXES,
    IMAGE_SUFFIXES,
    MEDICAL_IMAGE_SUFFIXES,
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_SUFFIXES,
    POINT_CLOUD_SUFFIXES,
    SCIENTIFIC_ARRAY_SUFFIXES,
    SIGNAL_SUFFIXES,
    VIDEO_SUFFIXES,
)
from kagglebot.model_artifacts import MODEL_DIRECTORY_ARTIFACT_SUFFIXES
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_ARTIFACT_KEYWORDS,
    ARCHIVE_SUBMISSION_EXTENSION_PATTERN,
    ARCHIVE_SUBMISSION_SUFFIXES,
    ARCHIVE_SUBMISSION_SUFFIXES_ORDERED,
    ARRAY_SUBMISSION_ARTIFACT_KEYWORDS,
    ARRAY_SUBMISSION_TOKEN_PATTERNS,
    BIO_SUBMISSION_ARTIFACT_KEYWORDS,
    BIO_SUBMISSION_TOKEN_PATTERNS,
    CODE_FENCE_LANG_SUFFIX_SPECS,
    CODE_FENCE_LANG_TO_SUFFIX,
    CODE_SUBMISSION_ARTIFACT_KEYWORDS,
    CODE_SUBMISSION_TOKEN_PATTERNS,
    COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES,
    COMPRESSION_TOKEN_PATTERN_SPECS,
    COMPRESSION_TOKEN_PATTERNS,
    DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS,
    DOCUMENT_SUBMISSION_TOKEN_PATTERNS,
    EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES,
    GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS,
    GEOSPATIAL_SUBMISSION_TOKEN_PATTERNS,
    GRAPH_SUBMISSION_ARTIFACT_KEYWORDS,
    GRAPH_SUBMISSION_TOKEN_PATTERNS,
    JSON_TEXT_NOISE_CONTEXT_MARKERS,
    MEDIA_SUBMISSION_ARTIFACT_KEYWORDS,
    MEDIA_SUBMISSION_TOKEN_PATTERNS,
    MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS,
    MEDICAL_SUBMISSION_TOKEN_PATTERNS,
    MISC_SUBMISSION_ARTIFACT_KEYWORD_ALIASES,
    MISC_SUBMISSION_ARTIFACT_KEYWORDS,
    MISC_SUBMISSION_TOKEN_PATTERN_SPECS,
    MODEL_BUNDLE_MARKERS,
    MODEL_SUBMISSION_ARTIFACT_KEYWORDS,
    MODEL_SUBMISSION_TOKEN_PATTERNS,
    MULTI_FILE_SUBMISSION_MARKERS,
    NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES,
    NON_TABULAR_SUBMISSION_SUFFIXES,
    POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS,
    POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS,
    SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS,
    SIGNAL_SUBMISSION_TOKEN_PATTERNS,
    SUBMISSION_ARTIFACT_KEYWORDS,
    SUBMISSION_TOKEN_PATTERN_SPECS,
    SUBMISSION_TOKEN_PATTERNS,
    TABULAR_SUBMISSION_ARTIFACT_KEYWORDS,
    TAR_ARCHIVE_SUBMISSION_SUFFIXES,
    ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES,
    compressed_variant_base_suffixes,
    drop_shadowed_geospatial_container_suffixes,
    drop_shadowed_geospatial_text_suffixes,
    drop_shadowed_graph_xml_suffixes,
    drop_shadowed_json_family_suffixes,
    drop_shadowed_json_lines_suffixes,
    drop_shadowed_model_index_component_suffixes,
    drop_shadowed_plain_tar_suffixes,
    drop_shadowed_submission_suffixes,
    submission_extension_pattern,
)


def test_submission_token_pattern_specs_mirror_compiled_patterns() -> None:
    assert SUBMISSION_TOKEN_PATTERN_SPECS == tuple(
        (pattern.pattern, suffix) for pattern, suffix in SUBMISSION_TOKEN_PATTERNS
    )


def test_submission_token_pattern_specs_are_serializable_primary_definitions() -> None:
    assert SUBMISSION_TOKEN_PATTERN_SPECS
    assert all(
        isinstance(pattern, str) and isinstance(suffix, str) for pattern, suffix in SUBMISSION_TOKEN_PATTERN_SPECS
    )


def test_compression_token_pattern_specs_mirror_compiled_patterns() -> None:
    assert COMPRESSION_TOKEN_PATTERN_SPECS == tuple(
        (pattern.pattern, suffix) for pattern, suffix in COMPRESSION_TOKEN_PATTERNS
    )


def test_code_fence_lang_suffix_specs_build_serializable_mapping() -> None:
    assert CODE_FENCE_LANG_TO_SUFFIX == dict(CODE_FENCE_LANG_SUFFIX_SPECS)
    assert CODE_FENCE_LANG_TO_SUFFIX["csv"] == ".csv"
    assert CODE_FENCE_LANG_TO_SUFFIX["json-lines"] == ".jsonl"
    assert CODE_FENCE_LANG_TO_SUFFIX["nd-json"] == ".ndjson"
    assert CODE_FENCE_LANG_TO_SUFFIX["html"] == ".html"
    assert all(isinstance(lang, str) and isinstance(suffix, str) for lang, suffix in CODE_FENCE_LANG_SUFFIX_SPECS)


def test_archive_submission_extension_pattern_is_registry_backed() -> None:
    assert ARCHIVE_SUBMISSION_SUFFIXES == frozenset(ARCHIVE_SUFFIXES)
    assert ARCHIVE_SUBMISSION_SUFFIXES_ORDERED == tuple(sorted(ARCHIVE_SUBMISSION_SUFFIXES, key=len, reverse=True))
    pattern = re.compile(rf"(?<![A-Za-z0-9_])\.({ARCHIVE_SUBMISSION_EXTENSION_PATTERN})\b", re.I)
    for suffix in ARCHIVE_SUFFIXES:
        assert pattern.search(f"upload `{suffix}`")


def test_archive_submission_suffix_derivatives_are_registry_backed() -> None:
    assert set(EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES) == ARCHIVE_SUBMISSION_SUFFIXES & {".7z", ".rar"}
    assert set(TAR_ARCHIVE_SUBMISSION_SUFFIXES) == ARCHIVE_SUBMISSION_SUFFIXES - {
        ".zip",
        *EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES,
    }
    assert set(ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES) == ARCHIVE_SUBMISSION_SUFFIXES & {".tar.zst", ".tzst"}


def test_submission_extension_pattern_normalizes_escapes_and_prefers_longest_suffixes() -> None:
    assert submission_extension_pattern(["json", ".jsonl", ".tar.gz", "", ".json"]) == r"tar\.gz|jsonl|json"


def test_drop_shadowed_model_index_component_suffixes_prefers_compound_model_suffixes() -> None:
    assert drop_shadowed_model_index_component_suffixes(
        [
            ".safetensors.index.json",
            ".safetensors",
            ".bin.index.json",
            ".bin",
            ".pt",
            ".json",
            ".ckpt.index",
            ".ckpt",
            ".mlmodelc",
            ".mlpackage",
        ]
    ) == [".safetensors.index.json", ".bin.index.json", ".ckpt.index", ".mlmodelc"]
    assert drop_shadowed_model_index_component_suffixes([".json", ".ckpt"]) == [".json", ".ckpt"]


def test_drop_shadowed_plain_tar_suffixes_prefers_compound_tar_suffixes() -> None:
    assert drop_shadowed_plain_tar_suffixes([".tar", ".tar.gz", ".zip"]) == [".tar.gz", ".zip"]
    assert drop_shadowed_plain_tar_suffixes([".tar", ".zip"]) == [".tar", ".zip"]


def test_drop_shadowed_json_lines_suffixes_prefers_named_json_lines_aliases() -> None:
    assert drop_shadowed_json_lines_suffixes([".jsonl", ".ndjson", ".jsonlines.zst", ".csv"]) == [
        ".ndjson",
        ".jsonlines.zst",
        ".csv",
    ]
    assert drop_shadowed_json_lines_suffixes([".jsonl", ".json"]) == [".jsonl", ".json"]


def test_drop_shadowed_json_family_suffixes_prefers_specific_json_variants() -> None:
    assert drop_shadowed_json_family_suffixes([".jsonld", ".json", ".csv"]) == [".jsonld", ".csv"]
    assert drop_shadowed_json_family_suffixes([".geojsonl.gz", ".json", ".geojson"]) == [
        ".geojsonl.gz",
        ".geojson",
    ]
    assert drop_shadowed_json_family_suffixes([".json", ".yaml"]) == [".json", ".yaml"]


def test_drop_shadowed_geospatial_container_suffixes_prefers_kmz_over_zip_and_kml() -> None:
    assert drop_shadowed_geospatial_container_suffixes([".kml", ".kmz", ".zip", ".geojson"]) == [
        ".kmz",
        ".geojson",
    ]
    assert drop_shadowed_geospatial_container_suffixes([".kml", ".zip"]) == [".kml", ".zip"]


def test_drop_shadowed_geospatial_text_suffixes_prefers_specific_geospatial_text_formats() -> None:
    assert drop_shadowed_geospatial_text_suffixes([".geojson", ".geojsonl.gz", ".topojson.zst"]) == [
        ".geojsonl.gz",
        ".topojson.zst",
    ]
    assert drop_shadowed_geospatial_text_suffixes([".xml.xz", ".osm.xz", ".osm.pbf"]) == [".osm.xz", ".osm.pbf"]
    assert drop_shadowed_geospatial_text_suffixes([".geojson", ".kml"]) == [".geojson", ".kml"]


def test_drop_shadowed_graph_xml_suffixes_prefers_graph_specific_xml_formats() -> None:
    assert drop_shadowed_graph_xml_suffixes([".gexf", ".xml", ".json"]) == [".gexf", ".json"]
    assert drop_shadowed_graph_xml_suffixes([".graphml", ".xml"]) == [".graphml"]
    assert drop_shadowed_graph_xml_suffixes([".rdf.gz", ".xml.gz", ".ttl.gz"]) == [".rdf.gz", ".ttl.gz"]
    assert drop_shadowed_graph_xml_suffixes([".xml", ".gml"]) == [".xml", ".gml"]


def test_drop_shadowed_submission_suffixes_applies_all_shadow_rules_in_order() -> None:
    assert drop_shadowed_submission_suffixes(
        [
            ".tar",
            ".tar.zst",
            ".jsonl",
            ".ndjson",
            ".jsonld",
            ".kml",
            ".kmz",
            ".zip",
            ".gexf",
            ".xml",
            ".safetensors.index.json",
            ".safetensors",
            ".json",
            ".mlmodelc",
            ".mlpackage",
        ]
    ) == [".tar.zst", ".ndjson", ".jsonld", ".kmz", ".gexf", ".safetensors.index.json", ".mlmodelc"]


def test_non_tabular_submission_suffixes_cover_shared_single_file_artifact_families() -> None:
    assert (
        IMAGE_SUFFIXES
        | MEDICAL_IMAGE_SUFFIXES
        | AUDIO_SUFFIXES
        | VIDEO_SUFFIXES
        | ARRAY_SUFFIXES
        | SCIENTIFIC_ARRAY_SUFFIXES
        | DIRECTORY_ARRAY_SUFFIXES
        | POINT_CLOUD_SUFFIXES
        | GEOSPATIAL_SUFFIXES
        | DOCUMENT_SUFFIXES
        | BIO_STRUCTURE_SUFFIXES
        | GRAPH_SUFFIXES
        | CODE_SUFFIXES
        | MODEL_ARTIFACT_SUFFIXES
        | MODEL_ARTIFACT_COMPOUND_SUFFIXES
        | MODEL_DIRECTORY_ARTIFACT_SUFFIXES
    ) <= NON_TABULAR_SUBMISSION_SUFFIXES
    assert {".webp", ".nii.gz", ".epub", ".smiles", ".graphml", ".py", ".savedmodel"} & (
        NON_TABULAR_SUBMISSION_SUFFIXES
    )
    assert {".topojson", ".geojsonseq.gz", ".osm.pbf", ".mbtiles", ".pmtiles", ".mvt"} <= (
        GEOSPATIAL_SUFFIXES & NON_TABULAR_SUBMISSION_SUFFIXES
    )


def test_non_directory_non_tabular_submission_suffixes_exclude_directory_artifacts() -> None:
    assert NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES == (
        NON_TABULAR_SUBMISSION_SUFFIXES - MODEL_DIRECTORY_ARTIFACT_SUFFIXES
    )
    assert not (NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES & MODEL_DIRECTORY_ARTIFACT_SUFFIXES)
    assert {".webp", ".nii.gz", ".onnx", ".safetensors.index.json"} <= NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES


def test_model_bundle_markers_cover_model_artifact_suffixes_and_common_bundle_terms() -> None:
    assert {"model weights", "weights", "inference script", "checkpoint"} <= set(MODEL_BUNDLE_MARKERS)
    assert {".onnx", ".keras", ".safetensors", ".savedmodel", ".tfcheckpoint"} <= set(MODEL_BUNDLE_MARKERS)


def test_multi_file_submission_markers_cover_common_archive_layout_terms() -> None:
    assert {
        "zip file containing",
        "one file per",
        "per-image",
        "per sample",
        "masks",
        ".tif",
        ".tiff",
        ".png",
    } <= set(MULTI_FILE_SUBMISSION_MARKERS)


def test_json_text_noise_context_markers_cover_common_non_artifact_mentions() -> None:
    assert {
        "topology json",
        "metadata json",
        "json metadata",
        "description json",
        "json description",
        "text description",
        "sample json",
        "input txt",
    } <= set(JSON_TEXT_NOISE_CONTEXT_MARKERS)


def test_archive_submission_artifact_keywords_are_registry_backed() -> None:
    assert "tarball" in ARCHIVE_SUBMISSION_ARTIFACT_KEYWORDS
    assert "zstd" in ARCHIVE_SUBMISSION_ARTIFACT_KEYWORDS
    for suffix in ARCHIVE_SUFFIXES:
        assert re.escape(suffix.removeprefix(".")) in ARCHIVE_SUBMISSION_ARTIFACT_KEYWORDS
    assert ARCHIVE_SUBMISSION_ARTIFACT_KEYWORDS in SUBMISSION_ARTIFACT_KEYWORDS


def test_tabular_submission_artifact_keywords_are_named_separately() -> None:
    assert "csv" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "parquet" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "jsonlines\\.zst" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "xlsx" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "excel" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "line-delimited\\s+json" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "comma[-\\s]*separated" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "semicolon[-\\s]*delimited" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert "sqlite\\s+(?:database|db|file|format)" in TABULAR_SUBMISSION_ARTIFACT_KEYWORDS
    assert TABULAR_SUBMISSION_ARTIFACT_KEYWORDS in SUBMISSION_ARTIFACT_KEYWORDS


def test_compressible_submission_keyword_suffixes_cover_shared_asset_text_families() -> None:
    assert compressed_variant_base_suffixes(NON_TABULAR_SUBMISSION_SUFFIXES) <= COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES
    assert {
        ".csv",
        ".jsonl",
        ".yaml",
        ".md",
        ".tex",
        ".rst",
        ".adoc",
        ".svg",
        ".fasta",
        ".smiles",
        ".graphml",
        ".edgelist",
        ".jsonld",
        ".ttl",
        ".nt",
        ".nq",
        ".rdf",
        ".owl",
        ".trig",
        ".vcf",
        ".gff3",
        ".ply",
        ".vtk",
        ".kml",
        ".hgt",
        ".fits",
        ".mrc",
        ".dcm",
        ".nrrd",
        ".vhdr",
        ".fif",
    } <= COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES


def test_archive_submission_token_patterns_handle_compressed_tarball_aliases() -> None:
    matched = {
        token: next(suffix for pattern, suffix in SUBMISSION_TOKEN_PATTERNS if pattern.search(token))
        for token in (
            "gzipped tarball",
            "gzip-compressed tar archive",
            "bzip2-compressed tarball",
            "xz-compressed tar archive",
            "zstd-compressed tarball",
            "tarball",
        )
    }

    assert matched["gzipped tarball"] == ".tar.gz"
    assert matched["gzip-compressed tar archive"] == ".tar.gz"
    assert matched["bzip2-compressed tarball"] == ".tar.bz2"
    assert matched["xz-compressed tar archive"] == ".tar.xz"
    assert matched["zstd-compressed tarball"] == ".tar.zst"
    assert matched["tarball"] == ".tar"


def test_submission_token_patterns_detect_delimited_tabular_aliases() -> None:
    matched = {
        token: next(suffix for pattern, suffix in SUBMISSION_TOKEN_PATTERNS if pattern.search(token))
        for token in (
            "comma-separated values",
            "comma delimited",
            "semicolon-separated values",
            "semicolon delimited",
            "pipe separated",
            "tab delimited",
            "SQLite database",
            "SQLite3 database",
            "pickle",
            "pickled",
        )
    }

    assert matched["comma-separated values"] == ".csv"
    assert matched["comma delimited"] == ".csv"
    assert matched["semicolon-separated values"] == ".csv"
    assert matched["semicolon delimited"] == ".csv"
    assert matched["pipe separated"] == ".psv"
    assert matched["tab delimited"] == ".tsv"
    assert matched["SQLite database"] == ".sqlite"
    assert matched["SQLite3 database"] == ".sqlite3"
    assert matched["pickle"] == ".pkl"
    assert matched["pickled"] == ".pkl"


def test_submission_token_patterns_detect_annotation_format_aliases() -> None:
    matched = {
        token: next(suffix for pattern, suffix in SUBMISSION_TOKEN_PATTERNS if pattern.search(token))
        for token in (
            "COCO annotations",
            "COCO JSON",
            "COCO panoptic JSON",
            "COCO keypoints",
            "LabelMe annotations",
            "Label Me JSON",
            "YOLO labels",
            "YOLOv5 labels",
            "YOLOv8 annotation text files",
            "YOLO annotation text files",
            "Pascal VOC annotations",
            "Pascal VOC XML files",
            "VOC annotation XML files",
            "Open Images annotations",
            "Open Images CSV files",
            "RLE masks",
            "run-length encoded masks",
            "run length encoding masks",
        )
    }

    assert matched["COCO annotations"] == ".json"
    assert matched["COCO JSON"] == ".json"
    assert matched["COCO panoptic JSON"] == ".json"
    assert matched["COCO keypoints"] == ".json"
    assert matched["LabelMe annotations"] == ".json"
    assert matched["Label Me JSON"] == ".json"
    assert matched["YOLO labels"] == ".txt"
    assert matched["YOLOv5 labels"] == ".txt"
    assert matched["YOLOv8 annotation text files"] == ".txt"
    assert matched["YOLO annotation text files"] == ".txt"
    assert matched["Pascal VOC annotations"] == ".xml"
    assert matched["Pascal VOC XML files"] == ".xml"
    assert matched["VOC annotation XML files"] == ".xml"
    assert matched["Open Images annotations"] == ".csv"
    assert matched["Open Images CSV files"] == ".csv"
    assert matched["RLE masks"] == ".csv"
    assert matched["run-length encoded masks"] == ".csv"
    assert matched["run length encoding masks"] == ".csv"


def test_misc_submission_artifact_keywords_are_named_aliases() -> None:
    assert {
        "coco",
        "coco\\s+(?:caption|captions|keypoint|keypoints|panoptic|segmentation|segmentations)",
        "labelme",
        "yolo",
        "yolov\\d+",
        "pascal\\s+voc",
        "open\\s+images?",
        "rle",
    } <= set(MISC_SUBMISSION_ARTIFACT_KEYWORD_ALIASES)
    assert "open\\s+images?" in MISC_SUBMISSION_ARTIFACT_KEYWORDS
    assert "pascal\\s+voc" in MISC_SUBMISSION_ARTIFACT_KEYWORDS
    assert "rle" in MISC_SUBMISSION_ARTIFACT_KEYWORDS
    assert MISC_SUBMISSION_ARTIFACT_KEYWORDS in SUBMISSION_ARTIFACT_KEYWORDS
    assert all(
        isinstance(pattern, str) and isinstance(suffix, str) for pattern, suffix in MISC_SUBMISSION_TOKEN_PATTERN_SPECS
    )


def test_media_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in MEDIA_SUBMISSION_TOKEN_PATTERNS}

    assert ".avif" in suffixes
    assert ".jp2" in suffixes
    assert ".jxl" in suffixes
    assert ".exr" in suffixes
    assert ".ppm" in suffixes
    assert ".flac" in suffixes
    assert ".svg.gz" in suffixes
    assert ".svgz" in suffixes
    assert ".mid" in suffixes
    assert ".opus" in suffixes
    assert ".mkv" in suffixes
    assert ".mp4" in suffixes
    assert ".mpg" in suffixes
    assert ".webm" in suffixes
    assert (IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES) - {
        ".aif",
        ".jpeg",
        ".midi",
        ".mpeg",
        ".svg",
        ".tiff",
    } <= suffixes


def test_media_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: next(suffix for pattern, suffix in MEDIA_SUBMISSION_TOKEN_PATTERNS if pattern.search(token))
        for token in (
            "jpeg xl",
            "JPEG 2000",
            "jp2",
            "openexr",
            "netpbm",
            "windows bitmap",
            "portable network graphics",
            "tif",
            "tiff",
            "jpg",
            "jpeg",
            "MPEG-4 video",
            "mp4",
            "mpeg",
            "QuickTime movie",
            "Matroska",
            "Flash Video",
            "Audio Video Interleave",
            "MPEG-4 audio",
            "Advanced Audio Coding",
            "Free Lossless Audio Codec",
            "Ogg Vorbis",
            "gzip-compressed svg",
            "zstd-compressed scalable vector graphics",
            "svgz",
            "scalable vector graphics",
            "svg",
            "aif",
            "aiff",
            "midi",
            "windows media audio",
        )
    }

    assert matched["jpeg xl"] == ".jxl"
    assert matched["JPEG 2000"] == ".jp2"
    assert matched["jp2"] == ".jp2"
    assert matched["openexr"] == ".exr"
    assert matched["netpbm"] == ".pnm"
    assert matched["windows bitmap"] == ".bmp"
    assert matched["portable network graphics"] == ".png"
    assert matched["tif"] == ".tif"
    assert matched["tiff"] == ".tif"
    assert matched["jpg"] == ".jpg"
    assert matched["jpeg"] == ".jpg"
    assert matched["MPEG-4 video"] == ".mp4"
    assert matched["mp4"] == ".mp4"
    assert matched["mpeg"] == ".mpg"
    assert matched["QuickTime movie"] == ".mov"
    assert matched["Matroska"] == ".mkv"
    assert matched["Flash Video"] == ".flv"
    assert matched["Audio Video Interleave"] == ".avi"
    assert matched["MPEG-4 audio"] == ".m4a"
    assert matched["Advanced Audio Coding"] == ".aac"
    assert matched["Free Lossless Audio Codec"] == ".flac"
    assert matched["Ogg Vorbis"] == ".ogg"
    assert matched["gzip-compressed svg"] == ".svg.gz"
    assert matched["zstd-compressed scalable vector graphics"] == ".svg.zst"
    assert matched["svgz"] == ".svgz"
    assert matched["scalable vector graphics"] == ".svg"
    assert matched["svg"] == ".svg"
    assert matched["aif"] == ".aiff"
    assert matched["aiff"] == ".aiff"
    assert matched["midi"] == ".mid"
    assert matched["windows media audio"] == ".wma"


def test_media_submission_artifact_keywords_are_registry_backed() -> None:
    assert "avif" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "jpeg\\s*2000|jp2" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "jpeg\\s+xl" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "open\\s*exr" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "matroska|mkv" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "free\\s+lossless\\s+audio\\s+codec|flac" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "opus" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "(?:gzip|gzipped|gzip[-\\s]*compressed)\\s+(?:svg|scalable\\s+vector\\s+graphics)" in (
        MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "scalable\\s+vector\\s+graphics|svg" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "svgz" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "mpg" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS
    assert "webm" in MEDIA_SUBMISSION_ARTIFACT_KEYWORDS


def test_document_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in DOCUMENT_SUBMISSION_TOKEN_PATTERNS}

    assert ".pdf" in suffixes
    assert ".epub" in suffixes
    assert ".odt" in suffixes
    assert ".pptx" in suffixes
    assert ".md.gz" in suffixes
    assert ".tex.zst" in suffixes
    assert ".vtt.zst" in suffixes
    assert (
        DOCUMENT_SUFFIXES
        - {
            ".adoc",
            ".docx",
            ".epub",
            ".htm",
            ".html",
            ".md",
            ".odp",
            ".odt",
            ".pptx",
            ".rst",
            ".rtf",
            ".srt",
            ".tex",
            ".vtt",
        }
        <= suffixes
    )


def test_document_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "EPUB",
            "word document",
            "microsoft word",
            "OpenDocument text",
            "PowerPoint",
            "LaTeX",
            "reStructuredText",
            "AsciiDoc",
            "rich text",
            "markdown",
            "webvtt",
            "subrip",
        )
        for pattern, suffix in DOCUMENT_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["EPUB"] == ".epub"
    assert matched["word document"] == ".docx"
    assert matched["microsoft word"] == ".docx"
    assert matched["OpenDocument text"] == ".odt"
    assert matched["PowerPoint"] == ".pptx"
    assert matched["LaTeX"] == ".tex"
    assert matched["reStructuredText"] == ".rst"
    assert matched["AsciiDoc"] == ".adoc"
    assert matched["rich text"] == ".rtf"
    assert matched["markdown"] == ".md"
    assert matched["webvtt"] == ".vtt"
    assert matched["subrip"] == ".srt"


def test_document_submission_artifact_keywords_are_registry_backed() -> None:
    assert "pdf" in DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS
    assert "epub" in DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS
    assert "power\\s*point|microsoft\\s+power\\s*point" in DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS
    assert "markdown" in DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS
    assert "tex\\.zst" in DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS
    assert "vtt\\.zst" in DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS


def test_medical_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in MEDICAL_SUBMISSION_TOKEN_PATTERNS}

    assert ".ima" in suffixes
    assert ".mrc.gz" in suffixes
    assert ".mrcs.zst" in suffixes
    assert ".nrrd.zst" in suffixes
    assert ".qptiff" in suffixes
    assert (
        MEDICAL_IMAGE_SUFFIXES
        - {
            ".ccp4",
            ".dcm",
            ".dicom",
            ".dm3",
            ".dm4",
            ".lif",
            ".lsm",
            ".mrc",
            ".mrcs",
            ".nd2",
            ".nii.gz",
            ".ome.tiff",
        }
        <= suffixes
    )


def test_medical_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "nifti",
            "nii.gz",
            "dicom",
            "dcm",
            "cryo-EM map",
            "mrc",
            "MRCS stack",
            "CCP4 map",
            "Nikon ND2",
            "Leica LIF",
            "Zeiss LSM",
            "Gatan DM3",
            "Gatan DM4",
            "ome.tif",
            "ome.tiff",
            "MetaImage",
            "NRRD",
            "Nearly Raw Raster Data",
            "Analyze 7.5",
        )
        for pattern, suffix in MEDICAL_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["nifti"] == ".nii.gz"
    assert matched["nii.gz"] == ".nii.gz"
    assert matched["dicom"] == ".dcm"
    assert matched["dcm"] == ".dcm"
    assert matched["cryo-EM map"] == ".mrc"
    assert matched["mrc"] == ".mrc"
    assert matched["MRCS stack"] == ".mrcs"
    assert matched["CCP4 map"] == ".ccp4"
    assert matched["Nikon ND2"] == ".nd2"
    assert matched["Leica LIF"] == ".lif"
    assert matched["Zeiss LSM"] == ".lsm"
    assert matched["Gatan DM3"] == ".dm3"
    assert matched["Gatan DM4"] == ".dm4"
    assert matched["ome.tif"] == ".ome.tif"
    assert matched["ome.tiff"] == ".ome.tif"
    assert matched["MetaImage"] == ".mha"
    assert matched["NRRD"] == ".nrrd"
    assert matched["Nearly Raw Raster Data"] == ".nrrd"
    assert matched["Analyze 7.5"] == ".hdr"


def test_medical_submission_artifact_keywords_are_registry_backed() -> None:
    assert "nifti" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "ima" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "mrc\\.gz" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "mrcs\\.zst" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "nikon\\s+nd2|nd2" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "gatan\\s+dm4|dm4" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "nrrd\\.zst" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "meta\\s*image|metaimage" in MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS


def test_point_cloud_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS}

    assert ".abc" in suffixes
    assert ".blend" in suffixes
    assert ".e57" in suffixes
    assert ".ply.gz" in suffixes
    assert ".vtk.gz" in suffixes
    assert ".vtu.zst" in suffixes
    assert ".step.gz" in suffixes
    assert ".ifc.zst" in suffixes
    assert ".gltf" in suffixes
    assert ".fbx" in suffixes
    assert ".usd" in suffixes
    assert ".usda.gz" in suffixes
    assert ".usdc" in suffixes
    assert ".usdz" in suffixes
    assert POINT_CLOUD_SUFFIXES - {".off"} <= suffixes


def test_point_cloud_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "Wavefront OBJ",
            "OBJ mesh",
            "Alembic scene",
            "ABC mesh",
            "Blender scene",
            "BLEND file",
            "COLLADA",
            "DAE scene",
            "binary glTF",
            "GLB scene",
            "USD scene",
            "USDA mesh",
            "USDC file",
            "USDZ archive",
            "VTK legacy",
            "VTK PolyData",
            "VTK unstructured grid",
            "Gmsh",
            "Medit mesh file",
            ".mesh",
            "Autodesk FBX",
            "FBX scene",
            "3D Studio",
            "3DS mesh",
            ".step",
            "STEP CAD file",
            ".stp",
            "STP geometry format",
            "IGES CAD model",
            "Initial Graphics Exchange Specification",
            "IGS geometry file",
            "IFC BIM model",
            "Industry Foundation Classes",
            "BREP geometry file",
            "boundary representation",
            ".off",
            "off mesh",
            "off file",
            "off format",
        )
        for pattern, suffix in POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["Wavefront OBJ"] == ".obj"
    assert matched["OBJ mesh"] == ".obj"
    assert matched["Alembic scene"] == ".abc"
    assert matched["ABC mesh"] == ".abc"
    assert matched["Blender scene"] == ".blend"
    assert matched["BLEND file"] == ".blend"
    assert matched["COLLADA"] == ".dae"
    assert matched["DAE scene"] == ".dae"
    assert matched["binary glTF"] == ".glb"
    assert matched["GLB scene"] == ".glb"
    assert matched["USD scene"] == ".usd"
    assert matched["USDA mesh"] == ".usda"
    assert matched["USDC file"] == ".usdc"
    assert matched["USDZ archive"] == ".usdz"
    assert matched["VTK legacy"] == ".vtk"
    assert matched["VTK PolyData"] == ".vtp"
    assert matched["VTK unstructured grid"] == ".vtu"
    assert matched["Gmsh"] == ".msh"
    assert matched["Medit mesh file"] == ".mesh"
    assert matched[".mesh"] == ".mesh"
    assert matched["Autodesk FBX"] == ".fbx"
    assert matched["FBX scene"] == ".fbx"
    assert matched["3D Studio"] == ".3ds"
    assert matched["3DS mesh"] == ".3ds"
    assert matched[".step"] == ".step"
    assert matched["STEP CAD file"] == ".step"
    assert matched[".stp"] == ".stp"
    assert matched["STP geometry format"] == ".stp"
    assert matched["IGES CAD model"] == ".iges"
    assert matched["Initial Graphics Exchange Specification"] == ".iges"
    assert matched["IGS geometry file"] == ".igs"
    assert matched["IFC BIM model"] == ".ifc"
    assert matched["Industry Foundation Classes"] == ".ifc"
    assert matched["BREP geometry file"] == ".brep"
    assert matched["boundary representation"] == ".brep"
    assert matched[".off"] == ".off"
    assert matched["off mesh"] == ".off"
    assert matched["off file"] == ".off"
    assert matched["off format"] == ".off"
    assert all(not pattern.search("turn off submissions") for pattern, _suffix in POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS)
    assert all(
        not pattern.search("mesh together predictions") for pattern, _suffix in POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS
    )
    assert all(
        not pattern.search("next step is validation") for pattern, _suffix in POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS
    )


def test_point_cloud_submission_artifact_keywords_are_registry_backed() -> None:
    assert "abc" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "alembic" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "blend" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "blender" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "dae" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "e57" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "ply\\.gz" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "vtk" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "gmsh" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "fbx" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "3d\\s+studio" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "step\\s+(?:cad|model|geometry|file|format)" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "industry\\s+foundation\\s+classes" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "boundary\\s+representation" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "x3d\\.zst" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "usda\\.gz" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "usdz" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "gltf" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS
    assert "wavefront\\s+obj" in POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS


def test_geospatial_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in GEOSPATIAL_SUBMISSION_TOKEN_PATTERNS}

    assert ".geojson.zst" in suffixes
    assert ".geojsonl.gz" in suffixes
    assert ".geojsonseq.zst" in suffixes
    assert ".topojson.zst" in suffixes
    assert ".osm.pbf" in suffixes
    assert ".mbtiles" in suffixes
    assert ".pmtiles" in suffixes
    assert ".mvt" in suffixes
    assert ".hgt" in suffixes
    assert ".dem.gz" in suffixes
    assert ".dt2.zst" in suffixes
    assert ".ecw" in suffixes
    assert ".sid" in suffixes
    assert ".kml.gz" in suffixes
    assert ".kmz" in suffixes
    assert GEOSPATIAL_SUFFIXES - {".geopackage", ".gpkg", ".shp"} <= suffixes


def test_geospatial_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "cloud optimized geotiff",
            "cog",
            "geotiff",
            "geo tiff",
            "ESRI shapefile",
            "shapefile",
            "shp",
            "geopackage",
            "gpkg",
            "GeoJSON Lines",
            "GeoJSON text sequence",
            "TopoJSON",
            "MBTiles",
            "PMTiles",
            "Mapbox vector tile",
            "OSM PBF",
            "OpenStreetMap XML",
            "Keyhole Markup Language",
            "KML",
            "zipped KML",
            "KMZ",
            "GDAL VRT",
            "virtual raster",
            "SRTM HGT",
            "HGT elevation",
            "DTED level 0",
            "DTED level 1",
            "DTED level 2",
            "digital elevation model",
            "DEM raster",
            "enhanced compression wavelet",
            "ECW raster",
            "MrSID",
            "SID raster",
            "MapInfo MIF/MID pair",
            "MIF",
            "ENVI raster header",
            "Band Interleaved by Line",
            "Band Interleaved by Pixel",
            "Band Sequential",
        )
        for pattern, suffix in GEOSPATIAL_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["cloud optimized geotiff"] == ".tif"
    assert matched["cog"] == ".tif"
    assert matched["geotiff"] == ".tif"
    assert matched["geo tiff"] == ".tif"
    assert matched["ESRI shapefile"] == ".shp"
    assert matched["shapefile"] == ".shp"
    assert matched["shp"] == ".shp"
    assert matched["geopackage"] == ".gpkg"
    assert matched["gpkg"] == ".gpkg"
    assert matched["GeoJSON Lines"] == ".geojsonl"
    assert matched["GeoJSON text sequence"] == ".geojsonseq"
    assert matched["TopoJSON"] == ".topojson"
    assert matched["MBTiles"] == ".mbtiles"
    assert matched["PMTiles"] == ".pmtiles"
    assert matched["Mapbox vector tile"] == ".mvt"
    assert matched["OSM PBF"] == ".osm.pbf"
    assert matched["OpenStreetMap XML"] == ".osm"
    assert matched["Keyhole Markup Language"] == ".kml"
    assert matched["KML"] == ".kml"
    assert matched["zipped KML"] == ".kmz"
    assert matched["KMZ"] == ".kmz"
    assert matched["GDAL VRT"] == ".vrt"
    assert matched["virtual raster"] == ".vrt"
    assert matched["SRTM HGT"] == ".hgt"
    assert matched["HGT elevation"] == ".hgt"
    assert matched["DTED level 0"] == ".dt0"
    assert matched["DTED level 1"] == ".dt1"
    assert matched["DTED level 2"] == ".dt2"
    assert matched["digital elevation model"] == ".dem"
    assert matched["DEM raster"] == ".dem"
    assert matched["enhanced compression wavelet"] == ".ecw"
    assert matched["ECW raster"] == ".ecw"
    assert matched["MrSID"] == ".sid"
    assert matched["SID raster"] == ".sid"
    assert matched["MapInfo MIF/MID pair"] == ".mif"
    assert matched["MIF"] == ".mif"
    assert matched["ENVI raster header"] == ".hdr"
    assert matched["Band Interleaved by Line"] == ".bil"
    assert matched["Band Interleaved by Pixel"] == ".bip"
    assert matched["Band Sequential"] == ".bsq"


def test_geospatial_submission_artifact_keywords_are_registry_backed() -> None:
    assert "geojson\\.zst" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "geojsonseq\\.zst" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "topojson\\.zst" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "osm\\.pbf" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "mbtiles" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "pmtiles" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "kml\\.gz" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "geopackage" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "cloud\\s+optimized\\s+geotiff" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "keyhole\\s+markup\\s+language|(?<!zipped\\s)kml" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "gdal\\s+vrt|virtual\\s+raster|vrt" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "srtm(?:\\s+(?:hgt|elevation|dem))?|hgt(?:\\s+(?:elevation|raster|file|format))?" in (
        GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "dem\\.gz" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "dt2\\.zst" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "mrsid|mr\\s*sid|sid(?:\\s+(?:raster|image|file|format))?" in (GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS)
    assert "mapinfo\\s+(?:interchange|mif(?:/mid)?(?:\\s+pair)?)|mif/mid|mif" in (
        GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "envi(?:\\s+raster)?\\s+header" in GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS


def test_bio_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in BIO_SUBMISSION_TOKEN_PATTERNS}

    assert ".mmcif" in suffixes
    assert ".vcf.gz" in suffixes
    assert ".sam.zst" in suffixes
    assert ".bam" in suffixes
    assert ".cram" in suffixes
    assert ".fasta.gz" in suffixes
    assert ".fastq.zst" in suffixes
    assert ".smiles.gz" in suffixes
    assert ".selfies.zst" in suffixes
    assert (
        BIO_STRUCTURE_SUFFIXES
        - {
            ".cif",
            ".fa",
            ".fq",
            ".inchi",
            ".mol",
            ".pdb",
            ".rxn",
            ".sdf",
            ".selfies",
            ".smi",
        }
        <= suffixes
    )


def test_bio_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "variant call format",
            "VCF",
            "binary alignment map",
            "BAM",
            "sequence alignment map",
            "SAM",
            "compressed reference-oriented alignment map",
            "CRAM",
            "general feature format 3",
            "GFF3",
            "gene transfer format",
            "GTF",
            "browser extensible data",
            "BED",
            "bigWig",
            "bigBed",
            "POD5",
            "protein data bank",
            "pdb",
            "macromolecular crystallographic information",
            "mmCIF",
            "crystallographic information",
            "cif",
            "structure data file",
            "sdf",
            "Tripos MOL2",
            "mol2",
            "MDL molfile",
            "molfile",
            "mol",
            "FASTA",
            "FA sequence",
            "FASTQ",
            "FQ reads",
            "Simplified Molecular Input Line Entry System",
            "SMI",
            "SMILES",
            "InChI",
            "SELFIES",
            "reaction file",
            "rxn",
        )
        for pattern, suffix in BIO_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["variant call format"] == ".vcf"
    assert matched["VCF"] == ".vcf"
    assert matched["binary alignment map"] == ".bam"
    assert matched["BAM"] == ".bam"
    assert matched["sequence alignment map"] == ".sam"
    assert matched["SAM"] == ".sam"
    assert matched["compressed reference-oriented alignment map"] == ".cram"
    assert matched["CRAM"] == ".cram"
    assert matched["general feature format 3"] == ".gff3"
    assert matched["GFF3"] == ".gff3"
    assert matched["gene transfer format"] == ".gtf"
    assert matched["GTF"] == ".gtf"
    assert matched["browser extensible data"] == ".bed"
    assert matched["BED"] == ".bed"
    assert matched["bigWig"] == ".bigwig"
    assert matched["bigBed"] == ".bigbed"
    assert matched["POD5"] == ".pod5"
    assert matched["protein data bank"] == ".pdb"
    assert matched["pdb"] == ".pdb"
    assert matched["macromolecular crystallographic information"] == ".mmcif"
    assert matched["mmCIF"] == ".mmcif"
    assert matched["crystallographic information"] == ".cif"
    assert matched["cif"] == ".cif"
    assert matched["structure data file"] == ".sdf"
    assert matched["sdf"] == ".sdf"
    assert matched["Tripos MOL2"] == ".mol2"
    assert matched["mol2"] == ".mol2"
    assert matched["MDL molfile"] == ".mol"
    assert matched["molfile"] == ".mol"
    assert matched["mol"] == ".mol"
    assert matched["FASTA"] == ".fasta"
    assert matched["FA sequence"] == ".fasta"
    assert matched["FASTQ"] == ".fastq"
    assert matched["FQ reads"] == ".fastq"
    assert matched["Simplified Molecular Input Line Entry System"] == ".smiles"
    assert matched["SMI"] == ".smiles"
    assert matched["SMILES"] == ".smiles"
    assert matched["InChI"] == ".inchi"
    assert matched["SELFIES"] == ".selfies"
    assert matched["reaction file"] == ".rxn"
    assert matched["rxn"] == ".rxn"


def test_bio_submission_artifact_keywords_are_registry_backed() -> None:
    assert "variant\\s+call\\s+format|vcf" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "binary\\s+alignment\\s+map|bam" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "gff3\\.zst" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "vcf\\.gz" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "bigwig" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "pod5" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "protein\\s+data\\s+bank" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "macromolecular\\s+crystallographic\\s+information|mmcif" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "tripos\\s+mol2|mol2" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "fastq|fq(?:\\s+(?:reads?|sequences?|file|format))?" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "fasta|fa(?:\\s+(?:sequences?|file|format))?" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "smiles" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "simplified\\s+molecular\\s+input\\s+line\\s+entry\\s+system|smiles|smi" in (
        BIO_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "inchi" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "fastq\\.zst" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "smiles\\.gz" in BIO_SUBMISSION_ARTIFACT_KEYWORDS
    assert "mmcif" in BIO_SUBMISSION_ARTIFACT_KEYWORDS


def test_graph_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in GRAPH_SUBMISSION_TOKEN_PATTERNS}

    assert ".graphml" in suffixes
    assert ".graphml.gz" in suffixes
    assert ".gexf" in suffixes
    assert ".gexf.zst" in suffixes
    assert ".gml" in suffixes
    assert ".ttl.gz" in suffixes
    assert ".jsonld.zst" in suffixes
    assert (
        GRAPH_SUFFIXES
        - {
            ".edges",
            ".edgelist",
            ".gexf",
            ".gml",
            ".graphml",
            ".jsonld",
            ".mtx",
            ".nq",
            ".nt",
            ".owl",
            ".rdf",
            ".trig",
            ".ttl",
        }
        <= suffixes
    )


def test_graph_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "Graph Markup Language",
            "GraphML",
            "Graph Exchange XML Format",
            "GEXF",
            "Graph Modeling Language",
            "Graph Modelling Language",
            "GML",
            "matrix market",
            "mtx",
            "JSON-LD",
            "linked data JSON",
            "Turtle RDF",
            "TTL RDF",
            "N-Triples",
            "NT RDF",
            "N-Quads",
            "NQ RDF",
            "RDF XML",
            "OWL ontology",
            "TriG RDF",
            "edge list",
            "edgelist",
            "edges",
        )
        for pattern, suffix in GRAPH_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["Graph Markup Language"] == ".graphml"
    assert matched["GraphML"] == ".graphml"
    assert matched["Graph Exchange XML Format"] == ".gexf"
    assert matched["GEXF"] == ".gexf"
    assert matched["Graph Modeling Language"] == ".gml"
    assert matched["Graph Modelling Language"] == ".gml"
    assert matched["GML"] == ".gml"
    assert matched["matrix market"] == ".mtx"
    assert matched["mtx"] == ".mtx"
    assert matched["JSON-LD"] == ".jsonld"
    assert matched["linked data JSON"] == ".jsonld"
    assert matched["Turtle RDF"] == ".ttl"
    assert matched["TTL RDF"] == ".ttl"
    assert matched["N-Triples"] == ".nt"
    assert matched["NT RDF"] == ".nt"
    assert matched["N-Quads"] == ".nq"
    assert matched["NQ RDF"] == ".nq"
    assert matched["RDF XML"] == ".rdf"
    assert matched["OWL ontology"] == ".owl"
    assert matched["TriG RDF"] == ".trig"
    assert matched["edge list"] == ".edgelist"
    assert matched["edgelist"] == ".edgelist"
    assert matched["edges"] == ".edges"


def test_graph_submission_artifact_keywords_are_registry_backed() -> None:
    assert "graphml\\.gz" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "gexf\\.zst" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "graph\\s+markup\\s+language|graphml" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "graph\\s+exchange\\s+xml\\s+format|gexf" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "graph\\s+model(?:l)?ing\\s+language|gml" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "matrix\\s+market" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "json[-\\s]*ld|linked\\s+data\\s+json" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "ttl\\.gz" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "jsonld\\.zst" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS
    assert "n[-\\s]*triples?|nt(?:\\s+(?:rdf|file|format))?" in GRAPH_SUBMISSION_ARTIFACT_KEYWORDS


def test_array_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in ARRAY_SUBMISSION_TOKEN_PATTERNS}

    assert ".fits.gz" in suffixes
    assert ".fts.zst" in suffixes
    assert ".nc4" in suffixes
    assert ".cdf" in suffixes
    assert ".loom" in suffixes
    assert ".n5" in suffixes
    assert ".ome.zarr" in suffixes
    assert ".zarr" in suffixes
    assert (ARRAY_SUFFIXES | SCIENTIFIC_ARRAY_SUFFIXES | DIRECTORY_ARRAY_SUFFIXES) - {
        ".cdf",
        ".fit",
        ".fits",
        ".grib",
        ".grib2",
        ".h5ad",
        ".loom",
        ".mat",
        ".nc",
        ".nc4",
        ".n5",
        ".ome.zarr",
        ".zarr",
    } <= suffixes


def test_array_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "netcdf",
            "NetCDF-4",
            "netcdf4",
            "nc4",
            "Common Data Format",
            "CDF",
            "grib",
            "grib2",
            "fits",
            "fit file",
            "fit format",
            "flexible image transport",
            "anndata",
            "h5ad",
            "loom",
            "loom file",
            "N5 store",
            "OME-Zarr store",
            "zarr",
            "zarr store",
            "matlab",
            "mat file",
            "compressed numpy array",
            "numpy archive",
            "numpy zip archive",
            "scipy sparse matrix",
            "numpy array",
        )
        for pattern, suffix in ARRAY_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["netcdf"] == ".nc"
    assert matched["NetCDF-4"] == ".nc4"
    assert matched["netcdf4"] == ".nc4"
    assert matched["nc4"] == ".nc4"
    assert matched["Common Data Format"] == ".cdf"
    assert matched["CDF"] == ".cdf"
    assert matched["grib"] == ".grib"
    assert matched["grib2"] == ".grib2"
    assert matched["fits"] == ".fits"
    assert matched["fit file"] == ".fit"
    assert matched["fit format"] == ".fit"
    assert matched["flexible image transport"] == ".fits"
    assert matched["anndata"] == ".h5ad"
    assert matched["h5ad"] == ".h5ad"
    assert matched["loom"] == ".loom"
    assert matched["loom file"] == ".loom"
    assert matched["N5 store"] == ".n5"
    assert matched["OME-Zarr store"] == ".ome.zarr"
    assert matched["zarr"] == ".zarr"
    assert matched["zarr store"] == ".zarr"
    assert matched["matlab"] == ".mat"
    assert matched["mat file"] == ".mat"
    assert matched["compressed numpy array"] == ".npz"
    assert matched["numpy archive"] == ".npz"
    assert matched["numpy zip archive"] == ".npz"
    assert matched["scipy sparse matrix"] == ".npz"
    assert matched["numpy array"] == ".npy"
    assert all(not pattern.search("fit a model") for pattern, _suffix in ARRAY_SUBMISSION_TOKEN_PATTERNS)


def test_array_submission_artifact_keywords_are_registry_backed() -> None:
    assert "npy" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "npz" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "numpy\\s+(?:zip\\s+)?archive" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "fits\\.gz" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "netcdf[-\\s]*4|netcdf4|nc4" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "common\\s+data\\s+format|cdf" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "loom(?:\\s+(?:file|format))?" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "ome[-\\s]*zarr(?:\\s+(?:store|directory|group|archive|format))?" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "(?<!ome[-\\s])zarr(?:\\s+(?:store|directory|group|archive|format))?" in (ARRAY_SUBMISSION_ARTIFACT_KEYWORDS)
    assert "\\bn5(?:\\s+(?:store|directory|group|archive|format))?\\b" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "fts\\.zst" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "netcdf" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert "scipy\\s+sparse\\s+matrix" in ARRAY_SUBMISSION_ARTIFACT_KEYWORDS
    assert ARRAY_SUBMISSION_ARTIFACT_KEYWORDS in SUBMISSION_ARTIFACT_KEYWORDS


def test_signal_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in SIGNAL_SUBMISSION_TOKEN_PATTERNS}

    assert ".hea.gz" in suffixes
    assert ".fif.gz" in suffixes
    assert ".vhdr.zst" in suffixes
    assert ".set.xz" in suffixes
    assert ".nwb" in suffixes
    assert (
        SIGNAL_SUFFIXES
        - {
            ".abf",
            ".bdf",
            ".cnt",
            ".edf",
            ".eeg",
            ".fdt",
            ".fif",
            ".hea",
            ".nwb",
            ".set",
            ".tdms",
            ".trc",
            ".vhdr",
            ".vmrk",
        }
        <= suffixes
    )


def test_signal_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: next(suffix for pattern, suffix in SIGNAL_SUBMISSION_TOKEN_PATTERNS if pattern.search(token))
        for token in (
            "European Data Format",
            "European Data Format Plus",
            "EDF+",
            "BioSemi Data Format",
            "Biomedical Data Format",
            "WaveForm DataBase",
            "WFDB header",
            "PhysioNet header",
            "Neurodata Without Borders",
            "Neurodata Without Borders Neurophysiology",
            "National Instruments Technical Data Management Streaming",
            "Technical Data Management Streaming",
            "Axon Binary File",
            "Axon Binary Format",
            "BrainVision header",
            "VHDR header",
            "BrainVision marker",
            "VMRK marker",
            "BrainVision EEG",
            "EEGLAB set",
            "EEGLAB FDT",
            "MNE FIF",
            "FIF EEG",
            "Neuroscan CNT",
            "Micromed TRC",
        )
    }

    assert matched["European Data Format"] == ".edf"
    assert matched["European Data Format Plus"] == ".edf"
    assert matched["EDF+"] == ".edf"
    assert matched["BioSemi Data Format"] == ".bdf"
    assert matched["Biomedical Data Format"] == ".bdf"
    assert matched["WaveForm DataBase"] == ".hea"
    assert matched["WFDB header"] == ".hea"
    assert matched["PhysioNet header"] == ".hea"
    assert matched["Neurodata Without Borders"] == ".nwb"
    assert matched["Neurodata Without Borders Neurophysiology"] == ".nwb"
    assert matched["National Instruments Technical Data Management Streaming"] == ".tdms"
    assert matched["Technical Data Management Streaming"] == ".tdms"
    assert matched["Axon Binary File"] == ".abf"
    assert matched["Axon Binary Format"] == ".abf"
    assert matched["BrainVision header"] == ".vhdr"
    assert matched["VHDR header"] == ".vhdr"
    assert matched["BrainVision marker"] == ".vmrk"
    assert matched["VMRK marker"] == ".vmrk"
    assert matched["BrainVision EEG"] == ".eeg"
    assert matched["EEGLAB set"] == ".set"
    assert matched["EEGLAB FDT"] == ".fdt"
    assert matched["MNE FIF"] == ".fif"
    assert matched["FIF EEG"] == ".fif"
    assert matched["Neuroscan CNT"] == ".cnt"
    assert matched["Micromed TRC"] == ".trc"


def test_signal_submission_artifact_keywords_are_registry_backed() -> None:
    assert "european\\s+data\\s+format(?:\\s+plus)?|edf\\+?" in SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "waveform\\s+database|wfdb(?:\\s+header)?|physionet\\s+header|ecg\\s+header" in (
        SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "(?:national\\s+instruments\\s+)?technical\\s+data\\s+management\\s+streaming|tdms" in (
        SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "brainvision\\s+header|vhdr(?:\\s+(?:header|file|format))?" in SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "eeglab(?:\\s+(?:set|dataset|file|format))?|set\\s+(?:eeg|file|format)" in (
        SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "fif\\.gz" in SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "vhdr\\.zst" in SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "hea\\.gz" in SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS


def test_code_submission_token_patterns_cover_single_file_code_artifacts() -> None:
    suffixes = {suffix for _pattern, suffix in CODE_SUBMISSION_TOKEN_PATTERNS}
    matched = {
        token: suffix
        for token in (
            "Python script",
            "Python source code",
            "Jupyter notebook",
            "IPython notebook",
            "R script",
            "Julia script",
        )
        for pattern, suffix in CODE_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert CODE_SUFFIXES <= suffixes
    assert matched == {
        "Python script": ".py",
        "Python source code": ".py",
        "Jupyter notebook": ".ipynb",
        "IPython notebook": ".ipynb",
        "R script": ".r",
        "Julia script": ".jl",
    }


def test_code_submission_artifact_keywords_are_registry_backed() -> None:
    assert "python\\s+(?:script|file|source|code)" in CODE_SUBMISSION_ARTIFACT_KEYWORDS
    assert "jupyter\\s+notebook|ipython\\s+notebook|notebook\\s+(?:file|source)" in (CODE_SUBMISSION_ARTIFACT_KEYWORDS)
    assert CODE_SUBMISSION_ARTIFACT_KEYWORDS in SUBMISSION_ARTIFACT_KEYWORDS


def test_model_submission_token_patterns_are_registry_backed() -> None:
    suffixes = {suffix for _pattern, suffix in MODEL_SUBMISSION_TOKEN_PATTERNS}

    assert ".onnx" in suffixes
    assert ".gguf" in suffixes
    assert ".joblib" in suffixes
    assert ".engine" in suffixes
    assert ".plan" in suffixes
    assert ".rknn" in suffixes
    assert ".pt" in suffixes
    assert ".pth" in suffixes
    assert ".mlmodel" in suffixes
    assert ".safetensors.index.json" in suffixes
    assert (
        MODEL_ARTIFACT_SUFFIXES
        - {
            ".bst",
            ".blob",
            ".dlc",
            ".engine",
            ".gguf",
            ".h5",
            ".hef",
            ".joblib",
            ".keras",
            ".mlmodel",
            ".msgpack",
            ".onnx",
            ".pb",
            ".plan",
            ".pmml",
            ".pt",
            ".pth",
            ".rknn",
            ".safetensors",
            ".skops",
            ".spm",
            ".tflite",
            ".ubj",
        }
        <= suffixes
    )
    assert MODEL_ARTIFACT_COMPOUND_SUFFIXES <= suffixes


def test_model_submission_token_patterns_keep_canonical_aliases() -> None:
    matched = {
        token: suffix
        for token in (
            "xgboost",
            "xgboost booster",
            "catboost",
            "catboost model",
            "lightgbm booster",
            "PyTorch checkpoint",
            "state dict",
            "TorchScript",
            "PyTorch weights",
            "Keras H5",
            "HDF5 model",
            "Core ML model",
            "Predictive Model Markup Language",
            "SentencePiece model",
            "XGBoost UBJSON",
            "skops model",
            "ONNX model",
            "Keras model",
            "GGUF model",
            "TensorRT engine",
            "TensorRT plan",
            "OpenVINO blob",
            "Hailo HEF",
            "Qualcomm DLC",
            "SNPE DLC",
            "Rockchip RKNN",
            "Msgpack model",
            "Joblib model",
            "safetensor",
            "safetensors",
            "tensorflow lite",
            "protobuf",
            "pb",
        )
        for pattern, suffix in MODEL_SUBMISSION_TOKEN_PATTERNS
        if pattern.search(token)
    }

    assert matched["xgboost"] == ".xgb"
    assert matched["xgboost booster"] == ".xgb"
    assert matched["catboost"] == ".cbm"
    assert matched["catboost model"] == ".cbm"
    assert matched["lightgbm booster"] == ".bst"
    assert matched["PyTorch checkpoint"] == ".pth"
    assert matched["state dict"] == ".pth"
    assert matched["TorchScript"] == ".pt"
    assert matched["PyTorch weights"] == ".pt"
    assert matched["Keras H5"] == ".h5"
    assert matched["HDF5 model"] == ".h5"
    assert matched["Core ML model"] == ".mlmodel"
    assert matched["Predictive Model Markup Language"] == ".pmml"
    assert matched["SentencePiece model"] == ".spm"
    assert matched["XGBoost UBJSON"] == ".ubj"
    assert matched["skops model"] == ".skops"
    assert matched["ONNX model"] == ".onnx"
    assert matched["Keras model"] == ".keras"
    assert matched["GGUF model"] == ".gguf"
    assert matched["TensorRT engine"] == ".engine"
    assert matched["TensorRT plan"] == ".plan"
    assert matched["OpenVINO blob"] == ".blob"
    assert matched["Hailo HEF"] == ".hef"
    assert matched["Qualcomm DLC"] == ".dlc"
    assert matched["SNPE DLC"] == ".dlc"
    assert matched["Rockchip RKNN"] == ".rknn"
    assert matched["Msgpack model"] == ".msgpack"
    assert matched["Joblib model"] == ".joblib"
    assert matched["safetensor"] == ".safetensors"
    assert matched["safetensors"] == ".safetensors"
    assert matched["tensorflow lite"] == ".tflite"
    assert matched["protobuf"] == ".pb"
    assert matched["pb"] == ".pb"


def test_model_submission_artifact_keywords_are_registry_backed() -> None:
    assert "onnx" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "gguf" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "tensorrt\\s+engine|engine(?:\\s+(?:model|file|format))?" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "rockchip\\s+rknn|rknn(?:\\s+(?:model|file|format))?" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "hailo\\s+hef|hef(?:\\s+(?:model|file|format))?" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "torchscript|pytorch\\s+(?:model|weights?)|pt\\s+(?:file|format)" in (
        MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "core\\s*ml(?!\\s+(?:model\\s+)?(?:package|directory|folder|bundle))(?:\\s+model)?" in (
        MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    )
    assert "predictive\\s+model\\s+markup\\s+language|pmml" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "sentencepiece(?:\\s+model)?|spm(?:\\s+(?:model|file|format))?" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
    assert "tensorflow\\s+lite" in MODEL_SUBMISSION_ARTIFACT_KEYWORDS
