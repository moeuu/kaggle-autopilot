from __future__ import annotations

import re
from collections.abc import Iterable

from kagglebot.asset_modality import (
    ARCHIVE_SUFFIXES,
    ARRAY_SUFFIXES,
    AUDIO_SUFFIXES,
    BIO_CHEM_TEXT_BASE_SUFFIXES,
    BIO_FASTQ_BASE_SUFFIXES,
    BIO_MOL_STRUCTURE_BASE_SUFFIXES,
    BIO_PDB_STRUCTURE_BASE_SUFFIXES,
    BIO_SEQUENCE_BASE_SUFFIXES,
    BIO_STRUCTURE_SUFFIXES,
    CODE_SUFFIXES,
    DIRECTORY_ARRAY_SUFFIXES,
    DOCUMENT_SUFFIXES,
    DOCUMENT_TEXT_METADATA_SUFFIXES,
    GENERIC_ARCHIVE_SUFFIXES,
    GEOSPATIAL_SUFFIXES,
    GRAPH_EDGE_LIST_BASE_SUFFIXES,
    GRAPH_RDF_BASE_SUFFIXES,
    GRAPH_SUFFIXES,
    GRAPH_XML_BASE_SUFFIXES,
    IMAGE_SUFFIXES,
    MEDICAL_IMAGE_SUFFIXES,
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_SUFFIXES,
    POINT_CLOUD_SUFFIXES,
    SCIENTIFIC_ARRAY_SUFFIXES,
    SIGNAL_SUFFIXES,
    TAR_ARCHIVE_SUFFIXES,
    VIDEO_SUFFIXES,
    ZSTD_TAR_ARCHIVE_SUFFIXES,
)
from kagglebot.compression_suffixes import strip_compression_suffix
from kagglebot.model_artifacts import MODEL_DIRECTORY_ARTIFACT_SUFFIXES
from kagglebot.submission_sample_discovery import TABULAR_SUBMISSION_SUFFIXES

_MEDIA_SUBMISSION_SUFFIXES = IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES
_ARRAY_SUBMISSION_SUFFIXES = ARRAY_SUFFIXES | SCIENTIFIC_ARRAY_SUFFIXES | DIRECTORY_ARRAY_SUFFIXES
_SIGNAL_SUBMISSION_SUFFIXES = SIGNAL_SUFFIXES
_MODEL_SUBMISSION_SUFFIXES = (
    MODEL_ARTIFACT_SUFFIXES | MODEL_ARTIFACT_COMPOUND_SUFFIXES | MODEL_DIRECTORY_ARTIFACT_SUFFIXES
)
MODEL_BUNDLE_MARKERS = (
    "model weights",
    "weights",
    "inference script",
    "checkpoint",
    *tuple(sorted(_MODEL_SUBMISSION_SUFFIXES)),
)
MULTI_FILE_SUBMISSION_MARKERS = (
    "zip file containing",
    "one file per",
    "per-image",
    "per image",
    "per-sample",
    "per sample",
    "masks",
    ".tif",
    ".tiff",
    ".png",
)
JSON_TEXT_NOISE_CONTEXT_MARKERS = (
    "example json",
    "topology json",
    "metadata json",
    "json metadata",
    "description json",
    "json description",
    "text description",
    "txt description",
    "sample json",
    "sample txt",
    "input json",
    "input txt",
)
NON_TABULAR_SUBMISSION_SUFFIXES = frozenset(
    IMAGE_SUFFIXES
    | MEDICAL_IMAGE_SUFFIXES
    | AUDIO_SUFFIXES
    | VIDEO_SUFFIXES
    | ARRAY_SUFFIXES
    | SCIENTIFIC_ARRAY_SUFFIXES
    | SIGNAL_SUFFIXES
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
)
NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES = frozenset(
    NON_TABULAR_SUBMISSION_SUFFIXES - MODEL_DIRECTORY_ARTIFACT_SUFFIXES
)
ARCHIVE_SUBMISSION_SUFFIXES = frozenset(ARCHIVE_SUFFIXES)
ARCHIVE_SUBMISSION_SUFFIXES_ORDERED = tuple(sorted(ARCHIVE_SUBMISSION_SUFFIXES, key=len, reverse=True))
EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES = tuple(
    sorted(ARCHIVE_SUBMISSION_SUFFIXES & GENERIC_ARCHIVE_SUFFIXES, key=len, reverse=True)
)
TAR_ARCHIVE_SUBMISSION_SUFFIXES = tuple(
    sorted(ARCHIVE_SUBMISSION_SUFFIXES & TAR_ARCHIVE_SUFFIXES, key=len, reverse=True)
)
ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES = tuple(
    sorted(ARCHIVE_SUBMISSION_SUFFIXES & ZSTD_TAR_ARCHIVE_SUFFIXES, key=len, reverse=True)
)


def submission_extension_pattern(suffixes: Iterable[str]) -> str:
    normalized = {suffix for raw_suffix in suffixes if (suffix := str(raw_suffix or "").strip().lower().lstrip("."))}
    return "|".join(re.escape(suffix) for suffix in sorted(normalized, key=len, reverse=True))


def drop_shadowed_model_index_component_suffixes(suffixes: list[str]) -> list[str]:
    shadows: set[str] = set()
    if ".safetensors.index.json" in suffixes:
        shadows.update({".safetensors", ".json"})
    if ".bin.index.json" in suffixes:
        shadows.update({".bin", ".json", ".pt"})
    if ".ckpt.index" in suffixes:
        shadows.add(".ckpt")
    if ".mlmodelc" in suffixes:
        shadows.add(".mlpackage")
    if not shadows:
        return suffixes
    return [suffix for suffix in suffixes if suffix not in shadows]


def drop_shadowed_plain_tar_suffixes(suffixes: list[str]) -> list[str]:
    if ".tar" not in suffixes:
        return suffixes
    if not any(suffix in TAR_ARCHIVE_SUFFIXES and suffix != ".tar" for suffix in suffixes):
        return suffixes
    return [suffix for suffix in suffixes if suffix != ".tar"]


def drop_shadowed_json_lines_suffixes(suffixes: list[str]) -> list[str]:
    base_suffixes = {strip_compression_suffix(suffix) for suffix in suffixes}
    if not ({".jsonlines", ".ndjson"} & base_suffixes):
        return suffixes
    return [suffix for suffix in suffixes if strip_compression_suffix(suffix) != ".jsonl"]


def drop_shadowed_json_family_suffixes(suffixes: list[str]) -> list[str]:
    base_suffixes = {strip_compression_suffix(suffix) for suffix in suffixes}
    shadows: set[str] = set()
    json_family_suffixes = {
        ".geojson",
        ".geojsonl",
        ".geojsonseq",
        ".jsonl",
        ".jsonlines",
        ".jsonld",
        ".ndjson",
        ".topojson",
    }
    if base_suffixes & json_family_suffixes:
        shadows.add(".json")
    if not shadows:
        return suffixes
    return [suffix for suffix in suffixes if strip_compression_suffix(suffix) not in shadows]


def drop_shadowed_geospatial_container_suffixes(suffixes: list[str]) -> list[str]:
    if ".kmz" not in suffixes:
        return suffixes
    return [suffix for suffix in suffixes if suffix not in {".kml", ".zip"}]


def drop_shadowed_geospatial_text_suffixes(suffixes: list[str]) -> list[str]:
    base_suffixes = {strip_compression_suffix(suffix) for suffix in suffixes}
    shadows: set[str] = set()
    if {".geojsonl", ".geojsonseq", ".topojson"} & base_suffixes:
        shadows.add(".geojson")
    if ".osm" in base_suffixes:
        shadows.add(".xml")
    if ".osm.pbf" in base_suffixes:
        shadows.add(".pbf")
    if not shadows:
        return suffixes
    return [suffix for suffix in suffixes if strip_compression_suffix(suffix) not in shadows]


def drop_shadowed_graph_xml_suffixes(suffixes: list[str]) -> list[str]:
    base_suffixes = {strip_compression_suffix(suffix) for suffix in suffixes}
    if ".xml" not in base_suffixes:
        return suffixes
    if not ({".graphml", ".gexf", ".rdf"} & base_suffixes):
        return suffixes
    return [suffix for suffix in suffixes if strip_compression_suffix(suffix) != ".xml"]


def drop_shadowed_submission_suffixes(suffixes: list[str]) -> list[str]:
    return drop_shadowed_model_index_component_suffixes(
        drop_shadowed_graph_xml_suffixes(
            drop_shadowed_geospatial_text_suffixes(
                drop_shadowed_geospatial_container_suffixes(
                    drop_shadowed_json_family_suffixes(
                        drop_shadowed_json_lines_suffixes(drop_shadowed_plain_tar_suffixes(suffixes))
                    )
                )
            )
        )
    )


ARCHIVE_SUBMISSION_EXTENSION_PATTERN = submission_extension_pattern(ARCHIVE_SUBMISSION_SUFFIXES)


def compressed_variant_base_suffixes(suffixes: Iterable[str]) -> set[str]:
    """Return base suffixes whose compressed variants are registered."""
    bases: set[str] = set()
    for suffix in suffixes:
        normalized = str(suffix or "").strip().lower()
        base = strip_compression_suffix(normalized)
        if base and base != normalized:
            bases.add(base)
    return bases


COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES = frozenset(
    {
        ".csv",
        ".tsv",
        ".tab",
        ".psv",
        ".txt",
        ".json",
        ".jsonl",
        ".jsonlines",
        ".ndjson",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".htm",
        ".md",
        ".pkl",
        ".pickle",
        ".rtf",
        ".srt",
        ".svg",
        ".vtt",
        ".mol2",
        ".gml",
        ".mtx",
        ".dae",
        ".ply",
        ".obj",
        ".off",
        ".pcd",
        ".pts",
        ".ptx",
        ".stl",
        ".x3d",
        ".xyz",
        ".fits",
        ".fit",
        ".fts",
        ".geojson",
        ".geojsonl",
        ".geojsonseq",
        ".kml",
        ".osm",
        ".topojson",
        ".dcm",
        ".dicom",
        ".ima",
        ".nii",
        ".hdr",
        ".img",
        ".mha",
        ".mhd",
        ".nhdr",
        ".nrrd",
    }
    | set(BIO_SEQUENCE_BASE_SUFFIXES)
    | set(BIO_CHEM_TEXT_BASE_SUFFIXES)
    | set(BIO_FASTQ_BASE_SUFFIXES)
    | set(BIO_PDB_STRUCTURE_BASE_SUFFIXES)
    | set(BIO_MOL_STRUCTURE_BASE_SUFFIXES)
    | set(GRAPH_XML_BASE_SUFFIXES)
    | set(GRAPH_EDGE_LIST_BASE_SUFFIXES)
    | set(GRAPH_RDF_BASE_SUFFIXES)
    | set(DOCUMENT_TEXT_METADATA_SUFFIXES)
    | compressed_variant_base_suffixes(NON_TABULAR_SUBMISSION_SUFFIXES)
)
COMPRESSION_TOKEN_PATTERN_SPECS: tuple[tuple[str, str], ...] = (
    (r"\bgzip(?:ped)?\b|\bgzipped\b|\.gz\b", ".gz"),
    (r"\bbzip2\b|\bbz2\b|\.bz2\b", ".bz2"),
    (r"\bxz\b|\.xz\b", ".xz"),
    (r"\bzstd\b|\bzstandard\b|\.zst\b", ".zst"),
)
COMPRESSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.I), suffix) for pattern, suffix in COMPRESSION_TOKEN_PATTERN_SPECS
)
_DOCUMENT_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"epub", ".epub"),
    (r"word\s+document|microsoft\s+word", ".docx"),
    (r"open\s*document\s+presentation|opendocument\s+presentation", ".odp"),
    (r"open\s*document\s+text|opendocument\s+text", ".odt"),
    (r"power\s*point|microsoft\s+power\s*point", ".pptx"),
    (r"latex", ".tex"),
    (r"restructured\s*text|rst", ".rst"),
    (r"ascii\s*doc|asciidoc", ".adoc"),
    (r"rich\s+text", ".rtf"),
    (r"html?", ".html"),
    (r"markdown", ".md"),
    (r"webvtt", ".vtt"),
    (r"subrip", ".srt"),
)
_DOCUMENT_SUBMISSION_ALIAS_SUFFIXES = {
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
_MEDIA_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"jpeg\s+xl", ".jxl"),
    (r"jpeg\s*2000|jp2", ".jp2"),
    (r"open\s*exr", ".exr"),
    (r"netpbm", ".pnm"),
    (r"windows\s+bitmap|bitmap\s+image|bmp", ".bmp"),
    (r"portable\s+network\s+graphics|png", ".png"),
    (r"(?:gzip|gzipped|gzip[-\s]*compressed)\s+(?:svg|scalable\s+vector\s+graphics)", ".svg.gz"),
    (r"(?:bzip2|bz2|bzip2[-\s]*compressed)\s+(?:svg|scalable\s+vector\s+graphics)", ".svg.bz2"),
    (r"(?:xz|xz[-\s]*compressed)\s+(?:svg|scalable\s+vector\s+graphics)", ".svg.xz"),
    (r"(?:zstd|zstandard|zst|zstd[-\s]*compressed)\s+(?:svg|scalable\s+vector\s+graphics)", ".svg.zst"),
    (r"svgz", ".svgz"),
    (r"scalable\s+vector\s+graphics|svg", ".svg"),
    (r"tiff?", ".tif"),
    (r"jpe?g(?!\s+(?:xl|2000))", ".jpg"),
    (r"mpeg[-\s]*4\s+video|mp4", ".mp4"),
    (r"mpeg(?![-\s]*4)", ".mpg"),
    (r"quicktime(?:\s+(?:movie|video|file|format))?|mov(?:\s+(?:video|file|format))?", ".mov"),
    (r"matroska|mkv", ".mkv"),
    (r"flash\s+video|flv", ".flv"),
    (r"audio\s+video\s+interleave|avi", ".avi"),
    (r"mpeg[-\s]*4\s+audio|m4a", ".m4a"),
    (r"advanced\s+audio\s+coding|aac", ".aac"),
    (r"free\s+lossless\s+audio\s+codec|flac", ".flac"),
    (r"ogg\s+vorbis|vorbis|ogg", ".ogg"),
    (r"aiff?", ".aiff"),
    (r"midi", ".mid"),
    (r"windows\s+media\s+audio", ".wma"),
)
_MEDIA_SUBMISSION_ALIAS_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".avi",
    ".bmp",
    ".exr",
    ".flac",
    ".flv",
    ".jpg",
    ".jpeg",
    ".jp2",
    ".jxl",
    ".m4a",
    ".mid",
    ".midi",
    ".mkv",
    ".mov",
    ".mpeg",
    ".mp4",
    ".mpg",
    ".ogg",
    ".pnm",
    ".png",
    ".svg",
    ".svg.bz2",
    ".svg.gz",
    ".svg.xz",
    ".svg.zst",
    ".svgz",
    ".tif",
    ".tiff",
    ".wma",
}
_MEDICAL_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"nii\.gz|nifti", ".nii.gz"),
    (r"dicom|dcm", ".dcm"),
    (r"mrcs(?:\s+(?:stack|image|file|format))?", ".mrcs"),
    (r"cryo[-\s]*em(?:\s+(?:map|volume|density))?|mrc", ".mrc"),
    (r"ccp4(?:\s+(?:map|volume|density))?", ".ccp4"),
    (r"nikon\s+nd2|nd2", ".nd2"),
    (r"leica\s+lif|lif(?:\s+(?:image|microscopy|file|format))?", ".lif"),
    (r"zeiss\s+lsm|lsm(?:\s+(?:image|microscopy|file|format))?", ".lsm"),
    (r"gatan\s+dm3|dm3", ".dm3"),
    (r"gatan\s+dm4|dm4", ".dm4"),
    (r"ome\.tiff?", ".ome.tif"),
    (r"meta\s*image|metaimage", ".mha"),
    (r"nrrd|nearly\s+raw\s+raster\s+data", ".nrrd"),
    (r"analyze(?:\s+7\.?5)?(?:\s+image)?", ".hdr"),
)
_MEDICAL_SUBMISSION_ALIAS_SUFFIXES = {
    ".ccp4",
    ".dcm",
    ".dicom",
    ".dm3",
    ".dm4",
    ".hdr",
    ".lif",
    ".lsm",
    ".mha",
    ".mrc",
    ".mrcs",
    ".nd2",
    ".nii.gz",
    ".nrrd",
    ".ome.tif",
    ".ome.tiff",
}
_POINT_CLOUD_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"3d\s+studio(?:\s+(?:scene|mesh|file|format))?|3ds\s+(?:scene|mesh|file|format)", ".3ds"),
    (r"alembic(?:\s+(?:scene|mesh|file|format|archive))?|abc\s+(?:scene|mesh|file|format)", ".abc"),
    (r"autodesk\s+fbx|fbx\s+(?:scene|mesh|file|format)", ".fbx"),
    (r"blender(?:\s+(?:scene|mesh|file|format))?|blend\s+(?:scene|mesh|file|format)", ".blend"),
    (r"\.step|step\s+(?:cad|model|geometry|file|format)", ".step"),
    (r"\.stp|stp\s+(?:cad|model|geometry|file|format)", ".stp"),
    (r"iges\s+(?:cad|model|geometry|file|format)|initial\s+graphics\s+exchange\s+specification", ".iges"),
    (r"igs\s+(?:cad|model|geometry|file|format)", ".igs"),
    (r"ifc\s+(?:bim|building|model|file|format)|industry\s+foundation\s+classes", ".ifc"),
    (r"brep\s+(?:cad|model|geometry|file|format)|boundary\s+representation", ".brep"),
    (r"gmsh|msh\s+(?:mesh|file|format)", ".msh"),
    (r"medit\s+mesh(?:\s+(?:file|format))?|\.mesh", ".mesh"),
    (r"vtk\s+polydata|vtp\s+(?:mesh|file|format)", ".vtp"),
    (r"vtk\s+unstructured\s+grid|vtu\s+(?:mesh|file|format)", ".vtu"),
    (r"vtk(?!\s+(?:polydata|unstructured\s+grid))(?:\s+(?:legacy|mesh|file|format))?", ".vtk"),
    (r"wavefront\s+obj|obj\s+(?:mesh|file|format)", ".obj"),
    (r"collada|dae\s+(?:scene|mesh|file|format)", ".dae"),
    (r"binary\s+gltf|glb\s+(?:scene|mesh|file|format)", ".glb"),
    (r"(?<!binary\s)gltf(?:\s+(?:scene|mesh|file|format))?", ".gltf"),
    (r"usdz(?:\s+(?:scene|mesh|file|format|archive))?", ".usdz"),
    (r"usda(?:\s+(?:scene|mesh|file|format))?", ".usda"),
    (r"usdc(?:\s+(?:scene|mesh|file|format))?", ".usdc"),
    (r"usd(?:\s+(?:scene|mesh|file|format))?", ".usd"),
    (r"\.off|off\s+(?:mesh|file|format)", ".off"),
)
_POINT_CLOUD_SUBMISSION_ALIAS_SUFFIXES = {
    ".abc",
    ".3ds",
    ".blend",
    ".brep",
    ".dae",
    ".fbx",
    ".glb",
    ".gltf",
    ".ifc",
    ".iges",
    ".igs",
    ".mesh",
    ".msh",
    ".obj",
    ".off",
    ".step",
    ".stp",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".vtk",
    ".vtp",
    ".vtu",
}
_GEOSPATIAL_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"cloud\s+optimized\s+geotiff|cog\b|geo\s*tiff?", ".tif"),
    (r"esri\s+shapefile|shapefile|shp", ".shp"),
    (r"geojson\s+(?:text\s+sequence|sequence|seq)|geojsonseq", ".geojsonseq"),
    (r"geojsonl|geojson\s+lines?", ".geojsonl"),
    (r"geojson(?!\s+(?:lines?|text\s+sequence|sequence|seq))", ".geojson"),
    (r"topojson|topology\s+json\s+(?:file|format|submission|output)", ".topojson"),
    (r"geopackage|gpkg", ".gpkg"),
    (r"mbtiles", ".mbtiles"),
    (r"pmtiles", ".pmtiles"),
    (r"mapbox\s+vector\s+tile|vector\s+tile|mvt", ".mvt"),
    (r"osm\s+pbf|openstreetmap\s+pbf", ".osm.pbf"),
    (r"openstreetmap|osm\s+(?:xml|file|format|submission|output)", ".osm"),
    (r"zipped\s+kml|kmz", ".kmz"),
    (r"keyhole\s+markup\s+language|(?<!zipped\s)kml", ".kml"),
    (r"gdal\s+vrt|virtual\s+raster|vrt", ".vrt"),
    (r"srtm(?:\s+(?:hgt|elevation|dem))?|hgt(?:\s+(?:elevation|raster|file|format))?", ".hgt"),
    (r"dted\s+level\s*0|dt0(?:\s+(?:elevation|raster|file|format))?", ".dt0"),
    (r"dted\s+level\s*1|dt1(?:\s+(?:elevation|raster|file|format))?", ".dt1"),
    (r"dted\s+level\s*2|dt2(?:\s+(?:elevation|raster|file|format))?", ".dt2"),
    (r"digital\s+elevation\s+model|dem(?:\s+(?:elevation|raster|file|format))?", ".dem"),
    (r"enhanced\s+compression\s+wavelet|ecw(?:\s+(?:raster|image|file|format))?", ".ecw"),
    (r"mrsid|mr\s*sid|sid(?:\s+(?:raster|image|file|format))?", ".sid"),
    (r"mapinfo\s+(?:interchange|mif(?:/mid)?(?:\s+pair)?)|mif/mid|mif", ".mif"),
    (r"envi(?:\s+raster)?\s+header", ".hdr"),
    (r"band\s+interleaved\s+by\s+line|bil(?:\s+raster)?", ".bil"),
    (r"band\s+interleaved\s+by\s+pixel|bip(?:\s+raster)?", ".bip"),
    (r"band\s+sequential|bsq(?:\s+raster)?", ".bsq"),
)
_GEOSPATIAL_SUBMISSION_ALIAS_SUFFIXES = {
    ".bil",
    ".bip",
    ".bsq",
    ".dem",
    ".dt0",
    ".dt1",
    ".dt2",
    ".ecw",
    ".gpkg",
    ".geopackage",
    ".hdr",
    ".hgt",
    ".geojson",
    ".geojsonl",
    ".geojsonseq",
    ".kml",
    ".kmz",
    ".mbtiles",
    ".mif",
    ".mvt",
    ".osm",
    ".osm.pbf",
    ".pmtiles",
    ".shp",
    ".sid",
    ".tif",
    ".tiff",
    ".topojson",
    ".vrt",
}
_BIO_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"variant\s+call\s+format|vcf", ".vcf"),
    (r"binary\s+variant\s+call\s+format|bcf", ".bcf"),
    (r"binary\s+alignment\s+map|bam", ".bam"),
    (r"sequence\s+alignment\s+map|sam", ".sam"),
    (r"compressed\s+reference[-\s]*oriented\s+alignment\s+map|cram", ".cram"),
    (r"general\s+feature\s+format\s*3|gff3", ".gff3"),
    (r"general\s+feature\s+format(?!\s*3)|\bgff\b", ".gff"),
    (r"gene\s+transfer\s+format|gtf", ".gtf"),
    (r"browser\s+extensible\s+data|bed", ".bed"),
    (r"bigwig|big\s*wig", ".bigwig"),
    (r"bigbed|big\s*bed", ".bigbed"),
    (r"pod5", ".pod5"),
    (r"protein\s+data\s+bank|pdb", ".pdb"),
    (r"macromolecular\s+crystallographic\s+information|mmcif", ".mmcif"),
    (r"(?<!macromolecular\s)crystallographic\s+information|cif", ".cif"),
    (r"structure[-\s]*data\s+file|sdf", ".sdf"),
    (r"tripos\s+mol2|mol2", ".mol2"),
    (r"mdl\s+molfile|molfile|mol", ".mol"),
    (r"fastq|fq(?:\s+(?:reads?|sequences?|file|format))?", ".fastq"),
    (r"fasta|fa(?:\s+(?:sequences?|file|format))?", ".fasta"),
    (r"simplified\s+molecular\s+input\s+line\s+entry\s+system|smiles|smi", ".smiles"),
    (r"inchi", ".inchi"),
    (r"selfies", ".selfies"),
    (r"reaction\s+file|rxn", ".rxn"),
)
_BIO_SUBMISSION_ALIAS_SUFFIXES = {
    ".bam",
    ".bcf",
    ".bed",
    ".bigbed",
    ".bigwig",
    ".cif",
    ".cram",
    ".fa",
    ".fasta",
    ".fastq",
    ".fq",
    ".gff",
    ".gff3",
    ".gtf",
    ".inchi",
    ".mmcif",
    ".mol",
    ".mol2",
    ".pod5",
    ".pdb",
    ".rxn",
    ".sam",
    ".sdf",
    ".selfies",
    ".smi",
    ".smiles",
    ".vcf",
}
_GRAPH_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"graph\s+markup\s+language|graphml", ".graphml"),
    (r"graph\s+exchange\s+xml\s+format|gexf", ".gexf"),
    (r"graph\s+model(?:l)?ing\s+language|gml", ".gml"),
    (r"matrix\s+market|mtx", ".mtx"),
    (r"json[-\s]*ld|linked\s+data\s+json", ".jsonld"),
    (r"turtle(?:\s+(?:rdf|file|format))?|ttl(?:\s+(?:rdf|file|format))?", ".ttl"),
    (r"n[-\s]*triples?|nt(?:\s+(?:rdf|file|format))?", ".nt"),
    (r"n[-\s]*quads?|nq(?:\s+(?:rdf|file|format))?", ".nq"),
    (r"rdf\s+xml(?:\s+(?:file|format))?", ".rdf"),
    (r"owl(?:\s+(?:ontology|file|format))?", ".owl"),
    (r"trig(?:\s+(?:rdf|file|format))?", ".trig"),
    (r"edge[-\s]*list|edgelist", ".edgelist"),
    (r"edges", ".edges"),
)
_GRAPH_SUBMISSION_ALIAS_SUFFIXES = {
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
_ARRAY_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"netcdf[-\s]*4|netcdf4|nc4", ".nc4"),
    (r"netcdf(?![-\s]*4)", ".nc"),
    (r"common\s+data\s+format|cdf", ".cdf"),
    (r"grib2", ".grib2"),
    (r"grib", ".grib"),
    (r"flexible\s+image\s+transport|fits", ".fits"),
    (r"fit\s+(?:file|format)", ".fit"),
    (r"anndata|h5ad", ".h5ad"),
    (r"loom(?:\s+(?:file|format))?", ".loom"),
    (r"ome[-\s]*zarr(?:\s+(?:store|directory|group|archive|format))?", ".ome.zarr"),
    (r"(?<!ome[-\s])zarr(?:\s+(?:store|directory|group|archive|format))?", ".zarr"),
    (r"\bn5(?:\s+(?:store|directory|group|archive|format))?\b", ".n5"),
    (r"matlab|mat\s+file", ".mat"),
    (
        r"(?:compressed\s+numpy\s+(?:array|archive)|numpy\s+(?:zip\s+)?archive|scipy\s+sparse\s+matrix"
        r"(?:\s+archive)?)",
        ".npz",
    ),
    (r"(?<!compressed\s)numpy\s+array", ".npy"),
)
_ARRAY_SUBMISSION_ALIAS_SUFFIXES = {
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
    ".npz",
    ".ome.zarr",
    ".zarr",
}
_SIGNAL_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"european\s+data\s+format(?:\s+plus)?|edf\+?", ".edf"),
    (r"biosemi\s+data\s+format|biomedical\s+data\s+format|bdf", ".bdf"),
    (r"waveform\s+database|wfdb(?:\s+header)?|physionet\s+header|ecg\s+header", ".hea"),
    (r"neurodata\s+without\s+borders(?:\s+neurophysiology)?|nwb", ".nwb"),
    (r"(?:national\s+instruments\s+)?technical\s+data\s+management\s+streaming|tdms", ".tdms"),
    (r"axon\s+binary\s+(?:file|format)|abf", ".abf"),
    (r"brainvision\s+header|vhdr(?:\s+(?:header|file|format))?", ".vhdr"),
    (r"brainvision\s+marker|vmrk(?:\s+(?:marker|file|format))?", ".vmrk"),
    (r"mne\s+fif|fif(?:\s+(?:eeg|meg|file|format))?", ".fif"),
    (r"brainvision\s+eeg|eeg(?:\s+(?:signal|file|format))?", ".eeg"),
    (r"eeglab\s+fdt|fdt(?:\s+(?:eeg|file|format))?", ".fdt"),
    (r"eeglab(?:\s+(?:set|dataset|file|format))?|set(?:\s+(?:eeg|file|format))?", ".set"),
    (r"neuroscan\s+cnt|cnt(?:\s+(?:eeg|file|format))?", ".cnt"),
    (r"micromed\s+trc|trc(?:\s+(?:eeg|file|format))?", ".trc"),
)
_SIGNAL_SUBMISSION_ALIAS_SUFFIXES = {
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
_MODEL_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"safetensors?\s+(?:index\s+json|json\s+index)", ".safetensors.index.json"),
    (r"(?:pytorch\s+)?(?:model\s+)?bin\s+(?:index\s+json|json\s+index)", ".bin.index.json"),
    (r"(?:tensorflow\s+)?(?:checkpoint|ckpt)\s+index", ".ckpt.index"),
    (r"(?:tensorflow\s+)?(?:saved\s*model)(?:\s+(?:directory|folder|bundle))?", ".savedmodel"),
    (r"(?:hugging\s*face|hf)\s+model(?:\s+(?:directory|folder|bundle))?", ".hfmodel"),
    (r"mlflow\s+model(?:\s+(?:directory|folder|bundle))?", ".mlflowmodel"),
    (
        r"(?:compiled\s+core\s*ml|core\s*ml\s+compiled)(?:\s+model)?(?:\s+(?:package|directory|folder|bundle))?",
        ".mlmodelc",
    ),
    (r"core\s*ml(?:\s+model)?\s+(?:package|directory|folder|bundle)", ".mlpackage"),
    (r"(?:tensorflow\s+)?(?:checkpoint|ckpt)\s+(?:directory|folder|bundle)", ".tfcheckpoint"),
    (r"xgboost(?!\s+ubj(?:son)?)(?:\s+(?:booster|model))?", ".xgb"),
    (r"catboost(?:\s+(?:model))?", ".cbm"),
    (r"light\s*gbm(?:\s+(?:booster|model))?", ".bst"),
    (r"pytorch\s+checkpoint|(?:pytorch\s+)?state[-_\s]*dict|pth(?:\s+(?:file|format))?", ".pth"),
    (r"torchscript|pytorch\s+(?:model|weights?)|pt(?:\s+(?:file|format))?", ".pt"),
    (r"keras\s+h5|hdf5\s+model|h5\s+model", ".h5"),
    (
        r"core\s*ml(?!\s+(?:model\s+)?(?:package|directory|folder|bundle))(?:\s+model)?"
        r"(?:\s+(?:file|format))?",
        ".mlmodel",
    ),
    (r"predictive\s+model\s+markup\s+language|pmml", ".pmml"),
    (r"sentencepiece(?:\s+model)?|spm(?:\s+(?:model|file|format))?", ".spm"),
    (r"xgboost\s+ubj(?:son)?|ubjson(?:\s+(?:model|file|format))?|ubj(?:\s+(?:model|file|format))?", ".ubj"),
    (r"skops(?:\s+(?:model|file|format))?", ".skops"),
    (r"onnx(?:\s+(?:model|file|format))?", ".onnx"),
    (r"keras(?!\s+h5)(?:\s+(?:model|file|format))?", ".keras"),
    (r"gguf(?:\s+(?:model|file|format))?", ".gguf"),
    (r"tensorrt\s+engine|engine(?:\s+(?:model|file|format))?", ".engine"),
    (r"tensorrt\s+plan|plan(?:\s+(?:model|file|format))?", ".plan"),
    (r"openvino\s+blob|blob(?:\s+(?:model|file|format))?", ".blob"),
    (r"hailo\s+hef|hef(?:\s+(?:model|file|format))?", ".hef"),
    (r"qualcomm\s+dlc|snpe\s+dlc|dlc(?:\s+(?:model|file|format))?", ".dlc"),
    (r"rockchip\s+rknn|rknn(?:\s+(?:model|file|format))?", ".rknn"),
    (r"msgpack(?:\s+(?:model|file|format))?", ".msgpack"),
    (r"joblib(?:\s+(?:model|file|format))?", ".joblib"),
    (r"safetensors?", ".safetensors"),
    (r"tensorflow\s+lite", ".tflite"),
    (r"protobuf|pb", ".pb"),
)
_MODEL_SUBMISSION_ALIAS_SUFFIXES = {
    ".bin.index.json",
    ".blob",
    ".bst",
    ".cbm",
    ".ckpt.index",
    ".dlc",
    ".engine",
    ".gguf",
    ".h5",
    ".hef",
    ".hfmodel",
    ".joblib",
    ".keras",
    ".mlmodel",
    ".mlflowmodel",
    ".mlmodelc",
    ".mlpackage",
    ".msgpack",
    ".onnx",
    ".pb",
    ".plan",
    ".pmml",
    ".pt",
    ".pth",
    ".rknn",
    ".safetensors",
    ".safetensors.index.json",
    ".savedmodel",
    ".skops",
    ".spm",
    ".tfcheckpoint",
    ".tflite",
    ".ubj",
    ".xgb",
}
_CODE_SUBMISSION_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"python\s+(?:script|file|source|code)", ".py"),
    (r"jupyter\s+notebook|ipython\s+notebook|notebook\s+(?:file|source)", ".ipynb"),
    (r"r\s+(?:script|file|source|code)", ".r"),
    (r"julia\s+(?:script|file|source|code)", ".jl"),
)
_CODE_SUBMISSION_ALIAS_SUFFIXES = {".ipynb", ".jl", ".py", ".r"}
_TABULAR_SUBMISSION_KEYWORD_ALIASES = (
    "excel",
    "spreadsheet",
    "opendocument\\s+spreadsheet",
    "macro[-\\s]*enabled\\s+excel",
    "macro[-\\s]*enabled\\s+workbook",
    "pickle",
    "stata",
    "arrow\\s+ipc",
    "json\\s+lines",
    "line-delimited\\s+json",
    "newline-delimited\\s+json",
    "comma[-\\s]*separated",
    "comma\\s+delimited",
    "semicolon[-\\s]*separated",
    "semicolon[-\\s]*delimited",
    "tab[-\\s]*separated",
    "tab\\s+delimited",
    "pipe[-\\s]*separated",
    "pipe\\s+delimited",
    "sqlite3\\s+(?:database|db|file|format)",
    "sqlite\\s+(?:database|db|file|format)",
)
MISC_SUBMISSION_ARTIFACT_KEYWORD_ALIASES = (
    "notebook",
    "code",
    "coco",
    "coco\\s+(?:caption|captions|keypoint|keypoints|panoptic|segmentation|segmentations)",
    "labelme",
    "label\\s*me",
    "yolo",
    "yolov\\d+",
    "pascal\\s+voc",
    "voc",
    "open\\s+images?",
    "run[-\\s]*length",
    "rle",
)
MISC_SUBMISSION_TOKEN_PATTERN_SPECS = (
    (
        r"\bcoco(?:\s+(?:annotation|annotations|caption|captions|detection|detections|instance|instances|"
        r"keypoint|keypoints|panoptic|segmentation|segmentations))?(?:\s+json)?\b",
        ".json",
    ),
    (r"\blabel\s*me(?:\s+(?:annotation|annotations))?(?:\s+json)?\b", ".json"),
    (r"\byolov?\d*\s+(?:label|labels|annotation|annotations)(?:\s+(?:txt|text|file|files))?\b", ".txt"),
    (r"\bpascal\s+voc(?:\s+(?:annotation|annotations))?(?:\s+(?:xml|file|files))?\b", ".xml"),
    (r"\bvoc\s+(?:annotation|annotations)(?:\s+(?:xml|file|files))?\b", ".xml"),
    (r"\bopen\s+images?(?:\s+(?:annotation|annotations|csv|file|files))?\b", ".csv"),
    (r"\brle\b|\brun[-\s]*length(?:\s+encoded)?(?:\s+(?:mask|masks|encoding|encodings))?\b", ".csv"),
)


def _suffix_keyword_pattern(suffix: str) -> str:
    return re.escape(suffix.removeprefix("."))


def _compile_submission_token_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9.])", re.I)


def _registry_submission_keyword_patterns(
    suffixes: set[str],
    alias_patterns: tuple[tuple[str, str], ...],
    alias_suffixes: set[str],
) -> tuple[str, ...]:
    del alias_suffixes
    generated = (_suffix_keyword_pattern(suffix) for suffix in sorted(suffixes, key=len, reverse=True))
    return tuple(pattern for pattern, _suffix in alias_patterns) + tuple(generated)


def _registry_submission_token_patterns(
    suffixes: set[str],
    alias_patterns: tuple[tuple[str, str], ...],
    alias_suffixes: set[str],
) -> tuple[tuple[re.Pattern[str], str], ...]:
    generated = (
        (_compile_submission_token_pattern(_suffix_keyword_pattern(suffix)), suffix)
        for suffix in sorted(suffixes - alias_suffixes, key=len, reverse=True)
    )
    aliases = ((_compile_submission_token_pattern(pattern), suffix) for pattern, suffix in alias_patterns)
    return tuple(aliases) + tuple(generated)


def _media_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        _MEDIA_SUBMISSION_SUFFIXES,
        _MEDIA_SUBMISSION_ALIAS_PATTERNS,
        _MEDIA_SUBMISSION_ALIAS_SUFFIXES,
    )


def _document_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        DOCUMENT_SUFFIXES,
        _DOCUMENT_SUBMISSION_ALIAS_PATTERNS,
        _DOCUMENT_SUBMISSION_ALIAS_SUFFIXES,
    )


def _medical_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        MEDICAL_IMAGE_SUFFIXES,
        _MEDICAL_SUBMISSION_ALIAS_PATTERNS,
        _MEDICAL_SUBMISSION_ALIAS_SUFFIXES,
    )


def _point_cloud_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        POINT_CLOUD_SUFFIXES,
        _POINT_CLOUD_SUBMISSION_ALIAS_PATTERNS,
        _POINT_CLOUD_SUBMISSION_ALIAS_SUFFIXES,
    )


def _geospatial_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        GEOSPATIAL_SUFFIXES,
        _GEOSPATIAL_SUBMISSION_ALIAS_PATTERNS,
        _GEOSPATIAL_SUBMISSION_ALIAS_SUFFIXES,
    )


def _bio_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        BIO_STRUCTURE_SUFFIXES,
        _BIO_SUBMISSION_ALIAS_PATTERNS,
        _BIO_SUBMISSION_ALIAS_SUFFIXES,
    )


def _graph_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        GRAPH_SUFFIXES,
        _GRAPH_SUBMISSION_ALIAS_PATTERNS,
        _GRAPH_SUBMISSION_ALIAS_SUFFIXES,
    )


def _array_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        _ARRAY_SUBMISSION_SUFFIXES,
        _ARRAY_SUBMISSION_ALIAS_PATTERNS,
        _ARRAY_SUBMISSION_ALIAS_SUFFIXES,
    )


def _signal_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        _SIGNAL_SUBMISSION_SUFFIXES,
        _SIGNAL_SUBMISSION_ALIAS_PATTERNS,
        _SIGNAL_SUBMISSION_ALIAS_SUFFIXES,
    )


def _model_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        _MODEL_SUBMISSION_SUFFIXES,
        _MODEL_SUBMISSION_ALIAS_PATTERNS,
        _MODEL_SUBMISSION_ALIAS_SUFFIXES,
    )


def _code_submission_keyword_patterns() -> tuple[str, ...]:
    return _registry_submission_keyword_patterns(
        CODE_SUFFIXES,
        _CODE_SUBMISSION_ALIAS_PATTERNS,
        _CODE_SUBMISSION_ALIAS_SUFFIXES,
    )


def _tabular_submission_keyword_patterns() -> tuple[str, ...]:
    generated = (
        _suffix_keyword_pattern(suffix) for suffix in sorted(TABULAR_SUBMISSION_SUFFIXES, key=len, reverse=True)
    )
    return (*_TABULAR_SUBMISSION_KEYWORD_ALIASES, *tuple(generated))


def _media_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        _MEDIA_SUBMISSION_SUFFIXES,
        _MEDIA_SUBMISSION_ALIAS_PATTERNS,
        _MEDIA_SUBMISSION_ALIAS_SUFFIXES,
    )


def _document_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        DOCUMENT_SUFFIXES,
        _DOCUMENT_SUBMISSION_ALIAS_PATTERNS,
        _DOCUMENT_SUBMISSION_ALIAS_SUFFIXES,
    )


def _medical_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        MEDICAL_IMAGE_SUFFIXES,
        _MEDICAL_SUBMISSION_ALIAS_PATTERNS,
        _MEDICAL_SUBMISSION_ALIAS_SUFFIXES,
    )


def _point_cloud_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        POINT_CLOUD_SUFFIXES,
        _POINT_CLOUD_SUBMISSION_ALIAS_PATTERNS,
        _POINT_CLOUD_SUBMISSION_ALIAS_SUFFIXES,
    )


def _geospatial_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        GEOSPATIAL_SUFFIXES,
        _GEOSPATIAL_SUBMISSION_ALIAS_PATTERNS,
        _GEOSPATIAL_SUBMISSION_ALIAS_SUFFIXES,
    )


def _bio_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        BIO_STRUCTURE_SUFFIXES,
        _BIO_SUBMISSION_ALIAS_PATTERNS,
        _BIO_SUBMISSION_ALIAS_SUFFIXES,
    )


def _graph_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        GRAPH_SUFFIXES,
        _GRAPH_SUBMISSION_ALIAS_PATTERNS,
        _GRAPH_SUBMISSION_ALIAS_SUFFIXES,
    )


def _array_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        _ARRAY_SUBMISSION_SUFFIXES,
        _ARRAY_SUBMISSION_ALIAS_PATTERNS,
        _ARRAY_SUBMISSION_ALIAS_SUFFIXES,
    )


def _signal_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        _SIGNAL_SUBMISSION_SUFFIXES,
        _SIGNAL_SUBMISSION_ALIAS_PATTERNS,
        _SIGNAL_SUBMISSION_ALIAS_SUFFIXES,
    )


def _model_submission_token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return _registry_submission_token_patterns(
        _MODEL_SUBMISSION_SUFFIXES,
        _MODEL_SUBMISSION_ALIAS_PATTERNS,
        _MODEL_SUBMISSION_ALIAS_SUFFIXES,
    )


MEDIA_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_media_submission_keyword_patterns())
DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_document_submission_keyword_patterns())
MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_medical_submission_keyword_patterns())
POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_point_cloud_submission_keyword_patterns())
GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_geospatial_submission_keyword_patterns())
BIO_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_bio_submission_keyword_patterns())
GRAPH_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_graph_submission_keyword_patterns())
ARRAY_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_array_submission_keyword_patterns())
SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_signal_submission_keyword_patterns())
MODEL_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_model_submission_keyword_patterns())
CODE_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_code_submission_keyword_patterns())

ARCHIVE_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(
    (
        "tarball",
        "zstd",
        "zst",
        *(_suffix_keyword_pattern(suffix) for suffix in ARCHIVE_SUBMISSION_SUFFIXES_ORDERED),
    )
)
TABULAR_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(_tabular_submission_keyword_patterns())
MISC_SUBMISSION_ARTIFACT_KEYWORDS = "|".join(MISC_SUBMISSION_ARTIFACT_KEYWORD_ALIASES)

SUBMISSION_ARTIFACT_KEYWORDS = (
    f"{TABULAR_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{ARCHIVE_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{MISC_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{MEDICAL_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{DOCUMENT_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{GEOSPATIAL_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{BIO_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{GRAPH_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{ARRAY_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{SIGNAL_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{MEDIA_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{POINT_CLOUD_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{MODEL_SUBMISSION_ARTIFACT_KEYWORDS}|"
    f"{CODE_SUBMISSION_ARTIFACT_KEYWORDS}"
)

CODE_FENCE_LANG_SUFFIX_SPECS: tuple[tuple[str, str], ...] = (
    ("csv", ".csv"),
    ("tsv", ".tsv"),
    ("tab", ".tab"),
    ("psv", ".psv"),
    ("jsonl", ".jsonl"),
    ("json-lines", ".jsonl"),
    ("json_lines", ".jsonl"),
    ("jsonlines", ".jsonlines"),
    ("ndjson", ".ndjson"),
    ("nd-json", ".ndjson"),
    ("json", ".json"),
    ("yaml", ".yaml"),
    ("yml", ".yaml"),
    ("parquet", ".parquet"),
    ("parq", ".parq"),
    ("pq", ".pq"),
    ("orc", ".orc"),
    ("avro", ".avro"),
    ("hdf", ".hdf5"),
    ("hdf5", ".hdf5"),
    ("feather", ".feather"),
    ("ftr", ".ftr"),
    ("arrow", ".arrow"),
    ("ipc", ".ipc"),
    ("xls", ".xls"),
    ("xlsm", ".xlsm"),
    ("xlsx", ".xlsx"),
    ("ods", ".ods"),
    ("pkl", ".pkl"),
    ("pickle", ".pkl"),
    ("dta", ".dta"),
    ("stata", ".dta"),
    ("xml", ".xml"),
    ("html", ".html"),
    ("htm", ".html"),
    ("txt", ".txt"),
)
CODE_FENCE_LANG_TO_SUFFIX = dict(CODE_FENCE_LANG_SUFFIX_SPECS)

MEDIA_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _media_submission_token_patterns()
DOCUMENT_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _document_submission_token_patterns()
MEDICAL_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _medical_submission_token_patterns()
POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    _point_cloud_submission_token_patterns()
)
GEOSPATIAL_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _geospatial_submission_token_patterns()
BIO_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _bio_submission_token_patterns()
GRAPH_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _graph_submission_token_patterns()
ARRAY_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _array_submission_token_patterns()
SIGNAL_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _signal_submission_token_patterns()
MODEL_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _model_submission_token_patterns()
CODE_SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _registry_submission_token_patterns(
    CODE_SUFFIXES,
    _CODE_SUBMISSION_ALIAS_PATTERNS,
    _CODE_SUBMISSION_ALIAS_SUFFIXES,
)

SUBMISSION_TOKEN_PATTERN_SPECS: tuple[tuple[str, str], ...] = (
    (r"\btar\.gz\b|\.tar\.gz\b", ".tar.gz"),
    (
        r"\b(?:gzip|gzipped|gzip[-\s]*compressed)\s+tar(?:ball|\s+archive)?\b|"
        r"\btar(?:ball|\s+archive)?\s+(?:gzip|gzipped|gzip[-\s]*compressed)\b",
        ".tar.gz",
    ),
    (r"\btgz\b|\.tgz\b", ".tgz"),
    (r"\btar\.bz2\b|\.tar\.bz2\b", ".tar.bz2"),
    (
        r"\b(?:bzip2|bz2|bzip2[-\s]*compressed)\s+tar(?:ball|\s+archive)?\b|"
        r"\btar(?:ball|\s+archive)?\s+(?:bzip2|bz2|bzip2[-\s]*compressed)\b",
        ".tar.bz2",
    ),
    (r"\btbz2\b|\.tbz2\b", ".tbz2"),
    (r"\btar\.xz\b|\.tar\.xz\b", ".tar.xz"),
    (
        r"\b(?:xz|xz[-\s]*compressed)\s+tar(?:ball|\s+archive)?\b|"
        r"\btar(?:ball|\s+archive)?\s+(?:xz|xz[-\s]*compressed)\b",
        ".tar.xz",
    ),
    (r"\btxz\b|\.txz\b", ".txz"),
    (r"\btar\.zst\b|\.tar\.zst\b", ".tar.zst"),
    (
        r"\b(?:zstd|zstandard|zst|zstd[-\s]*compressed)\s+tar(?:ball|\s+archive)?\b|"
        r"\btar(?:ball|\s+archive)?\s+(?:zstd|zstandard|zst|zstd[-\s]*compressed)\b",
        ".tar.zst",
    ),
    (r"\btzst\b|\.tzst\b", ".tzst"),
    (r"\btarball\b", ".tar"),
    (r"\btar\b|\.tar\b", ".tar"),
    (r"\bzip(?:ped)?\b", ".zip"),
    (r"\b7z\b|7[-\s]*zip", ".7z"),
    (r"\brar\b", ".rar"),
    *((pattern.pattern, suffix) for pattern, suffix in MEDICAL_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in DOCUMENT_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in GEOSPATIAL_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in BIO_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in GRAPH_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in ARRAY_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in SIGNAL_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in MEDIA_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in POINT_CLOUD_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in MODEL_SUBMISSION_TOKEN_PATTERNS),
    *((pattern.pattern, suffix) for pattern, suffix in CODE_SUBMISSION_TOKEN_PATTERNS),
    *MISC_SUBMISSION_TOKEN_PATTERN_SPECS,
    (r"\bjsonlines\b", ".jsonlines"),
    (r"\bndjson\b", ".ndjson"),
    (
        r"\bjson\s+lines\b|\bline-delimited\s+json\b|\bnewline-delimited\s+json\b|\bjsonl\b",
        ".jsonl",
    ),
    (r"\byaml\b|\byml\b", ".yaml"),
    (r"\btab[-\s]*(?:file|format)\b", ".tab"),
    (r"\bcomma[-\s]*separated(?:\s+values?)?\b|\bcomma\s+delimited\b", ".csv"),
    (r"\bsemicolon[-\s]*separated(?:\s+values?)?\b|\bsemicolon[-\s]*delimited\b", ".csv"),
    (r"\bpipe[-\s]*separated\b|\bpipe\s+delimited\b|\bpsv\b", ".psv"),
    (r"\btab[-\s]*separated\b|\btab\s+delimited\b|\btsv\b", ".tsv"),
    (r"\bparquet\b", ".parquet"),
    (r"\bparq\b", ".parq"),
    (r"\bpq\b", ".pq"),
    (r"\borc\b", ".orc"),
    (r"\bavro\b", ".avro"),
    (r"\bhdf5\b|\bhdf\b", ".hdf5"),
    (r"\bfeather\b", ".feather"),
    (r"\bftr\b", ".ftr"),
    (r"\barrow\b", ".arrow"),
    (r"\bipc\b", ".ipc"),
    (r"\bods\b|\bopendocument\s+spreadsheet\b", ".ods"),
    (r"\bxlsm\b|\bmacro[-\s]*enabled\s+excel\b|\bmacro[-\s]*enabled\s+workbook\b", ".xlsm"),
    (r"\bxlsx\b|\bexcel\b|\bspreadsheet\b", ".xlsx"),
    (r"\bxls\b", ".xls"),
    (r"\bpkl\b|\bpickled?\b", ".pkl"),
    (r"\bdta\b|\bstata\b", ".dta"),
    (r"\bsqlite3\s+(?:database|db|file|format)\b", ".sqlite3"),
    (r"\bsqlite\s+(?:database|db|file|format)\b", ".sqlite"),
    (r"\bxml\b", ".xml"),
    (r"\btsv\b|\btab[-\s]*separated\b", ".tsv"),
    (r"\bcsv\b", ".csv"),
    (r"\bjson\b", ".json"),
    (r"\btxt\b|\btext\s+file\b", ".txt"),
)
SUBMISSION_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.I), suffix) for pattern, suffix in SUBMISSION_TOKEN_PATTERN_SPECS
)
