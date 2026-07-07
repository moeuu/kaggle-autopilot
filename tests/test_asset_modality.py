from __future__ import annotations

from pathlib import Path

from kagglebot.asset_modality import (
    ANALYZE_IMAGE_PAIR_BASE_SUFFIXES,
    ANALYZE_IMAGE_PAIR_SUFFIXES,
    ANNOTATION_DIR_NAME_TOKENS,
    ANNOTATION_FILE_NAME_TOKENS,
    ANNOTATION_FILE_SUFFIXES,
    ANNOTATION_JSON_SUFFIXES,
    ANNOTATION_TEXT_SUFFIXES,
    ANNOTATION_XML_SUFFIXES,
    ARCHIVE_SUFFIXES,
    ASSET_COMPRESSION_SUFFIXES,
    BIO_CHEM_TEXT_BASE_SUFFIXES,
    BIO_FASTQ_BASE_SUFFIXES,
    BIO_GENOMIC_BASE_SUFFIXES,
    BIO_GENOMIC_TEXT_BASE_SUFFIXES,
    BIO_MOL_STRUCTURE_BASE_SUFFIXES,
    BIO_PDB_STRUCTURE_BASE_SUFFIXES,
    BIO_SEQUENCE_BASE_SUFFIXES,
    BIO_STRUCTURE_SUFFIXES,
    CODE_SUFFIXES,
    CRYOEM_IMAGE_BASE_SUFFIXES,
    CRYOEM_IMAGE_SUFFIXES,
    DATA_ASSET_SUFFIXES,
    DICOM_IMAGE_BASE_SUFFIXES,
    DICOM_IMAGE_SUFFIXES,
    DIRECTORY_ARRAY_SUFFIXES,
    DOCUMENT_HTML_BASE_SUFFIXES,
    DOCUMENT_SUFFIXES,
    DOCUMENT_TEXT_METADATA_SUFFIXES,
    GEOSPATIAL_RASTER_BASE_SUFFIXES,
    GEOSPATIAL_SUFFIXES,
    GRAPH_EDGE_LIST_BASE_SUFFIXES,
    GRAPH_RDF_BASE_SUFFIXES,
    GRAPH_SUFFIXES,
    GRAPH_XML_BASE_SUFFIXES,
    IMAGE_SUFFIXES,
    MEDICAL_HEADER_IMAGE_BASE_SUFFIXES,
    MEDICAL_HEADER_IMAGE_SUFFIXES,
    MEDICAL_IMAGE_SUFFIXES,
    MICROSCOPY_IMAGE_BASE_SUFFIXES,
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_FILENAMES,
    MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES,
    MODEL_ARTIFACT_NAME_TOKENS,
    MODEL_ARTIFACT_SUFFIXES,
    MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES,
    NIFTI_IMAGE_BASE_SUFFIXES,
    NIFTI_IMAGE_SUFFIXES,
    POINT_CLOUD_SUFFIXES,
    POINT_CLOUD_TEXT_METADATA_SUFFIXES,
    SCIENTIFIC_ARRAY_SUFFIXES,
    SIGNAL_NEUROPHYS_BASE_SUFFIXES,
    SIGNAL_SUFFIXES,
    TABULAR_DATA_SUFFIXES,
    VECTOR_IMAGE_SUFFIXES,
    archive_container,
    artifact_stem,
    artifact_suffix,
    asset_suffix,
    infer_asset_modality,
    infer_asset_modality_from_extensions,
    is_detection_annotation_path,
    is_model_artifact_path,
)
from kagglebot.submission_sample_discovery import TABULAR_GEOJSON_SUFFIXES


def test_asset_suffix_preserves_known_compound_suffixes() -> None:
    assert asset_suffix(Path("case_001.nii.gz")) == ".nii.gz"
    assert asset_suffix(Path("case_001.dcm.gz")) == ".dcm.gz"
    assert asset_suffix(Path("case_001.ima.xz")) == ".ima.xz"
    assert asset_suffix(Path("case_001.hdr.gz")) == ".hdr.gz"
    assert asset_suffix(Path("case_001.img.zst")) == ".img.zst"
    assert asset_suffix(Path("case_001.nhdr")) == ".nhdr"
    assert asset_suffix(Path("case_001.nrrd.zst")) == ".nrrd.zst"
    assert asset_suffix(Path("density.mrc.gz")) == ".mrc.gz"
    assert asset_suffix(Path("particles.mrcs.zst")) == ".mrcs.zst"
    assert asset_suffix(Path("slide.ome.tif")) == ".ome.tif"
    assert asset_suffix(Path("slide.ome.tiff")) == ".ome.tiff"
    assert asset_suffix(Path("features.geojson.zst")) == ".geojson.zst"
    assert asset_suffix(Path("features.geojsonseq.gz")) == ".geojsonseq.gz"
    assert asset_suffix(Path("roads.osm.pbf")) == ".osm.pbf"
    assert asset_suffix(Path("tiles.mbtiles")) == ".mbtiles"
    assert asset_suffix(Path("reads.fastq.gz")) == ".fastq.gz"
    assert asset_suffix(Path("proteins.fa.gz")) == ".fa.gz"
    assert asset_suffix(Path("train.csv.gz")) == ".csv.gz"
    assert asset_suffix(Path("records.jsonl.zst")) == ".jsonl.zst"
    assert asset_suffix(Path("diagram.svg.gz")) == ".svg.gz"
    assert asset_suffix(Path("submission.tar.gz")) == ".tar.gz"
    assert asset_suffix(Path("submission.tar.bz2")) == ".tar.bz2"
    assert asset_suffix(Path("submission.tar.xz")) == ".tar.xz"
    assert asset_suffix(Path("submission.tar.zst")) == ".tar.zst"
    assert asset_suffix(Path("submission.tzst")) == ".tzst"


def test_artifact_suffix_and_stem_preserve_archive_and_asset_compounds() -> None:
    assert artifact_suffix(Path("submission.tar.gz")) == ".tar.gz"
    assert artifact_stem(Path("submission.tar.gz")) == "submission"
    assert artifact_suffix(Path("answers.nii.gz")) == ".nii.gz"
    assert artifact_stem(Path("answers.nii.gz")) == "answers"
    assert artifact_suffix(Path("scan.dcm.gz")) == ".dcm.gz"
    assert artifact_stem(Path("scan.dcm.gz")) == "scan"
    assert artifact_suffix(Path("scan.ima.zst")) == ".ima.zst"
    assert artifact_stem(Path("scan.ima.zst")) == "scan"
    assert artifact_suffix(Path("predictions.zarr")) == ".zarr"
    assert artifact_stem(Path("predictions.zarr")) == "predictions"
    assert artifact_suffix(Path("labels.ome.zarr")) == ".ome.zarr"
    assert artifact_stem(Path("labels.ome.zarr")) == "labels"
    assert artifact_suffix(Path("volumes.n5")) == ".n5"
    assert artifact_stem(Path("volumes.n5")) == "volumes"
    assert artifact_suffix(Path("scene.usda.gz")) == ".usda.gz"
    assert artifact_stem(Path("scene.usda.gz")) == "scene"
    assert artifact_suffix(Path("scene.usdz")) == ".usdz"
    assert artifact_stem(Path("scene.usdz")) == "scene"
    assert artifact_suffix(Path("scene.abc")) == ".abc"
    assert artifact_stem(Path("scene.abc")) == "scene"
    assert artifact_suffix(Path("scene.blend")) == ".blend"
    assert artifact_stem(Path("scene.blend")) == "scene"
    assert artifact_suffix(Path("model.safetensors.index.json")) == ".safetensors.index.json"
    assert artifact_stem(Path("model.safetensors.index.json")) == "model"
    assert artifact_suffix(Path("model.ckpt.index")) == ".ckpt.index"
    assert artifact_stem(Path("model.ckpt.index")) == "model"


def test_archive_container_classifies_known_archive_suffixes() -> None:
    assert archive_container([".tar.gz"]) == "tar"
    assert archive_container([".tgz"]) == "tar"
    assert archive_container([".tar.zst"]) == "tar"
    assert archive_container([".7z"]) == "7z"
    assert archive_container([".rar"]) == "rar"
    assert archive_container([".zip"]) == "zip"
    assert archive_container([".csv"], default="file") == "file"
    assert archive_container([".csv"]) is None


def test_infer_asset_modality_prioritizes_medical_imaging_over_generic_archive(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "scan.svs").write_bytes(b"scan")
    (data_dir / "bundle.tar.gz").write_bytes(b"archive")

    assert infer_asset_modality(data_dir, include_code_artifact=True) == "medical_imaging"


def test_infer_asset_modality_recognizes_wsi_and_microscopy_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "slide.czi").write_bytes(b"microscopy")
    (data_dir / "tile.mrxs").write_bytes(b"wsi")
    (data_dir / "field.nd2").write_bytes(b"nd2")
    (data_dir / "section.dm4").write_bytes(b"dm4")

    assert infer_asset_modality(data_dir) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".qptiff"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".lif"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".lsm"}, include_code_artifact=True) == "medical_imaging"


def test_infer_asset_modality_recognizes_compressed_medical_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "scan.dcm.gz").write_bytes(b"compressed dicom")
    (data_dir / "slice.ima.xz").write_bytes(b"compressed ima")
    (data_dir / "mask.nrrd.zst").write_bytes(b"compressed nrrd")

    assert infer_asset_modality(data_dir) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".dicom.bz2"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".ima.gz"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".nii.xz"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".hdr.gz"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".img"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".mha.zst"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".mrc.gz"}, include_code_artifact=True) == "medical_imaging"
    assert infer_asset_modality_from_extensions({".ccp4.zst"}, include_code_artifact=True) == "medical_imaging"


def test_medical_image_suffix_groups_cover_compressed_variants() -> None:
    assert set(DICOM_IMAGE_BASE_SUFFIXES) == {".dcm", ".dicom", ".ima"}
    assert set(NIFTI_IMAGE_BASE_SUFFIXES) == {".nii"}
    assert set(ANALYZE_IMAGE_PAIR_BASE_SUFFIXES) == {".hdr", ".img"}
    assert set(MEDICAL_HEADER_IMAGE_BASE_SUFFIXES) == {".mha", ".mhd", ".nhdr", ".nrrd"}
    assert set(CRYOEM_IMAGE_BASE_SUFFIXES) == {".ccp4", ".mrc", ".mrcs"}
    assert {".dm3", ".dm4", ".lif", ".lsm", ".nd2", ".oir", ".vsi", ".zvi"} == set(MICROSCOPY_IMAGE_BASE_SUFFIXES)
    for base in ANALYZE_IMAGE_PAIR_BASE_SUFFIXES:
        assert base in ANALYZE_IMAGE_PAIR_SUFFIXES
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in ANALYZE_IMAGE_PAIR_SUFFIXES
    for base in DICOM_IMAGE_BASE_SUFFIXES:
        assert base in DICOM_IMAGE_SUFFIXES
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in DICOM_IMAGE_SUFFIXES
    for base in NIFTI_IMAGE_BASE_SUFFIXES:
        assert base in NIFTI_IMAGE_SUFFIXES
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in NIFTI_IMAGE_SUFFIXES
    for base in MEDICAL_HEADER_IMAGE_BASE_SUFFIXES:
        assert base in MEDICAL_HEADER_IMAGE_SUFFIXES
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in MEDICAL_HEADER_IMAGE_SUFFIXES
    for base in CRYOEM_IMAGE_BASE_SUFFIXES:
        assert base in CRYOEM_IMAGE_SUFFIXES
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in CRYOEM_IMAGE_SUFFIXES
    assert DICOM_IMAGE_SUFFIXES.issubset(MEDICAL_IMAGE_SUFFIXES)
    assert ANALYZE_IMAGE_PAIR_SUFFIXES.issubset(MEDICAL_IMAGE_SUFFIXES)
    assert CRYOEM_IMAGE_SUFFIXES.issubset(MEDICAL_IMAGE_SUFFIXES)
    assert NIFTI_IMAGE_SUFFIXES.issubset(MEDICAL_IMAGE_SUFFIXES)
    assert MEDICAL_HEADER_IMAGE_SUFFIXES.issubset(MEDICAL_IMAGE_SUFFIXES)
    assert set(MICROSCOPY_IMAGE_BASE_SUFFIXES).issubset(MEDICAL_IMAGE_SUFFIXES)


def test_asset_suffix_groups_cover_compressed_variants() -> None:
    for base in (".svg",):
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in VECTOR_IMAGE_SUFFIXES
    for base in (".fits", ".fit", ".fts"):
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in SCIENTIFIC_ARRAY_SUFFIXES
    for base in (
        ".mesh",
        ".msh",
        ".brep",
        ".ifc",
        ".iges",
        ".igs",
        ".ply",
        ".obj",
        ".off",
        ".pcd",
        ".pts",
        ".ptx",
        ".stl",
        ".step",
        ".stp",
        ".vtk",
        ".vtp",
        ".vtu",
        ".xyz",
    ):
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in POINT_CLOUD_SUFFIXES
    for base in (".rtf", ".html", ".htm", ".md", ".tex", ".rst", ".adoc", ".srt", ".vtt"):
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in DOCUMENT_SUFFIXES
    for base in (
        ".pdb",
        ".cif",
        ".mmcif",
        ".sdf",
        ".mol",
        ".mol2",
        ".smi",
        ".smiles",
        ".inchi",
        ".selfies",
        ".rxn",
        ".fasta",
        ".fastq",
        ".fq",
        ".bed",
        ".gff",
        ".gff3",
        ".gtf",
        ".sam",
        ".vcf",
    ):
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in BIO_STRUCTURE_SUFFIXES
    for base in (".graphml", ".gexf", ".gml", ".mtx", ".edgelist", ".edges"):
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in GRAPH_SUFFIXES
    for base in GRAPH_RDF_BASE_SUFFIXES:
        assert base in GRAPH_SUFFIXES
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in GRAPH_SUFFIXES
    for compression in ASSET_COMPRESSION_SUFFIXES:
        assert f".kml{compression}" in GEOSPATIAL_SUFFIXES
    for base in GEOSPATIAL_RASTER_BASE_SUFFIXES:
        assert base in GEOSPATIAL_SUFFIXES
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in GEOSPATIAL_SUFFIXES
    assert ".hea.gz" in SIGNAL_SUFFIXES
    for base in (".fif", ".set", ".vhdr", ".vmrk"):
        for compression in ASSET_COMPRESSION_SUFFIXES:
            assert f"{base}{compression}" in SIGNAL_SUFFIXES


def test_asset_modality_exposes_metadata_base_suffix_groups() -> None:
    assert POINT_CLOUD_TEXT_METADATA_SUFFIXES == (".xyz", ".pts", ".ptx")
    assert set(POINT_CLOUD_TEXT_METADATA_SUFFIXES) <= POINT_CLOUD_SUFFIXES
    assert {
        ".3ds",
        ".brep",
        ".fbx",
        ".ifc",
        ".iges",
        ".igs",
        ".mesh",
        ".msh",
        ".step",
        ".stp",
        ".vtk",
        ".vtp",
        ".vtu",
    } <= POINT_CLOUD_SUFFIXES
    assert GRAPH_XML_BASE_SUFFIXES == (".graphml", ".gexf")
    assert GRAPH_EDGE_LIST_BASE_SUFFIXES == (".edgelist", ".edges")
    assert GRAPH_RDF_BASE_SUFFIXES == (".jsonld", ".nq", ".nt", ".owl", ".rdf", ".trig", ".ttl")
    assert set(GRAPH_XML_BASE_SUFFIXES) | set(GRAPH_EDGE_LIST_BASE_SUFFIXES) | set(GRAPH_RDF_BASE_SUFFIXES) <= (
        GRAPH_SUFFIXES
    )
    assert BIO_FASTQ_BASE_SUFFIXES == (".fastq", ".fq")
    assert set(BIO_FASTQ_BASE_SUFFIXES) <= set(BIO_SEQUENCE_BASE_SUFFIXES)
    assert {".bam", ".bcf", ".cram", ".pod5"} <= set(BIO_GENOMIC_BASE_SUFFIXES)
    assert BIO_GENOMIC_TEXT_BASE_SUFFIXES == (".bed", ".gff", ".gff3", ".gtf", ".sam", ".vcf")
    assert set(BIO_GENOMIC_TEXT_BASE_SUFFIXES) <= set(BIO_GENOMIC_BASE_SUFFIXES)
    assert BIO_CHEM_TEXT_BASE_SUFFIXES == (".smi", ".smiles", ".inchi", ".selfies", ".rxn")
    assert BIO_PDB_STRUCTURE_BASE_SUFFIXES == (".pdb", ".cif", ".mmcif")
    assert BIO_MOL_STRUCTURE_BASE_SUFFIXES == (".sdf", ".mol")
    assert (
        set(BIO_SEQUENCE_BASE_SUFFIXES)
        | set(BIO_CHEM_TEXT_BASE_SUFFIXES)
        | set(BIO_GENOMIC_BASE_SUFFIXES)
        | set(BIO_PDB_STRUCTURE_BASE_SUFFIXES)
        | set(BIO_MOL_STRUCTURE_BASE_SUFFIXES)
        | {".mol2"}
    ) <= BIO_STRUCTURE_SUFFIXES
    assert DOCUMENT_HTML_BASE_SUFFIXES == (".html", ".htm")
    assert {".adoc", ".html", ".htm", ".md", ".rst", ".rtf", ".srt", ".tex", ".txt", ".vtt"} == set(
        DOCUMENT_TEXT_METADATA_SUFFIXES
    )
    assert set(DOCUMENT_TEXT_METADATA_SUFFIXES) - {".txt"} <= DOCUMENT_SUFFIXES
    assert {".json", ".json.gz", ".txt", ".txt.gz", ".xml", ".xml.gz"} <= ANNOTATION_FILE_SUFFIXES
    assert ANNOTATION_JSON_SUFFIXES <= ANNOTATION_FILE_SUFFIXES
    assert ANNOTATION_TEXT_SUFFIXES <= ANNOTATION_FILE_SUFFIXES
    assert ANNOTATION_XML_SUFFIXES <= ANNOTATION_FILE_SUFFIXES
    assert {"annotations", "labels", "bboxes", "segmentations"} <= ANNOTATION_DIR_NAME_TOKENS
    assert {"coco", "cvat", "instances", "labelme", "labelstudio", "pascal", "voc", "yolo"} <= (
        ANNOTATION_FILE_NAME_TOKENS
    )
    assert {".cnt", ".eeg", ".fdt", ".fif", ".set", ".trc", ".vhdr", ".vmrk"} == set(SIGNAL_NEUROPHYS_BASE_SUFFIXES)


def test_infer_asset_modality_recognizes_modern_media_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame.jxl").write_bytes(b"image")

    assert infer_asset_modality(data_dir) == "image"
    assert infer_asset_modality_from_extensions({".avif"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".exr"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".heif"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".jxl"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".ppm"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".svg"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".svg.gz"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".svgz"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".tga"}, include_code_artifact=True) == "image"
    assert infer_asset_modality_from_extensions({".mid"}, include_code_artifact=True) == "audio"
    assert infer_asset_modality_from_extensions({".opus"}, include_code_artifact=True) == "audio"
    assert infer_asset_modality_from_extensions({".aiff"}, include_code_artifact=True) == "audio"
    assert infer_asset_modality_from_extensions({".wma"}, include_code_artifact=True) == "audio"
    assert infer_asset_modality_from_extensions({".3gp"}, include_code_artifact=True) == "video"
    assert infer_asset_modality_from_extensions({".flv"}, include_code_artifact=True) == "video"
    assert infer_asset_modality_from_extensions({".m4v"}, include_code_artifact=True) == "video"
    assert infer_asset_modality_from_extensions({".mpg"}, include_code_artifact=True) == "video"
    assert infer_asset_modality_from_extensions({".wmv"}, include_code_artifact=True) == "video"
    assert infer_asset_modality_from_extensions({".edf"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".hea.gz"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".nwb"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".vhdr"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".fif.gz"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".ttl.gz"}, include_code_artifact=True) == "graph"
    assert infer_asset_modality_from_extensions({".jsonld"}, include_code_artifact=True) == "graph"


def test_infer_asset_modality_recognizes_genomics_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "variants.vcf.gz").write_bytes(b"##fileformat=VCFv4.3\n")
    (data_dir / "reads.bam").write_bytes(b"BAM\x01")

    assert infer_asset_modality(data_dir) == "bio"
    assert infer_asset_modality_from_extensions({".sam.gz"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".gff3.zst"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".bcf"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".cram"}, include_code_artifact=True) == "bio"


def test_infer_asset_modality_can_include_code_and_archive_artifacts() -> None:
    assert infer_asset_modality_from_extensions({".py"}, include_code_artifact=True) == "code"
    assert infer_asset_modality_from_extensions({".tar.gz"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".7z"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".rar"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".tar.gz"}, include_code_artifact=False) == "unknown"
    assert infer_asset_modality_from_extensions({".7z"}, include_code_artifact=False) == "unknown"


def test_infer_asset_modality_recognizes_huggingface_model_artifact_folder(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "model.safetensors").write_bytes(b"weights")
    (data_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (data_dir / "config.json").write_text("{}", encoding="utf-8")
    (data_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (data_dir / "vocab.txt").write_text("token\n", encoding="utf-8")

    assert infer_asset_modality(data_dir, include_code_artifact=True) == "artifact"
    assert infer_asset_modality(data_dir, include_code_artifact=False) == "tabular"
    assert asset_suffix(data_dir / "model.safetensors.index.json") == ".safetensors.index.json"
    assert is_model_artifact_path(data_dir / "tokenizer.json")
    assert is_model_artifact_path(data_dir / "spiece.model")
    assert ".model" not in MODEL_ARTIFACT_SUFFIXES
    assert ".bst" in MODEL_ARTIFACT_SUFFIXES
    assert ".cbm" in MODEL_ARTIFACT_SUFFIXES
    assert ".engine" in MODEL_ARTIFACT_SUFFIXES
    assert ".plan" in MODEL_ARTIFACT_SUFFIXES
    assert ".rknn" in MODEL_ARTIFACT_SUFFIXES
    assert ".hef" in MODEL_ARTIFACT_SUFFIXES
    assert ".dlc" in MODEL_ARTIFACT_SUFFIXES
    assert ".blob" in MODEL_ARTIFACT_SUFFIXES
    assert ".mlmodel" in MODEL_ARTIFACT_SUFFIXES
    assert ".pmml" in MODEL_ARTIFACT_SUFFIXES
    assert ".spm" in MODEL_ARTIFACT_SUFFIXES
    assert ".skops" in MODEL_ARTIFACT_SUFFIXES
    assert ".ubj" in MODEL_ARTIFACT_SUFFIXES
    assert ".xgb" in MODEL_ARTIFACT_SUFFIXES
    assert ".safetensors.index.json" in MODEL_ARTIFACT_COMPOUND_SUFFIXES
    assert ".ckpt.index" in MODEL_ARTIFACT_COMPOUND_SUFFIXES
    assert ".safetensors.index.json" in DATA_ASSET_SUFFIXES
    assert ".ckpt.index" in DATA_ASSET_SUFFIXES
    assert "tokenizer.json" in MODEL_ARTIFACT_FILENAMES
    assert "adapter_config.json" in MODEL_ARTIFACT_FILENAMES
    assert "preprocessor_config.json" in MODEL_ARTIFACT_FILENAMES
    assert "adapter_config.json" in MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES
    assert "vocab.txt" in MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES
    assert {
        "adapter",
        "booster",
        "catboost",
        "checkpoint",
        "coreml",
        "model",
        "pmml",
        "pytorch",
        "sklearn",
        "weights",
        "xgboost",
    }.issubset(MODEL_ARTIFACT_NAME_TOKENS)


def test_infer_asset_modality_recognizes_peft_adapter_artifact_folder(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (data_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (data_dir / "preprocessor_config.json").write_text('{"do_resize": true}', encoding="utf-8")
    (data_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    assert infer_asset_modality(data_dir, include_code_artifact=True) == "artifact"
    assert is_model_artifact_path(data_dir / "adapter_config.json")
    assert is_model_artifact_path(data_dir / "chat_template.jinja")


def test_infer_asset_modality_keeps_real_json_data_tabular_with_model_sidecar(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text('[{"id": 1, "target": 0}]', encoding="utf-8")
    (data_dir / "model.safetensors").write_bytes(b"weights")

    assert infer_asset_modality(data_dir, include_code_artifact=True) == "tabular"


def test_infer_asset_modality_recognizes_document_assets_as_text(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "prompt.pdf").write_bytes(b"%PDF")

    assert infer_asset_modality(data_dir) == "text"
    assert infer_asset_modality_from_extensions({".docx"}, include_code_artifact=True) == "text"
    assert infer_asset_modality_from_extensions({".epub"}, include_code_artifact=True) == "text"
    assert infer_asset_modality_from_extensions({".odt"}, include_code_artifact=True) == "text"
    assert infer_asset_modality_from_extensions({".pptx"}, include_code_artifact=True) == "text"
    assert infer_asset_modality_from_extensions({".tex.gz"}, include_code_artifact=True) == "text"


def test_infer_asset_modality_recognizes_signal_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "record.edf").write_bytes(b"edf")
    (data_dir / "record.hea.gz").write_bytes(b"compressed header")

    assert infer_asset_modality(data_dir) == "signal"
    assert infer_asset_modality_from_extensions({".bdf"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".tdms"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".abf"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".set"}, include_code_artifact=True) == "signal"
    assert infer_asset_modality_from_extensions({".cnt"}, include_code_artifact=True) == "signal"


def test_infer_asset_modality_recognizes_detection_annotation_datasets(tmp_path: Path) -> None:
    coco_dir = tmp_path / "coco"
    annotations_dir = coco_dir / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "instances_train.json").write_text(
        '{"images": [], "annotations": [], "categories": []}',
        encoding="utf-8",
    )

    yolo_dir = tmp_path / "yolo"
    labels_dir = yolo_dir / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "image_001.txt.gz").write_bytes(b"compressed")

    tabular_dir = tmp_path / "tabular"
    tabular_dir.mkdir()
    (tabular_dir / "train.json").write_text('[{"id": 1, "target": 0}]', encoding="utf-8")

    assert infer_asset_modality(coco_dir) == "image"
    assert infer_asset_modality(yolo_dir) == "image"
    assert infer_asset_modality(tabular_dir) == "tabular"
    assert is_detection_annotation_path(annotations_dir / "instances_train.json")
    assert is_detection_annotation_path(annotations_dir / "pascal_voc.xml")
    assert is_detection_annotation_path(annotations_dir / "cvat.xml.gz")
    assert is_detection_annotation_path(annotations_dir / "label-studio.xml")
    assert is_detection_annotation_path(labels_dir / "image_001.txt.gz")
    assert not is_detection_annotation_path(tabular_dir / "train.json")
    assert not is_detection_annotation_path(tabular_dir / "train.xml")


def test_infer_asset_modality_recognizes_geospatial_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "parcels.geojson").write_text("{}", encoding="utf-8")

    assert TABULAR_GEOJSON_SUFFIXES <= GEOSPATIAL_SUFFIXES
    assert infer_asset_modality(data_dir) == "geospatial"
    assert infer_asset_modality_from_extensions({".geojson.zst"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".geojsonl.gz"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".geojsonseq"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".topojson.zst"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".osm"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".osm.pbf"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".mbtiles"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".pmtiles"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".mvt"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".kml.gz"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".shp"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".gpkg"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".geopackage"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".bil"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".bsq"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".bip"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".hgt"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".dem.gz"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".dt2"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".ecw"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".sid.zst"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".mif"}, include_code_artifact=True) == "geospatial"
    assert infer_asset_modality_from_extensions({".vrt"}, include_code_artifact=True) == "geospatial"


def test_infer_asset_modality_recognizes_bio_structure_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "protein.pdb.gz").write_bytes(b"compressed")

    assert infer_asset_modality(data_dir) == "bio"
    assert infer_asset_modality_from_extensions({".sdf"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".sdf.xz"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".smi"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".smiles.gz"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".inchi"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".selfies"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".fasta"}, include_code_artifact=True) == "bio"
    assert infer_asset_modality_from_extensions({".fastq.gz"}, include_code_artifact=True) == "bio"


def test_infer_asset_modality_recognizes_graph_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "network.mtx.gz").write_bytes(b"compressed")

    assert infer_asset_modality(data_dir) == "graph"
    assert infer_asset_modality_from_extensions({".gexf"}, include_code_artifact=True) == "graph"
    assert infer_asset_modality_from_extensions({".edgelist"}, include_code_artifact=True) == "graph"
    assert infer_asset_modality_from_extensions({".graphml.gz"}, include_code_artifact=True) == "graph"


def test_infer_asset_modality_recognizes_scientific_array_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "weather.nc").write_bytes(b"netcdf")

    assert infer_asset_modality(data_dir) == "array"
    assert infer_asset_modality_from_extensions({".fits"}, include_code_artifact=True) == "array"
    assert infer_asset_modality_from_extensions({".fits.gz"}, include_code_artifact=True) == "array"
    assert infer_asset_modality_from_extensions({".nc4"}, include_code_artifact=True) == "array"
    assert infer_asset_modality_from_extensions({".grib2"}, include_code_artifact=True) == "array"


def test_infer_asset_modality_recognizes_array_store_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store_dir = data_dir / "counts.zarr"
    store_dir.mkdir(parents=True)
    (store_dir / ".zarray").write_text("{}", encoding="utf-8")

    assert infer_asset_modality(data_dir) == "array"
    assert infer_asset_modality_from_extensions({".h5ad"}, include_code_artifact=True) == "array"
    assert infer_asset_modality_from_extensions({".loom"}, include_code_artifact=True) == "array"
    assert infer_asset_modality_from_extensions({".zarr"}, include_code_artifact=True) == "array"
    assert infer_asset_modality_from_extensions({".ome.zarr"}, include_code_artifact=True) == "array"
    assert infer_asset_modality_from_extensions({".n5"}, include_code_artifact=True) == "array"


def test_infer_asset_modality_recognizes_additional_point_cloud_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "scan.ply.gz").write_bytes(b"compressed")

    assert infer_asset_modality(data_dir) == "point_cloud"
    assert infer_asset_modality_from_extensions({".dae.gz"}, include_code_artifact=True) == "point_cloud"
    assert infer_asset_modality_from_extensions({".x3d.xz"}, include_code_artifact=True) == "point_cloud"
    assert infer_asset_modality_from_extensions({".xyz"}, include_code_artifact=True) == "point_cloud"
    assert infer_asset_modality_from_extensions({".obj.xz"}, include_code_artifact=True) == "point_cloud"
    assert infer_asset_modality_from_extensions({".pts"}, include_code_artifact=True) == "point_cloud"
    assert infer_asset_modality_from_extensions({".ptx"}, include_code_artifact=True) == "point_cloud"
    assert infer_asset_modality_from_extensions({".off"}, include_code_artifact=True) == "point_cloud"


def test_infer_asset_modality_recognizes_tabular_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.parquet").write_bytes(b"PAR1")

    assert infer_asset_modality(data_dir) == "tabular"
    assert infer_asset_modality_from_extensions({".csv"}, include_code_artifact=True) == "tabular"
    assert infer_asset_modality_from_extensions({".csv.gz"}, include_code_artifact=True) == "tabular"
    assert infer_asset_modality_from_extensions({".jsonl.zst"}, include_code_artifact=True) == "tabular"
    assert infer_asset_modality_from_extensions({".xlsx"}, include_code_artifact=True) == "tabular"
    assert infer_asset_modality_from_extensions({".sqlite3"}, include_code_artifact=True) == "tabular"


def test_infer_asset_modality_prioritizes_heavy_assets_over_label_tables(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "image.png").write_bytes(b"png")
    (data_dir / "labels.csv").write_text("image,target\nimage.png,1\n", encoding="utf-8")

    assert infer_asset_modality(data_dir) == "image"


def test_infer_asset_modality_recognizes_model_artifacts_with_artifact_flag(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "model.safetensors").write_bytes(b"model")

    assert infer_asset_modality(data_dir) == "unknown"
    assert infer_asset_modality(data_dir, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".safetensors"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".gguf"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".msgpack"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".tflite"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".pb"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".joblib"}, include_code_artifact=True) == "artifact"
    assert infer_asset_modality_from_extensions({".safetensors.index.json"}, include_code_artifact=True) == "artifact"


def test_data_asset_suffixes_cover_runtime_data_assets_only() -> None:
    assert {
        ".jpg",
        ".avif",
        ".svg",
        ".svg.gz",
        ".svgz",
        ".heic",
        ".nii.gz",
        ".dcm.gz",
        ".dicom.bz2",
        ".ima.xz",
        ".nrrd.zst",
        ".czi",
        ".mrxs",
        ".qptiff",
        ".wav",
        ".opus",
        ".mp4",
        ".m4v",
        ".edf",
        ".hea.gz",
        ".npy",
        ".e57",
        ".glb",
        ".off",
        ".pts",
        ".csv",
        ".csv.gz",
        ".jsonl.zst",
        ".parquet",
        ".xlsx",
        ".sqlite3",
        ".onnx",
        ".safetensors",
        ".gguf",
        ".msgpack",
        ".tflite",
        ".pb",
        ".joblib",
        ".pdf",
        ".epub",
        ".odt",
        ".pptx",
        ".docx",
        ".geojson",
        ".gpkg",
        ".geopackage",
        ".kml",
        ".pdb",
        ".sdf",
        ".smi",
        ".smiles.gz",
        ".fasta",
        ".graphml",
        ".gexf",
        ".edgelist",
        ".nc",
        ".nc4",
        ".grib2",
        ".fits",
        ".h5ad",
        ".loom",
        ".mat",
        ".n5",
        ".ome.zarr",
        ".zarr",
    }.issubset(DATA_ASSET_SUFFIXES)
    assert VECTOR_IMAGE_SUFFIXES.issubset(IMAGE_SUFFIXES)
    assert DOCUMENT_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert MEDICAL_IMAGE_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert GEOSPATIAL_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert BIO_STRUCTURE_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert GRAPH_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert POINT_CLOUD_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert SCIENTIFIC_ARRAY_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert SIGNAL_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert DIRECTORY_ARRAY_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert TABULAR_DATA_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert MODEL_ARTIFACT_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert MODEL_ARTIFACT_COMPOUND_SUFFIXES.issubset(DATA_ASSET_SUFFIXES)
    assert DATA_ASSET_SUFFIXES.isdisjoint(CODE_SUFFIXES)
    assert DATA_ASSET_SUFFIXES.isdisjoint(ARCHIVE_SUFFIXES)
