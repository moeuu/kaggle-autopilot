from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest
import zstandard as zstd

from kagglebot.analyzer.analyze import analyze_competition
from kagglebot.paths import CompetitionPaths


def _write_rna_structure_analyzer_fixture(paths: CompetitionPaths) -> None:
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "target_id": ["RNA1", "RNA2"],
            "sequence": ["ACG", "GU"],
            "description": ["demo one", "demo two"],
        }
    ).to_csv(paths.data_dir / "train_sequences.csv", index=False)
    pd.DataFrame(
        {
            "target_id": ["RNA3"],
            "sequence": ["AC"],
            "description": ["demo test"],
        }
    ).to_csv(paths.data_dir / "test_sequences.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["RNA1_1", "RNA1_2", "RNA1_3", "RNA2_1", "RNA2_2"],
            "resname": ["A", "C", "G", "G", "U"],
            "resid": [1, 2, 3, 1, 2],
            "x_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_1": [1.5, 2.5, 3.5, 4.5, 5.5],
            "z_1": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(paths.data_dir / "train_labels.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["RNA3_1", "RNA3_2"],
            "resname": ["A", "C"],
            "resid": [1, 2],
            "x_1": [0.0, 0.0],
            "y_1": [0.0, 0.0],
            "z_1": [0.0, 0.0],
            "x_2": [0.0, 0.0],
            "y_2": [0.0, 0.0],
            "z_2": [0.0, 0.0],
        }
    ).to_csv(paths.data_dir / "sample_submission.csv", index=False)


def test_analyze_competition_uses_current_competition_paths(tmp_path):
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "feature": [0.5, 0.6]}).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "label": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    assert result.analysis_path == paths.analysis_path
    assert result.analysis_path.exists()
    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["slug"] == "demo"
    assert payload["schema"]["train_path"] == str(paths.data_raw / "train.csv")


def test_analyze_competition_reads_tsv_train_test_and_sample(tmp_path):
    paths = CompetitionPaths(slug="demo-tsv", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(paths.data_dir / "train.tsv", sep="\t", index=False)
    pd.DataFrame({"id": [5, 6], "feature": [0.5, 0.6]}).to_csv(
        paths.data_dir / "test.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame({"id": [5, 6], "label": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.tsv",
        sep="\t",
        index=False,
    )

    result = analyze_competition(
        slug="demo-tsv",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["slug"] == "demo-tsv"
    assert payload["schema"]["train_path"] == str(paths.data_raw / "train.tsv")
    assert payload["schema"]["sample_submission_path"] == str(paths.data_raw / "sample_submission.tsv")


def test_analyze_competition_reads_compressed_jsonl_train_test_and_sample(tmp_path):
    paths = CompetitionPaths(slug="demo-jsonl-gz", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    with gzip.open(paths.data_dir / "train.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    '{"id":1,"feature":0.1,"label":0}',
                    '{"id":2,"feature":0.2,"label":1}',
                    '{"id":3,"feature":0.3,"label":0}',
                    '{"id":4,"feature":0.4,"label":1}',
                ]
            )
            + "\n"
        )
    with gzip.open(paths.data_dir / "test.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"id":5,"feature":0.5}\n{"id":6,"feature":0.6}\n')
    with gzip.open(paths.data_dir / "sample_submission.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"id":5,"label":0.0}\n{"id":6,"label":0.0}\n')

    result = analyze_competition(
        slug="demo-jsonl-gz",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["train_path"] == str(paths.data_raw / "train.jsonl.gz")
    assert payload["schema"]["sample_submission_path"] == str(paths.data_raw / "sample_submission.jsonl.gz")
    assert payload["task"] == "classification"


def test_analyze_competition_reads_zstd_compressed_csv_train_test_and_sample(tmp_path):
    paths = CompetitionPaths(slug="demo-csv-zst", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    compressor = zstd.ZstdCompressor()
    (paths.data_dir / "train.csv.zst").write_bytes(
        compressor.compress(b"id,feature,label\n1,0.1,0\n2,0.2,1\n3,0.3,0\n4,0.4,1\n")
    )
    (paths.data_dir / "test.csv.zst").write_bytes(compressor.compress(b"id,feature\n5,0.5\n6,0.6\n"))
    (paths.data_dir / "sample_submission.csv.zst").write_bytes(compressor.compress(b"id,label\n5,0.0\n6,0.0\n"))

    result = analyze_competition(
        slug="demo-csv-zst",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["train_path"] == str(paths.data_raw / "train.csv.zst")
    assert payload["schema"]["sample_submission_path"] == str(paths.data_raw / "sample_submission.csv.zst")
    assert payload["task"] == "classification"


def test_analyze_competition_writes_non_tabular_asset_fallback_analysis(tmp_path):
    paths = CompetitionPaths(slug="demo-images", artifacts_dir=tmp_path / "artifacts")
    train_dir = paths.data_dir / "images" / "train"
    test_dir = paths.data_dir / "images" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "cat_001.jpg").write_bytes(b"fake-jpeg")
    (test_dir / "test_001.jpg").write_bytes(b"fake-jpeg")

    result = analyze_competition(
        slug="demo-images",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    assert result.analysis_path == paths.analysis_path
    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["slug"] == "demo-images"
    assert payload["type"] == "image"
    assert payload["task"] == "image"
    assert payload["prediction_kind"] == "artifact"
    assert payload["schema"]["target_columns"] == []
    assert payload["schema"]["train_path"] == str(paths.data_raw)
    assert "reference_notebook_baseline" in payload["strategy"]["models"]
    assert any("non-tabular asset fallback" in item for item in payload["assumptions"])


def test_analyze_competition_handles_object_detection_submission_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-detection", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2],
            "image_path": ["train/a.jpg", "train/b.jpg"],
            "prediction_string": ["0 0.9 0.5 0.5 0.2 0.2", "-"],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [3, 4], "image_path": ["test/c.jpg", "test/d.jpg"]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [3, 4], "prediction_string": ["-", "-"]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-detection",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["prediction_string"]
    assert payload["task"] == "object_detection"
    assert payload["prediction_kind"] == "prediction_string"
    assert payload["metric"] == "map"
    assert payload["metric_direction"] == "maximize"
    assert "parse_prediction_string_contract" in payload["strategy"]["preprocessing"]
    assert "object_detection_router" in payload["strategy"]["models"]
    assert any("object-detection prediction-string" in item for item in payload["assumptions"])


def test_analyze_competition_handles_segmentation_submission_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-segmentation", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["train_1", "train_2", "train_3"],
            "image_path": ["train/a.png", "train/b.png", "train/c.png"],
            "EncodedPixels": ["1 2 10 3", "", "4 1 20 2"],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": ["test_1", "test_2"],
            "image_path": ["test/a.png", "test/b.png"],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": ["test_1", "test_2"], "EncodedPixels": ["", ""]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-segmentation",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["EncodedPixels"]
    assert payload["task"] == "segmentation"
    assert payload["prediction_kind"] == "rle"
    assert payload["metric"] == "dice"
    assert payload["metric_direction"] == "maximize"
    assert "parse_mask_or_rle_submission_contract" in payload["strategy"]["preprocessing"]
    assert "rle_empty_mask_baseline" in payload["strategy"]["models"]
    assert any("segmentation mask/RLE" in item for item in payload["assumptions"])


def test_analyze_competition_non_tabular_fallback_records_data_sample_alias(tmp_path):
    paths = CompetitionPaths(slug="demo-image-template", artifacts_dir=tmp_path / "artifacts")
    train_dir = paths.data_dir / "images" / "train"
    test_dir = paths.data_dir / "images" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "cat_001.jpg").write_bytes(b"fake-jpeg")
    (test_dir / "test_001.jpg").write_bytes(b"fake-jpeg")
    (paths.data_dir / "AnswerTemplate.csv").write_text("image_id,label\ntest_001,cat\n", encoding="utf-8")

    result = analyze_competition(
        slug="demo-image-template",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["type"] == "image"
    assert payload["prediction_kind"] == "artifact"
    assert payload["schema"]["sample_submission_path"] == str(paths.data_raw / "AnswerTemplate.csv")
    assert any("AnswerTemplate.csv" in item for item in payload["assumptions"])


def test_analyze_competition_handles_rna_structure_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-rna-structure", artifacts_dir=tmp_path / "artifacts")
    _write_rna_structure_analyzer_fixture(paths)

    result = analyze_competition(
        slug="demo-rna-structure",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["type"] == "rna_structure"
    assert payload["task"] == "rna_structure"
    assert payload["prediction_kind"] == "residue_coordinates"
    assert payload["metric"] == "rmse"
    assert payload["metric_direction"] == "minimize"
    assert payload["schema"]["train_path"] == str(paths.data_raw / "train_sequences.csv")
    assert payload["schema"]["test_path"] == str(paths.data_raw / "test_sequences.csv")
    assert payload["schema"]["target_columns"] == ["x_1", "y_1", "z_1", "x_2", "y_2", "z_2"]
    assert payload["rna_structure"]["train_labels_path"] == str(paths.data_raw / "train_labels.csv")
    assert payload["rna_structure"]["sample_anchor_columns"] == ["ID", "resname", "resid"]
    assert "validate_coordinate_triplets" in payload["strategy"]["preprocessing"]
    assert "rna_coordinate_mean_baseline" in payload["strategy"]["models"]
    assert any("RNA sequence/structure layout inferred" in item for item in payload["assumptions"])


def test_analyze_competition_preserves_geospatial_modality(tmp_path):
    paths = CompetitionPaths(slug="demo-geospatial", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "latitude": [35.68, 34.69, 43.06, 33.59],
            "longitude": [139.76, 135.50, 141.35, 130.40],
            "feature": [10.0, 11.0, 12.0, 13.0],
            "target": [0, 1, 0, 1],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            "latitude": [35.18, 34.05],
            "longitude": [136.90, 131.00],
            "feature": [14.0, 15.0],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "target": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-geospatial",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["type"] == "tabular"
    assert payload["task"] == "geospatial"
    assert payload["metric"] == "accuracy"
    assert "derive_spatial_distance_and_bbox_features" in payload["strategy"]["preprocessing"]
    assert "geospatial_feature_gbdt" in payload["strategy"]["models"]
    assert any("geospatial modality inferred" in item for item in payload["assumptions"])


def test_analyze_competition_preserves_graph_modality(tmp_path):
    paths = CompetitionPaths(slug="demo-graph", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "source_node": ["A", "B", "C", "D"],
            "target_node": ["B", "C", "D", "E"],
            "weight": [0.4, 0.2, 0.8, 0.7],
            "label": [1, 0, 1, 0],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            "source_node": ["E", "F"],
            "target_node": ["F", "G"],
            "weight": [0.1, 0.9],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "label": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-graph",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["task"] == "graph"
    assert payload["metric"] == "accuracy"
    assert "derive_degree_and_topology_features" in payload["strategy"]["preprocessing"]
    assert "graph_topology_feature_gbdt" in payload["strategy"]["models"]
    assert any("graph modality inferred" in item for item in payload["assumptions"])


def test_analyze_competition_preserves_signal_modality(tmp_path):
    paths = CompetitionPaths(slug="demo-signal", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "waveform_path": ["signals/a.edf", "signals/b.hea.gz", "signals/c.nwb", "signals/d.tdms"],
            "feature": [0.4, 0.2, 0.8, 0.1],
            "target": [1, 0, 1, 0],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            "waveform_path": ["signals/e.edf", "signals/f.hea.gz"],
            "feature": [0.3, 0.9],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "target": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-signal",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["task"] == "signal"
    assert payload["metric"] == "accuracy"
    assert "derive_waveform_summary_and_frequency_features" in payload["strategy"]["preprocessing"]
    assert "signal_statistical_feature_gbdt" in payload["strategy"]["models"]
    assert any("signal modality inferred" in item for item in payload["assumptions"])


def test_analyze_competition_preserves_bio_modality(tmp_path):
    paths = CompetitionPaths(slug="demo-bio", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "smiles": ["CCO", "CN(C)C=O", "c1ccccc1", "O=C=O"],
            "feature": [0.4, 0.2, 0.8, 0.1],
            "target": [1, 0, 1, 0],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            "smiles": ["CCN", "CCCl"],
            "feature": [0.7, 0.3],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "target": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-bio",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["task"] == "bio"
    assert payload["metric"] == "accuracy"
    assert "derive_kmer_or_molecular_descriptor_features" in payload["strategy"]["preprocessing"]
    assert "sequence_kmer_feature_gbdt" in payload["strategy"]["models"]
    assert any("bio modality inferred" in item for item in payload["assumptions"])


@pytest.mark.parametrize(
    ("slug", "feature_column", "values", "expected_task", "expected_preprocessing", "expected_model"),
    [
        (
            "demo-image-ref",
            "filename",
            ["train/a.jpg", "train/b.png", "train/c.webp", "train/d.jpg"],
            "image",
            "derive_image_metadata_features",
            "image_metadata_feature_gbdt",
        ),
        (
            "demo-audio-ref",
            "audio_path",
            ["case_001", "case_002", "case_003", "case_004"],
            "audio",
            "derive_audio_metadata_features",
            "audio_metadata_feature_gbdt",
        ),
        (
            "demo-video-ref",
            "clip_filename",
            ["case_001", "case_002", "case_003", "case_004"],
            "video",
            "derive_video_metadata_features",
            "video_metadata_feature_gbdt",
        ),
        (
            "demo-medical-ref",
            "scan_path",
            ["case_001", "case_002", "case_003", "case_004"],
            "medical_imaging",
            "derive_medical_header_metadata_features",
            "medical_image_metadata_gbdt",
        ),
        (
            "demo-array-ref",
            "array_file",
            ["case_001", "case_002", "case_003", "case_004"],
            "array",
            "derive_array_shape_and_summary_features",
            "array_shape_stat_feature_gbdt",
        ),
        (
            "demo-point-ref",
            "lidar_file",
            ["case_001", "case_002", "case_003", "case_004"],
            "point_cloud",
            "derive_point_count_and_bbox_features",
            "point_cloud_metadata_feature_gbdt",
        ),
        (
            "demo-annotation-ref",
            "annotation_path",
            ["case_001", "case_002", "case_003", "case_004"],
            "annotation",
            "derive_annotation_count_bbox_mask_features",
            "annotation_format_feature_gbdt",
        ),
    ],
)
def test_analyze_competition_preserves_asset_reference_modalities(
    tmp_path,
    slug: str,
    feature_column: str,
    values: list[str],
    expected_task: str,
    expected_preprocessing: str,
    expected_model: str,
):
    paths = CompetitionPaths(slug=slug, artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            feature_column: values,
            "feature": [0.4, 0.2, 0.8, 0.1],
            "target": [1, 0, 1, 0],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            feature_column: values[:2],
            "feature": [0.7, 0.3],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "target": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug=slug,
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["task"] == expected_task
    assert payload["metric"] == "accuracy"
    assert expected_preprocessing in payload["strategy"]["preprocessing"]
    assert expected_model in payload["strategy"]["models"]
    assert any(f"{expected_task} modality inferred" in item for item in payload["assumptions"])


def test_analyze_competition_handles_single_label_probability_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-proba-cols", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "label": ["cat", "dog", "bird", "cat", "dog", "bird"],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [7, 8], "feature": [0.7, 0.8]}).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame(
        {
            "id": [7, 8],
            "class_bird": [1 / 3, 1 / 3],
            "class_cat": [1 / 3, 1 / 3],
            "class_dog": [1 / 3, 1 / 3],
        }
    ).to_csv(paths.data_dir / "sample_submission.csv", index=False)

    result = analyze_competition(
        slug="demo-proba-cols",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["label"]
    assert payload["prediction_kind"] == "probability_columns"
    assert payload["metric"] == "logloss"
    assert payload["metric_direction"] == "minimize"
    assert "map_sample_probability_columns_to_class_labels" in payload["strategy"]["preprocessing"]
    assert "renormalize_probability_columns" in payload["strategy"]["preprocessing"]
    assert "calibrated_extra_trees" in payload["strategy"]["models"]


def test_analyze_competition_handles_multi_output_regression_targets(tmp_path):
    paths = CompetitionPaths(slug="demo-multi-output-regression", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "target_x": [float(idx * 1.5) for idx in rows],
            "target_y": [float(idx * -0.2 + 5.0) for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "target_x": [0.0, 0.0], "target_y": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-multi-output-regression",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["target_x", "target_y"]
    assert payload["task"] == "multi_output_regression"
    assert payload["prediction_kind"] == "continuous_columns"
    assert payload["metric"] == "mcrmse"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_all_target_columns" in payload["strategy"]["preprocessing"]
    assert "align_2d_predictions_to_sample_columns" in payload["strategy"]["preprocessing"]
    assert "multi_output_ridge" in payload["strategy"]["models"]
    assert any("multi-output regression layout" in item for item in payload["assumptions"])


def test_analyze_competition_handles_coordinate_regression_targets(tmp_path):
    paths = CompetitionPaths(slug="demo-coordinate-regression", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "x": [float(idx * 1.5) for idx in rows],
            "y": [float(idx * -0.2 + 5.0) for idx in rows],
            "z": [float(idx * 0.7 - 3.0) for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-coordinate-regression",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["x", "y", "z"]
    assert payload["task"] == "coordinate_regression"
    assert payload["prediction_kind"] == "coordinate_columns"
    assert payload["metric"] == "rmse"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_coordinate_target_columns" in payload["strategy"]["preprocessing"]
    assert "align_coordinate_predictions_to_sample_columns" in payload["strategy"]["preprocessing"]
    assert "coordinate_gbdt_regressor" in payload["strategy"]["models"]
    assert any("coordinate target columns inferred" in item for item in payload["assumptions"])


def test_analyze_competition_handles_multi_target_classification_targets(tmp_path):
    paths = CompetitionPaths(slug="demo-multi-target-classification", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "label_a": ["cat" if idx % 2 else "dog" for idx in rows],
            "label_b": ["low", "mid", "high"] * 10,
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "label_a": ["cat", "cat"], "label_b": ["low", "low"]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-multi-target-classification",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["label_a", "label_b"]
    assert payload["task"] == "multi_target_classification"
    assert payload["prediction_kind"] == "target_columns"
    assert payload["metric"] == "f1"
    assert payload["metric_direction"] == "maximize"
    assert "encode_each_class_target_independently" in payload["strategy"]["preprocessing"]
    assert "format_class_or_probability_target_columns" in payload["strategy"]["preprocessing"]
    assert "multi_output_logreg" in payload["strategy"]["models"]
    assert any("multi-target classification layout" in item for item in payload["assumptions"])


def test_analyze_competition_handles_mixed_multi_task_targets(tmp_path):
    paths = CompetitionPaths(slug="demo-multi-task", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "label": ["cat" if idx % 2 else "dog" for idx in rows],
            "score": [float(idx * 0.75) for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "label": ["cat", "cat"], "score": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-multi-task",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["label", "score"]
    assert payload["task"] == "multi_task"
    assert payload["prediction_kind"] == "target_columns"
    assert payload["metric"] == "mean_target_metric"
    assert payload["metric_direction"] == "maximize"
    assert "infer_task_and_metric_per_target" in payload["strategy"]["preprocessing"]
    assert "align_outputs_to_sample_columns" in payload["strategy"]["preprocessing"]
    assert "per_target_type_router" in payload["strategy"]["models"]
    assert any("mixed multi-task target layout" in item for item in payload["assumptions"])


def test_analyze_competition_handles_probability_named_binary_target(tmp_path):
    paths = CompetitionPaths(slug="demo-fraud", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "isFraud": [0 if idx < 20 else 1 for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [40.0, 41.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "isFraud": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-fraud",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["isFraud"]
    assert payload["task"] == "classification"
    assert payload["prediction_kind"] == "probability"
    assert payload["metric"] == "logloss"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_probability_output_contract" in payload["strategy"]["preprocessing"]
    assert "calibrate_probabilities_when_possible" in payload["strategy"]["preprocessing"]
    assert "logreg_probability" in payload["strategy"]["models"]


def test_analyze_competition_handles_named_low_cardinality_numeric_regression(tmp_path):
    paths = CompetitionPaths(slug="demo-sales", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "sales": [idx % 4 for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [40.0, 41.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "sales": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-sales",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["sales"]
    assert payload["task"] == "regression"
    assert payload["prediction_kind"] == "continuous"
    assert payload["metric"] == "rmse"
    assert payload["metric_direction"] == "minimize"


def test_analyze_competition_handles_count_regression_target(tmp_path):
    paths = CompetitionPaths(slug="demo-count-regression", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "count": [idx for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "count": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-count-regression",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["count"]
    assert payload["task"] == "count_regression"
    assert payload["prediction_kind"] == "continuous"
    assert payload["metric"] == "rmsle"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_non_negative_count_target" in payload["strategy"]["preprocessing"]
    assert "evaluate_rmsle_on_validation" in payload["strategy"]["preprocessing"]
    assert "poisson_gradient_boosting" in payload["strategy"]["models"]
    assert any("non-negative integer count target" in item for item in payload["assumptions"])


def test_analyze_competition_handles_bounded_regression_target(tmp_path):
    paths = CompetitionPaths(slug="demo-bounded-regression", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(24))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "conversion_rate": [0.0, 0.25, 0.5, 0.75] * 6,
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [24.0, 25.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "conversion_rate": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-bounded-regression",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["conversion_rate"]
    assert payload["task"] == "bounded_regression"
    assert payload["prediction_kind"] == "continuous"
    assert payload["metric"] == "rmse"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_bounded_rate_or_ratio_target" in payload["strategy"]["preprocessing"]
    assert "clip_predictions_to_observed_bounds" in payload["strategy"]["preprocessing"]
    assert "bounded_ridge_regressor" in payload["strategy"]["models"]
    assert any("bounded rate/ratio target inferred" in item for item in payload["assumptions"])


def test_analyze_competition_handles_positive_skew_regression_target(tmp_path):
    paths = CompetitionPaths(slug="demo-positive-skew-regression", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    prices = [100, 110, 120, 130, 140, 150, 160, 170, 5000, 8000]
    pd.DataFrame(
        {
            "id": list(range(len(prices))),
            "feature": [float(idx) for idx in range(len(prices))],
            "SalePrice": prices,
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [10.0, 11.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "SalePrice": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-positive-skew-regression",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["SalePrice"]
    assert payload["task"] == "positive_skew_regression"
    assert payload["prediction_kind"] == "continuous"
    assert payload["metric"] == "rmsle"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_non_negative_skewed_target" in payload["strategy"]["preprocessing"]
    assert "evaluate_rmsle_on_validation" in payload["strategy"]["preprocessing"]
    assert "log1p_ridge_regressor" in payload["strategy"]["models"]
    assert any("positive skew non-negative target" in item for item in payload["assumptions"])


def test_analyze_competition_handles_delimited_multi_label_target(tmp_path):
    paths = CompetitionPaths(slug="demo-multilabel-text", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["red item", "blue item", "green item", "yellow item"],
            "labels": ["cat dog", "dog bird", "cat bird", "cat dog bird"],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "prompt": ["blue item", "green item"]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "labels": ["cat dog", "cat dog"]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-multilabel-text",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["labels"]
    assert payload["task"] == "multi_label"
    assert payload["prediction_kind"] == "text"
    assert payload["metric"] == "f1"
    assert payload["metric_direction"] == "maximize"
    assert "split_delimited_label_sets" in payload["strategy"]["preprocessing"]
    assert "format_label_sets_or_probability_columns" in payload["strategy"]["preprocessing"]
    assert "one_vs_rest_multilabel_logreg" in payload["strategy"]["models"]
    assert any("delimiter-based multi-label target" in item for item in payload["assumptions"])


def test_analyze_competition_handles_multi_label_probability_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-multilabel-columns", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    labels = ["cat dog", "dog bird", "cat bird", "cat dog", "dog bird", "cat bird", "cat dog", "dog bird"]
    pd.DataFrame(
        {
            "id": range(8),
            "feature": [float(idx) for idx in range(8)],
            "labels": labels,
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [1.0, 2.0]}).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "cat": [0.0, 0.0],
            "dog": [0.0, 0.0],
            "bird": [0.0, 0.0],
        }
    ).to_csv(paths.data_dir / "sample_submission.csv", index=False)

    result = analyze_competition(
        slug="demo-multilabel-columns",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["labels"]
    assert payload["task"] == "multi_label"
    assert payload["prediction_kind"] == "multi_label_columns"
    assert payload["metric"] == "f1"
    assert payload["metric_direction"] == "maximize"
    assert "map_label_columns_to_observed_labels" in payload["strategy"]["preprocessing"]
    assert "tune_per_label_thresholds" in payload["strategy"]["preprocessing"]
    assert "per_label_threshold_tuning" in payload["strategy"]["models"]
    assert any("delimiter-based multi-label target" in item for item in payload["assumptions"])


def test_analyze_competition_handles_multi_label_indicator_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-multilabel-indicators", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train_rows = []
    for idx in range(12):
        row = {
            "id": idx,
            "comment_text": f"example comment {idx}",
            "length": float(20 + idx),
        }
        for label_idx, label in enumerate(label_columns):
            row[label] = int((idx + label_idx) % 4 == 0)
        train_rows.append(row)
    pd.DataFrame(train_rows).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "comment_text": ["heldout comment one", "heldout comment two"],
            "length": [30.0, 31.0],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    sample = {"id": [100, 101]}
    sample.update({label: [0.0, 0.0] for label in label_columns})
    pd.DataFrame(sample).to_csv(paths.data_dir / "sample_submission.csv", index=False)

    result = analyze_competition(
        slug="demo-multilabel-indicators",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == label_columns
    assert payload["task"] == "multi_label"
    assert payload["prediction_kind"] == "multi_label_columns"
    assert payload["metric"] == "f1"
    assert payload["metric_direction"] == "maximize"
    assert "preserve_binary_indicator_label_columns" in payload["strategy"]["preprocessing"]
    assert "fit_one_vs_rest_label_heads" in payload["strategy"]["preprocessing"]
    assert "one_vs_rest_multilabel_logreg" in payload["strategy"]["models"]
    assert any("binary indicator multi-label target columns" in item for item in payload["assumptions"])


def test_analyze_competition_handles_quantile_submission_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-quantiles", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx * 2) for idx in range(30)],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "p10": [0.0, 0.0], "p50": [0.0, 0.0], "p90": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-quantiles",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["target"]
    assert payload["schema"]["feature_columns"] == ["feature"]
    assert payload["task"] == "regression"
    assert payload["prediction_kind"] == "quantile_columns"
    assert payload["metric"] == "pinball_loss"
    assert payload["metric_direction"] == "minimize"
    assert "map_quantile_submission_columns_to_target" in payload["strategy"]["preprocessing"]
    assert "enforce_non_crossing_quantiles" in payload["strategy"]["preprocessing"]
    assert "gradient_boosting_quantile_heads" in payload["strategy"]["models"]
    assert any("quantile submission columns inferred" in item for item in payload["assumptions"])


def test_analyze_competition_handles_prediction_interval_submission_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-interval", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx * 2) for idx in range(30)],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "lower": [0.0, 0.0], "upper": [1.0, 1.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-interval",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["target"]
    assert payload["schema"]["feature_columns"] == ["feature"]
    assert payload["task"] == "regression"
    assert payload["prediction_kind"] == "prediction_interval_columns"
    assert payload["metric"] == "interval_score"
    assert payload["metric_direction"] == "minimize"
    assert "map_interval_submission_columns_to_target" in payload["strategy"]["preprocessing"]
    assert "enforce_lower_upper_order" in payload["strategy"]["preprocessing"]
    assert "conformal_interval_regressor" in payload["strategy"]["models"]
    assert any("prediction-interval submission columns inferred" in item for item in payload["assumptions"])


def test_analyze_competition_handles_unlabeled_anomaly_score_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-anomaly", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "amount": [10.5, 11.2, 980.0],
            "velocity": [0.1, 0.2, 9.8],
            "country": ["JP", "US", "BR"],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [4, 5],
            "amount": [12.0, 1200.0],
            "velocity": [0.1, 12.5],
            "country": ["JP", "US"],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [4, 5], "anomaly_score": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-anomaly",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["anomaly_score"]
    assert payload["schema"]["feature_columns"] == ["amount", "velocity", "country"]
    assert payload["task"] == "unsupervised"
    assert payload["prediction_kind"] == "probability"
    assert payload["metric"] == "auc"
    assert payload["metric_direction"] == "maximize"
    assert "fit_unsupervised_scores_without_train_labels" in payload["strategy"]["preprocessing"]
    assert "robust_unsupervised_anomaly_score" in payload["strategy"]["models"]
    assert any("schema inferred from solver no-label layout" in item for item in payload["assumptions"])


def test_analyze_competition_handles_ctr_user_item_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-ctr", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "user_id": ["u1", "u1", "u2", "u3"],
            "ad_id": ["a1", "a2", "a1", "a3"],
            "device": ["mobile", "desktop", "mobile", "desktop"],
            "clicked": [1, 0, 0, 1],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            "user_id": ["u1", "u4"],
            "ad_id": ["a3", "a2"],
            "device": ["mobile", "desktop"],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "clicked": [0.5, 0.5]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-ctr",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["clicked"]
    assert payload["task"] == "ctr"
    assert payload["prediction_kind"] == "probability"
    assert payload["metric"] == "logloss"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_user_item_ids" in payload["strategy"]["preprocessing"]
    assert "calibrate_ctr_probabilities" in payload["strategy"]["preprocessing"]
    assert "ctr_gradient_boosting_calibrated" in payload["strategy"]["models"]
    assert any("user-item click/conversion layout" in item for item in payload["assumptions"])


def test_analyze_competition_handles_recommender_rating_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-recommender", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(1, 31))
    pd.DataFrame(
        {
            "id": rows,
            "user_id": [f"u{idx % 5}" for idx in rows],
            "item_id": [f"i{idx % 7}" for idx in rows],
            "context_feature": [idx * 0.2 for idx in rows],
            "rating": [1.0 + idx * 0.13 for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [31, 32],
            "user_id": ["u1", "u2"],
            "item_id": ["i3", "i4"],
            "context_feature": [6.2, 6.4],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [31, 32], "rating": [3.0, 3.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-recommender",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["rating"]
    assert payload["task"] == "recommender"
    assert payload["prediction_kind"] == "continuous"
    assert payload["metric"] == "rmse"
    assert payload["metric_direction"] == "minimize"
    assert "derive_user_item_aggregate_features" in payload["strategy"]["preprocessing"]
    assert "clip_predictions_to_observed_rating_range" in payload["strategy"]["preprocessing"]
    assert "matrix_factorization_baseline" in payload["strategy"]["models"]
    assert any("user-item rating/relevance layout" in item for item in payload["assumptions"])


def test_analyze_competition_handles_pairwise_matchup_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-pairwise", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "prompt": ["hello", "world", "test"],
            "model_a": ["alpha", "beta", "alpha"],
            "model_b": ["beta", "gamma", "gamma"],
            "winner_model_a": [1, 0, 0],
            "winner_model_b": [0, 1, 0],
            "winner_tie": [0, 0, 1],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [4, 5],
            "prompt": ["question", "answer"],
            "model_a": ["alpha", "beta"],
            "model_b": ["beta", "gamma"],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame(
        {
            "id": [4, 5],
            "winner_model_a": [0.33, 0.33],
            "winner_model_b": [0.33, 0.33],
            "winner_tie": [0.34, 0.34],
        }
    ).to_csv(paths.data_dir / "sample_submission.csv", index=False)

    result = analyze_competition(
        slug="demo-pairwise",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["winner_model_a", "winner_model_b", "winner_tie"]
    assert payload["task"] == "pairwise"
    assert payload["prediction_kind"] == "probability_columns"
    assert payload["metric"] == "logloss"
    assert payload["metric_direction"] == "minimize"
    assert "preserve_pair_entity_columns" in payload["strategy"]["preprocessing"]
    assert "derive_pairwise_difference_features" in payload["strategy"]["preprocessing"]
    assert "pairwise_difference_ranker" in payload["strategy"]["models"]
    assert any("paired entity matchup layout" in item for item in payload["assumptions"])


def test_analyze_competition_handles_survival_event_time_single_score_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-survival", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "efs": [idx % 2 for idx in range(30)],
            "efs_time": [float(30 - idx) for idx in range(30)],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [30.0, 31.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "prediction": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-survival",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["efs", "efs_time"]
    assert payload["task"] == "survival"
    assert payload["prediction_kind"] == "risk_score"
    assert payload["metric"] == "concordance_index"
    assert payload["metric_direction"] == "maximize"
    assert "preserve_event_and_time_targets" in payload["strategy"]["preprocessing"]
    assert "derive_single_risk_score_submission" in payload["strategy"]["preprocessing"]
    assert "survival_risk_score" in payload["strategy"]["models"]
    assert any("survival event/time targets inferred" in item for item in payload["assumptions"])


def test_analyze_competition_handles_learning_to_rank_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-ltr", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "query_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "document_id": ["d1", "d2", "d3", "d4", "d5", "d6"],
            "bm25": [3.1, 1.2, 0.4, 2.8, 1.7, 0.2],
            "doc_length": [120, 240, 80, 150, 90, 300],
            "relevance": [3, 1, 0, 2, 1, 0],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [7, 8],
            "query_id": ["q3", "q3"],
            "document_id": ["d7", "d8"],
            "bm25": [2.1, 0.9],
            "doc_length": [140, 210],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [7, 8], "relevance": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-ltr",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["relevance"]
    assert payload["task"] == "learning_to_rank"
    assert payload["prediction_kind"] == "ranking_score"
    assert payload["metric"] == "ndcg"
    assert payload["metric_direction"] == "maximize"
    assert "preserve_query_groups" in payload["strategy"]["preprocessing"]
    assert "grouped_ndcg_validation" in payload["strategy"]["preprocessing"]
    assert "lambdamart_ranker" in payload["strategy"]["models"]
    assert any("learning-to-rank" in item for item in payload["assumptions"])


def test_analyze_competition_handles_forecasting_layout(tmp_path):
    paths = CompetitionPaths(slug="demo-forecasting", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": range(24),
            "store_id": ["a"] * 12 + ["b"] * 12,
            "date_block_num": list(range(12)) + list(range(12)),
            "feature": [float(idx) for idx in range(24)],
            "target": [float(idx * 1.5) for idx in range(24)],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "store_id": ["a", "b"],
            "date_block_num": [12, 12],
            "feature": [24.0, 25.0],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "target": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-forecasting",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["target"]
    assert payload["task"] == "forecasting"
    assert payload["prediction_kind"] == "continuous"
    assert payload["metric"] == "rmse"
    assert payload["metric_direction"] == "minimize"
    assert "chronological_holdout_validation" in payload["strategy"]["preprocessing"]
    assert "derive_lag_and_rolling_features" in payload["strategy"]["preprocessing"]
    assert "lag_feature_gradient_boosting" in payload["strategy"]["models"]
    assert any("future temporal holdout inferred" in item for item in payload["assumptions"])


def test_analyze_competition_uses_ordinal_strategy_for_ordinal_target(tmp_path):
    paths = CompetitionPaths(slug="demo-ordinal", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(40))
    pd.DataFrame(
        {
            "id": rows,
            "feature": [float(idx) for idx in rows],
            "severity": [idx % 5 for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101], "feature": [40.0, 41.0]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101], "severity": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-ordinal",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["severity"]
    assert payload["task"] == "regression"
    assert payload["prediction_kind"] == "ordinal"
    assert payload["metric"] == "quadratic_weighted_kappa"
    assert payload["metric_direction"] == "maximize"
    assert "preserve_ordered_target_values" in payload["strategy"]["preprocessing"]
    assert "tune_monotonic_thresholds_on_validation" in payload["strategy"]["preprocessing"]
    assert "ordinal_threshold_tuning" in payload["strategy"]["models"]


def test_analyze_competition_handles_text_submission_columns(tmp_path):
    paths = CompetitionPaths(slug="demo-text", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "prompt": ["a", "b", "c"],
            "translation": ["alpha one", "beta two", "gamma three"],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [4, 5], "prompt": ["d", "e"]}).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [4, 5], "translation": ["", ""]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-text",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["translation"]
    assert payload["task"] == "text"
    assert payload["prediction_kind"] == "text"
    assert payload["metric"] == "text_similarity"
    assert payload["metric_direction"] == "maximize"
    assert "constant_text" in payload["strategy"]["models"]
    assert "normalize_text" in payload["strategy"]["preprocessing"]


def test_analyze_competition_preserves_text_feature_classification_modality(tmp_path):
    paths = CompetitionPaths(slug="demo-text-classification", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "review": [
                "Great acting and tight pacing",
                "Flat plot but nice cast",
                "Loved the ending and dialogue",
                "Too slow for me overall",
            ],
            "label": [1, 0, 1, 0],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [5, 6],
            "review": ["Funny and sharp script", "Weak script overall"],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "label": [0, 0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-text-classification",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["task"] == "text_classification"
    assert payload["prediction_kind"] == "class"
    assert payload["metric"] == "accuracy"
    assert "normalize_and_vectorize_text_features" in payload["strategy"]["preprocessing"]
    assert "tfidf_logreg_text_classifier" in payload["strategy"]["models"]
    assert any("text feature modality inferred" in item for item in payload["assumptions"])


def test_analyze_competition_preserves_text_feature_regression_modality(tmp_path):
    paths = CompetitionPaths(slug="demo-text-regression", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    rows = list(range(30))
    pd.DataFrame(
        {
            "id": rows,
            "description": [f"Detailed product description with useful signal number {idx}" for idx in rows],
            "sales": [float(idx * 1.7 + 3.0) for idx in rows],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [100, 101],
            "description": [
                "Detailed product description with useful signal number 30",
                "Detailed product description with useful signal number 31",
            ],
        }
    ).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [100, 101], "sales": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-text-regression",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["task"] == "text_regression"
    assert payload["prediction_kind"] == "continuous"
    assert payload["metric"] == "rmse"
    assert "blend_text_and_tabular_regression_features" in payload["strategy"]["preprocessing"]
    assert "tfidf_ridge_text_regressor" in payload["strategy"]["models"]
    assert any("text feature modality inferred" in item for item in payload["assumptions"])


def test_analyze_competition_handles_natural_language_target_column(tmp_path):
    paths = CompetitionPaths(slug="demo-natural-language-target", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prompt": ["red apple", "blue ocean", "green forest", "yellow flower"],
            "target": [
                "write a concise answer about red apples",
                "write a concise answer about blue oceans",
                "write a concise answer about green forests",
                "write a concise answer about yellow flowers",
            ],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "prompt": ["blue ocean", "green forest"]}).to_csv(
        paths.data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [5, 6], "target": ["placeholder", "placeholder"]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo-natural-language-target",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"]["target_columns"] == ["target"]
    assert payload["task"] == "text"
    assert payload["prediction_kind"] == "text"
    assert payload["metric"] == "text_similarity"
    assert payload["metric_direction"] == "maximize"
    assert "constant_text" in payload["strategy"]["models"]
    assert "normalize_text" in payload["strategy"]["preprocessing"]
