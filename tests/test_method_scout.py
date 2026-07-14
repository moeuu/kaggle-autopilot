from __future__ import annotations

import json
from pathlib import Path

from kagglebot.method_scout import (
    build_method_candidates,
    build_method_scout_queries,
    build_validation_registry,
    classify_source,
    effective_method_scout_mode,
    infer_modality,
    load_kaggle_discovery_sources,
    load_method_registry,
    load_source_registry,
    load_validation_registry,
    method_registry_path,
    render_method_registry_for_prompt,
    run_method_scout,
    source_registry_path,
    unsafe_method_reason,
)
from kagglebot.paths import CompetitionPaths


def test_query_builder_uses_modality_metric_and_validation_regression() -> None:
    queries = build_method_scout_queries(
        slug="demo-vision",
        problem_types=["image-classification"],
        dataset_profile={"target_metric": "accuracy"},
        metric="accuracy",
        campaign_state={
            "direction": "maximize",
            "latest_submission_score": 0.72,
            "champion_score": 0.80,
        },
        max_sources=8,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "validation split public leaderboard mismatch" in query_text
    assert "image" in query_text
    assert "accuracy" in query_text
    assert any(item["purpose"] == "validation_redesign" for item in queries)


def test_query_builder_adds_asset_modality_specific_queries() -> None:
    cases = {
        "audio": ["spectrogram", "MFCC"],
        "video": ["frame sampling", "temporal pooling"],
        "signal": ["ECG", "WFDB", "1D CNN"],
        "medical_imaging": ["DICOM", "IMA", "NIfTI", "NRRD", "MHA"],
        "array": ["Zarr", "OME-Zarr", "N5", "AnnData", "H5AD", "NetCDF"],
        "point_cloud": ["lidar", "voxel"],
        "geospatial": ["GeoJSON", "spatial cross validation"],
        "bio": ["SMILES", "FASTA", "PDB"],
        "graph": ["graph neural network", "NetworkX"],
        "annotation": ["COCO", "YOLO", "LabelMe", "RLE"],
        "artifact": ["ONNX", "safetensors", "checkpoint"],
        "model-artifact": ["ONNX", "safetensors", "CoreML"],
    }
    for modality, expected_terms in cases.items():
        queries = build_method_scout_queries(
            slug=f"demo-{modality}",
            problem_types=["tabular"],
            dataset_profile={"modality": modality},
            metric="accuracy",
            max_sources=12,
        )
        query_text = "\n".join(str(item["query"]) for item in queries)

        for term in expected_terms:
            assert term in query_text


def test_query_builder_adds_multimodal_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-vqa",
        problem_types=["multimodal", "classification"],
        dataset_profile={"modality": "multimodal"},
        metric="accuracy",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "multimodal image text fusion" in query_text
    assert "CLIP embeddings" in query_text
    assert "vision language dual encoder" in query_text


def test_query_builder_adds_multi_label_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-multilabel",
        problem_types=["tabular", "classification", "multi_label"],
        dataset_profile={"modality": "tabular", "target_semantics": "multi_label", "tags": ["multi_label"]},
        metric="f1",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "multi-label classification" in query_text
    assert "threshold optimization" in query_text
    assert "one-vs-rest" in query_text


def test_query_builder_adds_text_generation_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-translation",
        problem_types=["text", "text_generation"],
        dataset_profile={"modality": "text", "target_semantics": "text_generation", "tags": ["text_generation"]},
        metric="text_similarity",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "text generation translation summarization" in query_text
    assert "seq2seq transformer" in query_text
    assert "semantic similarity" in query_text


def test_query_builder_adds_document_file_reference_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-documents",
        problem_types=["text"],
        dataset_profile={"modality": "text", "columns": ["id", "document_path", "target"]},
        metric="accuracy",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "PDF DOCX Markdown document classification" in query_text
    assert "document file metadata text extraction embeddings" in query_text


def test_method_scout_infers_signal_modality_and_adds_signal_candidates() -> None:
    assert infer_modality(["ECG waveform classification"], {}) == "signal"
    assert infer_modality([], {"columns": ["record_id", "signal_path", "target"]}) == "signal"
    assert infer_modality([], {"columns": ["record_id", "file", "target"], "modality": "signal"}) == "signal"

    methods = build_method_candidates(
        slug="demo-signal",
        problem_types=["signal"],
        dataset_profile={"modality": "signal"},
        metric="auc",
        sources=[],
    )
    method_ids = [method.method_id for method in methods]

    assert "signal-statistical-feature-head" in method_ids
    assert "signal-1d-cnn-optional-branch" in method_ids


def test_query_builder_adds_multi_output_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-multioutput",
        problem_types=["tabular", "regression", "multi_output"],
        dataset_profile={
            "modality": "tabular",
            "target_semantics": "multi_output_regression",
            "tags": ["multi_output"],
        },
        metric="rmse",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "multi-output multi-target modeling" in query_text
    assert "multi-output regression" in query_text
    assert "per-target model" in query_text


def test_query_builder_adds_quantile_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-quantile",
        problem_types=["tabular", "regression", "quantile_regression"],
        dataset_profile={
            "modality": "tabular",
            "target_semantics": "quantile_regression",
            "tags": ["quantile_regression"],
        },
        metric="pinball_loss",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "quantile regression pinball loss" in query_text
    assert "prediction intervals conformal quantile regression" in query_text


def test_query_builder_adds_ordinal_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-ordinal",
        problem_types=["tabular", "ordinal_classification"],
        dataset_profile={
            "modality": "tabular",
            "target_semantics": "ordinal_classification",
            "tags": ["ordinal_classification"],
        },
        metric="quadratic_weighted_kappa",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "ordinal classification quadratic weighted kappa" in query_text
    assert "ordinal regression threshold optimization" in query_text


def test_query_builder_adds_sample_weight_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-weighted",
        problem_types=["tabular", "classification", "sample_weighted"],
        dataset_profile={
            "modality": "tabular",
            "sample_weight_column_hint": "sample_weight",
            "tags": ["sample_weighted"],
        },
        metric="auc",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "sample weights weighted metric" in query_text
    assert "sample_weight weighted loss validation" in query_text


def test_query_builder_adds_survival_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-survival",
        problem_types=["tabular", "survival"],
        dataset_profile={"modality": "tabular", "target_semantics": "survival", "tags": ["survival"]},
        metric="concordance_index",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "survival analysis" in query_text
    assert "concordance index" in query_text
    assert "event censoring" in query_text


def test_query_builder_adds_pairwise_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-pairwise",
        problem_types=["tabular", "pairwise"],
        dataset_profile={"modality": "tabular", "target_semantics": "pairwise", "tags": ["pairwise"]},
        metric="logloss",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "pairwise matchup ranking" in query_text
    assert "Bradley Terry" in query_text
    assert "feature difference" in query_text


def test_query_builder_adds_learning_to_rank_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-ltr",
        problem_types=["tabular", "learning_to_rank"],
        dataset_profile={"modality": "tabular", "target_semantics": "learning_to_rank", "tags": ["learning_to_rank"]},
        metric="ndcg",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "learning to rank NDCG LambdaMART" in query_text
    assert "query document relevance ranking" in query_text


def test_query_builder_adds_anomaly_detection_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-anomaly",
        problem_types=["tabular", "unsupervised", "anomaly_detection"],
        dataset_profile={
            "modality": "tabular",
            "task": "unsupervised",
            "target_semantics": "anomaly_detection",
            "tags": ["unsupervised", "anomaly_detection"],
        },
        metric="auc",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "anomaly detection isolation forest autoencoder" in query_text
    assert "unsupervised anomaly score calibration" in query_text


def test_query_builder_adds_ctr_and_recommender_queries() -> None:
    ctr_queries = build_method_scout_queries(
        slug="demo-ctr",
        problem_types=["tabular", "ctr"],
        dataset_profile={"modality": "tabular", "target_semantics": "ctr", "tags": ["ctr"]},
        metric="logloss",
        max_sources=12,
    )
    recommender_queries = build_method_scout_queries(
        slug="demo-recommender",
        problem_types=["tabular", "recommender"],
        dataset_profile={"modality": "tabular", "target_semantics": "recommender", "tags": ["recommender"]},
        metric="rmse",
        max_sources=12,
    )

    ctr_text = "\n".join(str(item["query"]) for item in ctr_queries)
    recommender_text = "\n".join(str(item["query"]) for item in recommender_queries)
    assert "click through rate CTR" in ctr_text
    assert "user item ad click prediction" in ctr_text
    assert "recommender system user item" in recommender_text
    assert "matrix factorization" in recommender_text


def test_query_builder_adds_forecasting_queries() -> None:
    queries = build_method_scout_queries(
        slug="demo-forecasting",
        problem_types=["timeseries", "forecasting"],
        dataset_profile={"modality": "timeseries", "target_semantics": "forecasting", "tags": ["forecasting"]},
        metric="rmse",
        max_sources=12,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "forecasting horizon lag rolling feature" in query_text
    assert "time series forecasting backtesting leakage" in query_text


def test_query_builder_adds_detection_and_segmentation_queries() -> None:
    detection_queries = build_method_scout_queries(
        slug="demo-detection",
        problem_types=["image", "object_detection"],
        dataset_profile={"modality": "image", "target_semantics": "object_detection", "tags": ["object_detection"]},
        metric="map",
        max_sources=12,
    )
    segmentation_queries = build_method_scout_queries(
        slug="demo-segmentation",
        problem_types=["image", "segmentation"],
        dataset_profile={"modality": "image", "target_semantics": "segmentation", "tags": ["segmentation"]},
        metric="dice",
        max_sources=12,
    )

    detection_text = "\n".join(str(item["query"]) for item in detection_queries)
    segmentation_text = "\n".join(str(item["query"]) for item in segmentation_queries)
    assert "object detection prediction_string" in detection_text
    assert "YOLO" in detection_text
    assert "segmentation RLE mask" in segmentation_text
    assert "U-Net" in segmentation_text


def test_method_scout_uses_profile_modality_for_point_cloud() -> None:
    assert infer_modality(["tabular"], {"modality": "point_cloud"}) == "point_cloud"

    methods = build_method_candidates(
        slug="demo-point-cloud",
        problem_types=["tabular"],
        dataset_profile={"modality": "point_cloud"},
        metric="rmse",
        sources=[],
    )

    assert methods
    assert methods[0].method_id.startswith("point-cloud-")
    assert methods[0].implementation_adapter["adapter"].startswith("point_cloud_")
    assert "torch" in methods[0].dependency_check["optional"]


def test_method_scout_uses_profile_modality_for_geospatial() -> None:
    assert infer_modality(["tabular"], {"modality": "geospatial"}) == "geospatial"

    methods = build_method_candidates(
        slug="demo-geospatial",
        problem_types=["tabular"],
        dataset_profile={"modality": "geospatial"},
        metric="rmse",
        sources=[],
    )

    assert methods
    assert methods[0].method_id.startswith("geospatial-")
    assert methods[0].implementation_adapter["adapter"].startswith("geospatial_")
    assert "geometry" in methods[0].dependency_check["fallback"].lower()


def test_method_scout_uses_profile_modality_for_bio() -> None:
    assert infer_modality(["tabular"], {"modality": "bio"}) == "bio"

    methods = build_method_candidates(
        slug="demo-bio",
        problem_types=["tabular"],
        dataset_profile={"modality": "bio"},
        metric="tm-score",
        sources=[],
    )

    assert methods
    assert methods[0].method_id.startswith("bio-")
    assert methods[0].implementation_adapter["adapter"].startswith("bio_")
    assert "domain-aware" in methods[0].summary


def test_method_scout_uses_profile_modality_for_graph() -> None:
    assert infer_modality(["tabular"], {"modality": "graph"}) == "graph"

    methods = build_method_candidates(
        slug="demo-graph",
        problem_types=["tabular"],
        dataset_profile={"modality": "graph"},
        metric="auc",
        sources=[],
    )

    assert methods
    assert methods[0].method_id.startswith("graph-")
    assert methods[0].implementation_adapter["adapter"].startswith("graph_")
    assert "networkx" in methods[0].dependency_check["optional"]


def test_method_scout_adds_multi_label_threshold_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-multilabel",
        problem_types=["tabular", "classification", "multi_label"],
        dataset_profile={"modality": "tabular", "target_semantics": "multi_label", "tags": ["multi_label"]},
        metric="f1",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-multi-label-one-vs-rest-thresholds" in method_ids
    multi_label_method = next(
        method for method in methods if method.method_id == "tabular-multi-label-one-vs-rest-thresholds"
    )
    assert "threshold" in multi_label_method.summary.lower()
    assert multi_label_method.implementation_adapter["adapter"] == "tabular_multi_label_thresholds"


def test_method_scout_adds_text_generation_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-translation",
        problem_types=["text", "text_generation"],
        dataset_profile={"modality": "text", "target_semantics": "text_generation", "tags": ["text_generation"]},
        metric="text_similarity",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "text-text-generation-retrieval-seq2seq" in method_ids
    generation_method = next(
        method for method in methods if method.method_id == "text-text-generation-retrieval-seq2seq"
    )
    assert "Text-generation submissions" in generation_method.summary
    assert generation_method.implementation_adapter["adapter"] == "text_text_generation_retrieval_seq2seq"


def test_method_scout_adds_document_file_reference_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-documents",
        problem_types=["text"],
        dataset_profile={"modality": "text", "columns": ["id", "pdf_file", "target"]},
        metric="accuracy",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "text-document-file-metadata-text-head" in method_ids
    document_method = next(method for method in methods if method.method_id == "text-document-file-metadata-text-head")
    assert "PDF/DOCX/Markdown metadata" in document_method.summary
    assert document_method.implementation_adapter["adapter"] == "text_document_file_metadata_text_head"
    assert "sklearn" in document_method.dependency_check["required"]


def test_method_scout_adds_multi_output_target_head_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-multioutput",
        problem_types=["tabular", "regression", "multi_output"],
        dataset_profile={
            "modality": "tabular",
            "target_semantics": "multi_output_regression",
            "tags": ["multi_output"],
        },
        metric="rmse",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-multi-output-target-heads" in method_ids
    multi_output_method = next(method for method in methods if method.method_id == "tabular-multi-output-target-heads")
    assert "per-target" in multi_output_method.summary
    assert multi_output_method.implementation_adapter["adapter"] == "tabular_multi_output_heads"


def test_method_scout_adds_quantile_interval_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-quantile",
        problem_types=["tabular", "regression", "quantile_regression"],
        dataset_profile={
            "modality": "tabular",
            "target_semantics": "quantile_regression",
            "tags": ["quantile_regression"],
        },
        metric="pinball_loss",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-quantile-interval-heads" in method_ids
    quantile_method = next(method for method in methods if method.method_id == "tabular-quantile-interval-heads")
    assert "ordered interval outputs" in quantile_method.summary
    assert quantile_method.implementation_adapter["adapter"] == "tabular_quantile_interval_heads"


def test_method_scout_adds_ordinal_threshold_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-ordinal",
        problem_types=["tabular", "ordinal_classification"],
        dataset_profile={
            "modality": "tabular",
            "target_semantics": "ordinal_classification",
            "tags": ["ordinal_classification"],
        },
        metric="quadratic_weighted_kappa",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-ordinal-threshold-qwk" in method_ids
    ordinal_method = next(method for method in methods if method.method_id == "tabular-ordinal-threshold-qwk")
    assert "ordered thresholds" in ordinal_method.summary
    assert ordinal_method.implementation_adapter["adapter"] == "tabular_ordinal_threshold_qwk"


def test_method_scout_adds_multimodal_fusion_candidates() -> None:
    methods = build_method_candidates(
        slug="demo-vqa",
        problem_types=["multimodal", "classification"],
        dataset_profile={"modality": "multimodal"},
        metric="accuracy",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "multimodal-vision-language-fusion" in method_ids
    fusion_method = next(method for method in methods if method.method_id == "multimodal-vision-language-fusion")
    assert "late-fusion" in fusion_method.summary
    assert fusion_method.implementation_adapter["adapter"].startswith("multimodal_")
    assert "torch" in fusion_method.dependency_check["optional"]
    assert "transformers" in fusion_method.dependency_check["optional"]


def test_method_scout_adds_sample_weight_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-weighted",
        problem_types=["tabular", "classification", "sample_weighted"],
        dataset_profile={
            "modality": "tabular",
            "sample_weight_column_hint": "sample_weight",
            "tags": ["sample_weighted"],
        },
        metric="auc",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-sample-weight-aware-training" in method_ids
    weight_method = next(method for method in methods if method.method_id == "tabular-sample-weight-aware-training")
    assert "sample_weight" in weight_method.fallback
    assert weight_method.implementation_adapter["adapter"] == "tabular_sample_weight_aware_training"


def test_method_scout_adds_survival_risk_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-survival",
        problem_types=["tabular", "survival"],
        dataset_profile={"modality": "tabular", "target_semantics": "survival", "tags": ["survival"]},
        metric="concordance_index",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-survival-risk-ranking" in method_ids
    survival_method = next(method for method in methods if method.method_id == "tabular-survival-risk-ranking")
    assert "censoring" in survival_method.fallback.lower()
    assert survival_method.implementation_adapter["adapter"] == "tabular_survival_risk_ranking"


def test_method_scout_adds_pairwise_ranking_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-pairwise",
        problem_types=["tabular", "pairwise"],
        dataset_profile={"modality": "tabular", "target_semantics": "pairwise", "tags": ["pairwise"]},
        metric="logloss",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-pairwise-difference-ranking" in method_ids
    pairwise_method = next(method for method in methods if method.method_id == "tabular-pairwise-difference-ranking")
    assert "feature differences" in pairwise_method.fallback
    assert pairwise_method.implementation_adapter["adapter"] == "tabular_pairwise_difference_ranking"


def test_method_scout_adds_learning_to_rank_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-ltr",
        problem_types=["tabular", "learning_to_rank"],
        dataset_profile={"modality": "tabular", "target_semantics": "learning_to_rank", "tags": ["learning_to_rank"]},
        metric="ndcg",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-learning-to-rank-lambdamart" in method_ids
    ltr_method = next(method for method in methods if method.method_id == "tabular-learning-to-rank-lambdamart")
    assert "query-group validation" in ltr_method.summary
    assert ltr_method.implementation_adapter["adapter"] == "tabular_learning_to_rank_lambdamart"


def test_method_scout_adds_anomaly_detection_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-anomaly",
        problem_types=["tabular", "unsupervised", "anomaly_detection"],
        dataset_profile={
            "modality": "tabular",
            "task": "unsupervised",
            "target_semantics": "anomaly_detection",
            "tags": ["unsupervised", "anomaly_detection"],
        },
        metric="auc",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "tabular-anomaly-score-ensemble" in method_ids
    anomaly_method = next(method for method in methods if method.method_id == "tabular-anomaly-score-ensemble")
    assert "score ensembles" in anomaly_method.summary
    assert anomaly_method.implementation_adapter["adapter"] == "tabular_anomaly_score_ensemble"


def test_method_scout_adds_ctr_and_recommender_candidates() -> None:
    ctr_methods = build_method_candidates(
        slug="demo-ctr",
        problem_types=["tabular", "ctr"],
        dataset_profile={"modality": "tabular", "target_semantics": "ctr", "tags": ["ctr"]},
        metric="logloss",
        sources=[],
    )
    recommender_methods = build_method_candidates(
        slug="demo-recommender",
        problem_types=["tabular", "recommender"],
        dataset_profile={"modality": "tabular", "target_semantics": "recommender", "tags": ["recommender"]},
        metric="rmse",
        sources=[],
    )

    ctr_ids = [method.method_id for method in ctr_methods]
    recommender_ids = [method.method_id for method in recommender_methods]
    assert "tabular-ctr-gbdt-calibration" in ctr_ids
    assert "tabular-recommender-user-item-features" in recommender_ids
    ctr_method = next(method for method in ctr_methods if method.method_id == "tabular-ctr-gbdt-calibration")
    recommender_method = next(
        method for method in recommender_methods if method.method_id == "tabular-recommender-user-item-features"
    )
    assert "calibrated probabilities" in ctr_method.summary
    assert ctr_method.implementation_adapter["adapter"] == "tabular_ctr_gbdt_calibration"
    assert "user-item interactions" in recommender_method.summary
    assert recommender_method.implementation_adapter["adapter"] == "tabular_recommender_user_item_features"


def test_method_scout_adds_forecasting_candidate() -> None:
    methods = build_method_candidates(
        slug="demo-forecasting",
        problem_types=["timeseries", "forecasting"],
        dataset_profile={"modality": "timeseries", "target_semantics": "forecasting", "tags": ["forecasting"]},
        metric="rmse",
        sources=[],
    )

    method_ids = [method.method_id for method in methods]
    assert "timeseries-forecasting-backtest-lag-gbdt" in method_ids
    forecasting_method = next(
        method for method in methods if method.method_id == "timeseries-forecasting-backtest-lag-gbdt"
    )
    assert "horizon-aware validation" in forecasting_method.summary
    assert forecasting_method.implementation_adapter["adapter"] == "timeseries_forecasting_backtest_lag_gbdt"


def test_method_scout_adds_detection_and_segmentation_candidates() -> None:
    detection_methods = build_method_candidates(
        slug="demo-detection",
        problem_types=["image", "object_detection"],
        dataset_profile={"modality": "image", "target_semantics": "object_detection", "tags": ["object_detection"]},
        metric="map",
        sources=[],
    )
    segmentation_methods = build_method_candidates(
        slug="demo-segmentation",
        problem_types=["image", "segmentation"],
        dataset_profile={"modality": "image", "target_semantics": "segmentation", "tags": ["segmentation"]},
        metric="dice",
        sources=[],
    )

    detection_ids = [method.method_id for method in detection_methods]
    segmentation_ids = [method.method_id for method in segmentation_methods]
    assert "image-object-detection-router" in detection_ids
    assert "image-segmentation-mask-rle" in segmentation_ids
    detection_method = next(
        method for method in detection_methods if method.method_id == "image-object-detection-router"
    )
    segmentation_method = next(
        method for method in segmentation_methods if method.method_id == "image-segmentation-mask-rle"
    )
    assert detection_method.implementation_adapter["adapter"] == "image_object_detection_router"
    assert segmentation_method.implementation_adapter["adapter"] == "image_segmentation_mask_rle"


def test_method_scout_seeds_artifact_modality_without_tabular_fallback() -> None:
    for modality in ("artifact", "model-artifact"):
        methods = build_method_candidates(
            slug="demo-artifact",
            problem_types=["tabular"],
            dataset_profile={"modality": modality},
            metric="accuracy",
            sources=[],
        )

        assert methods
        assert methods[0].method_id.startswith("artifact-")
        assert methods[0].implementation_adapter["adapter"].startswith("artifact_")
        assert "artifact validation" in methods[0].dependency_check["fallback"].lower()


def test_method_scout_seeds_annotation_modality_without_tabular_fallback() -> None:
    assert infer_modality(["COCO annotation detection"], {}) == "annotation"
    assert infer_modality([], {"columns": ["id", "annotation_path", "target"]}) == "annotation"

    methods = build_method_candidates(
        slug="demo-annotation",
        problem_types=["tabular"],
        dataset_profile={"modality": "annotation"},
        metric="map",
        sources=[],
    )

    assert methods
    assert methods[0].method_id.startswith("annotation-")
    assert methods[0].implementation_adapter["adapter"].startswith("annotation_")
    assert "annotation conversion" in methods[0].dependency_check["fallback"].lower()


def test_method_scout_seeds_array_and_medical_imaging_modalities() -> None:
    for modality, normalized_modality, prefix in (
        ("array", "array", "array-"),
        ("medical-imaging", "medical_imaging", "medical-imaging-"),
    ):
        methods = build_method_candidates(
            slug=f"demo-{modality}",
            problem_types=["tabular"],
            dataset_profile={"modality": modality},
            metric="accuracy",
            sources=[],
        )

        assert methods
        assert methods[0].method_id.startswith(prefix)
        assert methods[0].implementation_adapter["adapter"].startswith(f"{normalized_modality}_")
        assert "torch" in methods[0].dependency_check["optional"]


def test_method_scout_seeds_audio_and_video_modalities() -> None:
    for modality, prefix in (("audio", "audio-"), ("video", "video-")):
        methods = build_method_candidates(
            slug=f"demo-{modality}",
            problem_types=["tabular"],
            dataset_profile={"modality": modality},
            metric="accuracy",
            sources=[],
        )

        assert methods
        assert methods[0].method_id.startswith(prefix)
        assert methods[0].implementation_adapter["adapter"].startswith(f"{modality}_")
        assert "torch" in methods[0].dependency_check["optional"]


def test_method_scout_prefers_medical_image_over_generic_image() -> None:
    assert infer_modality(["medical image classification"], {}) == "medical_imaging"
    assert infer_modality(["DICOM volume segmentation"], {}) == "medical_imaging"
    assert infer_modality(["NRRD MHA scan spacing prediction"], {}) == "medical_imaging"
    assert infer_modality(["SMILES molecule property prediction"], {}) == "bio"
    assert infer_modality(["InChI chemical similarity"], {}) == "bio"


def test_method_scout_infers_asset_modalities_from_columns_without_generic_filename_bias() -> None:
    assert infer_modality([], {"columns": ["id", "filename", "target"]}) == "tabular"
    assert infer_modality([], {"columns": ["id", "audio_path", "target"]}) == "audio"
    assert infer_modality([], {"columns": ["id", "clip_filename", "target"]}) == "video"
    assert infer_modality([], {"columns": ["id", "scan_path", "target"]}) == "medical_imaging"
    assert infer_modality([], {"columns": ["id", "volume.nrrd", "target"]}) == "medical_imaging"
    assert infer_modality([], {"columns": ["id", "volume.nhdr", "target"]}) == "medical_imaging"
    assert infer_modality([], {"columns": ["id", "slice.ima", "target"]}) == "medical_imaging"
    assert infer_modality([], {"columns": ["id", "lidar_file", "target"]}) == "point_cloud"
    assert infer_modality([], {"columns": ["id", "geometry", "target"]}) == "geospatial"
    assert infer_modality([], {"columns": ["id", "protein_path", "target"]}) == "bio"
    assert infer_modality([], {"columns": ["id", "molecule.smi", "target"]}) == "bio"
    assert infer_modality([], {"columns": ["id", "compound.inchi", "target"]}) == "bio"
    assert infer_modality([], {"columns": ["id", "rna_sequence", "target"]}) == "rna"
    assert infer_modality([], {"columns": ["id", "edge_index", "target"]}) == "graph"
    assert infer_modality([], {"columns": ["id", "array_file", "target"]}) == "array"
    assert infer_modality([], {"columns": ["id", "netcdf_path", "target"]}) == "array"
    assert infer_modality([], {"columns": ["id", "h5ad_path", "target"]}) == "array"
    assert infer_modality([], {"columns": ["id", "labels.ome.zarr", "target"]}) == "array"
    assert infer_modality([], {"columns": ["id", "volume.n5", "target"]}) == "array"
    assert infer_modality([], {"columns": ["id", "pdf_file", "target"]}) == "text"
    assert infer_modality([], {"columns": ["id", "document_path", "target"]}) == "text"
    assert infer_modality([], {"columns": ["id", "report.md", "target"]}) == "text"
    assert infer_modality([], {"columns": ["id", "deck.pptx", "target"]}) == "text"
    assert infer_modality([], {"columns": ["id", "paper.tex.gz", "target"]}) == "text"
    assert infer_modality([], {"columns": ["id", "document_id", "target"]}) == "tabular"
    assert infer_modality([], {"columns": ["id", "image_path", "target"]}) == "image"
    assert infer_modality([], {"columns": ["id", "image_path", "question", "target"]}) == "multimodal"
    assert infer_modality([], {"columns": ["id", "document_path", "question", "target"]}) == "text"


def test_method_scout_infers_modern_media_suffix_modalities() -> None:
    assert infer_modality(["AVIF classification"], {}) == "image"
    assert infer_modality(["OPUS speech tagging"], {}) == "audio"
    assert infer_modality(["M4V clip scoring"], {}) == "video"
    assert infer_modality([], {"columns": ["id", "frame_heic_path", "target"]}) == "image"
    assert infer_modality([], {"columns": ["id", "clip_opus_path", "target"]}) == "audio"
    assert infer_modality([], {"columns": ["id", "movie_wmv_path", "target"]}) == "video"
    assert infer_modality([], {"columns": ["id", "frame_heif_path", "caption", "target"]}) == "multimodal"


def test_method_scout_infers_document_problem_types_as_text() -> None:
    assert infer_modality(["PDF document classification"], {}) == "text"
    assert infer_modality(["EPUB document classification"], {}) == "text"
    assert infer_modality(["document understanding"], {}) == "text"
    assert infer_modality(["Markdown report scoring"], {}) == "text"


def test_effective_method_scout_mode_disables_auto_outside_top1() -> None:
    assert effective_method_scout_mode(requested_mode="auto", campaign_mode="standard") == "off"
    assert effective_method_scout_mode(requested_mode="auto", campaign_mode="top1") == "auto"
    assert effective_method_scout_mode(requested_mode="refresh", campaign_mode="standard") == "refresh"
    assert effective_method_scout_mode(requested_mode="off", campaign_mode="top1") == "off"


def test_registry_loaders_return_objects_or_empty_dict(tmp_path: Path) -> None:
    method_path = tmp_path / "method_registry.json"
    source_path = tmp_path / "source_registry.json"
    validation_path = tmp_path / "validation_registry.json"
    method_path.write_text('{"methods": []}', encoding="utf-8")
    source_path.write_text("[]", encoding="utf-8")

    assert load_method_registry(method_path) == {"methods": []}
    assert load_source_registry(source_path) == {}
    assert load_validation_registry(validation_path) == {}


def test_source_quality_blocks_unsafe_leaderboard_proxy_method() -> None:
    source = {
        "url": "https://example.com/solution",
        "title": "LB proxy exact test match solution",
        "extracted_technique": "Use leaderboard proxy and exact matching on test rows.",
        "takeaway": "unsafe",
    }

    assert unsafe_method_reason("Use leaderboard proxy and exact matching on test rows")
    methods = build_method_candidates(
        slug="demo",
        problem_types=["tabular:binary"],
        dataset_profile={},
        metric="auc",
        sources=[source],
    )

    blocked = [method for method in methods if method.status == "blocked"]
    assert blocked
    assert blocked[0].blocked_reason


def test_method_registry_ranks_competition_specific_above_generic(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="playground-series-s6e2", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "research_sources.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "url": "https://www.kaggle.com/competitions/playground-series-s6e2/code",
                        "title": "Top S6E2 CatBoost XGBoost LightGBM blend",
                        "extracted_technique": "Train CatBoost, XGBoost, LightGBM and logit blend OOF predictions.",
                        "takeaway": "Competition-specific GBDT blend is strong.",
                        "query": "playground-series-s6e2 Kaggle winning solution",
                    }
                ),
                json.dumps(
                    {
                        "url": "https://example.com/generic-blog",
                        "title": "Generic tabular tips",
                        "extracted_technique": "Try random forest.",
                        "takeaway": "Generic baseline.",
                        "query": "tabular tips",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    registry = run_method_scout(
        paths=paths,
        slug=paths.slug,
        problem_types=["tabular:binary"],
        dataset_profile={"n_rows": 100000},
        metric="auc",
        mode="refresh",
    )

    assert registry["active_method_ids"]
    first_method = registry["methods"][0]
    assert isinstance(first_method["implementation_adapter"], dict)
    assert isinstance(first_method["dependency_check"], dict)
    assert first_method["source_type"] in {"competition_specific", "generic"}
    assert any(method["source_type"] == "competition_specific" for method in registry["methods"])
    assert method_registry_path(paths.context_dir).exists()
    source_registry = json.loads(source_registry_path(paths.context_dir).read_text(encoding="utf-8"))
    assert source_registry["active_source_ids"]
    assert source_registry["planned_query_ids"]


def test_load_kaggle_discovery_sources_prefers_relevant_surface_diversity(tmp_path: Path) -> None:
    path = tmp_path / "kaggle_discovery.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "surface": "datasets",
                        "url": "https://www.kaggle.com/datasets/example/arc-data",
                        "title": "ARC data",
                        "summary": "ARC task corpus",
                        "query": "arc agi",
                        "relevance_score": 4.0,
                        "rank_score": 4.5,
                    },
                    {
                        "surface": "datasets",
                        "url": "https://www.kaggle.com/datasets/example/arc-more-data",
                        "title": "More ARC data",
                        "relevance_score": 3.5,
                        "rank_score": 4.0,
                    },
                    {
                        "surface": "models",
                        "url": "https://www.kaggle.com/models/example/arc-model",
                        "title": "ARC model",
                        "relevance_score": 2.0,
                        "rank_score": 3.0,
                    },
                    {
                        "surface": "code",
                        "url": "https://www.kaggle.com/code/example/arc-solver",
                        "title": "ARC solver",
                        "relevance_score": "broken",
                        "rank_score": 5.0,
                    },
                    {
                        "surface": "discussions",
                        "url": "https://example.com/not-kaggle",
                        "title": "External candidate",
                        "relevance_score": 5.0,
                        "rank_score": 5.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    sources = load_kaggle_discovery_sources(path, limit=3)

    assert [source["kaggle_surface"] for source in sources] == ["datasets", "models", "datasets"]
    assert sources[0]["takeaway"] == "ARC task corpus"


def test_method_scout_merges_kaggle_discovery_into_source_registry(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="arc-prize", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.kaggle_discovery_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "surface": "code",
                        "url": "https://www.kaggle.com/code/example/arc-solver",
                        "title": "ARC solver notebook",
                        "summary": "A competition-specific solver to reproduce and validate.",
                        "query": "arc prize",
                        "relevance_score": 4.0,
                        "rank_score": 5.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_method_scout(
        paths=paths,
        slug=paths.slug,
        problem_types=["grid-reasoning"],
        dataset_profile={"modality": "array"},
        metric="accuracy",
        mode="refresh",
        max_sources=4,
    )

    source_registry = json.loads(source_registry_path(paths.context_dir).read_text(encoding="utf-8"))
    assert any(source["title"] == "ARC solver notebook" for source in source_registry["sources"])


def test_method_registry_prompt_lists_active_and_blocked_methods(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    registry = run_method_scout(
        paths=paths,
        slug="demo",
        problem_types=["tabular:binary"],
        dataset_profile={},
        metric="auc",
        mode="refresh",
    )

    prompt = render_method_registry_for_prompt(registry)

    assert "Competition-specific method scout is active" in prompt
    assert "Active method candidates" in prompt
    assert "adapter=" in prompt


def test_classify_source_types() -> None:
    assert (
        classify_source(
            {"url": "https://www.kaggle.com/competitions/demo/code/ref", "title": "Demo top notebook"},
            slug="demo",
        )
        == "competition_specific"
    )
    assert classify_source({"url": "https://arxiv.org/abs/2501.12345"}, slug="demo") == "paper"
    assert classify_source({"url": "https://github.com/example/method"}, slug="demo") == "official_repo"


def test_validation_registry_records_public_regression_split_candidates() -> None:
    registry = build_validation_registry(
        slug="demo",
        problem_types=["tabular:binary"],
        campaign_state={
            "direction": "maximize",
            "latest_submission_score": 0.75,
            "champion_score": 0.82,
            "offline_online_correlation": 0.1,
        },
        sources=[],
    )

    assert registry["priority"] is True
    assert registry["public_regression_signal"] is True
    assert registry["next_action"] == "validation_redesign"
    assert registry["active_profile"] == "group_or_proxy_cv"
    assert all("run_status" in item for item in registry["profiles"])
    assert any(item["profile_id"] == "adversarial_proxy_cv" for item in registry["profiles"])


def test_validation_registry_uses_dataset_group_hint() -> None:
    registry = build_validation_registry(
        slug="demo",
        problem_types=["tabular:binary"],
        dataset_profile={
            "modality": "tabular",
            "split_strategy_hint": "group_kfold",
            "group_column_hint": "patient_id",
        },
        campaign_state=None,
        sources=[],
    )

    assert registry["active_profile"] == "entity_group_cv"
    entity_profile = next(item for item in registry["profiles"] if item["profile_id"] == "entity_group_cv")
    assert entity_profile["split_family"] == "group"
    assert entity_profile["group_column_hint"] == "patient_id"
    assert "patient_id" in entity_profile["reason"]


def test_research_scout_off_skips_source_registry_sources(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "research_sources.jsonl").write_text(
        json.dumps({"url": "https://arxiv.org/abs/2501.1", "title": "Strong paper"}) + "\n",
        encoding="utf-8",
    )

    registry = run_method_scout(
        paths=paths,
        slug="demo",
        problem_types=["tabular:binary"],
        dataset_profile={},
        metric="auc",
        mode="refresh",
        research_mode="off",
    )

    source_registry = json.loads(source_registry_path(paths.context_dir).read_text(encoding="utf-8"))
    assert registry["research_mode"] == "off"
    assert registry["source_count"] == 0
    assert source_registry["source_count"] == 0
