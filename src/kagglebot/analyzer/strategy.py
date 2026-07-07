from __future__ import annotations

from kagglebot.analyzer.types import ModelingStrategy

_TEXT_TASKS = {"text", "translation", "text_generation"}
_ASSET_TASKS = {
    "image",
    "medical_imaging",
    "audio",
    "video",
    "signal",
    "array",
    "point_cloud",
    "geospatial",
    "bio",
    "rna",
    "graph",
    "annotation",
}
_ARTIFACT_TASKS = {"artifact", "code", "unknown"}


def build_strategy(
    task: str,
    *,
    prediction_kind: str | None = None,
    time_budget_minutes: int,
    cv_folds: int,
    models: list[str] | None,
    use_stacking: bool,
) -> ModelingStrategy:
    prediction_kind = (prediction_kind or "").strip().lower()
    if models:
        selected_models = [m.strip().lower() for m in models if m.strip()]
    else:
        if task == "unsupervised":
            selected_models = [
                "robust_unsupervised_anomaly_score",
                "isolation_forest_score",
                "one_class_svm_score",
                "autoencoder_anomaly_score",
            ]
        elif task == "survival":
            selected_models = [
                "survival_risk_score",
                "cox_ph_or_aft_baseline",
                "event_time_ranker",
                "censoring_aware_tree_ensemble",
            ]
        elif task == "learning_to_rank":
            selected_models = [
                "lambdamart_ranker",
                "groupwise_gradient_boosted_ranker",
                "pairwise_ranker",
                "ridge_relevance_score",
            ]
        elif task == "forecasting":
            selected_models = [
                "lag_feature_gradient_boosting",
                "rolling_window_ridge",
                "seasonal_naive_baseline",
                "grouped_time_series_ensemble",
            ]
        elif task == "count_regression":
            selected_models = [
                "poisson_gradient_boosting",
                "log1p_ridge_count_regressor",
                "tweedie_count_regressor",
                "rmsle_clipped_count_ensemble",
            ]
        elif task == "bounded_regression":
            selected_models = [
                "bounded_ridge_regressor",
                "beta_or_logit_transformed_regressor",
                "calibrated_gbdt_rate_regressor",
                "clipped_bounded_regression_ensemble",
            ]
        elif task == "positive_skew_regression":
            selected_models = [
                "log1p_ridge_regressor",
                "rmsle_gradient_boosting",
                "quantile_tail_robust_regressor",
                "non_negative_log_target_ensemble",
            ]
        elif task == "pairwise":
            selected_models = [
                "pairwise_difference_ranker",
                "bradley_terry_calibrator",
                "elo_feature_logistic",
                "matchup_gradient_boosting",
            ]
        elif task == "object_detection":
            selected_models = [
                "object_detection_router",
                "yolo_or_detectron_baseline",
                "prediction_string_formatter",
                "nms_tta_calibrator",
            ]
        elif task == "segmentation":
            selected_models = [
                "rle_empty_mask_baseline",
                "unet_mask_baseline",
                "mask_rcnn_or_segmentation_router",
                "rle_threshold_tuner",
            ]
        elif task == "multi_label":
            selected_models = [
                "one_vs_rest_multilabel_logreg",
                "calibrated_multilabel_extra_trees",
                "per_label_threshold_tuning",
                "tfidf_label_set_retrieval",
            ]
        elif task == "coordinate_regression":
            selected_models = [
                "coordinate_gbdt_regressor",
                "per_axis_ridge_regressor",
                "multi_output_coordinate_ensemble",
                "coordinate_scale_postprocessor",
            ]
        elif task == "multi_output_regression":
            selected_models = [
                "multi_output_ridge",
                "per_target_extra_trees_regressor",
                "multi_output_hist_gb_regressor",
                "target_correlation_blender",
            ]
        elif task == "multi_target_classification":
            selected_models = [
                "multi_output_logreg",
                "per_target_calibrated_extra_trees",
                "classifier_chain_baseline",
                "per_target_threshold_tuning",
            ]
        elif task == "multi_task":
            selected_models = [
                "per_target_type_router",
                "shared_feature_multi_head_baseline",
                "per_target_metric_blender",
                "submission_column_contract_validator",
            ]
        elif task == "rna_structure":
            selected_models = [
                "rna_coordinate_mean_baseline",
                "sequence_residue_coordinate_regressor",
                "per_residue_position_coordinate_prior",
                "coordinate_triplet_postprocessor",
            ]
        elif task == "geospatial":
            selected_models = [
                "geospatial_feature_gbdt",
                "spatial_group_validation_baseline",
                "geometry_distance_feature_head",
                "coordinate_interaction_ensemble",
            ]
        elif task == "graph":
            selected_models = [
                "graph_topology_feature_gbdt",
                "node_edge_degree_feature_head",
                "link_prediction_calibrator",
                "optional_graph_embedding_branch",
            ]
        elif task == "signal":
            selected_models = [
                "signal_statistical_feature_gbdt",
                "frequency_domain_feature_head",
                "ecg_eeg_waveform_encoder",
                "leak_safe_signal_validation",
            ]
        elif task == "annotation":
            selected_models = [
                "annotation_format_feature_gbdt",
                "coco_yolo_labelme_converter",
                "detection_segmentation_reference_router",
                "annotation_contract_validator",
            ]
        elif task in {"bio", "rna"}:
            selected_models = [
                "sequence_kmer_feature_gbdt",
                "molecule_or_sequence_descriptor_head",
                "domain_embedding_feature_head",
                "calibrated_bio_tabular_ensemble",
            ]
        elif task == "multimodal":
            selected_models = [
                "multimodal_late_fusion_baseline",
                "asset_metadata_text_feature_head",
                "clip_or_embedding_feature_branch",
                "tabular_metadata_blender",
            ]
        elif task == "text_classification":
            selected_models = [
                "tfidf_logreg_text_classifier",
                "linear_svm_text_classifier",
                "sentence_embedding_classifier",
                "calibrated_text_tabular_blend",
            ]
        elif task == "text_regression":
            selected_models = [
                "tfidf_ridge_text_regressor",
                "sentence_embedding_regressor",
                "text_length_stat_gbdt",
                "text_tabular_regression_blend",
            ]
        elif task == "ctr":
            selected_models = [
                "ctr_gradient_boosting_calibrated",
                "field_aware_factorization_machine",
                "target_encoded_linear_ctr",
                "catboost_user_item_ctr",
            ]
        elif task == "recommender":
            selected_models = [
                "user_item_aggregate_features",
                "matrix_factorization_baseline",
                "factorization_machine_regressor",
                "gradient_boosting_recommender",
            ]
        elif prediction_kind == "quantile_columns":
            selected_models = [
                "gradient_boosting_quantile_heads",
                "lightgbm_quantile_heads",
                "conformal_quantile_regressor",
                "ridge_quantile_residual_baseline",
            ]
        elif prediction_kind == "prediction_interval_columns":
            selected_models = [
                "conformal_interval_regressor",
                "gradient_boosting_interval_heads",
                "quantile_interval_ensemble",
                "ridge_residual_interval_baseline",
            ]
        elif prediction_kind == "ordinal":
            selected_models = [
                "ridge_ordered_score",
                "ordinal_threshold_tuning",
                "extra_trees_ordered_score",
                "catboost_ordered_target",
            ]
        elif prediction_kind == "probability_columns":
            selected_models = [
                "multinomial_logreg",
                "calibrated_extra_trees",
                "hist_gb_multiclass",
                "catboost_multiclass",
            ]
        elif prediction_kind == "probability":
            selected_models = [
                "logreg_probability",
                "calibrated_sgd_classifier",
                "calibrated_extra_trees",
                "hist_gb_probability",
                "catboost_probability",
            ]
        elif task in _TEXT_TASKS:
            selected_models = [
                "constant_text",
                "tfidf_nearest_neighbor",
                "sentence_embedding_retrieval",
                "seq2seq_text_runtime",
            ]
        elif task == "image":
            selected_models = [
                "reference_notebook_baseline",
                "image_metadata_feature_gbdt",
                "timm_or_torchvision_backbone",
                "image_embedding_tabular_head",
                "tta_calibrated_image_ensemble",
            ]
        elif task == "medical_imaging":
            selected_models = [
                "reference_notebook_baseline",
                "medical_image_metadata_gbdt",
                "medical_header_windowing_baseline",
                "slice_or_volume_embedding_head",
                "medical_imaging_tta_ensemble",
            ]
        elif task == "audio":
            selected_models = [
                "reference_notebook_baseline",
                "audio_metadata_feature_gbdt",
                "mel_spectrogram_cnn_baseline",
                "audio_embedding_tabular_head",
                "clip_duration_calibrated_ensemble",
            ]
        elif task == "video":
            selected_models = [
                "reference_notebook_baseline",
                "video_metadata_feature_gbdt",
                "frame_sampling_image_backbone",
                "temporal_pooling_embedding_head",
                "video_runtime_inference_pipeline",
            ]
        elif task == "array":
            selected_models = [
                "reference_notebook_baseline",
                "array_shape_stat_feature_gbdt",
                "scientific_array_summary_head",
                "npz_channel_pooling_baseline",
                "array_runtime_contract_validator",
            ]
        elif task == "point_cloud":
            selected_models = [
                "reference_notebook_baseline",
                "point_cloud_metadata_feature_gbdt",
                "geometric_stat_feature_head",
                "voxel_projection_baseline",
                "pointnet_optional_branch",
            ]
        elif task in {"geospatial", "bio", "rna", "graph"}:
            selected_models = [
                "reference_notebook_baseline",
                "domain_feature_extractor",
                "runtime_inference_pipeline",
            ]
        elif task in _ARTIFACT_TASKS:
            selected_models = [
                "reference_notebook_baseline",
                "submission_artifact_builder",
                "runtime_contract_validator",
            ]
        elif task == "classification":
            selected_models = [
                "logreg",
                "sgd_classifier",
                "extra_trees",
                "hist_gb",
                "catboost",
            ]
        else:
            selected_models = [
                "ridge",
                "sgd_regressor",
                "extra_trees",
                "hist_gb",
                "catboost",
            ]

    if task == "unsupervised":
        preprocessing = [
            "fit_unsupervised_scores_without_train_labels",
            "impute_median_numeric",
            "impute_mode_categorical",
            "robust_scale_numeric_features",
            "align_scores_to_sample_submission_ids",
        ]
    elif task == "survival":
        preprocessing = [
            "preserve_event_and_time_targets",
            "derive_single_risk_score_submission",
            "handle_censoring_in_validation",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "learning_to_rank":
        preprocessing = [
            "preserve_query_groups",
            "preserve_candidate_ids",
            "grouped_ndcg_validation",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "forecasting":
        preprocessing = [
            "sort_by_time_column",
            "chronological_holdout_validation",
            "derive_lag_and_rolling_features",
            "preserve_entity_group_columns",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "count_regression":
        preprocessing = [
            "preserve_non_negative_count_target",
            "log1p_transform_count_target_candidates",
            "clip_negative_count_predictions_to_zero",
            "evaluate_rmsle_on_validation",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "bounded_regression":
        preprocessing = [
            "preserve_bounded_rate_or_ratio_target",
            "infer_prediction_bounds_from_training_target",
            "clip_predictions_to_observed_bounds",
            "calibrate_bounded_regression_outputs",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "positive_skew_regression":
        preprocessing = [
            "preserve_non_negative_skewed_target",
            "log1p_transform_target_candidates",
            "evaluate_rmsle_on_validation",
            "clip_negative_predictions_to_zero",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "pairwise":
        preprocessing = [
            "preserve_pair_entity_columns",
            "derive_pairwise_difference_features",
            "calibrate_matchup_probabilities",
            "normalize_multi_outcome_probabilities",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "object_detection":
        preprocessing = [
            "preserve_image_file_references",
            "parse_prediction_string_contract",
            "validate_box_coordinate_format",
            "format_prediction_string_outputs",
            "preserve_required_submission_ids",
        ]
    elif task == "segmentation":
        preprocessing = [
            "preserve_image_file_references",
            "parse_mask_or_rle_submission_contract",
            "validate_rle_or_mask_artifact_format",
            "format_empty_or_predicted_masks_as_rle",
            "preserve_required_submission_ids",
        ]
    elif task == "multi_label":
        preprocessing = [
            "preserve_binary_indicator_label_columns",
            "split_delimited_label_sets",
            "map_label_columns_to_observed_labels",
            "fit_one_vs_rest_label_heads",
            "tune_per_label_thresholds",
            "format_label_sets_or_probability_columns",
        ]
    elif task == "coordinate_regression":
        preprocessing = [
            "preserve_coordinate_target_columns",
            "validate_coordinate_axis_groups",
            "fit_one_regression_head_per_coordinate_axis",
            "track_coordinate_rmse_by_axis",
            "align_coordinate_predictions_to_sample_columns",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "multi_output_regression":
        preprocessing = [
            "preserve_all_target_columns",
            "fit_one_regression_head_per_target",
            "track_per_target_validation_metrics",
            "align_2d_predictions_to_sample_columns",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "multi_target_classification":
        preprocessing = [
            "preserve_all_target_columns",
            "encode_each_class_target_independently",
            "fit_one_classifier_head_per_target",
            "track_per_target_classification_metrics",
            "format_class_or_probability_target_columns",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "multi_task":
        preprocessing = [
            "preserve_all_target_columns",
            "infer_task_and_metric_per_target",
            "route_each_target_to_matching_head",
            "combine_per_target_validation_summary",
            "align_outputs_to_sample_columns",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "rna_structure":
        preprocessing = [
            "preserve_sequence_tables",
            "join_residue_coordinate_labels_by_target_id",
            "preserve_residue_anchor_columns",
            "validate_coordinate_triplets",
            "evaluate_holdout_coordinate_rmse",
            "replicate_coordinates_across_required_triplets",
        ]
    elif task == "geospatial":
        preprocessing = [
            "preserve_latitude_longitude_or_geometry_columns",
            "derive_spatial_distance_and_bbox_features",
            "use_spatial_or_grouped_validation_when_possible",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "graph":
        preprocessing = [
            "preserve_node_edge_identifier_columns",
            "derive_degree_and_topology_features",
            "avoid_edge_leakage_in_validation_split",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task in {"bio", "rna"}:
        preprocessing = [
            "preserve_sequence_smiles_or_structure_columns",
            "derive_kmer_or_molecular_descriptor_features",
            "use_grouped_validation_for_shared_entities_when_possible",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "multimodal":
        preprocessing = [
            "preserve_asset_reference_and_text_columns",
            "derive_asset_metadata_features",
            "derive_text_or_embedding_features",
            "late_fuse_tabular_text_asset_features",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "text_classification":
        preprocessing = [
            "preserve_text_feature_columns",
            "normalize_and_vectorize_text_features",
            "derive_text_length_and_token_stats",
            "calibrate_text_classification_outputs",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "text_regression":
        preprocessing = [
            "preserve_text_feature_columns",
            "normalize_and_vectorize_text_features",
            "derive_text_length_and_token_stats",
            "blend_text_and_tabular_regression_features",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "ctr":
        preprocessing = [
            "preserve_user_item_ids",
            "derive_user_item_frequency_features",
            "target_encode_high_cardinality_ids_on_train_folds",
            "calibrate_ctr_probabilities",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif task == "recommender":
        preprocessing = [
            "preserve_user_item_ids",
            "derive_user_item_aggregate_features",
            "build_matrix_factorization_fallback_features",
            "clip_predictions_to_observed_rating_range",
            "impute_median_numeric",
            "impute_mode_categorical",
        ]
    elif prediction_kind == "quantile_columns":
        preprocessing = [
            "map_quantile_submission_columns_to_target",
            "impute_median_numeric",
            "impute_mode_categorical",
            "fit_quantile_heads_or_residual_offsets",
            "enforce_non_crossing_quantiles",
        ]
    elif prediction_kind == "prediction_interval_columns":
        preprocessing = [
            "map_interval_submission_columns_to_target",
            "impute_median_numeric",
            "impute_mode_categorical",
            "fit_interval_or_conformal_residuals",
            "enforce_lower_upper_order",
        ]
    elif prediction_kind == "ordinal":
        preprocessing = [
            "preserve_ordered_target_values",
            "impute_median_numeric",
            "impute_mode_categorical",
            "tune_monotonic_thresholds_on_validation",
            "round_and_clip_to_valid_labels",
        ]
    elif prediction_kind == "probability_columns":
        preprocessing = [
            "map_sample_probability_columns_to_class_labels",
            "impute_median_numeric",
            "impute_mode_categorical",
            "onehot_encode_for_linear_models",
            "renormalize_probability_columns",
        ]
    elif prediction_kind == "probability":
        preprocessing = [
            "preserve_probability_output_contract",
            "impute_median_numeric",
            "impute_mode_categorical",
            "onehot_encode_for_linear_models",
            "calibrate_probabilities_when_possible",
        ]
    elif task in _TEXT_TASKS:
        preprocessing = [
            "normalize_text",
            "preserve_ids_and_required_columns",
            "fallback_constant_or_nearest_neighbor",
            "optional_seq2seq_tokenization",
        ]
    elif task == "image":
        preprocessing = [
            "preserve_image_file_references",
            "derive_image_metadata_features",
            "prepare_optional_image_embedding_cache",
            "validate_image_submission_contract",
        ]
    elif task == "medical_imaging":
        preprocessing = [
            "preserve_medical_image_references",
            "derive_medical_header_metadata_features",
            "normalize_scan_shape_and_spacing_metadata",
            "validate_medical_imaging_submission_contract",
        ]
    elif task == "audio":
        preprocessing = [
            "preserve_audio_file_references",
            "derive_audio_metadata_features",
            "prepare_optional_spectrogram_features",
            "validate_audio_submission_contract",
        ]
    elif task == "video":
        preprocessing = [
            "preserve_video_file_references",
            "derive_video_metadata_features",
            "prepare_frame_sampling_plan",
            "validate_video_submission_contract",
        ]
    elif task == "signal":
        preprocessing = [
            "preserve_signal_file_references",
            "derive_waveform_summary_and_frequency_features",
            "prepare_optional_signal_windowing_plan",
            "validate_signal_submission_contract",
        ]
    elif task == "annotation":
        preprocessing = [
            "preserve_annotation_file_references",
            "derive_annotation_count_bbox_mask_features",
            "prepare_coco_yolo_labelme_conversion_plan",
            "validate_annotation_submission_contract",
        ]
    elif task == "array":
        preprocessing = [
            "preserve_array_file_references",
            "derive_array_shape_and_summary_features",
            "avoid_loading_full_array_payloads_when_possible",
            "validate_array_submission_contract",
        ]
    elif task == "point_cloud":
        preprocessing = [
            "preserve_point_cloud_references",
            "derive_point_count_and_bbox_features",
            "prepare_optional_voxel_or_projection_features",
            "validate_point_cloud_submission_contract",
        ]
    elif task in _ASSET_TASKS:
        preprocessing = [
            "preserve_file_structure",
            "derive_runtime_test_ids_from_assets",
            "use_reference_notebook_transforms",
            "validate_artifact_manifest_or_sample_contract",
        ]
    elif task in _ARTIFACT_TASKS:
        preprocessing = [
            "inspect_submission_format_and_rules",
            "preserve_required_archive_structure",
            "validate_artifact_manifest_or_sample_contract",
        ]
    else:
        preprocessing = [
            "impute_median_numeric",
            "impute_mode_categorical",
            "onehot_encode_for_linear_models",
            "ordinal_or_native_categorical_for_tree_models",
        ]

    return ModelingStrategy(
        preprocessing=preprocessing,
        models=selected_models,
        cv_folds=cv_folds,
        use_stacking=use_stacking,
        time_budget_minutes=time_budget_minutes,
    )
