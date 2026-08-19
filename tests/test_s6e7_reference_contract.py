from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.competition_artifact

ARTIFACT_ROOT = Path(os.environ.get("KAGGLEBOT_ARTIFACTS_DIR", "artifacts")) / "playground-series-s6e7"
KERNEL_PATH = ARTIFACT_ROOT / "kernel/kernel.py"
PLAN_PATH = ARTIFACT_ROOT / "plan.json"


@pytest.fixture(scope="module")
def kernel():
    spec = importlib.util.spec_from_file_location("s6e7_reference_contract_kernel", KERNEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probabilities(rows: int, favored: int = 0) -> np.ndarray:
    values = np.full((rows, 3), 0.1, dtype=np.float32)
    values[:, favored] = 0.8
    return values


def _candidate(kernel, name: str, rows: int, *, trusted: bool = True, source: str = "cv"):
    return kernel.CandidateResult(
        name=name,
        pipeline=name,
        oof=_probabilities(rows),
        test=_probabilities(2),
        writes=np.ones(rows, dtype=np.int8),
        fold_metrics=[],
        elapsed_minutes=0.0,
        details={
            "trusted_for_cv": trusted,
            "score_source": source,
            "class_order": list(kernel.CLASS_NAMES),
            "forbidden_provenance": [],
            "oof_coverage": 1.0,
        },
    )


def test_plan_is_one_three_fold_source_of_truth() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["cv_folds"] == 3
    assert plan["evaluation_protocol"]["n_folds"] == 3
    assert plan["toggles"]["TREE_FOLDS"] == 3
    assert plan["toggles"]["HEAVY_FOLDS"] == 3
    assert plan["stability_seeds"] == [2024, 777]
    assert plan["toggles"]["MAX_BLEND_MEMBERS"] == 6
    assert plan["toggles"]["BLEND_META_FOLDS"] == 5
    assert plan["toggles"]["BLEND_META_SEEDS"] == "3407,2024,777"
    assert plan["toggles"]["CALIBRATION_METHODS"] == "identity,temperature,multinomial_logit"
    assert plan["toggles"]["CALIBRATOR_C_GRID"] == "0.03,0.1,0.3,1.0"
    assert plan["toggles"]["TABICL_START_PROFILE_RTX3060"] == "ft"
    assert plan["toggles"]["TABICL_FORCE_PROBE"] is False


def test_target_encoding_is_self_excluding(kernel) -> None:
    rows = 18
    frame = pd.DataFrame({"unique_value": [f"row-{index}" for index in range(rows)]})
    y = np.tile(np.arange(3, dtype=np.int64), rows // 3)
    encoder = kernel.FoldLocalMulticlassTargetEncoder(columns=["unique_value"], smoothing=1.0, seed=42)

    encoded = encoder.fit_transform_oof(frame, y, inner_folds=3)

    assert encoded.shape == (rows, 3)
    np.testing.assert_allclose(encoded.sum(axis=1), 1.0, atol=1e-6)
    assert encoder.audit_["all_rows_exactly_once"] is True
    assert all(record["self_excluding"] for record in encoder.audit_["folds"])
    # Every key is unique, so a self-excluding validation transform must fall
    # back to the inner-training prior instead of reproducing its own label.
    assert np.max(encoded[np.arange(rows), y]) < 0.8


def test_target_encoding_unseen_category_uses_training_prior(kernel) -> None:
    frame = pd.DataFrame({"category": ["a", "a", "b", "b", "c", "c"]})
    y = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
    encoder = kernel.FoldLocalMulticlassTargetEncoder(columns=["category"], smoothing=2.0).fit(frame, y)

    unseen = encoder.transform(pd.DataFrame({"category": ["never-seen"]}))

    np.testing.assert_allclose(unseen[0], np.bincount(y, minlength=3) / len(y), atol=1e-7)


def test_probability_class_order_is_remapped_to_canonical(kernel) -> None:
    source = np.array([[0.60, 0.30, 0.10]], dtype=np.float32)

    remapped = kernel.reorder_probabilities(source, ["unhealthy", "at-risk", "fit"])

    np.testing.assert_allclose(remapped, [[0.30, 0.10, 0.60]], atol=1e-7)


def test_selection_requires_exact_full_oof_coverage(kernel) -> None:
    candidate = _candidate(kernel, "complete", 9)
    kernel.validate_full_oof_coverage([candidate], expected_rows=9)
    candidate.writes[-1] = 0

    with pytest.raises(kernel.DataContractError, match="cannot enter selection"):
        kernel.validate_full_oof_coverage([candidate], expected_rows=9)


def test_split_fingerprints_are_deterministic(kernel) -> None:
    y = np.tile(np.arange(3, dtype=np.int64), 20)
    regimes = np.zeros(len(y), dtype=np.int8)
    row_hash = np.arange(len(y), dtype=np.uint64)
    audit = {
        "duplicate_row_fraction": 0.0,
        "largest_group": 1,
        "conflicting_label_duplicate_groups": 0,
    }

    _, first_assignments, first = kernel.make_primary_splits(y, regimes, row_hash, audit)
    _, second_assignments, second = kernel.make_primary_splits(y, regimes, row_hash, audit)

    np.testing.assert_array_equal(first_assignments, second_assignments)
    assert first == second
    assert first["seed"] == 42
    assert first["assignment_sha256"] == kernel.sha256_array(first_assignments)


def test_probed_reference_output_is_excluded_from_cv_selection(kernel) -> None:
    honest = _candidate(kernel, "honest", 6)
    probed = _candidate(
        kernel,
        "submission_reference_probed",
        6,
        trusted=False,
        source="public_lb_reference",
    )

    selected = kernel.selectable_candidates([probed, honest], expected_rows=6)

    assert [candidate.name for candidate in selected] == ["honest"]
    assert kernel.candidate_is_trusted_for_cv(probed) is False


def test_submission_schema_and_sample_order_are_exact(kernel) -> None:
    test_ids = pd.Series([103, 101, 102], name="id")
    sample = pd.DataFrame({"id": test_ids, "health_condition": ["at-risk"] * 3})
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float32,
    )

    submission = kernel.build_submission(test_ids, sample, probabilities)

    assert list(submission.columns) == ["id", "health_condition"]
    assert submission["id"].tolist() == [103, 101, 102]
    assert submission["health_condition"].tolist() == ["at-risk", "fit", "unhealthy"]
    kernel.validate_submission(submission, sample, test_ids, probabilities)


def test_mandatory_reference_marker_is_exact() -> None:
    source = KERNEL_PATH.read_text(encoding="utf-8")
    assert "# KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED: najiama/post-processing-calibration-lb-0-95307" in source


def test_candidate_categories_and_metric_aliases(kernel) -> None:
    strong = _candidate(kernel, "te_hgbc_iter2_replay", 6)
    feature = _candidate(kernel, "xgb_native_categorical", 6)
    validation = _candidate(kernel, "meta_stability_audit", 6)
    validation.details["candidate_category"] = "validation_variant"
    calibration = _candidate(kernel, "calibrated_te_hgbc_iter2_replay", 6)
    calibration.details["candidate_category"] = "calibration"
    blend = _candidate(kernel, "nested_geometric", 6)
    blend.pipeline = "nested_oof_ensemble"

    assert kernel._candidate_category(strong) == "strong_single"
    assert kernel._candidate_category(feature) == "feature_variant"
    assert kernel._candidate_category(validation) == "validation_variant"
    assert kernel._candidate_category(calibration) == "calibration"
    assert kernel._candidate_category(blend) == "blend"

    aliases = kernel.finite_metric_aliases(0.95)
    numeric_aliases = ("score", "balanced_accuracy", "metric_value", "cv_balanced_accuracy")
    assert all(np.isfinite(aliases[name]) and aliases[name] == 0.95 for name in numeric_aliases)


def test_missingness_mixture_reliability_shrinkage(kernel) -> None:
    global_probability = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]], dtype=np.float32)
    specialist_probability = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    regime_rows = np.array([0.0, 900.0])

    mixed = kernel._regime_shrink(global_probability, specialist_probability, regime_rows, 100.0)

    np.testing.assert_allclose(mixed[0], global_probability[0], atol=1e-7)
    np.testing.assert_allclose(mixed[1], 0.1 * global_probability[1] + 0.9 * specialist_probability[1])


def test_reference_audit_isolates_public_components(kernel, tmp_path: Path) -> None:
    report = kernel.audit_required_reference(tmp_path)

    assert report["kernel_id"] == "najiama/post-processing-calibration-lb-0-95307"
    assert report["reference_gate_resolved"] is True
    assert report["blocks_novelty"] is False
    assert report["closest_leak_free_fallback"] == "cross-fitted member calibration plus nested OOF blend"
    assert report["public_reproduction"]["trusted_for_cv"] is False
    assert report["public_reproduction"]["default_submission"] is False
    public_components = [item for item in report["components"] if not item["trusted_for_cv"]]
    assert {item["component"] for item in public_components} >= {
        "public_submission_anchor",
        "public_score_weighting",
        "hard_label_voting",
        "row_corrections",
    }


def test_contract_smoke_covers_iteration3_attribution(kernel, tmp_path: Path) -> None:
    report = kernel.contract_smoke(tmp_path)
    decision = report["model_selection_decision"]

    assert report["status"] == "passed"
    assert decision["trusted"] is True
    assert decision["trust_errors"] == []
    required = {
        "metric",
        "source",
        "value",
        "direction",
        "fold_count",
        "seeds",
        "evaluated_rows",
        "data_file_hashes",
        "evaluation_mask_sha256",
        "class_list",
        "class_mapping_sha256",
        "fold_scores",
        "split_index_fingerprints",
        "split_assignment_sha256",
        "biometric_sha256",
        "mapping_sha256",
    }
    assert required <= decision.keys()
    assert all(decision[key] is not None for key in required)
    assert np.isfinite(decision["value"])
    assert report["member_calibration"]["self_excluding"] is True
    assert report["member_calibration"]["all_fold_overlaps_false"] is True
    assert report["all_member_blend_consideration"]["score_first_cap_used"] is False
    assert len(report["all_member_blend_consideration"]["kept_members"]) == 6
    assert report["candidate_contracts"]["complete_paths"] is True
    assert report["rtx3060_route"]["start_profile"] == "ft"
    assert report["rtx3060_route"]["force_probe"] is False
    assert report["submission_contract"]["columns"] == ["id", "health_condition"]

    index = json.loads(Path(report["candidate_contracts"]["index_path"]).read_text(encoding="utf-8"))
    assert index["candidate_contract_ingestion"]["ingested"] is True
    assert all(Path(item["oof_path"]).is_file() for item in index["contracts"])
    assert all(Path(item["test_path"]).is_file() for item in index["contracts"])
    categories = {item["candidate_category"] for item in index["contracts"]}
    assert {"strong_single", "feature_variant"} <= categories
