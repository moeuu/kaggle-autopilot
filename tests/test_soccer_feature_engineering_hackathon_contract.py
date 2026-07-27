from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ARTIFACT_ROOT = Path("/data/morita/kaggle-autopilot-artifacts/soccer-feature-engineering-hackathon")
KERNEL_PATH = ARTIFACT_ROOT / "kernel/kernel.py"
ARCHIVE_PATH = ARTIFACT_ROOT / "data/soccer-feature-engineering-hackathon.zip"
CANONICAL_SHA256 = "bdd20c5d9da6edadaab546dbf76403e464d48b33aac31bdbbc26b7c0f1cbb9b4"


@pytest.fixture(scope="session")
def soccer_kernel(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("soccer-kernel-import")
    previous_output = os.environ.get("KAGGLEBOT_OUTPUT_DIR")
    previous_archive = os.environ.get("KAGGLEBOT_COMPETITION_ARCHIVE")
    os.environ["KAGGLEBOT_OUTPUT_DIR"] = str(output)
    os.environ["KAGGLEBOT_COMPETITION_ARCHIVE"] = str(ARCHIVE_PATH)
    spec = importlib.util.spec_from_file_location("soccer_hackathon_kernel_contract", KERNEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    if previous_output is None:
        os.environ.pop("KAGGLEBOT_OUTPUT_DIR", None)
    else:
        os.environ["KAGGLEBOT_OUTPUT_DIR"] = previous_output
    if previous_archive is None:
        os.environ.pop("KAGGLEBOT_COMPETITION_ARCHIVE", None)
    else:
        os.environ["KAGGLEBOT_COMPETITION_ARCHIVE"] = previous_archive


@pytest.fixture(scope="session")
def canonical_data(soccer_kernel):
    sources, manifest = soccer_kernel.discover_event_sources()
    events = soccer_kernel.load_events(sources)
    return sources, manifest, events


def test_canonical_archive_hash_and_plan_manifest(soccer_kernel) -> None:
    observed = hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest()
    assert observed == CANONICAL_SHA256
    package = soccer_kernel.PLAN["input_manifest"]["package"]
    assert package == {
        "sha256": CANONICAL_SHA256,
        "expected_dynamic_event_members": 10,
        "expected_matches": 10,
        "expected_match_team_pairs": 20,
        "relative_path": "data/soccer-feature-engineering-hackathon.zip",
        "environment_override": "KAGGLEBOT_COMPETITION_ARCHIVE",
        "verify_hash_regardless_of_filename": True,
    }


def test_renamed_archive_discovery_verifies_hash(
    soccer_kernel, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    renamed = tmp_path / "authoritative-renamed-payload.zip"
    shutil.copyfile(ARCHIVE_PATH, renamed)
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_ARCHIVE", str(renamed))
    sources, manifest = soccer_kernel.discover_event_sources()
    assert len(sources) == 10
    assert manifest["package_level_verification"] == "verified"
    assert manifest["package_checks"][0]["path"] == str(renamed)
    assert manifest["package_checks"][0]["sha256"] == CANONICAL_SHA256


def test_wrong_hash_is_rejected_before_member_use(
    soccer_kernel, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wrong = tmp_path / "soccer-feature-engineering-hackathon.zip"
    with zipfile.ZipFile(wrong, "w") as archive:
        archive.writestr("1_dynamic_events.csv", "match_id,team_id,event_type\n1,10,player_possession\n")
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_ARCHIVE", str(wrong))
    with pytest.raises(soccer_kernel.DataDiscoveryError, match="SHA-256 mismatch"):
        soccer_kernel.discover_event_sources()


def test_unsafe_zip_member_is_rejected(soccer_kernel, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../1_dynamic_events.csv", "match_id,team_id,event_type\n1,10,player_possession\n")
    unsafe_sha = hashlib.sha256(unsafe.read_bytes()).hexdigest()
    monkeypatch.setenv("KAGGLEBOT_COMPETITION_ARCHIVE", str(unsafe))
    monkeypatch.setitem(soccer_kernel.INPUT_PACKAGE, "sha256", unsafe_sha)
    monkeypatch.setitem(soccer_kernel.INPUT_PACKAGE, "expected_dynamic_event_members", 1)
    with pytest.raises(ValueError, match="Unsafe ZIP member path"):
        soccer_kernel.discover_event_sources()


def test_ten_match_twenty_pair_contract(canonical_data) -> None:
    sources, manifest, events = canonical_data
    assert len(sources) == 10
    assert manifest["unique_match_count"] == 10
    assert events["match_id"].nunique() == 10
    assert events.groupby("match_id")["team_id"].nunique().eq(2).all()
    assert len(events[["match_id", "team_id"]].drop_duplicates()) == 20


def test_linked_candidate_delta_and_all_eleven_formulas(soccer_kernel, canonical_data) -> None:
    _, _, events = canonical_data
    matrix = soccer_kernel.build_linked_event_motif_graph_50(events)
    baseline = soccer_kernel.build_tactical_conversion_graph_49(events)
    added = [spec.name for spec in soccer_kernel.LINKED_MOTIF_ADDED_SPECS]
    assert len(matrix.columns) == 52
    assert len(added) == 11
    assert set(added) <= set(matrix.columns)
    assert soccer_kernel.LINKED_MOTIF_REMOVED_FEATURES.isdisjoint(matrix.columns)
    shared = [column for column in baseline if column in matrix and column not in soccer_kernel.KEY_COLS]
    assert soccer_kernel._frames_equal(
        baseline[soccer_kernel.KEY_COLS + shared], matrix[soccer_kernel.KEY_COLS + shared]
    )

    indexed = matrix.set_index(soccer_kernel.KEY_COLS)
    motif_masks = soccer_kernel.linked_motif_masks(events)
    for spec in soccer_kernel.LINKED_MOTIF_ADDED_SPECS:
        if spec.aggregation == "distinct_count":
            expected = soccer_kernel._group_distinct(events, motif_masks[spec.name], "pressing_chain_index")
        else:
            expected = soccer_kernel._group_count(events, motif_masks[spec.name])
        expected_values = expected.reindex(indexed.index, fill_value=0.0).to_numpy(dtype="float64")
        np.testing.assert_array_equal(indexed[spec.name].to_numpy(dtype="float64"), expected_values)


def test_linked_semantic_inequalities_and_full_matrix_quality(soccer_kernel, canonical_data) -> None:
    _, _, events = canonical_data
    matrix = soccer_kernel.build_linked_event_motif_graph_50(events)
    audit = soccer_kernel.validate_feature_matrix(matrix, soccer_kernel.LINKED_MOTIF_SPECS, events, final=True)
    quality = soccer_kernel.linked_candidate_quality_gate(matrix, audit)
    assert audit["n_rows"] == 20
    assert audit["n_features"] == 50
    assert audit["active_feature_count"] == 50
    assert audit["constant_features"] == []
    assert audit["exact_duplicate_pairs"] == []
    assert audit["spearman_pairs_at_or_above_0_95"] == []
    assert quality["passed"] is True
    assert (
        matrix["line_break_option_with_run_received_count"] <= matrix["line_break_option_with_run_targeted_count"]
    ).all()
    assert (matrix["line_break_option_with_run_targeted_count"] <= matrix["line_break_option_with_run_count"]).all()
    assert (matrix["received_line_break_inside_shape_count"] <= matrix["received_line_break_option_count"]).all()
    assert (matrix["long_pressing_chain_count"] <= matrix["pressing_chain_count"]).all()


def test_row_source_and_chunk_determinism(soccer_kernel, canonical_data) -> None:
    sources, _, events = canonical_data
    expected = soccer_kernel.build_linked_event_motif_graph_50(events)
    row_shuffled = soccer_kernel.stable_sort_events(events.sample(frac=1.0, random_state=73))
    source_reversed = soccer_kernel.load_events(list(reversed(sources)))
    alternate_chunk = soccer_kernel.load_events(sources, chunk_rows=997)
    assert soccer_kernel._frames_equal(soccer_kernel.build_linked_event_motif_graph_50(row_shuffled), expected)
    assert soccer_kernel._frames_equal(soccer_kernel.build_linked_event_motif_graph_50(source_reversed), expected)
    assert soccer_kernel._frames_equal(soccer_kernel.build_linked_event_motif_graph_50(alternate_chunk), expected)


def test_checkpoint_reuse_and_fingerprint_invalidation(soccer_kernel) -> None:
    fingerprints = {
        "source_manifest_sha256": "source",
        "plan_sha256": "plan",
        "feature_registry_sha256": "registry",
        "candidate": "linked_event_motif_graph_50",
        "seed": 42,
        "repeat": 1,
    }
    assignments = [{"fold": fold, "split_fingerprint": f"split-{fold}"} for fold in range(1, 6)]
    checkpoint = {
        "completed": True,
        "fingerprints": fingerprints,
        "fold_assignments": assignments,
        "completed_fold_records": [{"fold": fold} for fold in range(1, 6)],
    }
    assert soccer_kernel.checkpoint_can_resume(checkpoint, fingerprints, assignments, 5)
    changed_plan = {**fingerprints, "plan_sha256": "changed"}
    assert not soccer_kernel.checkpoint_can_resume(checkpoint, changed_plan, assignments, 5)
    changed_registry = {**fingerprints, "feature_registry_sha256": "changed"}
    assert not soccer_kernel.checkpoint_can_resume(checkpoint, changed_registry, assignments, 5)


def test_complete_thirty_fold_records_and_fingerprints(soccer_kernel) -> None:
    records = [{"split_fingerprint": f"split-{index}"} for index in range(30)]
    assert soccer_kernel.validate_complete_fold_records(records, n_folds=5, seed_count=3, repeats=2) == 30
    with pytest.raises(ValueError, match="Expected 30"):
        soccer_kernel.validate_complete_fold_records(records[:-1], n_folds=5, seed_count=3, repeats=2)


def test_candidate_readback_hash_supplies_real_four_point_proof(
    soccer_kernel, canonical_data, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, events = canonical_data
    matrix = soccer_kernel.build_linked_event_motif_graph_50(events)
    monkeypatch.setattr(soccer_kernel, "LOCAL_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(soccer_kernel, "OUTPUT_DIRS", [tmp_path])
    soccer_kernel.write_csv_all("candidate_linked_event_motif_graph_50.csv", matrix, output_dirs=[tmp_path])
    soccer_kernel.write_csv_all(
        "candidate_feature_dictionary_linked_event_motif_graph_50.csv",
        soccer_kernel.registry_frame(soccer_kernel.LINKED_MOTIF_SPECS),
        output_dirs=[tmp_path],
    )
    soccer_kernel.write_text_all(
        "candidate_feature_dictionary_linked_event_motif_graph_50.md",
        soccer_kernel.registry_markdown(soccer_kernel.LINKED_MOTIF_SPECS),
        output_dirs=[tmp_path],
    )
    artifact = soccer_kernel.validate_candidate_artifact(
        "linked_event_motif_graph_50", matrix, soccer_kernel.LINKED_MOTIF_SPECS
    )
    assert artifact["reread_hash_ok"] is True
    audit = soccer_kernel.validate_feature_matrix(matrix, soccer_kernel.LINKED_MOTIF_SPECS, events, final=True)
    common = dict(
        registry_ok=True,
        semantic_ok=True,
        orientation_ok=True,
        team_ok=True,
        isolation_ok=True,
        coverage_recorded=True,
        schema_ok=True,
        order_ok=True,
        provenance_ok=True,
        package_ok=True,
        phase_ok=True,
        matrix_audit=audit,
        case_study_ok=True,
        dictionary_writeup_ok=True,
        ablation_plot_ok=True,
    )
    pre, _ = soccer_kernel.contract_score(soccer_kernel.LINKED_MOTIF_SPECS, **common, reread_hash_ok=False)
    post, _ = soccer_kernel.contract_score(
        soccer_kernel.LINKED_MOTIF_SPECS,
        **common,
        reread_hash_ok=artifact["reread_hash_ok"],
    )
    assert pre == 96.0
    assert post == 100.0
    assert post - pre == 4.0


def test_forbidden_field_rejection(soccer_kernel, canonical_data) -> None:
    _, _, events = canonical_data
    bad = list(soccer_kernel.LINKED_MOTIF_SPECS)
    bad[0] = dataclasses.replace(bad[0], source_columns=("xthreat",), formula="count(xthreat)")
    with pytest.raises(soccer_kernel.CandidateUnavailableError, match="Forbidden"):
        soccer_kernel.audit_registry(bad, events)


def test_blend_stack_calibration_and_oof_are_blocked(soccer_kernel) -> None:
    categories = soccer_kernel.PLAN["candidate_categories"]
    for name in ("blend", "stack", "calibration", "oof_predictions"):
        assert categories[name] == "blocked_no_predictive_target_and_rules"
    assert soccer_kernel.SAVE_OOF_AND_TEST_NPY is False
    assert soccer_kernel.TRAINING_ENABLED is False
    assert soccer_kernel.PLAN["pipelines"][-1]["name"] == "geometry_only_raw_safety_30"
    assert "geometry_only_raw_safety_30" not in soccer_kernel.CANDIDATE_ORDER
    assert soccer_kernel.ENABLE_GEOMETRY_SAFETY_FALLBACK is False


def test_preflight_payload_is_serializable_and_complete(soccer_kernel, canonical_data) -> None:
    sources, manifest, events = canonical_data
    payload = soccer_kernel.write_input_preflight(sources, manifest, events)
    assert payload["archive_sha256"] == CANONICAL_SHA256
    assert payload["dynamic_event_member_count"] == 10
    assert payload["unique_match_count"] == 10
    assert payload["match_team_pair_count"] == 20
    assert len(payload["dynamic_event_members"]) == 10
    json.dumps(payload)
