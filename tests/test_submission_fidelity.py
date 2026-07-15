from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.hashing import sha256_file_or_none, sha256_text
from kagglebot.submission_fidelity import (
    QUARANTINE_STATE_KEY,
    enforce_submission_fidelity_quarantine,
    load_active_submission_fidelity_quarantine,
    persist_leaderboard_outcome_quarantine,
    reserve_quarantine_repair_attempt,
    validate_file_submission_fidelity,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    return _write(path, json.dumps(payload))


def _metrics(path: Path, **updates: object) -> Path:
    payload: dict[str, object] = {
        "chosen_pipeline": "catboost_blend",
        "metric": "accuracy",
        "direction": "maximize",
        "score_source": "cv",
        "score": 0.81,
        "authoritative": True,
        "fallback_submission": False,
        "prediction_source_distribution": {"sources": {"catboost": 3}},
    }
    payload.update(updates)
    return _write_json(path, payload)


def _semantic_report(path: Path, submission: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "submission_sha256": sha256_file_or_none(submission),
            "block_submit": False,
            "findings": [],
        },
    )


def _validate(
    tmp_path: Path,
    *,
    submission: Path,
    sample: Path,
    metrics: Path | None,
    code_fingerprint: str = "code-v2",
    quarantine_state: dict[str, object] | None = None,
) -> dict[str, object]:
    semantic = _semantic_report(tmp_path / "submission_semantic_preflight.json", submission)
    return validate_file_submission_fidelity(
        slug="demo",
        run_id="run-1",
        iteration=1,
        source_candidate_path=submission,
        prepared_submission_path=submission,
        code_fingerprint=code_fingerprint,
        metrics_path=metrics,
        semantic_report_path=semantic,
        sample_submission_path=sample,
        report_path=tmp_path / "logs" / "submission_fidelity_report-file.json",
        expected_contract_path=tmp_path / "logs" / "submission_fidelity_expected-file.json",
        quarantine_state=quarantine_state,
        score_value=0.81,
        score_direction="maximize",
    )


def test_file_fidelity_passes_with_exact_local_identity_and_trusted_provenance(tmp_path: Path) -> None:
    sample = _write(tmp_path / "sample_submission.csv", "id,target\n1,0\n2,0\n3,0\n")
    submission = _write(tmp_path / "submission.csv", "id,target\n1,0.1\n2,0.7\n3,0.2\n")

    report = _validate(tmp_path, submission=submission, sample=sample, metrics=_metrics(tmp_path / "metrics.json"))

    assert report["schema_version"] == 2
    assert report["report_type"] == "SubmissionFidelityReport"
    assert report["verdict"] == "pass"
    assert report["attestation_scope"] == "local_prepared_artifact"
    assert report["remote_runtime_attested"] is False
    assert report["metric_provenance"]["trusted"] is True
    assert report["selected_output"]["sha256"] == sha256_file_or_none(submission)
    assert report["prediction_evidence"]["identifier"]["composite_order_sha256"]
    assert report["report_fingerprint"]


def test_file_fidelity_keeps_ordinary_missing_provenance_legacy_compatible(tmp_path: Path) -> None:
    sample = _write(tmp_path / "sample_submission.csv", "id,target\n1,0\n2,0\n3,0\n")
    submission = _write(tmp_path / "submission.csv", "id,target\n1,0.1\n2,0.7\n3,0.2\n")

    report = _validate(tmp_path, submission=submission, sample=sample, metrics=None)

    assert report["verdict"] == "pass"
    assert report["warning_codes"] == ["legacy_unknown"]
    assert report["metric_provenance"]["trusted"] is False


def test_file_fidelity_rejects_ordered_identifier_mismatch(tmp_path: Path) -> None:
    sample = _write(tmp_path / "sample_submission.csv", "id,target\n1,0\n2,0\n3,0\n")
    submission = _write(tmp_path / "submission.csv", "id,target\n2,0.7\n1,0.1\n3,0.2\n")

    with pytest.raises(SubmissionValidationError) as exc_info:
        _validate(tmp_path, submission=submission, sample=sample, metrics=_metrics(tmp_path / "metrics.json"))

    assert "file_identifier_order_mismatch" in exc_info.value.reason_codes
    assert exc_info.value.fidelity_repair_required is True


@pytest.mark.parametrize(
    ("submission_text", "metrics_updates", "expected_code"),
    [
        ("id,target\n1,0.5\n2,0.5\n3,0.5\n", {}, "tabular_prediction_dispersion_collapsed"),
        (
            "id,target\n1,0.1\n2,0.7\n3,0.2\n",
            {"fallback_submission": True},
            "prediction_fallback_used",
        ),
    ],
)
def test_file_fidelity_rejects_constant_or_fallback_predictions(
    tmp_path: Path,
    submission_text: str,
    metrics_updates: dict[str, object],
    expected_code: str,
) -> None:
    sample = _write(tmp_path / "sample_submission.csv", "id,target\n1,0\n2,0\n3,0\n")
    submission = _write(tmp_path / "submission.csv", submission_text)

    with pytest.raises(SubmissionValidationError) as exc_info:
        _validate(
            tmp_path,
            submission=submission,
            sample=sample,
            metrics=_metrics(tmp_path / "metrics.json", **metrics_updates),
        )

    assert expected_code in exc_info.value.reason_codes


def test_quarantine_requires_changed_code_and_output_and_reserves_one_attempt(tmp_path: Path) -> None:
    sample = _write(tmp_path / "sample_submission.csv", "id,target\n1,0\n2,0\n3,0\n")
    old_submission = _write(tmp_path / "old.csv", "id,target\n1,0.2\n2,0.3\n3,0.4\n")
    old_output = str(sha256_file_or_none(old_submission))
    old_attempt = sha256_text("\0".join(("code-v1", old_output)))
    quarantine = {
        "status": "active",
        "anomaly": {"output_sha256": old_output, "package_fingerprint": "code-v1"},
        "failed_attempt_fingerprints": [old_attempt],
    }

    with pytest.raises(SubmissionValidationError) as exc_info:
        _validate(
            tmp_path / "unchanged",
            submission=old_submission,
            sample=sample,
            metrics=_metrics(
                tmp_path / "unchanged" / "metrics.json",
                submission_sha256=sha256_file_or_none(old_submission),
            ),
            code_fingerprint="code-v1",
            quarantine_state=quarantine,
        )
    assert "quarantine_output_fingerprint_unchanged" in exc_info.value.reason_codes
    assert "quarantine_package_fingerprint_unchanged" in exc_info.value.reason_codes
    assert "quarantine_failed_attempt_unchanged" in exc_info.value.reason_codes

    repaired = _write(tmp_path / "repaired.csv", "id,target\n1,0.9\n2,0.2\n3,0.6\n")
    report = _validate(
        tmp_path / "repaired",
        submission=repaired,
        sample=sample,
        metrics=_metrics(
            tmp_path / "repaired" / "metrics.json",
            submission_sha256=sha256_file_or_none(repaired),
        ),
        code_fingerprint="code-v2",
        quarantine_state=quarantine,
    )
    assert report["verdict"] == "pass"
    assert report["quarantine"]["repair_permit"] == "granted"

    saved: dict[str, object] = {}
    ledger = tmp_path / "submission_ledger.jsonl"
    assert reserve_quarantine_repair_attempt(
        report=report,
        quarantine_state=quarantine,
        save_run_state=saved.update,
        submission_ledger_path=ledger,
        slug="demo",
        run_id="run-2",
    )
    reserved = saved[QUARANTINE_STATE_KEY]
    assert reserved["pending_repair_attempt"]["attempt_fingerprint"] == report["attempt_fingerprint"]
    repeated = enforce_submission_fidelity_quarantine(report=report, quarantine_state=reserved)
    assert repeated["verdict"] == "fail"
    assert "quarantine_repair_permit_already_used" in repeated["reason_codes"]
    reloaded = load_active_submission_fidelity_quarantine(
        run_state={},
        submission_ledger_path=ledger,
        slug="demo",
    )
    assert reloaded["pending_repair_attempt"]["attempt_fingerprint"] == report["attempt_fingerprint"]


@pytest.mark.parametrize(
    ("metrics_updates", "expected_code"),
    [
        ({"authoritative": False}, "metric_provenance_untrusted"),
        ({}, "metric_artifact_binding_missing"),
    ],
)
def test_quarantine_fails_closed_on_untrusted_or_unbound_score_evidence(
    tmp_path: Path,
    metrics_updates: dict[str, object],
    expected_code: str,
) -> None:
    sample = _write(tmp_path / "sample_submission.csv", "id,target\n1,0\n2,0\n3,0\n")
    old = _write(tmp_path / "old.csv", "id,target\n1,0.2\n2,0.3\n3,0.4\n")
    repaired = _write(tmp_path / "repaired.csv", "id,target\n1,0.9\n2,0.2\n3,0.6\n")
    quarantine = {
        "status": "active",
        "anomaly": {
            "output_sha256": sha256_file_or_none(old),
            "package_fingerprint": "code-v1",
        },
        "failed_attempt_fingerprints": [],
    }
    if expected_code != "metric_artifact_binding_missing":
        metrics_updates["submission_sha256"] = sha256_file_or_none(repaired)

    with pytest.raises(SubmissionValidationError) as exc_info:
        _validate(
            tmp_path,
            submission=repaired,
            sample=sample,
            metrics=_metrics(tmp_path / "metrics.json", **metrics_updates),
            code_fingerprint="code-v2",
            quarantine_state=quarantine,
        )

    assert expected_code in exc_info.value.reason_codes


def test_quarantine_persists_anomaly_then_resolves_only_after_good_outcome(tmp_path: Path) -> None:
    ledger = tmp_path / "submission_ledger.jsonl"
    state: dict[str, object] = {}
    anomalous_fidelity = {
        "verdict": "pass",
        "attempt_fingerprint": "attempt-v1",
        "report_fingerprint": "report-v1",
        "package_fingerprint": "code-v1",
        "actual_hashes": {"output_sha256": "output-v1"},
        "metric_provenance": {"trusted": True},
        "reason_codes": [],
        "supporting_artifact_paths": [str(tmp_path / "report-v1.json")],
    }

    action = persist_leaderboard_outcome_quarantine(
        slug="demo",
        run_id="run-1",
        run_state=state,
        latest_submit_attempt={"ok": True, "sub_sha256": "output-v1", "submission_fidelity": anomalous_fidelity},
        anomaly={"signals": ["observed_bottom_two_percent"]},
        submission_ledger_path=ledger,
        save_run_state=state.update,
    )

    assert action == "activated"
    quarantine = state[QUARANTINE_STATE_KEY]
    assert quarantine["status"] == "active"
    assert quarantine["anomaly"]["output_sha256"] == "output-v1"
    assert quarantine["failed_attempt_fingerprints"] == ["attempt-v1"]
    cross_run = load_active_submission_fidelity_quarantine(
        run_state={},
        submission_ledger_path=ledger,
        slug="demo",
    )
    assert cross_run["anomaly"]["output_sha256"] == "output-v1"

    repaired_fidelity = {
        **anomalous_fidelity,
        "attempt_fingerprint": "attempt-v2",
        "report_fingerprint": "report-v2",
        "package_fingerprint": "code-v2",
        "actual_hashes": {"output_sha256": "output-v2"},
    }
    action = persist_leaderboard_outcome_quarantine(
        slug="demo",
        run_id="run-1",
        run_state=state,
        latest_submit_attempt={"ok": True, "sub_sha256": "output-v2", "submission_fidelity": repaired_fidelity},
        anomaly=None,
        submission_ledger_path=ledger,
        save_run_state=state.update,
    )

    assert action == "resolved"
    assert state[QUARANTINE_STATE_KEY]["status"] == "resolved"
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [event["action"] for event in events] == ["activated", "resolved"]
