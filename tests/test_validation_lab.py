from __future__ import annotations

import json
from pathlib import Path

from kagglebot.campaign import CampaignCandidate, candidate_registry_path, upsert_candidate
from kagglebot.validation_lab import run_validation_lab, validation_lab_report_path


def test_validation_lab_adopts_profile_with_best_evidence(tmp_path: Path) -> None:
    registry_path = candidate_registry_path(tmp_path / "context")
    for candidate in [
        CampaignCandidate(
            candidate_id="a",
            category="strong_single",
            run_id="run-1",
            iteration=1,
            direction="maximize",
            offline_score=0.70,
            public_score=0.69,
            validation_profile_id="default_cv",
        ),
        CampaignCandidate(
            candidate_id="b",
            category="strong_single",
            run_id="run-1",
            iteration=2,
            direction="maximize",
            offline_score=0.80,
            public_score=0.81,
            validation_profile_id="group_or_proxy_cv",
        ),
        CampaignCandidate(
            candidate_id="c",
            category="feature_variant",
            run_id="run-1",
            iteration=3,
            direction="maximize",
            offline_score=0.82,
            public_score=0.83,
            validation_profile_id="group_or_proxy_cv",
        ),
    ]:
        upsert_candidate(registry_path, candidate)

    report = run_validation_lab(
        context_dir=tmp_path / "context",
        validation_registry={
            "priority": True,
            "active_profile": "default_cv",
            "profiles": [
                {"profile_id": "default_cv", "priority": 0.3},
                {"profile_id": "group_or_proxy_cv", "priority": 0.9},
            ],
        },
        candidate_registry_path=registry_path,
        campaign_state={"offline_online_correlation": 0.1},
        mode="force",
    )

    assert report["status"] == "active"
    assert report["active_profile"] == "group_or_proxy_cv"
    registry = json.loads((tmp_path / "context" / "validation_registry.json").read_text(encoding="utf-8"))
    assert registry["active_profile"] == "group_or_proxy_cv"
    assert validation_lab_report_path(tmp_path / "context").exists()


def test_validation_lab_auto_monitors_when_trust_is_not_problematic(tmp_path: Path) -> None:
    report = run_validation_lab(
        context_dir=tmp_path / "context",
        validation_registry={"priority": False, "active_profile": "default_cv", "profiles": []},
        candidate_registry_path=tmp_path / "context" / "candidate_registry.json",
        campaign_state={"offline_online_correlation": 0.8},
        mode="auto",
    )

    assert report["status"] == "monitoring"
    assert report["reason"] == "validation_trust_not_yet_problematic"
