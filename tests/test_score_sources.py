from __future__ import annotations

from kagglebot.score_sources import (
    is_trusted_offline_score_source,
    normalize_score_source_list,
    normalize_score_source_name,
)


def test_normalize_score_source_name_handles_common_aliases() -> None:
    assert normalize_score_source_name("cross validation") == "cv"
    assert normalize_score_source_name("validation") == "holdout"
    assert normalize_score_source_name("lbproxy") == "lb_proxy"
    assert normalize_score_source_name(None) == "holdout"


def test_is_trusted_offline_score_source_rejects_public_or_proxy_sources() -> None:
    assert is_trusted_offline_score_source("cv") is True
    assert is_trusted_offline_score_source("consensus") is True
    assert is_trusted_offline_score_source("public_lb") is False
    assert is_trusted_offline_score_source("sample_smoke") is False


def test_normalize_score_source_list_deduplicates_in_order() -> None:
    assert normalize_score_source_list(["cv", "cross_validation", "validation", "holdout"]) == ["cv", "holdout"]
    assert normalize_score_source_list("cv") == []
