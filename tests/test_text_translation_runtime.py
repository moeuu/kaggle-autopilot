from __future__ import annotations

import pandas as pd

from kagglebot.kernel_runtime.text_translation import (
    build_retrieval_model,
    build_sentence_pairs,
    compute_text_metrics,
    retrieval_candidate_pools,
    select_mbr_candidate,
)


def test_build_sentence_pairs_keeps_grouping_and_alignment() -> None:
    df = pd.DataFrame(
        {
            "doc_id": ["doc-1"],
            "source_text": ["um-ma a-šur 3 GIN qí-bi-ma en-na"],
            "target_text": ["From Ashur: three shekels. Speak to Enna."],
        }
    )

    pairs = build_sentence_pairs(
        df,
        source_col="source_text",
        target_col="target_text",
        group_col="doc_id",
        boundary_tokens=("um-ma", "qí-bi-ma"),
    )

    assert len(pairs) == 2
    assert set(pairs["group_id"]) == {"doc-1"}
    assert pairs.iloc[0]["source"].startswith("um-ma")
    assert "qí-bi-ma" in pairs.iloc[1]["source"]


def test_compute_text_metrics_emits_sentence_and_document_scores() -> None:
    pair_df = pd.DataFrame(
        {
            "group_id": ["doc-1", "doc-1", "doc-2"],
            "sentence_index": [0, 1, 0],
            "target": ["alpha one", "alpha two", "beta one"],
            "doc_target": ["alpha one alpha two", "alpha one alpha two", "beta one"],
        }
    )

    sentence_metric, doc_metric = compute_text_metrics(pair_df, ["alpha one", "alpha two", "beta one"])

    assert sentence_metric["gmean"] > 99.0
    assert doc_metric["gmean"] > 99.0


def test_select_mbr_candidate_prefers_surface_agreement() -> None:
    chosen = select_mbr_candidate(
        [
            "Ashur sent three shekels.",
            "Ashur sent three shekels.",
            "Speak to Enna tomorrow.",
        ]
    )

    assert chosen == "Ashur sent three shekels."


def test_retrieval_candidate_pools_prefers_exact_memory() -> None:
    model = build_retrieval_model(
        ["um-ma a-šur", "qí-bi-ma en-na"],
        ["From Ashur", "Speak to Enna"],
    )

    pools, low_score = retrieval_candidate_pools(
        model,
        ["um-ma a-šur"],
        min_score=0.0,
        max_candidates=2,
    )

    assert low_score == 0
    assert pools[0][0] == ("From Ashur", "retrieval_exact", 1)
