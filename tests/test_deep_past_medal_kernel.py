from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.competition_artifact


def _load_kernel_module(*, env: dict[str, str] | None = None):
    kernel_path = (
        Path(__file__).resolve().parent.parent
        / "artifacts"
        / "deep-past-initiative-machine-translation"
        / "kernel"
        / "kernel.py"
    )
    if env:
        env_key = "|".join(f"{key}={value}" for key, value in sorted(env.items()))
        suffix = hashlib.sha1(env_key.encode("utf-8")).hexdigest()[:8]
        module_name = f"deep_past_medal_kernel_test_local_{suffix}"
    else:
        module_name = "deep_past_medal_kernel_test"
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(kernel_path))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(kernel_path)
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = module
    source = kernel_path.read_text(encoding="utf-8")
    if not env:
        exec(compile(source, str(kernel_path), "exec"), module.__dict__)
        return module

    previous_env: dict[str, str | None] = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        exec(compile(source, str(kernel_path), "exec"), module.__dict__)
        return module
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_fake_checkpoint(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (path / filename).write_text("x", encoding="utf-8")
    return str(path)


def test_build_pseudo_sentence_pairs_is_stable_and_keeps_parent_ids() -> None:
    mod = _load_kernel_module()
    lexicon = mod.LexiconResources(token_map={"ki-ma": "kima"})
    train_df = pd.DataFrame(
        {
            "oare_id": ["doc-a", "doc-b"],
            "transliteration": [
                "um-ma a-na be-li-im ki-ma qí-bi-ma 1 ma-na kaspam i-šu",
                "x x x 3 GÍN KÙ.BABBAR",
            ],
            "translation": [
                "Thus, speak to my lord. He owes one mina of silver.",
                "<gap> three shekels of silver",
            ],
        }
    )

    first = mod.build_pseudo_sentence_pairs(train_df, lexicon)
    second = mod.build_pseudo_sentence_pairs(train_df, lexicon)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["oare_id"]) == {"doc-a", "doc-b"}
    assert first["oare_id"].value_counts().loc["doc-a"] >= 2
    assert "transliteration_lex" in first.columns


def test_split_source_for_target_sentences_uses_monotonic_alignment() -> None:
    mod = _load_kernel_module()
    lexicon = mod.LexiconResources(token_map={})
    source = "um-ma a-šur 3 GÍN KÙ.BABBAR qí-bi-ma en-na"
    targets = ["From Ashur: three shekels of silver.", "Speak to Enna."]

    segments = mod.split_source_for_target_sentences(source, targets, lexicon)

    assert len(segments) == 2
    assert segments[0].startswith("um-ma")
    assert "qí-bi-ma" in segments[1]


def test_build_pseudo_sentence_pairs_degrades_safely_on_noisy_text() -> None:
    mod = _load_kernel_module()
    lexicon = mod.LexiconResources(token_map={})
    train_df = pd.DataFrame(
        {
            "oare_id": ["doc-noisy"],
            "transliteration": ["<gap> x x x x x"],
            "translation": ["broken text without clear punctuation"],
        }
    )

    pairs = mod.build_pseudo_sentence_pairs(train_df, lexicon)

    assert len(pairs) == 1
    assert pairs.iloc[0]["oare_id"] == "doc-noisy"
    assert pairs.iloc[0]["translation"] == "broken text without clear punctuation"


def test_iter_grouped_cv_splits_keeps_groups_isolated() -> None:
    mod = _load_kernel_module()
    groups = ["a", "a", "b", "b", "c", "c"]

    splits = list(mod.iter_grouped_cv_splits(groups, n_folds=3, seed=42, fast_dev=False))

    assert splits
    for tr_idx, va_idx in splits:
        train_groups = {groups[i] for i in tr_idx.tolist()}
        val_groups = {groups[i] for i in va_idx.tolist()}
        assert train_groups.isdisjoint(val_groups)


def test_compute_sentence_and_document_metrics_emits_both_levels() -> None:
    mod = _load_kernel_module()
    pair_df = pd.DataFrame(
        {
            "oare_id": ["doc-a", "doc-a", "doc-b"],
            "sentence_index": [0, 1, 0],
            "translation": [
                "this is one complete sentence",
                "this is another complete sentence",
                "this is the third complete sentence",
            ],
            "doc_translation": [
                "this is one complete sentence this is another complete sentence",
                "this is one complete sentence this is another complete sentence",
                "this is the third complete sentence",
            ],
        }
    )
    preds = [
        "this is one complete sentence",
        "this is another complete sentence",
        "this is the third complete sentence",
    ]

    sentence_metric, doc_metric = mod.compute_sentence_and_document_metrics(pair_df, preds)

    assert sentence_metric["gmean"] > 99.0
    assert doc_metric["gmean"] > 99.0


def test_apply_consistency_postprocess_uses_memory_and_group_consistency() -> None:
    mod = _load_kernel_module()
    source_texts = ["same src", "same src", "other src"]
    predictions = ["", "long canonical answer", "ok"]
    group_values = ["doc-1", "doc-1", "doc-2"]
    exact_memory = {"same src": "memory translation"}

    outputs, stats = mod.apply_consistency_postprocess(source_texts, predictions, group_values, exact_memory)

    assert outputs[0] == "memory translation"
    assert outputs[1] == "memory translation"
    assert stats["memory_rewrites"] >= 1
    assert stats["consistency_rewrites"] >= 1


def test_run_translation_seq2seq_aborts_degenerate_inference_only_lookup_submit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_kernel_module(env={"KAGGLEBOT_DO_TRAIN": "0", "KAGGLEBOT_DO_INFER": "1"})

    train_df = pd.DataFrame(
        {
            "oare_id": ["doc-1", "doc-2"],
            "transliteration": ["src-1", "src-2"],
            "translation": ["gold-1", "gold-2"],
        }
    )
    test_df = pd.DataFrame({"id": [0, 1], "transliteration": ["unseen-a", "unseen-b"]})
    sample_df = pd.DataFrame({"id": [0, 1], "translation": ["", ""]})
    pseudo_train_df = pd.DataFrame(
        {
            "oare_id": ["doc-1", "doc-2"],
            "transliteration": ["src-1", "src-2"],
            "transliteration_lex": ["src-1", "src-2"],
            "translation": ['"', '"'],
            "supervision_source": ["heuristic", "heuristic"],
            "pair_weight": [1.0, 1.0],
        }
    )
    baseline = mod.PipelineResult(
        name="lookup_baseline",
        cv_score=0.0,
        bleu=0.0,
        chrfpp=0.0,
        doc_score=0.0,
        doc_bleu=0.0,
        doc_chrfpp=0.0,
        unseen_sentence_score=0.0,
        unseen_sentence_bleu=0.0,
        unseen_sentence_chrfpp=0.0,
        unseen_document_score=0.0,
        unseen_document_bleu=0.0,
        unseen_document_chrfpp=0.0,
        complexity_rank=0,
        oof_predictions=np.zeros((1, len(pseudo_train_df)), dtype=object),
        test_predictions=np.zeros((1, len(test_df)), dtype=object),
        best_seed=42,
    )

    monkeypatch.setattr(mod, "_force_translation_metric_for_slug", lambda _slug: False)
    monkeypatch.setattr(mod, "preprocess_translation_df", lambda df, *_args: df.copy())
    monkeypatch.setattr(mod, "load_lexicon_resources", lambda _data_dir: mod.LexiconResources(token_map={}))
    monkeypatch.setattr(
        mod,
        "load_optional_metadata_frames",
        lambda _data_dir: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        mod,
        "build_merged_sentence_pairs",
        lambda *_args, **_kwargs: (
            pseudo_train_df.copy(),
            mod.MetadataSupervisionResult(pd.DataFrame(), 0, 0, 0),
        ),
    )
    monkeypatch.setattr(mod, "build_constraint_memories", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "build_exact_source_memory", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, "run_lookup_baseline_cv", lambda *_args, **_kwargs: baseline)
    monkeypatch.setattr(mod, "shortlisted_pipeline_names", lambda: [])
    monkeypatch.setattr(mod, "build_explicit_ensemble_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "summarize_results", lambda _results: pd.DataFrame({"pipeline": ["lookup_baseline"]}))
    monkeypatch.setattr(mod, "choose_best_result", lambda results: results[0])
    monkeypatch.setattr(mod, "_write_csv_all", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="degenerate lookup_baseline predictions"):
        mod.run_translation_seq2seq(
            train_df=train_df,
            test_df=test_df,
            sample_df=sample_df,
            data_dir=tmp_path,
            output_dirs=[tmp_path],
            kaggle_working_writable=False,
        )


def test_run_translation_seq2seq_submit_notebook_skips_cv_and_runs_direct_reference_inference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_kernel_module(
        env={
            "KAGGLEBOT_DO_TRAIN": "0",
            "KAGGLEBOT_DO_INFER": "1",
            "KAGGLEBOT_SUBMIT_NOTEBOOK": "1",
            "KAGGLEBOT_SUBMIT_SKIP_CV": "1",
        }
    )

    train_df = pd.DataFrame(
        {
            "oare_id": ["doc-1", "doc-2"],
            "transliteration": ["src-1", "src-2"],
            "translation": ["gold-1", "gold-2"],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [0, 1],
            "text_id": ["txt-1", "txt-2"],
            "transliteration": ["test-a", "test-b"],
        }
    )
    sample_df = pd.DataFrame({"id": [0, 1], "translation": ["", ""]})
    captured: dict[str, object] = {}

    submit_cfg = mod.PipelineConfig(
        name=mod.REFERENCE_PRIMARY_PIPELINE_NAME,
        model_hints=["model-a", "model-b"],
        max_source_len=512,
        max_new_tokens=384,
        num_beams=8,
        length_penalty=1.3,
        repetition_penalty=1.2,
        mbr_num_beam_cands=4,
        mbr_num_sample_cands=2,
        sample_temperatures=[0.6, 0.8, 1.05],
        mbr_top_p=0.92,
        mbr_pool_cap=32,
        use_mbr=True,
        use_multi_model_pool=True,
        use_lora=False,
        use_retrieval_candidates=False,
        use_context_window=False,
        allow_domain_adapted=False,
        strong_postprocess=True,
        complexity_rank=10,
        runtime_name="dual_checkpoint_public_mbr__exact_public_pair",
        reference_runtime_mode="exact_required_public_pair",
    )

    monkeypatch.setattr(mod, "preprocess_translation_df", lambda df, *_args: df.copy())
    monkeypatch.setattr(mod, "load_lexicon_resources", lambda _data_dir: mod.LexiconResources(token_map={}))
    monkeypatch.setattr(
        mod,
        "load_optional_metadata_frames",
        lambda _data_dir: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(mod, "build_constraint_memories", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "build_exact_source_memory", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        mod,
        "run_lookup_baseline_cv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("submit notebook path should skip CV")),
    )
    monkeypatch.setattr(mod, "get_pipeline_cfg", lambda _name: submit_cfg)
    monkeypatch.setattr(mod, "_prepare_reference_baseline_cfg", lambda cfg: cfg)

    def _fake_train_full_and_predict(*args, **kwargs):
        captured["train_full_and_predict"] = {"args": args, "kwargs": kwargs}
        return (
            ["pred-a", "pred-b"],
            "model-a,model-b",
            {"consistency_rewrites": 0},
            ["/kaggle/input/model-a", "/kaggle/input/model-b"],
        )

    monkeypatch.setattr(mod, "train_full_and_predict", _fake_train_full_and_predict)
    monkeypatch.setattr(
        mod,
        "write_submission",
        lambda df, *_args, **_kwargs: captured.setdefault("submission_df", df.copy()),
    )
    monkeypatch.setattr(
        mod,
        "_write_json_all",
        lambda _name, payload, _dirs: captured.setdefault("metrics", payload),
    )

    mod.run_translation_seq2seq(
        train_df=train_df,
        test_df=test_df,
        sample_df=sample_df,
        data_dir=tmp_path,
        output_dirs=[tmp_path],
        kaggle_working_writable=False,
    )

    train_call = captured["train_full_and_predict"]
    assert train_call["args"][2].name == mod.REFERENCE_PRIMARY_PIPELINE_NAME
    submission_df = captured["submission_df"]
    assert submission_df["translation"].tolist() == ["pred-a", "pred-b"]
    metrics = captured["metrics"]
    assert metrics["score_source"] == "submit_inference_only"
    assert metrics["config"]["submit_notebook_mode"] is True
    assert metrics["config"]["submit_skip_cv"] is True
    assert metrics["selected"]["name"] == mod.REFERENCE_PRIMARY_PIPELINE_NAME


def test_choose_best_result_prefers_seq2seq_path_over_retrieval_near_tie() -> None:
    mod = _load_kernel_module()
    zero_arr = np.zeros((1, 1), dtype=object)
    retrieval = mod.PipelineResult(
        name="retrieval_char_tfidf_knn",
        cv_score=35.00,
        bleu=35.0,
        chrfpp=35.0,
        doc_score=34.9,
        doc_bleu=34.9,
        doc_chrfpp=34.9,
        complexity_rank=1,
        oof_predictions=zero_arr,
        test_predictions=zero_arr,
        best_seed=42,
    )
    seq2seq = mod.PipelineResult(
        name="pooled_multi_byt5_mbr",
        cv_score=34.97,
        bleu=35.0,
        chrfpp=34.9,
        doc_score=35.1,
        doc_bleu=35.1,
        doc_chrfpp=35.1,
        complexity_rank=10,
        oof_predictions=zero_arr,
        test_predictions=zero_arr,
        best_seed=42,
    )

    chosen = mod.choose_best_result([retrieval, seq2seq])

    assert chosen.name == "pooled_multi_byt5_mbr"


def test_choose_best_result_keeps_stronger_retrieval_baseline_over_weaker_seq2seq() -> None:
    mod = _load_kernel_module()
    zero_arr = np.zeros((1, 1), dtype=object)
    retrieval = mod.PipelineResult(
        name="char_tfidf_knn_memory",
        cv_score=35.50,
        bleu=35.5,
        chrfpp=35.5,
        doc_score=35.4,
        doc_bleu=35.4,
        doc_chrfpp=35.4,
        complexity_rank=1,
        oof_predictions=zero_arr,
        test_predictions=zero_arr,
        best_seed=42,
    )
    seq2seq = mod.PipelineResult(
        name="dual_checkpoint_public_mbr",
        cv_score=34.80,
        bleu=34.8,
        chrfpp=34.8,
        doc_score=35.2,
        doc_bleu=35.2,
        doc_chrfpp=35.2,
        complexity_rank=10,
        oof_predictions=zero_arr,
        test_predictions=zero_arr,
        best_seed=42,
        executed_checkpoints=["/tmp/model-a", "/tmp/model-b"],
    )

    chosen = mod.choose_best_result([retrieval, seq2seq])

    assert chosen.name == "char_tfidf_knn_memory"


def test_choose_best_result_keeps_stronger_plan_blend_over_weaker_seq2seq() -> None:
    mod = _load_kernel_module()
    zero_arr = np.zeros((1, 1), dtype=object)
    blended = mod.PipelineResult(
        name="plan_mbr_blend",
        cv_score=35.20,
        bleu=35.2,
        chrfpp=35.2,
        doc_score=35.1,
        doc_bleu=35.1,
        doc_chrfpp=35.1,
        complexity_rank=2,
        oof_predictions=zero_arr,
        test_predictions=zero_arr,
        best_seed=42,
        ensemble_members=["char_tfidf_knn_memory", "retrieval_augmented_byt5_rerank"],
    )
    seq2seq = mod.PipelineResult(
        name="dual_checkpoint_public_mbr",
        cv_score=34.60,
        bleu=34.6,
        chrfpp=34.6,
        doc_score=34.8,
        doc_bleu=34.8,
        doc_chrfpp=34.8,
        complexity_rank=10,
        oof_predictions=zero_arr,
        test_predictions=zero_arr,
        best_seed=42,
        executed_checkpoints=["/tmp/model-a", "/tmp/model-b"],
    )

    chosen = mod.choose_best_result([blended, seq2seq])

    assert chosen.name == "plan_mbr_blend"


def test_deep_past_kernel_forces_competition_metric_and_reference_shortlist_order() -> None:
    mod = _load_kernel_module()

    assert mod.REPORTED_PRIMARY_METRIC == "Geometric Mean of the BLEU and the chrF++ scores"
    assert mod.shortlisted_pipeline_names()[0] == "dual_checkpoint_public_mbr"


def test_deep_past_kernel_keeps_reference_primary_in_shortlist_when_pipeline_1_toggle_is_disabled() -> None:
    mod = _load_kernel_module(env={"KAGGLEBOT_ENABLE_PIPELINE_1": "0"})

    shortlist = mod.shortlisted_pipeline_names()

    assert shortlist[0] == "dual_checkpoint_public_mbr"
    assert "dual_checkpoint_public_mbr" in shortlist


def test_deep_past_kernel_runs_full_seq2seq_shortlist_by_default() -> None:
    mod = _load_kernel_module()

    active = mod._active_plan_seq2seq_pipeline_names()

    assert active == {
        "contextual_byt5_curriculum_mbr",
        "dual_checkpoint_public_mbr",
        "retrieval_augmented_byt5_rerank",
    }


def test_deep_past_reference_only_mode_is_explicit_opt_in() -> None:
    mod = _load_kernel_module(env={"KAGGLEBOT_REFERENCE_MODE_ONLY": "1"})

    active = mod._active_plan_seq2seq_pipeline_names()

    assert active == {"dual_checkpoint_public_mbr"}


def test_resolve_artifact_dir_prefers_parent_for_source_tree_and_kernel_dir_for_submit_snapshot(tmp_path: Path) -> None:
    mod = _load_kernel_module()

    source_kernel_dir = tmp_path / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "plan.json").write_text("{}", encoding="utf-8")

    submit_kernel_dir = tmp_path / "submit-iter-1"
    submit_kernel_dir.mkdir(parents=True, exist_ok=True)
    (submit_kernel_dir / "plan.json").write_text("{}", encoding="utf-8")

    assert mod._resolve_artifact_dir(source_kernel_dir) == tmp_path
    assert mod._resolve_artifact_dir(submit_kernel_dir) == submit_kernel_dir


def test_deep_past_kernel_uses_faithful_default_shortlist_when_plan_pipelines_are_missing() -> None:
    mod = _load_kernel_module()
    original = mod.PLAN_PIPELINES
    try:
        mod.PLAN_PIPELINES = []
        shortlist = mod.shortlisted_pipeline_names()
    finally:
        mod.PLAN_PIPELINES = original

    assert shortlist[:4] == mod.FAITHFUL_TRANSLATION_DEFAULT_SHORTLIST


def test_local_reference_eval_frame_and_reduced_cfg_clamp_watchdog_budget() -> None:
    mod = _load_kernel_module()
    train_df = pd.DataFrame(
        {
            "oare_id": [f"doc-{idx:03d}" for idx in range(150)],
            "transliteration": [f"um-ma source {idx}" for idx in range(150)],
            "translation": [f"target text {idx}" for idx in range(150)],
        }
    )

    reduced_eval_df = mod._build_local_seq2seq_eval_frame(
        train_df,
        max_docs=mod.LOCAL_REFERENCE_FAST_EVAL_DOCS,
    )
    runtime_cfg = mod.PipelineConfig(
        **{
            **mod.get_pipeline_cfg("dual_checkpoint_public_mbr").__dict__,
            "model_hints": ["/tmp/model-a", "/tmp/model-b"],
            "use_multi_model_pool": True,
            "use_retrieval_candidates": True,
        }
    )
    reduced_cfg = mod._reduced_faithful_eval_cfg(runtime_cfg)

    assert reduced_eval_df["oare_id"].nunique() == min(len(train_df), mod.LOCAL_REFERENCE_FAST_EVAL_DOCS)
    assert reduced_cfg.use_multi_model_pool is True
    assert reduced_cfg.use_mbr is True
    assert reduced_cfg.num_beams == mod.LOCAL_REFERENCE_NUM_BEAMS
    assert reduced_cfg.mbr_num_beam_cands == mod.LOCAL_REFERENCE_NUM_BEAM_CANDIDATES
    assert reduced_cfg.mbr_num_sample_cands == mod.LOCAL_REFERENCE_NUM_SAMPLE_CANDIDATES
    assert reduced_cfg.mbr_pool_cap == mod.LOCAL_REFERENCE_MBR_POOL_CAP
    assert reduced_cfg.max_new_tokens == mod.LOCAL_REFERENCE_MAX_NEW_TOKENS
    assert reduced_cfg.use_retrieval_candidates is False


def test_local_kernel_mode_uses_watchdog_safe_reference_profile() -> None:
    mod = _load_kernel_module(env={"KAGGLEBOT_LOCAL_KERNEL": "1"})
    runtime_cfg = mod.PipelineConfig(
        **{
            **mod.get_pipeline_cfg("dual_checkpoint_public_mbr").__dict__,
            "model_hints": ["/tmp/model-a", "/tmp/model-b"],
            "use_multi_model_pool": True,
            "use_retrieval_candidates": True,
            "use_mbr": True,
            "mbr_num_beam_cands": 4,
            "mbr_num_sample_cands": 2,
            "sample_temperatures": [0.6, 0.8, 1.05],
        }
    )

    reduced_cfg = mod._reduced_faithful_eval_cfg(runtime_cfg)

    assert mod.LOCAL_KERNEL_MODE is True
    assert reduced_cfg.use_multi_model_pool is True
    assert reduced_cfg.use_mbr is False
    assert reduced_cfg.num_beams == mod.LOCAL_REFERENCE_WATCHDOG_NUM_BEAMS
    assert reduced_cfg.mbr_num_beam_cands == 1
    assert reduced_cfg.mbr_num_sample_cands == 0
    assert reduced_cfg.sample_temperatures == []
    assert reduced_cfg.max_new_tokens == mod.LOCAL_REFERENCE_WATCHDOG_MAX_NEW_TOKENS
    assert reduced_cfg.mbr_pool_cap == mod.LOCAL_REFERENCE_MBR_POOL_CAP
    assert reduced_cfg.use_retrieval_candidates is False


def test_deep_past_kernel_enables_lora_finetune_by_default() -> None:
    mod = _load_kernel_module()

    assert mod.USE_LORA_FINETUNE is True


def test_build_metadata_sentence_pairs_uses_high_precision_alias_join() -> None:
    mod = _load_kernel_module()
    lexicon = mod.LexiconResources(token_map={})
    train_df = pd.DataFrame(
        {
            "oare_id": ["doc-1"],
            "transliteration": ["um-ma a-šur 3 GÍN KÙ.BABBAR qí-bi-ma en-na"],
            "translation": ["From Ashur: three shekels of silver. Speak to Enna."],
        }
    )
    published_df = pd.DataFrame(
        {
            "oare_id": ["doc-1"],
            "aliases": ["ICK 1 146"],
            "label": ["Cuneiform Tablet ICK 1 146"],
            "cdli_id": ["P123456"],
        }
    )
    sentence_df = pd.DataFrame(
        {
            "display_name": ["(ICK 1 146)", "(ICK 1 146)"],
            "translation": ["From Ashur: three shekels of silver.", "Speak to Enna."],
            "first_word_transcription": [np.nan, "qí-bi-ma"],
            "first_word_spelling": ["um-ma", "qí-bi-ma"],
            "line_number": [1, 2],
            "side": [1, 1],
            "column": [1, 1],
            "sentence_obj_in_text": [1, 2],
        }
    )

    result = mod.build_metadata_sentence_pairs(train_df, published_df, sentence_df, lexicon)

    assert result.candidate_docs == 1
    assert result.matched_docs == 1
    assert len(result.pair_df) == 2
    assert set(result.pair_df["supervision_source"]) == {"sentence_metadata"}
    assert set(result.pair_df["pair_weight"]) == {1.5}
    assert result.pair_df.iloc[0]["transliteration"].startswith("um-ma")
    assert "qí-bi-ma" in result.pair_df.iloc[1]["transliteration"]


def test_build_metadata_sentence_pairs_rejects_ambiguous_display_clusters() -> None:
    mod = _load_kernel_module()
    lexicon = mod.LexiconResources(token_map={})
    train_df = pd.DataFrame(
        {
            "oare_id": ["doc-1"],
            "transliteration": ["um-ma a-šur qí-bi-ma en-na"],
            "translation": ["From Ashur. Speak to Enna."],
        }
    )
    published_df = pd.DataFrame(
        {
            "oare_id": ["doc-1"],
            "aliases": ["ICK 1 146"],
            "label": ["Cuneiform Tablet ICK 1 146"],
            "cdli_id": ["P123456"],
        }
    )
    sentence_df = pd.DataFrame(
        {
            "display_name": ["(ICK 1 146)", "(env ICK 1 146)"],
            "translation": ["From Ashur.", "Speak to Enna."],
            "first_word_transcription": [np.nan, "qí-bi-ma"],
            "first_word_spelling": ["um-ma", "qí-bi-ma"],
            "line_number": [1, 2],
            "side": [1, 1],
            "column": [1, 1],
            "sentence_obj_in_text": [1, 2],
        }
    )

    result = mod.build_metadata_sentence_pairs(train_df, published_df, sentence_df, lexicon)

    assert result.candidate_docs == 1
    assert result.matched_docs == 0
    assert result.rejected_docs == 1
    assert result.pair_df.empty


def test_build_merged_sentence_pairs_keeps_metadata_weights() -> None:
    mod = _load_kernel_module()
    lexicon = mod.LexiconResources(token_map={})
    train_df = pd.DataFrame(
        {
            "oare_id": ["doc-1"],
            "transliteration": ["um-ma a-šur 3 GÍN KÙ.BABBAR qí-bi-ma en-na"],
            "translation": ["From Ashur: three shekels of silver. Speak to Enna."],
        }
    )
    published_df = pd.DataFrame(
        {
            "oare_id": ["doc-1"],
            "aliases": ["ICK 1 146"],
            "label": ["Cuneiform Tablet ICK 1 146"],
            "cdli_id": ["P123456"],
        }
    )
    sentence_df = pd.DataFrame(
        {
            "display_name": ["(ICK 1 146)", "(ICK 1 146)"],
            "translation": ["From Ashur: three shekels of silver.", "Speak to Enna."],
            "first_word_transcription": [np.nan, "qí-bi-ma"],
            "first_word_spelling": ["um-ma", "qí-bi-ma"],
            "line_number": [1, 2],
            "side": [1, 1],
            "column": [1, 1],
            "sentence_obj_in_text": [1, 2],
        }
    )

    merged, metadata = mod.build_merged_sentence_pairs(train_df, lexicon, published_df, sentence_df)

    assert metadata.matched_docs == 1
    assert set(merged.loc[merged["supervision_source"] == "sentence_metadata", "pair_weight"]) == {1.5}
    assert not merged.duplicated(subset=["oare_id", "transliteration_lex", "translation"]).any()
    assert "heuristic" not in set(merged["supervision_source"])


def test_extract_monotonic_metadata_spans_rejects_non_monotonic_anchor() -> None:
    mod = _load_kernel_module()
    lexicon = mod.LexiconResources(token_map={})
    sentence_rows = pd.DataFrame(
        {
            "translation": ["row-1", "row-2"],
            "first_word_transcription": ["qí-bi-ma", np.nan],
            "first_word_spelling": ["qí-bi-ma", "um-ma"],
            "line_number": [1, 2],
            "side": [1, 1],
            "column": [1, 1],
            "sentence_obj_in_text": [1, 2],
        }
    )

    spans = mod.extract_monotonic_metadata_spans("um-ma a-šur qí-bi-ma en-na", sentence_rows, lexicon)

    assert spans == []


def test_pipeline_model_source_classes_prefers_domain_adapted_sources(monkeypatch) -> None:
    mod = _load_kernel_module()
    cfg = mod.get_pipeline_cfg("pooled_multi_byt5_mbr")

    monkeypatch.setattr(mod, "_pipeline_domain_adapted_model_hints", lambda _cfg: ["adapted-a"])
    monkeypatch.setattr(mod, "_pipeline_model_hints", lambda _cfg: ["adapted-a", "base-a", "base-b"])
    monkeypatch.setattr(mod, "_iter_local_model_sources", lambda hint: [f"/resolved/{hint}"])

    classes = mod._pipeline_model_source_classes(cfg)

    assert classes["domain_adapted"] == ["/resolved/adapted-a"]
    assert classes["base"] == ["/resolved/base-a", "/resolved/base-b"]


def test_constraint_memories_rewrite_entities_and_units() -> None:
    mod = _load_kernel_module()
    pair_df = pd.DataFrame(
        {
            "transliteration": ["A-šur", "A-šur", "3 GÍN", "3 GÍN"],
            "transliteration_lex": ["A-šur", "A-šur", "3 GÍN", "3 GÍN"],
            "translation": ["Ashur", "Ashur", "three shekels", "three shekels"],
            "doc_translation": ["Ashur", "Ashur", "three shekels", "three shekels"],
        }
    )
    memories = mod.build_constraint_memories(pair_df)
    stats = {"entity_rewrites": 0, "quantity_rewrites": 0, "unit_rewrites": 0, "constraint_bonus_hits": 0}

    entity_out = mod.apply_soft_constraint_rewrites("a-šur", "Asher", memories, stats)
    quantity_out = mod.apply_soft_constraint_rewrites("3 GÍN", "three talents", memories, stats)

    assert entity_out == "Ashur"
    assert quantity_out == "three shekels"
    assert stats["entity_rewrites"] >= 1
    assert (stats["unit_rewrites"] + stats["quantity_rewrites"]) >= 1


def test_candidate_mbr_utility_prefers_surface_match_gmean() -> None:
    mod = _load_kernel_module()

    close_score = mod.candidate_mbr_utility_score("Ashur sent three shekels.", "Ashur sent three shekels.")
    far_score = mod.candidate_mbr_utility_score("Ashur sent three shekels.", "Speak to Enna tomorrow.")

    assert close_score > far_score


def test_retrieval_candidate_pool_prefers_exact_memory_match() -> None:
    mod = _load_kernel_module()
    model = mod._fit_retrieval_model(
        ["um-ma a-šur", "qí-bi-ma en-na"],
        ["From Ashur", "Speak to Enna"],
    )

    pools, low_sim = mod._retrieval_candidate_pools(
        model=model,
        infer_src=["um-ma a-šur"],
        k=2,
        min_sim=0.0,
        max_candidates=2,
    )

    assert low_sim == 0
    assert pools[0][0][0] == "From Ashur"
    assert pools[0][0][1] == "retrieval_exact"


def test_compute_slice_metrics_emits_metadata_quantity_and_entity_keys() -> None:
    mod = _load_kernel_module()
    pair_df = pd.DataFrame(
        {
            "oare_id": ["doc-a", "doc-a", "doc-b"],
            "sentence_index": [0, 1, 0],
            "translation": [
                "this is the Ashur line",
                "this is the three shekels line",
                "this is the ordinary line",
            ],
            "doc_translation": [
                "this is the Ashur line this is the three shekels line",
                "this is the Ashur line this is the three shekels line",
                "this is the ordinary line",
            ],
            "supervision_source": ["sentence_metadata", "sentence_metadata", "heuristic"],
            "has_quantity_or_unit": [False, True, False],
            "has_entity_tokens": [True, False, False],
        }
    )
    preds = [
        "this is the Ashur line",
        "this is the three shekels line",
        "this is the ordinary line",
    ]

    metrics = mod.compute_slice_metrics(pair_df, preds)

    assert metrics["metadata_supervision_sentence_gmean"] > 99.0
    assert metrics["metadata_backed_sentence_gmean"] > 99.0
    assert metrics["metadata_supervision_document_gmean"] > 99.0
    assert metrics["heuristic_only_sentence_gmean"] > 99.0
    assert metrics["quantity_unit_sentence_gmean"] > 99.0
    assert metrics["entity_heavy_sentence_gmean"] > 99.0


def test_run_optional_lora_finetune_skips_cleanly_when_disabled() -> None:
    mod = _load_kernel_module()
    cfg = mod.get_pipeline_cfg("byt5_large_lora_finetune_plus_mbr")
    pair_df = pd.DataFrame(
        {
            "oare_id": ["doc-1", "doc-2"],
            "transliteration": ["um-ma a-šur", "qí-bi-ma en-na"],
            "transliteration_lex": ["um-ma a-šur", "qí-bi-ma en-na"],
            "translation": ["From Ashur", "Speak to Enna"],
            "doc_translation": ["From Ashur", "Speak to Enna"],
        }
    )

    result = mod.run_optional_lora_finetune(pair_df, pair_df.copy(), cfg, seed=42)

    assert result.ran is False
    assert (
        result.reason == "kernel_finetune_disabled"
        or result.reason == "lora_toggle_disabled"
        or result.reason == "pipeline_not_lora_enabled"
        or result.reason.startswith(("finetune_deps_unavailable:", "failed_to_load_base:"))
        or result.reason in {"cuda_unavailable", "no_local_model_sources"}
    )


def test_prepare_reference_baseline_cfg_prefers_strongest_cached_fallback_pair(monkeypatch, tmp_path: Path) -> None:
    mod = _load_kernel_module()
    cfg = mod.get_pipeline_cfg("dual_checkpoint_public_mbr")
    assiaben = _write_fake_checkpoint(tmp_path / "assiaben")
    artem = _write_fake_checkpoint(tmp_path / "artem")

    def fake_candidates(hint: str) -> tuple[list[str], list[str]]:
        mapping = {
            "assiaben/final-byt5": ([assiaben], []),
            "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6": (
                [],
                ["mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6 local asset exists but is empty"],
            ),
            "artemgoncarov/dpc-byt5-large": ([artem], []),
        }
        return mapping[hint]

    monkeypatch.setattr(mod, "_reference_model_candidates", fake_candidates)
    monkeypatch.setattr(mod, "_reference_cached_checkpoint_candidates", lambda: [assiaben, artem])

    resolved = mod._prepare_reference_baseline_cfg(cfg)

    assert resolved.reference_runtime_mode == "competition_faithful_fallback_pair"
    assert resolved.use_multi_model_pool is True
    assert resolved.use_retrieval_candidates is False
    assert resolved.model_hints == [assiaben, artem]
    assert resolved.runtime_name is not None
    assert resolved.reference_slot_meta is not None


def test_prepare_reference_baseline_cfg_blocks_when_no_distinct_second_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_kernel_module()
    cfg = mod.get_pipeline_cfg("dual_checkpoint_public_mbr")
    assiaben = _write_fake_checkpoint(tmp_path / "assiaben")

    def fake_candidates(hint: str) -> tuple[list[str], list[str]]:
        mapping = {
            "assiaben/final-byt5": ([assiaben], []),
            "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6": (
                [],
                ["mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6 local asset exists but is empty"],
            ),
            "artemgoncarov/dpc-byt5-large": ([], ["artem blocked"]),
        }
        return mapping[hint]

    monkeypatch.setattr(mod, "_reference_model_candidates", fake_candidates)
    monkeypatch.setattr(mod, "_reference_cached_checkpoint_candidates", lambda: [assiaben])

    resolved = mod._prepare_reference_baseline_cfg(cfg)

    assert resolved.reference_runtime_mode == "blocked_reference_runtime"
    assert resolved.model_hints == []
    assert resolved.use_multi_model_pool is False
    assert "blocked_reference_runtime" in (resolved.runtime_name or "")


def test_reference_model_candidates_continue_after_empty_exact_asset_and_find_nested_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_kernel_module()
    hint = "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6"
    exact_dir = tmp_path / "dataset__mattiaangeli__byt5-akkadian-mbr__PyTorch__default__6"
    exact_dir.mkdir(parents=True, exist_ok=True)
    source_root = tmp_path / "mirror" / "dataset__mattiaangeli__byt5-akkadian-mbr__PyTorch__default__6"
    nested_dir = _write_fake_checkpoint(source_root / "hf" / "checkpoint")

    monkeypatch.setattr(mod, "REFERENCE_EXACT_MODEL_ASSET_PATHS", {hint: (exact_dir,)})
    monkeypatch.setattr(mod, "_resolve_model_sources", lambda _hint: [source_root])

    candidates, blockers = mod._reference_model_candidates(hint)

    assert candidates == [nested_dir]
    assert any("local asset exists but is empty" in message for message in blockers)


def test_prepare_reference_baseline_cfg_keeps_reference_path_not_retrieval_augmented(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_kernel_module()
    cfg = mod.get_pipeline_cfg("dual_checkpoint_public_mbr")
    assiaben = _write_fake_checkpoint(tmp_path / "assiaben")
    mattia = _write_fake_checkpoint(tmp_path / "mattia")

    def fake_candidates(hint: str) -> tuple[list[str], list[str]]:
        mapping = {
            "assiaben/final-byt5": ([assiaben], []),
            "mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6": ([mattia], []),
            "artemgoncarov/dpc-byt5-large": ([], []),
        }
        return mapping[hint]

    monkeypatch.setattr(mod, "_reference_model_candidates", fake_candidates)

    resolved = mod._prepare_reference_baseline_cfg(cfg)

    assert resolved.reference_runtime_mode == "exact_required_public_pair"
    assert resolved.model_hints == [assiaben, mattia]
    assert resolved.use_multi_model_pool is True
    assert resolved.use_retrieval_candidates is False
