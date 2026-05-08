# Ranked candidate pipelines

1. **Contextual ByT5 curriculum + cross-checkpoint MBR**
Leak-free features/encodings: Unicode transliteration normalization, sentence-level pseudo pairs mined from `published_texts.csv` + `Sentences_Oare_FirstWord_LinNum.csv`, document-context windows keyed by `text_id`, lexicon-derived terminology coverage, quantity/unit preservation features fit on train folds only.  
Models + key hyperparameters: ByT5 checkpoint A `assiaben/final-byt5`, checkpoint B `artemgoncarov/dpc-byt5-large`, LoRA `r=32`, `alpha=64`, `lr=1.2e-4`, `warmup_ratio=0.08`, `batch_size=2`, `grad_accum=16`, `max_source_len=512`, `max_target_len=384`, `num_beams=8`, beam candidates `4`, sampled temps `0.65/0.85/1.05`, `top_p=0.92`, MBR pool cap `24`.  
Expected runtime/memory: about 6 to 8 hours on one 24 GB GPU for 3 seeds; 18 to 22 GB VRAM during training, 10 to 14 GB during inference.  
Leakage risk: medium if pseudo alignment or retrieval memory are built across fold boundaries; must group by document and fit indexes only on fold-train.  
Fallback if dependency unavailable: if `peft` is unavailable, switch to single-checkpoint inference-only MBR; if `sacrebleu` is unavailable, add it with `uv add sacrebleu`.

2. **Dual-checkpoint public-model MBR**
Leak-free features/encodings: same normalization and conservative regex corrections, but no extra sentence mining.  
Models + key hyperparameters: two frozen ByT5-family checkpoints, `num_beams=8`, `num_beam_candidates=4`, sampled temps `0.6/0.8/1.05`, `top_p=0.92`, `repetition_penalty=1.2`, MBR weights biased toward chrF++.  
Expected runtime/memory: 60 to 120 minutes inference-only, 10 to 14 GB VRAM.  
Leakage risk: low.  
Fallback if dependency unavailable: run a single checkpoint with beam search only.

3. **Retrieval-augmented ByT5 rerank**
Leak-free features/encodings: char-TFIDF and word-TFIDF retrieval memory built only from fold-train gold plus mined sentence pairs, exact-match and near-duplicate flags, lexicon terminology overlap, numeric/fraction preservation bonuses.  
Models + key hyperparameters: single ByT5 checkpoint with candidate generation, rerank using weighted score from MBR + lexical coverage + retrieval agreement; `k=32`, char n-grams `3-9`, `min_df=2`.  
Expected runtime/memory: 2 to 4 hours, 12 to 16 GB RAM for retrieval index plus 10 GB VRAM for inference.  
Leakage risk: medium, because naive retrieval can accidentally memorize held-out translations if fold partitioning is wrong.  
Fallback if dependency unavailable: use plain `sklearn` TF-IDF retrieval and skip rerank features that need extra tooling.

4. **Char-TFIDF kNN memory baseline**
Leak-free features/encodings: normalized transliteration strings only, fit TF-IDF on fold-train and apply to fold-val/test.  
Models + key hyperparameters: `k=16`, cosine similarity threshold `0.10`, char n-grams `3-9`.  
Expected runtime/memory: minutes, low memory.  
Leakage risk: low if grouped correctly.  
Fallback if dependency unavailable: exact-match lookup only.  
This is not a medal path; it is a calibration and ablation baseline.
