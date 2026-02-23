# research_summary.md (shortlist)

## 1) ByT5-large finetune + chrF-MBR decoding (Rank 1)
- Leak-free features/encodings: deterministic normalization of gaps/ellipses/subscripts; optional determinatives formatting; no test-derived stats.
- Model: `google/byt5-large` seq2seq; AdamW lr ~5e-5, label_smoothing 0.1, grad_checkpointing on, grad_accum 8–16, max_source_len 768–1024, max_target_len 256–512.
- Decoding: beam (6–8) + 2–6 sampled candidates; MBR rerank via chrF/chrF++ similarity across pool; length_penalty ~1.0–1.3.
- Runtime/memory: heavy; full 5-fold CV is multi-hour on 16–24GB GPU; inference moderate (batch 1–8).
- Leakage risk: low if training uses only provided/publicly allowed data and fold-scoped training; retrieval indices (if any) must be fit on fold-train only.
- Fallback: switch to `google/byt5-base`, reduce max lengths, reduce candidate pool for MBR.

## 2) ByT5-base finetune + stronger postprocess + MBR (Rank 2)
- Leak-free features: same normalization; add conservative de-duplication of repeated n-grams and spacing normalization.
- Model: `google/byt5-base`; lr 7e-5–1e-4; more epochs/steps to compensate; same decoding/MPR knobs.
- Runtime/memory: fits smaller GPUs; fastest iteration; good for FAST_DEV + ablation.
- Leakage risk: low; postprocess must be deterministic and not learned from test.
- Fallback: beam-only decoding if `sacrebleu` unavailable; select best by average logprob.

## 3) mT5/Flan-T5 finetune + beam search (Rank 3)
- Leak-free features: normalization only; avoid tokenization surprises by strict unicode normalization.
- Model: `google/mt5-base` or `google/flan-t5-base`; lr 3e-5–1e-4; label smoothing 0.1.
- Runtime/memory: similar or slightly lighter than ByT5-base.
- Leakage risk: low.
- Fallback: if tokenization degrades transliteration, drop to ByT5.

## 4) Translation-memory retrieval blend (Rank 4, optional)
- Leak-free features: char 3–5gram TF‑IDF fit on fold-train; retrieve top-k neighbors; blend neighbor translation with model output (e.g., choose between them by chrF similarity to model candidates in MBR pool).
- Runtime/memory: cheap; TF‑IDF build moderate; works CPU.
- Leakage risk: moderate if not fold-scoped; must never build index on full train when scoring CV folds.
- Fallback: disable retrieval toggle; keep pure seq2seq.
