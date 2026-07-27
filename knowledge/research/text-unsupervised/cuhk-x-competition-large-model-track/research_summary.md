## Ranked shortlist

### 1. Category-aware multimodal ensemble — recommended final

**Leak-free features/encodings:** Cross-fitted Qwen candidate logits, VideoMAE full-clip and quarter-window logits, optional skeleton/IMU/mmWave OOF logits, modality-presence flags, deterministic motion/clip metadata, and fold-train-only option frequency, answer-cardinality, co-occurrence, and precedence priors. Calibration temperatures and blend weights are learned only from OOF predictions; sample-submission values and test distribution statistics are excluded.

**Models + concrete hyperparameters:** Qwen3-VL-Embedding-2B recall plus Qwen3-VL-Reranker-2B QLoRA (`r=32`, `alpha=32`, dropout `0.05`, two epochs, LR `2e-5`, six hard negatives, top-eight rerank); VideoMAE base (`16x224`, batch 2, accumulation 8, four epochs, last two blocks unfrozen, backbone/head LR `1e-5/3e-4`); Qwen3-VL-8B-Instruct NF4 only for entropy `>0.35` or margin `<0.18`, with eight 448-pixel ordered composites and at most eight candidates. Blend search uses step `0.05` and component weights `0.10–0.80`.

**Expected runtime/memory:** Approximately 1,320 minutes end-to-end on one RTX3060, peak 11.8GB VRAM and about 14GB host RAM, with models loaded sequentially.

**Leakage risk:** Medium if subject/clip grouping or OOF blending is wrong; low after strict grouping and fold-fit statistics.

**Fallback:** Reduce batch, candidates, TTA, frames, and composite size before replacing 8B with 4B. If the Qwen custom reranker loader is unavailable, use frozen Qwen3-VL embeddings plus a trained listwise MLP; if optional sensor parsers fail, continue with visual branches and record the omission. This ranking is supported by CUHK-X’s modality-sensitive results and Qwen’s official video reranker design. ([arXiv][6])

### 2. Qwen3-VL 2B legal-candidate matcher/reranker — strongest single branch

**Leak-free features/encodings:** Synchronized motion-sampled modality frames; question and option semantics; legal complete candidate enumeration; source/category/option-count flags; fold-fit priors. `multi` uses every non-empty subset and `sequence` uses complete permutations.

**Models + concrete hyperparameters:** `Qwen/Qwen3-VL-Embedding-2B` for cached recall and `Qwen/Qwen3-VL-Reranker-2B` QLoRA with 12 frames at 224, NF4, batch 1, accumulation 16, two epochs, LR `2e-5`, max 24 candidates and top-eight reranking.

**Expected runtime/memory:** Roughly 700 minutes, 11.5GB peak VRAM, 12GB host RAM.

**Leakage risk:** Low when negatives, priors, and calibration are built inside each fold; high if candidate statistics are computed globally before CV.

**Fallback:** Freeze the backbone and train a small listwise MLP on cached embeddings; if PEFT/bitsandbytes is missing, use Qwen3-VL-4B deterministic candidate likelihood scoring. ([GitHub][2])

### 3. VideoMAE structured temporal + optional sensor fusion — diversity specialist

**Leak-free features/encodings:** Pseudo-RGB depth/IR/thermal/motion clips, full-clip and four quarter-window embeddings, candidate text embeddings reused from Qwen or fold-fit TF-IDF/SVD fallback, plus fold-train-only cardinality/co-occurrence/precedence features. Sensor normalization is fitted on fold train subjects only.

**Models + concrete hyperparameters:** `MCG-NJU/videomae-base-finetuned-kinetics`, 16 frames at 224, batch 2, accumulation 8, four epochs, last two blocks unfrozen, LR `1e-5` backbone and `3e-4` head, weight decay `0.05`, label smoothing `0.05`. Optional skeleton temporal CNN/transformer, IMU 1D CNN-transformer, and mmWave PointNet-lite heads use 128 normalized timesteps and late-logit fusion.

**Expected runtime/memory:** About 420–500 minutes; 8.5GB VRAM and 10–12GB host RAM.

**Leakage risk:** Low with subject+clip grouping; principal risk is domain transfer and defensive parsing of unknown sensor schemas.

**Fallback:** Freeze VideoMAE and train only the head on cached embeddings; if the checkpoint cannot resolve, use `torchvision.models.video.r3d_18` with identical grouped CV and decoder. ([GitHub][3])

### 4. Local reference Qwen3-VL-4B prompt baseline — mandatory sanity candidate

**Leak-free features/encodings:** Only the row’s frames, question, available options, source, and category. No retrieved validation answer, sample value, or test label enters prompts.

**Models + concrete hyperparameters:** Qwen3-VL-4B-Instruct 4-bit, eight ordered composite frames, temperature zero, max 96 new tokens, one strict parse-repair attempt.

**Expected runtime/memory:** 120–240 minutes for a representative grouped validation subset; approximately 7–10GB VRAM.

**Leakage risk:** Low, but repeated prompt tuning on one validation fold can overfit; freeze prompts after the fast ablation stage.

**Fallback:** Qwen3-VL-2B/compatible installed LLaVA path while preserving ordered frames and exact output validation. The required notebook’s 0.400 evidence makes this a contract check, not a competitive final.
